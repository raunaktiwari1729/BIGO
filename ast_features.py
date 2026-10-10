
import javalang

from javalang.tree import (
    ArrayCreator,
    Assignment,
    BinaryOperation,
    BreakStatement,
    ClassCreator,
    ContinueStatement,
    DoStatement,
    ForStatement,
    IfStatement,
    LambdaExpression,
    Literal,
    MethodDeclaration,
    MethodInvocation,
    ReturnStatement,
    SwitchStatement,
    WhileStatement,
)


AST_NAMES = [
    "ast_fail",
    "ast_loops",
    "ast_depth",
    "ast_lin_depth",
    "ast_log_depth",
    "ast_self_calls",
    "ast_rec_halving",
    "ast_any_rec",
    "ast_sort_in_loop",
    "ast_scan_in_loop",
    "ast_tree_in_loop",
    "ast_bs_in_loop",
    "ast_shift_cond",
    "ast_ifs",
    "ast_breaks",
    "ast_switch",
    "ast_continue",
    "ast_returns",
    "ast_lambda",
    "ast_arr2d",
    "ast_methods",
    "ast_stmt_in_loop",
    "ast_small_loops",
]

LOOPS = (ForStatement, WhileStatement, DoStatement)

SCAN = {
    "contains",
    "indexOf",
    "remove",
    "lastIndexOf",
    "containsAll",
}

TREE_OPS = {
    "poll",
    "offer",
    "put",
    "get",
    "add",
    "first",
    "last",
    "floor",
    "ceiling",
    "higher",
    "lower",
    "pollFirst",
    "pollLast",
}


def kids(n):
    """Yield AST nodes contained within a node."""
    for child in n.children:
        stack = [child]

        while stack:
            node = stack.pop()

            if isinstance(node, javalang.ast.Node):
                yield node
            elif isinstance(node, (list, tuple, set)):
                stack.extend(node)


def is_two(n):
    """Check whether a literal represents the value two."""
    return isinstance(n, Literal) and n.value in ("2", "2L")


def halving(a):
    """Detect assignments that may reduce a value geometrically."""
    if a.type in ("<<=", ">>=", ">>>="):
        return True

    if a.type in ("/=", "*="):
        return isinstance(a.value, Literal)

    if a.type == "=" and isinstance(a.value, BinaryOperation):
        value = a.value

        if value.operator in ("/", ">>", ">>>"):
            return True

        if value.operator == "*" and (
            is_two(value.operandl) or is_two(value.operandr)
        ):
            return True

    return False


def own_nodes(n):
    """Traverse descendants without entering nested loops."""
    for node in kids(n):
        yield node

        if not isinstance(node, LOOPS):
            yield from own_nodes(node)


def is_log_loop(loop):
    """Check for logarithmic-update patterns inside a loop."""
    parts = [loop.body]

    if (
        isinstance(loop, ForStatement)
        and hasattr(loop.control, "update")
        and loop.control.update
    ):
        parts.append(loop.control.update)

    for part in parts:
        items = part if isinstance(part, list) else [part]

        for item in items:
            if item is None:
                continue

            if isinstance(item, Assignment) and halving(item):
                return True

            if isinstance(item, javalang.ast.Node):
                for node in own_nodes(item):
                    if isinstance(node, Assignment) and halving(node):
                        return True

    return False


def find_consts(tree):
    """Find small initialized variables that appear unchanged."""
    assigned = set()

    for _, assignment in tree.filter(Assignment):
        if hasattr(assignment.expressionl, "member"):
            assigned.add(assignment.expressionl.member)

    for _, member in tree.filter(javalang.tree.MemberReference):
        if member.prefix_operators or member.postfix_operators:
            operators = (
                (member.prefix_operators or [])
                + (member.postfix_operators or [])
            )

            if any(op in ("++", "--") for op in operators):
                assigned.add(member.member)

    consts = set()

    for _, declaration in tree.filter(javalang.tree.VariableDeclarator):
        value = declaration.initializer

        if (
            isinstance(value, Literal)
            and value.value.isdigit()
            and 2 <= int(value.value) <= 100
            and declaration.name not in assigned
        ):
            consts.add(declaration.name)

    return consts


# Stores the small-variable bounds detected for the current program.
CONSTS = set()


def small_bound(loop):
    """Check whether a loop condition uses a small constant bound."""
    condition = None

    if isinstance(loop, ForStatement) and hasattr(
        loop.control, "condition"
    ):
        condition = loop.control.condition
    elif isinstance(loop, (WhileStatement, DoStatement)):
        condition = loop.condition

    if (
        not isinstance(condition, BinaryOperation)
        or condition.operator not in ("<", "<=", ">", ">=", "!=")
    ):
        return False

    for side in (condition.operandl, condition.operandr):
        if isinstance(side, Literal) and side.value.isdigit():
            if 2 <= int(side.value) <= 100:
                return True

        if (
            isinstance(side, javalang.tree.MemberReference)
            and side.member in CONSTS
        ):
            return True

    return False


def shift_in_cond(loop):
    """Detect bit-shift operations in a loop condition."""
    condition = None

    if isinstance(loop, ForStatement) and hasattr(
        loop.control, "condition"
    ):
        condition = loop.control.condition
    elif isinstance(loop, (WhileStatement, DoStatement)):
        condition = loop.condition

    if condition is None:
        return 0

    nodes = (
        [condition]
        + list(kids(condition))
        + [node for child in kids(condition) for node in kids(child)]
    )

    for node in nodes:
        if (
            isinstance(node, BinaryOperation)
            and node.operator in ("<<", ">>")
        ):
            return 1

    return 0


def new_stats():
    """Initialize the statistics collected during AST traversal."""
    return {
        "lin": 0,
        "log": 0,
        "depth": 0,
        "self_calls": 0,
        "rec_half": 0,
        "any_rec": 0,
        "sort_loop": 0,
        "scan_loop": 0,
        "tree_loop": 0,
        "bs_loop": 0,
        "shift": 0,
        "stmt_loop": 0,
        "const": 0,
    }


def merge(st, other, lin, log):
    """Merge statistics from a called method into the current method."""
    st["lin"] = max(st["lin"], lin + other["lin"])
    st["log"] = max(st["log"], log + other["log"])
    st["depth"] = max(st["depth"], lin + log + other["depth"])

    for key in (
        "any_rec",
        "sort_loop",
        "scan_loop",
        "tree_loop",
        "bs_loop",
        "shift",
        "const",
    ):
        st[key] += other[key]


def walk(n, name, methods, cache, stack, st, lin, log):
    """Traverse the AST and collect loop and method statistics."""
    if isinstance(n, LOOPS):
        is_logarithmic = is_log_loop(n)
        is_constant_bound = small_bound(n)

        st["const"] += is_constant_bound

        lin2 = lin + (0 if is_logarithmic or is_constant_bound else 1)
        log2 = log + (1 if is_logarithmic else 0)

        st["lin"] = max(st["lin"], lin2)
        st["log"] = max(st["log"], log2)
        st["depth"] = max(st["depth"], lin2 + log2)
        st["shift"] += shift_in_cond(n)

        for child in kids(n):
            walk(
                child,
                name,
                methods,
                cache,
                stack,
                st,
                lin2,
                log2,
            )

        return

    if isinstance(n, MethodInvocation):
        method_name = n.member
        in_loop = lin + log > 0

        if method_name in methods:
            if method_name == name:
                st["self_calls"] += 1
                st["any_rec"] = 1

                for argument in n.arguments:
                    nodes = [argument] + list(kids(argument))

                    for node in nodes:
                        if (
                            isinstance(node, BinaryOperation)
                            and node.operator in ("/", ">>")
                        ):
                            st["rec_half"] = 1

            elif method_name in stack:
                st["any_rec"] = 1

            else:
                merge(
                    st,
                    method_stats(method_name, methods, cache, stack),
                    lin,
                    log,
                )

        if method_name in ("sort", "parallelSort") and in_loop:
            st["sort_loop"] += 1

        if method_name in SCAN and in_loop:
            st["scan_loop"] += 1

        if method_name in TREE_OPS and in_loop:
            st["tree_loop"] += 1

        if method_name == "binarySearch" and in_loop:
            st["bs_loop"] += 1

    # Count statements encountered inside loops.
    if lin + log > 0 and n.__class__.__name__.endswith("Statement"):
        st["stmt_loop"] += 1

    for child in kids(n):
        walk(
            child,
            name,
            methods,
            cache,
            stack,
            st,
            lin,
            log,
        )


def method_stats(name, methods, cache, stack):
    """Calculate statistics for a method, reusing cached results."""
    if name in cache:
        return cache[name]

    stack.add(name)
    st = new_stats()

    for body in methods[name]:
        for statement in (body.body or []):
            walk(
                statement,
                name,
                methods,
                cache,
                stack,
                st,
                0,
                0,
            )

    stack.discard(name)
    cache[name] = st

    return st


def ast_features(code):
    """Extract the 23 AST-based features from Java source code."""
    try:
        tree = javalang.parse.parse(code)
    except Exception:
        # Mark parsing failure and return zeros for the other features.
        return [1] + [0] * (len(AST_NAMES) - 1)

    global CONSTS
    CONSTS = find_consts(tree)

    # Group method declarations by method name.
    methods = {}

    for _, method in tree.filter(MethodDeclaration):
        methods.setdefault(method.name, []).append(method)

    cache = {}

    called = {
        node.member
        for _, node in tree.filter(MethodInvocation)
    }

    roots = [
        name for name in methods
        if name not in called or name == "main"
    ]

    st = new_stats()

    # Aggregate statistics from root methods.
    for name in roots:
        stats = method_stats(name, methods, cache, set())

        for key in st:
            st[key] = max(st[key], stats[key])

    # Include recursion information from all methods.
    for name in methods:
        stats = method_stats(name, methods, cache, set())

        st["self_calls"] = max(st["self_calls"], stats["self_calls"])
        st["rec_half"] = max(st["rec_half"], stats["rec_half"])
        st["any_rec"] = max(st["any_rec"], stats["any_rec"])

    count = lambda node_type: sum(1 for _ in tree.filter(node_type))

    loops = sum(count(loop_type) for loop_type in LOOPS)

    arr2d = sum(
        1
        for _, array in tree.filter(ArrayCreator)
        if len(array.dimensions) >= 2
    )

    # Preserve the original 23-feature order.
    return [
        0,
        loops,
        st["depth"],
        st["lin"],
        st["log"],
        st["self_calls"],
        st["rec_half"],
        st["any_rec"],
        st["sort_loop"],
        st["scan_loop"],
        st["tree_loop"],
        st["bs_loop"],
        st["shift"],
        count(IfStatement),
        count(BreakStatement),
        count(SwitchStatement),
        count(ContinueStatement),
        count(ReturnStatement),
        count(LambdaExpression),
        arr2d,
        len(methods),
        st["stmt_loop"],
        st["const"],
    ]
v
