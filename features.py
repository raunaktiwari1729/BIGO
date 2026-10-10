
import re


# The order of these features must match the training pipeline.
FEATURE_NAMES = [
    "loc",
    "num_for",
    "num_while",
    "num_loops",
    "max_loop_depth",
    "nested2",
    "nested3",
    "num_sort",
    "num_hash_ds",
    "has_recursion",
    "num_log_loop",
    "num_binsearch",
    "num_shift",
    "num_methods",
]

# Additional features extracted from Java code.
EXTRA_NAMES = [
    "num_if",
    "num_break",
    "num_switch",
    "num_jump",
    "has_pq",
    "num_stmt",
    "num_vars",
]


def strip_code(code):
    """Remove comments and mask string and character literals."""
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r'"(\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(\\.|[^'\\])*'", "''", code)

    return code


def _loop_depth(code):
    """Estimate loop nesting depth using loop keywords and braces."""
    depth = 0
    loop_levels = []
    max_loop = 0
    nested2 = 0
    nested3 = 0
    pending = False

    # Track loops and braces in their order of appearance.
    for token in re.finditer(r"\bfor\b|\bwhile\b|\{|\}", code):
        symbol = token.group()

        if symbol in ("for", "while"):
            pending = True

        elif symbol == "{":
            depth += 1

            if pending:
                loop_levels.append(depth)
                current_depth = len(loop_levels)

                max_loop = max(max_loop, current_depth)

                if current_depth >= 2:
                    nested2 += 1

                if current_depth >= 3:
                    nested3 += 1

                pending = False

        elif symbol == "}":
            if loop_levels and loop_levels[-1] == depth:
                loop_levels.pop()

            depth = max(0, depth - 1)

    return max_loop, nested2, nested3


def hand_features(raw):
    """Extract the 14 primary handcrafted features."""
    code = strip_code(raw)

    # Basic code and loop statistics.
    loc = len([line for line in code.split("\n") if line.strip()])
    num_for = len(re.findall(r"\bfor\b", code))
    num_while = len(re.findall(r"\bwhile\b", code))
    num_loops = num_for + num_while

    # Detect common sorting and hash-based data structures.
    num_sort = len(
        re.findall(r"\.sort\s*\(|Arrays\.sort|Collections\.sort", code)
    )
    num_hash = len(
        re.findall(r"\b(HashMap|HashSet|TreeMap|TreeSet)\b", code)
    )

    # Detect operations and patterns associated with logarithmic algorithms.
    num_shift = len(re.findall(r"<<|>>", code))
    num_log = len(
        re.findall(r"[*/]=\s*2\b|<<=|>>=|\*\s*2\b|/\s*2\b", code)
    )
    num_bin = len(
        re.findall(r"\bmid\b|binarySearch|\b(lo|hi|low|high)\b", code)
    )

    # Count method declarations using the original regular expression.
    num_methods = len(
        re.findall(
            r"\b(?:public|private|protected|static)[\w<>\[\]]*"
            r"\s+\w+\s*\([^)]*\)\s*\{",
            code,
        )
    )

    # Estimate recursion by checking whether a method name occurs
    # more than once as a function call.
    names = re.findall(r"\b\w+\s+(\w+)\s*\([^)]*\)\s*\{", code)
    has_rec = 0

    for method_name in set(names):
        calls = re.findall(
            r"\b" + re.escape(method_name) + r"\s*\(",
            code,
        )

        if len(calls) > 1:
            has_rec = 1
            break

    max_depth, nested2, nested3 = _loop_depth(code)

    # Preserve the exact feature order used by the original model.
    return [
        loc,
        num_for,
        num_while,
        num_loops,
        max_depth,
        nested2,
        nested3,
        num_sort,
        num_hash,
        has_rec,
        num_log,
        num_bin,
        num_shift,
        num_methods,
    ]


def extra_features(raw):
    """Extract the seven additional handcrafted features."""
    code = strip_code(raw)

    # Count occurrences matching each original feature pattern.
    count = lambda pattern: len(re.findall(pattern, code))

    return [
        count(r"\bif\b"),
        count(r"\bbreak\b"),
        count(r"\bswitch\b"),
        count(r"\bcontinue\b|\breturn\b"),
        count(r"\bPriorityQueue\b"),
        count(r";"),
        count(r"\b(?:int|long|double|String|char|boolean)\s+\w+"),
    ]
