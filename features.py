
import re


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
    code = re.sub(r'"(\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(\\.|[^'\\])*'", "''", code)
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return code


def _loop_depth(code):
    """Estimate maximum loop nesting depth using braces."""
    depth = 0
    loop_levels = []
    max_loop = 0
    nested2 = 0
    nested3 = 0
    pending = False

    tokens = re.finditer(r"\bfor\b|\bwhile\b|\{|\}", code)

    for token in tokens:
        value = token.group()

        if value in ("for", "while"):
            pending = True

        elif value == "{":
            depth += 1

            if pending:
                loop_levels.append(depth)
                current = len(loop_levels)

                max_loop = max(max_loop, current)

                if current >= 2:
                    nested2 += 1

                if current >= 3:
                    nested3 += 1

                pending = False

        elif value == "}":
            if loop_levels and loop_levels[-1] == depth:
                loop_levels.pop()

            depth = max(0, depth - 1)

    return max_loop, nested2, nested3


def hand_features(raw):
    """Extract the original 14 handcrafted code features."""
    code = strip_code(raw)

    loc = sum(1 for line in code.splitlines() if line.strip())

    num_for = len(re.findall(r"\bfor\b", code))
    num_while = len(re.findall(r"\bwhile\b", code))
    num_loops = num_for + num_while

    num_sort = len(
        re.findall(r"\.sort\s*\(|Arrays\.sort|Collections\.sort", code)
    )

    num_hash = len(
        re.findall(r"\b(HashMap|HashSet|TreeMap|TreeSet)\b", code)
    )

    num_shift = len(re.findall(r"<<|>>", code))

    num_log = len(
        re.findall(r"[*/]=\s*2\b|<<=|>>=|\*\s*2\b|/\s*2\b", code)
    )

    num_bin = len(
        re.findall(r"\bmid\b|binarySearch|\b(lo|hi|low|high)\b", code)
    )

    num_methods = len(
        re.findall(
            r"\b(?:public|private|protected|static)"
            r"[\w<>\[\]]*\s+\w+\s*\([^)]*\)\s*\{",
            code,
        )
    )

    method_names = re.findall(
        r"\b\w+\s+(\w+)\s*\([^)]*\)\s*\{", code
    )

    has_recursion = 0

    for name in set(method_names):
        calls = re.findall(
            r"\b" + re.escape(name) + r"\s*\(", code
        )

        if len(calls) > 1:
            has_recursion = 1
            break

    max_loop_depth, nested2, nested3 = _loop_depth(code)

    return [
        loc,
        num_for,
        num_while,
        num_loops,
        max_loop_depth,
        nested2,
        nested3,
        num_sort,
        num_hash,
        has_recursion,
        num_log,
        num_bin,
        num_shift,
        num_methods,
    ]


def extra_features(raw):
    """Extract the original 7 additional code features."""
    code = strip_code(raw)

    def count(pattern):
        return len(re.findall(pattern, code))

    return [
        count(r"\bif\b"),
        count(r"\bbreak\b"),
        count(r"\bswitch\b"),
        count(r"\bcontinue\b|\breturn\b"),
        count(r"\bPriorityQueue\b"),
        count(r";"),
        count(r"\b(?:int|long|double|String|char|boolean)\s+\w+"),
    ]
