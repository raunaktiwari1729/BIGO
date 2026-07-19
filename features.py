import re
import numpy as np

FEATURE_NAMES = [
    "loc", "num_for", "num_while", "num_loops", "max_loop_depth",
    "nested2", "nested3", "num_sort", "num_hash_ds", "has_recursion",
    "num_log_loop", "num_binsearch", "num_shift", "num_methods",
]

def strip_code(code):
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r'"(\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(\\.|[^'\\])*'", "''", code)
    return code

def _loop_depth(code):
    depth = 0
    loop_levels = []
    max_loop = 0
    nested2 = nested3 = 0
    pending = False
    for t in re.finditer(r"\bfor\b|\bwhile\b|\{|\}", code):
        s = t.group()
        if s in ("for", "while"):
            pending = True
        elif s == "{":
            depth += 1
            if pending:
                loop_levels.append(depth)
                cur = len(loop_levels)
                max_loop = max(max_loop, cur)
                if cur >= 2: nested2 += 1
                if cur >= 3: nested3 += 1
                pending = False
        elif s == "}":
            if loop_levels and loop_levels[-1] == depth:
                loop_levels.pop()
            depth = max(0, depth - 1)
    return max_loop, nested2, nested3

def hand_features(raw):
    code = strip_code(raw)
    loc = len([l for l in code.split("\n") if l.strip()])
    num_for = len(re.findall(r"\bfor\b", code))
    num_while = len(re.findall(r"\bwhile\b", code))
    num_loops = num_for + num_while
    num_sort = len(re.findall(r"\.sort\s*\(|Arrays\.sort|Collections\.sort", code))
    num_hash = len(re.findall(r"\b(HashMap|HashSet|TreeMap|TreeSet)\b", code))
    num_shift = len(re.findall(r"<<|>>", code))
    num_log = len(re.findall(r"[*/]=\s*2\b|<<=|>>=|\*\s*2\b|/\s*2\b", code))
    num_bin = len(re.findall(r"\bmid\b|binarySearch|\b(lo|hi|low|high)\b", code))
    num_methods = len(re.findall(r"\b(?:public|private|protected|static)[\w<>\[\]]*\s+\w+\s*\([^)]*\)\s*\{", code))
    names = re.findall(r"\b\w+\s+(\w+)\s*\([^)]*\)\s*\{", code)
    has_rec = 0
    for m in set(names):
        if len(re.findall(r"\b" + re.escape(m) + r"\s*\(", code)) > 1:
            has_rec = 1
            break
    mx, n2, n3 = _loop_depth(code)
    return [loc, num_for, num_while, num_loops, mx, n2, n3,
            num_sort, num_hash, has_rec, num_log, num_bin, num_shift, num_methods]