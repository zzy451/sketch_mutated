from deap import gp
import hashlib, random, functools

rows_per_matrix = 16
cols_per_matrix = 16
planes = 3

INF = 1 << 60  # +inf 近似（overflow 映射用）

hash_functions = [hashlib.md5, hashlib.sha1, hashlib.sha256]

def str_slice(s: str, start: int, end: int) -> str:
    if start < 0: start = 0
    if end < 0: end = 0
    if start > len(s): start = len(s)
    if end > len(s): end = len(s)
    if end < start: end = start
    return s[start:end]

# -----------------------------
# Lazy storage init (与 update_language 保持一致，避免重复初始化清零)
# -----------------------------
def _ensure_storage():
    global count_matrices, overflow_state
    if "count_matrices" not in globals():
        count_matrices = [
            [[0 for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
            for _ in range(planes)
        ]
    if "overflow_state" not in globals():
        overflow_state = [
            [[False for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
            for _ in range(planes)
        ]

# -----------------------------
# init_dex 占位
# -----------------------------
if "init_dex" not in globals():
    def init_dex(e: str):
        return [(0, 0, 0), (0, 0, 1), (0, 0, 2)]

def abs_int(a: int) -> int:
    return a if a >= 0 else -a

def _loc(e: str, i: int):
    _ensure_storage()
    idx = i % 3
    dex = init_dex(e)
    if not isinstance(dex, (list, tuple)) or len(dex) < 3:
        dex = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    r, c, z = dex[idx]
    r = abs_int(r) % rows_per_matrix
    c = abs_int(c) % cols_per_matrix
    z = abs_int(z) % planes
    return r, c, z

# ===== counter ops =====
def cnt_rdval(e: str, i: int) -> int:
    r, c, z = _loc(e, i)
    return count_matrices[z][r][c]

def cnt_rdstate(e: str, i: int) -> bool:
    r, c, z = _loc(e, i)
    return overflow_state[z][r][c]

# 兼容关键字：query_date（默认把 overflow 映射为 +inf）
def query_date(e: str, i: int) -> int:
    r, c, z = _loc(e, i)
    if overflow_state[z][r][c]:
        return INF
    return count_matrices[z][r][c]

# ===== math/agg =====
def safe_add(a: int, b: int) -> int: return a + b
def safe_sub(a: int, b: int) -> int: return a - b
def safe_mul(a: int, b: int) -> int: return a * b
def safe_div(a: int, b: int) -> int: return a // (b if b != 0 else 1)
def safe_mod(a: int, b: int) -> int: return a % (b if b != 0 else 1)

def safe_min(a: int, b: int) -> int: return a if a < b else b
def safe_max(a: int, b: int) -> int: return a if a > b else b

def min3(a: int, b: int, c: int) -> int: return safe_min(a, safe_min(b, c))
def max3(a: int, b: int, c: int) -> int: return safe_max(a, safe_max(b, c))
def sum3(a: int, b: int, c: int) -> int: return a + b + c
def avg3(a: int, b: int, c: int) -> int: return (a + b + c) // 3

def median3(a: int, b: int, c: int) -> float:
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return b

# ===== compare/if =====
def lt(a: int, b: int) -> bool: return a < b
def gt(a: int, b: int) -> bool: return a > b
def eq(a: int, b: int) -> bool: return a == b

def and_bool(a: bool, b: bool) -> bool: return a and b
def or_bool(a: bool, b: bool) -> bool: return a or b
def not_bool(a: bool) -> bool: return not a

def if_then_else_int(cond: bool, out1: int, out2: int) -> int:
    return out1 if cond else out2

# ===== root: base_sel =====
def base_sel(mode: int, a: int, b: int, c: int) -> float:
    m = abs(mode) % 4
    if m == 0:   # min
        return float(min(a, b, c))
    if m == 1:   # max
        return float(max(a, b, c))
    if m == 2:   # median
        return float(sorted([a, b, c])[1])
    # avg
    return float((a + b + c) // 3)

# ===== PrimitiveSetTyped =====
pset = gp.PrimitiveSetTyped("QUERY", [str], float)
pset.renameArguments(ARG0="e")

pset.addPrimitive(str_slice,[str,int,int],str)
# int ops
pset.addPrimitive(safe_add, [int, int], int)
pset.addPrimitive(safe_sub, [int, int], int)
pset.addPrimitive(safe_mul, [int, int], int)
pset.addPrimitive(safe_div, [int, int], int)
pset.addPrimitive(safe_mod, [int, int], int)

pset.addPrimitive(safe_min, [int, int], int)
pset.addPrimitive(safe_max, [int, int], int)

# pset.addPrimitive(min3,    [int, int, int], int)
# pset.addPrimitive(max3,    [int, int, int], int)
# pset.addPrimitive(sum3,    [int, int, int], int)
# pset.addPrimitive(avg3,    [int, int, int], int)
# pset.addPrimitive(median3, [int, int, int], int)

# compare/if
pset.addPrimitive(lt, [int, int], bool)
pset.addPrimitive(gt, [int, int], bool)
pset.addPrimitive(eq, [int, int], bool)
# pset.addPrimitive(and_bool, [bool, bool], bool)
# pset.addPrimitive(or_bool,  [bool, bool], bool)
# pset.addPrimitive(not_bool, [bool], bool)
pset.addPrimitive(if_then_else_int, [bool, int, int], int)

# counter reads
# pset.addPrimitive(cnt_rdval,   [str, int], int)
# pset.addPrimitive(cnt_rdstate, [str, int], bool)
pset.addPrimitive(query_date,  [str, int], int)

# root
pset.addPrimitive(base_sel, [int,int, int, int], float)

# terminals
# for c in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1024,
#           rows_per_matrix, cols_per_matrix, INF]:
#     pset.addTerminal(int(c), int)

for c in [0, 1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 24, 31, 32,
          rows_per_matrix, cols_per_matrix, INF]:
    pset.addTerminal(int(c), int)

pset.addTerminal(True, bool)
pset.addTerminal(False, bool)
pset.addTerminal([], list)
pset.addEphemeralConstant("rand_mode", functools.partial(random.randint, 0, 3), int)
#pset.addEphemeralConstant("rand_small", functools.partial(random.randint, 0, 64), int)
pset.addEphemeralConstant("rand_small", functools.partial(random.randint, 0, 16), int)
pset.addEphemeralConstant("rand_idx",   functools.partial(random.randint, 0, 2), int)