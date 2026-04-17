from deap import gp
import hashlib
import random, functools

# ====== storage dims (must match init_dex) ======
rows_per_matrix = 16
cols_per_matrix = 16
planes = 3

# 32-bit 饱和计数
MAX_COUNTER = (1 << 32) - 1

hash_functions = [hashlib.md5, hashlib.sha1, hashlib.sha256]

# Lazy storage init
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
# init_dex (占位；评测时会被 evolve 的 init_dex 覆盖)
# -----------------------------
if "init_dex" not in globals():
    def init_dex(e: str):
        return [(0, 0, 0), (0, 0, 1), (0, 0, 2)]

# =============================
# 基础工具
# =============================
def abs_int(a: int) -> int:
    return a if a >= 0 else -a

def safe_add(a: int, b: int) -> int: return a + b
def safe_sub(a: int, b: int) -> int: return a - b
def safe_mul(a: int, b: int) -> int: return a * b
def safe_div(a: int, b: int) -> int: return a // (b if b != 0 else 1)
def safe_mod(a: int, b: int) -> int: return a % (b if b != 0 else 1)

# 位运算
def bit_xor(a: int, b: int) -> int: return a ^ b
def bit_and(a: int, b: int) -> int: return a & b
def bit_or(a: int, b: int) -> int: return a | b
def shl(a: int, k: int) -> int:
    k = k % 31
    return a << k
def shr(a: int, k: int) -> int:
    k = k % 31
    return a >> k

# 字符串切片（给 hash_on_slice 用；与 init_dex 的风格一致）
def str_slice(s: str, start: int, end: int) -> str:
    if start < 0: start = 0
    if end < 0: end = 0
    if start > len(s): start = len(s)
    if end > len(s): end = len(s)
    if end < start: end = start
    return s[start:end]

# 哈希（用于条件、采样、签名等）
def select_hash(i: int, e: str) -> int:
    f = hash_functions[i % len(hash_functions)]
    return int(f(e.encode("utf-8")).hexdigest(), 16)

def hash_salt(i: int, e: str, salt: int) -> int:
    return select_hash(i, e + "|" + str(salt))

def hash_on_slice(i: int, e: str, l: int, r: int) -> int:
    return select_hash(i, str_slice(e, l, r))

# =============================
# bool / if
# =============================
def greater(a: int, b: int) -> bool: return a > b
def less(a: int, b: int) -> bool: return a < b
def equal(a: int, b: int) -> bool: return a == b

def and_bool(a: bool, b: bool) -> bool: return a and b
def or_bool(a: bool, b: bool) -> bool: return a or b
def not_bool(a: bool) -> bool: return not a

def bit_test(x: int, k: int) -> bool:
    k = k % 31
    return ((x >> k) & 1) == 1

def is_even(x: int) -> bool: return (x & 1) == 0
def is_odd(x: int) -> bool: return (x & 1) == 1

def if_then_else_int(cond: bool, out1: int, out2: int) -> int:
    return out1 if cond else out2

# =============================
# location (对齐 query_language 的 _loc 逻辑：r/c/z 全部 clamp)
# =============================
def _loc(e: str, i: int):
    _ensure_storage()
    idx = i % 3
    dex = init_dex(e)
    # 防御：如果 dex 不合规，就兜底
    if not isinstance(dex, (list, tuple)) or len(dex) < 3:
        dex = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    r, c, z = dex[idx]
    r = abs_int(r) % rows_per_matrix
    c = abs_int(c) % cols_per_matrix
    z = abs_int(z) % planes
    return r, c, z

# =============================
# counter ops (读/写/溢出位)
# =============================
def cnt_rdval(e: str, i: int) -> int:
    r, c, z = _loc(e, i)
    return count_matrices[z][r][c]

def cnt_rdstate(e: str, i: int) -> bool:
    r, c, z = _loc(e, i)
    return overflow_state[z][r][c]

def cnt_wrtval(e: str, i: int, v: int) -> int:
    r, c, z = _loc(e, i)
    vv = v
    if vv < 0:
        vv = 0
    if vv > MAX_COUNTER:
        vv = MAX_COUNTER
        overflow_state[z][r][c] = True
    else:
        overflow_state[z][r][c] = False
    count_matrices[z][r][c] = vv
    return vv

def cnt_wrtstate(e: str, i: int, st: bool) -> bool:
    r, c, z = _loc(e, i)
    overflow_state[z][r][c] = bool(st)
    return overflow_state[z][r][c]

# 兼容旧名字
def query_count(e: str, i: int) -> int:
    return cnt_rdval(e, i)

def update_count(e: str, i: int, delta: int) -> int:
    r, c, z = _loc(e, i)

    cur = count_matrices[z][r][c]

    # PATCH(step5): 防止 bool/0 让更新变成 +0；同时把 delta 规范成 int
    d = int(delta)
    if d == 0:
        d = 1

    nxt = cur + d
    if nxt < 0:
        nxt = 0
        overflow_state[z][r][c] = False
    elif nxt > MAX_COUNTER:
        nxt = MAX_COUNTER
        overflow_state[z][r][c] = True
    else:
        overflow_state[z][r][c] = False

    count_matrices[z][r][c] = nxt
    return nxt

# =============================
# 关键：条件副作用（否则 if 包 update 会“照样更新”）
# =============================
def update_if(cond: bool, e: str, i: int, delta: int) -> int:
    if cond:
        return update_count(e, i, delta)
    else:
        return query_count(e, i)

def wrtval_if(cond: bool, e: str, i: int, v: int) -> int:
    if cond:
        return cnt_wrtval(e, i, v)
    else:
        return cnt_rdval(e, i)

def wrtstate_if(cond: bool, e: str, i: int, st: bool) -> bool:
    if cond:
        return cnt_wrtstate(e, i, st)
    else:
        return cnt_rdstate(e, i)

# =============================
# 宏：Conservative Update (CU)
# =============================
def cu_update(e: str, delta: int) -> int:
    c0 = query_count(e, 0)
    c1 = query_count(e, 1)
    c2 = query_count(e, 2)
    m = c0
    if c1 < m: m = c1
    if c2 < m: m = c2

    out = 0
    if c0 == m: out = update_count(e, 0, delta)
    if c1 == m: out = update_count(e, 1, delta)
    if c2 == m: out = update_count(e, 2, delta)
    return out

def cu_update_if(cond: bool, e: str, delta: int) -> int:
    if cond:
        return cu_update(e, delta)
    else:
        return 0

def base(a: int, b: int, c: int) -> float:
    aa = a
    bb = b
    cc = c
    return 0.0


# =============================
# PrimitiveSetTyped
# =============================
pset_update = gp.PrimitiveSetTyped("UPDATE", [str], float)
pset_update.renameArguments(ARG0="e")

# int ops
pset_update.addPrimitive(safe_add, [int, int], int)
pset_update.addPrimitive(safe_sub, [int, int], int)
pset_update.addPrimitive(safe_mul, [int, int], int)
pset_update.addPrimitive(safe_div, [int, int], int)
pset_update.addPrimitive(safe_mod, [int, int], int)
pset_update.addPrimitive(abs_int,  [int], int)

# bit ops
pset_update.addPrimitive(bit_xor, [int, int], int)
pset_update.addPrimitive(bit_and, [int, int], int)
pset_update.addPrimitive(bit_or,  [int, int], int)
pset_update.addPrimitive(shl,     [int, int], int)
pset_update.addPrimitive(shr,     [int, int], int)

# hash
pset_update.addPrimitive(select_hash,   [int, str], int)
pset_update.addPrimitive(hash_salt,     [int, str, int], int)
pset_update.addPrimitive(hash_on_slice, [int, str, int, int], int)
pset_update.addPrimitive(str_slice,[str,int,int],str)
# bool + ifelse
pset_update.addPrimitive(greater, [int, int], bool)
pset_update.addPrimitive(less,    [int, int], bool)
pset_update.addPrimitive(equal,   [int, int], bool)
# pset_update.addPrimitive(and_bool, [bool, bool], bool)
# pset_update.addPrimitive(or_bool,  [bool, bool], bool)
# pset_update.addPrimitive(not_bool, [bool], bool)
pset_update.addPrimitive(bit_test, [int, int], bool)
# pset_update.addPrimitive(is_even,  [int], bool)
# pset_update.addPrimitive(is_odd,   [int], bool)
pset_update.addPrimitive(if_then_else_int, [bool, int, int], int)

# counter reads/writes
# pset_update.addPrimitive(cnt_rdval,   [str, int], int)
pset_update.addPrimitive(cnt_rdstate, [str, int], bool)
# pset_update.addPrimitive(cnt_wrtval,  [str, int, int], int)
#pset_update.addPrimitive(cnt_wrtstate,[str, int, bool], bool)

pset_update.addPrimitive(query_count,  [str, int], int)
pset_update.addPrimitive(update_count, [str, int, int], int)

# side-effect-safe wrappers
pset_update.addPrimitive(update_if,   [bool, str, int, int], int)
#pset_update.addPrimitive(wrtval_if,   [bool, str, int, int], int)
pset_update.addPrimitive(wrtstate_if, [bool, str, int, bool], bool)

# macros
pset_update.addPrimitive(cu_update,    [str, int], int)
pset_update.addPrimitive(cu_update_if, [bool, str, int], int)

# root
pset_update.addPrimitive(base, [int, int, int], float)

# int terminals（含 -1 用于 delta）
# for c in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 101, 102, -1]:
#     pset_update.addTerminal(int(c), int)
for c in [0, 1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 24, 31, 32, -1]:
    pset_update.addTerminal(int(c), int)

# bool terminals
pset_update.addTerminal(True, bool)
pset_update.addTerminal(False, bool)
pset_update.addTerminal([], list)
# Ephemeral：索引与 delta
pset_update.addEphemeralConstant("rand_i",     functools.partial(random.randint, 0, 2), int)
pset_update.addEphemeralConstant("rand_delta", functools.partial(random.choice, [-1, 1, 2, -2, 3, -3]),int)
#pset_update.addEphemeralConstant("rand_small", functools.partial(random.randint, 0, 64), int)
pset_update.addEphemeralConstant("rand_small", functools.partial(random.randint, 0, 16), int)