from deap import gp
import hashlib, random, functools

hash_functions = [hashlib.md5, hashlib.sha1, hashlib.sha256]

rows_per_matrix = 16
cols_per_matrix = 16

# cap 候选集合（方案A：小集合）
# CAP_SET = [8, 16, 32, 64, 96, 102]
#SEG_SET = [2, 3, 4, 6, 8, 12, 16]
CAP_SET = [2, 4, 8, 12, 16]
SEG_SET = [2, 4, 8, 16]




# -----------------------------
# 工具函数
# -----------------------------
def abs_int(a: int) -> int:
    return a if a >= 0 else -a

def _cap(m: int) -> int:
    # mola/molb/molc 合法化并限制到 [1, cols_per_matrix]
    if m == 0:
        m = 1
    m = abs_int(m)
    if m > cols_per_matrix:
        m = cols_per_matrix
    if m <= 0:
        m = 1
    return m


# 根函数（必须是 list_3）

def list_3(a, a1, mola, b, b1, molb, c, c1, molc):
    mola = _cap(mola)
    molb = _cap(molb)
    molc = _cap(molc)
    return [
        (a % rows_per_matrix, a1 % mola, 0),
        (b % rows_per_matrix, b1 % molb, 1),
        (c % rows_per_matrix, c1 % molc, 2),
    ]

# 字符串原语
def str_concat(a: str, b: str) -> str:
    return a + b

def str_slice(s: str, start: int, end: int) -> str:
    if start < 0: start = 0
    if end < 0: end = 0
    if start > len(s): start = len(s)
    if end > len(s): end = len(s)
    if end < start: end = start
    return s[start:end]

def str_len(s: str) -> int:
    return len(s)

def int_to_str(x: int) -> str:
    return str(x)

# 哈希原语
def select_hash(i: int, e: str) -> int:
    f = hash_functions[i % len(hash_functions)]
    return int(f(e.encode("utf-8")).hexdigest(), 16)

def hash_salt(i: int, e: str, salt: int) -> int:
    return select_hash(i, e + "|" + str(salt))

def hash_on_slice(i: int, e: str, l: int, r: int) -> int:
    return select_hash(i, str_slice(e, l, r))


# 数学/位运算原语
def safe_add(a: int, b: int) -> int: return a + b
def safe_sub(a: int, b: int) -> int: return a - b
def safe_mul(a: int, b: int) -> int: return a * b
def safe_div(a: int, b: int) -> int: return a // (b if b != 0 else 1)
def safe_mod(a: int, b: int) -> int: return a % (b if b != 0 else 1)

def bit_xor(a: int, b: int) -> int: return a ^ b
def bit_and(a: int, b: int) -> int: return a & b
def bit_or(a: int, b: int) -> int: return a | b

def shl(a: int, k: int) -> int:
    k = k % 31
    return a << k

def shr(a: int, k: int) -> int:
    k = k % 31
    return a >> k

def affine(h1: int, h2: int, k: int) -> int:
    return h1 + k * h2

# 行分流（Row-locality routing）
def seg_id(h: int, seg_cnt: int) -> int:
    S = abs_int(seg_cnt)
    if S == 0:
        S = 1
    return abs_int(h) % S

def seg_row_index(h: int, seg_id_: int, seg_cnt: int) -> int:
    # 不丢行：把余数行均匀分给前若干段
    S = abs_int(seg_cnt)
    if S == 0:
        S = 1
    if S > rows_per_matrix:
        S = rows_per_matrix

    sid = abs_int(seg_id_) % S

    q = rows_per_matrix // S
    r = rows_per_matrix % S

    if sid < r:
        w = q + 1
        base = sid * (q + 1)
    else:
        w = q
        base = r * (q + 1) + (sid - r) * q

    if w <= 0:
        w = 1
    return base + (abs_int(h) % w)

# 合成原语：一条 primitive 完成 “特征 -> 段 -> 段内行”
def row_route(h_feat: int, h_full: int, seg_cnt: int) -> int:
    return seg_row_index(h_full, seg_id(h_feat, seg_cnt), seg_cnt)

# cap 映射：先“两档（2-tier）”，同时保留 quartile 以便后续精修
# =========================================================
def cap_half(seg_id_: int, seg_cnt: int, c_lo: int, c_hi: int) -> int:
    """
    两档：前 50% 段 -> c_lo，后 50% 段 -> c_hi
    """
    S = abs_int(seg_cnt)
    if S == 0:
        S = 1
    sid = abs_int(seg_id_) % S

    g = (sid * 2) // S  # 0..1
    cap = c_lo if g == 0 else c_hi

    cap = abs_int(cap)
    if cap == 0:
        cap = 1
    if cap > cols_per_matrix:
        cap = cols_per_matrix
    return cap

# 1~2 个宏原语（两档），让 GP 更快搜到“区域列宽不同”
def cap2_32_102(seg_id_: int, seg_cnt: int) -> int:
    return cap_half(seg_id_, seg_cnt, 32, 102)

def cap2_16_64(seg_id_: int, seg_cnt: int) -> int:
    return cap_half(seg_id_, seg_cnt, 16, 64)

def cap2_4_16(seg_id_: int, seg_cnt: int) -> int:
    return cap_half(seg_id_, seg_cnt, 4, 16)

def cap2_8_16(seg_id_: int, seg_cnt: int) -> int:
    return cap_half(seg_id_, seg_cnt, 8, 16)

# 保留通用 quartile（后续想做 4 档时可以直接用，不影响现在两档）
def cap_quartile(seg_id_: int, seg_cnt: int, c0: int, c1: int, c2: int, c3: int) -> int:
    S = abs_int(seg_cnt)
    if S == 0:
        S = 1
    sid = abs_int(seg_id_) % S

    g = (sid * 4) // S  # 0..3
    cap = c0 if g == 0 else (c1 if g == 1 else (c2 if g == 2 else c3))

    cap = abs_int(cap)
    if cap == 0:
        cap = 1
    if cap > cols_per_matrix:
        cap = cols_per_matrix
    return cap


# -----------------------------
# bool / if/else
# -----------------------------
def greater(a: int, b: int) -> bool: return a > b
def less(a: int, b: int) -> bool: return a < b
def equal(a: int, b: int) -> bool: return a == b

def and_bool(a: bool, b: bool) -> bool: return a and b
def or_bool(a: bool, b: bool) -> bool: return a or b
def not_bool(a: bool) -> bool: return not a

def if_then_else_int(cond: bool, out1: int, out2: int) -> int:
    return out1 if cond else out2

def if_then_else_str(cond: bool, out1: str, out2: str) -> str:
    return out1 if cond else out2

def if_then_else_bool(cond: bool, out1: bool, out2: bool) -> bool:
    return out1 if cond else out2


# -----------------------------
# PrimitiveSetTyped
# -----------------------------
pset = gp.PrimitiveSetTyped("INIT_DEX", [str], list)
pset.renameArguments(ARG0="e")

# hash
pset.addPrimitive(select_hash,   [int, str], int)
pset.addPrimitive(hash_salt,     [int, str, int], int)
pset.addPrimitive(hash_on_slice, [int, str, int, int], int)

# math
pset.addPrimitive(safe_add, [int, int], int)
pset.addPrimitive(safe_sub, [int, int], int)
pset.addPrimitive(safe_mul, [int, int], int)
pset.addPrimitive(safe_div, [int, int], int)
pset.addPrimitive(safe_mod, [int, int], int)
pset.addPrimitive(abs_int,  [int], int)

# bit
pset.addPrimitive(bit_xor, [int, int], int)
pset.addPrimitive(bit_and, [int, int], int)
pset.addPrimitive(bit_or,  [int, int], int)
pset.addPrimitive(shl,     [int, int], int)
pset.addPrimitive(shr,     [int, int], int)

# structure / routing
pset.addPrimitive(affine,        [int, int, int], int)
pset.addPrimitive(seg_id,        [int, int], int)
pset.addPrimitive(seg_row_index, [int, int, int], int)
pset.addPrimitive(row_route,     [int, int, int], int)

# cap mapping
pset.addPrimitive(cap_half,      [int, int, int, int], int)
# pset.addPrimitive(cap2_32_102,   [int, int], int)
# pset.addPrimitive(cap2_16_64,    [int, int], int)
pset.addPrimitive(cap2_4_16,     [int, int], int)
pset.addPrimitive(cap2_8_16,     [int, int], int)
pset.addPrimitive(cap_quartile,  [int, int, int, int, int, int], int)

# string
pset.addPrimitive(str_concat, [str, str], str)
pset.addPrimitive(str_slice,  [str, int, int], str)
pset.addPrimitive(str_len,    [str], int)
pset.addPrimitive(int_to_str, [int], str)

# bool / if
pset.addPrimitive(greater,   [int, int], bool)
pset.addPrimitive(less,      [int, int], bool)
pset.addPrimitive(equal,     [int, int], bool)
pset.addPrimitive(and_bool,  [bool, bool], bool)
pset.addPrimitive(or_bool,   [bool, bool], bool)
pset.addPrimitive(not_bool,  [bool], bool)

pset.addPrimitive(if_then_else_int,  [bool, int, int], int)
pset.addPrimitive(if_then_else_str,  [bool, str, str], str)
pset.addPrimitive(if_then_else_bool, [bool, bool, bool], bool)

# root
pset.addPrimitive(list_3, [int, int, int, int, int, int, int, int, int], list)


# -----------------------------
# terminals
# -----------------------------
# 常用小常量（切片/盐值/阈值/移位等）
for c in [0, 1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 24, 31, 32, 48, 63, 64]:
    pset.addTerminal(int(c), int)

# seg_cnt 只给固定集合
for c in SEG_SET:
    pset.addTerminal(int(c), int)

# cap 常量只给固定集合（方案A）
for c in CAP_SET:
    pset.addTerminal(int(c), int)

# 维度
pset.addTerminal(rows_per_matrix, int)
pset.addTerminal(cols_per_matrix, int)

pset.addTerminal([], list)

# bool 终端
pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

# Ephemeral：seg_cnt / cap 也只从集合里采样
pset.addEphemeralConstant("rand_seg", functools.partial(random.choice, SEG_SET), int)
pset.addEphemeralConstant("rand_cap", functools.partial(random.choice, CAP_SET), int)
#pset.addEphemeralConstant("rand_small", functools.partial(random.randint, 0, 64), int)
pset.addEphemeralConstant("rand_small", functools.partial(random.randint, 0, 16), int)