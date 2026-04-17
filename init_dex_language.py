from deap import gp, base, creator, tools
import hashlib, operator, random, functools

hash_functions = [hashlib.md5, hashlib.sha1, hashlib.sha256]

rows_per_matrix = 102
cols_per_matrix = 102
planes = 3

# 计数矩阵：3 x 102 x 102
count_matrix1 = [[0 for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
count_matrix2 = [[0 for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
count_matrix3 = [[0 for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
count_matrices = [count_matrix1, count_matrix2, count_matrix3]

# 溢出状态矩阵：3 x 102 x 102
overflow_matrix1 = [[False for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
overflow_matrix2 = [[False for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
overflow_matrix3 = [[False for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
overflow_matrices = [overflow_matrix1, overflow_matrix2, overflow_matrix3]


# counter 上限：PPT 里写的是“maximally 32-bits”
COUNTER_BITS = 32
MAX_COUNTER = (1 << COUNTER_BITS) - 1

# query_date 里把 overflow 映射成“足够大的 int”
INF = MAX_COUNTER + 1

def list_3(a,a1,mola,b,b1,molb,c,c1,molc):
    if mola == 0: mola = 1
    if molb == 0: molb = 1
    if molc == 0: molc = 1
    if mola < 0: mola = -mola
    if molb < 0: molb = -molb
    if molc < 0: molc = -molc
    return [(a % 102, a1 % mola, 0), (b % 102, b1 % molb, 1), (c % 102, c1 % molc, 2)]

#这里的mola,molb,molc以后可以扩展为根据输入变量e设置上限

def safe_mod(a: int, b: int) -> int:
    return a % (b if b != 0 else 1)

def abs_int(a: int) -> int:
    return a if a >= 0 else -a

def lt(a: int, b: int) -> bool: return a < b
def gt(a: int, b: int) -> bool: return a > b
def eq(a: int, b: int) -> bool: return a == b

def if_then_else_int(cond: bool, out1: int, out2: int) -> int:
    return out1 if cond else out2

def hash_salt(i: int, e: str, salt: int) -> int:
    return select_hash(i, e + "|" + str(salt))

def hash_on_slice(i: int, e: str, l: int, r: int) -> int:
    return select_hash(i, str_slice(e, l, r))


def str_concat(a, b):
    return a + b

def str_slice(s, start, end):
    return s[start:end]


def select_hash(i: int, e: str) -> int:
    #根据索引选择哈希函数并返回整数哈希值
    f = hash_functions[i % len(hash_functions)]
    return int(f(e.encode('utf-8')).hexdigest(), 16)


def safe_add(a: int, b: int) -> int: return a + b
def safe_sub(a: int, b: int) -> int: return a - b
def safe_mul(a: int, b: int) -> int: return a * b
def safe_div(a: int, b: int) -> int: return a // (b if b != 0 else 1)


def if_then_else(cond: bool, out1: int, out2: int) -> int:
    """if/else 结构"""
    return out1 if cond else out2



pset = gp.PrimitiveSetTyped("MAIN", [str],list)
pset.renameArguments(ARG0="e")

pset.addPrimitive(select_hash, [int, str], int)
pset.addPrimitive(hash_salt, [int, str, int], int)
pset.addPrimitive(hash_on_slice, [int, str, int, int], int)

pset.addPrimitive(safe_add, [int, int], int)
pset.addPrimitive(safe_sub, [int, int], int)
pset.addPrimitive(safe_mul, [int, int], int)
pset.addPrimitive(safe_div, [int, int], int)
pset.addPrimitive(safe_mod, [int, int], int)
pset.addPrimitive(abs_int, [int], int)

pset.addPrimitive(lt, [int, int], bool)
pset.addPrimitive(gt, [int, int], bool)
pset.addPrimitive(eq, [int, int], bool)
pset.addPrimitive(if_then_else_int, [bool, int, int], int)

pset.addPrimitive(str_slice, [str, int, int], str)

pset.addPrimitive(list_3, [int,int,int,int,int,int,int,int,int], list)

for c in [0,1,2,3,4,5,7,8,12,15,16,24,31,32,rows_per_matrix,cols_per_matrix]:
    pset.addTerminal(int(c), int)
pset.addTerminal([], list)
pset.addTerminal(True, bool)
pset.addTerminal(False, bool)

pset.addEphemeralConstant("rand_small",
    functools.partial(random.randint, 0, 16), int)
