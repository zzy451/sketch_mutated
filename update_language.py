from deap import gp
import hashlib
import operator, random
import functools

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

#counter 上限
COUNTER_BITS = 32#原来是32，用8更容易溢出
MAX_COUNTER = (1 << COUNTER_BITS) - 1

# query_date 里把 overflow 映射成“足够大的 int”
INF = MAX_COUNTER + 1

def init_dex(e):
    return [(0, 0, 0), (1, 0, 1), (2, 0, 2)]


def _loc(e: str, i: int):
    vars = init_dex(e)[i % 3]
    x = int(vars[0]) % 102
    y = int(vars[1]) % 102
    if isinstance(vars, (tuple, list)) and len(vars) > 2:
        z = int(vars[2]) % 3
    else:
        z = i % 3
    return x, y, z

def str_slice(s, start, end):
    return s[start:end]

def write_count(e:str,i:int,delta:int) -> int:
    x, y, z = _loc(e, i)
    vv = int(delta)
    if vv < 0:
        vv = 0
        overflow_matrices[z][x][y] = False
    elif vv > MAX_COUNTER:
        vv = MAX_COUNTER
        overflow_matrices[z][x][y] = True
    else:
        overflow_matrices[z][x][y] = False

    count_matrices[z][x][y] = vv
    return vv
def update_count(e: str, i: int, delta: int) -> int:
    x, y, z = _loc(e, i)
    count_matrices[z][x][y] +=delta
    if (count_matrices[z][x][y]<0):
        count_matrices[z][x][y] =0
        overflow_matrices[z][x][y] = False
    elif (count_matrices[z][x][y] > MAX_COUNTER):
        overflow_matrices[z][x][y] = True
        count_matrices[z][x][y] = MAX_COUNTER
    else :
        overflow_matrices[z][x][y] = False
    return count_matrices[z][x][y]

def update_state(e:str,i:int,st:bool) -> int:
    x, y, z = _loc(e, i)
    overflow_matrices[z][x][y] =st
    return 0

def query_count(e:str,i:int) -> int:
    x, y, z = _loc(e, i)
    return count_matrices[z][x][y]

def query_state(e:str,i:int) -> bool:
    x, y, z = _loc(e, i)
    return overflow_matrices[z][x][y]

def updatecount_if(cond: bool, e: str, i: int, delta: int) -> int:
    if not bool(cond):
        return query_count(e, i)
    return update_count(e, i, delta)

def writecount_if(cond: bool, e: str, i: int, v: int) -> int:
    if not bool(cond):
        return query_count(e, i)
    return write_count(e, i, v)


def writestate_if(cond: bool, e: str, i: int, st: bool) -> int:
    if not bool(cond):
        return query_state(e, i)
    return update_state(e, i, st)

def if_then_else(cond: bool, out1: int, out2: int) -> int:
    """if/else 结构"""
    return out1 if cond else out2

def base(a:int,b:int,c:int)->float:
    aa=a
    bb=b
    cc=c
    return 0.0

# 比较运算
def lt(a: int, b: int) -> bool: return a < b
def gt(a: int, b: int) -> bool: return a > b
def eq(a: int, b: int) -> bool: return a == b

# 算术操作符
def safe_add(a: int, b: int) -> int: return a + b
def safe_sub(a: int, b: int) -> int: return a - b
def safe_mul(a: int, b: int) -> int: return a * b
def safe_div(a: int, b: int) -> int: return a // (b if b != 0 else 1)
def safe_mod(a: int, b: int) -> int:
    bb = int(b)
    if bb == 0:
        bb = 1
    return int(a) % bb
def abs_int(a: int) -> int:
    aa = int(a)
    return aa if aa >= 0 else -aa

pset_update = gp.PrimitiveSetTyped("MAIN", [str], float)
pset_update.renameArguments(ARG0="e")

#字符类型操作
pset_update.addPrimitive(str_slice,[str,int,int],str)

#控制流
pset_update.addPrimitive(if_then_else, [bool, int, int], int)

# 矩阵操作与哈希函数
pset_update.addPrimitive(base, [int, int, int], float)
pset_update.addPrimitive(write_count,[str,int,int],int)
pset_update.addPrimitive(update_count,[str,int,int],int)
pset_update.addPrimitive(update_state,[str,int,bool],int)
pset_update.addPrimitive(query_count,[str,int],int)
pset_update.addPrimitive(query_state,[str,int],bool)
pset_update.addPrimitive(updatecount_if, [bool, str, int, int], int)
pset_update.addPrimitive(writecount_if, [bool, str, int, int], int)
pset_update.addPrimitive(writestate_if, [bool, str, int, bool], int)
# 比较运算
pset_update.addPrimitive(lt, [int, int], bool)
pset_update.addPrimitive(gt, [int, int], bool)
pset_update.addPrimitive(eq, [int, int], bool)

# 算术操作符
pset_update.addPrimitive(safe_add, [int, int], int)
pset_update.addPrimitive(safe_sub, [int, int], int)
pset_update.addPrimitive(safe_mul, [int, int], int)
pset_update.addPrimitive(safe_div, [int, int], int)
pset_update.addPrimitive(safe_mod, [int, int], int)
pset_update.addPrimitive(abs_int, [int], int)
# 常量
pset_update.addTerminal(0, int)
pset_update.addTerminal(1, int)
pset_update.addTerminal(2, int)
pset_update.addEphemeralConstant('rand_int', functools.partial(random.randint, 0, 102 - 1), int)
pset_update.addTerminal((0, 24), tuple)    # 示例：第0行第0列的索引元组
pset_update.addTerminal(-1, int)
pset_update.addTerminal([], list)
pset_update.addTerminal(0.0, float)

#布尔类型
pset_update.addTerminal(True, bool)   # 布尔值True
pset_update.addTerminal(False, bool)  # 布尔值False

pset_update.addEphemeralConstant("rand_i",
    functools.partial(random.randint, 0, 2), int)
pset_update.addEphemeralConstant("rand_delta",
    functools.partial(random.choice, [-1,1,2,-2,3,-3]), int)
pset_update.addEphemeralConstant("rand_small",
    functools.partial(random.randint, 0, 16), int)
