from deap import gp
import operator, random, functools
import hashlib
import math

# 全局常量
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
COUNTER_BITS = 32
MAX_COUNTER = (1 << COUNTER_BITS) - 1


INF = MAX_COUNTER + 1

# 算术操作符
def safe_add(a: int, b: int) -> int: return a + b
def safe_sub(a: int, b: int) -> int: return a - b
def safe_mul(a: int, b: int) -> int: return a * b
def safe_div(a: int, b: int) -> int: return a // (b if b != 0 else 1)
def abs_int(a: int) -> int:
    aa = int(a)
    return aa if aa >= 0 else -aa
def safe_mod(a: int, b: int) -> int:
    bb = int(b)
    if bb == 0:
        bb = 1
    return int(a) % bb
def safe_min(a: int, b: int) -> int:
    aa = int(a)
    bb = int(b)
    return aa if aa <= bb else bb
def safe_max(a: int, b: int) -> int:
    aa = int(a)
    bb = int(b)
    return aa if aa >= bb else bb
def sum3(a: int, b: int, c: int) -> int:
    return int(a) + int(b) + int(c)
def median3(a: int, b: int, c: int) -> int:
    aa, bb, cc = int(a), int(b), int(c)
    arr = [aa, bb, cc]
    arr.sort()
    return arr[1]

#字符串
def str_slice(s, start, end):
    return s[start:end]

def cnt_rdstate(e: str, i: int) -> bool:
    x, y, z = _loc(e, i)
    return bool(overflow_matrices[z][x][y])

def query_date(e:str,i:int) -> int:
    x, y, z = _loc(e, i)
    if overflow_matrices[z][x][y]:
        return INF
    return count_matrices[z][x][y]

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


# ---- 条件判断 ----
def if_then_else(cond, out1, out2):
    return out1 if cond else out2


def base_sel(mode: int, a: int, b: int, c: int) -> float:
    m = abs(mode) % 4
    if m == 0: return float(min(a, b, c))
    if m == 1: return float(max(a, b, c))
    if m == 2: return float(sorted([a, b, c])[1])
    return float((a + b + c) // 3)

def lt(a: int, b: int) -> bool: return a < b
def gt(a: int, b: int) -> bool: return a > b
def eq(a: int, b: int) -> bool: return a == b

# ---- 定义原语集 ----
pset = gp.PrimitiveSetTyped("QUERY", [str], float)
pset.renameArguments(ARG0="e")

# 数值计算
pset.addPrimitive(safe_add, [int, int], int)
pset.addPrimitive(safe_sub, [int, int], int)
pset.addPrimitive(safe_mul, [int, int], int)
pset.addPrimitive(safe_div, [int, int], int)
pset.addPrimitive(safe_mod, [int, int], int)
pset.addPrimitive(abs_int, [int], int)
pset.addPrimitive(safe_min, [int, int], int)
pset.addPrimitive(safe_max, [int, int], int)
pset.addPrimitive(sum3, [int, int, int], int)
pset.addPrimitive(median3, [int, int, int], int)
# 控制流
pset.addPrimitive(if_then_else, [bool, int, int], int)
pset.addPrimitive(base_sel, [int,int, int,int], float)

# 比较运算
pset.addPrimitive(lt, [int, int], bool)
pset.addPrimitive(gt, [int, int], bool)
pset.addPrimitive(eq, [int, int], bool)


pset.addPrimitive(query_date, [str, int], int)
pset.addPrimitive(cnt_rdstate, [str, int], bool)

#字符类型操作
pset.addPrimitive(str_slice,[str,int,int],str)

# 常量与辅助函数
pset.addTerminal(102, int)
pset.addTerminal(0.0, float)
pset.addTerminal(0, int)   # min
pset.addTerminal(1, int)   # max
pset.addTerminal(2, int)   # median
pset.addTerminal(3, int)   # avg
pset.addTerminal(4, int)
pset.addTerminal(8, int)
pset.addTerminal([], list)

# 给pset添加tuple类型终端
pset.addTerminal((0, 0), tuple)    # 示例：第0行第0列的索引元组
pset.addTerminal(True, bool)   # 布尔值True
pset.addTerminal(False, bool)  # 布尔值False

# 添加一个大数作为初始最小值
pset.addTerminal(INF, int)  # 模拟 float('inf')
#pset.addEphemeralConstant('rand_int', functools.partial(random.randint, 0, 102 - 1), int)
