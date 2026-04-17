# -*- coding: utf-8 -*-
import os
import random
import hashlib
import numpy as np
from glob import glob

# ===== dataset config baked from evolution run =====
DATASET_ROOT = '/data/8T/xgr/traces/univ2_trace'
PKTS = 20000
MAX_FILES = 8
START = 0
SHUFFLE = False
SEED = 20260327

def _to_pystr(x):
    try:
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode("utf-8", errors="ignore")
        if isinstance(x, np.generic):
            x = x.item()
        return str(x)
    except Exception:
        return str(x)

def find_flowid_files(root):
    patterns = ["*.flowid.npy", "*flowid*.npy", "univ2_pt*.npy", "*.npy"]
    files = []
    for pat in patterns:
        files.extend(glob(os.path.join(root, pat)))
    if not files:
        files = [p for p in glob(os.path.join(root, "*.npy"))
                 if ("omega" not in os.path.basename(p).lower())]
    return sorted(set(files))

def load_flow_stream(root, pkts, max_files, start, shuffle, seed):
    files = find_flowid_files(root)
    if not files:
        raise FileNotFoundError("no flowid files under: %s" % root)

    files = files[:max(1, int(max_files))]
    rng = random.Random(int(seed) & 0xFFFFFFFF)

    stream = []
    remain = int(pkts)

    for fp in files:
        try:
            arr = np.load(fp, mmap_mode="r")
        except Exception:
            arr = np.load(fp, allow_pickle=True)

        n = len(arr)
        if n <= int(start):
            continue
        if remain <= 0:
            break

        if shuffle:
            window_end = min(n, int(start) + max(remain * 20, remain))
            idx = list(range(int(start), window_end))
            rng.shuffle(idx)
            idx = idx[:remain]
            part = [_to_pystr(arr[i]) for i in idx]
        else:
            end = min(n, int(start) + remain)
            part = [_to_pystr(v) for v in arr[int(start):end]]

        stream.extend(part)
        remain = int(pkts) - len(stream)

    if not stream:
        raise RuntimeError("loaded 0 elements, please check pkts/start or file content.")

    return stream[:int(pkts)]

# ===== Count-Min Sketch parameters =====
try:
    from init_dex_language import *  # 同目录脚本导出模式
except Exception:
    from cmsketch_refactor.init_dex_language import *  # 包模式回退

rows_per_matrix = 102
cols_per_matrix = 102
planes = 3

count_matrices = [
    [[0 for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
    for _ in range(planes)
]
overflow_matrices = [
    [[False for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
    for _ in range(planes)
]

COUNTER_BITS = 32
MAX_COUNTER = (1 << COUNTER_BITS) - 1
INF = MAX_COUNTER + 1

# ===== helper functions =====
def str_concat(a, b):
    return a + b

def str_slice(s, start, end):
    return s[start:end]

def if_then_else(cond, out1, out2):
    return out1 if cond else out2

def base(a: int, b: int, c: int) -> float:
    return 0.0

def base_sel(mode: int, a: int, b: int, c: int) -> float:
    m = abs(int(mode)) % 4
    if m == 0:
        return float(min(int(a), int(b), int(c)))
    if m == 1:
        return float(max(int(a), int(b), int(c)))
    if m == 2:
        arr = [int(a), int(b), int(c)]
        arr.sort()
        return float(arr[1])
    return float((int(a) + int(b) + int(c)) // 3)

def lt(a: int, b: int) -> bool:
    return int(a) < int(b)

def gt(a: int, b: int) -> bool:
    return int(a) > int(b)

def eq(a: int, b: int) -> bool:
    return int(a) == int(b)

def safe_add(a: int, b: int) -> int:
    return int(a) + int(b)

def safe_sub(a: int, b: int) -> int:
    return int(a) - int(b)

def safe_mul(a: int, b: int) -> int:
    return int(a) * int(b)

def safe_div(a: int, b: int) -> int:
    return int(a) // (int(b) if int(b) != 0 else 1)

def safe_mod(a: int, b: int) -> int:
    bb = int(b)
    if bb == 0:
        bb = 1
    return int(a) % bb

def abs_int(a: int) -> int:
    aa = int(a)
    return aa if aa >= 0 else -aa

def safe_min(a: int, b: int) -> int:
    aa, bb = int(a), int(b)
    return aa if aa <= bb else bb

def safe_max(a: int, b: int) -> int:
    aa, bb = int(a), int(b)
    return aa if aa >= bb else bb

def sum3(a: int, b: int, c: int) -> int:
    return int(a) + int(b) + int(c)

def median3(a: int, b: int, c: int) -> int:
    arr = [int(a), int(b), int(c)]
    arr.sort()
    return arr[1]

# ---- init_dex (evolved) ----
def init_dex(e):
    return list_3(select_hash(0, e), safe_mod(hash_salt(0, e, 11), 102), 102, select_hash(1, e), safe_mod(hash_salt(1, e, 13), 102), 102, select_hash(2, e), safe_mod(hash_salt(2, e, 17), 102), 102)

def _loc(e: str, i: int):
    vars = init_dex(e)[i % 3]
    x = abs_int(int(vars[0])) % rows_per_matrix
    y = abs_int(int(vars[1])) % cols_per_matrix
    if isinstance(vars, (tuple, list)) and len(vars) > 2:
        z = abs_int(int(vars[2])) % planes
    else:
        z = i % planes
    return x, y, z

# ---- counter/state ops ----
def write_count(e: str, i: int, delta: int) -> int:
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
    count_matrices[z][x][y] += int(delta)
    if count_matrices[z][x][y] < 0:
        count_matrices[z][x][y] = 0
        overflow_matrices[z][x][y] = False
    elif count_matrices[z][x][y] > MAX_COUNTER:
        count_matrices[z][x][y] = MAX_COUNTER
        overflow_matrices[z][x][y] = True
    else:
        overflow_matrices[z][x][y] = False
    return count_matrices[z][x][y]

def update_state(e: str, i: int, st: bool) -> int:
    x, y, z = _loc(e, i)
    overflow_matrices[z][x][y] = bool(st)
    return 0

def query_count(e: str, i: int) -> int:
    x, y, z = _loc(e, i)
    return int(count_matrices[z][x][y])

def query_state(e: str, i: int) -> bool:
    x, y, z = _loc(e, i)
    return bool(overflow_matrices[z][x][y])

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

def cnt_rdstate(e: str, i: int) -> bool:
    x, y, z = _loc(e, i)
    return bool(overflow_matrices[z][x][y])

def query_date(e: str, i: int) -> int:
    x, y, z = _loc(e, i)
    if overflow_matrices[z][x][y]:
        return INF
    return int(count_matrices[z][x][y])

# ===== evolved functions =====
def update(e):
    return base(write_count(e, 0, safe_add(1, query_count(e, 0))), write_count(e, 1, safe_add(1, query_count(e, 1))), write_count(e, 2, safe_add(1, query_count(e, 2))))

def query(e):
    return base_sel(2, query_date(e, 0), query_date(e, 1), query_date(e, 2))

# ===== fallback synthetic data (only used if dataset loading fails) =====
def generate_test_data(size=10000):
    fruit_words = ["apple", "banana", "orange", "grape", "watermelon",
                   "strawberry", "pineapple", "mango", "peach", "pear",
                   "cherry", "kiwi", "lemon", "lime", "coconut",
                   "blueberry", "raspberry", "blackberry", "apricot", "plum"]
    animal_words = ["dog", "cat", "bird", "fish", "rabbit", "hamster",
                    "tiger", "lion", "elephant", "giraffe", "zebra",
                    "monkey", "bear", "wolf", "fox", "deer", "horse",
                    "cow", "sheep", "goat"]
    color_words = ["red", "blue", "green", "yellow", "black", "white",
                   "purple", "pink", "orange", "brown", "gray", "cyan",
                   "magenta", "gold", "silver"]
    base_words = fruit_words + animal_words + color_words

    test_data = []
    for i in range(int(size)):
        if i % 5 == 0:
            test_data.append("apple")
        elif i % 7 == 0:
            test_data.append("banana")
        elif i % 9 == 0:
            test_data.append("orange")
        elif i % 11 == 0:
            test_data.append("grape")
        else:
            gt = random.choice(["single", "with_num", "mixed_case", "combined"])
            if gt == "single":
                word = random.choice(base_words)
            elif gt == "with_num":
                word = "%s%d" % (random.choice(base_words), random.randint(1, 999))
            elif gt == "mixed_case":
                raw = random.choice(base_words)
                word = "".join(random.choice([c.upper(), c.lower()]) for c in raw)
            else:
                num_words = random.randint(2, 3)
                selected = random.sample(base_words, num_words)
                word = "_".join(selected)
            test_data.append(word)
    return test_data

if __name__ == "__main__":
    # 1) load real univ2 flow stream (preferred)
    try:
        data_stream = load_flow_stream(DATASET_ROOT, PKTS, MAX_FILES, START, SHUFFLE, SEED)
        print("[DATA] loaded univ2 flow stream: n=%d root=%s" % (len(data_stream), DATASET_ROOT))
    except Exception as ex:
        print("[WARN] failed to load univ2 flow stream, fallback to synthetic. reason:", ex)
        data_stream = generate_test_data(10000)
        print("[DATA] fallback synthetic stream: n=%d" % len(data_stream))

    # 2) update sketch
    for item in data_stream:
        update(item)

    # 3) compute true freq from this stream
    actual_freq = {}
    for item in data_stream:
        actual_freq[item] = actual_freq.get(item, 0) + 1

    # 4) evaluate top-K
    K = 30
    top_items = sorted(actual_freq.items(), key=lambda x: x[1], reverse=True)[:K]

    print()
    print("Top %d true freq vs CMS estimate:" % K)
    for item, actual in top_items:
        est = int(query(item))
        print("%-40s actual=%6d  est=%6d  abs_err=%6d" % (item, actual, est, abs(est - actual)))

    # 5) also test a few absent keys
    for item in ["dragonfruit", "starfruit", "passionfruit"]:
        est = int(query(item))
        print("%-40s actual=0     est=%6d  abs_err=%6d" % (item, est, abs(est)))
