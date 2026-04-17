import sys
import copy
from typing import Callable
from collections import Counter
sys.path.append("/home/xgr/zhangzy")
from deap import gp, creator, base, tools, algorithms
import random
import ast
import math
import statistics
import subprocess
import os
import copy
import string
import traceback
import hashlib
import functools
import time
import argparse
from openai import OpenAI
from operator import attrgetter
from glob import glob
import numpy as np
import concurrent.futures as cf
import multiprocessing as mp
import json
import contextlib
import re
from datetime import datetime
import urllib.request
import urllib.error

def _apply_proxy_mode_to_stream_path(path: str, dataset_mode: str, proxy_mode: str) -> str:
    """
    给 stage1_fixed_stream 派生出按 proxy_mode 区分的固定流文件名。
    - real 模式：原样返回
    - proxy 模式：在文件名末尾加上 _<proxy_mode>
    """
    path = str(path or "")
    if not path:
        return ""

    if str(dataset_mode) != "proxy":
        return path

    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".npy"
    return f"{root}_{proxy_mode}{ext}"


# 导入三个原语集
hash_functions = [hashlib.md5, hashlib.sha1, hashlib.sha256]
MAX_COUNTER = (1 << 32) - 1
INF = 1 << 60

def set_seed(seed: int):
    """Set Python RNG seed for reproducibility (this project uses `random` only)."""
    random.seed(int(seed) & 0xFFFFFFFF)


from init_dex_language import pset as init_dex_pset
from update_language import pset_update
from query_language import pset as query_pset

# 创建适应度类和个体类（适应度越高越好：FitnessMax）
# 说明：DEAP 的 creator 在同一解释器里重复 create 会报错，所以这里做一次健壮处理。
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# 有些环境里可能已经创建过 Individual（例如交互式多次运行）。
# 为避免类型不一致导致的隐性问题：若已存在但不是最大化版本，则创建一个新的 IndividualMax 来使用。
if "Individual" not in creator.__dict__:
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    INDIVIDUAL_CLS = creator.Individual
else:
    try:
        if getattr(creator.Individual, "fitness", None) is not None and tuple(creator.Individual.fitness.weights) == (
                1.0,):
            INDIVIDUAL_CLS = creator.Individual
        else:
            if "IndividualMax" not in creator.__dict__:
                creator.create("IndividualMax", gp.PrimitiveTree, fitness=creator.FitnessMax)
            INDIVIDUAL_CLS = creator.IndividualMax
    except Exception:
        if "IndividualMax" not in creator.__dict__:
            creator.create("IndividualMax", gp.PrimitiveTree, fitness=creator.FitnessMax)
        INDIVIDUAL_CLS = creator.IndividualMax


class CMSketchEvaluator:
    def __init__(self,
                 dataset_root: str = "/data/8T/xgr/traces/univ2_trace",
                 pkts: int = 30000,
                 max_files: int = 1,
                 start: int = 0,
                 shuffle: bool = False,
                 seed: int = 0,
                 dataset_mode: str = "real",
                 proxy_mode: str = "proxy_balanced",
                 proxy_pool_mul: int = 8,
                 proxy_min_u: int = 2500,
                 hard_case_enabled: bool = False,
                 hard_case_stage_topk: int = 24,
                 hard_case_absent_topk: int = 12,
                 hard_case_scan_mul: int = 3,
                 hard_case_decay: float = 0.85,
                 hard_case_weight: float = 0.50,
                 fixed_stream_path: str = ""):
        """评估器：把原来的“随机字符串流”替换为从 univ2_trace 读取的 flowid 流（默认只用很少数据）。
        - dataset_root: univ2_trace 目录或其子目录 univ2_npy（里面应有 *.flowid.npy）
        - pkts: 总共取多少个包对应的 flowid（越小越快）
        - max_files: 最多读取多少个分片文件
        - start: 每个文件从第几个元素开始取（用于避开文件头）
        - shuffle: 是否在局部窗口里打乱采样（默认不打乱，取前 pkts 个）
        - seed: 打乱采样的随机种子
        """
        self.dataset_root = dataset_root
        self.pkts = int(pkts)
        self.max_files = int(max_files)
        self.start = int(start)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.dataset_mode = str(dataset_mode)
        self.proxy_mode = str(proxy_mode)
        self.proxy_pool_mul = max(2, int(proxy_pool_mul))
        self.proxy_min_u = max(1, int(proxy_min_u))
        self.fixed_stream_path = str(fixed_stream_path) if fixed_stream_path else ""
        self.hard_case_enabled = bool(hard_case_enabled)
        self.hard_case_stage_topk = max(1, int(hard_case_stage_topk))
        self.hard_case_absent_topk = max(1, int(hard_case_absent_topk))
        self.hard_case_scan_mul = max(1, int(hard_case_scan_mul))
        self.hard_case_decay = min(0.999, max(0.0, float(hard_case_decay)))
        self.hard_case_weight = min(0.95, max(0.0, float(hard_case_weight)))
        # 读取真实数据集的 flowid（字符串化后供 GP 的字符串/哈希原语使用）
        if self.fixed_stream_path and os.path.exists(self.fixed_stream_path):
            self.test_data = self._load_fixed_stream(self.fixed_stream_path)
            #print(f"[DATA_FIXED_LOAD] path={self.fixed_stream_path} pkts={len(self.test_data)}", flush=True)
        else:
            self.test_data = self._load_univ2_flow_stream()
            if self.fixed_stream_path:
                self._save_fixed_stream(self.fixed_stream_path, self.test_data)
             #   print(f"[DATA_FIXED_SAVE] path={self.fixed_stream_path} pkts={len(self.test_data)}", flush=True)
        self.expected_freq = self._calculate_expected_freq()
        self.U = len(set(self.test_data))
        self.U_ratio = self.U / max(1, len(self.test_data))
        #print(f"[DATA] pkts={len(self.test_data)} U={self.U} U_ratio={self.U_ratio:.4f}", flush=True)

        # 归一化尺度（E0）：用于把误差映射到 (0,1] 的适应度；在初始种群评估后由 evolve_cmsketch 设定。
        self.E0 = None
        self.eval_cache = {}
        self.fec_cache = {}
        self.fec_hits = 0
        self.fec_misses = 0

        self._build_kpart_views()
        self.kpart_keep_fracs = [0.85, 0.55, 1.0]
        self.kpart_query_limits = [96, 384, None]
        self.kpart_upd_min = [1.4, 1.7, 2.0]
        self.kpart_upd_penalty_scale = [40000, 30000, 30000]
        self.kpart_avg_err_thresh = [260.0, 420.0, None]
        self.kpart_avg_err_scale = [32.0, 24.0, 0.0]
        self.kpart_cut_penalty = [300000.0, 450000.0, None]
        base_offset = max(20000.0, float(self.pkts) * 2.0)
        self.kpart_stage_err_offset = [base_offset * 2.0, base_offset, 0.0]
        # ---- proxy 模式下：把 evaluator 的“主统计口径”重绑到新的分布型 K-part 流 ----
        if self.dataset_mode == "proxy":
            merged = []
            for seg in getattr(self, "kpart_stage_streams", []):
                merged.extend(seg)

            if merged:
                self.test_data = list(merged)
                self.expected_freq = self._calc_freq_from_stream(self.test_data)
                self.U = len(set(self.test_data))
                self.U_ratio = self.U / max(1, len(self.test_data))
                # print(
                #     f"[DATA_REBIND_PROXY] pkts={len(self.test_data)} "
                #     f"U={self.U} U_ratio={self.U_ratio:.4f}",
                #     flush=True
                # )

        # ---- fec probe / absent keys 也跟着新的主统计口径走 ----
        self.fec_probe_update_n = min(160, len(self.test_data))
        self.fec_probe_present_n = min(48, len(self.expected_freq))
        self.fec_probe_absent_n = 16
        self.fec_absent_keys = self._build_fec_absent_keys(self.fec_probe_absent_n)
        self.stage_eval_cache = {0: {}, 1: {}}
        self.lexicase_stage_cases = 3
        self.lexicase_absent_cases = 4
        self.lexicase_total_cases = self.lexicase_stage_cases + self.lexicase_absent_cases
        self.lexicase_default_bad = 1e18
        self.hard_case_version = 0
        self.hard_case_absent_bank = self._build_absent_keys("hardcase_absent", 96)
        self.hard_case_state = {
            "stage_cases": [[], [], []],
            "stage_bucket_cases": [{}, {}, {}],
            "absent_cases": [],
        }
        self.debug_stats = {
            "eval_calls": 0,
            "eval_cache_hits": 0,
            "fec_cache_hits": 0,
            "hard_illegal": 0,
            "real_write_zero": 0,
            "query_date_zero": 0,
            "penalty_dominates": 0,
            "early_return_cut": 0,
            "penalty_sum": 0.0,
            "query_error_sum": 0.0,
            "total_error_sum": 0.0,
            "nonconst_hash_idx_total": 0,
            "nonconst_path_idx_total": 0,
            "bad_write_context_total": 0,
            "hard_illegal_reasons": Counter(),
        }

    def _debug_reset(self):
        self.debug_stats = {
            "eval_calls": 0,
            "eval_cache_hits": 0,
            "fec_cache_hits": 0,
            "hard_illegal": 0,
            "real_write_zero": 0,
            "query_date_zero": 0,
            "penalty_dominates": 0,
            "early_return_cut": 0,
            "penalty_sum": 0.0,
            "query_error_sum": 0.0,
            "total_error_sum": 0.0,
            "nonconst_hash_idx_total": 0,
            "nonconst_path_idx_total": 0,
            "bad_write_context_total": 0,
            "hard_illegal_reasons": Counter(),
        }

    def _debug_snapshot(self):
        out = {}
        for k, v in self.debug_stats.items():
            if isinstance(v, Counter):
                out[k] = dict(v)
            else:
                out[k] = v
        return out

    def _find_flowid_files(self):

        roots = [self.dataset_root]
        sub = os.path.join(self.dataset_root, "univ2_npy")
        if os.path.isdir(sub):
            roots.append(sub)

        patterns = ["*.flowid.npy", "*flowid*.npy", "univ2_pt*.npy", "*.npy"]
        files = []
        for root in roots:
            for pat in patterns:
                files.extend(glob(os.path.join(root, pat)))

        # 兜底：若没有 flowid.npy，就找所有 .npy（排除 omega/label 等）
        if not files:
            for root in roots:
                files.extend([p for p in glob(os.path.join(root, "*.npy"))
                              if ("omega" not in os.path.basename(p).lower())])

        files = sorted(set(files))
        return files

    @staticmethod
    def _to_pystr(x):
        """把 numpy 元素/bytes/整数等统一转为 Python str，供 GP 的 hash/字符串原语使用。"""
        try:
            if isinstance(x, (bytes, np.bytes_)):
                return x.decode("utf-8", errors="ignore")
            if isinstance(x, np.generic):
                x = x.item()
            return str(x)
        except Exception:
            return str(x)

    def _load_fixed_stream(self, path: str):
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        return [self._to_pystr(x) for x in arr]

    def _save_fixed_stream(self, path: str, stream):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.save(path, np.asarray(list(stream), dtype=object), allow_pickle=True)

    def _load_real_stream(self):
        """从 univ2_trace 目录读取 flowid 作为测试数据流。
        目标：
        1) 多文件均匀取样
        2) shuffle 时优先拉高 U
        3) 对 5000 包这类小样本，自动多次重采样，选 U 更高的一次
        """
        files = self._find_flowid_files()
        if not files:
            raise FileNotFoundError(
                f"在 {self.dataset_root} 下找不到可用的数据文件（已尝试 *.flowid.npy、*flowid*.npy、univ2_pt*.npy、*.npy）。"
            )

        files = files[:max(1, self.max_files)]
        target = max(0, self.pkts)
        if target == 0:
            return []

        # 对小样本训练，设置一个“最低可接受 U”
        # 5000 包时，阈值大约会落在 1500~1666 左右
        target_min_u = min(target, max(1500, target // 3))

        # 只有 shuffle 时才值得多次重采样
        retry_times = 5 if self.shuffle else 1

        def _sample_once(seed_bias: int):
            rng = random.Random((int(self.seed) + 10007 * int(seed_bias)) & 0xFFFFFFFF)

            local_files = files[:]
            rng.shuffle(local_files)

            stream = []
            num_files = max(1, len(local_files))
            per_file = max(1, (target + num_files - 1) // num_files)  # ceil(target / num_files)

            # ---------- 第一轮：每个文件先均匀取 ----------
            for fp in local_files:
                if len(stream) >= target:
                    break

                try:
                    arr = np.load(fp, mmap_mode="r")
                except Exception:
                    arr = np.load(fp, allow_pickle=True)

                n = len(arr)
                if n <= self.start:
                    continue

                take = min(per_file, target - len(stream))
                if take <= 0:
                    break

                if self.shuffle:
                    # 局部大窗口随机抽，尽量把 U 拉高
                    window_end = min(n, self.start + max(take * 20000, 50000))
                    idx = list(range(self.start, window_end))
                    rng.shuffle(idx)
                    idx = idx[:take]
                    part = [self._to_pystr(arr[i]) for i in idx]
                else:
                    end = min(n, self.start + take)
                    part = [self._to_pystr(v) for v in arr[self.start:end]]

                stream.extend(part)

            # ---------- 第二轮：不够再补 ----------
            if len(stream) < target:
                remain = target - len(stream)

                for fp in local_files:
                    if remain <= 0:
                        break

                    try:
                        arr = np.load(fp, mmap_mode="r")
                    except Exception:
                        arr = np.load(fp)

                    n = len(arr)
                    if n <= self.start:
                        continue

                    take = min(remain, max(1, (remain + num_files - 1) // num_files))
                    if take <= 0:
                        break

                    if self.shuffle:
                        window_end = min(n, self.start + max(take * 20000, 50000))
                        idx = list(range(self.start, window_end))
                        rng.shuffle(idx)
                        idx = idx[:take]
                        part = [self._to_pystr(arr[i]) for i in idx]
                    else:
                        end = min(n, self.start + take)
                        part = [self._to_pystr(v) for v in arr[self.start:end]]

                    stream.extend(part)
                    remain = target - len(stream)

            return stream[:target]

        best_stream = None
        best_u = -1

        for t in range(retry_times):
            cand = _sample_once(t)
            if not cand:
                continue

            cand_u = len(set(cand))
            if cand_u > best_u:
                best_stream = cand
                best_u = cand_u

            if cand_u >= target_min_u:
                break

        if not best_stream:
            raise RuntimeError(
                f"从 {files[0]} 读取不到任何元素，请检查 pkts/start 参数，或确认文件内容不是空的。"
            )

        if retry_times > 1:
            print(
               # f"[DATA_RETRY] retry_times={retry_times} best_U={best_u} target_min_U={target_min_u}",
                flush=True
            )

        return best_stream

    def _build_candidate_pool(self, files, rng, pool_target):
        pool = []
        if pool_target <= 0:
            return pool

        local_files = files[:]
        rng.shuffle(local_files)

        num_files = max(1, len(local_files))
        per_file = max(1, (pool_target + num_files - 1) // num_files)

        for fp in local_files:
            if len(pool) >= pool_target:
                break

            try:
                arr = np.load(fp, mmap_mode="r")
            except Exception:
                arr = np.load(fp, allow_pickle=True)

            n = len(arr)
            if n <= self.start:
                continue

            take = min(per_file, pool_target - len(pool))
            if take <= 0:
                break

            if self.shuffle:
                window_end = min(n, self.start + max(take * 20000, 80000))
                idx = list(range(self.start, window_end))
                rng.shuffle(idx)
                idx = idx[:take]
                part = [self._to_pystr(arr[i]) for i in idx]
            else:
                end = min(n, self.start + take)
                part = [self._to_pystr(v) for v in arr[self.start:end]]

            pool.extend(part)

        return pool[:pool_target]

    def _make_proxy_stream_from_pool(self, pool, rng, mode=None, target=None):
        target = max(0, self.pkts if target is None else int(target))
        if target == 0:
            return []

        uniq = list(dict.fromkeys(pool))
        rng.shuffle(uniq)

        if not uniq:
            return []

        mode = self.proxy_mode if mode is None else str(mode)

        def _cap_u(u, low_ratio, high_ratio):
            lo = max(32, int(target * float(low_ratio)))
            hi = max(lo, int(target * float(high_ratio)))
            return min(len(uniq), max(lo, min(int(u), hi)))

        def _build_stream(base, hot, hot_k=None, weight_mode="uniform"):
            base = list(base)
            if not base:
                return []

            hot = list(hot) if hot else list(base)
            if hot_k is not None:
                hot = hot[:max(1, min(len(hot), int(hot_k)))]
            if not hot:
                hot = list(base[:1])

            remain = max(0, target - len(base))
            if remain <= 0:
                stream = base[:target]
                rng.shuffle(stream)
                return stream

            if weight_mode == "descending":
                weights = [len(hot) - i for i in range(len(hot))]
                extra = rng.choices(hot, weights=weights, k=remain)
            else:
                extra = [rng.choice(hot) for _ in range(remain)]

            stream = base + extra
            rng.shuffle(stream)
            return stream[:target]

        # 中等 U，但必须保留足够 replay，不能让小 target 基本全 unique
        if mode == "proxy_balanced":
            u_goal = _cap_u(target * 0.50, low_ratio=0.30, high_ratio=0.55)
            base = uniq[:u_goal]
            hot_k = min(len(base), max(32, int(len(base) * 0.20)))
            return _build_stream(base, base, hot_k=hot_k, weight_mode="uniform")

        # 明显头部流：更低 U、更强 replay，保证 stage0 能看到可学习的 head
        if mode == "proxy_head":
            u_goal = _cap_u(target * 0.22, low_ratio=0.12, high_ratio=0.28)
            base = uniq[:u_goal]
            hot_k = min(len(base), max(16, int(len(base) * 0.10)))
            return _build_stream(base, base, hot_k=hot_k, weight_mode="descending")

        # 碰撞代理：保留相似字符串分组，但不要把 U 顶到接近 target
        if mode == "proxy_collision":
            uniq_sorted = sorted(
                uniq,
                key=lambda s: (len(str(s)), str(s)[:4], str(s)[-4:])
            )
            u_goal = min(
                len(uniq_sorted),
                max(max(64, int(target * 0.35)), min(int(target * 0.50), 1200))
            )
            base = uniq_sorted[:u_goal]
            hot_k = min(len(base), max(32, int(len(base) * 0.18)))
            return _build_stream(base, base, hot_k=hot_k, weight_mode="uniform")

        rng.shuffle(pool)
        return pool[:target]

    def _load_proxy_stream(self):
        files = self._find_flowid_files()
        if not files:
            raise FileNotFoundError(
                f"在 {self.dataset_root} 下找不到可用的数据文件（已尝试 *.flowid.npy、*flowid*.npy、univ2_pt*.npy、*.npy）。"
            )

        files = files[:max(1, self.max_files)]
        target = max(0, self.pkts)
        if target == 0:
            return []

        best_stream = None
        best_u = -1
        retry_times = 5

        for t in range(retry_times):
            rng = random.Random((int(self.seed) + 10007 * t) & 0xFFFFFFFF)
            pool_target = max(target * self.proxy_pool_mul, 40000)

            pool = self._build_candidate_pool(files, rng, pool_target)
            cand = self._make_proxy_stream_from_pool(
                pool,
                rng,
                mode=self.proxy_mode,
                target=target,
            )

            cand_u = len(set(cand))
            if cand_u > best_u:
                best_stream = cand
                best_u = cand_u

            if cand_u >= min(self.proxy_min_u, target):
                break

        print(
            f"[DATA_PROXY] mode={self.proxy_mode} retry_times={retry_times} "
            f"best_U={best_u} target_min_U={min(self.proxy_min_u, target)}",
            flush=True
        )

        if not best_stream:
            raise RuntimeError("proxy stream 构造失败")
        return best_stream

    def _load_univ2_flow_stream(self):
        if self.dataset_mode == "proxy":
            return self._load_proxy_stream()
        return self._load_real_stream()

    def _generate_large_test_data(self, size):
        """生成大规模测试数据，包含更多种类的字符串"""
        # 1. 扩充基础词汇库，涵盖更多类别
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

        # 合并所有基础词汇
        base_words = []
        base_words = fruit_words + animal_words + color_words

        # 2. 生成测试数据，增加字符串多样性
        test_data = []
        for i in range(size):
            # 保留原有频率控制逻辑（特定词汇高频出现）
            if i % 5 == 0:
                test_data.append("apple")
            elif i % 7 == 0:
                test_data.append("banana")
            elif i % 9 == 0:
                test_data.append("orange")
            elif i % 11 == 0:
                test_data.append("grape")
            else:
                # 3. 生成多样化的字符串
                # 随机选择生成方式：纯词汇 / 词汇+数字 / 混合大小写 / 多词汇拼接
                generate_type = random.choice(["single", "with_num", "mixed_case", "combined"])

                if generate_type == "single":
                    # 纯基础词汇
                    word = random.choice(base_words)
                elif generate_type == "with_num":
                    # 词汇 + 随机数字（1-3位）
                    word = f"{random.choice(base_words)}{random.randint(1, 999)}"
                elif generate_type == "mixed_case":
                    # 混合大小写的词汇
                    raw_word = random.choice(base_words)
                    word = ''.join(random.choice([c.upper(), c.lower()]) for c in raw_word)
                else:  # combined
                    # 2-3个词汇拼接
                    num_words = random.randint(2, 3)
                    selected_words = random.sample(base_words, num_words)
                    word = "_".join(selected_words)

                test_data.append(word)

        return test_data

    def _calculate_expected_freq(self):
        """计算预期频率"""
        freq = {}
        for item in self.test_data:
            freq[item] = freq.get(item, 0) + 1

        # 添加一些不存在的元素用于测试
        freq["dragonfruit"] = 0
        freq["starfruit"] = 0
        freq["passionfruit"] = 0

        return freq

    def _tree_stats(self, tree):
        counts = Counter()
        root = None

        if tree and len(tree) > 0:
            root = getattr(tree[0], "name", str(tree[0]))

        for node in tree:
            name = getattr(node, "name", None)
            if name is None:
                continue
            counts[name] += 1

        return {
            "root": root,
            "counts": counts,
            "size": len(tree),
        }

    def analyze_init_tree(self, init_dex_tree):
        stats = self._tree_stats(init_dex_tree)

        hash_names = {"select_hash", "hash_salt", "hash_on_slice"}
        forbidden = {
            "update_count", "write_count", "updatecount_if", "writecount_if",
            "update_state", "writestate_if",
            "query_count", "query_state",
            "query_date", "cnt_rdstate",
        }

        return {
            "root": stats["root"],
            "root_ok": (stats["root"] == "list_3"),
            "hash_calls": sum(stats["counts"][n] for n in hash_names),
            "forbidden_hits": {
                n: stats["counts"][n]
                for n in forbidden
                if stats["counts"][n] > 0
            },
            "size": stats["size"],
        }

    def analyze_update_tree(self, update_tree):
        stats = self._tree_stats(update_tree)

        writer_names = {
            "update_count", "write_count",
            "updatecount_if", "writecount_if",
            "update_state", "writestate_if",
        }

        forbidden = {
            "query_date", "cnt_rdstate",
        }

        return {
            "root": stats["root"],
            "root_ok": (stats["root"] == "base"),
            "write_calls": sum(stats["counts"][n] for n in writer_names),
            "forbidden_hits": {
                n: stats["counts"][n]
                for n in forbidden
                if stats["counts"][n] > 0
            },
            "size": stats["size"],
        }

    def analyze_query_tree(self, query_tree):
        stats = self._tree_stats(query_tree)

        read_names = {"query_date"}
        forbidden = {
            "update_count", "write_count",
            "updatecount_if", "writecount_if",
            "update_state", "writestate_if",
            "query_count", "query_state",
        }

        return {
            "root": stats["root"],
            "root_ok": (stats["root"] == "base_sel"),
            "read_calls": sum(stats["counts"][n] for n in read_names),
            "state_reads": int(stats["counts"]["cnt_rdstate"]),
            "forbidden_hits": {
                n: stats["counts"][n]
                for n in forbidden
                if stats["counts"][n] > 0
            },
            "size": stats["size"],
        }

    def _ast_term(self, text, const=None):
        return {
            "kind": "term",
            "text": text,
            "const": const,
            "pure": True,
        }

    def _ast_text(self, node):
        if node["kind"] == "term":
            return node["text"]
        return f'{node["name"]}(' + ", ".join(self._ast_text(ch) for ch in node["children"]) + ')'

    def _tree_to_ast(self, tree):
        def walk(pos):
            node = tree[pos]
            arity = int(getattr(node, "arity", 0))

            if arity == 0:
                value = getattr(node, "value", None)
                name0 = getattr(node, "name", None)

                const = None
                text = None

                # 1) 先识别输入变量 ARG0 -> e
                if name0 == "ARG0" or value == "ARG0":
                    text = "e"

                # 2) 纯常量终端
                elif isinstance(value, (bool, int, float, tuple, list)):
                    const = value
                    text = repr(value)

                # 3) 字符串常量：普通字符串保留 repr；Terminal 对象名字优先用名字
                elif isinstance(value, str):
                    if value == "ARG0":
                        text = "e"
                    else:
                        text = repr(value)

                # 4) 其次看 node.name
                elif isinstance(name0, str):
                    if name0 == "ARG0":
                        text = "e"
                    elif name0 == "True":
                        const = True
                        text = "True"
                    elif name0 == "False":
                        const = False
                        text = "False"
                    else:
                        text = name0

                # 5) 最后兜底，避免把对象地址直接打出来
                if text is None:
                    raw_text = str(node)
                    if raw_text == "ARG0":
                        text = "e"
                    elif raw_text == "True":
                        const = True
                        text = "True"
                    elif raw_text == "False":
                        const = False
                        text = "False"
                    else:
                        text = raw_text

                return self._ast_term(text, const=const), pos + 1

            name = getattr(node, "name", str(node))
            children = []
            nxt = pos + 1
            for _ in range(arity):
                ch, nxt = walk(nxt)
                children.append(ch)

            return {
                "kind": "prim",
                "name": name,
                "children": children,
                "pure": None,
                "const": None,
            }, nxt

        root, end = walk(0)
        if end != len(tree):
            raise ValueError(f"tree parse not fully consumed: end={end}, len={len(tree)}")
        return root

    def _const_of(self, node):
        return node.get("const", None)

    def _is_impure_primitive(self, name: str) -> bool:
        return name in {
            "update_count", "write_count",
            "updatecount_if", "writecount_if",
            "update_state", "writestate_if",
        }

    def _const_eval_primitive(self, name, vals):
        try:
            if name == "abs_int" and len(vals) == 1:
                return abs(int(vals[0]))

            if name == "safe_add" and len(vals) == 2:
                return int(vals[0]) + int(vals[1])

            if name == "safe_sub" and len(vals) == 2:
                return int(vals[0]) - int(vals[1])

            if name == "safe_mul" and len(vals) == 2:
                return int(vals[0]) * int(vals[1])

            if name == "safe_div" and len(vals) == 2:
                b = int(vals[1])
                return int(vals[0]) // (b if b != 0 else 1)

            if name == "safe_mod" and len(vals) == 2:
                b = int(vals[1])
                if b == 0:
                    b = 1
                return int(vals[0]) % b

            if name == "safe_min" and len(vals) == 2:
                return min(int(vals[0]), int(vals[1]))

            if name == "safe_max" and len(vals) == 2:
                return max(int(vals[0]), int(vals[1]))

            if name == "sum3" and len(vals) == 3:
                return int(vals[0]) + int(vals[1]) + int(vals[2])

            if name == "median3" and len(vals) == 3:
                arr = [int(vals[0]), int(vals[1]), int(vals[2])]
                arr.sort()
                return arr[1]

            if name == "lt" and len(vals) == 2:
                return int(vals[0]) < int(vals[1])

            if name == "gt" and len(vals) == 2:
                return int(vals[0]) > int(vals[1])

            if name == "eq" and len(vals) == 2:
                return vals[0] == vals[1]

            if name == "if_then_else" and len(vals) == 3:
                return vals[1] if bool(vals[0]) else vals[2]

            if name == "if_then_else_int" and len(vals) == 3:
                return int(vals[1]) if bool(vals[0]) else int(vals[2])

            if name == "base_sel" and len(vals) == 4:
                mode = abs(int(vals[0])) % 4
                a, b, c = int(vals[1]), int(vals[2]), int(vals[3])
                if mode == 0:
                    return float(min(a, b, c))
                if mode == 1:
                    return float(max(a, b, c))
                if mode == 2:
                    arr = [a, b, c]
                    arr.sort()
                    return float(arr[1])
                return float((a + b + c) // 3)

        except Exception:
            return None

        return None

    def _simplify_ast(self, node):
        if node["kind"] == "term":
            return node

        name = node["name"]
        children = [self._simplify_ast(ch) for ch in node["children"]]
        pure = (not self._is_impure_primitive(name)) and all(ch["pure"] for ch in children)

        def txt(i):
            return self._ast_text(children[i])

        def cval(i):
            return self._const_of(children[i])

        def is_const(i):
            return cval(i) is not None

        def make_prim(pname, pchildren, pure_override=None):
            return {
                "kind": "prim",
                "name": pname,
                "children": pchildren,
                "pure": pure if pure_override is None else bool(pure_override),
                "const": None,
            }

        def norm_mod3(ch):
            cv = self._const_of(ch)
            if cv is None:
                return ch
            try:
                v = int(cv) % 3
                return self._ast_term(str(v), const=v)
            except Exception:
                return ch

        # ---- init_dex hash 编号规范化：按 init_dex 原语集里的 3 个 hash 函数做 mod 3 ----
        if name == "select_hash" and len(children) == 2:
            children[0] = norm_mod3(children[0])

        elif name == "hash_salt" and len(children) == 3:
            children[0] = norm_mod3(children[0])

        elif name == "hash_on_slice" and len(children) == 4:
            children[0] = norm_mod3(children[0])

        # ---- update/query 路径参数规范化：按 create_pset_update/create_pset_query 的 i % 3 ----
        if name in {"query_count", "query_state", "query_date", "cnt_rdstate"} and len(children) == 2:
            children[1] = norm_mod3(children[1])

        elif name in {"update_count", "write_count", "update_state"} and len(children) == 3:
            children[1] = norm_mod3(children[1])

        elif name in {"updatecount_if", "writecount_if", "writestate_if"} and len(children) == 4:
            children[2] = norm_mod3(children[2])

        # ---- 条件折叠 ----
        if name in {"if_then_else", "if_then_else_int"} and len(children) == 3:
            if cval(0) is True:
                return children[1]
            if cval(0) is False:
                return children[2]
            if pure and txt(1) == txt(2):
                return children[1]

        # ---- sketch-aware 条件写原语折叠 ----
        if name == "updatecount_if" and len(children) == 4:
            if cval(0) is False:
                return make_prim("query_count", [children[1], children[2]], pure_override=False)
            if cval(0) is True:
                return make_prim("update_count", [children[1], children[2], children[3]], pure_override=False)

        if name == "writecount_if" and len(children) == 4:
            if cval(0) is False:
                return make_prim("query_count", [children[1], children[2]], pure_override=False)
            if cval(0) is True:
                return make_prim("write_count", [children[1], children[2], children[3]], pure_override=False)

        if name == "writestate_if" and len(children) == 4:
            if cval(0) is False:
                return make_prim("query_state", [children[1], children[2]], pure_override=False)
            if cval(0) is True:
                return make_prim("update_state", [children[1], children[2], children[3]], pure_override=False)

        # ---- 一元纯函数 ----
        if name == "abs_int" and len(children) == 1:
            if children[0]["kind"] == "prim" and children[0]["name"] == "abs_int":
                return children[0]
            if is_const(0):
                v = self._const_eval_primitive(name, [cval(0)])
                if v is not None:
                    return self._ast_term(repr(v), const=v)

        # ---- 二元简化 ----
        if name == "safe_add" and len(children) == 2:
            if is_const(0) and int(cval(0)) == 0:
                return children[1]
            if is_const(1) and int(cval(1)) == 0:
                return children[0]

        if name == "safe_sub" and len(children) == 2:
            if is_const(1) and int(cval(1)) == 0:
                return children[0]
            if pure and txt(0) == txt(1):
                return self._ast_term("0", const=0)

        if name == "safe_mul" and len(children) == 2:
            if is_const(0) and int(cval(0)) == 0:
                return self._ast_term("0", const=0)
            if is_const(1) and int(cval(1)) == 0:
                return self._ast_term("0", const=0)
            if is_const(0) and int(cval(0)) == 1:
                return children[1]
            if is_const(1) and int(cval(1)) == 1:
                return children[0]

        if name == "safe_div" and len(children) == 2:
            if is_const(0) and int(cval(0)) == 0:
                return self._ast_term("0", const=0)
            if is_const(1) and int(cval(1)) == 1:
                return children[0]

        if name == "safe_mod" and len(children) == 2:
            if is_const(0) and int(cval(0)) == 0:
                return self._ast_term("0", const=0)
            if is_const(1) and int(cval(1)) == 1:
                return self._ast_term("0", const=0)

            # safe_mod(safe_mod(x, m), m) -> safe_mod(x, m)
            if (
                    children[0]["kind"] == "prim"
                    and children[0]["name"] == "safe_mod"
                    and len(children[0]["children"]) == 2
                    and self._ast_text(children[0]["children"][1]) == txt(1)
            ):
                return make_prim("safe_mod", [children[0]["children"][0], children[1]], pure_override=pure)

        if name in {"safe_min", "safe_max", "eq"} and len(children) == 2 and pure and txt(0) == txt(1):
            if name == "eq":
                return self._ast_term("True", const=True)
            return children[0]

        if name in {"lt", "gt"} and len(children) == 2 and pure and txt(0) == txt(1):
            return self._ast_term("False", const=False)

        # ---- base_sel 结构化简 ----
        if name == "base_sel" and len(children) == 4:
            # base_sel(m, x, x, x) -> x
            if txt(1) == txt(2) == txt(3):
                return children[1]

            if is_const(0):
                mode = abs(int(cval(0))) % 4

                # min
                if mode == 0:
                    if txt(1) == txt(2):
                        return make_prim("safe_min", [children[1], children[3]], pure_override=pure)
                    if txt(1) == txt(3):
                        return make_prim("safe_min", [children[1], children[2]], pure_override=pure)
                    if txt(2) == txt(3):
                        return make_prim("safe_min", [children[1], children[2]], pure_override=pure)

                # max
                if mode == 1:
                    if txt(1) == txt(2):
                        return make_prim("safe_max", [children[1], children[3]], pure_override=pure)
                    if txt(1) == txt(3):
                        return make_prim("safe_max", [children[1], children[2]], pure_override=pure)
                    if txt(2) == txt(3):
                        return make_prim("safe_max", [children[1], children[2]], pure_override=pure)

                # median
                if mode == 2:
                    if txt(1) == txt(2):
                        return children[1]
                    if txt(1) == txt(3):
                        return children[1]
                    if txt(2) == txt(3):
                        return children[2]

        # ---- 常量折叠 ----
        vals = [self._const_of(ch) for ch in children]
        if pure and all(v is not None for v in vals):
            folded = self._const_eval_primitive(name, vals)
            if folded is not None:
                return self._ast_term(repr(folded), const=folded)

        # ---- 交换律/规范化：仅对纯子树做 ----
        if pure and name in {"safe_add", "safe_mul", "safe_min", "safe_max", "eq", "sum3", "median3"}:
            children = sorted(children, key=self._ast_text)

        return {
            "kind": "prim",
            "name": name,
            "children": children,
            "pure": pure,
            "const": None,
        }

    def _canonical_tree_str(self, tree):
        try:
            root = self._tree_to_ast(tree)
            simp = self._simplify_ast(root)
            return self._ast_text(simp)
        except Exception:
            return str(tree)

    def _canonical_triplet_key(self, init_dex_tree, update_tree, query_tree, stage_idx=None):
        init_key = self._canonical_tree_str(init_dex_tree)
        update_key = self._canonical_tree_str(update_tree)
        query_key = self._canonical_tree_str(query_tree)

        if stage_idx is None:
            return (init_key, update_key, query_key)
        return (int(stage_idx), init_key, update_key, query_key)

    def _ast_collect_names(self, node):
        names = Counter()
        depends_on_e = False

        def walk(nd):
            nonlocal depends_on_e
            if nd["kind"] == "term":
                if nd.get("text") == "e":
                    depends_on_e = True
                return

            names[nd["name"]] += 1
            for ch in nd["children"]:
                walk(ch)

        walk(node)
        return {
            "names": names,
            "depends_on_e": depends_on_e,
        }

    def _ast_legality_check(self, which: str, root):
        info = self._ast_collect_names(root)
        names = info["names"]
        hard_reasons = []

        if which == "init":
            bad = {
                "update_count", "write_count",
                "updatecount_if", "writecount_if",
                "update_state", "writestate_if",
                "query_count", "query_state",
                "query_date", "cnt_rdstate",
            }
            if any(names[n] > 0 for n in bad):
                hard_reasons.append("init_contains_counter_or_state_ops")

        elif which == "update":
            bad = {
                "query_date", "cnt_rdstate",
            }
            if any(names[n] > 0 for n in bad):
                hard_reasons.append("update_contains_query_only_ops")

        elif which == "query":
            bad = {
                "update_count", "write_count",
                "updatecount_if", "writecount_if",
                "update_state", "writestate_if",
                "query_count", "query_state",
            }
            if any(names[n] > 0 for n in bad):
                hard_reasons.append("query_contains_write_or_update_reads")

        return {
            "hard_illegal": bool(hard_reasons),
            "reasons": hard_reasons,
            "depends_on_e": info["depends_on_e"],
            "names": names,
        }

    def _ast_effect_summary(self, root):
        info = self._ast_collect_names(root)
        names = info["names"]

        return {
            "depends_on_e": info["depends_on_e"],
            "real_write_calls": (
                int(names["update_count"]) +
                int(names["write_count"]) +
                int(names["update_state"])
            ),
            "conditional_write_calls": (
                int(names["updatecount_if"]) +
                int(names["writecount_if"]) +
                int(names["writestate_if"])
            ),
            "query_date_calls": int(names["query_date"]),
            "counter_read_calls": int(names["query_count"]),
            "state_read_calls": int(names["query_state"]) + int(names["cnt_rdstate"]),
        }

    def _ast_pattern_summary(self, which: str, root):
        idx_pos_map = {
            "query_count": 1,
            "query_state": 1,
            "query_date": 1,
            "cnt_rdstate": 1,
            "update_count": 1,
            "write_count": 1,
            "update_state": 1,
            "updatecount_if": 2,
            "writecount_if": 2,
            "writestate_if": 2,
        }

        hash_names = {"select_hash", "hash_salt", "hash_on_slice"}

        write_names = {
            "update_count", "write_count", "update_state",
            "updatecount_if", "writecount_if", "writestate_if",
        }

        bad_write_parents = {
            "lt", "gt", "eq",
            "safe_add", "safe_sub", "safe_mul", "safe_div", "safe_mod",
            "safe_min", "safe_max", "sum3", "median3",
            "str_slice",
        }

        out = {
            "nonconst_hash_idx": 0,
            "nonconst_path_idx": 0,
            "bad_write_context": 0,
        }

        def walk(nd, parent_name=None, child_pos=None):
            if nd["kind"] == "term":
                return

            name = nd["name"]

            # init: hash 编号应该是固定的小常量
            if which == "init" and name in hash_names:
                idx_node = nd["children"][0]
                cv = self._const_of(idx_node)
                if cv is None:
                    out["nonconst_hash_idx"] += 1

            # update/query: 路径号 i 应该是固定的小常量 0/1/2
            if which in {"update", "query"} and name in idx_pos_map:
                idx_node = nd["children"][idx_pos_map[name]]
                cv = self._const_of(idx_node)
                if cv is None:
                    out["nonconst_path_idx"] += 1
                else:
                    try:
                        if int(cv) not in (0, 1, 2):
                            out["nonconst_path_idx"] += 1
                    except Exception:
                        out["nonconst_path_idx"] += 1

            # update: 写副作用不应该藏在比较/算术/切片/条件位里
            if which == "update" and name in write_names:
                if parent_name in bad_write_parents:
                    out["bad_write_context"] += 1
                elif parent_name in {"if_then_else", "if_then_else_int"} and child_pos == 0:
                    out["bad_write_context"] += 1

            for i, ch in enumerate(nd["children"]):
                walk(ch, name, i)

        walk(root)
        return out

    def _norm_fitness(self, total_error: float) -> float:
        """把误差映射为 (0,1] 的归一化适应度；越大越好。
        默认使用：fitness = 1 / (1 + error / E0)
        其中 E0 在初始种群评估后设置为“典型误差量级”（例如中位数）。
        """
        E0 = float(self.E0) if self.E0 not in (None, 0) else 1.0
        err = float(total_error)
        # 防止异常值/负值
        if not math.isfinite(err):
            return 0.0
        if err < 0:
            err = 0.0
        return 1.0 / (1.0 + (err / E0))

    def _build_absent_keys(self, tag: str, k: int):
        seen = set(self.expected_freq.keys())
        bases = list(self.expected_freq.keys())[:max(16, k * 4)]
        if not bases:
            bases = [f"{tag}_base_{i}" for i in range(max(1, k))]

        out = []
        i = 0
        while len(out) < k:
            base = str(bases[i % len(bases)])
            cand = f"{base}__{tag}__{i}"
            if cand not in seen:
                out.append(cand)
                seen.add(cand)
            i += 1
        return out

    def _build_fec_absent_keys(self, k: int):
        return self._build_absent_keys("fec_absent", k)

    def _empty_hard_case_state(self):
        return {
            "stage_cases": [[], [], []],
            "stage_bucket_cases": [{}, {}, {}],
            "absent_cases": [],
        }

    def _infer_failure_buckets(self, init_pat, update_pat, query_pat, update_eff, query_eff):
        buckets = []

        if int(update_eff.get("real_write_calls", 0)) == 0:
            buckets.append("real_write_zero")

        if int(query_eff.get("query_date_calls", 0)) == 0:
            buckets.append("query_date_zero")

        if int(init_pat.get("nonconst_hash_idx", 0)) > 0:
            buckets.append("nonconst_hash")

        if int(update_pat.get("nonconst_path_idx", 0)) > 0 or int(query_pat.get("nonconst_path_idx", 0)) > 0:
            buckets.append("nonconst_path")

        if int(update_pat.get("bad_write_context", 0)) > 0:
            buckets.append("bad_write_ctx")

        if not buckets:
            buckets.append("generic")

        # 去重但保序
        out = []
        seen = set()
        for b in buckets:
            bb = str(b)
            if bb not in seen:
                out.append(bb)
                seen.add(bb)
        return out

    def _pick_bucketed_replay_cases(self, stage_idx: int, replay_quota: int):
        replay_quota = max(0, int(replay_quota))
        if replay_quota <= 0:
            return []

        raw_stage_buckets = self.hard_case_state.get("stage_bucket_cases", [{}, {}, {}])
        cur = raw_stage_buckets[stage_idx] if stage_idx < len(raw_stage_buckets) else {}
        if not isinstance(cur, dict) or not cur:
            return []

        bucket_names = sorted(
            [str(k) for k in cur.keys()],
            key=lambda bk: (-len(cur.get(bk, [])), bk)
        )

        ptr = {bk: 0 for bk in bucket_names}
        chosen = []
        seen_items = set()

        progress = True
        while len(chosen) < replay_quota and progress:
            progress = False
            for bk in bucket_names:
                lst = cur.get(bk, [])
                while ptr[bk] < len(lst):
                    rec = lst[ptr[bk]]
                    ptr[bk] += 1
                    item = str(rec.get("item", ""))
                    if not item or item in seen_items:
                        continue
                    chosen.append(rec)
                    seen_items.add(item)
                    progress = True
                    break
                if len(chosen) >= replay_quota:
                    break

        return chosen

    def _clear_eval_caches(self):
        self.eval_cache = {}
        self.fec_cache = {}
        self.stage_eval_cache = {0: {}, 1: {}}

    def _sanitize_case_record(self, rec, default_expected=0):
        item = str(rec.get("item", ""))
        if not item:
            return None
        try:
            expected = int(rec.get("expected", default_expected))
        except Exception:
            expected = int(default_expected)
        try:
            score = float(rec.get("score", rec.get("err", 0.0)))
        except Exception:
            score = 0.0
        if score < 0:
            score = 0.0

        bucket = str(rec.get("bucket", "generic") or "generic")

        return {
            "item": item,
            "expected": expected,
            "score": score,
            "bucket": bucket,
        }

    def _dedup_case_records(self, records, keep_k: int, default_expected=0):
        mp_cases = {}
        for rec in records:
            rr = self._sanitize_case_record(rec, default_expected=default_expected)
            if rr is None:
                continue
            old = mp_cases.get(rr["item"])
            if (old is None) or (rr["score"] > old["score"]):
                mp_cases[rr["item"]] = rr

        vals = list(mp_cases.values())
        vals.sort(key=lambda d: (-float(d["score"]), str(d["item"])))
        return vals[:max(1, int(keep_k))]

    def export_hard_case_state(self):
        return {
            "version": int(self.hard_case_version),
            "stage_cases": copy.deepcopy(self.hard_case_state.get("stage_cases", [[], [], []])),
            "stage_bucket_cases": copy.deepcopy(self.hard_case_state.get("stage_bucket_cases", [{}, {}, {}])),
            "absent_cases": copy.deepcopy(self.hard_case_state.get("absent_cases", [])),
        }

    def import_hard_case_state(self, state):
        if not self.hard_case_enabled:
            self.hard_case_state = self._empty_hard_case_state()
            self.hard_case_version = 0
            self._clear_eval_caches()
            return

        if not state:
            self.hard_case_state = self._empty_hard_case_state()
            self.hard_case_version = 0
            self._clear_eval_caches()
            return

        raw_stage_cases = state.get("stage_cases", [[], [], []])
        norm_stage_cases = []
        for s in range(3):
            cur = raw_stage_cases[s] if s < len(raw_stage_cases) else []
            norm_stage_cases.append(
                self._dedup_case_records(cur, self.hard_case_stage_topk, default_expected=0)
            )

        raw_stage_bucket_cases = state.get("stage_bucket_cases", [{}, {}, {}])
        norm_stage_bucket_cases = []
        for s in range(3):
            cur_map = raw_stage_bucket_cases[s] if s < len(raw_stage_bucket_cases) else {}
            if not isinstance(cur_map, dict):
                cur_map = {}
            norm_map = {}
            for bk, recs in cur_map.items():
                norm_map[str(bk)] = self._dedup_case_records(
                    recs,
                    self.hard_case_stage_topk,
                    default_expected=0,
                )
            norm_stage_bucket_cases.append(norm_map)

        norm_absent = self._dedup_case_records(
            state.get("absent_cases", []),
            self.hard_case_absent_topk,
            default_expected=0,
        )

        self.hard_case_state = {
            "stage_cases": norm_stage_cases,
            "stage_bucket_cases": norm_stage_bucket_cases,
            "absent_cases": norm_absent,
        }
        self.hard_case_version = int(state.get("version", 0))
        self._clear_eval_caches()

    def _merge_hard_case_state(self, mined_state):
        if not self.hard_case_enabled or not mined_state:
            return

        old_state = self.export_hard_case_state()

        merged_stage_cases = []
        for s in range(3):
            prev = []
            for rec in self.hard_case_state.get("stage_cases", [[], [], []])[s]:
                rr = self._sanitize_case_record(rec, default_expected=0)
                if rr is None:
                    continue
                rr["score"] = float(rr["score"]) * float(self.hard_case_decay)
                prev.append(rr)

            cur_new = mined_state.get("stage_cases", [[], [], []])
            if s < len(cur_new):
                prev.extend(cur_new[s])

            merged_stage_cases.append(
                self._dedup_case_records(prev, self.hard_case_stage_topk, default_expected=0)
            )

        merged_stage_bucket_cases = []
        old_stage_bucket_cases = self.hard_case_state.get("stage_bucket_cases", [{}, {}, {}])
        new_stage_bucket_cases = mined_state.get("stage_bucket_cases", [{}, {}, {}])

        for s in range(3):
            prev_map = {}

            old_map = old_stage_bucket_cases[s] if s < len(old_stage_bucket_cases) else {}
            if isinstance(old_map, dict):
                for bk, recs in old_map.items():
                    keep = []
                    for rec in recs:
                        rr = self._sanitize_case_record(rec, default_expected=0)
                        if rr is None:
                            continue
                        rr["score"] = float(rr["score"]) * float(self.hard_case_decay)
                        keep.append(rr)
                    prev_map[str(bk)] = keep

            cur_map = new_stage_bucket_cases[s] if s < len(new_stage_bucket_cases) else {}
            if isinstance(cur_map, dict):
                for bk, recs in cur_map.items():
                    key = str(bk)
                    prev_map.setdefault(key, [])
                    prev_map[key].extend(recs)

            norm_map = {}
            for bk, recs in prev_map.items():
                norm_map[str(bk)] = self._dedup_case_records(
                    recs,
                    self.hard_case_stage_topk,
                    default_expected=0,
                )
            merged_stage_bucket_cases.append(norm_map)

        prev_absent = []
        for rec in self.hard_case_state.get("absent_cases", []):
            rr = self._sanitize_case_record(rec, default_expected=0)
            if rr is None:
                continue
            rr["score"] = float(rr["score"]) * float(self.hard_case_decay)
            prev_absent.append(rr)

        prev_absent.extend(mined_state.get("absent_cases", []))
        merged_absent = self._dedup_case_records(
            prev_absent,
            self.hard_case_absent_topk,
            default_expected=0,
        )

        self.hard_case_state = {
            "stage_cases": merged_stage_cases,
            "stage_bucket_cases": merged_stage_bucket_cases,
            "absent_cases": merged_absent,
        }

        if json.dumps(old_state, sort_keys=True, ensure_ascii=False) != json.dumps(self.export_hard_case_state(), sort_keys=True, ensure_ascii=False):
            self.hard_case_version += 1
            self._clear_eval_caches()

    def _get_stage_eval_items(self, stage_idx: int, expected_freq):
        items = list(expected_freq.items())
        items.sort(key=lambda kv: kv[1], reverse=True)

        base_limit = self.kpart_query_limits[stage_idx]
        if base_limit is None:
            base_limit = len(items)
        else:
            base_limit = max(1, int(base_limit))

        if (not self.hard_case_enabled) or (base_limit <= 0):
            return [(str(item), int(expected)) for item, expected in items[:base_limit]]

        replay_cases = self.hard_case_state.get("stage_cases", [[], [], []])[stage_idx]
        replay_quota = min(len(replay_cases), max(0, int(round(base_limit * float(self.hard_case_weight)))))
        base_quota = max(1, base_limit - replay_quota)

        picked = {}
        for item, expected in items[:base_quota]:
            picked[str(item)] = (str(item), int(expected))

        bucket_replay = self._pick_bucketed_replay_cases(stage_idx, replay_quota)

        for rec in bucket_replay:
            item = str(rec.get("item", ""))
            if not item:
                continue
            exp = int(rec.get("expected", expected_freq.get(item, 0)))
            picked[item] = (item, exp)
            if len(picked) >= base_limit:
                break

        if len(picked) < base_limit:
            for rec in replay_cases:
                item = str(rec.get("item", ""))
                if not item or item in picked:
                    continue
                exp = int(rec.get("expected", expected_freq.get(item, 0)))
                picked[item] = (item, exp)
                if len(picked) >= base_limit:
                    break

        if len(picked) < base_limit:
            for item, expected in items[base_quota:]:
                key = str(item)
                if key not in picked:
                    picked[key] = (key, int(expected))
                if len(picked) >= base_limit:
                    break

        return list(picked.values())

    def _get_absent_eval_items(self, limit=None):
        if limit is None:
            limit = max(int(self.lexicase_absent_cases), int(self.hard_case_absent_topk))
        limit = max(1, int(limit))

        items = []
        seen = set()

        if self.hard_case_enabled:
            for rec in self.hard_case_state.get("absent_cases", []):
                item = str(rec.get("item", ""))
                if item and item not in seen:
                    items.append((item, 0))
                    seen.add(item)
                if len(items) >= limit:
                    return items

        for item in self.hard_case_absent_bank:
            key = str(item)
            if key not in seen:
                items.append((key, 0))
                seen.add(key)
            if len(items) >= limit:
                break

        return items

    def _fec_bucket(self, v) -> int:
        try:
            vv = int(v)
        except Exception:
            return -1

        if vv < 0:
            vv = 0
        if vv <= 255:
            return vv
        if vv <= 4095:
            return (vv // 8) * 8
        if vv <= 65535:
            return (vv // 128) * 128
        return 65535

    def _make_fec_fingerprint(
            self,
            init_dex_func,
            probe_update_func,
            probe_query_func,
            probe_exec_stats,
            probe_overflow_matrices,
    ):
        touched = []
        probe_stream = self._get_probe_stream(limit=self.fec_probe_update_n, upto_stage=1)
        for item in probe_stream:
            probe_update_func(item)
            try:
                locs = init_dex_func(item)
                for i in range(3):
                    v = locs[i % 3]
                    x = int(v[0]) % 102
                    y = int(v[1]) % 102
                    if isinstance(v, (tuple, list)) and len(v) > 2:
                        z = int(v[2]) % 3
                    else:
                        z = i % 3
                    touched.append((x, y, z))
            except Exception:
                touched.append(("ERR", len(touched)))

        valid_touched = [t for t in touched if isinstance(t, tuple) and len(t) == 3 and isinstance(t[0], int)]

        probe_expected = self._calc_freq_from_stream(probe_stream)

        present_sig = []
        present_items = list(probe_expected.items())
        present_items.sort(key=lambda kv: (-kv[1], str(kv[0])))
        present_items = present_items[:self.fec_probe_present_n]
        for item, _ in present_items:
            present_sig.append(self._fec_bucket(probe_query_func(item)))

        absent_sig = []
        for item in self.fec_absent_keys:
            absent_sig.append(self._fec_bucket(probe_query_func(item)))

        overflow_hits = 0
        for x, y, z in set(valid_touched):
            if probe_overflow_matrices[z][x][y]:
                overflow_hits += 1

        raw = {
            "present": present_sig,
            "absent": absent_sig,
            "uniq_locs": len(set(valid_touched)),
            "overflow_hits": overflow_hits,
            "upd_ops": int(probe_exec_stats["upd_ops"]),
            "qry_reads": int(probe_exec_stats["qry_reads"]),
        }
        return hashlib.sha1(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()

    def _calc_freq_from_stream(self, stream):
        freq = {}
        for item in stream:
            freq[item] = freq.get(item, 0) + 1
        freq["dragonfruit"] = 0
        freq["starfruit"] = 0
        freq["passionfruit"] = 0
        return freq

    def _build_kpart_views(self):
        n = len(self.test_data)
        if n <= 0:
            self.kpart_stage_streams = [[], [], []]
            self.kpart_expected = [{}, {}, {}]
            self.kpart_pkt_cuts = [0, 0, 0]
            return

        # ---------- real 模式：保持你原来的前缀/分段逻辑 ----------
        if self.dataset_mode != "proxy":
            cut1 = max(1, n // 3)
            cut2 = max(cut1 + 1, (2 * n) // 3)
            cut3 = n
            if cut2 > n:
                cut2 = n
            if cut1 > cut2:
                cut1 = cut2

            s0 = list(self.test_data[:cut1])
            s1 = list(self.test_data[cut1:cut2])
            s2 = list(self.test_data[cut2:cut3])

            self.kpart_stage_streams = [s0, s1, s2]
            self.kpart_pkt_cuts = [len(s0), len(s0) + len(s1), len(s0) + len(s1) + len(s2)]

            cum = []
            self.kpart_expected = []
            for seg in self.kpart_stage_streams:
                cum.extend(seg)
                self.kpart_expected.append(self._calc_freq_from_stream(cum))
            return

        # ---------- proxy 模式：改成“分布切片”，不是前缀切片 ----------
        files = self._find_flowid_files()
        if not files:
            # 兜底：如果 proxy 文件异常，退回旧逻辑
            cut1 = max(1, n // 3)
            cut2 = max(cut1 + 1, (2 * n) // 3)
            cut3 = n
            if cut2 > n:
                cut2 = n
            if cut1 > cut2:
                cut1 = cut2

            s0 = list(self.test_data[:cut1])
            s1 = list(self.test_data[cut1:cut2])
            s2 = list(self.test_data[cut2:cut3])

            self.kpart_stage_streams = [s0, s1, s2]
            self.kpart_pkt_cuts = [len(s0), len(s0) + len(s1), len(s0) + len(s1) + len(s2)]

            cum = []
            self.kpart_expected = []
            for seg in self.kpart_stage_streams:
                cum.extend(seg)
                self.kpart_expected.append(self._calc_freq_from_stream(cum))
            return

        files = files[:max(1, self.max_files)]

        base_rng = random.Random((int(self.seed) + 424242) & 0xFFFFFFFF)
        pool_target = max(int(self.pkts) * int(self.proxy_pool_mul), 40000)
        pool = self._build_candidate_pool(files, base_rng, pool_target)

        # 三个 stage 的增量训练包数：不要让前两段太小，否则很容易退化成“几乎全是 1/2 次频率”
        n0 = max(1, min(int(self.pkts), max(int(self.pkts * 0.25), 1200)))
        n1 = max(1, min(int(self.pkts) - n0, max(int(self.pkts * 0.35), 1500)))
        n2 = max(1, int(self.pkts) - n0 - n1)

        if n0 + n1 + n2 < int(self.pkts):
            n2 += int(self.pkts) - (n0 + n1 + n2)
        elif n0 + n1 + n2 > int(self.pkts):
            n2 = max(1, n2 - ((n0 + n1 + n2) - int(self.pkts)))

        rng0 = random.Random((int(self.seed) + 50001) & 0xFFFFFFFF)
        rng1 = random.Random((int(self.seed) + 50002) & 0xFFFFFFFF)
        rng2 = random.Random((int(self.seed) + 50003) & 0xFFFFFFFF)

        s0 = self._make_proxy_stream_from_pool(pool, rng0, mode="proxy_head", target=n0)
        s1 = self._make_proxy_stream_from_pool(pool, rng1, mode="proxy_balanced", target=n1)
        s2 = self._make_proxy_stream_from_pool(pool, rng2, mode="proxy_collision", target=n2)

        print(
            f"[KPART_PROXY] "
            f"n0={len(s0)} U0={len(set(s0))} U0_ratio={len(set(s0)) / max(1, len(s0)):.4f} "
            f"n1={len(s1)} U1={len(set(s1))} U1_ratio={len(set(s1)) / max(1, len(s1)):.4f} "
            f"n2={len(s2)} U2={len(set(s2))} U2_ratio={len(set(s2)) / max(1, len(s2)):.4f}",
            flush=True
        )

        self.kpart_stage_streams = [s0, s1, s2]
        self.kpart_pkt_cuts = [len(s0), len(s0) + len(s1), len(s0) + len(s1) + len(s2)]

        cum = []
        self.kpart_expected = []
        for seg in self.kpart_stage_streams:
            cum.extend(seg)
            self.kpart_expected.append(self._calc_freq_from_stream(cum))

    def _get_probe_stream(self, limit=None, upto_stage=1):
        """
        统一辅助评估口径：
        - 若已经有 kpart_stage_streams，就优先用 stage0..upto_stage 的累计流
        - 否则回退到 self.test_data
        """
        stream = []

        stage_streams = getattr(self, "kpart_stage_streams", None)
        if stage_streams:
            try:
                upto = min(max(0, int(upto_stage)), len(stage_streams) - 1)
                for s in range(upto + 1):
                    stream.extend(stage_streams[s])
            except Exception:
                stream = []

        if not stream:
            stream = list(self.test_data)

        if limit is not None:
            stream = stream[:max(0, int(limit))]

        return stream

    def _query_error_on_items(self, query_func, expected_freq, items=None, max_items=None, return_records: bool = False):
        if items is None:
            items = list(expected_freq.items())
            items.sort(key=lambda kv: kv[1], reverse=True)
            if max_items is not None:
                items = items[:max_items]
            norm_items = [(str(item), int(expected)) for item, expected in items]
        else:
            norm_items = []
            for rec in items:
                if isinstance(rec, (tuple, list)) and len(rec) >= 2:
                    item = str(rec[0])
                    try:
                        expected = int(rec[1])
                    except Exception:
                        expected = int(expected_freq.get(item, 0))
                else:
                    item = str(rec)
                    expected = int(expected_freq.get(item, 0))
                norm_items.append((item, expected))

        total_err = 0.0
        records = []
        for item, expected in norm_items:
            estimated = query_func(item)
            err = abs(float(estimated) - float(expected))
            total_err += err
            if return_records:
                records.append({
                    "item": str(item),
                    "expected": int(expected),
                    "estimated": float(estimated),
                    "err": float(err),
                })

        avg_err = total_err / max(1, len(norm_items))
        if return_records:
            records.sort(key=lambda d: (-float(d["err"]), str(d["item"])))
            return total_err, avg_err, len(norm_items), records
        return total_err, avg_err, len(norm_items)

    def _evaluate_individual_core(self, init_dex_tree, update_tree, query_tree, stage_idx: int = 2, return_case_vec: bool = False, return_hard_cases: bool = False):
        try:
            stage_idx = int(stage_idx)
            self.debug_stats["eval_calls"] += 1

            def _format_cached_result(cached_obj):
                if isinstance(cached_obj, dict):
                    res = cached_obj.get("res", (0.0, 2_000_000_000.0))
                    vec = tuple(cached_obj.get("case_vec", ()))
                    hc_bundle = copy.deepcopy(
                        cached_obj.get(
                            "hard_case_bundle",
                            self._empty_hard_case_state(),
                        )
                    )
                else:
                    # 兼容旧缓存：老版本 cache 里只有 (fit, err)
                    res = cached_obj
                    vec = ()
                    hc_bundle = self._empty_hard_case_state()

                fit = float(res[0])
                err = float(res[1])
                bad = float(self.lexicase_default_bad)

                if return_case_vec and return_hard_cases:
                    if len(vec) < int(self.lexicase_total_cases):
                        vec = vec + (bad,) * (int(self.lexicase_total_cases) - len(vec))
                    return (fit, err, vec, hc_bundle)

                if return_case_vec:
                    if len(vec) < int(self.lexicase_total_cases):
                        vec = vec + (bad,) * (int(self.lexicase_total_cases) - len(vec))
                    return (fit, err, vec)

                if return_hard_cases:
                    return (fit, err, hc_bundle)

                return (fit, err)

            def _format_cached_result(cached_obj):
                if isinstance(cached_obj, dict):
                    res = cached_obj.get("res", (0.0, 2_000_000_000.0))
                    vec = tuple(cached_obj.get("case_vec", ()))
                    hc_bundle = copy.deepcopy(
                        cached_obj.get(
                            "hard_case_bundle",
                            self._empty_hard_case_state()
                        )
                    )
                else:
                    # 兼容旧缓存：老版本 cache 里只有 (fit, err)
                    res = cached_obj
                    vec = ()
                    hc_bundle = self._empty_hard_case_state()

                fit = float(res[0])
                err = float(res[1])
                bad = float(self.lexicase_default_bad)

                if return_case_vec and return_hard_cases:
                    if len(vec) < int(self.lexicase_total_cases):
                        vec = vec + (bad,) * (int(self.lexicase_total_cases) - len(vec))
                    return (fit, err, vec, hc_bundle)

                if return_case_vec:
                    if len(vec) < int(self.lexicase_total_cases):
                        vec = vec + (bad,) * (int(self.lexicase_total_cases) - len(vec))
                    return (fit, err, vec)

                if return_hard_cases:
                    return (fit, err, hc_bundle)

                return (fit, err)

            if stage_idx >= 2:
                cache_key = self._canonical_triplet_key(init_dex_tree, update_tree, query_tree)
                cached = self.eval_cache.get(cache_key)
                if cached is not None:
                    self.debug_stats["eval_cache_hits"] += 1
                    return _format_cached_result(cached)
            else:
                cache_key = self._canonical_triplet_key(
                    init_dex_tree, update_tree, query_tree, stage_idx=stage_idx
                )
                cached = self.stage_eval_cache[stage_idx].get(cache_key)
                if cached is not None:
                    self.debug_stats["eval_cache_hits"] += 1
                    return _format_cached_result(cached)

            def _ret(total_error: float, fec_fp=None):
                total_error = float(total_error)
                res = (self._norm_fitness(total_error), total_error)
                payload = {
                    "res": res,
                    "case_vec": tuple(case_vec),
                    "hard_case_bundle": copy.deepcopy(hard_case_bundle),
                }

                if stage_idx >= 2:
                    self.eval_cache[cache_key] = payload
                    if fec_fp is not None:
                        self.fec_cache[fec_fp] = payload
                else:
                    self.stage_eval_cache[stage_idx][cache_key] = payload

                if return_case_vec and return_hard_cases:
                    bad = float(self.lexicase_default_bad)
                    vec = tuple(case_vec)
                    if len(vec) < int(self.lexicase_total_cases):
                        vec = vec + (bad,) * (int(self.lexicase_total_cases) - len(vec))
                    return (res[0], res[1], vec, copy.deepcopy(hard_case_bundle))

                if return_case_vec:
                    bad = float(self.lexicase_default_bad)
                    vec = tuple(case_vec)
                    if len(vec) < int(self.lexicase_total_cases):
                        vec = vec + (bad,) * (int(self.lexicase_total_cases) - len(vec))
                    return (res[0], res[1], vec)

                if return_hard_cases:
                    return (res[0], res[1], copy.deepcopy(hard_case_bundle))

                return res

            penalty = 0
            case_vec = []
            hard_case_bundle = self._empty_hard_case_state()

            init_info = self.analyze_init_tree(init_dex_tree)
            update_info = self.analyze_update_tree(update_tree)
            query_info = self.analyze_query_tree(query_tree)

            try:
                init_ast = self._simplify_ast(self._tree_to_ast(init_dex_tree))
                update_ast = self._simplify_ast(self._tree_to_ast(update_tree))
                query_ast = self._simplify_ast(self._tree_to_ast(query_tree))
            except Exception:
                return _ret(1_900_000_000.0)

            init_ast_chk = self._ast_legality_check("init", init_ast)
            update_ast_chk = self._ast_legality_check("update", update_ast)
            query_ast_chk = self._ast_legality_check("query", query_ast)

            init_eff = self._ast_effect_summary(init_ast)
            update_eff = self._ast_effect_summary(update_ast)
            query_eff = self._ast_effect_summary(query_ast)

            init_pat = self._ast_pattern_summary("init", init_ast)
            update_pat = self._ast_pattern_summary("update", update_ast)
            query_pat = self._ast_pattern_summary("query", query_ast)
            active_hard_buckets = self._infer_failure_buckets(
                init_pat=init_pat,
                update_pat=update_pat,
                query_pat=query_pat,
                update_eff=update_eff,
                query_eff=query_eff,
            )
            primary_hard_bucket = active_hard_buckets[0] if active_hard_buckets else "generic"

            hash_calls = init_info["hash_calls"]
            update_calls = update_info["write_calls"]
            query_calls = query_info["read_calls"]

            # ---- init_dex 约束 ----
            if not init_info["root_ok"]:
                penalty += 120000

            if init_info["forbidden_hits"]:
                penalty += 220000 + 30000 * sum(init_info["forbidden_hits"].values())

            if hash_calls == 0:
                penalty += 80000
            elif hash_calls < 2:
                penalty += (2 - hash_calls) * 12000

            # ---- update 约束 ----
            if not update_info["root_ok"]:
                penalty += 120000

            if update_info["forbidden_hits"]:
                penalty += 180000 + 25000 * sum(update_info["forbidden_hits"].values())

            if update_calls == 0:
                penalty += 100000
            elif update_calls < 2:
                penalty += (2 - update_calls) * 18000

            # ---- query 约束 ----
            if not query_info["root_ok"]:
                penalty += 120000

            if query_info["forbidden_hits"]:
                penalty += 260000 + 40000 * sum(query_info["forbidden_hits"].values())

            if query_calls == 0:
                penalty += 100000
            elif query_calls < 2:
                penalty += (2 - query_calls) * 18000

            # ---- AST 级硬合法性检查 ----
            if init_ast_chk["hard_illegal"] or update_ast_chk["hard_illegal"] or query_ast_chk["hard_illegal"]:
                self.debug_stats["hard_illegal"] += 1
                for r in init_ast_chk.get("reasons", []):
                    self.debug_stats["hard_illegal_reasons"][f"init:{r}"] += 1
                for r in update_ast_chk.get("reasons", []):
                    self.debug_stats["hard_illegal_reasons"][f"update:{r}"] += 1
                for r in query_ast_chk.get("reasons", []):
                    self.debug_stats["hard_illegal_reasons"][f"query:{r}"] += 1

                penalty += 450000
                if penalty >= 450000:
                    self.debug_stats["early_return_cut"] += 1
                    self.debug_stats["penalty_sum"] += float(penalty)
                    self.debug_stats["total_error_sum"] += float(penalty)
                    return _ret(float(penalty))

            # ---- AST 级退化检测 ----
            if not update_eff["depends_on_e"]:
                penalty += 90000

            if not query_eff["depends_on_e"]:
                penalty += 90000

            # 化简后若已经没有真实写入，说明它基本是伪 update
            if update_eff["real_write_calls"] == 0:
                self.debug_stats["real_write_zero"] += 1
                if update_eff["conditional_write_calls"] == 0:
                    penalty += 180000
                else:
                    penalty += 90000

            # 化简后 query_date 太少，说明 query 大概率已经退化
            if query_eff["query_date_calls"] == 0:
                self.debug_stats["query_date_zero"] += 1
                cut_penalty = max(float(penalty) + 260000.0, 450000.0)
                self.debug_stats["early_return_cut"] += 1
                self.debug_stats["penalty_sum"] += cut_penalty
                self.debug_stats["total_error_sum"] += cut_penalty
                return _ret(cut_penalty)
            elif query_eff["query_date_calls"] < 2:
                penalty += (2 - query_eff["query_date_calls"]) * 22000

            # ---- AST 级结构约束：按当前原语集语义约束 hash/path id ----
            self.debug_stats["nonconst_hash_idx_total"] += int(init_pat["nonconst_hash_idx"])
            self.debug_stats["nonconst_path_idx_total"] += int(update_pat["nonconst_path_idx"])
            self.debug_stats["nonconst_path_idx_total"] += int(query_pat["nonconst_path_idx"])
            self.debug_stats["bad_write_context_total"] += int(update_pat["bad_write_context"])

            if init_pat["nonconst_hash_idx"] > 0:
                penalty += init_pat["nonconst_hash_idx"] * 18000

            if update_pat["nonconst_path_idx"] > 0:
                penalty += update_pat["nonconst_path_idx"] * 50000

            if query_pat["nonconst_path_idx"] > 0:
                penalty += query_pat["nonconst_path_idx"] * 50000

            if update_pat["bad_write_context"] > 0:
                bwc = int(update_pat["bad_write_context"])
                penalty += bwc * 90000

                # bad_write_ctx 一旦达到 2，基本已经不是干净的 update 结构了
                if bwc >= 2:
                    cut_penalty = max(float(penalty), 450000.0)
                    self.debug_stats["early_return_cut"] += 1
                    self.debug_stats["penalty_sum"] += cut_penalty
                    self.debug_stats["total_error_sum"] += cut_penalty
                    return _ret(cut_penalty)

            init_dex_func = gp.compile(init_dex_tree, init_dex_pset)

            def _norm_loc(v, fallback_z):
                x = int(v[0]) % 102
                y = int(v[1]) % 102
                if isinstance(v, (tuple, list)) and len(v) > 2:
                    z = int(v[2]) % 3
                else:
                    z = fallback_z % 3
                return (x, y, z)

            loc_probe_stream = self._get_probe_stream(limit=128, upto_stage=1)
            sample_n = len(loc_probe_stream)
            uniq_sum = 0.0
            for e0 in loc_probe_stream:
                locs = init_dex_func(e0)
                norm_locs = [_norm_loc(locs[i % 3], i) for i in range(3)]
                uniq_sum += len(set(norm_locs))
            avg_unique_locs = uniq_sum / max(1, sample_n)
            if avg_unique_locs < 2.0:
                penalty += int((2.0 - avg_unique_locs) * 25000)

            if penalty >= 1500000:
                return _ret(float(penalty))

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
            exec_stats = {"upd_ops": 0, "qry_reads": 0}

            def create_pset_update(count_matrices, overflow_matrices, exec_stats):
                def str_slice(s, start, end):
                    return s[start:end]

                def write_count(e: str, i: int, delta: int) -> int:
                    vars = init_dex_func(e)[i % 3]
                    x = vars[0] % 102
                    y = vars[1] % 102
                    z = i % 3
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
                    exec_stats["upd_ops"] += 1
                    return vv

                def update_count(e: str, i: int, delta: int) -> int:
                    vars = init_dex_func(e)[i % 3]
                    x = vars[0] % 102
                    y = vars[1] % 102
                    z = i % 3
                    count_matrices[z][x][y] += delta
                    if count_matrices[z][x][y] < 0:
                        count_matrices[z][x][y] = 0
                        overflow_matrices[z][x][y] = False
                    elif count_matrices[z][x][y] > MAX_COUNTER:
                        overflow_matrices[z][x][y] = True
                        count_matrices[z][x][y] = MAX_COUNTER
                    else:
                        overflow_matrices[z][x][y] = False
                    exec_stats["upd_ops"] += 1
                    return count_matrices[z][x][y]

                def update_state(e: str, i: int, st: bool) -> int:
                    vars = init_dex_func(e)[i % 3]
                    x = vars[0] % 102
                    y = vars[1] % 102
                    z = i % 3
                    overflow_matrices[z][x][y] = st
                    return 0

                def query_count(e: str, i: int) -> int:
                    vars = init_dex_func(e)[i % 3]
                    x = vars[0] % 102
                    y = vars[1] % 102
                    return count_matrices[i % 3][x][y]

                def query_state(e: str, i: int) -> bool:
                    vars = init_dex_func(e)[i % 3]
                    x = vars[0] % 102
                    y = vars[1] % 102
                    z = i % 3
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
                    return out1 if cond else out2

                def base(a: int, b: int, c: int) -> float:
                    return 0.0

                def lt(a: int, b: int) -> bool:
                    return a < b

                def gt(a: int, b: int) -> bool:
                    return a > b

                def eq(a: int, b: int) -> bool:
                    return a == b

                def safe_add(a: int, b: int) -> int:
                    return a + b

                def safe_sub(a: int, b: int) -> int:
                    return a - b

                def safe_mul(a: int, b: int) -> int:
                    return a * b

                def safe_div(a: int, b: int) -> int:
                    return a // (b if b != 0 else 1)

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
                pset_update.addPrimitive(str_slice, [str, int, int], str)
                pset_update.addPrimitive(if_then_else, [bool, int, int], int)
                pset_update.addPrimitive(base, [int, int, int], float)
                pset_update.addPrimitive(write_count, [str, int, int], int)
                pset_update.addPrimitive(update_count, [str, int, int], int)
                pset_update.addPrimitive(update_state, [str, int, bool], int)
                pset_update.addPrimitive(query_count, [str, int], int)
                pset_update.addPrimitive(query_state, [str, int], bool)
                pset_update.addPrimitive(updatecount_if, [bool, str, int, int], int)
                pset_update.addPrimitive(writecount_if, [bool, str, int, int], int)
                pset_update.addPrimitive(writestate_if, [bool, str, int, bool], int)
                pset_update.addPrimitive(lt, [int, int], bool)
                pset_update.addPrimitive(gt, [int, int], bool)
                pset_update.addPrimitive(eq, [int, int], bool)
                pset_update.addPrimitive(safe_add, [int, int], int)
                pset_update.addPrimitive(safe_sub, [int, int], int)
                pset_update.addPrimitive(safe_mul, [int, int], int)
                pset_update.addPrimitive(safe_div, [int, int], int)
                pset_update.addPrimitive(safe_mod, [int, int], int)
                pset_update.addPrimitive(abs_int, [int], int)
                pset_update.addTerminal(0, int)
                pset_update.addTerminal(1, int)
                pset_update.addTerminal(2, int)
                pset_update.addEphemeralConstant('rand_int', functools.partial(random.randint, 0, 102 - 1), int)
                pset_update.addTerminal((0, 24), tuple)
                pset_update.addTerminal(-1, int)
                pset_update.addTerminal([], list)
                pset_update.addTerminal(0.0, float)
                pset_update.addTerminal(True, bool)
                pset_update.addTerminal(False, bool)
                pset_update.addEphemeralConstant("rand_i", functools.partial(random.randint, 0, 2), int)
                pset_update.addEphemeralConstant("rand_delta", functools.partial(random.choice, [-1, 1, 2, -2, 3, -3]), int)
                pset_update.addEphemeralConstant("rand_small", functools.partial(random.randint, 0, 16), int)
                return pset_update

            def create_pset_query(count_matrices, overflow_matrices, exec_stats):
                def safe_add(a: int, b: int) -> int:
                    return a + b

                def safe_sub(a: int, b: int) -> int:
                    return a - b

                def safe_mul(a: int, b: int) -> int:
                    return a * b

                def safe_div(a: int, b: int) -> int:
                    return a // (b if b != 0 else 1)

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

                def str_slice(s, start, end):
                    return s[start:end]

                def cnt_rdstate(e: str, i: int) -> bool:
                    vars = init_dex_func(e)[i % 3]
                    x = vars[0] % 102
                    y = vars[1] % 102
                    z = i % 3
                    return bool(overflow_matrices[z][x][y])

                def query_date(e: str, i: int) -> int:
                    vars = init_dex_func(e)[i % 3]
                    x = vars[0] % 102
                    y = vars[1] % 102
                    z = i % 3
                    exec_stats["qry_reads"] += 1
                    if overflow_matrices[z][x][y]:
                        return INF
                    return count_matrices[z][x][y]

                def if_then_else(cond, out1, out2):
                    return out1 if cond else out2

                def base_sel(mode: int, a: int, b: int, c: int) -> float:
                    m = abs(mode) % 4
                    if m == 0:
                        return float(min(a, b, c))
                    if m == 1:
                        return float(max(a, b, c))
                    if m == 2:
                        return float(sorted([a, b, c])[1])
                    return float((a + b + c) // 3)

                def lt(a: int, b: int) -> bool:
                    return a < b

                def gt(a: int, b: int) -> bool:
                    return a > b

                def eq(a: int, b: int) -> bool:
                    return a == b

                pset = gp.PrimitiveSetTyped("QUERY", [str], float)
                pset.renameArguments(ARG0="e")
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
                pset.addPrimitive(if_then_else, [bool, int, int], int)
                pset.addPrimitive(base_sel, [int, int, int, int], float)
                pset.addPrimitive(lt, [int, int], bool)
                pset.addPrimitive(gt, [int, int], bool)
                pset.addPrimitive(eq, [int, int], bool)
                pset.addPrimitive(query_date, [str, int], int)
                pset.addPrimitive(cnt_rdstate, [str, int], bool)
                pset.addPrimitive(str_slice, [str, int, int], str)
                pset.addTerminal(102, int)
                pset.addTerminal(0.0, float)
                pset.addTerminal(0, int)
                pset.addTerminal(1, int)
                pset.addTerminal(2, int)
                pset.addTerminal(3, int)
                pset.addTerminal(4, int)
                pset.addTerminal(8, int)
                pset.addTerminal([], list)
                pset.addTerminal((0, 0), tuple)
                pset.addTerminal(True, bool)
                pset.addTerminal(False, bool)
                pset.addTerminal(INF, int)
                return pset

            pset_update_replaced = create_pset_update(count_matrices, overflow_matrices, exec_stats)
            pset_query_replaced = create_pset_query(count_matrices, overflow_matrices, exec_stats)
            update_func = gp.compile(update_tree, pset_update_replaced)
            query_func = gp.compile(query_tree, pset_query_replaced)

            fec_fp = None
            if stage_idx >= 2:
                probe_count_matrices = [
                    [[0 for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
                    for _ in range(planes)
                ]
                probe_overflow_matrices = [
                    [[False for _ in range(cols_per_matrix)] for _ in range(rows_per_matrix)]
                    for _ in range(planes)
                ]
                probe_exec_stats = {"upd_ops": 0, "qry_reads": 0}
                probe_pset_update = create_pset_update(probe_count_matrices, probe_overflow_matrices, probe_exec_stats)
                probe_pset_query = create_pset_query(probe_count_matrices, probe_overflow_matrices, probe_exec_stats)
                probe_update_func = gp.compile(update_tree, probe_pset_update)
                probe_query_func = gp.compile(query_tree, probe_pset_query)
                fec_fp = self._make_fec_fingerprint(
                    init_dex_func=init_dex_func,
                    probe_update_func=probe_update_func,
                    probe_query_func=probe_query_func,
                    probe_exec_stats=probe_exec_stats,
                    probe_overflow_matrices=probe_overflow_matrices,
                )
                fec_cached = self.fec_cache.get(fec_fp)
                if fec_cached is not None:
                    self.fec_hits += 1
                    self.debug_stats["fec_cache_hits"] += 1
                    self.eval_cache[cache_key] = fec_cached
                    return _format_cached_result(fec_cached)
                self.fec_misses += 1

            stage_streams = self.kpart_stage_streams
            stage_expected = self.kpart_expected
            stage_query_error = 0.0
            final_qn = 0
            seen_pkts = 0

            for s in range(stage_idx + 1):
                cur_stream = stage_streams[s]
                try:
                    for item in cur_stream:
                        update_func(item)
                except Exception:
                    return _ret(2_000_000_000.0, fec_fp=fec_fp)

                seen_pkts += len(cur_stream)

                avg_upd_ops = exec_stats["upd_ops"] / max(1, seen_pkts)
                min_ops = float(self.kpart_upd_min[s])
                if avg_upd_ops < min_ops:
                    penalty += int((min_ops - avg_upd_ops) * float(self.kpart_upd_penalty_scale[s]))

                stage_items = self._get_stage_eval_items(s, stage_expected[s])

                try:
                    if return_hard_cases:
                        stage_query_error, stage_avg_err, final_qn, hard_records = self._query_error_on_items(
                            query_func,
                            stage_expected[s],
                            items=stage_items,
                            return_records=True,
                        )
                        top_stage_records = [
                            {
                                "item": str(rec["item"]),
                                "expected": int(rec["expected"]),
                                "score": float(rec["err"]),
                                "bucket": primary_hard_bucket,
                            }
                            for rec in hard_records[:int(self.hard_case_stage_topk)]
                        ]
                        hard_case_bundle["stage_cases"][s] = top_stage_records

                        for bk in active_hard_buckets:
                            hard_case_bundle["stage_bucket_cases"][s][str(bk)] = [
                                {
                                    "item": str(rec["item"]),
                                    "expected": int(rec["expected"]),
                                    "score": float(rec["err"]),
                                    "bucket": str(bk),
                                }
                                for rec in hard_records[:int(self.hard_case_stage_topk)]
                            ]
                    else:
                        stage_query_error, stage_avg_err, final_qn = self._query_error_on_items(
                            query_func,
                            stage_expected[s],
                            items=stage_items,
                            return_records=False,
                        )
                except Exception:
                    return _ret(2_000_000_000.0, fec_fp=fec_fp)

                err_thr = self.kpart_avg_err_thresh[s]
                if err_thr is not None and stage_avg_err > float(err_thr):
                    penalty += int(stage_avg_err * float(self.kpart_avg_err_scale[s]))

                case_vec.append(float(stage_avg_err))

                cut_pen = self.kpart_cut_penalty[s]
                if cut_pen is not None and penalty >= float(cut_pen):
                    self.debug_stats["early_return_cut"] += 1
                    self.debug_stats["penalty_sum"] += float(penalty)
                    self.debug_stats["query_error_sum"] += float(stage_query_error)
                    self.debug_stats["total_error_sum"] += float(stage_query_error + penalty)
                    if float(penalty) > float(stage_query_error):
                        self.debug_stats["penalty_dominates"] += 1
                    return _ret(float(stage_query_error + penalty), fec_fp=fec_fp)

            absent_eval_items = self._get_absent_eval_items()

            if return_case_vec or return_hard_cases:
                try:
                    if return_hard_cases:
                        _, _, _, absent_records = self._query_error_on_items(
                            query_func,
                            stage_expected[min(stage_idx, len(stage_expected) - 1)],
                            items=absent_eval_items,
                            return_records=True,
                        )
                        hard_case_bundle["absent_cases"] = [
                            {
                                "item": str(rec["item"]),
                                "expected": 0,
                                "score": float(rec["err"]),
                            }
                            for rec in absent_records[:int(self.hard_case_absent_topk)]
                        ]
                    if return_case_vec:
                        for item, _ in absent_eval_items[:int(self.lexicase_absent_cases)]:
                            try:
                                est = int(query_func(item))
                                case_vec.append(float(abs(est)))
                            except Exception:
                                case_vec.append(float(self.lexicase_default_bad))
                except Exception:
                    if return_case_vec:
                        while len(case_vec) < int(self.lexicase_stage_cases + self.lexicase_absent_cases):
                            case_vec.append(float(self.lexicase_default_bad))

            avg_qry_reads = exec_stats["qry_reads"] / max(1, final_qn)
            if avg_qry_reads < 2.0:
                penalty += int((2.0 - avg_qry_reads) * 30000)

            total_error = float(stage_query_error + penalty)
            self.debug_stats["penalty_sum"] += float(penalty)
            self.debug_stats["query_error_sum"] += float(stage_query_error)
            self.debug_stats["total_error_sum"] += float(total_error)
            if float(penalty) > float(stage_query_error):
                self.debug_stats["penalty_dominates"] += 1

            return _ret(total_error, fec_fp=fec_fp)


        except Exception as e:

            print(f"DEBUG: 整体评估异常(stage={stage_idx}): {e}")

            total_error = 2_000_000_000.0

            res = (self._norm_fitness(total_error), total_error)
            bad = (float(self.lexicase_default_bad),) * int(self.lexicase_total_cases)
            payload = {
                "res": res,
                "case_vec": bad,
                "hard_case_bundle": copy.deepcopy(hard_case_bundle),
            }

            try:

                if stage_idx >= 2:

                    self.eval_cache[cache_key] = payload

                else:

                    self.stage_eval_cache[stage_idx][cache_key] = payload

            except Exception:

                pass

            if return_case_vec and return_hard_cases:
                return (res[0], res[1], bad, copy.deepcopy(hard_case_bundle))

            if return_case_vec:
                return (res[0], res[1], bad)

            if return_hard_cases:
                return (res[0], res[1], copy.deepcopy(hard_case_bundle))

            return res

    def evaluate_individual(self, init_dex_tree, update_tree, query_tree, return_case_vec: bool = False, return_hard_cases: bool = False):
        return self._evaluate_individual_core(
            init_dex_tree,
            update_tree,
            query_tree,
            stage_idx=2,
            return_case_vec=return_case_vec,
            return_hard_cases=return_hard_cases,
        )

    def mine_hard_cases(self, init_dex_tree, update_tree, query_tree):
        ret = self.evaluate_individual(
            init_dex_tree,
            update_tree,
            query_tree,
            return_case_vec=False,
            return_hard_cases=True,
        )
        if isinstance(ret, tuple) and len(ret) >= 3:
            return copy.deepcopy(ret[2])
        return self._empty_hard_case_state()

    def generate_complete_code(self, init_dex_tree, update_tree, query_tree):
        init_dex_str = self._canonical_tree_str(init_dex_tree)
        update_str = self._canonical_tree_str(update_tree)
        query_str = self._canonical_tree_str(query_tree)

        # 把本次评测用到的数据集参数“烘焙”进导出的脚本里，保证测试一致
        baked_root = self.dataset_root
        baked_pkts = int(self.pkts)
        baked_files = int(self.max_files)
        baked_start = int(self.start)
        baked_shuffle = bool(self.shuffle)
        baked_seed = int(self.seed)

        code = f'''# -*- coding: utf-8 -*-
import os
import random
import hashlib
import numpy as np
from glob import glob

# ===== dataset config baked from evolution run =====
DATASET_ROOT = {baked_root!r}
PKTS = {baked_pkts}
MAX_FILES = {baked_files}
START = {baked_start}
SHUFFLE = {baked_shuffle}
SEED = {baked_seed}

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
from init_dex_language import *  # 这里要求你本地的 init_dex_language.py 与当前 init_dex 原语集一致

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
    return {init_dex_str}

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
    return {update_str}

def query(e):
    return {query_str}

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
    actual_freq = {{}}
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
'''
        return code





class Stage1MultiProxyEvaluator:
    """Stage1 evaluator wrapper: multi-proxy aggregation only.
    仅做 multi-proxy 聚合，不引入 holdout / curriculum。
    """

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.stage1_multi_proxy = bool(self.cfg.get("stage1_multi_proxy", False))
        modes = list(self.cfg.get("stage1_proxy_modes", []))
        if not modes:
            modes = [str(self.cfg.get("proxy_mode", "proxy_balanced"))]
        self.stage1_proxy_modes = [str(x) for x in modes]

        self.lexicase_stage_cases = 3
        self.lexicase_absent_cases = 4
        self.lexicase_total_cases = self.lexicase_stage_cases + self.lexicase_absent_cases
        self.lexicase_default_bad = 1e18
        self.E0 = None
        self.fec_hits = 0
        self.fec_misses = 0

        self.eval_specs = self._build_eval_specs()
        self.primary_spec = self.eval_specs[0]
        self.primary = self.primary_spec["evaluator"]

        self.dataset_root = self.primary.dataset_root
        self.pkts = self.primary.pkts
        self.max_files = self.primary.max_files
        self.start = self.primary.start
        self.shuffle = self.primary.shuffle
        self.seed = self.primary.seed
        self.dataset_mode = self.primary.dataset_mode
        self.proxy_mode = "multi_proxy"
        self.proxy_pool_mul = self.primary.proxy_pool_mul
        self.proxy_min_u = self.primary.proxy_min_u
        self.fixed_stream_path = getattr(self.primary, "fixed_stream_path", "")
        self.test_data = list(getattr(self.primary, "test_data", []))
        self.expected_freq = dict(getattr(self.primary, "expected_freq", {}))
        self.U = int(getattr(self.primary, "U", 0))
        self.U_ratio = float(getattr(self.primary, "U_ratio", 0.0))

    def __getattr__(self, name):
        return getattr(self.primary, name)

    def _norm_fitness(self, total_error: float) -> float:
        E0 = float(self.E0) if self.E0 not in (None, 0) else 1.0
        err = float(total_error)
        if not math.isfinite(err):
            return 0.0
        if err < 0:
            err = 0.0
        return 1.0 / (1.0 + (err / E0))

    def _base_eval_kwargs(self):
        return dict(
            dataset_root=self.cfg["dataset_root"],
            pkts=int(self.cfg["pkts"]),
            max_files=int(self.cfg["files"]),
            start=int(self.cfg["start"]),
            shuffle=bool(self.cfg["shuffle"]),
            dataset_mode=str(self.cfg["dataset_mode"]),
            proxy_pool_mul=int(self.cfg["proxy_pool_mul"]),
            proxy_min_u=int(self.cfg["proxy_min_u"]),
            hard_case_enabled=self.cfg.get("hard_case_replay", False),
            hard_case_stage_topk=self.cfg.get("hard_case_stage_topk", 24),
            hard_case_absent_topk=self.cfg.get("hard_case_absent_topk", 12),
            hard_case_scan_mul=self.cfg.get("hard_case_scan_mul", 3),
            hard_case_decay=self.cfg.get("hard_case_decay", 0.85),
            hard_case_weight=self.cfg.get("hard_case_weight", 0.50),
        )

    def _build_eval_specs(self):
        base_kwargs = self._base_eval_kwargs()
        dataset_seed = int(self.cfg.get("dataset_seed", 0)) & 0xFFFFFFFF
        base_fixed_stream = str(self.cfg.get("fixed_stream_path", "") or "")

        specs = []
        n = max(1, len(self.stage1_proxy_modes))
        for idx, mode in enumerate(self.stage1_proxy_modes):
            kwargs = dict(base_kwargs)
            kwargs.update({
                "seed": (dataset_seed + 1009 * idx) & 0xFFFFFFFF,
                "proxy_mode": str(mode),
                "fixed_stream_path": _apply_proxy_mode_to_stream_path(base_fixed_stream, kwargs["dataset_mode"], str(mode)),
            })
            ev = CMSketchEvaluator(**kwargs)
            specs.append({
                "name": f"train:{mode}",
                "mode": str(mode),
                "weight": 1.0 / float(n),
                "evaluator": ev,
            })

        if not specs:
            kwargs = dict(base_kwargs)
            kwargs.update({
                "seed": dataset_seed,
                "proxy_mode": str(self.cfg.get("proxy_mode", "proxy_balanced")),
                "fixed_stream_path": base_fixed_stream,
            })
            ev = CMSketchEvaluator(**kwargs)
            specs.append({
                "name": f"train:{kwargs['proxy_mode']}",
                "mode": str(kwargs["proxy_mode"]),
                "weight": 1.0,
                "evaluator": ev,
            })
        return specs

    def _agg_case_vec(self, pieces):
        if not pieces:
            return ()
        dim = max(len(vec) for _, vec in pieces)
        if dim <= 0:
            return ()
        out = []
        for i in range(dim):
            num = 0.0
            den = 0.0
            for w, vec in pieces:
                if i >= len(vec):
                    continue
                num += float(w) * float(vec[i])
                den += float(w)
            out.append(num / max(1e-12, den))
        return tuple(out)

    def _merge_debug_snapshot(self):
        merged = {
            "eval_calls": 0,
            "eval_cache_hits": 0,
            "fec_cache_hits": 0,
            "hard_illegal": 0,
            "real_write_zero": 0,
            "query_date_zero": 0,
            "penalty_dominates": 0,
            "early_return_cut": 0,
            "penalty_sum": 0.0,
            "query_error_sum": 0.0,
            "total_error_sum": 0.0,
            "nonconst_hash_idx_total": 0,
            "nonconst_path_idx_total": 0,
            "bad_write_context_total": 0,
            "hard_illegal_reasons": Counter(),
        }
        fec_hits = 0
        fec_misses = 0
        for sp in self.eval_specs:
            ev = sp["evaluator"]
            dbg = ev._debug_snapshot()
            for k, v in dbg.items():
                if isinstance(v, Counter):
                    merged[k].update(v)
                elif k in merged:
                    merged[k] += v
                else:
                    merged[k] = v
            fec_hits += int(getattr(ev, "fec_hits", 0))
            fec_misses += int(getattr(ev, "fec_misses", 0))
        self.fec_hits = fec_hits
        self.fec_misses = fec_misses
        return merged

    def _debug_reset(self):
        for sp in self.eval_specs:
            sp["evaluator"]._debug_reset()

    def _debug_snapshot(self):
        return self._merge_debug_snapshot()

    def export_hard_case_state(self):
        return self.primary.export_hard_case_state()

    def import_hard_case_state(self, state):
        for sp in self.eval_specs:
            sp["evaluator"].import_hard_case_state(copy.deepcopy(state))

    def mine_hard_cases(self, init_dex_tree, update_tree, query_tree):
        return self.primary.mine_hard_cases(init_dex_tree, update_tree, query_tree)

    def evaluate_individual(self, init_dex_tree, update_tree, query_tree, return_case_vec: bool = False, return_hard_cases: bool = False):
        total_err = 0.0
        case_pieces = []
        hard_piece = None

        for sp in self.eval_specs:
            ev = sp["evaluator"]
            w = float(sp["weight"])
            if return_case_vec and return_hard_cases:
                fit_i, err_i, vec_i, hc_i = ev.evaluate_individual(
                    init_dex_tree, update_tree, query_tree, return_case_vec=True, return_hard_cases=True
                )
                case_pieces.append((w, tuple(vec_i)))
                if hard_piece is None:
                    hard_piece = copy.deepcopy(hc_i)
            elif return_case_vec:
                fit_i, err_i, vec_i = ev.evaluate_individual(
                    init_dex_tree, update_tree, query_tree, return_case_vec=True, return_hard_cases=False
                )
                case_pieces.append((w, tuple(vec_i)))
            elif return_hard_cases:
                fit_i, err_i, hc_i = ev.evaluate_individual(
                    init_dex_tree, update_tree, query_tree, return_case_vec=False, return_hard_cases=True
                )
                if hard_piece is None:
                    hard_piece = copy.deepcopy(hc_i)
            else:
                fit_i, err_i = ev.evaluate_individual(
                    init_dex_tree, update_tree, query_tree, return_case_vec=False, return_hard_cases=False
                )
            total_err += w * float(err_i)

        fit = float(self._norm_fitness(total_err))
        if return_case_vec and return_hard_cases:
            return fit, float(total_err), self._agg_case_vec(case_pieces), hard_piece if hard_piece is not None else self.primary._empty_hard_case_state()
        if return_case_vec:
            return fit, float(total_err), self._agg_case_vec(case_pieces)
        if return_hard_cases:
            return fit, float(total_err), hard_piece if hard_piece is not None else self.primary._empty_hard_case_state()
        return fit, float(total_err)

def _build_gp_context(max_size: int = 80):
    toolboxes = {'init_dex': base.Toolbox(), 'update': base.Toolbox(), 'query': base.Toolbox()}
    pset_map = {'init_dex': init_dex_pset, 'update': pset_update, 'query': query_pset}
    rettype_map = {'init_dex': list, 'update': float, 'query': float}
    height_limit_cfg = {
        "init_dex": 6,
        "update": 5,
        "query": 5,
    }
    for key, tb in toolboxes.items():
        pset = pset_map[key]
        rtype = rettype_map[key]
        tree_hmax = height_limit_cfg[key]

        tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4, type_=rtype)
        tb.register("individual", tools.initIterate, INDIVIDUAL_CLS, tb.expr)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("clone", copy.deepcopy)

        tb.register("expr_mut", gp.genFull, pset=pset, min_=1, max_=2, type_=rtype)

        tb.register("mate_one_point", gp.cxOnePoint)
        tb.register("mate_leaf_biased", gp.cxOnePointLeafBiased, termpb=0.1)

        tb.register("mut_uniform", gp.mutUniform, expr=tb.expr_mut, pset=pset)
        tb.register("mut_node_replace", gp.mutNodeReplacement, pset=pset)
        tb.register("mut_insert", gp.mutInsert, pset=pset)
        tb.register("mut_shrink", gp.mutShrink)
        if hasattr(gp, "mutEphemeral"):
            tb.register("mut_ephemeral", gp.mutEphemeral, mode="one")
        else:
            tb.register("mut_ephemeral", gp.mutNodeReplacement, pset=pset)

        for op in ["mate_one_point", "mate_leaf_biased",
                   "mut_uniform", "mut_node_replace", "mut_insert", "mut_shrink", "mut_ephemeral"]:
            tb.decorate(op, gp.staticLimit(key=attrgetter("height"), max_value=tree_hmax))
            tb.decorate(op, gp.staticLimit(key=len, max_value=max_size))

    def _ind_from_str(pset, s):
        return INDIVIDUAL_CLS.from_string(s, pset)

    skeleton_exprs = {
        "init_dex": [
            "list_3(select_hash(0,e), safe_mod(select_hash(0,e),102), 102, "
            "select_hash(1,e), safe_mod(select_hash(1,e),102), 102, "
            "select_hash(2,e), safe_mod(select_hash(2,e),102), 102)",

            "list_3(hash_on_slice(0,e,0,4), safe_mod(hash_on_slice(0,e,0,4),102), 102, "
            "hash_on_slice(1,e,0,4), safe_mod(hash_on_slice(1,e,0,4),102), 102, "
            "hash_on_slice(2,e,0,4), safe_mod(hash_on_slice(2,e,0,4),102), 102)"
        ],
        "update": [
            "base(update_count(e,0,1), update_count(e,1,1), update_count(e,2,1))",

            "base(updatecount_if(True,e,0,1), updatecount_if(True,e,1,1), updatecount_if(True,e,2,1))"
        ],
        "query": [
            "base_sel(0, query_date(e,0), query_date(e,1), query_date(e,2))",
            "base_sel(3, query_date(e,0), query_date(e,1), query_date(e,2))"
        ],
    }
    seed_exprs = {
        "init_dex": [
            "list_3(hash_salt(0,e,1), safe_mod(hash_salt(0,e,1),102), 102, "
            "hash_salt(1,e,1), safe_mod(hash_salt(1,e,1),102), 102, "
            "hash_salt(2,e,1), safe_mod(hash_salt(2,e,1),102), 102)"
        ],
        "update": [
            "base(write_count(e,0,safe_add(query_count(e,0),1)), "
            "write_count(e,1,safe_add(query_count(e,1),1)), "
            "write_count(e,2,safe_add(query_count(e,2),1)))"
        ],
        "query": [
            "base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2))"
        ],
    }

    skeleton_bank = {}
    seed_bank = {}
    for which in ("init_dex", "update", "query"):
        sb = []
        for expr in skeleton_exprs.get(which, []):
            try:
                sb.append(_ind_from_str(pset_map[which], expr))
            except Exception as e:
                print(f"[SKELETON_SKIP] which={which} expr={expr} reason={e}", flush=True)
        if not sb:
            try:
                sb = [toolboxes[which].individual()]
            except Exception:
                sb = []
        skeleton_bank[which] = sb

        tb = []
        for expr in seed_exprs.get(which, []):
            try:
                tb.append(_ind_from_str(pset_map[which], expr))
            except Exception as e:
                print(f"[SEED_SKIP] which={which} expr={expr} reason={e}", flush=True)
        if not tb:
            tb = [toolboxes[which].clone(x) for x in sb] if sb else []
        seed_bank[which] = tb
    llm_seed_bank = {
        "init_dex": [],
        "update": [],
        "query": [],
    }
    llm_team_bank = []

    return {
        "toolboxes": toolboxes,
        "pset_map": pset_map,
        "seed_bank": seed_bank,
        "skeleton_bank": skeleton_bank,
        "llm_seed_bank": llm_seed_bank,
        "llm_team_bank": llm_team_bank,
    }

def _build_local_llm_seed_teams():
    return [
        {
            "name": "cm_min_basic",
            "init_dex": "list_3(hash_salt(0,e,1), safe_mod(hash_salt(0,e,1),102), 102, "
                        "hash_salt(1,e,1), safe_mod(hash_salt(1,e,1),102), 102, "
                        "hash_salt(2,e,1), safe_mod(hash_salt(2,e,1),102), 102)",
            "update": "base(update_count(e,0,1), update_count(e,1,1), update_count(e,2,1))",
            "query": "base_sel(0, query_date(e,0), query_date(e,1), query_date(e,2))",
        },
        {
            "name": "read_before_write_median",
            "init_dex": "list_3(select_hash(0,e), safe_mod(hash_salt(0,e,11),102), 102, "
                        "select_hash(1,e), safe_mod(hash_salt(1,e,13),102), 102, "
                        "select_hash(2,e), safe_mod(hash_salt(2,e,17),102), 102)",
            "update": "base(write_count(e,0,safe_add(query_count(e,0),1)), "
                      "write_count(e,1,safe_add(query_count(e,1),1)), "
                      "write_count(e,2,safe_add(query_count(e,2),1)))",
            "query": "base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2))",
        },
        {
            "name": "conditional_write_avg",
            "init_dex": "list_3(hash_on_slice(0,e,0,8), safe_mod(hash_on_slice(0,e,0,8),102), 102, "
                        "hash_on_slice(1,e,4,12), safe_mod(hash_on_slice(1,e,4,12),102), 102, "
                        "hash_on_slice(2,e,8,16), safe_mod(hash_on_slice(2,e,8,16),102), 102)",
            "update": "base(updatecount_if(lt(query_count(e,0),query_count(e,1)),e,0,1), "
                      "updatecount_if(lt(query_count(e,1),query_count(e,2)),e,1,1), "
                      "updatecount_if(lt(query_count(e,2),query_count(e,0)),e,2,1))",
            "query": "base_sel(3, query_date(e,0), query_date(e,1), query_date(e,2))",
        },
    ]

def _populate_llm_seed_bank_from_cfg(gp_ctx, cfg):
    if "llm_seed_bank" not in gp_ctx:
        gp_ctx["llm_seed_bank"] = {
            "init_dex": [],
            "update": [],
            "query": [],
        }
    if "llm_team_bank" not in gp_ctx:
        gp_ctx["llm_team_bank"] = []

    mode = str(cfg.get("llm_mode", "off"))
    enabled = bool(cfg.get("llm_enable", False))

    if (not enabled) or (mode not in {"seed", "immigrant", "seed+immigrant"}):
        return gp_ctx

    expr_bank = _build_local_llm_seed_exprs()
    team_bank = _build_local_llm_seed_teams()
    keep_k = max(1, int(cfg.get("llm_num_candidates", 8)))

    for which in ("init_dex", "update", "query"):
        pset = gp_ctx["pset_map"][which]
        built = []
        seen = set()

        for ind in gp_ctx.get("skeleton_bank", {}).get(which, []):
            seen.add(str(ind))
        for ind in gp_ctx.get("seed_bank", {}).get(which, []):
            seen.add(str(ind))

        for expr in expr_bank.get(which, []):
            expr = str(expr).strip()
            if not expr or expr in seen:
                continue
            try:
                ind = INDIVIDUAL_CLS.from_string(expr, pset)
                built.append(ind)
                seen.add(str(ind))
            except Exception as e:
                print(f"[LLM_SEED_SKIP] which={which} expr={expr} reason={e}", flush=True)
            if len(built) >= keep_k:
                break
        gp_ctx["llm_seed_bank"][which] = built

    built_teams = []
    team_seen = set()
    for spec in team_bank:
        try:
            init_ind = INDIVIDUAL_CLS.from_string(str(spec["init_dex"]).strip(), gp_ctx["pset_map"]["init_dex"])
            update_ind = INDIVIDUAL_CLS.from_string(str(spec["update"]).strip(), gp_ctx["pset_map"]["update"])
            query_ind = INDIVIDUAL_CLS.from_string(str(spec["query"]).strip(), gp_ctx["pset_map"]["query"])
            key = (str(init_ind), str(update_ind), str(query_ind))
            if key in team_seen:
                continue
            built_teams.append({
                "name": str(spec.get("name", f"team_{len(built_teams)}")),
                "init_dex": init_ind,
                "update": update_ind,
                "query": query_ind,
            })
            team_seen.add(key)
        except Exception as e:
            print(f"[LLM_TEAM_SKIP] name={spec.get('name', 'unknown')} reason={e}", flush=True)

    gp_ctx["llm_team_bank"] = built_teams[:keep_k]
    return gp_ctx

def _apply_mutation_with_ctx(toolboxes, op_name: str, which: str, ind):
    tb = toolboxes[which]
    try:
        ret = getattr(tb, op_name)(ind)
        if isinstance(ret, tuple) and ret:
            return ret[0]
        return ind
    except Exception:
        try:
            ret = tb.mut_uniform(ind)
            if isinstance(ret, tuple) and ret:
                return ret[0]
            return ind
        except Exception:
            return ind

def _skeleton_individual_from_ctx(ctx, which):
    bank = ctx.get("skeleton_bank", {}).get(which, [])
    if bank:
        return ctx["toolboxes"][which].clone(random.choice(bank))
    return ctx["toolboxes"][which].individual()


def _seeded_individual_from_ctx(ctx, which):
    bank = ctx.get("seed_bank", {}).get(which, [])
    if bank:
        return ctx["toolboxes"][which].clone(random.choice(bank))
    return _skeleton_individual_from_ctx(ctx, which)


def _llm_seeded_individual_from_ctx(ctx, which):
    bank = ctx.get("llm_seed_bank", {}).get(which, [])
    if not bank:
        # 这里先回退到普通 seed，保证第一步不改搜索行为
        return _seeded_individual_from_ctx(ctx, which)
    return ctx["toolboxes"][which].clone(random.choice(bank))


def _normalize_init_probs(p_skeleton=0.70, p_seed=0.20, p_llm_seed=0.0):
    p_skeleton = max(0.0, float(p_skeleton))
    p_seed = max(0.0, float(p_seed))
    p_llm_seed = max(0.0, float(p_llm_seed))

    total = p_skeleton + p_seed + p_llm_seed
    if total >= 1.0:
        # 留一点概率给随机初始化
        scale = 0.95 / total
        p_skeleton *= scale
        p_seed *= scale
        p_llm_seed *= scale
    return p_skeleton, p_seed, p_llm_seed


def _init_individual_from_ctx(ctx, which, p_skeleton=0.70, p_seed=0.20, p_llm_seed=0.0):
    p_skeleton, p_seed, p_llm_seed = _normalize_init_probs(
        p_skeleton=p_skeleton,
        p_seed=p_seed,
        p_llm_seed=p_llm_seed,
    )

    r = random.random()
    if r < p_skeleton:
        return _skeleton_individual_from_ctx(ctx, which)
    if r < p_skeleton + p_seed:
        return _seeded_individual_from_ctx(ctx, which)
    if r < p_skeleton + p_seed + p_llm_seed:
        return _llm_seeded_individual_from_ctx(ctx, which)
    return ctx["toolboxes"][which].individual()


def _llm_team_from_ctx(ctx):
    bank = ctx.get("llm_team_bank", [])
    if not bank:
        return None
    team = random.choice(bank)
    return {
        "init_dex": ctx["toolboxes"]["init_dex"].clone(team["init_dex"]),
        "update": ctx["toolboxes"]["update"].clone(team["update"]),
        "query": ctx["toolboxes"]["query"].clone(team["query"]),
        "name": str(team.get("name", "llm_team")),
    }


def _prepare_llm_team_bank_for_cfg(cfg):
    gp_ctx = _build_gp_context(max_size=cfg["max_size"])
    gp_ctx = _populate_llm_seed_bank_from_cfg(gp_ctx, cfg)
    evaluator = _make_evaluator_from_cfg(cfg)
    gp_ctx = _filter_llm_team_bank_with_evaluator(gp_ctx, evaluator, cfg)
    return list(gp_ctx.get("llm_team_bank", []))


def _inject_llm_immigrants_into_state(state, cfg, llm_team_bank):
    if not llm_team_bank:
        return state, 0

    m = min(
        max(1, int(cfg.get("llm_immigrant_count", 2))),
        len(llm_team_bank),
        len(state["fits"]),
    )
    if m <= 0:
        return state, 0

    evaluator = _make_evaluator_from_cfg(cfg)
    if cfg.get("hard_case_replay", False):
        evaluator.import_hard_case_state(state.get("hard_case_state"))

    chosen = list(llm_team_bank) if len(llm_team_bank) <= m else random.sample(llm_team_bank, k=m)
    repl_idx = sorted(range(len(state["birth"])), key=lambda j: state["birth"][j])[:m]

    inserted = 0
    for rep_idx, spec in zip(repl_idx, chosen):
        init_ind = copy.deepcopy(spec["init_dex"])
        update_ind = copy.deepcopy(spec["update"])
        query_ind = copy.deepcopy(spec["query"])
        try:
            fit, err, case_vec = evaluator.evaluate_individual(
                init_ind, update_ind, query_ind, return_case_vec=True
            )
        except Exception as e:
            fit, err, case_vec = _outer_eval_fail_result(
                evaluator,
                phase="legacy_llm_immigrant",
                exc=e,
                island_idx=state.get("island_idx", None),
            )

        state["pops"]["init_dex"][rep_idx] = init_ind
        state["pops"]["update"][rep_idx] = update_ind
        state["pops"]["query"][rep_idx] = query_ind
        state["fits"][rep_idx] = (float(fit), float(err))
        state["case_vecs"][rep_idx] = tuple(float(x) for x in case_vec)
        state["birth"][rep_idx] = int(state["step_counter"])
        state["step_counter"] = int(state["step_counter"]) + 1
        inserted += 1

    _, best_fit, best_err, _ = _refresh_island_best(state)
    state["best_fitness"] = float(best_fit)
    state["best_error"] = float(best_err)
    if cfg.get("hard_case_replay", False):
        state["scored_hard_case_version"] = int(state.get("hard_case_state", {}).get("version", 0))
    return state, inserted


def _filter_llm_team_bank_with_evaluator(gp_ctx, evaluator, cfg):
    team_bank = list(gp_ctx.get("llm_team_bank", []))
    if not team_bank:
        return gp_ctx

    scored = []
    for spec in team_bank:
        try:
            fit, err, case_vec = evaluator.evaluate_individual(
                spec["init_dex"],
                spec["update"],
                spec["query"],
                return_case_vec=True,
            )
            scored.append((float(fit), float(err), tuple(float(x) for x in case_vec), spec))
        except Exception as e:
            print(f"[LLM_TEAM_FILTER_SKIP] name={spec.get('name', 'unknown')} reason={e}", flush=True)

    scored.sort(key=lambda x: (-x[0], x[1], str(x[3].get("name", ""))))
    keep_k = max(1, min(len(scored), int(cfg.get("llm_num_candidates", 8))))

    kept = []
    for fit, err, case_vec, spec in scored[:keep_k]:
        kept.append({
            "name": str(spec.get("name", f"team_{len(kept)}")),
            "init_dex": spec["init_dex"],
            "update": spec["update"],
            "query": spec["query"],
            "fitness": float(fit),
            "error": float(err),
            "case_vec": tuple(float(x) for x in case_vec),
        })

    gp_ctx["llm_team_bank"] = kept
    if kept:
        print(f"[LLM_TEAM_FILTER] total={len(team_bank)} kept={len(kept)} best_fit={kept[0]['fitness']:.6f} best_err={kept[0]['error']:.2f}", flush=True)
    else:
        print(f"[LLM_TEAM_FILTER] total={len(team_bank)} kept=0", flush=True)
    return gp_ctx

def _normalize_llm_mode(mode: str) -> str:
    m = str(mode or "none").strip().lower()
    alias = {
        "off": "none",
        "seed": "seeds",
        "immigrant": "stagnation",
        "seed+immigrant": "both",
    }
    return alias.get(m, m)


def _parse_llm_target_funcs(raw) -> set:
    if isinstance(raw, (list, tuple, set)):
        toks = [str(x).strip() for x in raw]
    else:
        toks = [s.strip() for s in str(raw or "").split(",")]
    out = set()
    for tok in toks:
        if tok in {"init", "init_dex"}:
            out.add("init_dex")
        elif tok == "update":
            out.add("update")
        elif tok == "query":
            out.add("query")
    if not out:
        out = {"update", "query"}
    return out


def _runtime_primitive_names(pset):
    out = set()
    for _, prims in getattr(pset, "primitives", {}).items():
        for prim in prims:
            out.add(str(getattr(prim, "name", "")))
    return out


def _tree_primitive_names(tree):
    out = set()
    for node in tree:
        try:
            if int(getattr(node, "arity", 0)) > 0:
                out.add(str(getattr(node, "name", "")))
        except Exception:
            continue
    return out


class PrimitiveSpecReference:
    ROOT_TYPE_EXPECTED = {
        "init_dex": list,
        "update": float,
        "query": float,
    }
    FORBIDDEN = {
        "init_dex": {
            "update_count", "write_count", "updatecount_if", "writecount_if",
            "update_state", "writestate_if",
            "query_count", "query_state", "query_date", "cnt_rdstate",
        },
        "update": {"query_date", "cnt_rdstate"},
        "query": {
            "update_count", "write_count", "updatecount_if", "writecount_if",
            "update_state", "writestate_if", "query_count", "query_state",
        },
    }
    INIT_HASH_NAMES = {"select_hash", "hash_salt", "hash_on_slice"}
    UPDATE_REAL_WRITE_NAMES = {"update_count", "write_count"}
    QUERY_REAL_READ_NAMES = {"query_date"}

    def __init__(self, pset_map, cfg, logger):
        self.pset_map = pset_map
        self.cfg = cfg
        self.logger = logger
        self.runtime_allowed = {
            k: _runtime_primitive_names(pset_map[k])
            for k in ("init_dex", "update", "query")
        }
        self.ref_allowed = {}
        self.ref_source = {}
        self._load_reference_files()

    def _warn(self, msg, **extra):
        if self.logger is not None:
            self.logger.warn(msg, **extra)
        else:
            print(f"[LLM_WARN] {msg} {extra}", flush=True)

    @staticmethod
    def _extract_allowed_from_file(path: str):
        allowed = set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
        except Exception:
            return allowed
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "addPrimitive":
                continue
            if not node.args:
                continue
            head = node.args[0]
            if isinstance(head, ast.Name):
                allowed.add(str(head.id))
            elif isinstance(head, ast.Attribute):
                allowed.add(str(head.attr))
            elif isinstance(head, ast.Constant) and isinstance(head.value, str):
                allowed.add(str(head.value))
        return allowed

    def _load_reference_files(self):
        path_map = {
            "init_dex": str(self.cfg.get("llm_ref_init_pset_path", "") or "").strip(),
            "update": str(self.cfg.get("llm_ref_update_pset_path", "") or "").strip(),
            "query": str(self.cfg.get("llm_ref_query_pset_path", "") or "").strip(),
        }
        for which in ("init_dex", "update", "query"):
            path = path_map[which]
            if path and os.path.exists(path):
                allowed = self._extract_allowed_from_file(path)
                if allowed:
                    self.ref_allowed[which] = allowed
                    self.ref_source[which] = path
                    continue
                self._warn("failed to parse reference file, fallback runtime pset", which=which, path=path)
            elif path:
                self._warn("reference file missing, fallback runtime pset", which=which, path=path)
            self.ref_allowed[which] = set(self.runtime_allowed[which])
            self.ref_source[which] = "runtime_pset"

    def check_root_type(self, which: str, tree):
        exp_t = self.ROOT_TYPE_EXPECTED[which]
        try:
            ret_t = getattr(tree[0], "ret", None)
        except Exception:
            return False, f"{which}_root_type_unavailable"
        if ret_t is exp_t:
            return True, ""
        if str(getattr(ret_t, "__name__", ret_t)).lower() == str(getattr(exp_t, "__name__", exp_t)).lower():
            return True, ""
        return False, f"{which}_root_type_mismatch expected={exp_t} got={ret_t}"

    def check_dual_source_primitives(self, which: str, tree):
        names = _tree_primitive_names(tree)
        bad = []
        warns = []
        for n in names:
            if n not in self.runtime_allowed[which]:
                bad.append(f"{which}_primitive_not_in_runtime:{n}")
                continue
            if self.ref_allowed[which] and n not in self.ref_allowed[which]:
                warns.append(f"{which}_primitive_not_in_ref_but_runtime_compatible:{n}")
        return bad, warns

    def check_forbidden(self, which: str, tree):
        names = _tree_primitive_names(tree)
        hit = sorted([n for n in names if n in self.FORBIDDEN[which]])
        return [f"{which}_forbidden_primitive:{n}" for n in hit]

    def quick_prescreen(self, team):
        reasons = []
        if len(_tree_primitive_names(team["init_dex"]).intersection(self.INIT_HASH_NAMES)) <= 0:
            reasons.append("init_dex_missing_hash_primitive")
        if len(_tree_primitive_names(team["update"]).intersection(self.UPDATE_REAL_WRITE_NAMES)) <= 0:
            reasons.append("update_missing_real_write_primitive")
        if len(_tree_primitive_names(team["query"]).intersection(self.QUERY_REAL_READ_NAMES)) <= 0:
            reasons.append("query_missing_real_read_primitive")
        return reasons


class LLMRunLogger:
    def __init__(self, log_path: str = ""):
        self.log_path = str(log_path or "").strip()
        if self.log_path:
            parent = os.path.dirname(self.log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    def _write_jsonl(self, event, payload):
        if not self.log_path:
            return
        rec = {"ts": datetime.utcnow().isoformat() + "Z", "event": str(event)}
        rec.update(payload)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def info(self, msg, **payload):
        self._write_jsonl("info", {"message": str(msg), **payload})
        print(f"[LLM] {msg} {payload}", flush=True)

    def warn(self, msg, **payload):
        self._write_jsonl("warning", {"message": str(msg), **payload})
        print(f"[LLM_WARN] {msg} {payload}", flush=True)


class LLMProposalEngine:
    """
    offline_json / jsonl candidate protocol (v1, fixed):
    team:
      {"mode":"team","init_dex":"...","update":"...","query":"...","rationale":"..."}
    single_tree:
      {"mode":"single_tree","target":"query","expr":"...","rationale":"..."}
    v1 seeds/stagnation consume only mode=team; single_tree is parsed but not applied.
    """

    def __init__(self, cfg, ref_spec: PrimitiveSpecReference, logger: LLMRunLogger):
        self.cfg = cfg
        self.ref_spec = ref_spec
        self.logger = logger
        self.mode = _normalize_llm_mode(cfg.get("llm_mode", "none"))
        self.provider = str(cfg.get("llm_provider", "none") or "none").strip().lower()
        self.target_funcs = _parse_llm_target_funcs(cfg.get("llm_target_funcs", "update,query"))

    def _extract_json(self, text):
        txt = str(text or "").strip()
        if not txt:
            return None
        if txt.startswith("{") or txt.startswith("["):
            return txt
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", txt, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        l = txt.find("{")
        r = txt.rfind("}")
        if l >= 0 and r > l:
            return txt[l:r + 1]
        return None

    def _build_report(self, evaluator, team=None, fit=None, err=None, case_vec=None, hard_cases=None, extra_prompt_hints=None):
        if team is None:
            return {"target_funcs": sorted(self.target_funcs)}
        ast_summary = {}
        try:
            init_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["init_dex"]))
            upd_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["update"]))
            qry_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["query"]))
            ast_summary = {
                "legality": {
                    "init": evaluator._ast_legality_check("init", init_ast),
                    "update": evaluator._ast_legality_check("update", upd_ast),
                    "query": evaluator._ast_legality_check("query", qry_ast),
                },
                "effect": {
                    "init": evaluator._ast_effect_summary(init_ast),
                    "update": evaluator._ast_effect_summary(upd_ast),
                    "query": evaluator._ast_effect_summary(qry_ast),
                },
                "pattern": {
                    "init": evaluator._ast_pattern_summary("init", init_ast),
                    "update": evaluator._ast_pattern_summary("update", upd_ast),
                    "query": evaluator._ast_pattern_summary("query", qry_ast),
                },
            }
        except Exception:
            ast_summary = {}

        analysis = {
            "init_dex": evaluator.analyze_init_tree(team["init_dex"]),
            "update": evaluator.analyze_update_tree(team["update"]),
            "query": evaluator.analyze_query_tree(team["query"]),
        }

        prompt_hints = self._derive_prompt_hints(analysis, ast_summary)
        prompt_hints = _merge_prompt_hint_dicts(prompt_hints, extra_prompt_hints or {})

        return {
            "canonical": {
                "init_dex": evaluator._canonical_tree_str(team["init_dex"]),
                "update": evaluator._canonical_tree_str(team["update"]),
                "query": evaluator._canonical_tree_str(team["query"]),
            },
            "fitness": None if fit is None else float(fit),
            "error": None if err is None else float(err),
            "case_vec": list(case_vec) if case_vec is not None else None,
            "analysis": analysis,
            "ast_summary": ast_summary,
            "constraints": {
                "init_dex": "index only; no counter/state read/write",
                "update": "read/write variable+counter/state path",
                "query": "read-only counter/state path; no writes",
                "output": "team expressions only (init_dex/update/query), not full python code",
            },
            "hard_cases": hard_cases,
            "target_funcs": sorted(self.target_funcs),
            "prompt_hints": prompt_hints,
        }

    def _derive_prompt_hints(self, analysis: dict, ast_summary: dict):
        patt = ast_summary.get("pattern", {}) if isinstance(ast_summary, dict) else {}
        eff = ast_summary.get("effect", {}) if isinstance(ast_summary, dict) else {}
        hints = {
            "failure_buckets": [],
            "hard_avoid": [],
            "prefer": [],
            "repair_focus": [],
        }

        upd_pat = patt.get("update", {}) if isinstance(patt, dict) else {}
        qry_pat = patt.get("query", {}) if isinstance(patt, dict) else {}
        init_pat = patt.get("init", {}) if isinstance(patt, dict) else {}
        upd_eff = eff.get("update", {}) if isinstance(eff, dict) else {}
        qry_eff = eff.get("query", {}) if isinstance(eff, dict) else {}

        def add_unique(dst, item):
            s = str(item).strip()
            if s and s not in dst:
                dst.append(s)

        if int(upd_eff.get("real_write_calls", 0)) == 0:
            add_unique(hints["failure_buckets"], "real_write_zero")
            add_unique(hints["hard_avoid"], "Do not generate update expressions with zero real writes.")
            add_unique(hints["repair_focus"], "Make update perform at least one real counter write.")
        if int(qry_eff.get("query_date_calls", 0)) == 0:
            add_unique(hints["failure_buckets"], "query_date_zero")
            add_unique(hints["hard_avoid"], "Do not generate query expressions with zero query_date reads.")
            add_unique(hints["repair_focus"], "Make query perform stable real reads via query_date-related access.")
        if int(init_pat.get("nonconst_hash_idx", 0)) > 0:
            add_unique(hints["failure_buckets"], "nonconst_hash")
            add_unique(hints["hard_avoid"], "Avoid dynamic hash ids; prefer small stable constant hash ids.")
            add_unique(hints["repair_focus"], "Stabilize hash ids into small constants when possible.")
        if int(upd_pat.get("nonconst_path_idx", 0)) > 0 or int(qry_pat.get("nonconst_path_idx", 0)) > 0:
            add_unique(hints["failure_buckets"], "nonconst_path")
            add_unique(hints["hard_avoid"], "Avoid dynamic path ids; prefer small stable constant path ids shared by update and query.")
            add_unique(hints["repair_focus"], "Reduce non-constant path ids and keep update/query access shape aligned.")
        if int(upd_pat.get("bad_write_context", 0)) > 0:
            add_unique(hints["failure_buckets"], "bad_write_ctx")
            add_unique(hints["hard_avoid"], "Do not place real writes inside comparisons, boolean gates, arithmetic-only wrappers, or nested min/max guards.")
            add_unique(hints["repair_focus"], "Move writes into a clean update context.")

        q_info = analysis.get("query", {}) if isinstance(analysis, dict) else {}
        u_info = analysis.get("update", {}) if isinstance(analysis, dict) else {}
        i_info = analysis.get("init_dex", {}) if isinstance(analysis, dict) else {}

        if not bool(i_info.get("root_ok", True)):
            add_unique(hints["prefer"], "Keep init_dex shaped like a list_3-style index constructor.")
        if not bool(u_info.get("root_ok", True)):
            add_unique(hints["prefer"], "Keep update rooted like base-style composition.")
        if not bool(q_info.get("root_ok", True)):
            add_unique(hints["prefer"], "Keep query rooted like base_sel-style selection.")

        add_unique(hints["prefer"], "Keep update/query shared access shape stable.")
        add_unique(hints["prefer"], "Prefer simple, stable sketch structure over clever but fragile expressions.")

        return hints

    def _build_repair_feedback(self, failed_records):
        if not failed_records:
            return ""
        counts = Counter()
        samples = []
        for rec in failed_records:
            for r in rec.get("reasons", []):
                key = str(r).strip()
                if not key:
                    continue
                counts[key] += 1
                if len(samples) < 6 and key not in samples:
                    samples.append(key)
        if not counts:
            return ""
        top = [f"{k} x{v}" for k, v in counts.most_common(6)]
        lines = [
            "Previous candidate(s) were rejected or failed.",
            "Top failure reasons:",
        ]
        lines.extend([f"- {x}" for x in top])
        lines.append("Repair goal: keep the general sketch idea, but explicitly eliminate the listed failure modes.")
        return "\n".join(lines)

    def _build_prompt(self, phase: str, report: dict, repair_feedback: str = ""):
        schema = {
            "mode": "team",
            "init_dex": "<expr>",
            "update": "<expr>",
            "query": "<expr>",
            "rationale": "<short>"
        }
        hints = report.get("prompt_hints", {}) if isinstance(report, dict) else {}
        hard_avoid = list(hints.get("hard_avoid", [])) if isinstance(hints, dict) else []
        prefer = list(hints.get("prefer", [])) if isinstance(hints, dict) else []
        repair_focus = list(hints.get("repair_focus", [])) if isinstance(hints, dict) else []
        failure_buckets = list(hints.get("failure_buckets", [])) if isinstance(hints, dict) else []

        lines = [
            "Return JSON only. No markdown, no explanations, no code fences.",
            "Output exactly ONE JSON object with this schema:",
            json.dumps(schema, ensure_ascii=False),
            "Do not output an array.",
            "Do not output a top-level key named candidates.",
            "Each field init_dex/update/query must be a single DEAP expression string, not Python code.",
            "You are proposing a sketch team, not full Python source code.",
            f"phase={phase}",
            f"target_funcs={json.dumps(report.get('target_funcs', []), ensure_ascii=False)}",
            "Hard requirements:",
            "- init_dex must stay index-only and must not perform counter/state read/write.",
            "- update must keep real counter writes in a clean update context.",
            "- query must stay read-only and must not write.",
        ]

        if hard_avoid:
            lines.append("Hard avoid:")
            lines.extend([f"- {x}" for x in hard_avoid])
        if prefer:
            lines.append("Prefer:")
            lines.extend([f"- {x}" for x in prefer])
        if failure_buckets:
            lines.append("Current dominant failure buckets:")
            lines.extend([f"- {x}" for x in failure_buckets])
        if repair_focus:
            lines.append("Primary repair targets:")
            lines.extend([f"- {x}" for x in repair_focus])
        if repair_feedback:
            lines.append("Repair feedback from previous failed proposal:")
            lines.append(repair_feedback)

        lines.append("Current team/context JSON:")
        lines.append(json.dumps(report, ensure_ascii=False))
        return "\n".join(lines)

    def _load_offline(self):
        path = str(self.cfg.get("llm_offline_candidates_path", "") or "").strip()
        if not path:
            self.logger.warn("offline candidate path empty")
            return []
        if not os.path.exists(path):
            self.logger.warn("offline candidate path missing", path=path)
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            if path.lower().endswith(".jsonl"):
                out = []
                for ln in text.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        out.append(json.loads(ln))
                    except Exception:
                        self.logger.warn("invalid jsonl line skipped", sample=ln[:100])
                return out
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and isinstance(obj.get("candidates"), list):
                return obj["candidates"]
            if isinstance(obj, dict):
                return [obj]
            return []
        except Exception as e:
            self.logger.warn("failed loading offline candidates", error=str(e), path=path)
            return []

    def _call_openai_compatible(self, prompt_text: str):
        import os
        import json

        model = str(self.cfg.get("llm_model", "") or "").strip()
        base_url = str(self.cfg.get("llm_base_url", "") or "").strip()
        api_key_env = str(self.cfg.get("llm_api_key_env", "") or "").strip()
        timeout_s = float(self.cfg.get("llm_timeout", 30.0))

        if not model:
            self.logger.warn("openai_compatible request skipped", error="empty llm_model")
            return ""

        if not base_url:
            self.logger.warn("openai_compatible request skipped", error="empty llm_base_url")
            return ""

        if not api_key_env:
            self.logger.warn("openai_compatible request skipped", error="empty llm_api_key_env")
            return ""

        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            self.logger.warn(
                "openai_compatible request skipped",
                error=f"environment variable not set: {api_key_env}",
            )
            return ""

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url.rstrip("/"),
                timeout=timeout_s,
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0,
            )

            text = resp.choices[0].message.content
            return text if text is not None else ""

        except Exception as e:
            self.logger.warn(
                "openai_compatible request failed",
                error=str(e),
                base_url=base_url,
                model=model,
            )
            return ""

    def fetch_raw_candidates(self, phase: str, report: dict, repair_feedback: str = ""):
        if self.provider == "offline_json":
            return self._load_offline()

        if self.provider == "openai_compatible":
            prompt_text = self._build_prompt(phase, report, repair_feedback=repair_feedback)
            raw_text = self._call_openai_compatible(prompt_text)
            if not raw_text:
                return []
            return [raw_text]

        self.logger.info("llm provider disabled", provider=self.provider, phase=phase)
        return []

    def parse_candidate_objects(self, raw):
        out = []

        def _consume(cur):
            if cur is None:
                return

            if isinstance(cur, str):
                jt = self._extract_json(cur)
                if not jt:
                    return
                try:
                    cur = json.loads(jt)
                except Exception:
                    return

            if isinstance(cur, list):
                for x in cur:
                    _consume(x)
                return

            if isinstance(cur, dict) and isinstance(cur.get("candidates"), list):
                for x in cur["candidates"]:
                    _consume(x)
                return

            if not isinstance(cur, dict):
                return

            mode = str(cur.get("mode", "")).strip().lower()

            # 允许没有 mode，但字段齐全时按 team 处理
            if (not mode) and any(k in cur for k in ("init_dex", "update", "query")):
                mode = "team"

            if mode == "team":
                out.append({
                    "mode": "team",
                    "init_dex": str(cur.get("init_dex", "")).strip(),
                    "update": str(cur.get("update", "")).strip(),
                    "query": str(cur.get("query", "")).strip(),
                    "rationale": str(cur.get("rationale", "")).strip(),
                })
                return

            if mode == "single_tree":
                out.append({
                    "mode": "single_tree",
                    "target": str(cur.get("target", "")).strip(),
                    "expr": str(cur.get("expr", "")).strip(),
                    "rationale": str(cur.get("rationale", "")).strip(),
                })
                return

        for item in raw:
            _consume(item)

        return out

    def materialize_team(self, obj, pset_map, base_team=None):
        if obj.get("mode") != "team":
            return None, ["not_team_mode"]
        target_funcs = self.target_funcs
        out = {}
        errs = []
        for which in ("init_dex", "update", "query"):
            try:
                if (which not in target_funcs) and (base_team is not None):
                    out[which] = copy.deepcopy(base_team[which])
                    continue
                expr = str(obj.get(which, "")).strip()
                if not expr:
                    if base_team is not None:
                        out[which] = copy.deepcopy(base_team[which])
                        continue
                    errs.append(f"missing_{which}")
                    continue
                out[which] = INDIVIDUAL_CLS.from_string(expr, pset_map[which])
            except Exception as e:
                errs.append(f"parse_{which}_failed:{e}")
        if errs:
            return None, errs
        return out, []

    def validate_team_candidate(self, team, evaluator, existing_canon=None):
        reasons = []
        warns = []
        for which in ("init_dex", "update", "query"):
            ok, msg = self.ref_spec.check_root_type(which, team[which])
            if not ok:
                reasons.append(msg)
            bad_rt, warn_rt = self.ref_spec.check_dual_source_primitives(which, team[which])
            reasons.extend(bad_rt)
            warns.extend(warn_rt)
            reasons.extend(self.ref_spec.check_forbidden(which, team[which]))

        reasons.extend(self.ref_spec.quick_prescreen(team))
        if reasons:
            return {"ok": False, "reasons": sorted(set(reasons)), "warnings": sorted(set(warns))}

        try:
            init_info = evaluator.analyze_init_tree(team["init_dex"])
            update_info = evaluator.analyze_update_tree(team["update"])
            query_info = evaluator.analyze_query_tree(team["query"])
            if init_info.get("forbidden_hits"):
                reasons.append("init_forbidden_hits")
            if update_info.get("forbidden_hits"):
                reasons.append("update_forbidden_hits")
            if query_info.get("forbidden_hits"):
                reasons.append("query_forbidden_hits")
            init_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["init_dex"]))
            upd_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["update"]))
            qry_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["query"]))
            if evaluator._ast_legality_check("init", init_ast).get("hard_illegal"):
                reasons.append("init_ast_hard_illegal")
            if evaluator._ast_legality_check("update", upd_ast).get("hard_illegal"):
                reasons.append("update_ast_hard_illegal")
            if evaluator._ast_legality_check("query", qry_ast).get("hard_illegal"):
                reasons.append("query_ast_hard_illegal")
        except Exception as e:
            reasons.append(f"ast_validate_failed:{e}")
        if reasons:
            return {"ok": False, "reasons": sorted(set(reasons)), "warnings": sorted(set(warns))}

        key = evaluator._canonical_triplet_key(team["init_dex"], team["update"], team["query"])
        if existing_canon is not None and key in existing_canon:
            return {"ok": False, "reasons": ["duplicate_canonical_team"], "warnings": sorted(set(warns))}

        try:
            fit, err, case_vec = evaluator.evaluate_individual(
                team["init_dex"], team["update"], team["query"], return_case_vec=True
            )
        except Exception as e:
            return {"ok": False, "reasons": [f"evaluate_failed:{e}"], "warnings": sorted(set(warns))}
        return {
            "ok": True,
            "team": team,
            "fit": float(fit),
            "err": float(err),
            "case_vec": tuple(float(x) for x in case_vec),
            "key": key,
            "warnings": sorted(set(warns)),
        }

    def prepare_phase_candidates(self, phase, gp_ctx, evaluator, base_team, existing_canon, limit, extra_prompt_hints=None):
        if not bool(self.cfg.get("llm_enable", False)):
            return []
        mode = _normalize_llm_mode(self.cfg.get("llm_mode", "none"))
        if phase == "seed" and mode not in {"seeds", "both"}:
            return []
        if phase == "stagnation" and mode not in {"stagnation", "both"}:
            return []
        fit = None
        err = None
        vec = None
        hard = None
        try:
            fit, err, vec = evaluator.evaluate_individual(
                base_team["init_dex"], base_team["update"], base_team["query"], return_case_vec=True
            )
            if not bool(self.cfg.get("llm_use_case_vec", False)):
                vec = None
            if bool(self.cfg.get("llm_use_hard_cases", False)):
                hard = evaluator.mine_hard_cases(base_team["init_dex"], base_team["update"], base_team["query"])
        except Exception:
            fit, err, vec, hard = None, None, None, None
        report = self._build_report(
            evaluator=evaluator,
            team=base_team,
            fit=fit,
            err=err,
            case_vec=vec,
            hard_cases=hard,
            extra_prompt_hints=extra_prompt_hints,
        )

        out = []
        seen = set(existing_canon or set())
        failed_records = []
        n_materialized = 0
        n_validated = 0
        n_evaluated = 0
        total_raw = 0
        total_parsed = 0
        total_team_parsed = 0

        def _consume_raw(raw_list):
            nonlocal n_materialized, n_validated, n_evaluated, total_raw, total_parsed, total_team_parsed, out, seen, failed_records
            total_raw += len(raw_list)
            parsed = self.parse_candidate_objects(raw_list)
            total_parsed += len(parsed)
            parsed_teams = [x for x in parsed if x.get("mode") == "team"]
            total_team_parsed += len(parsed_teams)
            n_single = len([x for x in parsed if x.get("mode") == "single_tree"])
            if n_single > 0:
                self.logger.info("single_tree parsed but not consumed in v1", phase=phase, count=n_single)

            for obj in parsed_teams:
                team, perr = self.materialize_team(obj, gp_ctx["pset_map"], base_team=base_team)
                if perr:
                    failed_records.append({"stage": "materialize", "reasons": list(perr)})
                    continue
                n_materialized += 1
                chk = self.validate_team_candidate(team, evaluator, existing_canon=seen)
                if not chk.get("ok", False):
                    failed_records.append({"stage": "validate", "reasons": list(chk.get("reasons", []))})
                    continue
                n_validated += 1
                n_evaluated += 1
                if chk.get("warnings"):
                    self.logger.warn("runtime/reference conflict accepted by runtime compatibility", warnings=chk.get("warnings"))
                seen.add(chk["key"])
                out.append({
                    "team": chk["team"],
                    "fit": chk["fit"],
                    "err": chk["err"],
                    "case_vec": chk["case_vec"],
                    "rationale": obj.get("rationale", ""),
                    "source": self.provider,
                })
                if len(out) >= max(1, int(limit)):
                    break

        raw = self.fetch_raw_candidates(phase, report)
        _consume_raw(raw)

        repair_rounds = 0
        max_repair_rounds = 1 if self.provider == "openai_compatible" else 0
        if "llm_repair_rounds" in self.cfg:
            try:
                max_repair_rounds = max(0, int(self.cfg.get("llm_repair_rounds", max_repair_rounds)))
            except Exception:
                pass

        while len(out) < max(1, int(limit)) and repair_rounds < max_repair_rounds and failed_records:
            repair_rounds += 1
            repair_feedback = self._build_repair_feedback(failed_records)
            self.logger.info(
                "llm repair round",
                phase=phase,
                round=repair_rounds,
                failures=len(failed_records),
                accepted=len(out),
            )
            repair_raw = self.fetch_raw_candidates(phase, report, repair_feedback=repair_feedback)
            if not repair_raw:
                break
            failed_records = []
            _consume_raw(repair_raw)

        self.logger.info(
            "llm candidate summary",
            phase=phase,
            source=self.provider,
            raw=total_raw,
            parsed=total_parsed,
            team_parsed=total_team_parsed,
            materialized=n_materialized,
            validated=n_validated,
            evaluated=n_evaluated,
            accepted=len(out),
            repair_rounds=repair_rounds,
            use_case_vec=bool(self.cfg.get("llm_use_case_vec", False)),
            use_hard_cases=bool(self.cfg.get("llm_use_hard_cases", False)),
            failure_buckets=report.get("prompt_hints", {}).get("failure_buckets", []),
        )
        return out

def _merge_prompt_hint_dicts(*hint_dicts):
    out = {
        "failure_buckets": [],
        "hard_avoid": [],
        "prefer": [],
        "repair_focus": [],
    }

    for hints in hint_dicts:
        if not isinstance(hints, dict):
            continue
        for k in out.keys():
            vals = hints.get(k, [])
            if not isinstance(vals, (list, tuple)):
                continue
            for item in vals:
                s = str(item).strip()
                if s and s not in out[k]:
                    out[k].append(s)
    return out


def _collect_recent_failure_hints_from_states(island_states):
    bucket_counts = Counter()
    metric_counts = Counter()

    for st in island_states or []:
        if not isinstance(st, dict):
            continue

        hc = st.get("hard_case_state", {})
        if isinstance(hc, dict):
            stage_bucket_cases = hc.get("stage_bucket_cases", [])
            if isinstance(stage_bucket_cases, list):
                for mp in stage_bucket_cases:
                    if not isinstance(mp, dict):
                        continue
                    for bk, recs in mp.items():
                        key = str(bk).strip()
                        if not key:
                            continue
                        try:
                            bucket_counts[key] += len(recs or [])
                        except Exception:
                            bucket_counts[key] += 1

        recent_hist = st.get("recent_diag_history", [])
        if isinstance(recent_hist, list):
            for dbg in recent_hist:
                if not isinstance(dbg, dict):
                    continue
                metric_counts["real_write_zero"] += int(dbg.get("real_write_zero", 0))
                metric_counts["query_date_zero"] += int(dbg.get("query_date_zero", 0))
                metric_counts["nonconst_hash"] += int(dbg.get("nonconst_hash_idx_total", 0))
                metric_counts["nonconst_path"] += int(dbg.get("nonconst_path_idx_total", 0))
                metric_counts["bad_write_ctx"] += int(dbg.get("bad_write_context_total", 0))

    ordered = []
    seen = set()

    for bk, _ in bucket_counts.most_common():
        if bk and bk not in seen and bk != "generic":
            ordered.append(bk)
            seen.add(bk)

    for key in ("query_date_zero", "bad_write_ctx", "nonconst_path", "nonconst_hash", "real_write_zero"):
        if metric_counts.get(key, 0) > 0 and key not in seen:
            ordered.append(key)
            seen.add(key)

    hints = {
        "failure_buckets": ordered[:5],
        "hard_avoid": [],
        "prefer": [],
        "repair_focus": [],
    }

    text_map = {
        "real_write_zero": (
            "Do not generate update expressions with zero real writes.",
            "Make update perform at least one real counter write.",
        ),
        "query_date_zero": (
            "Do not generate query expressions with zero query_date reads.",
            "Make query perform stable real reads via query_date-related access.",
        ),
        "nonconst_hash": (
            "Avoid dynamic hash ids; prefer small stable constant hash ids.",
            "Stabilize hash ids into small constants when possible.",
        ),
        "nonconst_path": (
            "Avoid dynamic path ids; prefer small stable constant path ids shared by update and query.",
            "Reduce non-constant path ids and keep update/query access shape aligned.",
        ),
        "bad_write_ctx": (
            "Do not place real writes inside comparisons, boolean gates, arithmetic-only wrappers, or nested min/max guards.",
            "Move writes into a clean update context.",
        ),
    }

    for bk in hints["failure_buckets"]:
        if bk not in text_map:
            continue
        hard_avoid_text, repair_focus_text = text_map[bk]
        if hard_avoid_text not in hints["hard_avoid"]:
            hints["hard_avoid"].append(hard_avoid_text)
        if repair_focus_text not in hints["repair_focus"]:
            hints["repair_focus"].append(repair_focus_text)

    if hints["failure_buckets"]:
        hints["prefer"].append("Prioritize repairing the currently dominant island-level failure modes.")
        hints["prefer"].append("Prefer stable sketch structure that directly fixes recent bad patterns.")

    return hints

def _serialize_team_spec(team, rationale="", source="offline_json"):
    return {
        "init_dex": str(team["init_dex"]),
        "update": str(team["update"]),
        "query": str(team["query"]),
        "rationale": str(rationale or ""),
        "source": str(source or ""),
    }


def _deserialize_team_spec(spec, pset_map):
    return {
        "init_dex": INDIVIDUAL_CLS.from_string(str(spec["init_dex"]), pset_map["init_dex"]),
        "update": INDIVIDUAL_CLS.from_string(str(spec["update"]), pset_map["update"]),
        "query": INDIVIDUAL_CLS.from_string(str(spec["query"]), pset_map["query"]),
    }


def _compose_team_with_targets(base_team, cand_team, target_funcs):
    out = {}
    for which in ("init_dex", "update", "query"):
        out[which] = copy.deepcopy(cand_team[which] if which in target_funcs else base_team[which])
    return out




def _safe_triplet_fec_key(evaluator, init_dex_tree, update_tree, query_tree):
    try:
        return evaluator._canonical_triplet_key(init_dex_tree, update_tree, query_tree)
    except Exception:
        return "RAW::" + repr((str(init_dex_tree), str(update_tree), str(query_tree)))


def _state_fec_summary(state):
    keys = list(state.get("fec_keys", []))
    if not keys:
        return 0, 0
    cnt = Counter(keys)
    return len(cnt), (max(cnt.values()) if cnt else 0)


def _rebuild_state_fec_index(state, evaluator):
    pop_size = len(state.get("fits", []))
    pops = state["pops"]
    fec_keys = []
    cnt = Counter()
    for i in range(pop_size):
        key = _safe_triplet_fec_key(
            evaluator,
            pops["init_dex"][i],
            pops["update"][i],
            pops["query"][i],
        )
        fec_keys.append(key)
        cnt[key] += 1
    state["fec_keys"] = fec_keys
    state["fec_counts"] = dict(cnt)
    return state


def _replacement_score(fits, birth, fec_keys, step_counter, idx):
    cnt = Counter(fec_keys)
    age = max(0, int(step_counter) - int(birth[idx]))
    fit, err = fits[idx]
    crowd = cnt.get(fec_keys[idx], 1)

    crowd_pressure = max(0, crowd - 2)

    return (crowd_pressure, age, -float(fit), float(err))


def _rank_replacement_targets(state):
    pop_size = len(state.get("fits", []))
    if pop_size <= 0:
        return []
    fec_keys = list(state.get("fec_keys", []))
    if len(fec_keys) != pop_size:
        fec_keys = [None] * pop_size
    step_counter = int(state.get("step_counter", pop_size))
    return sorted(
        range(pop_size),
        key=lambda j: _replacement_score(state["fits"], state["birth"], fec_keys, step_counter, j),
        reverse=True,
    )


def _choose_replacement_idx_from_lists(fits, birth, fec_keys, step_counter):
    if not fits:
        return -1
    return max(
        range(len(fits)),
        key=lambda j: _replacement_score(fits, birth, fec_keys, step_counter, j),
    )


def _select_diverse_migration_indices(state, k):
    pop_size = len(state.get("fits", []))
    if pop_size <= 0 or int(k) <= 0:
        return []

    fec_keys = list(state.get("fec_keys", []))
    if len(fec_keys) != pop_size:
        fec_keys = [f"__fallback__:{j}" for j in range(pop_size)]

    elite_k = max(1, int(k) // 2)
    rand_k = max(0, int(k) - elite_k)

    order = sorted(
        range(pop_size),
        key=lambda j: (float(state["fits"][j][0]), -float(state["fits"][j][1])),
        reverse=True,
    )

    chosen = []
    seen = set()

    for j in order:
        key = fec_keys[j]
        if key in seen:
            continue
        chosen.append(j)
        seen.add(key)
        if len(chosen) >= elite_k:
            break

    rest = [j for j in range(pop_size) if j not in chosen]
    random.shuffle(rest)

    for j in rest:
        key = fec_keys[j]
        if key in seen and len(rest) > max(2, rand_k * 2):
            continue
        chosen.append(j)
        seen.add(key)
        if len(chosen) >= int(k):
            break

    if len(chosen) < int(k):
        for j in order:
            if j not in chosen:
                chosen.append(j)
                if len(chosen) >= int(k):
                    break

    return chosen[:int(k)]

def replace_individual_in_state(state, idx, team, fit_err_casevec, source_meta=None, fec_key=None):
    fit, err, case_vec = fit_err_casevec
    pop_size = len(state["fits"])
    if idx < 0 or idx >= pop_size:
        return state

    fec_keys = state.get("fec_keys")
    if not (isinstance(fec_keys, list) and len(fec_keys) == pop_size):
        fec_keys = [None] * pop_size

    counts = Counter(state.get("fec_counts", {}))
    old_key = fec_keys[idx]

    state["pops"]["init_dex"][idx] = copy.deepcopy(team["init_dex"])
    state["pops"]["update"][idx] = copy.deepcopy(team["update"])
    state["pops"]["query"][idx] = copy.deepcopy(team["query"])
    state["fits"][idx] = (float(fit), float(err))
    state["case_vecs"][idx] = tuple(float(x) for x in case_vec)
    state["birth"][idx] = int(state["step_counter"])
    state["step_counter"] = int(state["step_counter"]) + 1

    if fec_key is None:
        fec_key = old_key

    if old_key in counts:
        counts[old_key] -= 1
        if counts[old_key] <= 0:
            counts.pop(old_key, None)

    fec_keys[idx] = fec_key
    counts[fec_key] += 1
    state["fec_keys"] = fec_keys
    state["fec_counts"] = dict(counts)

    for k, v in list(state.items()):
        if k in {"pops", "fits", "case_vecs", "birth", "step_counter", "fec_keys", "fec_counts"}:
            continue
        if isinstance(v, list) and len(v) == pop_size:
            if k == "ages":
                v[idx] = 0
            elif source_meta is not None and ("meta" in k.lower()):
                v[idx] = copy.deepcopy(source_meta)

    if source_meta is not None:
        if "llm_meta" not in state or not (isinstance(state["llm_meta"], list) and len(state["llm_meta"]) == pop_size):
            state["llm_meta"] = [None] * pop_size
        state["llm_meta"][idx] = copy.deepcopy(source_meta)
    return state

def _collect_existing_canonical_keys_from_states(island_states, evaluator):
    keys = set()
    for st in island_states:
        n = len(st.get("fits", []))
        for i in range(n):
            try:
                keys.add(evaluator._canonical_triplet_key(
                    st["pops"]["init_dex"][i], st["pops"]["update"][i], st["pops"]["query"][i]
                ))
            except Exception:
                continue
    return keys


def _apply_llm_seed_specs_to_state(state, cfg, gp_ctx, evaluator):
    specs = list(cfg.get("llm_seed_specs", []))
    if not specs:
        return state, 0
    if not bool(cfg.get("llm_enable", False)):
        return state, 0
    if _normalize_llm_mode(cfg.get("llm_mode", "none")) not in {"seeds", "both"}:
        return state, 0
    pop_size = len(state.get("fits", []))
    if pop_size <= 0:
        return state, 0
    seed_ratio = max(0.0, float(cfg.get("llm_seed_ratio", 0.0)))
    seed_max = max(0, int(cfg.get("llm_seed_max", 0)))
    inject_n = int(round(pop_size * seed_ratio))
    if seed_max > 0:
        inject_n = min(inject_n, seed_max)
    inject_n = min(pop_size, max(0, inject_n))
    if inject_n <= 0:
        return state, 0
    if not (isinstance(state.get("fec_keys"), list) and len(state.get("fec_keys", [])) == pop_size):
        state = _rebuild_state_fec_index(state, evaluator)
    target_funcs = _parse_llm_target_funcs(cfg.get("llm_target_funcs", "update,query"))
    repl_order = _rank_replacement_targets(state)
    idxs = repl_order[:inject_n] if repl_order else random.sample(range(pop_size), k=inject_n)
    inserted = 0
    for rank, idx in enumerate(idxs):
        spec = specs[rank % len(specs)]
        try:
            cand_team = _deserialize_team_spec(spec, gp_ctx["pset_map"])
            base_team = {
                "init_dex": state["pops"]["init_dex"][idx],
                "update": state["pops"]["update"][idx],
                "query": state["pops"]["query"][idx],
            }
            team = _compose_team_with_targets(base_team, cand_team, target_funcs)
            fit, err, case_vec = evaluator.evaluate_individual(
                team["init_dex"], team["update"], team["query"], return_case_vec=True
            )
            fec_key = _safe_triplet_fec_key(
                evaluator,
                team["init_dex"],
                team["update"],
                team["query"],
            )
            replace_individual_in_state(
                state,
                idx,
                team,
                (fit, err, case_vec),
                source_meta={"phase": "seed", "source": spec.get("source", "offline_json"), "rationale": spec.get("rationale", "")},
                fec_key=fec_key,
            )
            print(f"[LLM_SEED_APPLY] idx={idx} fit={float(fit):.6f} err={float(err):.6f}", flush=True)
            inserted += 1
        except Exception:
            continue
    return state, inserted


def _inject_llm_immigrants_with_engine(state, cfg, gp_ctx, llm_engine, candidate_specs, success_budget):
    if success_budget <= 0 or (not candidate_specs):
        return state, 0, []
    evaluator = _make_evaluator_from_cfg(cfg)
    if cfg.get("hard_case_replay", False):
        evaluator.import_hard_case_state(state.get("hard_case_state"))
    target_funcs = _parse_llm_target_funcs(cfg.get("llm_target_funcs", "update,query"))
    pop_size = len(state.get("fits", []))
    if pop_size <= 0:
        return state, 0, []

    if not (isinstance(state.get("fec_keys"), list) and len(state.get("fec_keys", [])) == pop_size):
        state = _rebuild_state_fec_index(state, evaluator)

    repl_idx = _rank_replacement_targets(state)
    inserted = 0
    accepted_specs = []
    ptr = 0
    for spec in candidate_specs:
        if inserted >= int(success_budget):
            break
        if ptr >= len(repl_idx):
            break
        idx = repl_idx[ptr]
        ptr += 1
        try:
            cand_team = _deserialize_team_spec(spec, gp_ctx["pset_map"])
            base_team = {
                "init_dex": state["pops"]["init_dex"][idx],
                "update": state["pops"]["update"][idx],
                "query": state["pops"]["query"][idx],
            }
            team = _compose_team_with_targets(base_team, cand_team, target_funcs)
            chk = llm_engine.validate_team_candidate(team, evaluator, existing_canon=None)
            if not chk.get("ok", False):
                continue
            replace_individual_in_state(
                state,
                idx,
                chk["team"],
                (chk["fit"], chk["err"], chk["case_vec"]),
                source_meta={"phase": "stagnation", "source": spec.get("source", "offline_json"), "rationale": spec.get("rationale", "")},
                fec_key=chk["key"],
            )
            print(f"[LLM_IMMIGRANT_APPLY] idx={idx} fit={float(chk['fit']):.6f} err={float(chk['err']):.6f}",
                  flush=True)
            inserted += 1
            accepted_specs.append(spec)
        except Exception:
            continue
    return state, inserted, accepted_specs


def _make_evaluator_from_cfg(cfg):
    use_stage1_multi_proxy = (
        str(cfg.get("dataset_mode", "real")) == "proxy"
        and bool(cfg.get("stage1_multi_proxy", False))
        and len(list(cfg.get("stage1_proxy_modes", []))) > 1
    )

    if use_stage1_multi_proxy:
        ev = Stage1MultiProxyEvaluator(cfg)
    else:
        ev = CMSketchEvaluator(
            dataset_root=cfg["dataset_root"],
            pkts=cfg["pkts"],
            max_files=cfg["files"],
            start=cfg["start"],
            shuffle=cfg["shuffle"],
            seed=int(cfg["dataset_seed"]) & 0xFFFFFFFF,
            dataset_mode=cfg["dataset_mode"],
            proxy_mode=cfg["proxy_mode"],
            proxy_pool_mul=cfg["proxy_pool_mul"],
            proxy_min_u=cfg["proxy_min_u"],
            hard_case_enabled=cfg.get("hard_case_replay", False),
            hard_case_stage_topk=cfg.get("hard_case_stage_topk", 24),
            hard_case_absent_topk=cfg.get("hard_case_absent_topk", 12),
            hard_case_scan_mul=cfg.get("hard_case_scan_mul", 3),
            hard_case_decay=cfg.get("hard_case_decay", 0.85),
            hard_case_weight=cfg.get("hard_case_weight", 0.50),
            fixed_stream_path=str(cfg.get("fixed_stream_path", "") or ""),
        )

    e0_value = cfg.get("e0_value", None)
    if e0_value is not None:
        try:
            ev.E0 = max(1.0, float(e0_value))
        except Exception:
            ev.E0 = 1.0

    return ev

def _outer_eval_fail_result(evaluator, phase: str, exc: Exception, island_idx=None):
    err = 2_000_000_000.0
    fit = float(evaluator._norm_fitness(err))
    case_vec = (float(evaluator.lexicase_default_bad),) * int(evaluator.lexicase_total_cases)

    if island_idx is None:
        print(f"[OUTER_EVAL_FAIL] phase={phase} err={repr(exc)}", flush=True)
    else:
        print(f"[OUTER_EVAL_FAIL] phase={phase} island={int(island_idx)} err={repr(exc)}", flush=True)

    return fit, err, case_vec

def _reevaluate_population_with_evaluator(state, evaluator):
    pops = state["pops"]
    pop_size = len(state["fits"])
    fits = []
    case_vecs = []
    for i in range(pop_size):
        try:
            fit, err, case_vec = evaluator.evaluate_individual(
                pops["init_dex"][i],
                pops["update"][i],
                pops["query"][i],
                return_case_vec=True,
            )
        except Exception as e:
            fit, err, case_vec = _outer_eval_fail_result(
                evaluator,
                phase=f"reeval_pop_idx_{i}",
                exc=e,
                island_idx=state.get("island_idx", None),
            )
        fits.append((float(fit), float(err)))
        case_vecs.append(tuple(float(x) for x in case_vec))
    state["fits"] = fits
    state["case_vecs"] = case_vecs
    state = _rebuild_state_fec_index(state, evaluator)
    return state

def _refresh_island_best(state):
    fits = state["fits"]
    pops = state["pops"]
    pop_size = len(fits)
    i = max(range(pop_size), key=lambda j: fits[j][0])
    fit, err = fits[i]
    team = {
        "init_dex": copy.deepcopy(pops["init_dex"][i]),
        "update": copy.deepcopy(pops["update"][i]),
        "query": copy.deepcopy(pops["query"][i]),
    }
    return i, float(fit), float(err), team

def _select_parent_idx_lexicase(fits, case_vecs, tournament_k: int, sample_cases: int = 4, epsilon: float = 1e-9):
    pop_size = len(fits)
    k = min(max(1, int(tournament_k)), pop_size)
    cand = random.sample(range(pop_size), k=k)

    usable = [j for j in cand if j < len(case_vecs) and case_vecs[j] is not None]
    if len(usable) < 2:
        return max(cand, key=lambda j: fits[j][0])

    case_dim = len(case_vecs[usable[0]])
    chosen_cases = random.sample(range(case_dim), k=min(max(1, int(sample_cases)), case_dim))

    survivors = usable[:]
    for c in chosen_cases:
        best_val = min(case_vecs[j][c] for j in survivors)
        survivors = [j for j in survivors if case_vecs[j][c] <= best_val + float(epsilon)]
        if len(survivors) <= 1:
            break

    if not survivors:
        survivors = usable

    return max(survivors, key=lambda j: fits[j][0])

def _init_island_state(cfg, island_idx: int):
    gp_ctx = _build_gp_context(max_size=cfg["max_size"])
    gp_ctx = _populate_llm_seed_bank_from_cfg(gp_ctx, cfg)

    toolboxes = gp_ctx["toolboxes"]
    pop_size = int(cfg["population_size"])

    if cfg.get("rng_state") is not None:
        random.setstate(cfg["rng_state"])
    else:
        init_seed = (int(cfg["base_seed"]) + 1000003 * int(island_idx)) & 0xFFFFFFFF
        set_seed(init_seed)

    evaluator = _make_evaluator_from_cfg(cfg)
    init_p_skeleton = float(cfg.get("init_p_skeleton", 0.70))
    init_p_seed = float(cfg.get("init_p_seed", 0.20))
    llm_seed_prob = 0.0

    pops = {
        'init_dex': [
            _init_individual_from_ctx(
                gp_ctx,
                'init_dex',
                p_skeleton=init_p_skeleton,
                p_seed=init_p_seed,
                p_llm_seed=llm_seed_prob,
            )
            for _ in range(pop_size)
        ],
        'update': [
            _init_individual_from_ctx(
                gp_ctx,
                'update',
                p_skeleton=init_p_skeleton,
                p_seed=init_p_seed,
                p_llm_seed=llm_seed_prob,
            )
            for _ in range(pop_size)
        ],
        'query': [
            _init_individual_from_ctx(
                gp_ctx,
                'query',
                p_skeleton=init_p_skeleton,
                p_seed=init_p_seed,
                p_llm_seed=llm_seed_prob,
            )
            for _ in range(pop_size)
        ],
    }
    triples = [
        (pops['init_dex'][i], pops['update'][i], pops['query'][i])
        for i in range(pop_size)
    ]
    fits = []
    case_vecs = []
    for init_t, upd_t, qry_t in triples:
        try:
            fit, err, case_vec = evaluator.evaluate_individual(
                init_t, upd_t, qry_t, return_case_vec=True
            )
        except Exception as e:
            fit, err, case_vec = _outer_eval_fail_result(
                evaluator,
                phase="init_population",
                exc=e,
                island_idx=island_idx,
            )
        fits.append((float(fit), float(err)))
        case_vecs.append(tuple(float(x) for x in case_vec))

    init_errs = [float(err) for _, err in fits]
    finite_init_errs = [x for x in init_errs if math.isfinite(x)]
    init_err_min = min(finite_init_errs) if finite_init_errs else float("inf")
    init_err_med = statistics.median(finite_init_errs) if finite_init_errs else float("inf")
    init_err_max = max(finite_init_errs) if finite_init_errs else float("inf")

    dbg = evaluator._debug_snapshot()
    eval_calls = max(1, int(dbg["eval_calls"]))
    top_hard_illegal = sorted(
        dbg["hard_illegal_reasons"].items(),
        key=lambda kv: (-kv[1], kv[0])
    )[:3]

    print(
        f"[DIAG_INIT] island={island_idx} "
        f"E0={evaluator.E0} "
        f"err_min={init_err_min:.2f} err_med={init_err_med:.2f} err_max={init_err_max:.2f} "
        f"eval_calls={dbg['eval_calls']} cache_hits={dbg['eval_cache_hits']} fec_hits={dbg['fec_cache_hits']} "
        f"hard_illegal={dbg['hard_illegal']} real_write_zero={dbg['real_write_zero']} "
        f"query_date_zero={dbg['query_date_zero']} penalty_dom={dbg['penalty_dominates']} "
        f"penalty_avg={dbg['penalty_sum'] / eval_calls:.2f} "
        f"query_avg={dbg['query_error_sum'] / eval_calls:.2f} "
        f"top_hard_illegal={top_hard_illegal}",
        flush=True
    )

    state = {
        'island_idx': int(island_idx),
        'pops': pops,
        'birth': list(range(pop_size)),
        'fits': fits,
        'case_vecs': case_vecs,
        'step_counter': int(pop_size),
        'rng_state': random.getstate(),
                "hard_case_state": evaluator.export_hard_case_state() if cfg.get("hard_case_replay", False) else {"version": 0, **evaluator._empty_hard_case_state()},
        "scored_hard_case_version": 0,
        "recent_diag_history": [copy.deepcopy(dbg)],
    }
    state = _rebuild_state_fec_index(state, evaluator)
    _, best_fit, best_err, _ = _refresh_island_best(state)
    state['best_fitness'] = float(best_fit)
    state['best_error'] = float(best_err)

    # 初始化后按比例混入 llm_seed_bank（team 级），并同步刷新 pop/fits/case_vecs/birth 等字段
    try:
        state, seed_inserted = _apply_llm_seed_specs_to_state(state, cfg, gp_ctx, evaluator)
        if int(seed_inserted) > 0:
            _, best_fit2, best_err2, _ = _refresh_island_best(state)
            state['best_fitness'] = float(best_fit2)
            state['best_error'] = float(best_err2)
            print(f"[LLM_SEED_INJECT] island={island_idx} inserted={int(seed_inserted)}", flush=True)
    except Exception as e:
        print(f"[LLM_SEED_INJECT_SKIP] island={island_idx} reason={e}", flush=True)

    if 'scored_hard_case_version' not in state:
        state['scored_hard_case_version'] = int(state.get('hard_case_state', {}).get('version', 0))
    return state


def _evolve_island_chunk(job):
    state, local_gens, cfg = job
    island_idx = int(state["island_idx"]) if state is not None else int(cfg.get("island_idx", 0))
    gp_ctx = _build_gp_context(max_size=cfg["max_size"])
    toolboxes = gp_ctx["toolboxes"]
    evaluator = _make_evaluator_from_cfg(cfg)

    if state is None:
        state = _init_island_state(cfg, island_idx)
    elif state.get('rng_state') is not None:
        random.setstate(state['rng_state'])
    else:
        init_seed = (int(cfg["base_seed"]) + 1000003 * int(island_idx)) & 0xFFFFFFFF
        set_seed(init_seed)

    if cfg.get("hard_case_replay", False):
        evaluator.import_hard_case_state(state.get("hard_case_state"))
        cur_version = int(state.get("hard_case_state", {}).get("version", 0))
        scored_version = int(state.get("scored_hard_case_version", -1))
        if cur_version != scored_version:
            state = _reevaluate_population_with_evaluator(state, evaluator)
            state["scored_hard_case_version"] = cur_version

    pops = state['pops']
    birth = state['birth']
    fits = state['fits']
    case_vecs = state.get('case_vecs', [None] * len(fits))
    step_counter = int(state.get('step_counter', len(birth)))
    if not (isinstance(state.get("fec_keys"), list) and len(state.get("fec_keys", [])) == len(fits)):
        state = _rebuild_state_fec_index(state, evaluator)
    fec_keys = list(state.get("fec_keys", []))

    pop_size = int(cfg['population_size'])
    mut_ops = {
        'init_dex': ["mut_uniform", "mut_node_replace", "mut_insert", "mut_shrink", "mut_ephemeral"],
        'update': ["mut_uniform", "mut_node_replace", "mut_insert", "mut_shrink", "mut_ephemeral"],
        'query': ["mut_uniform", "mut_node_replace", "mut_insert", "mut_shrink", "mut_ephemeral"],
    }

    history = []
    for _gen in range(int(local_gens)):
        _, prev_best_fit, prev_best_error, _ = _refresh_island_best({
            'pops': pops,
            'fits': fits,
        })


        for _ in range(pop_size):
            if str(cfg.get("parent_selector", "lexicase")) == "tournament":
                k = min(int(cfg['tournament_size']), pop_size)
                cand = random.sample(range(pop_size), k=k)
                parent_idx = max(cand, key=lambda j: fits[j][0])
            else:
                parent_idx = _select_parent_idx_lexicase(
                    fits,
                    case_vecs,
                    tournament_k=int(cfg['tournament_size']),
                    sample_cases=int(cfg.get('lexicase_cases', 4)),
                    epsilon=float(cfg.get('lexicase_epsilon', 1e-9)),
                )

            child = {
                'init_dex': toolboxes['init_dex'].clone(pops['init_dex'][parent_idx]),
                'update': toolboxes['update'].clone(pops['update'][parent_idx]),
                'query': toolboxes['query'].clone(pops['query'][parent_idx]),
            }

            _reset_prob = float(cfg['reset_prob'])
            _mutation_prob = float(cfg['mutation_prob'])
            _reset_whole_prob = float(cfg['reset_whole_prob'])

            if random.random() < _reset_whole_prob:
                which = random.choice(["init_dex", "update", "query"])
                child[which] = toolboxes[which].individual()
            else:
                if random.random() < _reset_prob:
                    which = random.choice(["init_dex", "update", "query"])
                    child[which] = toolboxes[which].individual()


                if random.random() < _mutation_prob:
                    which = random.choice(["init_dex", "update", "query"])
                    op = random.choice(mut_ops[which])
                    child[which] = _apply_mutation_with_ctx(toolboxes, op, which, child[which])

            try:
                fit, err, case_vec = evaluator.evaluate_individual(
                    child['init_dex'],
                    child['update'],
                    child['query'],
                    return_case_vec=True,
                )
            except Exception as e:
                fit, err, case_vec = _outer_eval_fail_result(
                    evaluator,
                    phase="child_eval",
                    exc=e,
                    island_idx=island_idx,
                )

            child_key = _safe_triplet_fec_key(
                evaluator,
                child['init_dex'],
                child['update'],
                child['query'],
            )

            for which in ("init_dex", "update", "query"):
                pops[which].append(child[which])
            fits.append((float(fit), float(err)))
            case_vecs.append(tuple(float(x) for x in case_vec))
            birth.append(step_counter)
            fec_keys.append(child_key)
            step_counter += 1

            drop_idx = _choose_replacement_idx_from_lists(
                fits,
                birth,
                fec_keys,
                step_counter,
            )

            for which in ("init_dex", "update", "query"):
                pops[which].pop(drop_idx)
            fits.pop(drop_idx)
            birth.pop(drop_idx)
            case_vecs.pop(drop_idx)
            fec_keys.pop(drop_idx)



        _, cur_best_fit, cur_best_err, _ = _refresh_island_best({
            'pops': pops,
            'fits': fits,
        })
        avg_fit = sum(x[0] for x in fits) / max(1, len(fits))
        avg_err = sum(x[1] for x in fits) / max(1, len(fits))
        history.append({
            'avg_fit': float(avg_fit),
            'avg_err': float(avg_err),
            'best_fit': float(cur_best_fit),
            'best_err': float(cur_best_err),
        })




    state['pops'] = pops
    state['birth'] = birth
    state['fits'] = fits
    state['case_vecs'] = case_vecs
    state['step_counter'] = int(step_counter)
    state['fec_keys'] = fec_keys
    state['fec_counts'] = dict(Counter(fec_keys))
    state['rng_state'] = random.getstate()
    _, best_fit, best_err, best_team = _refresh_island_best(state)

    if cfg.get("hard_case_replay", False):
        try:
            mined_state = evaluator.mine_hard_cases(
                best_team['init_dex'],
                best_team['update'],
                best_team['query'],
            )
            evaluator._merge_hard_case_state(mined_state)
            state['hard_case_state'] = evaluator.export_hard_case_state()
        except Exception:
            state['hard_case_state'] = state.get('hard_case_state',
                                                 {"version": 0, **evaluator._empty_hard_case_state()})
    else:
        state['hard_case_state'] = {"version": 0, **evaluator._empty_hard_case_state()}

    state['best_fitness'] = float(best_fit)
    state['best_error'] = float(best_err)

    dbg = evaluator._debug_snapshot()
    recent_diag_history = list(state.get("recent_diag_history", []))
    recent_diag_history.append(copy.deepcopy(dbg))
    keep_recent = max(2, int(cfg.get("llm_recent_diag_keep", 4)))
    state["recent_diag_history"] = recent_diag_history[-keep_recent:]
    eval_calls = max(1, int(dbg["eval_calls"]))
    fec_unique, fec_max_cluster = _state_fec_summary(state)
    top_hard_illegal = sorted(
        dbg["hard_illegal_reasons"].items(),
        key=lambda kv: (-kv[1], kv[0])
    )[:3]

    print(
        f"[DIAG_CHUNK] island={island_idx} local_gens={int(local_gens)} "
        f"E0={evaluator.E0} "
        f"eval_calls={dbg['eval_calls']} cache_hits={dbg['eval_cache_hits']} fec_hits={dbg['fec_cache_hits']} "
        f"fec_unique={fec_unique} fec_max_cluster={fec_max_cluster} "
        f"hard_illegal={dbg['hard_illegal']} real_write_zero={dbg['real_write_zero']} "
        f"query_date_zero={dbg['query_date_zero']} penalty_dom={dbg['penalty_dominates']} "
        f"early_cut={dbg['early_return_cut']} "
        f"nonconst_hash_total={dbg['nonconst_hash_idx_total']} "
        f"nonconst_path_total={dbg['nonconst_path_idx_total']} "
        f"bad_write_ctx_total={dbg['bad_write_context_total']} "
        f"penalty_avg={dbg['penalty_sum'] / eval_calls:.2f} "
        f"query_avg={dbg['query_error_sum'] / eval_calls:.2f} "
        f"total_avg={dbg['total_error_sum'] / eval_calls:.2f} "
        f"top_hard_illegal={top_hard_illegal}",
        flush=True
    )

    return {
        'state': state,
        'history': history,
        'best_fitness': float(best_fit),
        'best_error': float(best_err),
        'best_team': best_team,
    }


def _migrate_island_states(island_states, mig_k: int):
    islands = len(island_states)
    if islands <= 1:
        return island_states

    k = max(1, int(mig_k))

    for isl in range(islands):
        src = island_states[isl]
        dst = island_states[(isl + 1) % islands]

        pop_size = len(src['fits'])
        if pop_size <= 0:
            continue

        mig_idx = _select_diverse_migration_indices(src, k)

        src_fec_keys = list(src.get("fec_keys", []))
        if len(src_fec_keys) != pop_size:
            src_fec_keys = [f"__fallback__:{isl}:{j}" for j in range(pop_size)]

        src_case_vecs = src.get("case_vecs", [()] * pop_size)

        migrants = []
        for j in mig_idx:
            migrants.append({
                "team": {
                    "init_dex": copy.deepcopy(src['pops']['init_dex'][j]),
                    "update": copy.deepcopy(src['pops']['update'][j]),
                    "query": copy.deepcopy(src['pops']['query'][j]),
                },
                "fit": float(src['fits'][j][0]),
                "err": float(src['fits'][j][1]),
                "case_vec": tuple(float(x) for x in src_case_vecs[j]),
                "fec_key": src_fec_keys[j],
            })

        for mig in migrants:
            dst_counts = Counter(dst.get("fec_keys", []))
            if dst_counts.get(mig["fec_key"], 0) >= 2:
                continue

            repl_order = _rank_replacement_targets(dst)
            if not repl_order:
                break
            rep_idx = repl_order[0]

            replace_individual_in_state(
                dst,
                rep_idx,
                mig["team"],
                (mig["fit"], mig["err"], mig["case_vec"]),
                source_meta={
                    "phase": "migration",
                    "source_island": int(isl),
                    "fec_key": mig["fec_key"],
                },
                fec_key=mig["fec_key"],
            )

    return island_states

def evolve_cmsketch(
        population_size=300,
        generations=20,
        seed=None,
        dataset_root: str = "/data/8T/xgr/traces/univ2_trace",
        pkts: int = 30000,
        max_files: int = 1,
        start: int = 0,
        shuffle: bool = False,
        dataset_mode: str = "real",
        proxy_mode: str = "proxy_balanced",
        stage1_proxy_modes=None,
        stage1_multi_proxy: bool = True,
        proxy_pool_mul: int = 8,
        proxy_min_u: int = 2500,
        islands: int = 4,
        tournament_size: int = 5,
        reset_prob: float = 0.10,
        parent_selector: str = "lexicase",
        lexicase_cases: int = 4,
        lexicase_epsilon: float = 1e-9,
        hard_case_replay: bool = False,
        hard_case_stage_topk: int = 24,
        hard_case_absent_topk: int = 12,
        hard_case_scan_mul: int = 3,
        hard_case_decay: float = 0.85,
        hard_case_weight: float = 0.50,
        llm_enable: bool = False,
        llm_mode: str = "none",
        llm_provider: str = "none",
        llm_model: str = "",
        llm_base_url: str = "",
        llm_api_key_env: str = "",
        llm_timeout: float = 30.0,
        llm_seed_ratio: float = 0.0,
        llm_seed_max: int = 0,
        llm_stagnation_patience: int = 2,
        llm_stagnation_num_candidates: int = 6,
        llm_stagnation_max_inject: int = 2,
        llm_offline_candidates_path: str = "",
        llm_log_path: str = "",
        llm_target_funcs: str = "update,query",
        llm_use_case_vec: bool = False,
        llm_use_hard_cases: bool = False,
        llm_ref_init_pset_path: str = "",
        llm_ref_update_pset_path: str = "",
        llm_ref_query_pset_path: str = "",
        init_p_skeleton: float = 0.70,
        init_p_seed: float = 0.20,
        reset_whole_prob: float = 0.02,
        mutation_prob: float = 0.90,
        mig_period: int = 8,
        mig_k: int = 3,
        max_size: int = 80,
        dataset_seed=None,
        fixed_stream_path: str = "",
):
    """推荐版：一岛一进程，周期性 migration。"""

    if seed is None:
        seed = time.time_ns() % (2 ** 32)
    set_seed(seed)
    if dataset_seed is None:
        dataset_seed = seed
    dataset_seed = int(dataset_seed) & 0xFFFFFFFF
    print(f"[SEED] {seed}")

    islands = max(1, int(islands))
    population_size = int(population_size)


    try:
        baseline_proxy_mode = str(proxy_mode)
        if stage1_proxy_modes:
            try:
                baseline_proxy_mode = str(list(stage1_proxy_modes)[0])
            except Exception:
                baseline_proxy_mode = str(proxy_mode)

        baseline_eval = CMSketchEvaluator(
            dataset_root=dataset_root,
            pkts=pkts,
            max_files=max_files,
            start=start,
            shuffle=shuffle,
            seed=int(dataset_seed) & 0xFFFFFFFF,
            dataset_mode=dataset_mode,
            proxy_mode=baseline_proxy_mode,
            proxy_pool_mul=proxy_pool_mul,
            proxy_min_u=proxy_min_u,
            fixed_stream_path=str(fixed_stream_path or ""),
        )

        def _calc_standard_cms_error(test_data, expected_freq, rows=3, cols=10240):
            local_hash_functions = [hashlib.md5, hashlib.sha1, hashlib.sha256]
            matrix = [[0] * cols for _ in range(rows)]
            for item in test_data:
                b = str(item).encode('utf-8', errors='ignore')
                for i, hf in enumerate(local_hash_functions):
                    y = int(hf(b).hexdigest(), 16) % cols
                    matrix[i][y] += 1
            total = 0.0
            n_items = 0
            for item, exp in expected_freq.items():
                if float(exp) <= 0.0:
                    continue
                b = str(item).encode('utf-8', errors='ignore')
                ests = []
                for i, hf in enumerate(local_hash_functions):
                    y = int(hf(b).hexdigest(), 16) % cols
                    ests.append(matrix[i][y])
                total += abs(min(ests) - float(exp))
                n_items += 1
            return float(total / max(1, n_items))

        E_base = _calc_standard_cms_error(baseline_eval.test_data, baseline_eval.expected_freq)
        print(f"[Baseline CMS] AAE_base={E_base:.6f}")
    except Exception as _e:
        print(f"[Baseline CMS] 计算失败: {_e}")

    cfg = {
        'population_size': population_size,
        'dataset_root': dataset_root,
        'pkts': int(pkts),
        'files': int(max_files),
        'start': int(start),
        'shuffle': bool(shuffle),
        'dataset_mode': str(dataset_mode),
        'proxy_mode': str(proxy_mode),
        'proxy_pool_mul': int(proxy_pool_mul),
        'proxy_min_u': int(proxy_min_u),
        'base_seed': int(seed) & 0xFFFFFFFF,
        'dataset_seed': int(dataset_seed) & 0xFFFFFFFF,
        'fixed_stream_path': str(fixed_stream_path or ""),
        'stage1_proxy_modes': list(stage1_proxy_modes or [str(proxy_mode)]),
        'stage1_multi_proxy': bool(stage1_multi_proxy),
        'tournament_size': int(tournament_size),
        'parent_selector': str(parent_selector),
        'lexicase_cases': int(lexicase_cases),
        'lexicase_epsilon': float(lexicase_epsilon),
        'hard_case_replay': bool(hard_case_replay),
        'hard_case_stage_topk': int(hard_case_stage_topk),
        'hard_case_absent_topk': int(hard_case_absent_topk),
        'hard_case_scan_mul': int(hard_case_scan_mul),
        'hard_case_decay': float(hard_case_decay),
        'hard_case_weight': float(hard_case_weight),
        'llm_enable': bool(llm_enable),
        'llm_mode': str(_normalize_llm_mode(llm_mode)),
        'llm_provider': str(llm_provider),
        'llm_model': str(llm_model),
        'llm_base_url': str(llm_base_url),
        'llm_api_key_env': str(llm_api_key_env),
        'llm_timeout': float(llm_timeout),
        'llm_seed_ratio': float(llm_seed_ratio),
        'llm_seed_max': int(llm_seed_max),
        'llm_stagnation_patience': int(llm_stagnation_patience),
        'llm_stagnation_num_candidates': int(llm_stagnation_num_candidates),
        'llm_stagnation_max_inject': int(llm_stagnation_max_inject),
        'llm_offline_candidates_path': str(llm_offline_candidates_path),
        'llm_log_path': str(llm_log_path),
        'llm_target_funcs': str(llm_target_funcs),
        'llm_use_case_vec': bool(llm_use_case_vec),
        'llm_use_hard_cases': bool(llm_use_hard_cases),
        'llm_ref_init_pset_path': str(llm_ref_init_pset_path),
        'llm_ref_update_pset_path': str(llm_ref_update_pset_path),
        'llm_ref_query_pset_path': str(llm_ref_query_pset_path),
        'init_p_skeleton': float(init_p_skeleton),
        'init_p_seed': float(init_p_seed),
        'reset_prob': float(reset_prob),
        'reset_whole_prob': float(reset_whole_prob),
        'mutation_prob': float(mutation_prob),
        'max_size': int(max_size),
    }
    cfg['llm_target_funcs'] = ",".join(sorted(_parse_llm_target_funcs(cfg.get("llm_target_funcs", "update,query"))))
    cfg['llm_seed_specs'] = []

    gp_ctx_main = _build_gp_context(max_size=cfg["max_size"])
    llm_logger = LLMRunLogger(cfg.get("llm_log_path", ""))
    llm_ref = PrimitiveSpecReference(gp_ctx_main["pset_map"], cfg, llm_logger)
    llm_engine = LLMProposalEngine(cfg, llm_ref, llm_logger)
    main_eval_for_llm = _make_evaluator_from_cfg(cfg)

    # seed 阶段候选在主进程预取与预校验，worker 只做注入替换，避免热路径联网
    try:
        if bool(cfg.get("llm_enable", False)) and str(cfg.get("llm_mode", "none")) in {"seeds", "both"}:
            base_seed_team = {
                "init_dex": _skeleton_individual_from_ctx(gp_ctx_main, "init_dex"),
                "update": _skeleton_individual_from_ctx(gp_ctx_main, "update"),
                "query": _skeleton_individual_from_ctx(gp_ctx_main, "query"),
            }
            seed_limit = max(1, int(cfg.get("llm_seed_max", 0) or 8))
            seed_candidates = llm_engine.prepare_phase_candidates(
                phase="seed",
                gp_ctx=gp_ctx_main,
                evaluator=main_eval_for_llm,
                base_team=base_seed_team,
                existing_canon=set(),
                limit=seed_limit,
            )
            for i, c in enumerate(seed_candidates):
                llm_logger.info(
                    "seed candidate accepted",
                    idx=int(i),
                    source=str(c.get("source", "offline_json")),
                    fitness=float(c.get("fit", 0.0)),
                    error=float(c.get("err", 0.0)),
                )
            cfg["llm_seed_specs"] = [
                _serialize_team_spec(c["team"], rationale=c.get("rationale", ""), source=c.get("source", "offline_json"))
                for c in seed_candidates
            ]
            llm_logger.info("seed candidate bank prepared", total=len(cfg["llm_seed_specs"]))
    except Exception as e:
        llm_logger.warn("seed candidate prepare failed", error=str(e))

    no_improve_chunks = 0

    best_fitness = -float('inf')
    best_error = float('inf')
    best_team = None

    ctx = None
    try:
        ctx = mp.get_context('fork')
    except Exception:
        ctx = None

    max_workers = max(1, islands)
    if ctx is not None:
        pool = cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    else:
        pool = cf.ProcessPoolExecutor(max_workers=max_workers)

    try:
        init_jobs = []
        for isl in range(islands):
            job_cfg = dict(cfg)
            job_cfg['island_idx'] = int(isl)
            init_jobs.append((None, 0, job_cfg))

        init_results = list(pool.map(_evolve_island_chunk, init_jobs))
        island_states = [r['state'] for r in init_results]

        init_errs = []
        total_init = 0
        fail_like = 0

        for st in island_states:
            for _, err in st["fits"]:
                total_init += 1
                try:
                    e = float(err)
                except Exception:
                    continue

                if not math.isfinite(e):
                    continue

                # 2e9 是你现在的兜底失败值，不要拿它参与 E0 标定
                if e >= 2_000_000_000.0:
                    fail_like += 1
                    continue

                init_errs.append(e)

        if init_errs:
            e0_value = max(1.0, float(statistics.median(init_errs)))
        else:
            e0_value = 1.0

        cfg["e0_value"] = float(e0_value)
        main_eval_for_llm.E0 = float(e0_value)

        print(
            f"[E0_INIT] e0_value={e0_value:.6f} "
            f"valid={len(init_errs)} total={total_init} fail_like={fail_like}",
            flush=True
        )

        # 用新的 E0 只重算 fitness，不重算 error
        for st in island_states:
            new_fits = []
            for _, err in st["fits"]:
                new_fits.append((float(main_eval_for_llm._norm_fitness(err)), float(err)))
            st["fits"] = new_fits

        best_fitness = -float('inf')
        best_error = float('inf')
        best_team = None

        for st in island_states:
            _, st_best_fit, st_best_err, st_best_team = _refresh_island_best(st)
            st["best_fitness"] = float(st_best_fit)
            st["best_error"] = float(st_best_err)

            if (st_best_fit > best_fitness) or (st_best_fit == best_fitness and st_best_err < best_error):
                best_fitness = float(st_best_fit)
                best_error = float(st_best_err)
                best_team = st_best_team

        print(f"初始最佳fitness: {best_fitness:.6f} (AAE={best_error:.6f})")

        gens_done = 0
        mig_period_eff = int(mig_period) if int(mig_period) > 0 else int(generations)

        while gens_done < int(generations):
            chunk_gens = min(mig_period_eff, int(generations) - gens_done)
            print(f"\n=== 代块 {gens_done + 1}-{gens_done + chunk_gens}/{generations} ===", flush=True)

            jobs = []
            for isl, st in enumerate(island_states):
                job_cfg = dict(cfg)
                job_cfg['island_idx'] = int(isl)
                jobs.append((st, int(chunk_gens), job_cfg))

            results = list(pool.map(_evolve_island_chunk, jobs))
            island_states = [r['state'] for r in results]
            gens_done += int(chunk_gens)

            all_fits = [ft for st in island_states for ft in st['fits']]
            avg_fit = sum(x[0] for x in all_fits) / max(1, len(all_fits))
            avg_err = sum(x[1] for x in all_fits) / max(1, len(all_fits))

            improved = False
            for r in results:
                if (r['best_fitness'] > best_fitness) or (r['best_fitness'] == best_fitness and r['best_error'] < best_error):
                    best_fitness = float(r['best_fitness'])
                    best_error = float(r['best_error'])
                    best_team = r['best_team']
                    improved = True

            print(
                f"平均fitness: {avg_fit:.6f} (平均AAE={avg_err:.6f})，最佳fitness: {best_fitness:.6f} (AAE={best_error:.6f})",
                flush=True,
            )
            if improved:
                no_improve_chunks = 0
            else:
                no_improve_chunks += 1

            if improved and best_team is not None:
                print(f"发现新的历史最佳误差: {best_error:.6f} (fitness={best_fitness:.6f})", flush=True)
                try:
                    tmp_ev = _make_evaluator_from_cfg(cfg)
                    print("当前最佳个体表达式：")
                    print("init_dex: ", tmp_ev._canonical_tree_str(best_team['init_dex']))
                    print("update: ", tmp_ev._canonical_tree_str(best_team['update']))
                    print("query: ", tmp_ev._canonical_tree_str(best_team['query']))
                except Exception:
                    pass

            if islands > 1 and gens_done < int(generations) and int(mig_period) > 0:
                print(f"[MIGRATE] after_gen={gens_done} k={mig_k} islands={islands}", flush=True)
                island_states = _migrate_island_states(island_states, mig_k=mig_k)

            if (
                bool(cfg.get("llm_enable", False))
                and str(cfg.get("llm_mode", "none")) in {"stagnation", "both"}
                and gens_done < int(generations)
                and no_improve_chunks >= max(1, int(cfg.get("llm_stagnation_patience", 2)))
            ):
                try:
                    existing_keys = _collect_existing_canonical_keys_from_states(island_states, main_eval_for_llm)
                    base_team_for_llm = best_team
                    if base_team_for_llm is None:
                        try:
                            base_team_for_llm = {
                                "init_dex": _skeleton_individual_from_ctx(gp_ctx_main, "init_dex"),
                                "update": _skeleton_individual_from_ctx(gp_ctx_main, "update"),
                                "query": _skeleton_individual_from_ctx(gp_ctx_main, "query"),
                            }
                        except Exception:
                            base_team_for_llm = None

                    cand_limit = max(1, int(cfg.get("llm_stagnation_num_candidates", 6)))
                    phase_extra_prompt_hints = _collect_recent_failure_hints_from_states(island_states)
                    phase_candidates = llm_engine.prepare_phase_candidates(
                        phase="stagnation",
                        gp_ctx=gp_ctx_main,
                        evaluator=main_eval_for_llm,
                        base_team=base_team_for_llm,
                        existing_canon=existing_keys,
                        limit=cand_limit,
                        extra_prompt_hints=phase_extra_prompt_hints,
                    )
                    for i, c in enumerate(phase_candidates):
                        llm_logger.info(
                            "stagnation candidate accepted",
                            idx=int(i),
                            source=str(c.get("source", "offline_json")),
                            fitness=float(c.get("fit", 0.0)),
                            error=float(c.get("err", 0.0)),
                        )
                    candidate_specs = [
                        _serialize_team_spec(c["team"], rationale=c.get("rationale", ""), source=c.get("source", "offline_json"))
                        for c in phase_candidates
                    ]
                    budget = max(0, int(cfg.get("llm_stagnation_max_inject", 2)))
                    budget = min(budget, len(candidate_specs), len(island_states))
                    total_injected = 0
                    if candidate_specs and budget > 0:
                        new_states = []
                        remain = budget
                        remaining_specs = list(candidate_specs)

                        for st in island_states:
                            if remain <= 0 or not remaining_specs:
                                new_states.append(st)
                                continue

                            st, inserted, accepted_specs = _inject_llm_immigrants_with_engine(
                                st,
                                cfg,
                                gp_ctx_main,
                                llm_engine,
                                remaining_specs,
                                success_budget=1,
                            )

                            total_injected += int(inserted)
                            remain = max(0, remain - int(inserted))

                            if accepted_specs:
                                accepted_ids = {id(spec) for spec in accepted_specs}
                                remaining_specs = [
                                    spec for spec in remaining_specs
                                    if id(spec) not in accepted_ids
                                ]

                            new_states.append(st)

                        island_states = new_states

                    print(
                        f"[LLM_IMMIGRANT_TRIGGER] after_gen={gens_done} stagnation_chunks={no_improve_chunks} "
                        f"candidates={len(candidate_specs)} budget={budget} injected={total_injected}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[LLM_IMMIGRANT_TRIGGER_SKIP] after_gen={gens_done} reason={e}", flush=True)
                no_improve_chunks = 0

                for st in island_states:
                    _, fit_i, err_i, team_i = _refresh_island_best(st)
                    if (fit_i > best_fitness) or (fit_i == best_fitness and err_i < best_error):
                        best_fitness = float(fit_i)
                        best_error = float(err_i)
                        best_team = team_i

            if best_error <= 0:
                print("找到完美解!", flush=True)
                break


    finally:

        if pool is not None:

            try:

                pool.shutdown(wait=True, cancel_futures=True)

            except TypeError:

                pool.shutdown(wait=True)

    return best_team, best_fitness, best_error

def test_generated_code(code):
    """测试生成的代码"""
    # 将代码写入临时文件
    with open("temp_cmsketch.py", "w", encoding="utf-8") as f:
        f.write(code)

    try:
        # 运行临时文件
        result = subprocess.run([sys.executable, "temp_cmsketch.py"],
                                capture_output=True, text=True, timeout=30)  # 增加超时时间

        if result.returncode == 0:
            print(" 生成的代码运行成功!")
            print("输出结果:")
            print(result.stdout)
            return True
        else:
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(" 生成的代码运行超时!")
        return False
    except Exception as e:
        print(f" 测试过程中发生错误: {e}")
        return False


def _run_one_restart_job(job):
    """
    job = (run_idx, run_seed, args_dict)
    返回:
    (stage2_error, stage1_fitness, stage1_error, run_seed, code_path, exprs_dict, proxy_mode)
    """
    run_idx, run_seed, args_dict = job
    out_dir = args_dict.get("out_dir", "runs")
    os.makedirs(out_dir, exist_ok=True)

    proxy_modes = args_dict.get("proxy_modes", ["proxy_balanced", "proxy_head", "proxy_collision"])
    if not proxy_modes:
        proxy_modes = ["proxy_balanced", "proxy_head", "proxy_collision"]

    if bool(args_dict.get("stage1_multi_proxy", False)) and str(args_dict.get("stage1_dataset_mode", "proxy")) == "proxy":
        run_proxy_mode = "multi_proxy"
    elif str(args_dict.get("stage1_dataset_mode", "proxy")) == "proxy":
        run_proxy_mode = proxy_modes[run_idx % len(proxy_modes)]
    else:
        run_proxy_mode = "proxy_balanced"

    log_path = os.path.join(out_dir, f"run_{run_idx + 1:03d}_seed{run_seed}.log")
    code_path = os.path.join(out_dir, f"best_seed{run_seed}.py")
    meta_path = os.path.join(out_dir, f"best_seed{run_seed}.json")

    try:
        with open(log_path, "w", encoding="utf-8") as lf, \
                contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):

            print( f"[RUN {run_idx + 1}] seed={run_seed} stage1_dataset_mode={args_dict.get('stage1_dataset_mode', 'proxy')} proxy_mode_arg={run_proxy_mode}")

            # ---------- 阶段1：proxy 搜索 ----------
            best_team, best_fitness, best_error = evolve_cmsketch(
                population_size=args_dict["pop"],
                generations=args_dict["gen"],
                seed=run_seed,
                dataset_root=args_dict["dataset_root"],
                pkts=args_dict["pkts"],
                max_files=args_dict["files"],
                start=args_dict["start"],
                shuffle=args_dict["shuffle"],
                dataset_mode=args_dict["stage1_dataset_mode"],
                proxy_mode=(args_dict.get("proxy_modes", ["proxy_balanced"])[0] if args_dict.get("stage1_multi_proxy", False) else run_proxy_mode),
                stage1_proxy_modes=args_dict.get("proxy_modes", ["proxy_balanced"]),
                stage1_multi_proxy=args_dict.get("stage1_multi_proxy", False),
                proxy_pool_mul=args_dict["proxy_pool_mul"],
                proxy_min_u=args_dict["proxy_min_u"],
                islands=args_dict["islands"],
                tournament_size=args_dict["tournament_size"],
                parent_selector=args_dict["parent_selector"],
                lexicase_cases=args_dict["lexicase_cases"],
                lexicase_epsilon=args_dict["lexicase_epsilon"],
                hard_case_replay=args_dict["hard_case_replay"],
                hard_case_stage_topk=args_dict["hard_case_stage_topk"],
                hard_case_absent_topk=args_dict["hard_case_absent_topk"],
                hard_case_scan_mul=args_dict["hard_case_scan_mul"],
                hard_case_decay=args_dict["hard_case_decay"],
                hard_case_weight=args_dict["hard_case_weight"],
                llm_enable=args_dict["llm_enable"],
                llm_mode=args_dict["llm_mode"],
                llm_provider=args_dict["llm_provider"],
                llm_model=args_dict["llm_model"],
                llm_base_url=args_dict["llm_base_url"],
                llm_api_key_env=args_dict["llm_api_key_env"],
                llm_timeout=args_dict["llm_timeout"],
                llm_seed_ratio=args_dict["llm_seed_ratio"],
                llm_seed_max=args_dict["llm_seed_max"],
                llm_stagnation_patience=args_dict["llm_stagnation_patience"],
                llm_stagnation_num_candidates=args_dict["llm_stagnation_num_candidates"],
                llm_stagnation_max_inject=args_dict["llm_stagnation_max_inject"],
                llm_offline_candidates_path=args_dict["llm_offline_candidates_path"],
                llm_log_path=args_dict["llm_log_path"],
                llm_target_funcs=args_dict["llm_target_funcs"],
                llm_use_case_vec=args_dict["llm_use_case_vec"],
                llm_use_hard_cases=args_dict["llm_use_hard_cases"],
                llm_ref_init_pset_path=args_dict["llm_ref_init_pset_path"],
                llm_ref_update_pset_path=args_dict["llm_ref_update_pset_path"],
                llm_ref_query_pset_path=args_dict["llm_ref_query_pset_path"],
                init_p_skeleton=args_dict["init_p_skeleton"],
                init_p_seed=args_dict["init_p_seed"],
                reset_prob=args_dict["reset_prob"],
                reset_whole_prob=args_dict["reset_whole_prob"],
                mutation_prob=args_dict["mutation_prob"],
                mig_period=args_dict["mig_period"],
                mig_k=args_dict["mig_k"],
                max_size=args_dict["max_size"],
                dataset_seed=args_dict["stage1_dataset_seed"],
                fixed_stream_path=(
                    str(args_dict["stage1_fixed_stream"])
                    if args_dict.get("stage1_multi_proxy", False)
                    else _apply_proxy_mode_to_stream_path(
                        args_dict["stage1_fixed_stream"], args_dict["stage1_dataset_mode"], run_proxy_mode
                    )
                ),
            )
            print(f"[STAGE1] proxy_mode_arg={run_proxy_mode} best_fitness={best_fitness:.6f} best_error={best_error:.2f}")

            # ---------- 阶段2：真实流复评 ----------
            val_seed = int(args_dict["stage2_dataset_seed"]) & 0xFFFFFFFF
            val_evaluator = CMSketchEvaluator(
                dataset_root=args_dict["dataset_root"],
                pkts=args_dict["stage2_pkts"],
                max_files=args_dict["stage2_files"],
                start=args_dict["stage2_start"],
                shuffle=args_dict["stage2_shuffle"],
                seed=val_seed,
                dataset_mode="real",
                proxy_mode="proxy_balanced",
                proxy_pool_mul=args_dict["proxy_pool_mul"],
                proxy_min_u=args_dict["proxy_min_u"],
                fixed_stream_path=args_dict["stage2_fixed_stream"],
            )
            val_evaluator.E0 = 1.0

            _, stage2_error = val_evaluator.evaluate_individual(
                best_team["init_dex"],
                best_team["update"],
                best_team["query"],
            )
            print(
                f"[STAGE2] real_error={stage2_error:.2f} "
                f"pkts={len(val_evaluator.test_data)} U={val_evaluator.U} U_ratio={val_evaluator.U_ratio:.4f}"
            )

            exprs = {
                "init_dex": str(best_team["init_dex"]),
                "update": str(best_team["update"]),
                "query": str(best_team["query"]),
            }

            # 导出的最终代码，烘焙阶段2真实流配置
            ev = CMSketchEvaluator.__new__(CMSketchEvaluator)
            ev.dataset_root = args_dict["dataset_root"]
            ev.pkts = int(args_dict["stage2_pkts"])
            ev.max_files = int(args_dict["stage2_files"])
            ev.start = int(args_dict["stage2_start"])
            ev.shuffle = bool(args_dict["stage2_shuffle"])
            ev.seed = int(val_seed) & 0xFFFFFFFF
            ev.E0 = None

            best_code = CMSketchEvaluator.generate_complete_code(
                ev,
                best_team["init_dex"],
                best_team["update"],
                best_team["query"],
            )

            with open(code_path, "w", encoding="utf-8") as f:
                f.write(best_code)

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "seed": int(run_seed),
                        "proxy_mode": run_proxy_mode,
                        "stage1_best_fitness": float(best_fitness),
                        "stage1_best_error": float(best_error),
                        "stage2_real_error": float(stage2_error),
                        "exprs": exprs,
                        "code_path": code_path,
                        "log_path": log_path,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(
                f"[DONE] proxy_mode={run_proxy_mode} "
                f"stage1_best_fitness={best_fitness:.6f} "
                f"stage1_best_error={best_error:.2f} "
                f"stage2_real_error={stage2_error:.2f}"
            )

            return (
                float(stage2_error),
                float(best_fitness),
                float(best_error),
                int(run_seed),
                code_path,
                exprs,
                str(run_proxy_mode),
            )

    except Exception as e:
        return 2_000_000_000.0, 0.0, 2_000_000_000.0, int(run_seed), "", {"error": str(e)}, "error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolve CMS variants with DEAP GP on univ2_trace flowid stream.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed. If not set, a time-based seed is used.")
    parser.add_argument("--restarts", type=int, default=1, help="Number of independent restarts (multi-start).")
    parser.add_argument("--pop", type=int, default=100, help="Population size.")
    parser.add_argument("--gen", type=int, default=400, help="Number of generations.")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数（用于并行 restarts）")
    parser.add_argument("--out_dir", type=str, default="runs", help="并行模式下每个 restart 的输出目录（log/py/json）")
    # 数据集：默认使用 /data/8T/xgr/traces/univ2_trace/univ2_npy 下的 *.flowid.npy（少量采样）
    parser.add_argument("--dataset_root", type=str, default="/data/8T/xgr/traces/univ2_trace",
                        help="univ2_trace 目录或 univ2_npy 子目录（里面应包含 *.flowid.npy）")
    parser.add_argument("--pkts", type=int, default=10000,
                        help="从数据集中一共取多少个 flowid（越小越快）")
    parser.add_argument("--files", type=int, default=1,
                        help="最多读取多少个分片文件（univ2_pt0/1/2...）")
    # 兼容你之前的命令行写法：--max_files 等价于 --files
    parser.add_argument("--max_files", dest="files", type=int, default=1,
                        help="(alias) 同 --files")
    parser.add_argument("--start", type=int, default=0,
                        help="每个文件从第几个元素开始取")
    parser.add_argument("--shuffle", action="store_true",
                        help="是否在局部窗口里打乱采样（默认不打乱，取前 pkts 个）")
    parser.add_argument("--stage1_dataset_mode", type=str, default="proxy",
                        choices=["proxy", "real"],
                        help="阶段1搜索使用的数据模式")
    parser.add_argument("--proxy_modes", type=str, default="proxy_balanced,proxy_head,proxy_collision",
                        help="阶段1多代理流模式，逗号分隔")
    parser.add_argument("--stage1_multi_proxy", dest="stage1_multi_proxy", action="store_true",
                        help="阶段1启用 multi-proxy 聚合（默认开启）")
    parser.add_argument("--no_stage1_multi_proxy", dest="stage1_multi_proxy", action="store_false",
                        help="关闭阶段1 multi-proxy，退回单 proxy")
    parser.add_argument("--proxy_pool_mul", type=int, default=8,
                        help="proxy候选池倍数，pool_target = pkts * proxy_pool_mul")
    parser.add_argument("--proxy_min_u", type=int, default=2500,
                        help="proxy训练流最小目标U")

    parser.add_argument("--stage2_pkts", type=int, default=20000,
                        help="阶段2真实流复评包数")
    parser.add_argument("--stage2_files", type=int, default=16,
                        help="阶段2真实流复评文件数")
    parser.add_argument("--stage2_start", type=int, default=0,
                        help="阶段2真实流复评起点")
    parser.add_argument("--stage2_shuffle", action="store_true",
                        help="阶段2真实流是否shuffle")
    parser.add_argument("--stage1_dataset_seed", type=int, default=20250319,
                        help="阶段1数据抽样seed；与搜索seed分离")
    parser.add_argument("--stage2_dataset_seed", type=int, default=20250320,
                        help="阶段2真实流复评seed；与搜索seed分离")
    parser.add_argument("--stage1_fixed_stream", type=str, default="",
                        help="阶段1固定流npy路径；存在则直接加载，不存在则首次生成并保存")
    parser.add_argument("--stage2_fixed_stream", type=str, default="",
                        help="阶段2固定流npy路径；存在则直接加载，不存在则首次生成并保存")
    # 决策层（三大件 + 精英 + 爆发）
    parser.add_argument("--islands", type=int, default=4, help="岛模型数量（>1 开启迁移）")
    parser.add_argument("--tournament_size", type=int, default=5, help="锦标赛大小")
    parser.add_argument("--parent_selector", type=str, default="lexicase", choices=["tournament", "lexicase"],
                        help="父代选择方式：tournament 或 lexicase")
    parser.add_argument("--lexicase_cases", type=int, default=4,
                        help="每次父代选择时随机抽取多少个 case 做 sampled lexicase")
    parser.add_argument("--lexicase_epsilon", type=float, default=1e-9,
                        help="lexicase 的误差容忍 epsilon")
    parser.add_argument("--hard_case_replay", action="store_true",
                        help="开启 hard-case replay / counterexample-driven case 池")
    parser.add_argument("--hard_case_stage_topk", type=int, default=24,
                        help="每个 stage 保留多少个 hardest present cases")
    parser.add_argument("--hard_case_absent_topk", type=int, default=12,
                        help="保留多少个 absent / false-positive hardest cases")
    parser.add_argument("--hard_case_scan_mul", type=int, default=3,
                        help="保留参数占位：扫描倍率（当前版本主要用于配置兼容）")
    parser.add_argument("--hard_case_decay", type=float, default=0.85,
                        help="旧 hard cases 的衰减系数")
    parser.add_argument("--hard_case_weight", type=float, default=0.50,
                        help="每个 stage 评估时 replay cases 的混入比例")
    parser.add_argument("--llm_enable", action="store_true",
                        help="Enable LLM proposal path (default off).")
    parser.add_argument("--llm_mode", type=str, default="none",
                        choices=["none", "seeds", "stagnation", "both"],
                        help="LLM mode: none / seeds / stagnation / both")
    parser.add_argument("--llm_provider", type=str, default="none",
                        choices=["none", "offline_json", "openai_compatible"],
                        help="LLM candidate provider")
    parser.add_argument("--llm_model", type=str, default="", help="openai-compatible model name")
    parser.add_argument("--llm_base_url", type=str, default="", help="openai-compatible base url")
    parser.add_argument("--llm_api_key_env", type=str, default="", help="API key environment variable name")
    parser.add_argument("--llm_timeout", type=float, default=30.0, help="LLM request timeout seconds")

    parser.add_argument("--init_p_skeleton", type=float, default=0.70,
                        help="Initial skeleton sampling probability")
    parser.add_argument("--init_p_seed", type=float, default=0.20,
                        help="Initial manual seed sampling probability")

    parser.add_argument("--llm_seed_ratio", type=float, default=0.0,
                        help="Initial population LLM seed injection ratio")
    parser.add_argument("--llm_seed_max", type=int, default=0,
                        help="Maximum successful LLM seed injections during initialization")
    parser.add_argument("--llm_stagnation_patience", type=int, default=2,
                        help="Trigger stagnation immigrants after this many non-improving chunks")
    parser.add_argument("--llm_stagnation_num_candidates", type=int, default=6,
                        help="Number of candidates requested per stagnation trigger")
    parser.add_argument("--llm_stagnation_max_inject", type=int, default=2,
                        help="Per-trigger successful immigrant injection budget")

    parser.add_argument("--llm_offline_candidates_path", type=str, default="",
                        help="Offline candidates file path (.jsonl/.json)")
    parser.add_argument("--llm_log_path", type=str, default="",
                        help="LLM path JSONL log path")
    parser.add_argument("--llm_target_funcs", type=str, default="update,query",
                        help="Functions allowed for LLM edits: update,query or update,query,init_dex")
    parser.add_argument("--llm_use_case_vec", action="store_true",
                        help="Include case_vec in LLM report")
    parser.add_argument("--llm_use_hard_cases", action="store_true",
                        help="Include hard-case state in LLM report")

    parser.add_argument("--llm_ref_init_pset_path", type=str, default="",
                        help="Init primitive reference file path (optional)")
    parser.add_argument("--llm_ref_update_pset_path", type=str, default="",
                        help="Update primitive reference file path (optional)")
    parser.add_argument("--llm_ref_query_pset_path", type=str, default="",
                        help="Query primitive reference file path (optional)")
    parser.add_argument("--reset_prob", type=float, default=0.10, help="组件重置概率（随机重采样某组件树）")
    parser.add_argument("--reset_whole_prob", type=float, default=0.02,
                        help="更强的组件重置概率（直接重采样某组件，不做其它变异）")
    parser.add_argument("--mutation_prob", type=float, default=0.90, help="常规变异概率")

    parser.add_argument("--mig_period", type=int, default=8, help="岛间迁移周期（代）")
    parser.add_argument("--mig_k", type=int, default=3, help="每次迁移 top-k team")
    parser.add_argument("--max_size", type=int, default=80, help="GP树最大节点数（bloat约束）")

    parser.set_defaults(stage1_multi_proxy=True)

    args = parser.parse_args()

    if not args.stage1_fixed_stream:
        args.stage1_fixed_stream = os.path.join(
            "fixed_streams",
            f"stage1_{args.stage1_dataset_mode}_{args.pkts}pkts_{args.files}f_"
            f"start{args.start}_sh{int(args.shuffle)}_seed{int(args.stage1_dataset_seed)}.npy"
        )

    if not args.stage2_fixed_stream:
        args.stage2_fixed_stream = os.path.join(
            "fixed_streams",
            f"stage2_real_{args.stage2_pkts}pkts_{args.stage2_files}f_"
            f"start{args.stage2_start}_sh{int(args.stage2_shuffle)}_seed{int(args.stage2_dataset_seed)}.npy"
        )

    proxy_modes = [s.strip() for s in str(args.proxy_modes).split(",") if s.strip()]
    if not proxy_modes:
        proxy_modes = ["proxy_balanced", "proxy_head", "proxy_collision"]
    args.proxy_modes = proxy_modes

    print("开始演化 Count-Min Sketch 变体（univ2_trace 真实 flowid 流）...")
    print(
        f"dataset_root={args.dataset_root} pkts={args.pkts} files={args.files} start={args.start} shuffle={args.shuffle}")

    base_seed = args.seed if args.seed is not None else (time.time_ns() % (2 ** 32))

    best_pack = None  # (fitness, error, seed, code_path, exprs, best_team_or_None)

    use_parallel = (int(args.workers) > 1) and (int(args.restarts) > 1)
    out_dir = args.out_dir
    args_dict = vars(args).copy()
    args_dict["out_dir"] = out_dir

    if use_parallel:
        os.makedirs(out_dir, exist_ok=True)

        jobs = []
        for r in range(max(1, args.restarts)):
            run_seed = (base_seed + r) % (2 ** 32)
            jobs.append((r, run_seed, args_dict))

        maxw = min(int(args.workers), len(jobs))
        print(f"[PARALLEL] restarts={len(jobs)} workers={maxw} out_dir={out_dir}")

        with cf.ProcessPoolExecutor(max_workers=maxw) as ex:
            futs = [ex.submit(_run_one_restart_job, job) for job in jobs]
            for fut in cf.as_completed(futs):
                stage2_err, stage1_fit, stage1_err, seed, code_path, exprs, proxy_mode = fut.result()
                print(
                    f"[RUN DONE] seed={seed} proxy_mode={proxy_mode} "
                    f"stage2_real_error={stage2_err:.2f} "
                    f"stage1_best_fitness={stage1_fit:.6f} "
                    f"stage1_best_error={stage1_err:.2f}"
                )
                if (best_pack is None) or (stage2_err < best_pack[0]):
                    best_pack = (stage2_err, stage1_fit, stage1_err, seed, proxy_mode, code_path, exprs, None)

    else:
        for r in range(max(1, args.restarts)):
            run_seed = (base_seed + r) % (2 ** 32)
            run_proxy_mode = args.proxy_modes[r % len(args.proxy_modes)]

            print(f"\n===== RUN {r + 1}/{max(1, args.restarts)}  seed={run_seed}  pop={args.pop}  gen={args.gen} =====")
            print(f"[STAGE1] dataset_mode={args.stage1_dataset_mode} proxy_mode={run_proxy_mode}")

            best_team, best_fitness, best_error = evolve_cmsketch(
                population_size=args.pop,
                generations=args.gen,
                seed=run_seed,
                dataset_root=args.dataset_root,
                pkts=args.pkts,
                max_files=args.files,
                start=args.start,
                shuffle=args.shuffle,
                dataset_mode=args.stage1_dataset_mode,
                proxy_mode=(args.proxy_modes[0] if args.stage1_multi_proxy else run_proxy_mode),
                stage1_proxy_modes=args.proxy_modes,
                stage1_multi_proxy=args.stage1_multi_proxy,
                proxy_pool_mul=args.proxy_pool_mul,
                proxy_min_u=args.proxy_min_u,
                islands=args.islands,
                tournament_size=args.tournament_size,
                parent_selector=args.parent_selector,
                lexicase_cases=args.lexicase_cases,
                lexicase_epsilon=args.lexicase_epsilon,
                hard_case_replay=args.hard_case_replay,
                hard_case_stage_topk=args.hard_case_stage_topk,
                hard_case_absent_topk=args.hard_case_absent_topk,
                hard_case_scan_mul=args.hard_case_scan_mul,
                hard_case_decay=args.hard_case_decay,
                hard_case_weight=args.hard_case_weight,
                llm_enable=args.llm_enable,
                llm_mode=args.llm_mode,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                llm_base_url=args.llm_base_url,
                llm_api_key_env=args.llm_api_key_env,
                llm_timeout=args.llm_timeout,
                llm_seed_ratio=args.llm_seed_ratio,
                llm_seed_max=args.llm_seed_max,
                llm_stagnation_patience=args.llm_stagnation_patience,
                llm_stagnation_num_candidates=args.llm_stagnation_num_candidates,
                llm_stagnation_max_inject=args.llm_stagnation_max_inject,
                llm_offline_candidates_path=args.llm_offline_candidates_path,
                llm_log_path=args.llm_log_path,
                llm_target_funcs=args.llm_target_funcs,
                llm_use_case_vec=args.llm_use_case_vec,
                llm_use_hard_cases=args.llm_use_hard_cases,
                llm_ref_init_pset_path=args.llm_ref_init_pset_path,
                llm_ref_update_pset_path=args.llm_ref_update_pset_path,
                llm_ref_query_pset_path=args.llm_ref_query_pset_path,
                init_p_skeleton=args.init_p_skeleton,
                init_p_seed=args.init_p_seed,
                reset_prob=args.reset_prob,
                reset_whole_prob=args.reset_whole_prob,
                mutation_prob=args.mutation_prob,
                mig_period=args.mig_period,
                mig_k=args.mig_k,
                max_size=args.max_size,
                dataset_seed=args.stage1_dataset_seed,
                fixed_stream_path=(
                    str(args.stage1_fixed_stream)
                    if args.stage1_multi_proxy
                    else _apply_proxy_mode_to_stream_path(
                        args.stage1_fixed_stream, args.stage1_dataset_mode, run_proxy_mode
                    )
                ),
            )

            print(f"[RUN {r + 1}][STAGE1] best_fitness={best_fitness:.6f} best_error={best_error:.2f}")

            val_seed = int(args.stage2_dataset_seed) & 0xFFFFFFFF
            val_evaluator = CMSketchEvaluator(
                dataset_root=args.dataset_root,
                pkts=args.stage2_pkts,
                max_files=args.stage2_files,
                start=args.stage2_start,
                shuffle=args.stage2_shuffle,
                seed=val_seed,
                dataset_mode="real",
                proxy_mode="proxy_balanced",
                proxy_pool_mul=args.proxy_pool_mul,
                proxy_min_u=args.proxy_min_u,
                fixed_stream_path=args.stage2_fixed_stream,
            )
            val_evaluator.E0 = 1.0

            _, stage2_error = val_evaluator.evaluate_individual(
                best_team["init_dex"],
                best_team["update"],
                best_team["query"]
            )

            print(
                f"[RUN {r + 1}][STAGE2] proxy_mode={run_proxy_mode} "
                f"real_error={stage2_error:.2f} "
                f"pkts={len(val_evaluator.test_data)} U={val_evaluator.U} U_ratio={val_evaluator.U_ratio:.4f}"
            )

            exprs = {
                "init_dex": str(best_team["init_dex"]),
                "update": str(best_team["update"]),
                "query": str(best_team["query"]),
            }

            if (best_pack is None) or (stage2_error < best_pack[0]):
                best_pack = (stage2_error, best_fitness, best_error, run_seed, run_proxy_mode, "", exprs, best_team)

    best_stage2_error, best_stage1_fitness, best_stage1_error, best_seed, best_proxy_mode, best_code_path, best_exprs, best_team = best_pack
    print(
        f"\n[FINAL BEST] seed={best_seed} proxy_mode={best_proxy_mode} "
        f"stage2_real_error={best_stage2_error:.2f} "
        f"stage1_best_fitness={best_stage1_fitness:.6f} "
        f"stage1_best_error={best_stage1_error:.2f}"
    )

    # 拿到 best_code：并行模式直接读子进程产出的文件；非并行模式现场生成
    if best_code_path:
        with open(best_code_path, "r", encoding="utf-8") as f:
            best_code = f.read()
    else:
        set_seed(best_seed)
        val_seed = int(args.stage2_dataset_seed) & 0xFFFFFFFF

        ev = CMSketchEvaluator.__new__(CMSketchEvaluator)
        ev.dataset_root = args.dataset_root
        ev.pkts = int(args.stage2_pkts)
        ev.max_files = int(args.stage2_files)
        ev.start = int(args.stage2_start)
        ev.shuffle = bool(args.stage2_shuffle)
        ev.seed = int(val_seed) & 0xFFFFFFFF
        ev.E0 = None

        best_code = CMSketchEvaluator.generate_complete_code(
            ev,
            best_team["init_dex"],
            best_team["update"],
            best_team["query"],
        )

    # 保存最终最佳代码（保持你原来的文件名不变）
    with open("best_mutated_cmsketch_large.py", "w", encoding="utf-8") as f:
        f.write(best_code)

    print("\n最佳变异版本已保存为 'best_mutated_cmsketch_large.py'")
    print("\n最佳表达式树:")
    if isinstance(best_exprs, dict):
        print(f"init_dex: {best_exprs.get('init_dex')}")
        print(f"update: {best_exprs.get('update')}")
        print(f"query: {best_exprs.get('query')}")

    print("\n测试生成的代码.")
    test_generated_code(best_code)

    if os.path.exists("temp_cmsketch.py"):
        os.remove("temp_cmsketch.py")

    print("\n演化过程完成!")
