import sys
import copy
from typing import Callable
from dataclasses import dataclass, asdict, field
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
        self.kpart_avg_err_scale = [20.0, 16.0, 0.0]
        self.kpart_cut_penalty = [360000.0, 520000.0, None]
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
                penalty += 20000
            elif update_calls < 2:
                penalty += (2 - update_calls) * 5000

            # ---- query 约束 ----
            if not query_info["root_ok"]:
                penalty += 120000

            if query_info["forbidden_hits"]:
                penalty += 260000 + 40000 * sum(query_info["forbidden_hits"].values())

            pass

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
                penalty += (2 - query_eff["query_date_calls"]) * 10000

            # ---- AST 级结构约束：按当前原语集语义约束 hash/path id ----
            self.debug_stats["nonconst_hash_idx_total"] += int(init_pat["nonconst_hash_idx"])
            self.debug_stats["nonconst_path_idx_total"] += int(update_pat["nonconst_path_idx"])
            self.debug_stats["nonconst_path_idx_total"] += int(query_pat["nonconst_path_idx"])
            self.debug_stats["bad_write_context_total"] += int(update_pat["bad_write_context"])

            if init_pat["nonconst_hash_idx"] > 0:
                penalty += init_pat["nonconst_hash_idx"] * 8000

            if update_pat["nonconst_path_idx"] > 0:
                penalty += update_pat["nonconst_path_idx"] * 15000

            if query_pat["nonconst_path_idx"] > 0:
                penalty += query_pat["nonconst_path_idx"] * 15000

            if update_pat["bad_write_context"] > 0:
                bwc = int(update_pat["bad_write_context"])
                penalty += bwc * 40000

                # bad_write_ctx 一旦达到 2，基本已经不是干净的 update 结构了
                if bwc >= 3:
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
                penalty += int((2.0 - avg_qry_reads) * 12000)

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
    llm_team_bank_by_family = {}

    return {
        "toolboxes": toolboxes,
        "pset_map": pset_map,
        "seed_bank": seed_bank,
        "skeleton_bank": skeleton_bank,
        "llm_seed_bank": llm_seed_bank,
        "llm_team_bank": llm_team_bank,
        "llm_team_bank_by_family": llm_team_bank_by_family,
    }

def _build_local_llm_seed_exprs():
    return {
        "init_dex": [
            "list_3(hash_salt(0,e,1), safe_mod(hash_salt(0,e,1),102), 102, "
            "hash_salt(1,e,1), safe_mod(hash_salt(1,e,1),102), 102, "
            "hash_salt(2,e,1), safe_mod(hash_salt(2,e,1),102), 102)",
            "list_3(select_hash(0,e), safe_mod(hash_salt(0,e,11),102), 102, "
            "select_hash(1,e), safe_mod(hash_salt(1,e,13),102), 102, "
            "select_hash(2,e), safe_mod(hash_salt(2,e,17),102), 102)",
            "list_3(hash_on_slice(0,e,0,8), safe_mod(hash_on_slice(0,e,0,8),102), 102, "
            "hash_on_slice(1,e,4,12), safe_mod(hash_on_slice(1,e,4,12),102), 102, "
            "hash_on_slice(2,e,8,16), safe_mod(hash_on_slice(2,e,8,16),102), 102)"
        ],
        "update": [
            "base(update_count(e,0,1), update_count(e,1,1), update_count(e,2,1))",
            "base(write_count(e,0,safe_add(query_count(e,0),1)), "
            "write_count(e,1,safe_add(query_count(e,1),1)), "
            "write_count(e,2,safe_add(query_count(e,2),1)))",
            "base(updatecount_if(lt(query_count(e,0),query_count(e,1)),e,0,1), "
            "updatecount_if(lt(query_count(e,1),query_count(e,2)),e,1,1), "
            "updatecount_if(lt(query_count(e,2),query_count(e,0)),e,2,1))"
        ],
        "query": [
            "base_sel(0, query_date(e,0), query_date(e,1), query_date(e,2))",
            "base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2))",
            "base_sel(3, query_date(e,0), query_date(e,1), query_date(e,2))",
            "base_sel(cnt_rdstate(e,0), query_date(e,0), query_date(e,1), query_date(e,2))"
        ],
    }

def _build_local_llm_seed_teams():
    return [
        {
            "name": "cm_min_basic",
            "family_tag": "symmetric-init/triple-write/min-query",
            "init_dex": "list_3(hash_salt(0,e,1), safe_mod(hash_salt(0,e,1),102), 102, "
                        "hash_salt(1,e,1), safe_mod(hash_salt(1,e,1),102), 102, "
                        "hash_salt(2,e,1), safe_mod(hash_salt(2,e,1),102), 102)",
            "update": "base(update_count(e,0,1), update_count(e,1,1), update_count(e,2,1))",
            "query": "base_sel(0, query_date(e,0), query_date(e,1), query_date(e,2))",
        },
        {
            "name": "read_before_write_median",
            "family_tag": "asymmetric-init/read-before-write/median-query",
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
            "family_tag": "slice-hash-init/double-write/avg-query",
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
                "family_tag": str(spec.get("family_tag", _team_family_tag({"init_dex": init_ind, "update": update_ind, "query": query_ind}))),
                "family_parts": _team_family_parts({"init_dex": init_ind, "update": update_ind, "query": query_ind}),
            })
            team_seen.add(key)
        except Exception as e:
            print(f"[LLM_TEAM_SKIP] name={spec.get('name', 'unknown')} reason={e}", flush=True)

    gp_ctx["llm_team_bank"] = built_teams[:keep_k]
    gp_ctx = _rebuild_llm_team_bank_by_family(gp_ctx)
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


def _append_unique_str(dst, item):
    s = str(item).strip()
    if s and s not in dst:
        dst.append(s)



def _tree_text_for_family(tree):
    try:
        return str(tree)
    except Exception:
        return repr(tree)


def _classify_init_family(tree):
    txt = _tree_text_for_family(tree)
    has_slice = ("hash_on_slice(" in txt) or ("str_slice(" in txt)
    has_select = ("select_hash(" in txt)
    has_salt = ("hash_salt(" in txt)
    has_nested_hash = ("hash_salt(hash_salt(" in txt) or ("select_hash(hash_salt(" in txt) or ("hash_on_slice(hash_salt(" in txt)
    hash_kind_n = int(has_select) + int(has_salt) + int("hash_on_slice(" in txt)
    if has_slice and (has_select or has_salt):
        return "hybrid-slice-init"
    if has_nested_hash or txt.count("hash_salt(") >= 4:
        return "layered-init"
    if has_slice:
        return "slice-hash-init"
    if hash_kind_n >= 2:
        return "asymmetric-init"
    return "symmetric-init"


def _classify_update_family(tree):
    txt = _tree_text_for_family(tree)
    real_writes = txt.count("update_count(") + txt.count("write_count(")
    cond_writes = txt.count("updatecount_if(") + txt.count("writecount_if(") + txt.count("writestate_if(")
    state_writes = txt.count("update_state(") + txt.count("writestate_if(")
    has_read_before = ("query_count(" in txt) and (("write_count(" in txt) or ("writecount_if(" in txt) or ("update_count(" in txt) or ("updatecount_if(" in txt))
    if state_writes > 0 and has_read_before:
        return "rescue-write"
    if state_writes > 0:
        return "stateful-write"
    if cond_writes >= 2 and real_writes <= 1:
        return "conditional-write"
    if has_read_before:
        return "read-before-write"
    if real_writes >= 3 or cond_writes >= 3:
        return "triple-write"
    if real_writes == 2 or cond_writes == 2:
        return "double-write"
    return "mixed-write"


def _classify_query_family(tree):
    txt = _tree_text_for_family(tree)
    has_state = ("cnt_rdstate(" in txt)
    has_mix_agg = (("base_sel(" in txt) and (("safe_min(" in txt) or ("safe_max(" in txt) or ("median3(" in txt))) or ((txt.count("safe_min(") + txt.count("safe_max(") + txt.count("median3(")) >= 2)
    if has_state and has_mix_agg:
        return "fallback-query"
    if has_mix_agg:
        return "mixed-aggregator-query"
    if has_state:
        return "state-gated-query"
    if "base_sel(0" in txt:
        return "min-query"
    if "base_sel(1" in txt:
        return "max-query"
    if "base_sel(2" in txt:
        return "median-query"
    if "base_sel(3" in txt:
        return "avg-query"
    return "free-query"


def _irregular_family_catalog():
    return {
        "init_dex": {"layered-init", "hybrid-slice-init"},
        "update": {"stateful-write", "rescue-write", "conditional-write"},
        "query": {"fallback-query", "mixed-aggregator-query", "state-gated-query"},
    }


def _is_irregular_family_part(which, fam):
    fam = str(fam or "")
    return fam in _irregular_family_catalog().get(which, set())


def _is_irregular_family_tag(tag: str) -> bool:
    parts = str(tag or "").split("/")
    if len(parts) != 3:
        return False
    return (
        _is_irregular_family_part("init_dex", parts[0])
        or _is_irregular_family_part("update", parts[1])
        or _is_irregular_family_part("query", parts[2])
    )


def _component_family_of_tree(which, tree):
    if which == "init_dex":
        return _classify_init_family(tree)
    if which == "update":
        return _classify_update_family(tree)
    return _classify_query_family(tree)


def _team_family_parts(team):
    return {
        "init_dex": _component_family_of_tree("init_dex", team["init_dex"]),
        "update": _component_family_of_tree("update", team["update"]),
        "query": _component_family_of_tree("query", team["query"]),
    }


def _team_family_tag(team):
    parts = _team_family_parts(team)
    return f'{parts["init_dex"]}/{parts["update"]}/{parts["query"]}'


@dataclass
class ArchitectureSchema:
    arch_type: str = "regular"
    num_branches: int = 3
    branch_roles: list = field(default_factory=list)
    handoff_policy: str = "none"
    query_fusion: str = "base_sel"
    state_usage: str = "none"
    layout_style: str = "regular"
    innovation_note: str = ""
    primary_innovation_axis: str = ""


@dataclass
class MotifSignature:
    is_regular: bool = True
    uses_overflow_state: bool = False
    has_handoff: bool = False
    has_sidecar_branch: bool = False
    is_layered: bool = False
    query_ignores_inf: bool = False
    query_fuses_multi_branch: bool = False
    update_is_conditional: bool = False
    update_is_read_before_write: bool = False
    update_is_branch_asymmetric: bool = False
    init_is_asymmetric: bool = False
    branch_count: int = 3
    layout_family: str = "regular"
    init_family: str = "symmetric-init"
    update_family: str = "triple-write"
    query_family: str = "min-query"


@dataclass
class CandidateMeta:
    family_tag: str = ""
    key_v1: tuple = field(default_factory=tuple)
    key_v2: tuple = field(default_factory=tuple)
    repair_dup_key: tuple = field(default_factory=tuple)
    architecture_schema: dict = field(default_factory=dict)
    motif_signature: dict = field(default_factory=dict)
    schema_hash: str = ""
    motif_key: str = ""
    arch_type: str = "regular"


def _architecture_schema_hash(schema):
    try:
        payload = schema if isinstance(schema, dict) else asdict(schema)
    except Exception:
        payload = dict(schema or {}) if isinstance(schema, dict) else {"arch_type": "regular"}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _motif_signature_key(sig):
    try:
        payload = sig if isinstance(sig, dict) else asdict(sig)
    except Exception:
        payload = dict(sig or {}) if isinstance(sig, dict) else {"layout_family": "regular"}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _infer_architecture_schema_from_team(team):
    fam_parts = _team_family_parts(team)
    init_fam = str(fam_parts.get("init_dex", ""))
    upd_fam = str(fam_parts.get("update", ""))
    qry_fam = str(fam_parts.get("query", ""))
    init_txt = _tree_text_for_family(team.get("init_dex"))
    upd_txt = _tree_text_for_family(team.get("update"))
    qry_txt = _tree_text_for_family(team.get("query"))

    uses_state = ("update_state(" in upd_txt) or ("writestate_if(" in upd_txt) or ("cnt_rdstate(" in qry_txt)
    has_handoff = uses_state or ("query_count(" in upd_txt and ("write_count(" in upd_txt or "update_count(" in upd_txt or "writecount_if(" in upd_txt or "updatecount_if(" in upd_txt))
    layered = (init_fam in {"hybrid-slice-init", "layered-init"}) or ("hash_on_slice(" in init_txt and ("hash_salt(" in init_txt or "select_hash(" in init_txt))
    sidecar = (qry_fam in {"fallback-query", "mixed-aggregator-query", "state-gated-query"}) or ("cnt_rdstate(" in qry_txt)
    conditional = upd_fam in {"conditional-write", "stateful-write", "rescue-write"}

    arch_type = "regular"
    handoff_policy = "none"
    state_usage = "overflow_state" if uses_state else "none"
    layout_style = "regular"
    innovation_note = "regular-cms-like"
    primary_axis = "balanced"

    if uses_state and sidecar:
        arch_type = "overflow" if ("safe_min(" in qry_txt or "base_sel(0" in qry_txt) else "diamond"
        handoff_policy = "overflow_to_sidecar"
        layout_style = "sidecar_heavy"
        innovation_note = "state-gated overflow handoff"
        primary_axis = "update_state_handoff"
    elif layered and conditional:
        arch_type = "pyramid"
        handoff_policy = "layered_correction"
        layout_style = "layered"
        innovation_note = "layered correction over regular body"
        primary_axis = "layout_then_update"
    elif layered and sidecar:
        arch_type = "elastic"
        handoff_policy = "branch_fallback"
        layout_style = "asymmetric_dual_path"
        innovation_note = "main regular body with sidecar branch"
        primary_axis = "layout_sidecar"
    elif layered or conditional or sidecar:
        arch_type = "hybrid"
        handoff_policy = "branch_fallback" if sidecar else ("layered_correction" if layered else "none")
        layout_style = "layered" if layered else ("asymmetric_dual_path" if sidecar else "regular")
        innovation_note = "hybrid irregular sketch assembled from existing primitives"
        primary_axis = "existing_primitives_only"

    branch_roles = ["main", "aux", "aux"]
    if arch_type in {"overflow", "diamond"}:
        branch_roles = ["main", "overflow", "correction"]
    elif arch_type == "pyramid":
        branch_roles = ["main", "layered", "correction"]
    elif arch_type == "elastic":
        branch_roles = ["main", "sidecar", "fallback"]
    elif arch_type == "hybrid":
        branch_roles = ["main", "aux", "aux"]

    query_fusion = "base_sel"
    if qry_fam == "median-query":
        query_fusion = "median"
    elif qry_fam == "min-query":
        query_fusion = "min"
    elif qry_fam == "avg-query":
        query_fusion = "avg"
    elif qry_fam == "state-gated-query":
        query_fusion = "state_gated_min"
    elif qry_fam in {"fallback-query", "mixed-aggregator-query"}:
        query_fusion = "branch_conditional"

    return ArchitectureSchema(
        arch_type=str(arch_type),
        num_branches=3,
        branch_roles=list(branch_roles),
        handoff_policy=str(handoff_policy),
        query_fusion=str(query_fusion),
        state_usage=str(state_usage),
        layout_style=str(layout_style),
        innovation_note=str(innovation_note),
        primary_innovation_axis=str(primary_axis),
    )


def _extract_motif_signature(team, schema=None):
    schema = schema or _infer_architecture_schema_from_team(team)
    schema_dict = schema if isinstance(schema, dict) else asdict(schema)
    fam_parts = _team_family_parts(team)
    init_fam = str(fam_parts.get("init_dex", ""))
    upd_fam = str(fam_parts.get("update", ""))
    qry_fam = str(fam_parts.get("query", ""))
    init_txt = _tree_text_for_family(team.get("init_dex"))
    upd_txt = _tree_text_for_family(team.get("update"))
    qry_txt = _tree_text_for_family(team.get("query"))
    uses_overflow_state = ("update_state(" in upd_txt) or ("writestate_if(" in upd_txt) or ("cnt_rdstate(" in qry_txt)
    sig = MotifSignature(
        is_regular=bool(str(schema_dict.get("arch_type", "regular")) == "regular"),
        uses_overflow_state=bool(uses_overflow_state),
        has_handoff=bool(str(schema_dict.get("handoff_policy", "none")) != "none"),
        has_sidecar_branch=bool(str(schema_dict.get("layout_style", "regular")) in {"sidecar_heavy", "asymmetric_dual_path"} or qry_fam in {"fallback-query", "mixed-aggregator-query", "state-gated-query"}),
        is_layered=bool(str(schema_dict.get("layout_style", "regular")) == "layered" or init_fam in {"hybrid-slice-init", "layered-init"}),
        query_ignores_inf=bool("safe_min(" in qry_txt or "base_sel(0" in qry_txt),
        query_fuses_multi_branch=bool(qry_fam in {"fallback-query", "mixed-aggregator-query", "state-gated-query", "median-query", "avg-query"}),
        update_is_conditional=bool(upd_fam in {"conditional-write", "stateful-write", "rescue-write"} or "updatecount_if(" in upd_txt or "writecount_if(" in upd_txt),
        update_is_read_before_write=bool("query_count(" in upd_txt and ("write_count(" in upd_txt or "update_count(" in upd_txt or "writecount_if(" in upd_txt or "updatecount_if(" in upd_txt)),
        update_is_branch_asymmetric=bool(upd_fam in {"stateful-write", "rescue-write", "conditional-write"}),
        init_is_asymmetric=bool(init_fam in {"asymmetric-init", "slice-hash-init", "hybrid-slice-init", "layered-init"}),
        branch_count=int(schema_dict.get("num_branches", 3) or 3),
        layout_family=str(schema_dict.get("layout_style", "regular")),
        init_family=init_fam or "symmetric-init",
        update_family=upd_fam or "triple-write",
        query_family=qry_fam or "min-query",
    )
    return sig


def _candidate_meta_from_team(evaluator, team):
    schema = _infer_architecture_schema_from_team(team)
    motif = _extract_motif_signature(team, schema=schema)
    schema_dict = asdict(schema)
    motif_dict = asdict(motif)
    init_key = evaluator._canonical_tree_str(team["init_dex"])
    update_key = evaluator._canonical_tree_str(team["update"])
    query_key = evaluator._canonical_tree_str(team["query"])
    schema_hash = _architecture_schema_hash(schema_dict)
    motif_key = _motif_signature_key(motif_dict)
    return CandidateMeta(
        family_tag=str(_team_family_tag(team)),
        key_v1=evaluator._canonical_triplet_key(team["init_dex"], team["update"], team["query"]),
        key_v2=(init_key, update_key, query_key, schema_hash, motif_key),
        repair_dup_key=(init_key, update_key, schema_hash, motif_key),
        architecture_schema=schema_dict,
        motif_signature=motif_dict,
        schema_hash=str(schema_hash),
        motif_key=str(motif_key),
        arch_type=str(schema_dict.get("arch_type", "regular")),
    )


def _innovation_archive_entries(state):
    cur = state.get("innovation_archive", []) if isinstance(state, dict) else []
    return list(cur or [])


def _innovation_archive_append(state, team=None, candidate_meta=None, fit=None, err=None, source=""):
    if not isinstance(state, dict):
        return state
    if candidate_meta is None and team is None:
        return state
    if candidate_meta is None:
        class _MiniEval:
            def _canonical_tree_str(self, tree):
                return str(tree)
            def _canonical_triplet_key(self, a, b, c):
                return (str(a), str(b), str(c))
        candidate_meta = _candidate_meta_from_team(_MiniEval(), team)
    if isinstance(candidate_meta, CandidateMeta):
        meta_dict = asdict(candidate_meta)
    else:
        meta_dict = dict(candidate_meta or {})
    archive = list(state.get("innovation_archive", []))
    key = (str(meta_dict.get("schema_hash", "")), str(meta_dict.get("motif_key", "")))
    found = False
    for rec in archive:
        if (str(rec.get("schema_hash", "")), str(rec.get("motif_key", ""))) == key:
            found = True
            old_err = float(rec.get("err", 1e18))
            if err is not None and float(err) <= old_err:
                rec.update({
                    "fit": None if fit is None else float(fit),
                    "err": None if err is None else float(err),
                    "family_tag": str(meta_dict.get("family_tag", "")),
                    "arch_type": str(meta_dict.get("arch_type", "regular")),
                    "schema_hash": str(meta_dict.get("schema_hash", "")),
                    "motif_key": str(meta_dict.get("motif_key", "")),
                    "architecture_schema": copy.deepcopy(meta_dict.get("architecture_schema", {})),
                    "motif_signature": copy.deepcopy(meta_dict.get("motif_signature", {})),
                    "source": str(source or rec.get("source", "")),
                })
            break
    if not found:
        archive.append({
            "fit": None if fit is None else float(fit),
            "err": None if err is None else float(err),
            "family_tag": str(meta_dict.get("family_tag", "")),
            "arch_type": str(meta_dict.get("arch_type", "regular")),
            "schema_hash": str(meta_dict.get("schema_hash", "")),
            "motif_key": str(meta_dict.get("motif_key", "")),
            "architecture_schema": copy.deepcopy(meta_dict.get("architecture_schema", {})),
            "motif_signature": copy.deepcopy(meta_dict.get("motif_signature", {})),
            "source": str(source or ""),
        })
    keep_k = max(16, int(state.get("structure_keep_quota", 8)) * 8)
    archive = sorted(archive, key=lambda d: (float(d.get("err", 1e18)), str(d.get("motif_key", ""))))[:keep_k]
    state["innovation_archive"] = archive
    return state


def _motif_histogram_from_state(state):
    hist = Counter()
    if not isinstance(state, dict):
        return hist
    for rec in list(state.get("innovation_archive", []) or []):
        key = str(rec.get("motif_key", ""))
        if key:
            hist[key] += 1
    llm_meta = state.get("llm_meta", [])
    if isinstance(llm_meta, list):
        for rec in llm_meta:
            if isinstance(rec, dict):
                key = str(rec.get("motif_key", ""))
                if key:
                    hist[key] += 1
    pops = state.get("pops", {}) if isinstance(state.get("pops", {}), dict) else {}
    n = len(state.get("fits", [])) if isinstance(state.get("fits", []), list) else 0
    for i in range(n):
        try:
            team = {"init_dex": pops["init_dex"][i], "update": pops["update"][i], "query": pops["query"][i]}
            sig = _extract_motif_signature(team)
            hist[_motif_signature_key(sig)] += 1
        except Exception:
            continue
    return hist


def _mainstream_motif_signature_from_state(state):
    motif_counts = Counter()
    motif_payloads = {}
    if not isinstance(state, dict):
        return None
    pops = state.get("pops", {}) if isinstance(state.get("pops", {}), dict) else {}
    n = len(state.get("fits", [])) if isinstance(state.get("fits", []), list) else 0
    for i in range(n):
        try:
            team = {"init_dex": pops["init_dex"][i], "update": pops["update"][i], "query": pops["query"][i]}
            sig = asdict(_extract_motif_signature(team))
            key = _motif_signature_key(sig)
            motif_counts[key] += 1
            motif_payloads[key] = sig
        except Exception:
            continue
    for rec in list(state.get("innovation_archive", []) or []):
        sig = rec.get("motif_signature", {}) if isinstance(rec, dict) else {}
        if isinstance(sig, dict) and sig:
            key = _motif_signature_key(sig)
            motif_counts[key] += 1
            motif_payloads[key] = sig
    if not motif_counts:
        return None
    best_key = motif_counts.most_common(1)[0][0]
    return motif_payloads.get(best_key)


def _motif_distance(sig_a, sig_b):
    if not isinstance(sig_a, dict) or not isinstance(sig_b, dict):
        return 0.0
    bool_keys = [
        "is_regular", "uses_overflow_state", "has_handoff", "has_sidecar_branch", "is_layered",
        "query_ignores_inf", "query_fuses_multi_branch", "update_is_conditional",
        "update_is_read_before_write", "update_is_branch_asymmetric", "init_is_asymmetric",
    ]
    dist = 0.0
    for k in bool_keys:
        dist += 1.0 if bool(sig_a.get(k, False)) != bool(sig_b.get(k, False)) else 0.0
    for k in ("layout_family", "init_family", "update_family", "query_family"):
        dist += 1.0 if str(sig_a.get(k, "")) != str(sig_b.get(k, "")) else 0.0
    return float(dist)


def _compute_motif_score_terms(team, state, profile=None, novelty_subtype="stable", candidate_meta=None):
    if candidate_meta is None:
        schema = _infer_architecture_schema_from_team(team)
        sig = _extract_motif_signature(team, schema=schema)
        schema_dict = asdict(schema)
        sig_dict = asdict(sig)
        motif_key = _motif_signature_key(sig_dict)
    else:
        schema_dict = dict(candidate_meta.get("architecture_schema", {}) or {})
        sig_dict = dict(candidate_meta.get("motif_signature", {}) or {})
        motif_key = str(candidate_meta.get("motif_key", "") or _motif_signature_key(sig_dict))
    mainstream = _mainstream_motif_signature_from_state(state)
    motif_distance_score = _motif_distance(sig_dict, mainstream) if isinstance(mainstream, dict) and mainstream else 0.0
    hist = _motif_histogram_from_state(state)
    freq = int(hist.get(motif_key, 0))
    innovation_bonus = max(0.0, 4.0 - float(freq))
    arch_type = str(schema_dict.get("arch_type", "regular"))
    if novelty_subtype in {"irregular", "innovation"}:
        if arch_type != "regular":
            innovation_bonus += 2.5
        if bool(sig_dict.get("has_handoff", False)):
            innovation_bonus += 1.5
        if bool(sig_dict.get("uses_overflow_state", False)):
            innovation_bonus += 1.5
        if bool(sig_dict.get("query_fuses_multi_branch", False)):
            innovation_bonus += 1.0
    return {
        "motif_distance_score": float(motif_distance_score),
        "innovation_bonus": float(innovation_bonus),
        "motif_frequency": int(freq),
        "motif_key": str(motif_key),
        "arch_type": arch_type,
        "architecture_schema": schema_dict,
        "motif_signature": sig_dict,
    }


def _validate_irregular_candidate_meta(candidate_meta):
    if isinstance(candidate_meta, CandidateMeta):
        meta = asdict(candidate_meta)
    else:
        meta = dict(candidate_meta or {})
    schema = dict(meta.get("architecture_schema", {}) or {})
    sig = dict(meta.get("motif_signature", {}) or {})
    reasons = []
    if str(schema.get("arch_type", "regular")) == "regular":
        reasons.append("irregular_arch_collapsed_to_regular")
    strength = 0
    for k in ("uses_overflow_state", "has_handoff", "has_sidecar_branch", "is_layered", "query_fuses_multi_branch", "update_is_branch_asymmetric"):
        if bool(sig.get(k, False)):
            strength += 1
    if strength < 2:
        reasons.append("irregular_motif_too_weak")
    if str(schema.get("handoff_policy", "none")) != "none" and not bool(sig.get("has_handoff", False)):
        reasons.append("irregular_handoff_not_materialized")
    return sorted(set(reasons))


def _rebuild_innovation_archive_from_population(state):
    if not isinstance(state, dict):
        return state
    state.setdefault("innovation_archive", [])
    pops = state.get("pops", {}) if isinstance(state.get("pops", {}), dict) else {}
    fits = list(state.get("fits", [])) if isinstance(state.get("fits", []), list) else []
    for i in range(len(fits)):
        try:
            team = {"init_dex": pops["init_dex"][i], "update": pops["update"][i], "query": pops["query"][i]}
            fit, err = fits[i]
            schema = _infer_architecture_schema_from_team(team)
            sig = _extract_motif_signature(team, schema=schema)
            meta = {
                "family_tag": _team_family_tag(team),
                "arch_type": str(asdict(schema).get("arch_type", "regular")),
                "schema_hash": _architecture_schema_hash(asdict(schema)),
                "motif_key": _motif_signature_key(asdict(sig)),
                "architecture_schema": asdict(schema),
                "motif_signature": asdict(sig),
            }
            _innovation_archive_append(state, candidate_meta=meta, fit=fit, err=err, source="population_init")
        except Exception:
            continue
    return state



def _empty_family_histogram():
    return {
        "exact": Counter(),
        "components": {
            "init_dex": Counter(),
            "update": Counter(),
            "query": Counter(),
        },
        "total": 0,
    }


def _spec_family_parts(spec, pset_map=None):
    fam_parts = spec.get("family_parts") if isinstance(spec, dict) else None
    if isinstance(fam_parts, dict):
        return {
            "init_dex": str(fam_parts.get("init_dex", "symmetric-init")),
            "update": str(fam_parts.get("update", "triple-write")),
            "query": str(fam_parts.get("query", "min-query")),
        }
    if isinstance(spec, dict) and all(k in spec for k in ("init_dex", "update", "query")):
        try:
            return _team_family_parts(spec)
        except Exception:
            pass
    if pset_map is not None and isinstance(spec, dict):
        try:
            team = _deserialize_team_spec(spec, pset_map)
            return _team_family_parts(team)
        except Exception:
            pass
    return {
        "init_dex": "symmetric-init",
        "update": "triple-write",
        "query": "min-query",
    }


def _spec_family_tag(spec, pset_map=None):
    if isinstance(spec, dict) and spec.get("family_tag"):
        return str(spec.get("family_tag"))
    parts = _spec_family_parts(spec, pset_map=pset_map)
    return f'{parts["init_dex"]}/{parts["update"]}/{parts["query"]}'


def _rebuild_llm_team_bank_by_family(gp_ctx):
    bank = list(gp_ctx.get("llm_team_bank", []))
    grouped = {}
    for spec in bank:
        tag = _spec_family_tag(spec, pset_map=gp_ctx.get("pset_map"))
        grouped.setdefault(tag, []).append(spec)
    gp_ctx["llm_team_bank_by_family"] = grouped
    return gp_ctx


def _flatten_llm_team_bank_by_family(bank_by_family):
    out = []
    for _, specs in sorted((bank_by_family or {}).items(), key=lambda kv: str(kv[0])):
        out.extend(list(specs or []))
    return out


def _record_team_family(hist, team):
    if not team:
        return hist
    try:
        parts = _team_family_parts(team)
        tag = _team_family_tag(team)
    except Exception:
        return hist
    hist["exact"][str(tag)] += 1
    for which in ("init_dex", "update", "query"):
        hist["components"][which][str(parts[which])] += 1
    hist["total"] += 1
    return hist


def _family_histogram_from_team_specs(team_specs, pset_map=None):
    hist = _empty_family_histogram()
    for spec in list(team_specs or []):
        try:
            if isinstance(spec, dict) and all(k in spec for k in ("init_dex", "update", "query")) and pset_map is None:
                team = spec
            else:
                team = _deserialize_team_spec(spec, pset_map) if pset_map is not None else None
            hist = _record_team_family(hist, team)
        except Exception:
            continue
    return hist


def _family_histogram_from_state(state):
    hist = _empty_family_histogram()
    pops = state.get("pops", {}) if isinstance(state, dict) else {}
    fits = list(state.get("fits", [])) if isinstance(state, dict) else []
    n = len(fits)
    for i in range(n):
        try:
            team = {
                "init_dex": pops["init_dex"][i],
                "update": pops["update"][i],
                "query": pops["query"][i],
            }
            hist = _record_team_family(hist, team)
        except Exception:
            continue
    return hist


def _family_histogram_from_states(island_states):
    hist = _empty_family_histogram()
    for st in list(island_states or []):
        cur = _family_histogram_from_state(st)
        hist["exact"].update(cur["exact"])
        for which in ("init_dex", "update", "query"):
            hist["components"][which].update(cur["components"][which])
        hist["total"] += int(cur.get("total", 0))
    return hist


def _family_summary_lines(family_hist, topn=4):
    if not isinstance(family_hist, dict):
        return []
    exact = family_hist.get("exact", Counter())
    comps = family_hist.get("components", {})
    lines = []
    if exact:
        top_exact = [f"{tag} x{cnt}" for tag, cnt in exact.most_common(max(1, int(topn)))]
        if top_exact:
            lines.append("top_exact: " + ", ".join(top_exact))
    for which in ("init_dex", "update", "query"):
        cur = comps.get(which, Counter())
        if cur:
            top_cur = [f"{tag} x{cnt}" for tag, cnt in cur.most_common(max(1, int(topn)))]
            if top_cur:
                lines.append(f"top_{which}: " + ", ".join(top_cur))
    return lines


def _preferred_family_lists_from_buckets(failure_buckets, prefer_irregular: bool = False):
    out = {
        "init_dex": [],
        "update": [],
        "query": [],
    }
    buckets = [str(x) for x in list(failure_buckets or [])]
    if any(b in {"nonconst_hash", "collision"} for b in buckets):
        out["init_dex"].extend(["slice-hash-init", "asymmetric-init"])
        if prefer_irregular:
            out["init_dex"] = ["hybrid-slice-init", "layered-init"] + out["init_dex"]
    if any(b in {"query_date_zero", "generic_read", "read_error"} for b in buckets):
        out["query"].extend(["median-query", "avg-query", "state-gated-query", "free-query"])
        if prefer_irregular:
            out["query"] = ["fallback-query", "mixed-aggregator-query", "state-gated-query"] + out["query"]
    if any(b in {"bad_write_ctx", "real_write_zero"} for b in buckets):
        out["update"].extend(["read-before-write", "double-write", "mixed-write"])
        if prefer_irregular:
            out["update"] = ["rescue-write", "stateful-write", "conditional-write"] + out["update"]
    return out


def _profile_family_default_order(profile_name: str, prefer_irregular: bool = False):
    prof = str(profile_name or "").strip().lower()
    if prof == "init_explore":
        return {
            "init_dex": ["hybrid-slice-init", "layered-init", "asymmetric-init", "slice-hash-init", "symmetric-init"],
            "update": ["read-before-write", "triple-write", "double-write", "conditional-write", "stateful-write", "rescue-write", "mixed-write"],
            "query": ["median-query", "min-query", "avg-query", "mixed-aggregator-query", "fallback-query", "state-gated-query", "free-query"],
        }
    if prof == "update_explore":
        return {
            "init_dex": ["asymmetric-init", "hybrid-slice-init", "layered-init", "symmetric-init", "slice-hash-init"],
            "update": ["rescue-write", "stateful-write", "conditional-write", "read-before-write", "double-write", "triple-write", "mixed-write"],
            "query": ["median-query", "state-gated-query", "fallback-query", "mixed-aggregator-query", "min-query", "avg-query", "free-query"],
        }
    if prof == "irregular_architecture":
        return {
            "init_dex": ["hybrid-slice-init", "layered-init", "asymmetric-init", "slice-hash-init", "symmetric-init"],
            "update": ["rescue-write", "stateful-write", "conditional-write", "read-before-write", "double-write", "mixed-write", "triple-write"],
            "query": ["fallback-query", "mixed-aggregator-query", "state-gated-query", "median-query", "avg-query", "min-query", "free-query"],
        }
    return {
        "init_dex": ["symmetric-init", "asymmetric-init", "slice-hash-init", "hybrid-slice-init", "layered-init"],
        "update": ["triple-write", "read-before-write", "double-write", "mixed-write", "conditional-write", "stateful-write", "rescue-write"],
        "query": ["min-query", "median-query", "avg-query", "state-gated-query", "free-query", "mixed-aggregator-query", "fallback-query"],
    }


def _profile_novelty_constraints(profile, novelty_subtype: str = "stable"):
    prof = str((profile or {}).get("name", "") or "").strip().lower()
    novelty_subtype = str(novelty_subtype or "stable").strip().lower()
    irregular = novelty_subtype in {"irregular", "innovation"}

    base = {
        "novelty_component_weights": {"init_dex": 1.6, "update": 1.6, "query": 1.6},
        "novelty_distance_weights": {"init_dex": 1.0, "update": 1.0, "query": 1.0},
        "novelty_must_match": [],
        "novelty_keep_stable": [],
        "novelty_min_match_score": 2.4 if not irregular else 1.8,
        "novelty_min_distance_score": 0.9 if not irregular else 0.7,
        "novelty_near_miss_distance_margin": 0.12 if not irregular else 0.18,
        "proposal_focus": [
            "让新颖性落在真正有价值的结构位，而不是所有部件一起乱动。",
            "优先避免把创新预算浪费在脆弱的 update 花样上。",
        ],
    }

    if prof == "init_explore":
        base.update({
            "novelty_component_weights": {"init_dex": 3.8, "update": 1.2, "query": 1.8},
            "novelty_distance_weights": {"init_dex": 3.0, "update": 0.4, "query": 1.2},
            "novelty_must_match": ["init_dex"],
            "novelty_keep_stable": ["update"],
            "novelty_min_match_score": 4.0 if not irregular else 3.0,
            "novelty_min_distance_score": 1.6 if not irregular else 1.1,
            "novelty_near_miss_distance_margin": 0.18 if not irregular else 0.26,
            "proposal_focus": [
                "Primary novelty budget: init_dex。优先做新的索引/布局家族。",
                "Keep update conservative and clean; do not spend most novelty on update tricks.",
                "Query can move, but as a secondary axis behind init_dex.",
            ],
        })
    elif prof == "update_explore":
        base.update({
            "novelty_component_weights": {"init_dex": 1.4, "update": 3.8, "query": 1.4},
            "novelty_distance_weights": {"init_dex": 0.8, "update": 2.8, "query": 0.9},
            "novelty_must_match": ["update"],
            "novelty_keep_stable": ["query"],
            "novelty_min_match_score": 3.3 if not irregular else 2.6,
            "novelty_min_distance_score": 1.0 if not irregular else 0.7,
            "novelty_near_miss_distance_margin": 0.14 if not irregular else 0.20,
            "proposal_focus": [
                "Primary novelty budget: update / handoff / state usage。",
                "Do not spend the main novelty budget on query-only rewrites.",
                "Keep query reliable enough to preserve holdout-lite behavior.",
            ],
        })
    elif prof == "irregular_architecture":
        base.update({
            "novelty_component_weights": {"init_dex": 2.4, "update": 2.8, "query": 2.2},
            "novelty_distance_weights": {"init_dex": 1.8, "update": 2.0, "query": 1.6},
            "novelty_must_match": [],
            "novelty_keep_stable": [],
            "novelty_min_match_score": 2.3 if not irregular else 1.8,
            "novelty_min_distance_score": 1.2 if not irregular else 0.8,
            "novelty_near_miss_distance_margin": 0.20 if not irregular else 0.28,
            "proposal_focus": [
                "Irregular architecture island: favor non-regular motifs built from existing primitives.",
                "Prioritize update/state/handoff and layout roles over query-only novelty.",
                "Allow structure-keeping novelty candidates to survive a bit longer before regularizing them.",
            ],
        })
    else:
        base.update({
            "proposal_focus": [
                "Prefer family-consistent novelty over random structural noise.",
                "Do not spend the novelty budget mainly on unstable update changes.",
            ],
        })
    return base


def _build_family_guidance(profile, family_hist, available_specs=None, failure_buckets=None, prefer_irregular: bool = False):
    family_hist = family_hist if isinstance(family_hist, dict) else _empty_family_histogram()
    allowed_map = dict(profile.get("allowed_family_labels", {})) if isinstance(profile, dict) else {}
    prof_name = str((profile or {}).get("name", "") or "").strip().lower()
    avail_parts = {"init_dex": set(), "update": set(), "query": set()}
    for spec in list(available_specs or []):
        try:
            parts = _spec_family_parts(spec)
            for which in ("init_dex", "update", "query"):
                avail_parts[which].add(str(parts[which]))
        except Exception:
            continue
    preferred = _preferred_family_lists_from_buckets(failure_buckets, prefer_irregular=prefer_irregular)
    target_parts = {}
    defaults = _profile_family_default_order(prof_name, prefer_irregular=prefer_irregular)
    novelty_constraints = _profile_novelty_constraints(profile, novelty_subtype=("irregular" if prefer_irregular else "stable"))
    irregular_order = {
        "init_dex": ["hybrid-slice-init", "layered-init"],
        "update": ["rescue-write", "stateful-write", "conditional-write"],
        "query": ["fallback-query", "mixed-aggregator-query", "state-gated-query"],
    }
    for which in ("init_dex", "update", "query"):
        allowed = list(allowed_map.get(which, []))
        if not allowed:
            allowed = list(defaults[which])
        feasible = [fam for fam in allowed if (not avail_parts[which]) or (fam in avail_parts[which])]
        if feasible:
            allowed = feasible
        comp_counts = family_hist.get("components", {}).get(which, Counter())
        pref_rank = {fam: idx for idx, fam in enumerate(preferred.get(which, []))}
        irregular_rank = {fam: idx for idx, fam in enumerate(irregular_order.get(which, []))}
        default_rank = {fam: idx for idx, fam in enumerate(defaults.get(which, []))}
        allowed = sorted(
            allowed,
            key=lambda fam: (
                0 if fam in pref_rank else 1,
                pref_rank.get(fam, 999),
                default_rank.get(fam, 999),
                0 if (prefer_irregular and fam in irregular_rank) else 1,
                irregular_rank.get(fam, 999),
                int(comp_counts.get(fam, 0)),
                str(fam),
            ),
        )
        target_parts[which] = str(allowed[0])
    target_family_tag = f'{target_parts["init_dex"]}/{target_parts["update"]}/{target_parts["query"]}'
    avoid_exact = [str(tag) for tag, _ in family_hist.get("exact", Counter()).most_common(3)]
    return {
        "profile_name": prof_name,
        "target_family_parts": target_parts,
        "target_family_tag": target_family_tag,
        "avoid_exact_families": avoid_exact,
        "current_family_summary": _family_summary_lines(family_hist, topn=4),
        "failure_buckets": [str(x) for x in list(failure_buckets or [])],
        "prefer_irregular": bool(prefer_irregular),
        "novelty_component_weights": dict(novelty_constraints.get("novelty_component_weights", {})),
        "novelty_distance_weights": dict(novelty_constraints.get("novelty_distance_weights", {})),
        "novelty_must_match": list(novelty_constraints.get("novelty_must_match", [])),
        "novelty_keep_stable": list(novelty_constraints.get("novelty_keep_stable", [])),
        "novelty_min_match_score": float(novelty_constraints.get("novelty_min_match_score", 2.0)),
        "novelty_min_distance_score": float(novelty_constraints.get("novelty_min_distance_score", 1.0)),
        "proposal_focus": list(novelty_constraints.get("proposal_focus", [])),
    }


def _pick_spec_family_injection_sort_key(spec, profile, family_hist=None, pset_map=None, target_parts=None):
    parts = _spec_family_parts(spec, pset_map=pset_map)
    allowed_map = dict(profile.get("allowed_family_labels", {})) if isinstance(profile, dict) else {}
    constraints = _profile_novelty_constraints(profile, novelty_subtype=str((spec or {}).get("novelty_subtype", "stable") or "stable"))
    match_weights = dict(constraints.get("novelty_component_weights", {}))
    keep_stable = {str(x) for x in list(constraints.get("novelty_keep_stable", []))}
    allowed_hits = 0
    target_score = 0.0
    stable_violation = 0
    comp_count = 0
    family_hist = family_hist if isinstance(family_hist, dict) else _empty_family_histogram()
    exact_count = int(family_hist.get("exact", Counter()).get(_spec_family_tag(spec, pset_map=pset_map), 0))
    for which in ("init_dex", "update", "query"):
        fam = str(parts.get(which, ""))
        if fam in set(allowed_map.get(which, set())):
            allowed_hits += 1
        if isinstance(target_parts, dict) and fam == str(target_parts.get(which, "")):
            target_score += float(match_weights.get(which, 1.0))
        elif which in keep_stable:
            stable_violation += 1
        comp_count += int(family_hist.get("components", {}).get(which, Counter()).get(fam, 0))
    return (-float(target_score), stable_violation, -allowed_hits, exact_count, comp_count, str(_spec_family_tag(spec, pset_map=pset_map)))


def _llm_candidate_channel(spec):
    ch = str((spec or {}).get("channel", "") or "").strip().lower()
    if ch in {"repair", "novelty"}:
        return ch
    if ch in {"irregular_novelty", "innovation", "innovative_novelty"}:
        return "novelty"
    edit_mode = str((spec or {}).get("edit_mode", "") or "").strip().lower()
    if edit_mode == "single_tree":
        return "repair"
    return "novelty"


def _llm_candidate_subtype(spec):
    ch = str((spec or {}).get("channel", "") or "").strip().lower()
    if ch in {"irregular_novelty", "innovation", "innovative_novelty"}:
        return "irregular"
    sub = str((spec or {}).get("novelty_subtype", "") or "").strip().lower()
    if sub in {"irregular", "innovation"}:
        return "irregular"
    return "stable"


def _pick_llm_candidate_spec_sort_key(spec, profile, family_hist=None, pset_map=None, target_parts=None):
    channel = _llm_candidate_channel(spec)
    fam_key = _pick_spec_family_injection_sort_key(
        spec,
        profile,
        family_hist=family_hist,
        pset_map=pset_map,
        target_parts=target_parts,
    )
    fit = float((spec or {}).get("fit", 0.0) or 0.0)
    err = float((spec or {}).get("err", 1e18) or 1e18)
    if channel == "novelty":
        return fam_key + (err, -fit, str((spec or {}).get("source", "")))
    return (err, -fit) + fam_key + (str((spec or {}).get("source", "")),)


def _dominant_failure_bucket_from_hints(hints):
    if not isinstance(hints, dict):
        return ""
    buckets = [str(x).strip() for x in list(hints.get("failure_buckets", []))]
    buckets = [b for b in buckets if b]
    return buckets[0] if buckets else ""


def _adaptive_single_tree_target_from_hints(profile, hints, allowed_targets=None, fallback="update"):
    allowed = []
    for x in list(allowed_targets or []):
        sx = str(x).strip().lower()
        if sx in {"init", "init_dex"}:
            sx = "init_dex"
        if sx in {"init_dex", "update", "query"} and sx not in allowed:
            allowed.append(sx)
    if not allowed:
        allowed = ["update", "query", "init_dex"]
    prof_name = str((profile or {}).get("name", "") or "").strip().lower()
    dominant = _dominant_failure_bucket_from_hints(hints)
    target = str(fallback or "update")
    reason = "fallback_default"
    if dominant in {"bad_write_ctx", "real_write_zero"} and "update" in allowed:
        target = "update"
        reason = f"{dominant}_dominates"
    elif dominant in {"query_date_zero", "read_error", "generic_read"} and "query" in allowed:
        target = "query"
        reason = f"{dominant}_dominates"
    elif dominant == "nonconst_path":
        if "query" in allowed:
            target = "query"
            reason = "nonconst_path_prefers_query"
        elif "update" in allowed:
            target = "update"
            reason = "nonconst_path_fallback_update"
    elif dominant in {"nonconst_hash", "collision"}:
        if prof_name == "init_explore" and "init_dex" in allowed:
            target = "init_dex"
            reason = f"{dominant}_dominates_init_explore"
        elif "query" in allowed:
            target = "query"
            reason = f"{dominant}_fallback_query"
        elif "update" in allowed:
            target = "update"
            reason = f"{dominant}_fallback_update"
    if target not in allowed:
        target = allowed[0]
        reason = f"normalize_to_allowed:{reason}"
    return target, reason


def _local_mainstream_family_parts(family_hist):
    family_hist = family_hist if isinstance(family_hist, dict) else _empty_family_histogram()
    parts = {}
    for which in ("init_dex", "update", "query"):
        ctr = family_hist.get("components", {}).get(which, Counter())
        if ctr:
            parts[which] = str(ctr.most_common(1)[0][0])
        else:
            parts[which] = ""
    return parts


def _team_or_spec_family_parts(team_or_spec, pset_map=None):
    if isinstance(team_or_spec, dict) and all(k in team_or_spec for k in ("init_dex", "update", "query")):
        try:
            return _team_family_parts(team_or_spec)
        except Exception:
            pass
    return _spec_family_parts(team_or_spec, pset_map=pset_map)


def _novelty_family_metrics(team_or_spec, family_hist, target_parts, pset_map=None, profile=None, guidance=None, novelty_subtype="stable"):
    cand_parts = _team_or_spec_family_parts(team_or_spec, pset_map=pset_map)
    mainstream = _local_mainstream_family_parts(family_hist)
    constraints = guidance if isinstance(guidance, dict) else _profile_novelty_constraints(profile, novelty_subtype=novelty_subtype)
    match_weights = dict(constraints.get("novelty_component_weights", {}))
    distance_weights = dict(constraints.get("novelty_distance_weights", {}))
    must_match = [str(x) for x in list(constraints.get("novelty_must_match", [])) if str(x)]
    keep_stable = [str(x) for x in list(constraints.get("novelty_keep_stable", [])) if str(x)]

    match_count = 0
    mainstream_distance = 0
    target_match_score = 0.0
    mainstream_distance_score = 0.0
    part_match = {}
    part_distance = {}
    stable_violations = []

    for which in ("init_dex", "update", "query"):
        fam = str(cand_parts.get(which, ""))
        target_fam = str(target_parts.get(which, "")) if isinstance(target_parts, dict) else ""
        mainstream_fam = str(mainstream.get(which, ""))
        matched = bool(fam and target_fam and fam == target_fam)
        off_mainstream = bool(fam and mainstream_fam and fam != mainstream_fam)
        part_match[which] = matched
        part_distance[which] = off_mainstream
        if matched:
            match_count += 1
            target_match_score += float(match_weights.get(which, 1.0))
        if off_mainstream:
            mainstream_distance += 1
            mainstream_distance_score += float(distance_weights.get(which, 1.0))
        if which in keep_stable and target_fam and fam != target_fam:
            stable_violations.append(which)

    missing_must = [which for which in must_match if not bool(part_match.get(which, False))]
    return {
        "parts": cand_parts,
        "target_match": int(match_count),
        "target_match_score": float(target_match_score),
        "mainstream_distance": int(mainstream_distance),
        "mainstream_distance_score": float(mainstream_distance_score),
        "mainstream_parts": mainstream,
        "part_match": part_match,
        "part_distance": part_distance,
        "must_match_missing": missing_must,
        "keep_stable_violations": stable_violations,
        "must_match": must_match,
        "keep_stable": keep_stable,
    }


def _novelty_family_gate(team_or_spec, family_hist, target_parts, pset_map=None, min_match=2, min_distance=1, profile=None, guidance=None, novelty_subtype="stable"):
    metrics = _novelty_family_metrics(
        team_or_spec,
        family_hist,
        target_parts,
        pset_map=pset_map,
        profile=profile,
        guidance=guidance,
        novelty_subtype=novelty_subtype,
    )
    constraints = guidance if isinstance(guidance, dict) else _profile_novelty_constraints(profile, novelty_subtype=novelty_subtype)
    reasons = []
    missing = list(metrics.get("must_match_missing", []))
    if missing:
        reasons.append("novelty_primary_part_miss:" + ",".join(missing))
    min_match_score = max(float(min_match), float(constraints.get("novelty_min_match_score", float(min_match))))
    min_distance_score = max(float(min_distance), float(constraints.get("novelty_min_distance_score", float(min_distance))))
    near_miss_distance_margin = max(0.0, float(constraints.get("novelty_near_miss_distance_margin", 0.0)))
    target_match_score = float(metrics.get("target_match_score", 0.0))
    mainstream_distance_score = float(metrics.get("mainstream_distance_score", 0.0))
    distance_near_miss_used = False
    if float(target_match_score) < float(min_match_score):
        reasons.append(f"novelty_family_match_score_too_low:{float(target_match_score):.2f}<{float(min_match_score):.2f}")
    if float(mainstream_distance_score) < float(min_distance_score):
        if (not missing) and float(target_match_score) >= float(min_match_score) and float(mainstream_distance_score + near_miss_distance_margin) >= float(min_distance_score):
            distance_near_miss_used = True
        else:
            reasons.append(f"novelty_family_distance_score_too_low:{float(mainstream_distance_score):.2f}<{float(min_distance_score):.2f}")
    metrics["distance_near_miss_used"] = bool(distance_near_miss_used)
    metrics["near_miss_distance_margin"] = float(near_miss_distance_margin)
    metrics["effective_min_match_score"] = float(min_match_score)
    metrics["effective_min_distance_score"] = float(min_distance_score)
    return (len(reasons) == 0), metrics, reasons


def _novelty_quality_gate(chk_fit, chk_err, island_best_fit, island_best_err, cfg):
    err_mult = max(1.0, float(cfg.get("llm_novelty_err_mult", 3.0)))
    err_cap = max(1.0, float(cfg.get("llm_novelty_err_cap", 2000.0)))
    fit_delta = max(0.0, float(cfg.get("llm_novelty_fit_delta", 0.01)))
    err_threshold = min(err_cap, err_mult * max(1.0, float(island_best_err)))
    if float(chk_err) <= float(err_threshold):
        return True, f"err_gate:{float(chk_err):.2f}<={float(err_threshold):.2f}"
    if float(chk_fit) >= float(island_best_fit) - float(fit_delta):
        return True, f"fit_gate:{float(chk_fit):.6f}>={float(island_best_fit) - float(fit_delta):.6f}"
    return False, f"novelty_quality_fail err={float(chk_err):.2f} thr={float(err_threshold):.2f} fit={float(chk_fit):.6f} ref={float(island_best_fit):.6f}"


def _empty_llm_novelty_stats():
    return {
        "proposed": 0,
        "validated": 0,
        "pass_family": 0,
        "pass_saturation": 0,
        "pass_quality": 0,
        "pass_holdout": 0,
        "pass_score": 0,
        "injected": 0,
        "rejected_reasons": Counter(),
        "accepted_families": Counter(),
    }


def _ensure_llm_novelty_stats_in_state(state):
    cur = state.get("llm_novelty_stats")
    if not isinstance(cur, dict):
        cur = _empty_llm_novelty_stats()
        state["llm_novelty_stats"] = cur
    if not isinstance(cur.get("rejected_reasons"), Counter):
        cur["rejected_reasons"] = Counter(dict(cur.get("rejected_reasons", {})))
    if not isinstance(cur.get("accepted_families"), Counter):
        cur["accepted_families"] = Counter(dict(cur.get("accepted_families", {})))
    for k in ("proposed", "validated", "pass_family", "pass_saturation", "pass_quality", "pass_holdout", "pass_score", "injected"):
        cur[k] = int(cur.get(k, 0))
    return cur


def _record_llm_novelty_reject(state, reason):
    stats = _ensure_llm_novelty_stats_in_state(state)
    stats["rejected_reasons"][str(reason)] += 1
    return stats


def _aggregate_llm_novelty_stats(island_states):
    out = _empty_llm_novelty_stats()
    for st in list(island_states or []):
        cur = _ensure_llm_novelty_stats_in_state(st)
        for k in ("proposed", "validated", "pass_family", "pass_saturation", "pass_quality", "pass_holdout", "pass_score", "injected"):
            out[k] += int(cur.get(k, 0))
        out["rejected_reasons"].update(cur.get("rejected_reasons", Counter()))
        out["accepted_families"].update(cur.get("accepted_families", Counter()))
    return out


def _derive_llm_novelty_holdout_fixed_stream_path(cfg, pkts, dataset_seed):
    base = str(cfg.get("fixed_stream_path", "") or "").strip()
    if not base:
        return ""
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".npy"
    return f"{root}__llm_novelty_holdout_{int(pkts)}pk_seed{int(dataset_seed) & 0xFFFFFFFF}{ext}"


def _build_llm_novelty_holdout_evaluator_from_cfg(cfg):
    base_pkts = max(1, int(cfg.get("pkts", 1)))
    holdout_pkts = int(cfg.get("llm_novelty_holdout_pkts", min(base_pkts, max(1200, int(round(base_pkts * 0.30))))))
    holdout_pkts = max(1, holdout_pkts)
    holdout_seed = int(cfg.get("llm_novelty_holdout_seed", int(cfg.get("dataset_seed", 0)) + 99173)) & 0xFFFFFFFF
    ev = CMSketchEvaluator(
        dataset_root=cfg["dataset_root"],
        pkts=holdout_pkts,
        max_files=int(cfg.get("files", 1)),
        start=int(cfg.get("start", 0)),
        shuffle=bool(cfg.get("shuffle", False)),
        seed=holdout_seed,
        dataset_mode=str(cfg.get("dataset_mode", "real")),
        proxy_mode=str(cfg.get("proxy_mode", "proxy_balanced")),
        proxy_pool_mul=int(cfg.get("proxy_pool_mul", 8)),
        proxy_min_u=int(cfg.get("proxy_min_u", 2500)),
        hard_case_enabled=False,
        fixed_stream_path=_derive_llm_novelty_holdout_fixed_stream_path(cfg, holdout_pkts, holdout_seed),
    )
    try:
        ev.E0 = max(1.0, float(cfg.get("e0_value", 1.0)))
    except Exception:
        ev.E0 = 1.0
    try:
        ev.kpart_query_limits = [min(48, int(ev.kpart_query_limits[0] or 48)), min(96, int(ev.kpart_query_limits[1] or 96)), 160]
        ev.fec_probe_update_n = min(96, len(ev.test_data))
        ev.fec_probe_present_n = min(24, len(ev.expected_freq))
        ev.fec_probe_absent_n = min(8, int(ev.fec_probe_absent_n))
        ev.fec_absent_keys = ev._build_fec_absent_keys(ev.fec_probe_absent_n)
    except Exception:
        pass
    return ev


def _team_failure_bucket_summary(evaluator, team):
    try:
        init_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["init_dex"]))
        update_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["update"]))
        query_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["query"]))
        init_pat = evaluator._ast_pattern_summary("init", init_ast)
        update_pat = evaluator._ast_pattern_summary("update", update_ast)
        query_pat = evaluator._ast_pattern_summary("query", query_ast)
        update_eff = evaluator._ast_effect_summary(update_ast)
        query_eff = evaluator._ast_effect_summary(query_ast)
        buckets = evaluator._infer_failure_buckets(init_pat, update_pat, query_pat, update_eff, query_eff)
        return {
            "failure_buckets": [str(x) for x in list(buckets or []) if str(x)],
            "pattern": {
                "init": init_pat,
                "update": update_pat,
                "query": query_pat,
            },
            "effect": {
                "update": update_eff,
                "query": query_eff,
            },
        }
    except Exception as e:
        return {
            "failure_buckets": [f"failure_summary_error:{e}"],
            "pattern": {},
            "effect": {},
        }


def _novelty_saturation_gate(team_or_spec, family_hist, pset_map=None, exact_cap=1, component_cap=12, min_distance_if_saturated=2):
    family_hist = family_hist if isinstance(family_hist, dict) else _empty_family_histogram()
    cand_parts = _team_or_spec_family_parts(team_or_spec, pset_map=pset_map)
    mainstream = _local_mainstream_family_parts(family_hist)
    mainstream_distance = 0
    component_count = 0
    for which in ("init_dex", "update", "query"):
        fam = str(cand_parts.get(which, ""))
        component_count += int(family_hist.get("components", {}).get(which, Counter()).get(fam, 0))
        if fam and fam != str(mainstream.get(which, "")):
            mainstream_distance += 1
    try:
        tag = _team_family_tag(team_or_spec) if isinstance(team_or_spec, dict) and all(k in team_or_spec for k in ("init_dex", "update", "query")) else _spec_family_tag(team_or_spec, pset_map=pset_map)
    except Exception:
        tag = ""
    exact_count = int(family_hist.get("exact", Counter()).get(str(tag), 0))
    reasons = []
    if exact_count >= int(exact_cap):
        reasons.append(f"novelty_exact_family_saturated:{exact_count}>={int(exact_cap)}")
    if component_count >= int(component_cap) and int(mainstream_distance) < int(min_distance_if_saturated):
        reasons.append(f"novelty_component_saturation:{component_count}>={int(component_cap)}")
    return (len(reasons) == 0), {
        "tag": str(tag),
        "parts": cand_parts,
        "mainstream_parts": mainstream,
        "mainstream_distance": int(mainstream_distance),
        "exact_count": int(exact_count),
        "component_count": int(component_count),
    }, reasons


def _novelty_holdout_gate(holdout_err, chk_err, island_best_err, cfg):
    holdout_err = float(holdout_err)
    chk_err = float(chk_err)
    island_best_err = max(1.0, float(island_best_err))
    err_mult = max(1.0, float(cfg.get("llm_novelty_holdout_err_mult", 2.5)))
    err_cap = max(1.0, float(cfg.get("llm_novelty_holdout_err_cap", 1200.0)))
    delta_cap = max(0.0, float(cfg.get("llm_novelty_holdout_delta_cap", 180.0)))
    threshold = min(err_cap, max(chk_err + delta_cap, island_best_err * err_mult))
    if holdout_err <= threshold:
        return True, f"holdout_gate:{holdout_err:.2f}<={threshold:.2f}"
    return False, f"novelty_holdout_fail err={holdout_err:.2f} thr={threshold:.2f} chk={chk_err:.2f} ref={island_best_err:.2f}"


def _compute_novelty_score(chk_fit, chk_err, island_best_fit, island_best_err, family_metrics, saturation_metrics, failure_summary, holdout_err=None, profile=None, guidance=None, novelty_subtype="stable"):
    target_match = int((family_metrics or {}).get("target_match", 0))
    target_match_score = float((family_metrics or {}).get("target_match_score", float(target_match)))
    mainstream_distance = int((family_metrics or {}).get("mainstream_distance", 0))
    mainstream_distance_score = float((family_metrics or {}).get("mainstream_distance_score", float(mainstream_distance)))
    exact_count = int((saturation_metrics or {}).get("exact_count", 0))
    component_count = int((saturation_metrics or {}).get("component_count", 0))
    failure_buckets = [str(x) for x in list((failure_summary or {}).get("failure_buckets", [])) if str(x) and str(x) != "generic"]
    must_match_missing = [str(x) for x in list((family_metrics or {}).get("must_match_missing", [])) if str(x)]
    keep_stable_violations = [str(x) for x in list((family_metrics or {}).get("keep_stable_violations", [])) if str(x)]
    island_best_err = max(1.0, float(island_best_err))
    stage_rel = max(0.0, float(chk_err) / island_best_err)

    score = 9.0 * float(target_match_score) + 7.0 * float(mainstream_distance_score)
    score -= 14.0 * float(exact_count)
    score -= min(18.0, 0.75 * float(component_count))
    score -= 6.0 * float(len(failure_buckets))
    score -= 5.0 * float(len(keep_stable_violations))
    score -= 6.0 * float(len(must_match_missing))
    score -= max(0.0, stage_rel - 1.0) * 8.0
    if not must_match_missing and float(target_match_score) > 0.0:
        score += 2.0
    if holdout_err is not None:
        hold_rel = max(0.0, float(holdout_err) / island_best_err)
        score -= max(0.0, hold_rel - 1.0) * 10.0
        if float(holdout_err) <= float(chk_err):
            score += 3.0
    if float(chk_fit) >= float(island_best_fit):
        score += 2.0
    return {
        "score": float(score),
        "target_match": int(target_match),
        "target_match_score": float(target_match_score),
        "mainstream_distance": int(mainstream_distance),
        "mainstream_distance_score": float(mainstream_distance_score),
        "exact_count": int(exact_count),
        "component_count": int(component_count),
        "failure_bucket_count": int(len(failure_buckets)),
        "failure_buckets": failure_buckets,
        "must_match_missing": must_match_missing,
        "keep_stable_violations": keep_stable_violations,
        "holdout_err": None if holdout_err is None else float(holdout_err),
    }


def _compute_irregular_novelty_bonus(team, family_metrics=None, failure_summary=None):
    fam_parts = _team_family_parts(team)
    init_fam = str(fam_parts.get("init_dex", ""))
    upd_fam = str(fam_parts.get("update", ""))
    qry_fam = str(fam_parts.get("query", ""))
    init_txt = _tree_text_for_family(team.get("init_dex"))
    upd_txt = _tree_text_for_family(team.get("update"))
    qry_txt = _tree_text_for_family(team.get("query"))
    bonus = 0.0
    reasons = []

    if _is_irregular_family_part("init_dex", init_fam):
        bonus += 8.0
        reasons.append(f"irregular_init:{init_fam}")
    if _is_irregular_family_part("update", upd_fam):
        bonus += 10.0
        reasons.append(f"irregular_update:{upd_fam}")
    if _is_irregular_family_part("query", qry_fam):
        bonus += 10.0
        reasons.append(f"irregular_query:{qry_fam}")
    if "update_state(" in upd_txt or "writestate_if(" in upd_txt:
        bonus += 6.0
        reasons.append("stateful_control")
    if "cnt_rdstate(" in qry_txt:
        bonus += 5.0
        reasons.append("state_read")
    if (("base_sel(" in qry_txt) and (("safe_min(" in qry_txt) or ("safe_max(" in qry_txt) or ("median3(" in qry_txt))) or ((qry_txt.count("safe_min(") + qry_txt.count("safe_max(") + qry_txt.count("median3(")) >= 2):
        bonus += 5.0
        reasons.append("mixed_aggregator")
    if ("hash_on_slice(" in init_txt and ("hash_salt(" in init_txt or "select_hash(" in init_txt)) or ("hash_salt(hash_salt(" in init_txt):
        bonus += 4.0
        reasons.append("layered_or_hybrid_hash")
    mainstream_distance = int((family_metrics or {}).get("mainstream_distance", 0))
    if mainstream_distance >= 2:
        bonus += 3.0
        reasons.append("mainstream_distance>=2")
    failure_buckets = [str(x) for x in list((failure_summary or {}).get("failure_buckets", [])) if str(x) and str(x) != "generic"]
    if len(failure_buckets) >= 3:
        bonus -= 4.0
        reasons.append("failure_penalty")
    return {
        "bonus": float(bonus),
        "reasons": reasons,
        "family_parts": fam_parts,
    }


def _filter_bank_by_family(which, bank, allowed_family_labels=None):
    bank = list(bank or [])
    allowed = set(allowed_family_labels or [])
    if not bank or not allowed:
        return bank
    out = []
    for ind in bank:
        try:
            if _component_family_of_tree(which, ind) in allowed:
                out.append(ind)
        except Exception:
            continue
    return out or bank


def _sample_component_from_bank(ctx, which, bank_name: str, allowed_family_labels=None):
    bank = ctx.get(bank_name, {}).get(which, []) if isinstance(ctx.get(bank_name), dict) else []
    bank = _filter_bank_by_family(which, bank, allowed_family_labels)
    if bank:
        return ctx["toolboxes"][which].clone(random.choice(bank))
    return None


def _is_llm_seed_mode_enabled(cfg) -> bool:
    return bool(cfg.get("llm_enable", False)) and str(cfg.get("llm_mode", "none")) in {"seeds", "both"}


def _choose_weighted_key(weight_map, default_key="query"):
    items = [(str(k), max(0.0, float(v))) for k, v in dict(weight_map or {}).items()]
    items = [(k, v) for k, v in items if v > 0.0]
    if not items:
        return str(default_key)
    total = sum(v for _, v in items)
    r = random.random() * total
    acc = 0.0
    for k, v in items:
        acc += v
        if acc >= r:
            return k
    return items[-1][0]


def _get_island_profile(cfg, island_idx: int):
    llm_on = _is_llm_seed_mode_enabled(cfg)
    base_llm = 0.05 if llm_on else 0.0
    profiles = [
        {
            "name": "baseline",
            "role": "baseline_exploit",
            "structure_keep_quota": 1,
            "init_p_skeleton": 0.72,
            "init_p_seed": 0.23,
            "init_p_llm_seed": base_llm,
            "allowed_family_labels": {
                "init_dex": {"symmetric-init", "asymmetric-init"},
                "update": {"triple-write", "read-before-write"},
                "query": {"min-query", "median-query"},
            },
            "innovation_family_labels": {
                "init_dex": {"hybrid-slice-init", "layered-init"},
                "update": {"conditional-write", "stateful-write", "rescue-write"},
                "query": {"mixed-aggregator-query", "fallback-query", "state-gated-query"},
            },
            "mutation_weights": {"init_dex": 0.25, "update": 0.35, "query": 0.40},
            "reset_weights": {"init_dex": 0.25, "update": 0.35, "query": 0.40},
            "family_jump_prob": 0.06,
            "family_jump_weights": {"init_dex": 0.20, "update": 0.20, "query": 0.60},
        },
        {
            "name": "init_explore",
            "role": "layout_innovation",
            "structure_keep_quota": 2,
            "init_p_skeleton": 0.55,
            "init_p_seed": 0.30,
            "init_p_llm_seed": 0.15 if llm_on else 0.0,
            "allowed_family_labels": {
                "init_dex": {"asymmetric-init", "slice-hash-init", "hybrid-slice-init", "layered-init"},
                "update": {"triple-write", "read-before-write", "conditional-write", "stateful-write", "rescue-write"},
                "query": {"min-query", "median-query", "avg-query", "mixed-aggregator-query", "fallback-query", "state-gated-query"},
            },
            "innovation_family_labels": {
                "init_dex": {"hybrid-slice-init", "layered-init"},
                "update": {"conditional-write", "stateful-write", "rescue-write"},
                "query": {"mixed-aggregator-query", "fallback-query", "state-gated-query"},
            },
            "mutation_weights": {"init_dex": 0.62, "update": 0.18, "query": 0.20},
            "reset_weights": {"init_dex": 0.65, "update": 0.15, "query": 0.20},
            "family_jump_prob": 0.16,
            "family_jump_weights": {"init_dex": 0.75, "update": 0.05, "query": 0.20},
        },
        {
            "name": "update_explore",
            "role": "update_handoff_innovation",
            "structure_keep_quota": 2,
            "init_p_skeleton": 0.58,
            "init_p_seed": 0.27,
            "init_p_llm_seed": 0.15 if llm_on else 0.0,
            "allowed_family_labels": {
                "init_dex": {"symmetric-init", "asymmetric-init", "hybrid-slice-init", "layered-init"},
                "update": {"read-before-write", "double-write", "mixed-write", "triple-write", "conditional-write", "stateful-write", "rescue-write"},
                "query": {"min-query", "median-query", "avg-query", "mixed-aggregator-query", "fallback-query", "state-gated-query"},
            },
            "innovation_family_labels": {
                "init_dex": {"hybrid-slice-init", "layered-init"},
                "update": {"conditional-write", "stateful-write", "rescue-write"},
                "query": {"mixed-aggregator-query", "fallback-query", "state-gated-query"},
            },
            "mutation_weights": {"init_dex": 0.15, "update": 0.70, "query": 0.15},
            "reset_weights": {"init_dex": 0.15, "update": 0.70, "query": 0.15},
            "family_jump_prob": 0.18,
            "family_jump_weights": {"init_dex": 0.10, "update": 0.80, "query": 0.10},
        },
        {
            "name": "irregular_architecture",
            "role": "irregular_architecture",
            "structure_keep_quota": 3,
            "init_p_skeleton": 0.58,
            "init_p_seed": 0.27,
            "init_p_llm_seed": 0.18 if llm_on else 0.0,
            "allowed_family_labels": {
                "init_dex": {"hybrid-slice-init", "layered-init", "asymmetric-init", "slice-hash-init", "symmetric-init"},
                "update": {"conditional-write", "stateful-write", "rescue-write", "read-before-write", "double-write", "mixed-write", "triple-write"},
                "query": {"fallback-query", "mixed-aggregator-query", "state-gated-query", "median-query", "avg-query", "free-query", "min-query"},
            },
            "innovation_family_labels": {
                "init_dex": {"hybrid-slice-init", "layered-init"},
                "update": {"conditional-write", "stateful-write", "rescue-write"},
                "query": {"fallback-query", "mixed-aggregator-query", "state-gated-query"},
            },
            "mutation_weights": {"init_dex": 0.32, "update": 0.38, "query": 0.30},
            "reset_weights": {"init_dex": 0.28, "update": 0.40, "query": 0.32},
            "family_jump_prob": 0.24,
            "family_jump_weights": {"init_dex": 0.35, "update": 0.40, "query": 0.25},
        },
    ]
    prof = copy.deepcopy(profiles[int(island_idx) % len(profiles)])
    prof["island_idx"] = int(island_idx)
    return prof


def _pick_spec_family_match_score(spec, profile, pset_map):
    try:
        fam_parts = spec.get("family_parts")
        if not isinstance(fam_parts, dict):
            team = _deserialize_team_spec(spec, pset_map)
            fam_parts = _team_family_parts(team)
        score = 0
        for which in ("init_dex", "update", "query"):
            allowed = set(profile.get("allowed_family_labels", {}).get(which, set()))
            if fam_parts.get(which) in allowed:
                score += 1
        return score
    except Exception:
        return 0


def _init_individual_from_ctx(ctx, which, p_skeleton=0.70, p_seed=0.20, p_llm_seed=0.0, allowed_family_labels=None):
    p_skeleton, p_seed, p_llm_seed = _normalize_init_probs(
        p_skeleton=p_skeleton,
        p_seed=p_seed,
        p_llm_seed=p_llm_seed,
    )

    r = random.random()
    if r < p_skeleton:
        cand = _sample_component_from_bank(ctx, which, "skeleton_bank", allowed_family_labels)
        if cand is not None:
            return cand
        return _skeleton_individual_from_ctx(ctx, which)
    if r < p_skeleton + p_seed:
        cand = _sample_component_from_bank(ctx, which, "seed_bank", allowed_family_labels)
        if cand is not None:
            return cand
        return _seeded_individual_from_ctx(ctx, which)
    if r < p_skeleton + p_seed + p_llm_seed:
        cand = _sample_component_from_bank(ctx, which, "llm_seed_bank", allowed_family_labels)
        if cand is not None:
            return cand
        return _llm_seeded_individual_from_ctx(ctx, which)
    return ctx["toolboxes"][which].individual()


def _llm_team_from_ctx(ctx, target_family_tag: str = ""):
    bank_by_family = ctx.get("llm_team_bank_by_family", {}) if isinstance(ctx, dict) else {}
    bank = list(ctx.get("llm_team_bank", [])) if isinstance(ctx, dict) else []
    if target_family_tag and target_family_tag in bank_by_family:
        bank = list(bank_by_family.get(target_family_tag, []))
    if not bank:
        bank = _flatten_llm_team_bank_by_family(bank_by_family)
    if not bank:
        return None
    team = random.choice(bank)
    return {
        "init_dex": ctx["toolboxes"]["init_dex"].clone(team["init_dex"]),
        "update": ctx["toolboxes"]["update"].clone(team["update"]),
        "query": ctx["toolboxes"]["query"].clone(team["query"]),
        "name": str(team.get("name", "llm_team")),
        "family_tag": str(team.get("family_tag", _spec_family_tag(team))),
    }


def _prepare_llm_team_bank_for_cfg(cfg):
    gp_ctx = _build_gp_context(max_size=cfg["max_size"])
    gp_ctx = _populate_llm_seed_bank_from_cfg(gp_ctx, cfg)
    evaluator = _make_evaluator_from_cfg(cfg)
    gp_ctx = _filter_llm_team_bank_with_evaluator(gp_ctx, evaluator, cfg)
    gp_ctx = _rebuild_llm_team_bank_by_family(gp_ctx)
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
        fam_parts = spec.get("family_parts") if isinstance(spec.get("family_parts"), dict) else _team_family_parts(spec)
        kept.append({
            "name": str(spec.get("name", f"team_{len(kept)}")),
            "init_dex": spec["init_dex"],
            "update": spec["update"],
            "query": spec["query"],
            "fitness": float(fit),
            "error": float(err),
            "case_vec": tuple(float(x) for x in case_vec),
            "family_tag": str(spec.get("family_tag", _team_family_tag(spec))),
            "family_parts": fam_parts,
        })

    gp_ctx["llm_team_bank"] = kept
    gp_ctx = _rebuild_llm_team_bank_by_family(gp_ctx)
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
    Supports both team proposals and single_tree edit proposals.
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

    def _build_report(self, evaluator, team=None, fit=None, err=None, case_vec=None, hard_cases=None, extra_prompt_hints=None, family_guidance=None):
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

        base_family_tag = _team_family_tag(team) if isinstance(team, dict) and team is not None else ""
        candidate_meta = {}
        if isinstance(team, dict) and team is not None:
            try:
                candidate_meta = asdict(_candidate_meta_from_team(evaluator, team))
            except Exception:
                candidate_meta = {}
        return {
            "family_tag": base_family_tag,
            "architecture_schema": copy.deepcopy(candidate_meta.get("architecture_schema", {})),
            "motif_signature": copy.deepcopy(candidate_meta.get("motif_signature", {})),
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
            "family_guidance": copy.deepcopy(family_guidance) if isinstance(family_guidance, dict) else {},
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

        add_unique = _append_unique_str

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

    def _prefer_single_tree_for_phase(self, phase: str, report: dict = None) -> bool:
        if isinstance(report, dict) and report.get("_force_single_tree") is not None:
            return bool(report.get("_force_single_tree"))
        mode = str(self.cfg.get("llm_single_tree_mode", "stagnation") or "stagnation").strip().lower()
        if mode == "none":
            return False
        if mode == "both":
            return True
        if mode == "stagnation":
            return str(phase) == "stagnation"
        if mode == "seeds":
            return str(phase) == "seed"
        return str(phase) == "stagnation"

    def _default_single_tree_target(self, phase: str, report: dict):
        forced = str((report or {}).get("_force_single_tree_target", "") or "").strip().lower() if isinstance(report, dict) else ""
        target = forced or str(self.cfg.get("llm_single_tree_target", "update") or "update").strip().lower()
        if target in {"init", "init_dex"}:
            target = "init_dex"
        if target not in {"init_dex", "update", "query"}:
            target = "update"
        allowed = list(report.get("target_funcs", [])) if isinstance(report, dict) else []
        norm_allowed = []
        for x in allowed:
            sx = str(x).strip().lower()
            if sx in {"init", "init_dex"}:
                sx = "init_dex"
            if sx in {"init_dex", "update", "query"} and sx not in norm_allowed:
                norm_allowed.append(sx)

        if isinstance(report, dict) and not forced:
            hints = report.get("prompt_hints", {}) or {}
            buckets = [str(x) for x in list(hints.get("failure_buckets", []))]
            preferred = None
            if any(b in {"bad_write_ctx", "real_write_zero"} for b in buckets):
                preferred = "update"
            elif any(b in {"query_date_zero", "read_error", "generic_read"} for b in buckets):
                preferred = "query"
            elif any(b in {"nonconst_hash", "collision"} for b in buckets):
                preferred = "init_dex"
            elif any(b in {"nonconst_path"} for b in buckets):
                preferred = "query" if "query" in norm_allowed else "update"
            if preferred and ((not norm_allowed) or preferred in norm_allowed):
                target = preferred

        if norm_allowed and target not in norm_allowed:
            if "update" in norm_allowed:
                target = "update"
            else:
                target = norm_allowed[0]
        return target

    def _build_prompt(self, phase: str, report: dict, repair_feedback: str = ""):
        use_single_tree = self._prefer_single_tree_for_phase(phase, report=report)
        single_target = self._default_single_tree_target(phase, report)
        if use_single_tree:
            schema = {
                "mode": "single_tree",
                "target": single_target,
                "expr": "<expr>",
                "rationale": "<short>"
            }
        else:
            schema = {
                "mode": "team",
                "architecture_hint": {
                    "arch_type": "regular|pyramid|diamond|overflow|elastic|hybrid",
                    "handoff_policy": "none|overflow_to_sidecar|layered_correction|branch_fallback",
                    "query_fusion": "min|median|base_sel|state_gated_min|branch_conditional",
                    "state_usage": "none|overflow_state|branch_state|mixed",
                    "layout_style": "regular|layered|asymmetric_dual_path|sidecar_heavy"
                },
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
        candidate_channel = str((report or {}).get("_candidate_channel", "") or "").strip().lower() if isinstance(report, dict) else ""
        canonical = (report or {}).get("canonical", {}) if isinstance(report, dict) else {}

        lines = [
            "Return JSON only. No markdown, no explanations, no code fences.",
            "Output exactly ONE JSON object with this schema:",
            json.dumps(schema, ensure_ascii=False),
            "Do not output an array.",
            "Do not output a top-level key named candidates.",
            f"phase={phase}",
            f"target_funcs={json.dumps(report.get('target_funcs', []), ensure_ascii=False)}",
            "Quality priority order:",
            "1) directly parseable DEAP expression strings, 2) satisfy sketch semantics, 3) avoid canonical duplicates, 4) then pursue novelty or local gain.",
            "Hard requirements:",
            "- init_dex must stay index-only and must not perform counter/state read/write.",
            "- update must keep real counter writes in a clean update context.",
            "- query must stay read-only and must not write.",
        ]

        if use_single_tree:
            lines.extend([
                f"Single-tree edit mode is enabled. Edit ONLY the target function: {single_target}.",
                "Keep the other two functions unchanged; do not rewrite the whole team.",
                "The expr field must be one single DEAP expression string for the target function only.",
                "Prefer small, local repairs over complete rewrites.",
            ])
        else:
            lines.extend([
                "Each field init_dex/update/query must be a single DEAP expression string, not Python code.",
                "You are proposing a sketch team, not full Python source code.",
                "Do not paraphrase the current canonical team with cosmetic rewrites only.",
            ])

        if candidate_channel == "repair":
            lines.append("Repair channel: first fix the current dominant failure buckets; keep the overall family stable unless a failure bucket clearly points to a different component.")
        elif candidate_channel == "novelty":
            lines.append("Novelty channel: match the target family, avoid saturated exact families, and make one principal structural change instead of three noisy changes.")
        elif candidate_channel in {"irregular_novelty", "innovation", "innovative_novelty"}:
            lines.append("Irregular novelty channel: propose a genuinely different but still legal sketch family; keep the output parseable and do not collapse back to a trivial mainstream family.")

        lines.extend([
            "Architecture-first guidance:",
            "- First decide a high-level structure, then instantiate it with existing primitives only.",
            "- Motif card / regular_cms_like: symmetric init + stable triple-write + simple min/median query.",
            "- Motif card / overflow_handoff: use update_state or state-gated query so one branch can hand off when overflow-like conditions appear.",
            "- Motif card / layered_correction: use layered or hybrid init, keep one main branch and one correction branch, and let query fuse them.",
            "- Motif card / sidecar_heavy: keep a main regular body plus a sidecar/fallback branch, and let query decide how to fuse them.",
            "- Motif card / asymmetric_dual_path: let different branches play different roles instead of being three symmetric copies.",
            "- Do not add new primitives. Use existing primitives only.",
            "init_dex format guidance:",
            "- Prefer one single list_3(...) expression.",
            "- Good init example: list_3(hash_salt(0,e,1), safe_mod(hash_salt(0,e,1),102), 102, hash_salt(1,e,1), safe_mod(hash_salt(1,e,1),102), 102, hash_salt(2,e,1), safe_mod(hash_salt(2,e,1),102), 102)",
            "- Avoid init outputs that read or write counters/state.",
            "update format guidance:",
            "- update must be one single-line DEAP expression rooted at base(...).",
            "- update must contain at least one real write primitive: update_count or write_count or a conditional write wrapper around them.",
            "- Prefer one of these skeleton styles: plain triple-write, read-before-write, or conditional-write.",
            "- Good update example A: base(update_count(e,0,1), update_count(e,1,1), update_count(e,2,1))",
            "- Good update example B: base(write_count(e,0,safe_add(query_count(e,0),1)), write_count(e,1,safe_add(query_count(e,1),1)), write_count(e,2,safe_add(query_count(e,2),1)))",
            "- Good update example C: base(updatecount_if(lt(query_count(e,0),query_count(e,1)),e,0,1), updatecount_if(lt(query_count(e,1),query_count(e,2)),e,1,1), updatecount_if(lt(query_count(e,2),query_count(e,0)),e,2,1))",
            "- Bad update example: base(if_then_else(True,1,1), 1, 1)  # rejected because there is no real write primitive.",
            "- Do not emit update:, update =, expr:, expr =, return, markdown fences, or multi-line pseudo-code.",
            "- Do not use query_date or cnt_rdstate inside update.",
            "query format guidance:",
            "- query must be exactly one single-line DEAP expression string.",
            "- query should normally be rooted at base_sel(mode,a,b,c).",
            "- query should prefer query_date(e,0), query_date(e,1), query_date(e,2) as the three main read branches.",
            "- Good query example A: base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2))",
            "- Good query example B: base_sel(0, query_date(e,0), query_date(e,1), query_date(e,2))",
            "- Bad query example: query: base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2))  # bad because of the prefix.",
            "- Bad query example: return base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2))  # bad because of the return wrapper.",
            "- query must not use update_count/write_count/update_state/updatecount_if/writecount_if/writestate_if.",
            "- query must not use query_count or query_state.",
        ])

        if use_single_tree and single_target == "update":
            lines.extend([
                "Single-tree update task: output ONLY the update expr, rooted at base(...).",
                "Single-tree update task: make the smallest legal change that removes the dominant failure buckets.",
                "Single-tree update task: if unsure, prefer the simple triple-write template over a clever but fragile update.",
            ])
        if use_single_tree and single_target == "query":
            lines.extend([
                "Single-tree query task: output ONLY the query expr, rooted at base_sel(...).",
                "Single-tree query task: do not emit query:, expr:, query =, expr =, return, or wrappers like query(...).",
                "Single-tree query task: if unsure, prefer base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2)).",
            ])
        if use_single_tree and single_target == "init_dex":
            lines.extend([
                "Single-tree init_dex task: output ONLY the init_dex expr, ideally one list_3(...).",
                "Single-tree init_dex task: keep hash ids and path ids stable and small when possible.",
            ])

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

        family_guidance = report.get("family_guidance", {}) if isinstance(report, dict) else {}
        if isinstance(family_guidance, dict) and family_guidance:
            cur_summary = list(family_guidance.get("current_family_summary", []))
            avoid_exact = list(family_guidance.get("avoid_exact_families", []))
            target_family_tag = str(family_guidance.get("target_family_tag", "") or "")
            target_family_parts = family_guidance.get("target_family_parts", {})
            if cur_summary:
                lines.append("Current family distribution summary:")
                lines.extend([f"- {x}" for x in cur_summary])
            if avoid_exact:
                lines.append("Avoid saturated exact families:")
                lines.extend([f"- {x}" for x in avoid_exact])
            if target_family_tag:
                lines.append(f"Target family for this proposal: {target_family_tag}")
            if isinstance(target_family_parts, dict) and target_family_parts:
                lines.append("Target family parts:")
                lines.extend([f"- {k}: {v}" for k, v in target_family_parts.items()])
            focus_lines = list(family_guidance.get("proposal_focus", []))
            if focus_lines:
                lines.append("Proposal focus:")
                lines.extend([f"- {x}" for x in focus_lines])
            must_match = list(family_guidance.get("novelty_must_match", []))
            if must_match:
                lines.append("Primary family parts that should match the target family:")
                lines.extend([f"- {x}" for x in must_match])
            keep_stable = list(family_guidance.get("novelty_keep_stable", []))
            if keep_stable:
                lines.append("Keep these parts conservative unless there is a very strong reason:")
                lines.extend([f"- {x}" for x in keep_stable])

        cur_init = str(canonical.get("init_dex", "") or "").strip()
        cur_update = str(canonical.get("update", "") or "").strip()
        cur_query = str(canonical.get("query", "") or "").strip()
        if cur_init or cur_update or cur_query:
            lines.append("Current canonical expressions (do not merely rephrase these with equivalent syntax):")
            if cur_init:
                lines.append(f"- current_init_dex: {cur_init}")
            if cur_update:
                lines.append(f"- current_update: {cur_update}")
            if cur_query:
                lines.append(f"- current_query: {cur_query}")

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

    @staticmethod
    def _strip_outer_wrapper_quotes(text: str) -> str:
        s = str(text or "").strip()
        changed = True
        while changed and len(s) >= 2:
            changed = False
            if s[:3] == '"""' and s[-3:] == '"""':
                s = s[3:-3].strip()
                changed = True
                continue
            if s[:3] == "'''" and s[-3:] == "'''":
                s = s[3:-3].strip()
                changed = True
                continue
            if s[0] == s[-1] and s[0] in {'"', "'", '`'}:
                s = s[1:-1].strip()
                changed = True
        return s

    @staticmethod
    def _extract_balanced_call_text(text: str, func_name: str):
        src = str(text or "")
        m = re.search(rf'\b{re.escape(func_name)}\s*\(', src)
        if not m:
            return None
        start = m.start()
        i = src.find('(', m.start())
        if i < 0:
            return None
        depth = 0
        for j in range(i, len(src)):
            ch = src[j]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return src[start:j + 1]
        return None

    def _query_template_exprs_from_text(self, text: str):
        raw = str(text or "")
        preferred = [2, 0, 3]
        if "base_sel(0" in raw or "min-query" in raw:
            preferred = [0, 2, 3]
        elif "base_sel(3" in raw or "avg-query" in raw:
            preferred = [3, 2, 0]
        elif "cnt_rdstate(" in raw:
            return [
                "base_sel(cnt_rdstate(e,0), query_date(e,0), query_date(e,1), query_date(e,2))",
                "base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2))",
                "base_sel(0, query_date(e,0), query_date(e,1), query_date(e,2))",
            ]
        out = [
            f"base_sel({mode}, query_date(e,0), query_date(e,1), query_date(e,2))"
            for mode in preferred
        ]
        out.append("base_sel(cnt_rdstate(e,0), query_date(e,0), query_date(e,1), query_date(e,2))")
        return out

    def _normalize_query_candidate_text(self, text):
        src = str(text or "")
        if not src.strip():
            return ""
        m = re.search(r"```(?:json|python|text)?\s*([\s\S]*?)```", src, flags=re.IGNORECASE)
        if m:
            src = m.group(1)
        src = src.replace("\r\n", "\n").replace("\r", "\n").strip()
        src = self._strip_outer_wrapper_quotes(src)
        src = src.strip().strip(",;")

        prefix_patterns = [
            r'^\s*query\s*[:=]\s*(.+)$',
            r'^\s*expr\s*[:=]\s*(.+)$',
            r'^\s*return\s+(.+)$',
        ]
        changed = True
        while changed:
            changed = False
            for pat in prefix_patterns:
                mm = re.match(pat, src, flags=re.IGNORECASE | re.DOTALL)
                if mm:
                    src = self._strip_outer_wrapper_quotes(mm.group(1).strip()).strip().strip(",;")
                    changed = True

        lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
        if len(lines) > 1:
            rich = [
                ln for ln in lines
                if any(tok in ln for tok in (
                    "base_sel(", "query_date(", "cnt_rdstate(", "safe_", "sum3(",
                    "median3(", "lt(", "gt(", "eq(", "if_then_else(", "str_slice(",
                    "query:", "query =", "expr:", "expr =", "return ",
                ))
            ]
            src = " ".join(rich or lines)

        return src.strip().strip(",;")

    def _extract_query_core_expr(self, text):
        src = self._normalize_query_candidate_text(text)
        if not src:
            return ""

        base_sel_call = self._extract_balanced_call_text(src, "base_sel")
        if base_sel_call:
            return base_sel_call.strip()

        wrapped_query = self._extract_balanced_call_text(src, "query")
        if wrapped_query and wrapped_query.strip().startswith("query(") and wrapped_query.strip().endswith(")"):
            inner = wrapped_query.strip()[len("query("):-1].strip()
            inner = self._normalize_query_candidate_text(inner)
            inner_base = self._extract_balanced_call_text(inner, "base_sel")
            return (inner_base or inner).strip()

        mm = re.match(r'^\s*(?:query|expr)\s*[:=]\s*(.+)$', src, flags=re.IGNORECASE | re.DOTALL)
        if mm:
            src = mm.group(1).strip()
        mm = re.match(r'^\s*return\s+(.+)$', src, flags=re.IGNORECASE | re.DOTALL)
        if mm:
            src = mm.group(1).strip()
        return src.strip().strip(",;")

    def _auto_repair_query_expr(self, expr_text):
        expr = self._extract_query_core_expr(expr_text)
        cands = []

        def add(x):
            sx = self._normalize_query_candidate_text(x)
            if sx and sx not in cands:
                cands.append(sx)

        add(expr)

        if expr.lower().startswith("base("):
            add("base_sel(" + expr[len("base("):])

        if expr.lower().startswith("query(") and expr.endswith(")"):
            inner = expr[len("query("):-1].strip()
            add(inner)
            if inner.lower().startswith("base("):
                add("base_sel(" + inner[len("base("):])

        extracted_base_sel = self._extract_balanced_call_text(expr, "base_sel")
        if extracted_base_sel:
            add(extracted_base_sel)

        if expr and not expr.startswith("base_sel("):
            add(f"base_sel(2, {expr}, query_date(e,1), query_date(e,2))")
            add(f"base_sel(0, {expr}, query_date(e,1), query_date(e,2))")
            add(f"base_sel(3, {expr}, query_date(e,1), query_date(e,2))")

        for tpl in self._query_template_exprs_from_text(expr_text):
            add(tpl)

        return cands

    def _parse_query_expr_with_fallbacks(self, expr_text, pset_map):
        pset = pset_map["query"]
        normalized = self._normalize_query_candidate_text(expr_text)
        core_expr = self._extract_query_core_expr(normalized)
        candidates = []

        def add(x):
            sx = self._normalize_query_candidate_text(x)
            if sx and sx not in candidates:
                candidates.append(sx)

        # 优先尝试已经显式修到 base_sel(...) 形式的候选，再尝试原始文本。
        for cand in self._auto_repair_query_expr(core_expr or normalized):
            add(cand)
        add(core_expr)
        add(normalized)

        parse_errors = []
        primitive_allowed = _runtime_primitive_names(pset)
        for idx, cand in enumerate(candidates):
            try:
                tree = INDIVIDUAL_CLS.from_string(cand, pset)
            except Exception as e:
                parse_errors.append(str(e))
                continue

            try:
                root = tree[0]
                root_name = str(getattr(root, "name", "") or "")
                root_ret = getattr(root, "ret", None)
                if root_ret is not float:
                    parse_errors.append(f"query_root_type_mismatch expected=float got={root_ret} root={root_name}")
                    continue
                if root_name != "base_sel":
                    parse_errors.append(f"query_root_not_base_sel:{root_name}")
                    continue

                prim_names = _tree_primitive_names(tree)
                bad_runtime = sorted([n for n in prim_names if n not in primitive_allowed])
                if bad_runtime:
                    parse_errors.append(f"query_primitive_not_in_runtime:{','.join(bad_runtime[:5])}")
                    continue
                if "query_date" not in prim_names:
                    parse_errors.append("query_missing_query_date")
                    continue
                if any(n in prim_names for n in {"update_count", "write_count", "updatecount_if", "writecount_if", "update_state", "writestate_if"}):
                    parse_errors.append("query_contains_forbidden_write_ops")
                    continue
                if any(n in prim_names for n in {"query_count", "query_state"}):
                    parse_errors.append("query_contains_query_count_or_query_state")
                    continue

                return tree, {
                    "used_expr": cand,
                    "normalized_text": normalized,
                    "core_expr": core_expr,
                    "used_autofix": bool(idx >= 1 or cand != core_expr),
                }, []
            except Exception as e:
                parse_errors.append(f"query_ast_simplify_failed:{type(e).__name__}:{e}")

        top_reason = parse_errors[0] if parse_errors else "unknown"
        return None, {
            "normalized_text": normalized,
            "core_expr": core_expr,
            "parse_errors": parse_errors[:8],
        }, [f"parse_query_failed:{top_reason}"]

    def _update_template_exprs_from_text(self, text: str):
        raw = str(text or "")
        default_templates = [
            "base(update_count(e,0,1), update_count(e,1,1), update_count(e,2,1))",
            "base(write_count(e,0,safe_add(query_count(e,0),1)), write_count(e,1,safe_add(query_count(e,1),1)), write_count(e,2,safe_add(query_count(e,2),1)))",
            "base(updatecount_if(lt(query_count(e,0),query_count(e,1)),e,0,1), updatecount_if(lt(query_count(e,1),query_count(e,2)),e,1,1), updatecount_if(lt(query_count(e,2),query_count(e,0)),e,2,1))",
        ]
        if "write_count(" in raw or "read-before-write" in raw:
            return [default_templates[1], default_templates[0], default_templates[2]]
        if "updatecount_if(" in raw or "conditional-write" in raw:
            return [default_templates[2], default_templates[0], default_templates[1]]
        return list(default_templates)

    def _normalize_update_candidate_text(self, text):
        src = str(text or "")
        if not src.strip():
            return ""
        m = re.search(r"```(?:json|python|text)?\s*([\s\S]*?)```", src, flags=re.IGNORECASE)
        if m:
            src = m.group(1)
        src = src.replace("\r\n", "\n").replace("\r", "\n").strip()
        src = self._strip_outer_wrapper_quotes(src)
        src = src.strip().strip(",;")
        prefix_patterns = [
            r'^\s*update\s*[:=]\s*(.+)$',
            r'^\s*expr\s*[:=]\s*(.+)$',
            r'^\s*return\s+(.+)$',
        ]
        changed = True
        while changed:
            changed = False
            for pat in prefix_patterns:
                mm = re.match(pat, src, flags=re.IGNORECASE | re.DOTALL)
                if mm:
                    src = self._strip_outer_wrapper_quotes(mm.group(1).strip()).strip().strip(",;")
                    changed = True
        lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
        if len(lines) > 1:
            rich = [
                ln for ln in lines
                if any(tok in ln for tok in (
                    "base(", "update_count(", "write_count(", "updatecount_if(", "writecount_if(",
                    "query_count(", "if_then_else(", "safe_", "lt(", "gt(", "eq(",
                    "update:", "update =", "expr:", "expr =", "return ",
                ))
            ]
            src = " ".join(rich or lines)
        return src.strip().strip(",;")

    def _extract_update_core_expr(self, text):
        src = self._normalize_update_candidate_text(text)
        if not src:
            return ""
        base_call = self._extract_balanced_call_text(src, "base")
        if base_call:
            return base_call.strip()
        wrapped_update = self._extract_balanced_call_text(src, "update")
        if wrapped_update and wrapped_update.strip().startswith("update(") and wrapped_update.strip().endswith(")"):
            inner = wrapped_update.strip()[len("update("):-1].strip()
            inner = self._normalize_update_candidate_text(inner)
            inner_base = self._extract_balanced_call_text(inner, "base")
            return (inner_base or inner).strip()
        mm = re.match(r'^\s*(?:update|expr)\s*[:=]\s*(.+)$', src, flags=re.IGNORECASE | re.DOTALL)
        if mm:
            src = mm.group(1).strip()
        mm = re.match(r'^\s*return\s+(.+)$', src, flags=re.IGNORECASE | re.DOTALL)
        if mm:
            src = mm.group(1).strip()
        return src.strip().strip(",;")

    def _auto_repair_update_expr(self, expr_text):
        expr = self._extract_update_core_expr(expr_text)
        cands = []
        def add(x):
            sx = self._normalize_update_candidate_text(x)
            if sx and sx not in cands:
                cands.append(sx)
        add(expr)
        extracted_base = self._extract_balanced_call_text(expr, "base")
        if extracted_base:
            add(extracted_base)
        if expr.lower().startswith("update(") and expr.endswith(")"):
            inner = expr[len("update("):-1].strip()
            add(inner)
            inner_base = self._extract_balanced_call_text(inner, "base")
            if inner_base:
                add(inner_base)
        if expr and not expr.startswith("base("):
            add(f"base({expr}, update_count(e,1,1), update_count(e,2,1))")
            add(f"base({expr}, write_count(e,1,safe_add(query_count(e,1),1)), write_count(e,2,safe_add(query_count(e,2),1)))")
        for tpl in self._update_template_exprs_from_text(expr_text):
            add(tpl)
        return cands

    def _parse_update_expr_with_fallbacks(self, expr_text, pset_map):
        pset = pset_map["update"]
        normalized = self._normalize_update_candidate_text(expr_text)
        core_expr = self._extract_update_core_expr(normalized)
        candidates = []
        def add(x):
            sx = self._normalize_update_candidate_text(x)
            if sx and sx not in candidates:
                candidates.append(sx)
        for cand in self._auto_repair_update_expr(core_expr or normalized):
            add(cand)
        add(core_expr)
        add(normalized)
        parse_errors = []
        primitive_allowed = _runtime_primitive_names(pset)
        write_names = {"update_count", "write_count", "updatecount_if", "writecount_if", "update_state", "writestate_if"}
        forbidden_names = {"query_date", "cnt_rdstate"}
        for idx, cand in enumerate(candidates):
            try:
                tree = INDIVIDUAL_CLS.from_string(cand, pset)
            except Exception as e:
                parse_errors.append(str(e))
                continue
            try:
                root = tree[0]
                root_name = str(getattr(root, "name", "") or "")
                root_ret = getattr(root, "ret", None)
                if root_ret is not float:
                    parse_errors.append(f"update_root_type_mismatch expected=float got={root_ret} root={root_name}")
                    continue
                if root_name != "base":
                    parse_errors.append(f"update_root_not_base:{root_name}")
                    continue
                prim_names = _tree_primitive_names(tree)
                bad_runtime = sorted([n for n in prim_names if n not in primitive_allowed])
                if bad_runtime:
                    parse_errors.append(f"update_primitive_not_in_runtime:{','.join(bad_runtime[:5])}")
                    continue
                if not any(n in prim_names for n in write_names):
                    parse_errors.append("update_missing_real_write_primitive")
                    continue
                if any(n in prim_names for n in forbidden_names):
                    parse_errors.append("update_contains_query_only_ops")
                    continue
                return tree, {
                    "used_expr": cand,
                    "normalized_text": normalized,
                    "core_expr": core_expr,
                    "used_autofix": bool(idx >= 1 or cand != core_expr),
                }, []
            except Exception as e:
                parse_errors.append(f"update_ast_simplify_failed:{type(e).__name__}:{e}")
        top_reason = parse_errors[0] if parse_errors else "unknown"
        return None, {
            "normalized_text": normalized,
            "core_expr": core_expr,
            "parse_errors": parse_errors[:8],
        }, [f"parse_update_failed:{top_reason}"]

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
                expr = str(obj.get(which, "") or "").strip()
                if not expr:
                    if base_team is not None:
                        out[which] = copy.deepcopy(base_team[which])
                        continue
                    errs.append(f"missing_{which}")
                    continue
                if which == "query":
                    qtree, _qmeta, qerrs = self._parse_query_expr_with_fallbacks(expr, pset_map)
                    if qerrs:
                        errs.extend(list(qerrs))
                        continue
                    out[which] = qtree
                elif which == "update":
                    utree, _umeta, uerrs = self._parse_update_expr_with_fallbacks(expr, pset_map)
                    if uerrs:
                        errs.extend(list(uerrs))
                        continue
                    out[which] = utree
                else:
                    out[which] = INDIVIDUAL_CLS.from_string(expr, pset_map[which])
            except Exception as e:
                errs.append(f"parse_{which}_failed:{e}")
        if errs:
            return None, errs
        return out, []

    def materialize_single_tree(self, obj, pset_map, base_team=None):
        if obj.get("mode") != "single_tree":
            return None, ["not_single_tree_mode"]
        if base_team is None:
            return None, ["single_tree_requires_base_team"]

        raw_target = str(obj.get("target", "") or "").strip().lower()
        if raw_target in {"init", "init_dex"}:
            target = "init_dex"
        else:
            target = raw_target
        if target not in {"init_dex", "update", "query"}:
            target = self._default_single_tree_target("stagnation", {"target_funcs": sorted(self.target_funcs)})
        if target not in self.target_funcs:
            return None, [f"single_tree_target_not_allowed:{target}"]

        expr = str(obj.get("expr", "") or "").strip()
        if not expr:
            return None, ["single_tree_missing_expr"]

        try:
            if target == "query":
                new_tree, _qmeta, qerrs = self._parse_query_expr_with_fallbacks(expr, pset_map)
                if qerrs:
                    return None, [f"parse_single_tree_failed:{target}:{'; '.join(qerrs)}"]
            elif target == "update":
                new_tree, _umeta, uerrs = self._parse_update_expr_with_fallbacks(expr, pset_map)
                if uerrs:
                    return None, [f"parse_single_tree_failed:{target}:{'; '.join(uerrs)}"]
            else:
                new_tree = INDIVIDUAL_CLS.from_string(expr, pset_map[target])
        except Exception as e:
            return None, [f"parse_single_tree_failed:{target}:{e}"]

        max_ratio = max(1.0, float(self.cfg.get("llm_single_tree_max_ratio", 1.5)))
        base_size = max(1, len(base_team[target]))
        new_size = len(new_tree)
        max_allowed = max(base_size + 3, int(math.ceil(base_size * max_ratio)))
        if new_size > max_allowed:
            return None, [f"single_tree_too_large:{target}:{new_size}>{max_allowed}"]

        out = {
            "init_dex": copy.deepcopy(base_team["init_dex"]),
            "update": copy.deepcopy(base_team["update"]),
            "query": copy.deepcopy(base_team["query"]),
        }
        out[target] = new_tree
        return out, []

    def _normalize_failure_reason(self, reason):
        s = str(reason or "").strip()
        if not s:
            return "unknown"
        if s.startswith("parse_single_tree_failed:"):
            parts = s.split(":", 2)
            return ":".join(parts[:2])
        if s.startswith("single_tree_too_large:"):
            parts = s.split(":", 2)
            return ":".join(parts[:2])
        if s.startswith("parse_init_dex_failed:") or s.startswith("parse_update_failed:") or s.startswith("parse_query_failed:"):
            return s.split(":", 1)[0]
        if s.startswith("evaluate_failed:") or s.startswith("ast_validate_failed:"):
            return s.split(":", 1)[0]
        if s.startswith("query_ast_simplify_failed:") or s.startswith("query_ast_illegal:"):
            return s.split(":", 1)[0]
        if s.startswith("query_root_not_base_sel:"):
            return "query_root_not_base_sel"
        if "_root_type_mismatch" in s and " expected=" in s:
            return s.split(" expected=", 1)[0]
        if s.startswith("single_tree_target_not_allowed:"):
            return s.split(":", 1)[0]
        return s

    def _summarize_failed_records(self, failed_records, topk=5):
        topk = max(1, int(topk))
        stage_counts = Counter()
        raw_reason_counts = Counter()
        norm_reason_counts = Counter()
        validate_reason_counts = Counter()
        materialize_reason_counts = Counter()
        stage_reason_counts = {}

        for rec in list(failed_records or []):
            stage = str(rec.get("stage", "unknown") or "unknown")
            stage_counts[stage] += 1
            reasons = list(rec.get("reasons", [])) or ["unknown"]
            for rr in reasons:
                raw = str(rr or "unknown").strip() or "unknown"
                norm = self._normalize_failure_reason(raw)
                raw_reason_counts[raw] += 1
                norm_reason_counts[norm] += 1
                stage_reason_counts.setdefault(stage, Counter())
                stage_reason_counts[stage][norm] += 1
                if "validate" in stage:
                    validate_reason_counts[norm] += 1
                elif "materialize" in stage:
                    materialize_reason_counts[norm] += 1

        return {
            "total_failures": int(sum(stage_counts.values())),
            "stage_counts": dict(stage_counts),
            "top_raw_reasons": [f"{k} x{v}" for k, v in raw_reason_counts.most_common(topk)],
            "top_reasons": [f"{k} x{v}" for k, v in norm_reason_counts.most_common(topk)],
            "validate_top_reasons": [f"{k} x{v}" for k, v in validate_reason_counts.most_common(topk)],
            "materialize_top_reasons": [f"{k} x{v}" for k, v in materialize_reason_counts.most_common(topk)],
            "stage_top_reasons": {
                str(stage): [f"{k} x{v}" for k, v in ctr.most_common(topk)]
                for stage, ctr in stage_reason_counts.items()
            },
        }

    def _validate_query_tree_details(self, evaluator, query_tree):
        reasons = []
        forbidden_write_names = {"update_count", "write_count", "updatecount_if", "writecount_if", "update_state", "writestate_if"}
        try:
            qinfo = evaluator.analyze_query_tree(query_tree)
            if not bool(qinfo.get("root_ok", False)):
                reasons.append("query_root_not_base_sel")
            if int(qinfo.get("read_calls", 0)) <= 0:
                reasons.append("query_missing_query_date")
            forb = dict(qinfo.get("forbidden_hits", {}) or {})
            if any(str(k) in forbidden_write_names for k in forb.keys()):
                reasons.append("query_contains_forbidden_write_ops")
            if any(str(k) in {"query_count", "query_state"} for k in forb.keys()):
                reasons.append("query_contains_query_count_or_query_state")
        except Exception as e:
            reasons.append(f"query_ast_simplify_failed:{type(e).__name__}")
            return sorted(set(reasons))

        try:
            qry_ast = evaluator._simplify_ast(evaluator._tree_to_ast(query_tree))
        except Exception as e:
            reasons.append(f"query_ast_simplify_failed:{type(e).__name__}")
            return sorted(set(reasons))

        try:
            qleg = evaluator._ast_legality_check("query", qry_ast)
            if qleg.get("hard_illegal"):
                leg_reasons = [str(x) for x in list(qleg.get("reasons", []))]
                if any(rr == "query_contains_write_or_update_reads" for rr in leg_reasons):
                    reasons.append("query_contains_forbidden_write_ops")
                for rr in leg_reasons:
                    if rr != "query_contains_write_or_update_reads":
                        reasons.append(f"query_ast_illegal:{rr}")
            qeff = evaluator._ast_effect_summary(qry_ast)
            if int(qeff.get("query_date_calls", 0)) <= 0:
                reasons.append("query_missing_query_date")
            qpat = evaluator._ast_pattern_summary("query", qry_ast)
            if int(qpat.get("nonconst_path_idx", 0)) > 0:
                reasons.append("query_dynamic_path_invalid")
            names = evaluator._ast_collect_names(qry_ast).get("names", Counter())
            if int(names.get("query_count", 0)) > 0 or int(names.get("query_state", 0)) > 0:
                reasons.append("query_contains_query_count_or_query_state")
            if any(int(names.get(n, 0)) > 0 for n in forbidden_write_names):
                reasons.append("query_contains_forbidden_write_ops")
        except Exception as e:
            reasons.append(f"query_ast_simplify_failed:{type(e).__name__}")
        return sorted(set(reasons))

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
            if init_info.get("forbidden_hits"):
                reasons.append("init_forbidden_hits")
            if update_info.get("forbidden_hits"):
                reasons.append("update_forbidden_hits")
            init_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["init_dex"]))
            upd_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["update"]))
            if evaluator._ast_legality_check("init", init_ast).get("hard_illegal"):
                reasons.append("init_ast_hard_illegal")
            if evaluator._ast_legality_check("update", upd_ast).get("hard_illegal"):
                reasons.append("update_ast_hard_illegal")
        except Exception as e:
            reasons.append(f"ast_validate_failed:{e}")

        reasons.extend(self._validate_query_tree_details(evaluator, team["query"]))
        if reasons:
            return {"ok": False, "reasons": sorted(set(reasons)), "warnings": sorted(set(warns))}

        key = evaluator._canonical_triplet_key(team["init_dex"], team["update"], team["query"])
        try:
            candidate_meta = asdict(_candidate_meta_from_team(evaluator, team))
        except Exception:
            candidate_meta = {
                "family_tag": str(_team_family_tag(team)),
                "key_v1": key,
                "key_v2": key,
                "repair_dup_key": key,
                "architecture_schema": {},
                "motif_signature": {},
                "schema_hash": "",
                "motif_key": "",
                "arch_type": "regular",
            }
        key_v2 = tuple(candidate_meta.get("key_v2", key)) if isinstance(candidate_meta.get("key_v2", key), (list, tuple)) else key
        repair_dup_key = tuple(candidate_meta.get("repair_dup_key", key)) if isinstance(candidate_meta.get("repair_dup_key", key), (list, tuple)) else key
        if existing_canon is not None:
            if key in existing_canon or key_v2 in existing_canon:
                return {"ok": False, "reasons": ["duplicate_canonical_team"], "warnings": sorted(set(warns))}
            if repair_dup_key in existing_canon:
                return {"ok": False, "reasons": ["duplicate_structural_team"], "warnings": sorted(set(warns))}

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
            "key_v2": key_v2,
            "repair_dup_key": repair_dup_key,
            "candidate_meta": candidate_meta,
            "warnings": sorted(set(warns)),
        }

    def prepare_phase_candidates(self, phase, gp_ctx, evaluator, base_team, existing_canon, limit, extra_prompt_hints=None, family_guidance=None, force_single_tree=None, force_single_tree_target=None, target_funcs_override=None, candidate_channel="", adaptive_reason=""):
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
            family_guidance=family_guidance,
        )
        if target_funcs_override is not None:
            report["target_funcs"] = sorted(_parse_llm_target_funcs(target_funcs_override))
        if force_single_tree is not None:
            report["_force_single_tree"] = bool(force_single_tree)
        if force_single_tree_target is not None:
            report["_force_single_tree_target"] = str(force_single_tree_target)
        if candidate_channel:
            report["_candidate_channel"] = str(candidate_channel)
        if adaptive_reason:
            report["_adaptive_reason"] = str(adaptive_reason)

        out = []
        seen = set(existing_canon or set())
        failed_records = []
        all_failed_records = []
        n_materialized = 0
        n_validated = 0
        n_evaluated = 0
        total_raw = 0
        total_parsed = 0
        total_team_parsed = 0
        total_single_parsed = 0

        def _record_failed(stage, reasons):
            rec = {"stage": str(stage), "reasons": list(reasons or ["unknown"])}
            failed_records.append(rec)
            all_failed_records.append(copy.deepcopy(rec))

        def _consume_raw(raw_list):
            nonlocal n_materialized, n_validated, n_evaluated, total_raw, total_parsed, total_team_parsed, total_single_parsed, out, seen, failed_records, all_failed_records
            total_raw += len(raw_list)
            parsed = self.parse_candidate_objects(raw_list)
            total_parsed += len(parsed)
            parsed_teams = [x for x in parsed if x.get("mode") == "team"]
            parsed_single = [x for x in parsed if x.get("mode") == "single_tree"]
            total_team_parsed += len(parsed_teams)
            total_single_parsed += len(parsed_single)

            for obj in parsed_single:
                team, perr = self.materialize_single_tree(obj, gp_ctx["pset_map"], base_team=base_team)
                if perr:
                    _record_failed("materialize_single_tree", list(perr))
                    continue
                n_materialized += 1
                chk = self.validate_team_candidate(team, evaluator, existing_canon=seen)
                if not chk.get("ok", False):
                    _record_failed("validate_single_tree", list(chk.get("reasons", [])))
                    continue
                n_validated += 1
                n_evaluated += 1
                if chk.get("warnings"):
                    self.logger.warn("runtime/reference conflict accepted by runtime compatibility", warnings=chk.get("warnings"))
                seen.add(chk["key"])
                seen.add(chk.get("key_v2", chk["key"]))
                seen.add(chk.get("repair_dup_key", chk["key"]))
                out.append({
                    "team": chk["team"],
                    "fit": chk["fit"],
                    "err": chk["err"],
                    "case_vec": chk["case_vec"],
                    "rationale": obj.get("rationale", ""),
                    "source": self.provider,
                    "edit_mode": "single_tree",
                    "edit_target": str(obj.get("target", "") or self._default_single_tree_target(phase, report)),
                    "channel": str(candidate_channel or "repair"),
                    "novelty_subtype": "stable",
                    "family_tag": str(chk.get("candidate_meta", {}).get("family_tag", _team_family_tag(chk["team"]))),
                    "family_parts": _team_family_parts(chk["team"]),
                    "architecture_schema": copy.deepcopy(chk.get("candidate_meta", {}).get("architecture_schema", {})),
                    "motif_signature": copy.deepcopy(chk.get("candidate_meta", {}).get("motif_signature", {})),
                    "arch_type": str(chk.get("candidate_meta", {}).get("arch_type", "regular")),
                    "motif_key": str(chk.get("candidate_meta", {}).get("motif_key", "")),
                    "schema_hash": str(chk.get("candidate_meta", {}).get("schema_hash", "")),
                })
                if len(out) >= max(1, int(limit)):
                    return

            for obj in parsed_teams:
                team, perr = self.materialize_team(obj, gp_ctx["pset_map"], base_team=base_team)
                if perr:
                    _record_failed("materialize", list(perr))
                    continue
                n_materialized += 1
                chk = self.validate_team_candidate(team, evaluator, existing_canon=seen)
                if not chk.get("ok", False):
                    _record_failed("validate", list(chk.get("reasons", [])))
                    continue
                n_validated += 1
                n_evaluated += 1
                if chk.get("warnings"):
                    self.logger.warn("runtime/reference conflict accepted by runtime compatibility", warnings=chk.get("warnings"))
                seen.add(chk["key"])
                seen.add(chk.get("key_v2", chk["key"]))
                seen.add(chk.get("repair_dup_key", chk["key"]))
                out.append({
                    "team": chk["team"],
                    "fit": chk["fit"],
                    "err": chk["err"],
                    "case_vec": chk["case_vec"],
                    "rationale": obj.get("rationale", ""),
                    "source": self.provider,
                    "edit_mode": "team",
                    "edit_target": "team",
                    "channel": str(candidate_channel or "novelty"),
                    "novelty_subtype": ("irregular" if str(candidate_channel or "").strip().lower() in {"irregular_novelty", "innovation", "innovative_novelty"} else "stable"),
                    "family_tag": str(chk.get("candidate_meta", {}).get("family_tag", _team_family_tag(chk["team"]))),
                    "family_parts": _team_family_parts(chk["team"]),
                    "architecture_schema": copy.deepcopy(chk.get("candidate_meta", {}).get("architecture_schema", {})),
                    "motif_signature": copy.deepcopy(chk.get("candidate_meta", {}).get("motif_signature", {})),
                    "arch_type": str(chk.get("candidate_meta", {}).get("arch_type", "regular")),
                    "motif_key": str(chk.get("candidate_meta", {}).get("motif_key", "")),
                    "schema_hash": str(chk.get("candidate_meta", {}).get("schema_hash", "")),
                })
                if len(out) >= max(1, int(limit)):
                    return

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

        fail_summary = self._summarize_failed_records(
            all_failed_records,
            topk=max(1, int(self.cfg.get("llm_fail_reason_topk", 5))),
        )
        if int(fail_summary.get("total_failures", 0)) > 0:
            self.logger.info(
                "llm candidate fail summary",
                phase=phase,
                source=self.provider,
                channel=str(candidate_channel or ("repair" if self._prefer_single_tree_for_phase(phase, report=report) else "novelty")),
                total_failures=int(fail_summary.get("total_failures", 0)),
                failed_stage_counts=fail_summary.get("stage_counts", {}),
                failed_reason_topk=fail_summary.get("top_reasons", []),
                validate_reason_topk=fail_summary.get("validate_top_reasons", []),
                materialize_reason_topk=fail_summary.get("materialize_top_reasons", []),
                stage_top_reasons=fail_summary.get("stage_top_reasons", {}),
            )
        if total_parsed > 0 and n_validated == 0:
            self.logger.warn(
                "llm validation bottleneck",
                phase=phase,
                source=self.provider,
                parsed=total_parsed,
                materialized=n_materialized,
                validated=n_validated,
                failed_stage_counts=fail_summary.get("stage_counts", {}),
                failed_reason_topk=fail_summary.get("top_reasons", []),
                validate_reason_topk=fail_summary.get("validate_top_reasons", []),
                materialize_reason_topk=fail_summary.get("materialize_top_reasons", []),
            )

        self.logger.info(
            "llm candidate summary",
            phase=phase,
            source=self.provider,
            raw=total_raw,
            parsed=total_parsed,
            team_parsed=total_team_parsed,
            single_tree_parsed=total_single_parsed,
            materialized=n_materialized,
            validated=n_validated,
            evaluated=n_evaluated,
            accepted=len(out),
            repair_rounds=repair_rounds,
            failed_total=int(fail_summary.get("total_failures", 0)),
            failed_stage_counts=fail_summary.get("stage_counts", {}),
            failed_reason_topk=fail_summary.get("top_reasons", []),
            validate_reason_topk=fail_summary.get("validate_top_reasons", []),
            materialize_reason_topk=fail_summary.get("materialize_top_reasons", []),
            use_case_vec=bool(self.cfg.get("llm_use_case_vec", False)),
            use_hard_cases=bool(self.cfg.get("llm_use_hard_cases", False)),
            failure_buckets=report.get("prompt_hints", {}).get("failure_buckets", []),
            single_tree_mode=self._prefer_single_tree_for_phase(phase),
            single_tree_target=self._default_single_tree_target(phase, report),
            target_family=report.get("family_guidance", {}).get("target_family_tag", ""),
            channel=str(candidate_channel or ("repair" if self._prefer_single_tree_for_phase(phase, report=report) else "novelty")),
            adaptive_reason=str(report.get("_adaptive_reason", "")),
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

    _append_unique_str(hints["hard_avoid"], "Do not prefix query with query:, query =, expr:, expr =, or return.")
    _append_unique_str(hints["hard_avoid"], "Do not use update_count/write_count/update_state/updatecount_if/writecount_if/writestate_if inside query.")
    _append_unique_str(hints["hard_avoid"], "Do not use query_count or query_state inside query.")
    _append_unique_str(hints["prefer"], "Output query as a single-line DEAP expression rooted at base_sel(mode,a,b,c).")
    _append_unique_str(hints["prefer"], "Prefer query_date(e,0), query_date(e,1), query_date(e,2) as the three main query reads.")

    return hints

def _serialize_team_spec(team, rationale="", source="offline_json"):
    fam_tag = str(team.get("family_tag", _team_family_tag(team))) if isinstance(team, dict) else ""
    fam_parts = team.get("family_parts") if isinstance(team, dict) else None
    if not isinstance(fam_parts, dict) and isinstance(team, dict):
        fam_parts = _team_family_parts(team)
    return {
        "init_dex": str(team["init_dex"]),
        "update": str(team["update"]),
        "query": str(team["query"]),
        "rationale": str(rationale or ""),
        "source": str(source or ""),
        "family_tag": fam_tag,
        "family_parts": copy.deepcopy(fam_parts) if isinstance(fam_parts, dict) else None,
    }


def _deserialize_team_spec(spec, pset_map):
    team = {
        "init_dex": INDIVIDUAL_CLS.from_string(str(spec["init_dex"]), pset_map["init_dex"]),
        "update": INDIVIDUAL_CLS.from_string(str(spec["update"]), pset_map["update"]),
        "query": INDIVIDUAL_CLS.from_string(str(spec["query"]), pset_map["query"]),
    }
    fam_parts = spec.get("family_parts")
    if isinstance(fam_parts, dict):
        team["family_parts"] = copy.deepcopy(fam_parts)
    else:
        team["family_parts"] = _team_family_parts(team)
    team["family_tag"] = str(spec.get("family_tag", _team_family_tag(team)))
    return team


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
        try:
            if str(source_meta.get("phase", "")) in {"seed", "stagnation"} or str(source_meta.get("channel", "")) == "novelty":
                candidate_meta = {
                    "family_tag": str(source_meta.get("family_tag", "")),
                    "arch_type": str(source_meta.get("arch_type", "regular")),
                    "schema_hash": str(source_meta.get("schema_hash", "")),
                    "motif_key": str(source_meta.get("motif_key", "")),
                    "architecture_schema": copy.deepcopy(source_meta.get("architecture_schema", {})),
                    "motif_signature": copy.deepcopy(source_meta.get("motif_signature", {})),
                }
                _innovation_archive_append(state, candidate_meta=candidate_meta, fit=float(fit_err_casevec[0]), err=float(fit_err_casevec[1]), source=str(source_meta.get("source", "")))
        except Exception:
            pass
    return state

def _collect_existing_canonical_keys_from_states(island_states, evaluator):
    keys = set()
    for st in island_states:
        n = len(st.get("fits", []))
        for i in range(n):
            try:
                team = {
                    "init_dex": st["pops"]["init_dex"][i],
                    "update": st["pops"]["update"][i],
                    "query": st["pops"]["query"][i],
                }
                keys.add(evaluator._canonical_triplet_key(
                    team["init_dex"], team["update"], team["query"]
                ))
                meta = asdict(_candidate_meta_from_team(evaluator, team))
                key_v2 = meta.get("key_v2", ())
                repair_dup_key = meta.get("repair_dup_key", ())
                if key_v2:
                    keys.add(tuple(key_v2) if isinstance(key_v2, (list, tuple)) else key_v2)
                if repair_dup_key:
                    keys.add(tuple(repair_dup_key) if isinstance(repair_dup_key, (list, tuple)) else repair_dup_key)
            except Exception:
                continue
    return keys


def _apply_llm_seed_specs_to_state(state, cfg, gp_ctx, evaluator, target_funcs_override=None):
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
    target_funcs = _parse_llm_target_funcs(target_funcs_override if target_funcs_override is not None else cfg.get("llm_target_funcs", "update,query"))
    island_profile = _get_island_profile(cfg, int(state.get("island_idx", 0)))
    family_hist = _family_histogram_from_state(state)
    family_guidance = _build_family_guidance(island_profile, family_hist, available_specs=specs, failure_buckets=[])
    if len(specs) > 1:
        specs = sorted(
            specs,
            key=lambda sp: _pick_spec_family_injection_sort_key(
                sp,
                island_profile,
                family_hist=family_hist,
                pset_map=gp_ctx["pset_map"],
                target_parts=family_guidance.get("target_family_parts", {}),
            ) + (str(sp.get("source", "")),)
        )
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
            seed_meta = asdict(_candidate_meta_from_team(evaluator, team))
            replace_individual_in_state(
                state,
                idx,
                team,
                (fit, err, case_vec),
                source_meta={"phase": "seed", "source": spec.get("source", "offline_json"), "rationale": spec.get("rationale", ""), "family_tag": seed_meta.get("family_tag", spec.get("family_tag", "")), "architecture_schema": copy.deepcopy(seed_meta.get("architecture_schema", {})), "motif_signature": copy.deepcopy(seed_meta.get("motif_signature", {})), "schema_hash": str(seed_meta.get("schema_hash", "")), "motif_key": str(seed_meta.get("motif_key", "")), "arch_type": str(seed_meta.get("arch_type", "regular"))},
                fec_key=fec_key,
            )
            print(f"[LLM_SEED_APPLY] idx={idx} fit={float(fit):.6f} err={float(err):.6f}", flush=True)
            inserted += 1
        except Exception:
            continue
    return state, inserted


def _inject_llm_immigrants_with_engine(state, cfg, gp_ctx, llm_engine, candidate_specs, success_budget, target_funcs_override=None):
    if success_budget <= 0 or (not candidate_specs):
        return state, 0, []
    evaluator = _make_evaluator_from_cfg(cfg)
    if cfg.get("hard_case_replay", False):
        evaluator.import_hard_case_state(state.get("hard_case_state"))
    target_funcs = _parse_llm_target_funcs(target_funcs_override if target_funcs_override is not None else cfg.get("llm_target_funcs", "update,query"))
    island_profile = _get_island_profile(cfg, int(state.get("island_idx", 0)))
    family_hist = _family_histogram_from_state(state)
    failure_hints = _collect_recent_failure_hints_from_states([state])
    family_guidance = _build_family_guidance(
        island_profile,
        family_hist,
        available_specs=candidate_specs,
        failure_buckets=failure_hints.get("failure_buckets", []),
        prefer_irregular=False,
    )
    irregular_family_guidance = _build_family_guidance(
        island_profile,
        family_hist,
        available_specs=candidate_specs,
        failure_buckets=failure_hints.get("failure_buckets", []),
        prefer_irregular=True,
    )
    if len(candidate_specs) > 1:
        candidate_specs = sorted(
            list(candidate_specs),
            key=lambda sp: _pick_llm_candidate_spec_sort_key(
                sp,
                island_profile,
                family_hist=family_hist,
                pset_map=gp_ctx["pset_map"],
                target_parts=family_guidance.get("target_family_parts", {}),
            )
        )
    pop_size = len(state.get("fits", []))
    if pop_size <= 0:
        return state, 0, []

    if not (isinstance(state.get("fec_keys"), list) and len(state.get("fec_keys", [])) == pop_size):
        state = _rebuild_state_fec_index(state, evaluator)

    novelty_stats = _ensure_llm_novelty_stats_in_state(state)
    novelty_holdout_ev = None
    if any(_llm_candidate_channel(sp) == "novelty" for sp in list(candidate_specs or [])):
        try:
            novelty_holdout_ev = _build_llm_novelty_holdout_evaluator_from_cfg(cfg)
        except Exception as e:
            llm_engine.logger.warn("novelty holdout-lite evaluator build failed", island=int(state.get("island_idx", 0)), error=str(e))
            novelty_holdout_ev = None

    _, island_best_fit, island_best_err, _ = _refresh_island_best(state)
    repl_idx = _rank_replacement_targets(state)
    inserted = 0
    accepted_specs = []
    ptr = 0
    for spec in candidate_specs:
        if inserted >= int(success_budget):
            break
        if ptr >= len(repl_idx):
            break
        channel = _llm_candidate_channel(spec)
        novelty_subtype = _llm_candidate_subtype(spec)
        if channel == "novelty":
            novelty_stats["proposed"] += 1
        if channel == "novelty" and str(island_profile.get("name", "")) == "baseline":
            _record_llm_novelty_reject(state, "baseline_island_skip")
            llm_engine.logger.info("novelty candidate rejected", island=int(state.get("island_idx", 0)), reason="baseline_island_skip")
            continue
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
                if channel == "novelty":
                    rej_reason = "validate_fail:" + "|".join([str(x) for x in chk.get("reasons", [])[:3]])
                    _record_llm_novelty_reject(state, rej_reason)
                    llm_engine.logger.info(
                        "novelty candidate rejected",
                        island=int(state.get("island_idx", 0)),
                        family_tag=str(spec.get("family_tag", "")),
                        reason=rej_reason,
                    )
                continue
            if channel == "novelty":
                novelty_stats["validated"] += 1
                guidance_local = irregular_family_guidance if novelty_subtype == "irregular" else family_guidance
                fam_ok, fam_metrics, fam_reasons = _novelty_family_gate(
                    chk["team"],
                    family_hist,
                    guidance_local.get("target_family_parts", {}),
                    pset_map=gp_ctx["pset_map"],
                    min_match=(1 if novelty_subtype == "irregular" else 2),
                    min_distance=(1 if novelty_subtype == "irregular" else 1),
                    profile=island_profile,
                    guidance=guidance_local,
                    novelty_subtype=novelty_subtype,
                )
                if not fam_ok:
                    rej_reason = str(fam_reasons[0] if fam_reasons else "novelty_family_gate_fail")
                    _record_llm_novelty_reject(state, rej_reason)
                    llm_engine.logger.info(
                        "novelty candidate rejected",
                        island=int(state.get("island_idx", 0)),
                        family_tag=str(_team_family_tag(chk["team"])),
                        reason=rej_reason,
                        target_match=int(fam_metrics.get("target_match", 0)),
                        family_distance=float(fam_metrics.get("mainstream_distance_score", fam_metrics.get("mainstream_distance", 0))),
                    )
                    continue
                novelty_stats["pass_family"] += 1

                sat_ok, sat_metrics, sat_reasons = _novelty_saturation_gate(
                    chk["team"],
                    family_hist,
                    pset_map=gp_ctx["pset_map"],
                    exact_cap=int(cfg.get("llm_irregular_exact_cap", 2) if novelty_subtype == "irregular" else cfg.get("llm_novelty_exact_cap", 1)),
                    component_cap=int(cfg.get("llm_irregular_component_cap", 18) if novelty_subtype == "irregular" else cfg.get("llm_novelty_component_cap", 12)),
                    min_distance_if_saturated=int(cfg.get("llm_irregular_saturation_min_distance", 1) if novelty_subtype == "irregular" else cfg.get("llm_novelty_saturation_min_distance", 2)),
                )
                if not sat_ok:
                    rej_reason = str(sat_reasons[0] if sat_reasons else "novelty_saturation_gate_fail")
                    _record_llm_novelty_reject(state, rej_reason)
                    llm_engine.logger.info(
                        "novelty candidate rejected",
                        island=int(state.get("island_idx", 0)),
                        family_tag=str(sat_metrics.get("tag", _team_family_tag(chk["team"]))),
                        reason=rej_reason,
                        exact_count=int(sat_metrics.get("exact_count", 0)),
                        component_count=int(sat_metrics.get("component_count", 0)),
                    )
                    continue
                novelty_stats["pass_saturation"] += 1

                qual_ok, qual_reason = _novelty_quality_gate(
                    chk["fit"], chk["err"], island_best_fit, island_best_err, cfg
                )
                if not qual_ok:
                    _record_llm_novelty_reject(state, str(qual_reason))
                    llm_engine.logger.info(
                        "novelty candidate rejected",
                        island=int(state.get("island_idx", 0)),
                        family_tag=str(sat_metrics.get("tag", _team_family_tag(chk["team"]))),
                        reason=str(qual_reason),
                        fit=float(chk["fit"]),
                        err=float(chk["err"]),
                    )
                    continue
                novelty_stats["pass_quality"] += 1

                holdout_lite_err = None
                holdout_lite_reason = "holdout_lite_disabled"
                if novelty_holdout_ev is not None:
                    try:
                        _, holdout_lite_err = novelty_holdout_ev.evaluate_individual(
                            chk["team"]["init_dex"],
                            chk["team"]["update"],
                            chk["team"]["query"],
                        )
                        holdout_ok, holdout_lite_reason = _novelty_holdout_gate(
                            holdout_lite_err,
                            chk["err"],
                            island_best_err,
                            cfg,
                        )
                    except Exception as e:
                        holdout_ok = False
                        holdout_lite_reason = f"novelty_holdout_eval_fail:{e}"
                    if not holdout_ok:
                        _record_llm_novelty_reject(state, str(holdout_lite_reason))
                        llm_engine.logger.info(
                            "novelty candidate rejected",
                            island=int(state.get("island_idx", 0)),
                            family_tag=str(sat_metrics.get("tag", _team_family_tag(chk["team"]))),
                            reason=str(holdout_lite_reason),
                            holdout_lite_err=None if holdout_lite_err is None else float(holdout_lite_err),
                        )
                        continue
                novelty_stats["pass_holdout"] += 1

                failure_summary = _team_failure_bucket_summary(evaluator, chk["team"])
                candidate_meta_local = dict(chk.get("candidate_meta", {}) or {})
                score_info = _compute_novelty_score(
                    chk["fit"],
                    chk["err"],
                    island_best_fit,
                    island_best_err,
                    fam_metrics,
                    sat_metrics,
                    failure_summary,
                    holdout_err=holdout_lite_err,
                    profile=island_profile,
                    guidance=guidance_local,
                    novelty_subtype=novelty_subtype,
                )
                motif_terms = _compute_motif_score_terms(
                    chk["team"],
                    state,
                    profile=island_profile,
                    novelty_subtype=novelty_subtype,
                    candidate_meta=candidate_meta_local,
                )
                score_info["motif_distance_score"] = float(motif_terms.get("motif_distance_score", 0.0))
                score_info["motif_frequency"] = int(motif_terms.get("motif_frequency", 0))
                score_info["arch_type"] = str(motif_terms.get("arch_type", candidate_meta_local.get("arch_type", "regular")))
                score_info["score"] = float(score_info.get("score", 0.0)) + 6.0 * float(motif_terms.get("motif_distance_score", 0.0)) + float(motif_terms.get("innovation_bonus", 0.0))
                score_info["innovation_bonus"] = float(score_info.get("innovation_bonus", 0.0)) + float(motif_terms.get("innovation_bonus", 0.0))
                score_info.setdefault("innovation_reasons", [])
                score_info["innovation_reasons"] = list(score_info.get("innovation_reasons", [])) + [f"motif_freq={int(motif_terms.get('motif_frequency', 0))}", f"arch_type={str(motif_terms.get('arch_type', candidate_meta_local.get('arch_type', 'regular')))}"]
                if novelty_subtype == "irregular":
                    irr_meta_reasons = _validate_irregular_candidate_meta(candidate_meta_local)
                    if irr_meta_reasons:
                        rej_reason = str(irr_meta_reasons[0])
                        _record_llm_novelty_reject(state, rej_reason)
                        llm_engine.logger.info(
                            "novelty candidate rejected",
                            island=int(state.get("island_idx", 0)),
                            family_tag=str(sat_metrics.get("tag", _team_family_tag(chk["team"]))),
                            reason=rej_reason,
                            arch_type=str(candidate_meta_local.get("arch_type", "regular")),
                        )
                        continue
                    irregular_bonus_info = _compute_irregular_novelty_bonus(
                        chk["team"],
                        family_metrics=fam_metrics,
                        failure_summary=failure_summary,
                    )
                    score_info["score"] = float(score_info.get("score", 0.0)) + float(irregular_bonus_info.get("bonus", 0.0))
                    score_info["innovation_bonus"] = float(score_info.get("innovation_bonus", 0.0)) + float(irregular_bonus_info.get("bonus", 0.0))
                    score_info["innovation_reasons"] = list(score_info.get("innovation_reasons", [])) + list(irregular_bonus_info.get("reasons", []))
                min_score = float(cfg.get("llm_irregular_min_score", 15.0) if novelty_subtype == "irregular" else cfg.get("llm_novelty_min_score", 18.0))
                if float(score_info.get("score", -1e18)) < float(min_score):
                    rej_reason = f"novelty_score_too_low:{float(score_info.get('score', 0.0)):.2f}<{float(min_score):.2f}"
                    _record_llm_novelty_reject(state, rej_reason)
                    llm_engine.logger.info(
                        "novelty candidate rejected",
                        island=int(state.get("island_idx", 0)),
                        family_tag=str(sat_metrics.get("tag", _team_family_tag(chk["team"]))),
                        reason=rej_reason,
                        novelty_score=float(score_info.get("score", 0.0)),
                        holdout_lite_err=None if holdout_lite_err is None else float(holdout_lite_err),
                    )
                    continue
                novelty_stats["pass_score"] += 1
                spec["family_match"] = int(fam_metrics.get("target_match", 0))
                spec["family_match_score"] = float(fam_metrics.get("target_match_score", fam_metrics.get("target_match", 0)))
                spec["family_distance"] = int(fam_metrics.get("mainstream_distance", 0))
                spec["family_distance_score"] = float(fam_metrics.get("mainstream_distance_score", fam_metrics.get("mainstream_distance", 0)))
                spec["distance_near_miss_used"] = bool(fam_metrics.get("distance_near_miss_used", False))
                spec["effective_min_distance_score"] = float(fam_metrics.get("effective_min_distance_score", 0.0))
                spec["quality_gate_reason"] = str(qual_reason)
                spec["holdout_lite_err"] = None if holdout_lite_err is None else float(holdout_lite_err)
                spec["holdout_lite_reason"] = str(holdout_lite_reason)
                spec["novelty_score"] = float(score_info.get("score", 0.0))
                spec["innovation_bonus"] = float(score_info.get("innovation_bonus", 0.0))
                spec["innovation_reasons"] = list(score_info.get("innovation_reasons", []))
                spec["failure_buckets"] = list(score_info.get("failure_buckets", []))
                spec["exact_family_count"] = int(sat_metrics.get("exact_count", 0))
                spec["component_family_count"] = int(sat_metrics.get("component_count", 0))
                spec["novelty_subtype"] = str(novelty_subtype)
                spec["motif_distance_score"] = float(score_info.get("motif_distance_score", 0.0))
                spec["motif_frequency"] = int(score_info.get("motif_frequency", 0))
                spec["arch_type"] = str(score_info.get("arch_type", spec.get("arch_type", "regular")))
                spec.setdefault("architecture_schema", copy.deepcopy(candidate_meta_local.get("architecture_schema", {})))
                spec.setdefault("motif_signature", copy.deepcopy(candidate_meta_local.get("motif_signature", {})))
                spec.setdefault("schema_hash", str(candidate_meta_local.get("schema_hash", "")))
                spec.setdefault("motif_key", str(candidate_meta_local.get("motif_key", "")))
            replace_individual_in_state(
                state,
                idx,
                chk["team"],
                (chk["fit"], chk["err"], chk["case_vec"]),
                source_meta={
                    "phase": "stagnation",
                    "source": spec.get("source", "offline_json"),
                    "rationale": spec.get("rationale", ""),
                    "family_tag": spec.get("family_tag", ""),
                    "channel": channel,
                    "novelty_score": float(spec.get("novelty_score", 0.0)) if channel == "novelty" else None,
                    "holdout_lite_err": spec.get("holdout_lite_err", None) if channel == "novelty" else None,
                    "family_match": int(spec.get("family_match", 0)) if channel == "novelty" else None,
                    "family_distance": int(spec.get("family_distance", 0)) if channel == "novelty" else None,
                    "family_match_score": float(spec.get("family_match_score", 0.0)) if channel == "novelty" else None,
                    "family_distance_score": float(spec.get("family_distance_score", 0.0)) if channel == "novelty" else None,
                    "distance_near_miss_used": bool(spec.get("distance_near_miss_used", False)) if channel == "novelty" else None,
                    "novelty_subtype": str(spec.get("novelty_subtype", "stable")) if channel == "novelty" else None,
                    "innovation_bonus": float(spec.get("innovation_bonus", 0.0)) if channel == "novelty" else None,
                    "architecture_schema": copy.deepcopy(spec.get("architecture_schema", {})),
                    "motif_signature": copy.deepcopy(spec.get("motif_signature", {})),
                    "schema_hash": str(spec.get("schema_hash", "")),
                    "motif_key": str(spec.get("motif_key", "")),
                    "arch_type": str(spec.get("arch_type", "regular")),
                    "motif_distance_score": float(spec.get("motif_distance_score", 0.0)) if channel == "novelty" else None,
                    "motif_frequency": int(spec.get("motif_frequency", 0)) if channel == "novelty" else None,
                },
                fec_key=chk["key"],
            )
            extra = ""
            if channel == "novelty":
                novelty_stats["injected"] += 1
                novelty_stats["accepted_families"][str(_team_family_tag(chk["team"]))] += 1
                extra = (
                    f" subtype={str(spec.get('novelty_subtype', 'stable'))}"
                    f" family_match={int(spec.get('family_match', 0))}"
                    f" family_match_score={float(spec.get('family_match_score', 0.0)):.2f}"
                    f" family_distance={int(spec.get('family_distance', 0))}"
                    f" family_distance_score={float(spec.get('family_distance_score', 0.0)):.2f}"
                    f" distance_near_miss={int(bool(spec.get('distance_near_miss_used', False)))}"
                    f" novelty_score={float(spec.get('novelty_score', 0.0)):.2f}"
                    f" innovation_bonus={float(spec.get('innovation_bonus', 0.0)):.2f}"
                    f" motif_distance_score={float(spec.get('motif_distance_score', 0.0)):.2f}"
                    f" motif_freq={int(spec.get('motif_frequency', 0))}"
                    f" arch_type={str(spec.get('arch_type', 'regular'))}"
                    f" holdout_lite_err={float(spec.get('holdout_lite_err', 0.0)) if spec.get('holdout_lite_err', None) is not None else 'NA'}"
                    f" gate={spec.get('quality_gate_reason', '')}"
                )
                llm_engine.logger.info(
                    "novelty candidate injected",
                    island=int(state.get("island_idx", 0)),
                    idx=int(idx),
                    family_tag=str(_team_family_tag(chk["team"])),
                    novelty_score=float(spec.get("novelty_score", 0.0)),
                    holdout_lite_err=spec.get("holdout_lite_err", None),
                    target_match=int(spec.get("family_match", 0)),
                    family_distance=int(spec.get("family_distance", 0)),
                    distance_near_miss_used=bool(spec.get("distance_near_miss_used", False)),
                    novelty_subtype=str(spec.get("novelty_subtype", "stable")),
                    innovation_bonus=float(spec.get("innovation_bonus", 0.0)),
                    motif_distance_score=float(spec.get("motif_distance_score", 0.0)),
                    motif_frequency=int(spec.get("motif_frequency", 0)),
                    arch_type=str(spec.get("arch_type", "regular")),
                )
            print(f"[LLM_IMMIGRANT_APPLY] channel={channel} idx={idx} fit={float(chk['fit']):.6f} err={float(chk['err']):.6f}{extra}", flush=True)
            inserted += 1
            accepted_specs.append(spec)
        except Exception as e:
            if channel == "novelty":
                _record_llm_novelty_reject(state, f"inject_exception:{e}")
            continue
    return state, inserted, accepted_specs


def _make_evaluator_from_cfg(cfg):
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


def _collect_top_stage1_candidates_from_states(island_states, evaluator, top_k: int = 5):
    top_k = max(1, int(top_k))
    pool = {}
    for st in island_states or []:
        fits = list(st.get("fits", []))
        case_vecs = list(st.get("case_vecs", []))
        pops = st.get("pops", {})
        n = len(fits)
        for i in range(n):
            try:
                team = {
                    "init_dex": copy.deepcopy(pops["init_dex"][i]),
                    "update": copy.deepcopy(pops["update"][i]),
                    "query": copy.deepcopy(pops["query"][i]),
                }
                key = evaluator._canonical_triplet_key(team["init_dex"], team["update"], team["query"])
                fit_i, err_i = fits[i]
                rec = {
                    "team": team,
                    "stage1_fit": float(fit_i),
                    "stage1_err": float(err_i),
                    "case_vec": tuple(float(x) for x in (case_vecs[i] if i < len(case_vecs) else ())),
                    "island_idx": int(st.get("island_idx", -1)),
                }
                old = pool.get(key)
                if old is None or (rec["stage1_fit"] > old["stage1_fit"]) or (rec["stage1_fit"] == old["stage1_fit"] and rec["stage1_err"] < old["stage1_err"]):
                    pool[key] = rec
            except Exception:
                continue
    vals = list(pool.values())
    vals.sort(key=lambda d: (-float(d["stage1_fit"]), float(d["stage1_err"]), int(d.get("island_idx", -1))))
    return vals[:top_k]




def _team_expr_key_for_holdout(team):
    return (str(team["init_dex"]), str(team["update"]), str(team["query"]))


def _append_historical_best_to_holdout_candidates(candidates, best_team, best_fitness, best_error):
    candidates = list(candidates or [])
    if not best_team:
        return candidates

    best_rec = {
        "team": {
            "init_dex": copy.deepcopy(best_team["init_dex"]),
            "update": copy.deepcopy(best_team["update"]),
            "query": copy.deepcopy(best_team["query"]),
        },
        "stage1_fit": float(best_fitness),
        "stage1_err": float(best_error),
        "case_vec": tuple(),
        "island_idx": -1,
        "from_history_best": True,
    }

    best_key = _team_expr_key_for_holdout(best_rec["team"])
    found_same = False
    for cand in candidates:
        try:
            cand_key = _team_expr_key_for_holdout(cand["team"])
        except Exception:
            continue
        if cand_key == best_key:
            cand["from_history_best"] = True
            # 保留更好的 stage1 指标，避免日志里看到同一表达式却用了较差的记录
            try:
                if float(best_rec["stage1_fit"]) > float(cand.get("stage1_fit", -1e18)):
                    cand["stage1_fit"] = float(best_rec["stage1_fit"])
                if float(best_rec["stage1_err"]) < float(cand.get("stage1_err", 2_000_000_000.0)):
                    cand["stage1_err"] = float(best_rec["stage1_err"])
            except Exception:
                pass
            found_same = True
            break

    if not found_same:
        candidates.append(best_rec)
    return candidates

def _build_real_evaluator_from_stage_cfg(args_dict, *, pkts, files, start, shuffle, dataset_seed, fixed_stream_path):
    ev = CMSketchEvaluator(
        dataset_root=args_dict["dataset_root"],
        pkts=int(pkts),
        max_files=int(files),
        start=int(start),
        shuffle=bool(shuffle),
        seed=int(dataset_seed) & 0xFFFFFFFF,
        dataset_mode="real",
        proxy_mode="proxy_balanced",
        proxy_pool_mul=args_dict["proxy_pool_mul"],
        proxy_min_u=args_dict["proxy_min_u"],
        fixed_stream_path=str(fixed_stream_path or ""),
    )
    ev.E0 = 1.0
    return ev


def _rerank_candidates_with_real_holdout(args_dict, candidates, log_prefix="[STAGE2_HOLDOUT]"):
    candidates = list(candidates or [])
    if not candidates:
        return None, [], None

    holdout_ev = _build_real_evaluator_from_stage_cfg(
        args_dict,
        pkts=args_dict["stage2_holdout_pkts"],
        files=args_dict["stage2_holdout_files"],
        start=args_dict["stage2_holdout_start"],
        shuffle=args_dict["stage2_holdout_shuffle"],
        dataset_seed=args_dict["stage2_holdout_dataset_seed"],
        fixed_stream_path=args_dict["stage2_holdout_fixed_stream"],
    )

    scored = []
    for rank, cand in enumerate(candidates, start=1):
        team = cand["team"]
        try:
            _, holdout_err = holdout_ev.evaluate_individual(
                team["init_dex"],
                team["update"],
                team["query"],
            )
        except Exception:
            holdout_err = 2_000_000_000.0
        rec = dict(cand)
        rec["holdout_err"] = float(holdout_err)
        scored.append(rec)
        src_tag = " [HISTORY_BEST]" if bool(rec.get("from_history_best", False)) else ""
        print(
            f"{log_prefix} rank={rank} stage1_fit={rec['stage1_fit']:.6f} stage1_err={rec['stage1_err']:.2f} holdout_err={rec['holdout_err']:.2f}{src_tag}",
            flush=True,
        )

    scored.sort(key=lambda d: (float(d["holdout_err"]), -float(d["stage1_fit"]), float(d["stage1_err"])))
    chosen = scored[0] if scored else None
    if chosen is not None:
        chosen_tag = " [HISTORY_BEST]" if bool(chosen.get("from_history_best", False)) else ""
        print(
            f"{log_prefix} chosen_holdout_err={chosen['holdout_err']:.2f} chosen_stage1_fit={chosen['stage1_fit']:.6f} chosen_stage1_err={chosen['stage1_err']:.2f}{chosen_tag}",
            flush=True,
        )
    return chosen, scored, holdout_ev

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
    island_profile = _get_island_profile(cfg, island_idx)
    init_p_skeleton = float(island_profile.get("init_p_skeleton", cfg.get("init_p_skeleton", 0.70)))
    init_p_seed = float(island_profile.get("init_p_seed", cfg.get("init_p_seed", 0.20)))
    llm_seed_prob = float(island_profile.get("init_p_llm_seed", 0.0))
    fam_allowed = island_profile.get("allowed_family_labels", {})

    pops = {
        'init_dex': [
            _init_individual_from_ctx(
                gp_ctx,
                'init_dex',
                p_skeleton=init_p_skeleton,
                p_seed=init_p_seed,
                p_llm_seed=llm_seed_prob,
                allowed_family_labels=fam_allowed.get('init_dex', set()),
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
                allowed_family_labels=fam_allowed.get('update', set()),
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
                allowed_family_labels=fam_allowed.get('query', set()),
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
        f"[DIAG_INIT] island={island_idx} profile={island_profile.get('name','baseline')} "
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
        'island_profile': copy.deepcopy(island_profile),
                "hard_case_state": evaluator.export_hard_case_state() if cfg.get("hard_case_replay", False) else {"version": 0, **evaluator._empty_hard_case_state()},
        "scored_hard_case_version": 0,
        "recent_diag_history": [copy.deepcopy(dbg)],
        "llm_novelty_stats": _empty_llm_novelty_stats(),
        "innovation_archive": [],
        "recent_repair_history": [],
        "structure_keep_quota": int(island_profile.get("structure_keep_quota", 2 if str(island_profile.get("role", "")) == "irregular_architecture" else 1)),
    }
    state = _rebuild_innovation_archive_from_population(state)
    state = _rebuild_state_fec_index(state, evaluator)
    _, best_fit, best_err, _ = _refresh_island_best(state)
    state['best_fitness'] = float(best_fit)
    state['best_error'] = float(best_err)

    # 初始化后按比例混入 llm_seed_bank（team 级），并同步刷新 pop/fits/case_vecs/birth 等字段
    try:
        state, seed_inserted = _apply_llm_seed_specs_to_state(state, cfg, gp_ctx, evaluator, target_funcs_override=cfg.get("llm_target_funcs", "update,query"))
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

    island_profile = copy.deepcopy(state.get("island_profile", _get_island_profile(cfg, island_idx)))
    fam_allowed = island_profile.get("allowed_family_labels", {})

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
            mut_weights = island_profile.get("mutation_weights", {"init_dex": 1.0, "update": 1.0, "query": 1.0})
            reset_weights = island_profile.get("reset_weights", {"init_dex": 1.0, "update": 1.0, "query": 1.0})
            family_jump_prob = float(island_profile.get("family_jump_prob", 0.0))
            family_jump_weights = island_profile.get("family_jump_weights", mut_weights)
            family_jump_done = False

            if random.random() < family_jump_prob:
                which = _choose_weighted_key(family_jump_weights, default_key="query")
                child[which] = _init_individual_from_ctx(
                    gp_ctx,
                    which,
                    p_skeleton=0.15,
                    p_seed=0.55,
                    p_llm_seed=0.30 if _is_llm_seed_mode_enabled(cfg) else 0.0,
                    allowed_family_labels=fam_allowed.get(which, set()),
                )
                family_jump_done = True

            if random.random() < _reset_whole_prob:
                which = _choose_weighted_key(reset_weights, default_key="query")
                child[which] = _init_individual_from_ctx(
                    gp_ctx,
                    which,
                    p_skeleton=float(island_profile.get("init_p_skeleton", cfg.get("init_p_skeleton", 0.70))),
                    p_seed=float(island_profile.get("init_p_seed", cfg.get("init_p_seed", 0.20))),
                    p_llm_seed=float(island_profile.get("init_p_llm_seed", 0.0)),
                    allowed_family_labels=fam_allowed.get(which, set()),
                )
            else:
                if random.random() < _reset_prob:
                    which = _choose_weighted_key(reset_weights, default_key="query")
                    child[which] = _init_individual_from_ctx(
                        gp_ctx,
                        which,
                        p_skeleton=float(island_profile.get("init_p_skeleton", cfg.get("init_p_skeleton", 0.70))),
                        p_seed=float(island_profile.get("init_p_seed", cfg.get("init_p_seed", 0.20))),
                        p_llm_seed=float(island_profile.get("init_p_llm_seed", 0.0)),
                        allowed_family_labels=fam_allowed.get(which, set()),
                    )

                if random.random() < _mutation_prob:
                    which = _choose_weighted_key(mut_weights, default_key="query")
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
        f"[DIAG_CHUNK] island={island_idx} profile={island_profile.get('name','baseline')} local_gens={int(local_gens)} "
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
        llm_single_tree_mode: str = "stagnation",
        llm_single_tree_target: str = "update",
        llm_single_tree_max_ratio: float = 1.5,
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
        return_top_candidates: bool = False,
        final_stage1_topk: int = 5,
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
        baseline_eval = CMSketchEvaluator(
            dataset_root=dataset_root,
            pkts=pkts,
            max_files=max_files,
            start=start,
            shuffle=shuffle,
            seed=int(dataset_seed) & 0xFFFFFFFF,
            dataset_mode=dataset_mode,
            proxy_mode=proxy_mode,
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
        'llm_single_tree_mode': str(llm_single_tree_mode),
        'llm_single_tree_target': str(llm_single_tree_target),
        'llm_single_tree_max_ratio': float(llm_single_tree_max_ratio),
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
    cfg['llm_single_tree_mode'] = str(cfg.get('llm_single_tree_mode', 'stagnation') or 'stagnation').strip().lower()
    cfg['llm_single_tree_target'] = str(cfg.get('llm_single_tree_target', 'update') or 'update').strip().lower()
    cfg['llm_seed_specs'] = []

    gp_ctx_main = _build_gp_context(max_size=cfg["max_size"])
    llm_logger = LLMRunLogger(cfg.get("llm_log_path", ""))
    llm_ref = PrimitiveSpecReference(gp_ctx_main["pset_map"], cfg, llm_logger)
    llm_engine = LLMProposalEngine(cfg, llm_ref, llm_logger)
    main_eval_for_llm = _make_evaluator_from_cfg(cfg)

    # seed 阶段候选在主进程预取与预校验，worker 只做注入替换，避免热路径联网
    try:
        if bool(cfg.get("llm_enable", False)) and str(cfg.get("llm_mode", "none")) in {"seeds", "both"}:
            seed_limit = max(1, int(cfg.get("llm_seed_max", 0) or 8))
            per_profile_limit = max(1, int(math.ceil(seed_limit / max(1, int(islands)))))
            seed_candidates = []
            seed_seen = set()
            global_seed_hist = _empty_family_histogram()
            for prof_idx in range(max(1, int(islands))):
                island_profile = _get_island_profile(cfg, prof_idx)
                family_guidance = _build_family_guidance(island_profile, global_seed_hist, available_specs=[], failure_buckets=[])
                base_seed_team = {
                    "init_dex": _skeleton_individual_from_ctx(gp_ctx_main, "init_dex"),
                    "update": _skeleton_individual_from_ctx(gp_ctx_main, "update"),
                    "query": _skeleton_individual_from_ctx(gp_ctx_main, "query"),
                }
                cur_candidates = llm_engine.prepare_phase_candidates(
                    phase="seed",
                    gp_ctx=gp_ctx_main,
                    evaluator=main_eval_for_llm,
                    base_team=base_seed_team,
                    existing_canon=seed_seen,
                    limit=per_profile_limit,
                    family_guidance=family_guidance,
                )
                for c in cur_candidates:
                    try:
                        c_key = main_eval_for_llm._canonical_triplet_key(c["team"]["init_dex"], c["team"]["update"], c["team"]["query"])
                    except Exception:
                        c_key = None
                    if c_key is not None and c_key in seed_seen:
                        continue
                    if c_key is not None:
                        seed_seen.add(c_key)
                    c["team"]["family_parts"] = _team_family_parts(c["team"])
                    c["team"]["family_tag"] = _team_family_tag(c["team"])
                    seed_candidates.append(c)
                    global_seed_hist = _record_team_family(global_seed_hist, c["team"])
                    if len(seed_candidates) >= seed_limit:
                        break
                if len(seed_candidates) >= seed_limit:
                    break
            for i, c in enumerate(seed_candidates):
                llm_logger.info(
                    "seed candidate accepted",
                    idx=int(i),
                    source=str(c.get("source", "offline_json")),
                    fitness=float(c.get("fit", 0.0)),
                    error=float(c.get("err", 0.0)),
                    family_tag=str(c["team"].get("family_tag", "")),
                )
            cfg["llm_seed_specs"] = [
                _serialize_team_spec({**c["team"], "family_tag": _team_family_tag(c["team"]), "family_parts": _team_family_parts(c["team"])}, rationale=c.get("rationale", ""), source=c.get("source", "offline_json"))
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
                    repair_limit = max(1, int(math.ceil(cand_limit * 0.5)))
                    irregular_limit = 0
                    if bool(cfg.get("llm_irregular_enable", True)) and cand_limit >= 3:
                        irregular_limit = max(1, int(math.ceil(cand_limit * float(cfg.get("llm_irregular_fraction", 0.25)))))
                    novelty_limit = max(1, cand_limit - repair_limit - irregular_limit)
                    phase_extra_prompt_hints = _collect_recent_failure_hints_from_states(island_states)
                    phase_candidates = []
                    phase_seen = set(existing_keys)
                    global_family_hist = _family_histogram_from_states(island_states)
                    per_profile_repair = max(1, int(math.ceil(repair_limit / max(1, len(island_states)))))
                    per_profile_novelty = max(1, int(math.ceil(novelty_limit / max(1, len(island_states)))))
                    per_profile_irregular = max(1, int(math.ceil(max(0, irregular_limit) / max(1, len(island_states))))) if irregular_limit > 0 else 0
                    for st in island_states:
                        if len(phase_candidates) >= cand_limit:
                            break
                        island_profile_cur = copy.deepcopy(st.get("island_profile", _get_island_profile(cfg, int(st.get("island_idx", 0)))))
                        state_family_hist = _family_histogram_from_state(st)
                        state_hint = _collect_recent_failure_hints_from_states([st])
                        merged_hints = _merge_prompt_hint_dicts(phase_extra_prompt_hints, state_hint)
                        family_guidance = _build_family_guidance(
                            island_profile_cur,
                            state_family_hist if state_family_hist.get("total", 0) > 0 else global_family_hist,
                            available_specs=cfg.get("llm_seed_specs", []),
                            failure_buckets=merged_hints.get("failure_buckets", []),
                            prefer_irregular=False,
                        )
                        irregular_profile = copy.deepcopy(island_profile_cur)
                        if isinstance(irregular_profile, dict):
                            ir_map = dict(irregular_profile.get("innovation_family_labels", {}))
                            if ir_map:
                                irregular_profile["allowed_family_labels"] = ir_map
                        irregular_family_guidance = _build_family_guidance(
                            irregular_profile,
                            state_family_hist if state_family_hist.get("total", 0) > 0 else global_family_hist,
                            available_specs=cfg.get("llm_seed_specs", []),
                            failure_buckets=merged_hints.get("failure_buckets", []),
                            prefer_irregular=True,
                        )
                        repair_allowed_targets = _parse_llm_target_funcs(cfg.get("llm_target_funcs", "update,query"))
                        adaptive_target, adaptive_reason = _adaptive_single_tree_target_from_hints(
                            island_profile_cur,
                            merged_hints,
                            allowed_targets=repair_allowed_targets,
                            fallback=str(cfg.get("llm_single_tree_target", "update")),
                        )
                        repair_phase = llm_engine.prepare_phase_candidates(
                            phase="stagnation",
                            gp_ctx=gp_ctx_main,
                            evaluator=main_eval_for_llm,
                            base_team=base_team_for_llm,
                            existing_canon=phase_seen,
                            limit=min(per_profile_repair, cand_limit - len(phase_candidates)),
                            extra_prompt_hints=merged_hints,
                            family_guidance=family_guidance,
                            force_single_tree=True,
                            force_single_tree_target=adaptive_target,
                            target_funcs_override=cfg.get("llm_target_funcs", "update,query"),
                            candidate_channel="repair",
                            adaptive_reason=adaptive_reason,
                        )
                        for c in repair_phase:
                            try:
                                c_key = main_eval_for_llm._canonical_triplet_key(c["team"]["init_dex"], c["team"]["update"], c["team"]["query"])
                            except Exception:
                                c_key = None
                            if c_key is not None and c_key in phase_seen:
                                continue
                            if c_key is not None:
                                phase_seen.add(c_key)
                            c["team"]["family_parts"] = _team_family_parts(c["team"])
                            c["team"]["family_tag"] = _team_family_tag(c["team"])
                            c["channel"] = "repair"
                            c["adaptive_reason"] = str(adaptive_reason)
                            phase_candidates.append(c)
                            if len(phase_candidates) >= cand_limit:
                                break
                        if len(phase_candidates) >= cand_limit:
                            continue
                        novelty_phase = []
                        irregular_phase = []
                        if str(island_profile_cur.get("name", "")) != "baseline":
                            novelty_phase = llm_engine.prepare_phase_candidates(
                                phase="stagnation",
                                gp_ctx=gp_ctx_main,
                                evaluator=main_eval_for_llm,
                                base_team=base_team_for_llm,
                                existing_canon=phase_seen,
                                limit=min(per_profile_novelty, cand_limit - len(phase_candidates)),
                                extra_prompt_hints=merged_hints,
                                family_guidance=family_guidance,
                                force_single_tree=False,
                                target_funcs_override="init_dex,update,query",
                                candidate_channel="novelty",
                            )
                            if irregular_limit > 0 and len(phase_candidates) < cand_limit:
                                irregular_phase = llm_engine.prepare_phase_candidates(
                                    phase="stagnation",
                                    gp_ctx=gp_ctx_main,
                                    evaluator=main_eval_for_llm,
                                    base_team=base_team_for_llm,
                                    existing_canon=phase_seen,
                                    limit=min(per_profile_irregular, cand_limit - len(phase_candidates)),
                                    extra_prompt_hints=merged_hints,
                                    family_guidance=irregular_family_guidance,
                                    force_single_tree=False,
                                    target_funcs_override="init_dex,update,query",
                                    candidate_channel="irregular_novelty",
                                )
                        for c in novelty_phase:
                            try:
                                c_key = main_eval_for_llm._canonical_triplet_key(c["team"]["init_dex"], c["team"]["update"], c["team"]["query"])
                            except Exception:
                                c_key = None
                            if c_key is not None and c_key in phase_seen:
                                continue
                            if c_key is not None:
                                phase_seen.add(c_key)
                            c["team"]["family_parts"] = _team_family_parts(c["team"])
                            c["team"]["family_tag"] = _team_family_tag(c["team"])
                            c["channel"] = "novelty"
                            c["novelty_subtype"] = str(c.get("novelty_subtype", "stable"))
                            phase_candidates.append(c)
                            if len(phase_candidates) >= cand_limit:
                                break
                        for c in irregular_phase:
                            try:
                                c_key = main_eval_for_llm._canonical_triplet_key(c["team"]["init_dex"], c["team"]["update"], c["team"]["query"])
                            except Exception:
                                c_key = None
                            if c_key is not None and c_key in phase_seen:
                                continue
                            if c_key is not None:
                                phase_seen.add(c_key)
                            c["team"]["family_parts"] = _team_family_parts(c["team"])
                            c["team"]["family_tag"] = _team_family_tag(c["team"])
                            c["channel"] = "irregular_novelty"
                            c["novelty_subtype"] = "irregular"
                            phase_candidates.append(c)
                            if len(phase_candidates) >= cand_limit:
                                break
                    for i, c in enumerate(phase_candidates):
                        llm_logger.info(
                            "stagnation candidate accepted",
                            idx=int(i),
                            source=str(c.get("source", "offline_json")),
                            fitness=float(c.get("fit", 0.0)),
                            error=float(c.get("err", 0.0)),
                            family_tag=str(c["team"].get("family_tag", _team_family_tag(c["team"]))),
                            channel=str(c.get("channel", "repair")),
                            adaptive_reason=str(c.get("adaptive_reason", "")),
                        )
                    candidate_specs = []
                    for c in phase_candidates:
                        spec = _serialize_team_spec({**c["team"], "family_tag": _team_family_tag(c["team"]), "family_parts": _team_family_parts(c["team"])}, rationale=c.get("rationale", ""), source=c.get("source", "offline_json"))
                        spec["channel"] = str(c.get("channel", "repair"))
                        spec["edit_mode"] = str(c.get("edit_mode", "team"))
                        spec["fit"] = float(c.get("fit", 0.0))
                        spec["err"] = float(c.get("err", 1e18))
                        spec["target_funcs_override"] = cfg.get("llm_target_funcs", "update,query") if spec["channel"] == "repair" else "init_dex,update,query"
                        spec["adaptive_reason"] = str(c.get("adaptive_reason", ""))
                        candidate_specs.append(spec)
                    budget = max(0, int(cfg.get("llm_stagnation_max_inject", 2)))
                    budget = min(budget, len(candidate_specs), len(island_states))
                    total_injected = 0
                    repair_injected = 0
                    novelty_injected = 0
                    if candidate_specs and budget > 0:
                        repair_specs = [sp for sp in candidate_specs if _llm_candidate_channel(sp) == "repair"]
                        novelty_specs = [sp for sp in candidate_specs if _llm_candidate_channel(sp) == "novelty"]
                        repair_budget = 1 if repair_specs and budget > 0 else 0
                        novelty_budget = 1 if novelty_specs and (budget - repair_budget) > 0 else 0
                        remaining_budget = max(0, budget - repair_budget - novelty_budget)
                        if remaining_budget > 0:
                            if len(repair_specs) >= len(novelty_specs):
                                repair_budget += remaining_budget
                            else:
                                novelty_budget += remaining_budget

                        def _apply_channel(island_states_local, specs_local, slot_budget, target_funcs_override, channel_name):
                            if slot_budget <= 0 or not specs_local:
                                return island_states_local, 0
                            new_states_local = []
                            remain_local = int(slot_budget)
                            remaining_specs_local = list(specs_local)
                            injected_local = 0
                            for st_local in island_states_local:
                                prof_local = copy.deepcopy(st_local.get("island_profile", _get_island_profile(cfg, int(st_local.get("island_idx", 0)))))
                                if channel_name == "novelty" and str(prof_local.get("name", "")) == "baseline":
                                    new_states_local.append(st_local)
                                    continue
                                if remain_local <= 0 or not remaining_specs_local:
                                    new_states_local.append(st_local)
                                    continue
                                st_local, inserted_local, accepted_specs_local = _inject_llm_immigrants_with_engine(
                                    st_local,
                                    cfg,
                                    gp_ctx_main,
                                    llm_engine,
                                    remaining_specs_local,
                                    success_budget=1,
                                    target_funcs_override=target_funcs_override,
                                )
                                injected_local += int(inserted_local)
                                remain_local = max(0, remain_local - int(inserted_local))
                                if accepted_specs_local:
                                    accepted_ids_local = {id(spec) for spec in accepted_specs_local}
                                    remaining_specs_local = [spec for spec in remaining_specs_local if id(spec) not in accepted_ids_local]
                                new_states_local.append(st_local)
                            return new_states_local, injected_local

                        island_states, repair_injected = _apply_channel(island_states, repair_specs, repair_budget, cfg.get("llm_target_funcs", "update,query"), "repair")
                        island_states, novelty_injected = _apply_channel(island_states, novelty_specs, novelty_budget, "init_dex,update,query", "novelty")
                        total_injected = int(repair_injected) + int(novelty_injected)

                    novelty_summary = _aggregate_llm_novelty_stats(island_states)
                    top_novelty_rejects = novelty_summary.get("rejected_reasons", Counter()).most_common(3)
                    print(
                        f"[LLM_IMMIGRANT_TRIGGER] after_gen={gens_done} stagnation_chunks={no_improve_chunks} "
                        f"candidates={len(candidate_specs)} budget={budget} repair_injected={repair_injected} novelty_injected={novelty_injected} injected={total_injected} "
                        f"novelty_proposed={int(novelty_summary.get('proposed', 0))} "
                        f"novelty_validated={int(novelty_summary.get('validated', 0))} "
                        f"novelty_pass_family={int(novelty_summary.get('pass_family', 0))} "
                        f"novelty_pass_saturation={int(novelty_summary.get('pass_saturation', 0))} "
                        f"novelty_pass_quality={int(novelty_summary.get('pass_quality', 0))} "
                        f"novelty_pass_holdout={int(novelty_summary.get('pass_holdout', 0))} "
                        f"novelty_pass_score={int(novelty_summary.get('pass_score', 0))} "
                        f"novelty_reject_top3={top_novelty_rejects}",
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

    top_candidates = _collect_top_stage1_candidates_from_states(
        island_states,
        main_eval_for_llm,
        top_k=max(1, int(final_stage1_topk)),
    )
    top_candidates = _append_historical_best_to_holdout_candidates(
        top_candidates,
        best_team,
        best_fitness,
        best_error,
    )
    if return_top_candidates:
        return best_team, best_fitness, best_error, top_candidates
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

    if str(args_dict.get("stage1_dataset_mode", "proxy")) == "proxy":
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
            best_team, best_fitness, best_error, top_candidates = evolve_cmsketch(
                population_size=args_dict["pop"],
                generations=args_dict["gen"],
                seed=run_seed,
                dataset_root=args_dict["dataset_root"],
                pkts=args_dict["pkts"],
                max_files=args_dict["files"],
                start=args_dict["start"],
                shuffle=args_dict["shuffle"],
                dataset_mode=args_dict["stage1_dataset_mode"],
                proxy_mode=run_proxy_mode,
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
                llm_single_tree_mode=args_dict["llm_single_tree_mode"],
                llm_single_tree_target=args_dict["llm_single_tree_target"],
                llm_single_tree_max_ratio=args_dict["llm_single_tree_max_ratio"],
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
                fixed_stream_path=_apply_proxy_mode_to_stream_path(
                    args_dict["stage1_fixed_stream"], args_dict["stage1_dataset_mode"], run_proxy_mode
                ),
                return_top_candidates=True,
                final_stage1_topk=args_dict["stage2_holdout_topk"],
            )
            print(f"[STAGE1] proxy_mode_arg={run_proxy_mode} best_fitness={best_fitness:.6f} best_error={best_error:.2f}")

            # ---------- 阶段2：小 real holdout 复排 + 全量真实流复评 ----------
            selected_team = best_team
            selected_stage1_fit = float(best_fitness)
            selected_stage1_err = float(best_error)

            if bool(args_dict.get("stage2_holdout_enable", False)) and top_candidates:
                holdout_candidates = _append_historical_best_to_holdout_candidates(
                    top_candidates[:max(1, int(args_dict["stage2_holdout_topk"]))],
                    best_team,
                    best_fitness,
                    best_error,
                )
                chosen, holdout_scored, _ = _rerank_candidates_with_real_holdout(
                    args_dict,
                    holdout_candidates,
                    log_prefix="[STAGE2_HOLDOUT]",
                )
                if chosen is not None:
                    selected_team = chosen["team"]
                    selected_stage1_fit = float(chosen["stage1_fit"])
                    selected_stage1_err = float(chosen["stage1_err"])

            val_seed = int(args_dict["stage2_dataset_seed"]) & 0xFFFFFFFF
            val_evaluator = _build_real_evaluator_from_stage_cfg(
                args_dict,
                pkts=args_dict["stage2_pkts"],
                files=args_dict["stage2_files"],
                start=args_dict["stage2_start"],
                shuffle=args_dict["stage2_shuffle"],
                dataset_seed=val_seed,
                fixed_stream_path=args_dict["stage2_fixed_stream"],
            )

            _, stage2_error = val_evaluator.evaluate_individual(
                selected_team["init_dex"],
                selected_team["update"],
                selected_team["query"],
            )
            print(
                f"[STAGE2] real_error={stage2_error:.2f} "
                f"pkts={len(val_evaluator.test_data)} U={val_evaluator.U} U_ratio={val_evaluator.U_ratio:.4f}"
            )

            exprs = {
                "init_dex": str(selected_team["init_dex"]),
                "update": str(selected_team["update"]),
                "query": str(selected_team["query"]),
            }
            best_team = selected_team
            best_fitness = selected_stage1_fit
            best_error = selected_stage1_err

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
    parser.add_argument("--stage2_holdout_enable", action="store_true", default=True,
                        help="是否启用 stage1 top-K 的小 real holdout 复排（默认开启）")
    parser.add_argument("--stage2_holdout_topk", type=int, default=5,
                        help="进入小 real holdout 复排的 stage1 top-K 候选数")
    parser.add_argument("--stage2_holdout_pkts", type=int, default=6000,
                        help="小 real holdout 的包数")
    parser.add_argument("--stage2_holdout_files", type=int, default=8,
                        help="小 real holdout 的文件数")
    parser.add_argument("--stage2_holdout_start", type=int, default=0,
                        help="小 real holdout 的起点")
    parser.add_argument("--stage2_holdout_shuffle", action="store_true",
                        help="小 real holdout 是否 shuffle")
    parser.add_argument("--stage2_holdout_dataset_seed", type=int, default=20250321,
                        help="小 real holdout 的数据抽样 seed")
    parser.add_argument("--stage2_holdout_fixed_stream", type=str, default="",
                        help="小 real holdout 固定流 npy 路径；存在则直接加载，不存在则首次生成并保存")
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
    parser.add_argument("--llm_single_tree_mode", type=str, default="stagnation",
                        choices=["none", "seeds", "stagnation", "both"],
                        help="Prefer single-tree edit proposals in these phases")
    parser.add_argument("--llm_single_tree_target", type=str, default="update",
                        choices=["init_dex", "update", "query"],
                        help="Default target function for single-tree edit mode")
    parser.add_argument("--llm_single_tree_max_ratio", type=float, default=1.5,
                        help="Maximum allowed size ratio for a single-tree edit relative to the base tree")
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
    if not args.stage2_holdout_fixed_stream:
        args.stage2_holdout_fixed_stream = os.path.join(
            "fixed_streams",
            f"stage2_holdout_real_{args.stage2_holdout_pkts}pkts_{args.stage2_holdout_files}f_"
            f"start{args.stage2_holdout_start}_sh{int(args.stage2_holdout_shuffle)}_seed{int(args.stage2_holdout_dataset_seed)}.npy"
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

            best_team, best_fitness, best_error, top_candidates = evolve_cmsketch(
                population_size=args.pop,
                generations=args.gen,
                seed=run_seed,
                dataset_root=args.dataset_root,
                pkts=args.pkts,
                max_files=args.files,
                start=args.start,
                shuffle=args.shuffle,
                dataset_mode=args.stage1_dataset_mode,
                proxy_mode=run_proxy_mode,
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
                llm_single_tree_mode=args.llm_single_tree_mode,
                llm_single_tree_target=args.llm_single_tree_target,
                llm_single_tree_max_ratio=args.llm_single_tree_max_ratio,
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
                fixed_stream_path=_apply_proxy_mode_to_stream_path(
                    args.stage1_fixed_stream, args.stage1_dataset_mode, run_proxy_mode
                ),
                return_top_candidates=True,
                final_stage1_topk=args.stage2_holdout_topk,
            )

            print(f"[RUN {r + 1}][STAGE1] best_fitness={best_fitness:.6f} best_error={best_error:.2f}")

            selected_team = best_team
            selected_stage1_fit = float(best_fitness)
            selected_stage1_err = float(best_error)

            if bool(args.stage2_holdout_enable) and top_candidates:
                holdout_candidates = _append_historical_best_to_holdout_candidates(
                    top_candidates[:max(1, int(args.stage2_holdout_topk))],
                    best_team,
                    best_fitness,
                    best_error,
                )
                chosen, holdout_scored, _ = _rerank_candidates_with_real_holdout(
                    vars(args),
                    holdout_candidates,
                    log_prefix=f"[RUN {r + 1}][STAGE2_HOLDOUT]",
                )
                if chosen is not None:
                    selected_team = chosen["team"]
                    selected_stage1_fit = float(chosen["stage1_fit"])
                    selected_stage1_err = float(chosen["stage1_err"])

            val_seed = int(args.stage2_dataset_seed) & 0xFFFFFFFF
            val_evaluator = _build_real_evaluator_from_stage_cfg(
                vars(args),
                pkts=args.stage2_pkts,
                files=args.stage2_files,
                start=args.stage2_start,
                shuffle=args.stage2_shuffle,
                dataset_seed=val_seed,
                fixed_stream_path=args.stage2_fixed_stream,
            )

            _, stage2_error = val_evaluator.evaluate_individual(
                selected_team["init_dex"],
                selected_team["update"],
                selected_team["query"]
            )

            print(
                f"[RUN {r + 1}][STAGE2] proxy_mode={run_proxy_mode} "
                f"real_error={stage2_error:.2f} "
                f"pkts={len(val_evaluator.test_data)} U={val_evaluator.U} U_ratio={val_evaluator.U_ratio:.4f}"
            )

            exprs = {
                "init_dex": str(selected_team["init_dex"]),
                "update": str(selected_team["update"]),
                "query": str(selected_team["query"]),
            }
            best_team = selected_team
            best_fitness = selected_stage1_fit
            best_error = selected_stage1_err

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
