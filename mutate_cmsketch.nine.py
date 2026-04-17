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
from operator import attrgetter
from glob import glob
import numpy as np
import concurrent.futures as cf
import multiprocessing as mp
import json
import contextlib

# 导入三个原语集
hash_functions = [hashlib.md5, hashlib.sha1, hashlib.sha256]
MAX_COUNTER = (1 << 32) - 1
INF = 1 << 60
_GLOBAL_EVALUATOR = None


def _eval_triplet_safe(triple):
    """triple = (init_tree, update_tree, query_tree) -> (fitness, error)"""
    global _GLOBAL_EVALUATOR
    init_t, upd_t, qry_t = triple
    try:
        return _GLOBAL_EVALUATOR.evaluate_individual(init_t, upd_t, qry_t)
    except Exception:
        return (0.0, 2_000_000_000.0)


def _eval_triplet_stage_safe(payload):
    """payload = (stage_idx, (init_tree, update_tree, query_tree)) -> (fitness, error)"""
    global _GLOBAL_EVALUATOR
    stage_idx, triple = payload
    init_t, upd_t, qry_t = triple
    try:
        return _GLOBAL_EVALUATOR.evaluate_individual_stage(init_t, upd_t, qry_t, stage_idx=stage_idx)
    except Exception:
        return (0.0, 2_000_000_000.0)


def _parallel_eval_triplets_stage(triples, stage_idx, evaluator, eval_pool, eval_workers):
    """批量评估某一 stage；返回 [(fitness, error), ...]。fitness 始终按主进程当前 E0 重算。"""
    if not triples:
        return []

    raw = []
    if eval_pool is None or int(eval_workers) <= 1 or len(triples) == 1:
        for init_t, upd_t, qry_t in triples:
            try:
                raw.append(evaluator.evaluate_individual_stage(init_t, upd_t, qry_t, stage_idx=stage_idx))
            except Exception:
                raw.append((0.0, 2_000_000_000.0))
    else:
        payloads = [(int(stage_idx), t) for t in triples]
        chunksize = max(1, len(payloads) // max(1, int(eval_workers) * 4))
        raw = list(eval_pool.map(_eval_triplet_stage_safe, payloads, chunksize=chunksize))

    out = []
    for _, err in raw:
        err = float(err)
        fit = float(evaluator._norm_fitness(err))
        out.append((fit, err))
    return out


def _parallel_eval_triplets(triples, evaluator, eval_pool, eval_workers):
    """批量评估入口：升级为种群级 top-fraction K-part。"""
    if not triples:
        return []
    return evaluator.evaluate_population_kpart(triples, eval_pool=eval_pool, eval_workers=eval_workers)


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
                 dataset_root: str = "/data/8T/xgr/traces/univ2_trace/univ2_npy",
                 pkts: int = 30000,
                 max_files: int = 1,
                 start: int = 0,
                 shuffle: bool = False,
                 seed: int = 0,
                 dataset_mode: str = "real",
                 proxy_mode: str = "proxy_balanced",
                 proxy_pool_mul: int = 8,
                 proxy_min_u: int = 2500):
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
        # 读取真实数据集的 flowid（字符串化后供 GP 的字符串/哈希原语使用）
        self.test_data = self._load_univ2_flow_stream()
        self.expected_freq = self._calculate_expected_freq()
        self.U = len(set(self.test_data))
        self.U_ratio = self.U / max(1, len(self.test_data))
        print(f"[DATA] pkts={len(self.test_data)} U={self.U} U_ratio={self.U_ratio:.4f}", flush=True)

        # 归一化尺度（E0）：用于把误差映射到 (0,1] 的适应度；在初始种群评估后由 evolve_cmsketch 设定。
        self.E0 = None
        self.eval_cache = {}
        self.fec_cache = {}
        self.fec_hits = 0
        self.fec_misses = 0

        self._build_kpart_views()
        self.kpart_keep_fracs = [0.60, 0.32, 1.0]
        self.kpart_query_limits = [192, 768, None]
        self.kpart_upd_min = [1.6, 1.9, 2.1]
        self.kpart_upd_penalty_scale = [50000, 40000, 30000]
        self.kpart_avg_err_thresh = [6.0, 2.0, None]
        self.kpart_avg_err_scale = [1200.0, 800.0, 0.0]
        self.kpart_cut_penalty = [220000.0, 300000.0, None]
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
                print(
                    f"[DATA_REBIND_PROXY] pkts={len(self.test_data)} "
                    f"U={self.U} U_ratio={self.U_ratio:.4f}",
                    flush=True
                )

        # ---- fec probe / absent keys 也跟着新的主统计口径走 ----
        self.fec_probe_update_n = min(512, len(self.test_data))
        self.fec_probe_present_n = min(96, len(self.expected_freq))
        self.fec_probe_absent_n = 32
        self.fec_absent_keys = self._build_fec_absent_keys(self.fec_probe_absent_n)
        self.stage_eval_cache = {0: {}, 1: {}}

    def _find_flowid_files(self):
        """在 dataset_root 下寻找 flowid 的 .npy 文件（优先 *.flowid.npy）。
        兼容两种组织：
        - dataset_root 本身就是 univ2_npy 目录
        - dataset_root 是 univ2_trace 根目录，内部有 univ2_npy 子目录
        """
        roots = [self.dataset_root]
        sub = os.path.join(self.dataset_root, "univ2_npy")
        if os.path.isdir(sub):
            roots.append(sub)

        patterns = ["*.flowid.npy", "*flowid*.npy"]
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
                f"在 {self.dataset_root} 下找不到可用的 flowid .npy 文件（优先找 *.flowid.npy）。\n"
                "你现在的 univ2_flowid 位置看起来像：/data/8T/xgr/traces/univ2_trace/univ2_npy/univ2_pt*.flowid.npy\n"
                "建议：把 --dataset_root 指到 univ2_npy 目录，或把 flowid.npy 链接/拷贝到 univ2_trace 根目录。"
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
                    arr = np.load(fp)

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
                f"[DATA_RETRY] retry_times={retry_times} best_U={best_u} target_min_U={target_min_u}",
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
                arr = np.load(fp)

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

        # 高U + 少量热流
        if mode == "proxy_balanced":
            u_goal = min(len(uniq), max(int(target * 0.70), min(self.proxy_min_u, target)))
            base = uniq[:u_goal]

            remain = target - len(base)
            hot = base[:min(len(base), max(32, u_goal // 20))]
            extra = [rng.choice(hot) for _ in range(max(0, remain))]

            stream = base + extra
            rng.shuffle(stream)
            return stream[:target]

        # 头部流更明显
        if mode == "proxy_head":
            u_goal = min(len(uniq), max(int(target * 0.40), 1500))
            base = uniq[:u_goal]

            remain = target - len(base)
            hot = base[:min(len(base), max(32, u_goal // 16))]
            weights = [len(hot) - i for i in range(len(hot))]
            extra = rng.choices(hot, weights=weights, k=max(0, remain))

            stream = base + extra
            rng.shuffle(stream)
            return stream[:target]

        # 保持较高U，但让字符串在局部更相似
        if mode == "proxy_collision":
            uniq_sorted = sorted(
                uniq,
                key=lambda s: (len(str(s)), str(s)[:4], str(s)[-4:])
            )
            u_goal = min(len(uniq_sorted), max(int(target * 0.55), 2000))
            base = uniq_sorted[:u_goal]

            remain = target - len(base)
            hot = base[:min(len(base), max(64, u_goal // 12))]
            extra = [rng.choice(hot) for _ in range(max(0, remain))]

            stream = base + extra
            rng.shuffle(stream)
            return stream[:target]

        rng.shuffle(pool)
        return pool[:target]

    def _load_proxy_stream(self):
        files = self._find_flowid_files()
        if not files:
            raise FileNotFoundError(
                f"在 {self.dataset_root} 下找不到可用的 flowid .npy 文件（优先找 *.flowid.npy）。"
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

    def _build_fec_absent_keys(self, k: int):
        seen = set(self.expected_freq.keys())
        bases = list(self.expected_freq.keys())[:max(16, k * 4)]
        if not bases:
            bases = [f"fec_base_{i}" for i in range(max(1, k))]

        out = []
        i = 0
        while len(out) < k:
            base = str(bases[i % len(bases)])
            cand = f"{base}__fec_absent__{i}"
            if cand not in seen:
                out.append(cand)
                seen.add(cand)
            i += 1
        return out

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

        # 三个 stage 的“增量训练包数”
        # 总和约等于 pkts，但分布不同
        n0 = max(1, min(int(self.pkts), max(int(self.pkts * 0.20), 1000)))
        n1 = max(1, min(int(self.pkts) - n0, max(int(self.pkts * 0.30), 1500)))
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

    def _query_error_on_items(self, query_func, expected_freq, max_items=None):
        items = list(expected_freq.items())
        items.sort(key=lambda kv: kv[1], reverse=True)
        if max_items is not None:
            items = items[:max_items]

        total_err = 0.0
        for item, expected in items:
            estimated = query_func(item)
            total_err += abs(estimated - expected)
        avg_err = total_err / max(1, len(items))
        return total_err, avg_err, len(items)

    def _evaluate_individual_core(self, init_dex_tree, update_tree, query_tree, stage_idx: int = 2):
        try:
            stage_idx = int(stage_idx)
            if stage_idx >= 2:
                cache_key = self._canonical_triplet_key(init_dex_tree, update_tree, query_tree)
                cached = self.eval_cache.get(cache_key)
                if cached is not None:
                    return cached
            else:
                cache_key = self._canonical_triplet_key(init_dex_tree, update_tree, query_tree, stage_idx=stage_idx)
                cached = self.stage_eval_cache[stage_idx].get(cache_key)
                if cached is not None:
                    return cached

            def _ret(total_error: float, fec_fp=None):
                total_error = float(total_error)
                res = (self._norm_fitness(total_error), total_error)
                if stage_idx >= 2:
                    self.eval_cache[cache_key] = res
                    if fec_fp is not None:
                        self.fec_cache[fec_fp] = res
                else:
                    self.stage_eval_cache[stage_idx][cache_key] = res
                return res

            penalty = 0

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
                penalty += (2 - hash_calls) * 6000

            # ---- update 约束 ----
            if not update_info["root_ok"]:
                penalty += 120000

            if update_info["forbidden_hits"]:
                penalty += 180000 + 25000 * sum(update_info["forbidden_hits"].values())

            if update_calls == 0:
                penalty += 100000
            elif update_calls < 2:
                penalty += (2 - update_calls) * 8000

            # ---- query 约束 ----
            if not query_info["root_ok"]:
                penalty += 120000

            if query_info["forbidden_hits"]:
                penalty += 260000 + 40000 * sum(query_info["forbidden_hits"].values())

            if query_calls == 0:
                penalty += 100000
            elif query_calls < 2:
                penalty += (2 - query_calls) * 8000

            # ---- AST 级硬合法性检查 ----
            if init_ast_chk["hard_illegal"] or update_ast_chk["hard_illegal"] or query_ast_chk["hard_illegal"]:
                penalty += 450000
                if penalty >= 450000:
                    return _ret(float(penalty))

            # ---- AST 级退化检测 ----
            if not update_eff["depends_on_e"]:
                penalty += 90000

            if not query_eff["depends_on_e"]:
                penalty += 90000

            # 化简后若已经没有真实写入，说明它基本是伪 update
            if update_eff["real_write_calls"] == 0:
                if update_eff["conditional_write_calls"] == 0:
                    penalty += 180000
                else:
                    penalty += 90000

            # 化简后 query_date 太少，说明 query 大概率已经退化
            if query_eff["query_date_calls"] == 0:
                penalty += 180000
            elif query_eff["query_date_calls"] < 2:
                penalty += (2 - query_eff["query_date_calls"]) * 18000

            # ---- AST 级结构约束：按当前原语集语义约束 hash/path id ----
            if init_pat["nonconst_hash_idx"] > 0:
                penalty += init_pat["nonconst_hash_idx"] * 6000

            if update_pat["nonconst_path_idx"] > 0:
                penalty += update_pat["nonconst_path_idx"] * 8000

            if query_pat["nonconst_path_idx"] > 0:
                penalty += query_pat["nonconst_path_idx"] * 8000

            # ---- AST 级结构约束：惩罚把写副作用藏进比较/算术/切片/条件 ----
            if update_pat["bad_write_context"] > 0:
                penalty += update_pat["bad_write_context"] * 45000

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
            if avg_unique_locs < 1.8:
                penalty += int((1.8 - avg_unique_locs) * 12000)

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
                    self.eval_cache[cache_key] = fec_cached
                    return fec_cached
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

                try:
                    stage_query_error, stage_avg_err, final_qn = self._query_error_on_items(
                        query_func,
                        stage_expected[s],
                        max_items=self.kpart_query_limits[s],
                    )
                except Exception:
                    return _ret(2_000_000_000.0, fec_fp=fec_fp)

                err_thr = self.kpart_avg_err_thresh[s]
                if err_thr is not None and stage_avg_err > float(err_thr):
                    penalty += int(stage_avg_err * float(self.kpart_avg_err_scale[s]))

                cut_pen = self.kpart_cut_penalty[s]
                if cut_pen is not None and penalty >= float(cut_pen):
                    return _ret(float(stage_avg_err + penalty / max(1, final_qn)), fec_fp=fec_fp)

            avg_qry_reads = exec_stats["qry_reads"] / max(1, final_qn)
            if avg_qry_reads < 1.8:
                penalty += int((1.8 - avg_qry_reads) * 15000)

            total_error = float(stage_avg_err + penalty / max(1, final_qn))
            return _ret(total_error, fec_fp=fec_fp)

        except Exception as e:
            print(f"DEBUG: 整体评估异常(stage={stage_idx}): {e}")
            total_error = 2_000_000_000.0
            res = (self._norm_fitness(total_error), total_error)
            try:
                if stage_idx >= 2:
                    self.eval_cache[cache_key] = res
                else:
                    self.stage_eval_cache[stage_idx][cache_key] = res
            except Exception:
                pass
            return res

    def evaluate_individual(self, init_dex_tree, update_tree, query_tree):
        return self._evaluate_individual_core(init_dex_tree, update_tree, query_tree, stage_idx=2)

    def evaluate_individual_stage(self, init_dex_tree, update_tree, query_tree, stage_idx: int):
        return self._evaluate_individual_core(init_dex_tree, update_tree, query_tree, stage_idx=stage_idx)

    def evaluate_population_kpart(self, triples, eval_pool=None, eval_workers=1):
        if not triples:
            return []
        n = len(triples)
        if n == 1:
            return [self.evaluate_individual(*triples[0])]

        results = [None] * n
        keep1 = max(1, min(n, int(math.ceil(n * float(self.kpart_keep_fracs[0])))))
        keep2 = max(1, min(keep1, int(math.ceil(n * float(self.kpart_keep_fracs[1])))))

        stage0_res = _parallel_eval_triplets_stage(triples, 0, self, eval_pool, eval_workers)
        rank0 = sorted(range(n), key=lambda i: (stage0_res[i][1], -stage0_res[i][0]))
        surv1 = rank0[:keep1]
        surv1_set = set(surv1)
        off0 = float(self.kpart_stage_err_offset[0])
        for i in range(n):
            if i not in surv1_set:
                err = float(stage0_res[i][1]) + off0
                results[i] = (self._norm_fitness(err), err)

        stage1_triples = [triples[i] for i in surv1]
        stage1_res_local = _parallel_eval_triplets_stage(stage1_triples, 1, self, eval_pool, eval_workers)
        stage1_res = {idx: res for idx, res in zip(surv1, stage1_res_local)}
        rank1 = sorted(surv1, key=lambda i: (stage1_res[i][1], -stage1_res[i][0]))
        surv2 = rank1[:keep2]
        surv2_set = set(surv2)
        off1 = float(self.kpart_stage_err_offset[1])
        for i in surv1:
            if i not in surv2_set:
                err = float(stage1_res[i][1]) + off1
                results[i] = (self._norm_fitness(err), err)

        stage2_triples = [triples[i] for i in surv2]
        stage2_res_local = _parallel_eval_triplets_stage(stage2_triples, 2, self, eval_pool, eval_workers)
        for i, res in zip(surv2, stage2_res_local):
            err = float(res[1])
            results[i] = (self._norm_fitness(err), err)

        for i in range(n):
            if results[i] is None:
                results[i] = (0.0, 2_000_000_000.0)
        return results

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
    patterns = ["*.flowid.npy", "*flowid*.npy"]
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
            arr = np.load(fp)

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


# 判断演化进程
def get_evolution_phase(gen, total_gens):
    """根据当前代数判断进化阶段"""
    progress = gen / total_gens
    if progress < 0.3:
        return "early"  # 早期阶段：探索
    elif progress < 0.7:
        return "mid"  # 中期阶段：优化
    else:
        return "late"  # 后期阶段：精炼


def evolve_cmsketch(
        population_size=300,
        generations=20,
        seed=None,
        # 数据集采样参数
        dataset_root: str = "/data/8T/xgr/traces/univ2_trace/univ2_npy",
        pkts: int = 30000,
        max_files: int = 1,
        start: int = 0,
        shuffle: bool = False,
        dataset_mode: str = "real",
        proxy_mode: str = "proxy_balanced",
        proxy_pool_mul: int = 8,
        proxy_min_u: int = 2500,
        # 三大件 + 精英保留 + 停滞爆发
        islands: int = 4,
        tournament_size: int = 5,
        elite_rate: float = 0.0,  # 每代强制保留 top-elite_rate 的 team
        reset_prob: float = 0.10,  # 组件重置：随机把某一棵树整棵重采样（大步跳）
        reset_whole_prob: float = 0.02,  # 更强重置：直接重采样某组件，不做其它变异
        recombine_prob: float = 0.0,  # 弱 donor-copy：保留但默认关闭
        crossover_prob: float = 0.25,  # 真正 GP crossover
        leaf_biased_cx_prob: float = 0.50,  # crossover 时选择 leaf-biased 的概率
        mutation_prob: float = 0.90,  # 常规变异概率（对一个随机组件做一次变异）
        stagnation_limit: int = 10,  # 连续多少代无提升触发“爆发期”
        burst_gens: int = 0,  # 爆发持续代数（PPT模式默认关闭）
        immigrant_rate: float = 0.0,  # 爆发期每代向种群注入 immigrants 的比例（默认关闭）
        mixed_seeded_prob: float = 0.0,
        immigration_period: int = 6,  # 周期性随机移民周期
        periodic_immigration_rate: float = 0.08,  # 周期性随机移民比例
        mig_period: int = 8,  # 岛间迁移周期（代）
        mig_k: int =3,  # 每次迁移 top-k team
        max_height: int = 8,
        max_size: int = 80,
        eval_workers: int = 1,
):
    """使用遗传编程演化 CMSketch（只改策略层：三大件 + 精英保留 + 停滞爆发）。
    - 三大件：Aging / Component reset / Migration
    - 精英保留：每代强制把 top elite_rate 的 team 保留下来
    - 停滞爆发：连续 stagnation_limit 代无提升 -> burst_gens 代概率加大 + immigrants 换血
    注意：不改 evaluator 的误差/适应度定义（evaluate_individual 返回 (fitness, error)）。
    """

    if seed is None:
        seed = time.time_ns() % (2 ** 32)
    set_seed(seed)
    print(f"[SEED] {seed}")
    mixed_seeded_prob = max(0.0, min(1.0, float(mixed_seeded_prob)))

    evaluator = CMSketchEvaluator(
        dataset_root=dataset_root,
        pkts=pkts,
        max_files=max_files,
        start=start,
        shuffle=shuffle,
        seed=int(seed) & 0xFFFFFFFF,
        dataset_mode=dataset_mode,
        proxy_mode=proxy_mode,
        proxy_pool_mul=proxy_pool_mul,
        proxy_min_u=proxy_min_u,
    )
    # ====== 单 seed 多核：并行评估个体 ======
    eval_workers = max(1, int(eval_workers))
    eval_pool = None
    if eval_workers > 1:
        global _GLOBAL_EVALUATOR
        _GLOBAL_EVALUATOR = evaluator
        try:
            ctx = mp.get_context("fork")  # Linux 推荐 fork，避免 pickling evaluator
            eval_pool = cf.ProcessPoolExecutor(max_workers=eval_workers, mp_context=ctx)
        except Exception:
            # 兜底：不用 mp_context
            eval_pool = cf.ProcessPoolExecutor(max_workers=eval_workers)
        print(f"[EVAL_PAR] eval_workers={eval_workers}", flush=True)

    # ---- 演化前：输出“标准 CMS(3行, min查询)”在当前数据流上的基线误差 E_base ----
    def _calc_standard_cms_error(test_data, expected_freq, rows=3, cols=10240):
        hash_functions = [hashlib.md5, hashlib.sha1, hashlib.sha256]
        matrix = [[0] * cols for _ in range(rows)]
        for item in test_data:
            b = str(item).encode('utf-8', errors='ignore')
            for i, hf in enumerate(hash_functions):
                y = int(hf(b).hexdigest(), 16) % cols
                matrix[i][y] += 1
        total = 0
        for item, exp in expected_freq.items():
            b = str(item).encode('utf-8', errors='ignore')
            ests = []
            for i, hf in enumerate(hash_functions):
                y = int(hf(b).hexdigest(), 16) % cols
                ests.append(matrix[i][y])
            total += abs(min(ests) - exp)
        return float(total)

    try:
        E_base = _calc_standard_cms_error(evaluator.test_data, evaluator.expected_freq)
        E_base_aae = E_base / max(1, len(evaluator.expected_freq))
        print(
            f"[Baseline CMS] SAE={E_base:.2f}  AAE={E_base_aae:.6f}  "
            f"(标准CMS在当前采样流上的总绝对误差/平均绝对误差)"
        )
    except Exception as _e:
        E_base = float("nan")
        print(f"[Baseline CMS] 计算失败: {_e}")

    # ---- 为三个函数分别创建工具箱（修正 expr_mut 注册，加入 bloat 限制） ----
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

        # 先注册 expr / expr_mut
        if key == "init_dex":
            tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2, type_=rtype)
            tb.register("expr_mut", gp.genFull, pset=pset, min_=0, max_=1, type_=rtype)
        else:
            tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3, type_=rtype)
            tb.register("expr_mut", gp.genFull, pset=pset, min_=1, max_=2, type_=rtype)

        # 再注册 individual / population
        tb.register("individual", tools.initIterate, INDIVIDUAL_CLS, tb.expr)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("clone", copy.deepcopy)

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

        # 树规模约束（防止 bloat）
        for op in ["mate_one_point", "mate_leaf_biased",
                   "mut_uniform", "mut_node_replace", "mut_insert", "mut_shrink", "mut_ephemeral"]:
            tb.decorate(op, gp.staticLimit(key=attrgetter("height"), max_value=tree_hmax))
            tb.decorate(op, gp.staticLimit(key=len, max_value=max_size))

    def _apply_mutation(tb, op_name: str, ind):
        """deap 的 mut/cx 有些返回 (ind,)；这里统一返回变异后的 ind。"""
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

    def _apply_crossover(tb, op_name: str, ind1, ind2):
        """统一返回 crossover 后的第一个子代。"""
        try:
            ret = getattr(tb, op_name)(ind1, ind2)
            if isinstance(ret, tuple) and len(ret) >= 1:
                return ret[0]
            return ind1
        except Exception:
            try:
                ret = tb.mate_one_point(ind1, ind2)
                if isinstance(ret, tuple) and len(ret) >= 1:
                    return ret[0]
                return ind1
            except Exception:
                return ind1

    def _ind_from_str(pset, s):
        return INDIVIDUAL_CLS.from_string(s, pset)

    seed_exprs = {
        "init_dex": [
            "list_3(select_hash(0,e), safe_mod(select_hash(0,e),102), 102, select_hash(1,e), safe_mod(select_hash(1,e),102), 102, select_hash(2,e), safe_mod(select_hash(2,e),102), 102)",
            "list_3(hash_salt(0,e,1), safe_mod(hash_salt(0,e,1),102), 102, hash_salt(1,e,1), safe_mod(hash_salt(1,e,1),102), 102, hash_salt(2,e,1), safe_mod(hash_salt(2,e,1),102), 102)",
            "list_3(hash_on_slice(0,e,0,4), safe_mod(hash_on_slice(0,e,0,4),102), 102, hash_on_slice(1,e,0,4), safe_mod(hash_on_slice(1,e,0,4),102), 102, hash_on_slice(2,e,0,4), safe_mod(hash_on_slice(2,e,0,4),102), 102)"
        ],
        "update": [
            "base(update_count(e,0,1), update_count(e,1,1), update_count(e,2,1))",
            "base(updatecount_if(True,e,0,1), updatecount_if(True,e,1,1), updatecount_if(True,e,2,1))",
            "base(write_count(e,0,safe_add(query_count(e,0),1)), write_count(e,1,safe_add(query_count(e,1),1)), write_count(e,2,safe_add(query_count(e,2),1)))"
        ],
        "query": [
            "base_sel(0, query_date(e,0), query_date(e,1), query_date(e,2))",
            "base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2))",
            "base_sel(3, query_date(e,0), query_date(e,1), query_date(e,2))"
        ],
    }

    seed_bank = {
        k: [_ind_from_str(pset_map[k], s) for s in v]
        for k, v in seed_exprs.items()
    }

    def _seeded_individual(which):
        return toolboxes[which].clone(random.choice(seed_bank[which]))

    def _mixed_immigrant_individual(which, seeded_prob=0.5):
        if random.random() < float(seeded_prob):
            return _seeded_individual(which)
        return toolboxes[which].individual()

    # ---- 初始化 islands ----
    islands = max(1, int(islands))
    population_size = int(population_size)
    elite_size = max(0, int(round(population_size * float(elite_rate))))

    island_states = []
    for _ in range(islands):
        pops = {
            'init_dex': toolboxes['init_dex'].population(n=population_size),
            'update': toolboxes['update'].population(n=population_size),
            'query': toolboxes['query'].population(n=population_size),
        }
        island_states.append({
            'pops': pops,
            'birth': [0] * population_size,
            'fits': [(0.0, float("inf"))] * population_size,  # (fitness, error)
        })

    # ---- 初始评估：并行收集 error 设定 E0 ----
    init_results_all = []
    for st in island_states:
        pops = st['pops']
        triples = [
            (pops['init_dex'][i], pops['update'][i], pops['query'][i])
            for i in range(population_size)
        ]
        local_results = _parallel_eval_triplets(triples, evaluator, eval_pool, eval_workers)
        init_results_all.extend(local_results)

    init_errors = [float(err) for _, err in init_results_all]

    try:
        finite_errs = [float(e) for e in init_errors if math.isfinite(float(e))]
        evaluator.E0 = max(1.0, float(statistics.median(finite_errs))) if finite_errs else 1.0
    except Exception:
        evaluator.E0 = 1.0

    # 直接复用刚才的 error，不再重复评估第二遍
    offset = 0
    for st in island_states:
        local_results = init_results_all[offset: offset + population_size]
        offset += population_size
        st['fits'] = [
            (float(evaluator._norm_fitness(err)), float(err))
            for _, err in local_results
        ]
        st['birth'] = list(range(population_size))

    # ---- 全局最优（fitness 越大越好） ----
    best_fitness = -1.0
    best_error = float("inf")
    best_team = None

    def _refresh_global_best():
        nonlocal best_fitness, best_error, best_team
        for st in island_states:
            fits = st['fits']
            pops = st['pops']
            i = max(range(population_size), key=lambda j: fits[j][0])
            fit, err = fits[i]
            if (fit > best_fitness) or (fit == best_fitness and err < best_error):
                best_fitness, best_error = fit, err
                best_team = {
                    'init_dex': pops['init_dex'][i],
                    'update': pops['update'][i],
                    'query': pops['query'][i],
                }

    _refresh_global_best()
    print(f"初始最佳归一化适应度: {best_fitness:.6f} (误差={best_error:.2f}, E0={evaluator.E0:.2f})")

    # ---- 老化进化主循环 ----
    step_counter = population_size
    no_improve = 0
    burst_left = 0

    mut_ops = {
        'init_dex': ["mut_uniform", "mut_node_replace", "mut_insert", "mut_shrink", "mut_ephemeral"],
        'update': ["mut_uniform", "mut_node_replace", "mut_insert", "mut_shrink", "mut_ephemeral"],
        'query': ["mut_uniform", "mut_node_replace", "mut_insert", "mut_shrink", "mut_ephemeral"],
    }
    mut_weights = {
        "early": [0.10, 0.15, 0.40, 0.25, 0.10],  # uniform, node_replace, insert, shrink, ephemeral
        "mid": [0.10, 0.25, 0.30, 0.20, 0.15],
        "late": [0.10, 0.30, 0.15, 0.20, 0.25],
    }

    for gen in range(int(generations)):
        phase = get_evolution_phase(gen, generations)
        print(f"\n=== 第 {gen + 1}/{generations} 代 ===", flush=True)
        prev_best_error = best_error

        # ====== 每岛：保存精英（防止被 aging 删除） ======
        elites_per_island = []
        for st in island_states:
            fits = st['fits']
            pops = st['pops']
            elite_idx = sorted(range(population_size), key=lambda j: fits[j][0], reverse=True)[:elite_size]
            elites = []
            for j in elite_idx:
                elites.append((
                    toolboxes['init_dex'].clone(pops['init_dex'][j]),
                    toolboxes['update'].clone(pops['update'][j]),
                    toolboxes['query'].clone(pops['query'][j]),
                    fits[j],
                ))
            elites_per_island.append(elites)

        # ====== 每代做 population_size 次“生-死”更新（aging）======
        # 改成“批量生成 -> 并行评估 -> 依次插回”，单 seed 才能真正吃多核
        batch_steps = 1
        produced = 0

        while produced < 1:
            cur_steps = min(batch_steps, population_size - produced)
            pending = []  # [(isl, child), ...]

            # 先批量生成 child
            for _ in range(cur_steps):
                for isl, st in enumerate(island_states):
                    pops = st['pops']
                    fits = st['fits']

                    k = min(int(tournament_size), population_size)
                    cand = random.sample(range(population_size), k=k)
                    parent_idx = max(cand, key=lambda j: fits[j][0])

                    child = {
                        'init_dex': toolboxes['init_dex'].clone(pops['init_dex'][parent_idx]),
                        'update': toolboxes['update'].clone(pops['update'][parent_idx]),
                        'query': toolboxes['query'].clone(pops['query'][parent_idx]),
                    }

                    _reset_prob = float(reset_prob)
                    _recombine_prob = float(recombine_prob)
                    _crossover_prob = float(crossover_prob)
                    _leaf_biased_cx_prob = float(leaf_biased_cx_prob)
                    _mutation_prob = float(mutation_prob)
                    _reset_whole_prob = float(reset_whole_prob)

                    if phase == "early":
                        _crossover_prob = max(_crossover_prob, 0.30)
                    elif phase == "late":
                        _crossover_prob = min(_crossover_prob, 0.15)

                    if random.random() < _reset_whole_prob:
                        which = random.choice(["init_dex", "update", "query"])
                        child[which] = toolboxes[which].individual()
                    else:
                        if random.random() < _reset_prob:
                            which = random.choice(["init_dex", "update", "query"])
                            child[which] = toolboxes[which].individual()

                        if random.random() < _crossover_prob:
                            mate_cand = random.sample(range(population_size), k=k)
                            mate_idx = max(mate_cand, key=lambda j: fits[j][0])

                            which = random.choice(["init_dex", "update", "query"])
                            donor = toolboxes[which].clone(pops[which][mate_idx])

                            cx_op = "mate_leaf_biased" if random.random() < _leaf_biased_cx_prob else "mate_one_point"
                            child[which] = _apply_crossover(toolboxes[which], cx_op, child[which], donor)

                        elif random.random() < _recombine_prob:
                            donor_idx = random.randrange(population_size)
                            which = random.choice(["init_dex", "update", "query"])
                            child[which] = toolboxes[which].clone(pops[which][donor_idx])

                        if random.random() < _mutation_prob:
                            which = random.choice(["init_dex", "update", "query"])
                            op = random.choices(mut_ops[which], weights=mut_weights[phase], k=1)[0]
                            child[which] = _apply_mutation(toolboxes[which], op, child[which])

                    pending.append((isl, child))

            # 再批量并行评估
            triples = [
                (child['init_dex'], child['update'], child['query'])
                for _, child in pending
            ]
            results = _parallel_eval_triplets(triples, evaluator, eval_pool, eval_workers)

            # 最后按顺序插回各岛，保持 aging 逻辑
            for (isl, child), (fit, err) in zip(pending, results):
                st = island_states[isl]
                pops = st['pops']
                fits = st['fits']
                birth = st['birth']

                for which in ("init_dex", "update", "query"):
                    pops[which].append(child[which])
                fits.append((float(fit), float(err)))
                birth.append(step_counter)
                step_counter += 1

                oldest_idx = min(range(len(birth)), key=lambda j: birth[j])
                for which in ("init_dex", "update", "query"):
                    pops[which].pop(oldest_idx)
                fits.pop(oldest_idx)
                birth.pop(oldest_idx)

                while len(birth) > population_size:
                    oldest_idx = min(range(len(birth)), key=lambda j: birth[j])
                    for which in ("init_dex", "update", "query"):
                        pops[which].pop(oldest_idx)
                    fits.pop(oldest_idx)
                    birth.pop(oldest_idx)

                st['fits'] = fits
                st['birth'] = birth

            produced += cur_steps

        # ====== 精英保留：把保存下来的 elites 强制塞回去（替换掉最差的） ======
        for st, elites in zip(island_states, elites_per_island):
            pops = st['pops']
            fits = st['fits']
            birth = st['birth']

            if not elites:
                continue

            # 找到当前最差的 elite_size 个位置来替换
            worst_idx = sorted(range(population_size), key=lambda j: fits[j][0])[:len(elites)]
            for rep_idx, (e_init, e_upd, e_qry, e_fit) in zip(worst_idx, elites):
                pops['init_dex'][rep_idx] = e_init
                pops['update'][rep_idx] = e_upd
                pops['query'][rep_idx] = e_qry
                fits[rep_idx] = (float(e_fit[0]), float(e_fit[1]))
                birth[rep_idx] = step_counter
                step_counter += 1

            st['fits'] = fits
            st['birth'] = birth

        # ====== 迁移（岛模型） ======
        if islands > 1 and int(mig_period) > 0 and ((gen + 1) % int(mig_period) == 0):
            print(f"[MIGRATE] gen={gen + 1} period={mig_period} k={mig_k} mode=elite+random", flush=True)
            k = max(1, int(mig_k))
            for isl in range(islands):
                src = island_states[isl]
                dst = island_states[(isl + 1) % islands]

                src_fits = src['fits']
                src_pops = src['pops']

                elite_k = max(1, k // 2)
                rand_k = max(0, k - elite_k)

                top_idx = sorted(
                    range(population_size),
                    key=lambda j: src_fits[j][0],
                    reverse=True
                )[:elite_k]

                rest_pool = [j for j in range(population_size) if j not in top_idx]
                if rand_k > 0 and rest_pool:
                    rand_idx = random.sample(rest_pool, k=min(rand_k, len(rest_pool)))
                else:
                    rand_idx = []

                mig_idx = top_idx + rand_idx

                migrants = []
                for j in mig_idx:
                    migrants.append((
                        toolboxes['init_dex'].clone(src_pops['init_dex'][j]),
                        toolboxes['update'].clone(src_pops['update'][j]),
                        toolboxes['query'].clone(src_pops['query'][j]),
                        src_fits[j],
                    ))

                dst_birth = dst['birth']
                dst_pops = dst['pops']
                dst_fits = dst['fits']

                # 目标岛尽量不要覆盖自己的精英：先保护 top elite_size
                protect_idx = set(
                    sorted(
                        range(population_size),
                        key=lambda j: dst_fits[j][0],
                        reverse=True
                    )[:elite_size]
                )

                candidate_repl = [
                    j for j in sorted(range(population_size), key=lambda j: dst_birth[j])
                    if j not in protect_idx
                ]

                if len(candidate_repl) < len(migrants):
                    candidate_repl = sorted(range(population_size), key=lambda j: dst_birth[j])

                repl_idx = candidate_repl[:len(migrants)]

                for rep_idx, mig in zip(repl_idx, migrants):
                    dst_pops['init_dex'][rep_idx] = mig[0]
                    dst_pops['update'][rep_idx] = mig[1]
                    dst_pops['query'][rep_idx] = mig[2]
                    dst_fits[rep_idx] = mig[3]
                    dst_birth[rep_idx] = step_counter
                    step_counter += 1

                dst['birth'] = dst_birth
                dst['fits'] = dst_fits

        # ====== 周期性随机移民（PPT风格：固定周期，小比例，全random） ======
        if int(immigration_period) > 0 and ((gen + 1) % int(immigration_period) == 0):
            m = max(1, int(population_size * float(periodic_immigration_rate)))
            print(f"[IMMIGRATION] gen={gen + 1} period={immigration_period} m={m} mode=random", flush=True)

            pending = []

            for st in island_states:
                pops = st['pops']
                fits = st['fits']
                birth = st['birth']

                protect_idx = set(
                    sorted(range(population_size), key=lambda j: fits[j][0], reverse=True)[:elite_size]
                )

                repl = [
                    j for j in sorted(range(population_size), key=lambda j: birth[j])
                    if j not in protect_idx
                ][:m]

                if len(repl) < m:
                    repl = sorted(range(population_size), key=lambda j: birth[j])[:m]

                for j in repl:
                    new_init = toolboxes['init_dex'].individual()
                    new_upd = toolboxes['update'].individual()
                    new_qry = toolboxes['query'].individual()

                    pops['init_dex'][j] = new_init
                    pops['update'][j] = new_upd
                    pops['query'][j] = new_qry

                    pending.append((st, j, new_init, new_upd, new_qry))

            triples = [(a, b, c) for _, _, a, b, c in pending]
            results = _parallel_eval_triplets(triples, evaluator, eval_pool, eval_workers)

            for (st, j, _, _, _), (fit, err) in zip(pending, results):
                st['fits'][j] = (float(fit), float(err))
                st['birth'][j] = step_counter
                step_counter += 1

        # ====== 爆发期随机移民：换血（替换最老 m 个，尽量不动精英） ======
        if burst_left > 0:
            m = max(1, int(population_size * float(immigrant_rate)))
            if burst_left == int(burst_gens):
                print(
                    f"[BURST] mixed immigrants: seeded_prob={mixed_seeded_prob:.2f} "
                    f"random_prob={1.0 - mixed_seeded_prob:.2f} m={m}",
                    flush=True
                )
            pending = []  # [(st, j, init, upd, qry), ...]

            for st in island_states:
                pops = st['pops']
                birth = st['birth']
                repl = sorted(range(population_size), key=lambda j: birth[j])[:m]

                for j in repl:
                    new_init = _mixed_immigrant_individual('init_dex', seeded_prob=mixed_seeded_prob)
                    new_upd = _mixed_immigrant_individual('update', seeded_prob=mixed_seeded_prob)
                    new_qry = _mixed_immigrant_individual('query', seeded_prob=mixed_seeded_prob)

                    pops['init_dex'][j] = new_init
                    pops['update'][j] = new_upd
                    pops['query'][j] = new_qry

                    pending.append((st, j, new_init, new_upd, new_qry))

            triples = [(a, b, c) for _, _, a, b, c in pending]
            results = _parallel_eval_triplets(triples, evaluator, eval_pool, eval_workers)

            for (st, j, _, _, _), (fit, err) in zip(pending, results):
                st['fits'][j] = (float(fit), float(err))
                st['birth'][j] = step_counter
                step_counter += 1

            burst_left -= 1

        # ====== 本代统计与停滞检测 ======
        _refresh_global_best()
        all_fits = [f for st in island_states for f in st['fits']]
        avg_fit = sum(x[0] for x in all_fits) / len(all_fits)
        avg_err = sum(x[1] for x in all_fits) / len(all_fits)
        print(
            f"平均归一化适应度: {avg_fit:.6f} (平均误差={avg_err:.2f})，最佳: {best_fitness:.6f} (误差={best_error:.2f})",
            flush=True)

        if best_error < prev_best_error:
            print(f"发现新的历史最佳误差: {best_error:.2f} (fitness={best_fitness:.6f})", flush=True)
            if best_team is not None:
                print("当前最佳个体表达式：")
                print("init_dex: ", evaluator._canonical_tree_str(best_team['init_dex']))
                print("update: ", evaluator._canonical_tree_str(best_team['update']))
                print("query: ", evaluator._canonical_tree_str(best_team['query']))
            no_improve = 0
        else:
            no_improve += 1

        if int(burst_gens) > 0 and no_improve >= int(stagnation_limit):
            print(f"[STAGNATION] 连续 {no_improve} 代无提升 -> 进入爆发期 burst_gens={burst_gens}（mixed immigrants）",
                  flush=True)
            burst_left = int(burst_gens)
            no_improve = 0

        if best_error <= 0:
            print("找到完美解!", flush=True)
            break
    if eval_pool is not None:
        eval_pool.shutdown(wait=True, cancel_futures=True)
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
        run_proxy_mode = "proxy_balanced"
    else:
        run_proxy_mode = proxy_modes[run_idx % len(proxy_modes)]

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
                proxy_mode=run_proxy_mode,
                proxy_pool_mul=args_dict["proxy_pool_mul"],
                proxy_min_u=args_dict["proxy_min_u"],
                islands=args_dict["islands"],
                tournament_size=args_dict["tournament_size"],
                elite_rate=args_dict["elite_rate"],
                reset_prob=args_dict["reset_prob"],
                reset_whole_prob=args_dict["reset_whole_prob"],
                recombine_prob=args_dict["recombine_prob"],
                crossover_prob=args_dict["crossover_prob"],
                leaf_biased_cx_prob=args_dict["leaf_biased_cx_prob"],
                mutation_prob=args_dict["mutation_prob"],
                stagnation_limit=args_dict["stagnation_limit"],
                burst_gens=args_dict["burst_gens"],
                immigrant_rate=args_dict["immigrant_rate"],
                mixed_seeded_prob=args_dict["mixed_seeded_prob"],
                immigration_period=args_dict["immigration_period"],
                periodic_immigration_rate=args_dict["periodic_immigration_rate"],
                mig_period=args_dict["mig_period"],
                mig_k=args_dict["mig_k"],
                max_height=args_dict["max_height"],
                max_size=args_dict["max_size"],
                eval_workers=args_dict["eval_workers"],
            )
            print(f"[STAGE1] proxy_mode_arg={run_proxy_mode} best_fitness={best_fitness:.6f} best_error={best_error:.2f}")

            # ---------- 阶段2：真实流复评 ----------
            val_seed = (int(run_seed) + 99991) % (2 ** 32)
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
    parser.add_argument("--eval_workers", type=int, default=1, help="单个 seed 内部用于并行评估个体的进程数")
    parser.add_argument("--out_dir", type=str, default="runs", help="并行模式下每个 restart 的输出目录（log/py/json）")
    # 数据集：默认使用 /data/8T/xgr/traces/univ2_trace/univ2_npy 下的 *.flowid.npy（少量采样）
    parser.add_argument("--dataset_root", type=str, default="/data/8T/xgr/traces/univ2_trace/univ2_npy",
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
    # 决策层（三大件 + 精英 + 爆发）
    parser.add_argument("--islands", type=int, default=4, help="岛模型数量（>1 开启迁移）")
    parser.add_argument("--tournament_size", type=int, default=5, help="锦标赛大小")
    parser.add_argument("--elite_rate", type=float, default=0.0, help="每代强制保留的精英比例")
    parser.add_argument("--reset_prob", type=float, default=0.10, help="组件重置概率（随机重采样某组件树）")
    parser.add_argument("--reset_whole_prob", type=float, default=0.02,
                        help="更强的组件重置概率（直接重采样某组件，不做其它变异）")
    parser.add_argument("--recombine_prob", type=float, default=0.0, help="弱 donor-copy 概率（默认关闭）")
    parser.add_argument("--crossover_prob", type=float, default=0.25, help="真实 GP crossover 概率")
    parser.add_argument("--leaf_biased_cx_prob", type=float, default=0.50, help="leaf-biased crossover 概率")
    parser.add_argument("--mutation_prob", type=float, default=0.90, help="常规变异概率")
    parser.add_argument("--stagnation_limit", type=int, default=10, help="连续多少代无提升触发爆发期")
    parser.add_argument("--burst_gens", type=int, default=0, help="爆发期持续代数（PPT模式默认关闭）")
    parser.add_argument("--immigrant_rate", type=float, default=0.0, help="爆发期每代注入 immigrants 的比例（默认关闭）")
    parser.add_argument(
        "--mixed_seeded_prob",
        type=float,
        default=0.0,
        help="爆发期 immigrants 中 seeded 个体的比例，0=全random，1=全seeded"
    )
    parser.add_argument("--immigration_period", type=int, default=6, help="周期性随机移民周期")
    parser.add_argument("--periodic_immigration_rate", type=float, default=0.08, help="周期性随机移民比例")
    parser.add_argument("--mig_period", type=int, default=8, help="岛间迁移周期（代）")
    parser.add_argument("--mig_k", type=int, default=3, help="每次迁移 top-k team")
    parser.add_argument("--max_height", type=int, default=8, help="GP树最大高度（bloat约束）")
    parser.add_argument("--max_size", type=int, default=80, help="GP树最大节点数（bloat约束）")

    args = parser.parse_args()
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
                proxy_mode=run_proxy_mode,
                proxy_pool_mul=args.proxy_pool_mul,
                proxy_min_u=args.proxy_min_u,
                islands=args.islands,
                tournament_size=args.tournament_size,
                elite_rate=args.elite_rate,
                reset_prob=args.reset_prob,
                reset_whole_prob=args.reset_whole_prob,
                recombine_prob=args.recombine_prob,
                crossover_prob=args.crossover_prob,
                leaf_biased_cx_prob=args.leaf_biased_cx_prob,
                mutation_prob=args.mutation_prob,
                stagnation_limit=args.stagnation_limit,
                burst_gens=args.burst_gens,
                immigrant_rate=args.immigrant_rate,
                mixed_seeded_prob=args.mixed_seeded_prob,
                immigration_period=args.immigration_period,
                periodic_immigration_rate=args.periodic_immigration_rate,
                mig_period=args.mig_period,
                mig_k=args.mig_k,
                max_height=args.max_height,
                max_size=args.max_size,
                eval_workers=args.eval_workers,
            )

            print(f"[RUN {r + 1}][STAGE1] best_fitness={best_fitness:.6f} best_error={best_error:.2f}")

            val_seed = (int(run_seed) + 99991) % (2 ** 32)
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
        val_seed = (int(best_seed) + 99991) % (2 ** 32)

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