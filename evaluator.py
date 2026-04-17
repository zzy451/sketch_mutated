try:
    from .common import *
except ImportError:
    from common import *


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
                def _resolve_loc(e: str, i: int):
                    vars = init_dex_func(e)[i % 3]
                    x = abs_int(int(vars[0])) % rows_per_matrix
                    y = abs_int(int(vars[1])) % cols_per_matrix
                    if isinstance(vars, (tuple, list)) and len(vars) > 2:
                        z = abs_int(int(vars[2])) % planes
                    else:
                        z = i % planes
                    return x, y, z

                def str_slice(s, start, end):
                    return s[start:end]

                def write_count(e: str, i: int, delta: int) -> int:
                    x, y, z = _resolve_loc(e, i)
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
                    x, y, z = _resolve_loc(e, i)
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
                    x, y, z = _resolve_loc(e, i)
                    overflow_matrices[z][x][y] = st
                    return 0

                def query_count(e: str, i: int) -> int:
                    x, y, z = _resolve_loc(e, i)
                    return count_matrices[z][x][y]

                def query_state(e: str, i: int) -> bool:
                    x, y, z = _resolve_loc(e, i)
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
                def _resolve_loc(e: str, i: int):
                    vars = init_dex_func(e)[i % 3]
                    x = abs_int(int(vars[0])) % rows_per_matrix
                    y = abs_int(int(vars[1])) % cols_per_matrix
                    if isinstance(vars, (tuple, list)) and len(vars) > 2:
                        z = abs_int(int(vars[2])) % planes
                    else:
                        z = i % planes
                    return x, y, z

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
                    x, y, z = _resolve_loc(e, i)
                    return bool(overflow_matrices[z][x][y])

                def query_date(e: str, i: int) -> int:
                    x, y, z = _resolve_loc(e, i)
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

# expose private helpers needed by sibling modules via import *
__all__ = [name for name in dir() if not name.startswith('__')]
