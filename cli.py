try:
    from .common import *
    from .evolution import *
    from .evaluator import CMSketchEvaluator
except ImportError:
    from common import *
    from evolution import *
    from evaluator import CMSketchEvaluator


def main(argv=None):
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

    args = parser.parse_args(argv)

    primitive_ref_paths = PRIMITIVE_REFERENCE_PATHS
    if not args.llm_ref_init_pset_path:
        args.llm_ref_init_pset_path = primitive_ref_paths.get("init_dex", "")
    if not args.llm_ref_update_pset_path:
        args.llm_ref_update_pset_path = primitive_ref_paths.get("update", "")
    if not args.llm_ref_query_pset_path:
        args.llm_ref_query_pset_path = primitive_ref_paths.get("query", "")

    primitive_report = PRIMITIVE_CONSISTENCY_REPORT if isinstance(PRIMITIVE_CONSISTENCY_REPORT, dict) else _primitive_consistency_report()
    if not primitive_report.get("ok", True):
        print(f"[PRIMITIVE_WARN] mismatches={primitive_report.get('mismatches', {})}")
    init_layout_protocol = primitive_report.get("init_layout_protocol", {}) if isinstance(primitive_report, dict) else {}

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
    print(
        f"[PRIMITIVE_REGISTRY] init={args.llm_ref_init_pset_path} update={args.llm_ref_update_pset_path} query={args.llm_ref_query_pset_path}")
    print(
        f"[PRIMITIVE_PROTOCOL] root={init_layout_protocol.get('root', '')} "
        f"tuple_arity={init_layout_protocol.get('tuple_arity', [])} "
        f"lane_planes={init_layout_protocol.get('lane_planes', [])} "
        f"triplet_ok={init_layout_protocol.get('uses_triplet_protocol', False)}")

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
    with open("mutate_cmsketch.py", "w", encoding="utf-8") as f:
        f.write(best_code)

    print("\n最佳变异版本已保存为 'mutate_cmsketch.py'")
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


if __name__ == "__main__":
    main()
