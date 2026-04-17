try:
    from .common import *
    from .evaluator import CMSketchEvaluator
    from .helpers import *
except ImportError:
    from common import *
    from evaluator import CMSketchEvaluator
    from helpers import *


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
        self.duplicate_escape_streaks = Counter()
        self.single_tree_size_escape_streaks = Counter()
        self._current_escape_key = ""

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
        schema_frontier = {}
        incubator_summary = {}
        if isinstance(family_guidance, dict):
            schema_frontier = copy.deepcopy(family_guidance.get("schema_frontier", {}))
            if isinstance(schema_frontier, dict):
                incubator_summary = {
                    "incubator_size": int(schema_frontier.get("incubator_size", 0) or 0),
                    "incubator_top": copy.deepcopy(schema_frontier.get("incubator_top", [])),
                }
        return {
            "family_tag": base_family_tag,
            "architecture_schema": copy.deepcopy(candidate_meta.get("architecture_schema", {})),
            "motif_signature": copy.deepcopy(candidate_meta.get("motif_signature", {})),
            "mechanism_schema": copy.deepcopy(candidate_meta.get("mechanism_schema", {})),
            "schema_hash": str(candidate_meta.get("schema_hash", "")),
            "motif_key": str(candidate_meta.get("motif_key", "")),
            "mechanism_key": str(candidate_meta.get("mechanism_key", "")),
            "mechanism_cluster": str(candidate_meta.get("mechanism_cluster", "")),
            "mechanism_distance_from_cms": float(candidate_meta.get("mechanism_distance_from_cms", 0.0)),
            "arch_type": str(candidate_meta.get("arch_type", "regular")),
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
                "architecture_first": "first propose a structure schema, then materialize it with existing primitives only",
                "real_write_anchor": "novelty structures must keep at least one unconditional real update/write anchor",
            },
            "hard_cases": hard_cases,
            "target_funcs": sorted(self.target_funcs),
            "prompt_hints": prompt_hints,
            "family_guidance": copy.deepcopy(family_guidance) if isinstance(family_guidance, dict) else {},
            "schema_frontier": schema_frontier,
            "mechanism_frontier": copy.deepcopy(family_guidance.get("mechanism_frontier", {})) if isinstance(family_guidance, dict) else {},
            "incubator_summary": incubator_summary,
            "current_structure_axes": {
                "arch_type": str(candidate_meta.get("architecture_schema", {}).get("arch_type", "regular")) if isinstance(candidate_meta.get("architecture_schema", {}), dict) else "regular",
                "layout_style": str(candidate_meta.get("architecture_schema", {}).get("layout_style", "regular")) if isinstance(candidate_meta.get("architecture_schema", {}), dict) else "regular",
                "handoff_policy": str(candidate_meta.get("architecture_schema", {}).get("handoff_policy", "none")) if isinstance(candidate_meta.get("architecture_schema", {}), dict) else "none",
                "state_usage": str(candidate_meta.get("architecture_schema", {}).get("state_usage", "none")) if isinstance(candidate_meta.get("architecture_schema", {}), dict) else "none",
                "query_fusion": str(candidate_meta.get("architecture_schema", {}).get("query_fusion", "base_sel")) if isinstance(candidate_meta.get("architecture_schema", {}), dict) else "base_sel",
            },
            "current_mechanism_axes": {
                "mechanism_family": str(candidate_meta.get("mechanism_schema", {}).get("mechanism_family", "cms_like")) if isinstance(candidate_meta.get("mechanism_schema", {}), dict) else "cms_like",
                "state_contract": str(candidate_meta.get("mechanism_schema", {}).get("state_contract", "none")) if isinstance(candidate_meta.get("mechanism_schema", {}), dict) else "none",
                "query_contract": str(candidate_meta.get("mechanism_schema", {}).get("query_contract", "simple_reduce")) if isinstance(candidate_meta.get("mechanism_schema", {}), dict) else "simple_reduce",
                "replication_budget": int(candidate_meta.get("mechanism_schema", {}).get("replication_budget", 3) or 3) if isinstance(candidate_meta.get("mechanism_schema", {}), dict) else 3,
            },
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
            add_unique(hints["prefer"], "Keep init_dex shaped like a triplet list_3-style index constructor.")
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
        candidate_channel = str((report or {}).get("_candidate_channel", "") or "").strip().lower() if isinstance(report, dict) else ""
        if candidate_channel in {"novelty", "irregular_novelty", "innovation", "innovative_novelty"}:
            return False
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
            "- If you claim a non-regular architecture_schema, the final expressions must materially realize it.",
            "- Novel structures must keep at least one unconditional real write anchor in update.",
            "- Prefer one principal schema-axis change over many noisy micro-edits.",
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
            lines.append("Irregular novelty channel: propose a genuinely different mechanism, not merely another cosmetic CMS-family variant.")
            lines.append("For irregular novelty, define lane roles / handoff / state contract first, then materialize expressions with existing primitives only.")

        duplicate_escape_mode = str((report or {}).get("_duplicate_escape_mode", "") or "").strip()
        if duplicate_escape_mode:
            lines.append(f"Duplicate-basin escape mode: {duplicate_escape_mode}.")
            lines.append("Do not stay inside the same canonical basin. Prefer a coordinated architecture/schema rewrite over another tiny local rewrite.")

        lines.extend([
            "Mechanism-first guidance:",
            "- Before writing expressions, decide lane roles, lane relations, a state contract, and a query contract.",
            "- You may output mode=mechanism_team with a mechanism_schema plus optional expressions. If expressions are omitted, the runtime compiler will materialize a primitive-only team.",
            "- A novelty candidate must not merely be another three-replica counter sketch with cosmetic constant changes.",
            "- At least one lane should play a non-replica role such as scout / witness / rescue / fallback / promotion when proposing genuine novelty.",
            "Architecture-first guidance:",
            "- First decide a high-level structure, then instantiate it with existing primitives only.",
            "- Mechanism schema fields to think about: mechanism_family, lane_roles, lane_relations, state_contract, query_contract, replication_budget, novelty_axes.",
            "- Motif card / regular_cms_like: symmetric init + stable triple-write + simple min/median query.",
            "- Motif card / overflow_handoff: use update_state or state-gated query so one branch can hand off when overflow-like conditions appear.",
            "- Motif card / layered_correction: use layered or hybrid init, keep one main branch and one correction branch, and let query fuse them.",
            "- Motif card / sidecar_heavy: keep a main regular body plus a sidecar/fallback branch, and let query decide how to fuse them.",
            "- Motif card / asymmetric_dual_path: let different branches play different roles instead of being three symmetric copies.",
            "- Do not add new primitives. Use existing primitives only.",
            "- Avoid returning a candidate whose mechanism is still basically cms_like unless you are in repair mode.",
            "init_dex format guidance:",
            "- Prefer one single list_3(...) expression.",
            "- list_3(...) should represent three (x, y, z) lane tuples, not old two-coordinate pairs.",
            "- Keep the lane-plane contract stable: the three list_3 lanes map to z/plane 0, 1, 2 in order.",
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
                "Single-tree init_dex task: keep the three lanes aligned with triplet output semantics and stable z/plane roles 0, 1, 2.",
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
            schema_frontier = family_guidance.get("schema_frontier", {}) if isinstance(family_guidance, dict) else {}
            if isinstance(schema_frontier, dict) and schema_frontier:
                axes = list(schema_frontier.get("axes", []))
                if axes:
                    lines.append("Self-discovered structure frontier:")
                    for ax in axes[:5]:
                        axis = str(ax.get("axis", "") or "")
                        under = ", ".join(str(x) for x in list(ax.get("underexplored", []))[:3])
                        sat = ", ".join(str(x) for x in list(ax.get("saturated", []))[:2])
                        if axis:
                            lines.append(f"- {axis}: underexplored=[{under}] saturated=[{sat}]")
                rare_examples = list(schema_frontier.get("rare_examples", []))
                if rare_examples:
                    lines.append("Rare but promising archived schema examples:")
                    for ex in rare_examples[:3]:
                        lines.append(f"- {json.dumps(ex, ensure_ascii=False)}")
                if int(schema_frontier.get("incubator_size", 0) or 0) > 0:
                    lines.append(f"Innovation incubator size: {int(schema_frontier.get('incubator_size', 0) or 0)}")
            mechanism_frontier = report.get("mechanism_frontier", {}) if isinstance(report, dict) else {}
            if isinstance(mechanism_frontier, dict) and mechanism_frontier:
                fams = ", ".join(str(x) for x in list(mechanism_frontier.get("mechanism_family_underexplored", []))[:3])
                states = ", ".join(str(x) for x in list(mechanism_frontier.get("state_contract_underexplored", []))[:3])
                queries = ", ".join(str(x) for x in list(mechanism_frontier.get("query_contract_underexplored", []))[:3])
                lines.append("Mechanism frontier summary:")
                if fams:
                    lines.append(f"- mechanism_family underexplored: [{fams}]")
                if states:
                    lines.append(f"- state_contract underexplored: [{states}]")
                if queries:
                    lines.append(f"- query_contract underexplored: [{queries}]")
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
        max_retries = max(0, int(self.cfg.get("llm_transport_retries", 2)))
        retry_backoff = max(0.0, float(self.cfg.get("llm_transport_retry_backoff", 1.5)))

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

        last_error = ""
        for attempt in range(max_retries + 1):
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
                last_error = str(e)
                self.logger.warn(
                    "llm_transport_fail",
                    error=last_error,
                    base_url=base_url,
                    model=model,
                    attempt=int(attempt + 1),
                    max_attempts=int(max_retries + 1),
                )
                if attempt < max_retries:
                    try:
                        time.sleep(retry_backoff * float(attempt + 1))
                    except Exception:
                        pass

        self.logger.warn(
            "openai_compatible request failed",
            error=last_error,
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

            # 允许没有 mode；若有 mechanism_schema 则优先按 mechanism_team 处理
            if (not mode) and isinstance(cur.get("mechanism_schema", {}), dict) and cur.get("mechanism_schema"):
                mode = "mechanism_team"
            elif (not mode) and any(k in cur for k in ("init_dex", "update", "query")):
                mode = "team"

            if mode in {"team", "mechanism_team", "schema_team", "mechanism"}:
                out.append({
                    "mode": "mechanism_team" if mode in {"mechanism_team", "schema_team", "mechanism"} else "team",
                    "init_dex": str(cur.get("init_dex", "")).strip(),
                    "update": str(cur.get("update", "")).strip(),
                    "query": str(cur.get("query", "")).strip(),
                    "rationale": str(cur.get("rationale", "")).strip(),
                    "architecture_schema": _sanitize_architecture_schema_claim(cur.get("architecture_schema", {})),
                    "mechanism_schema": _sanitize_mechanism_schema_claim(cur.get("mechanism_schema", {})),
                })
                return

            if mode == "single_tree":
                out.append({
                    "mode": "single_tree",
                    "target": str(cur.get("target", "")).strip(),
                    "expr": str(cur.get("expr", "")).strip(),
                    "rationale": str(cur.get("rationale", "")).strip(),
                    "architecture_schema": _sanitize_architecture_schema_claim(cur.get("architecture_schema", {})),
                    "mechanism_schema": _sanitize_mechanism_schema_claim(cur.get("mechanism_schema", {})),
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
        if obj.get("mode") not in {"team", "mechanism_team"}:
            return None, ["not_team_mode"]
        target_funcs = self.target_funcs
        out = {}
        errs = []
        mechanism_schema = _sanitize_mechanism_schema_claim(obj.get("mechanism_schema", {}))
        compiled_spec = None
        if obj.get("mode") == "mechanism_team" or (mechanism_schema and (not str(obj.get("init_dex", "") or "").strip()) and (not str(obj.get("update", "") or "").strip()) and (not str(obj.get("query", "") or "").strip())):
            try:
                compiled_spec = _compile_mechanism_schema_to_team_spec(mechanism_schema, variant=int(time.time()) % 3)
            except Exception as e:
                errs.append(f"compile_mechanism_schema_failed:{type(e).__name__}")
        for which in ("init_dex", "update", "query"):
            try:
                if (which not in target_funcs) and (base_team is not None):
                    out[which] = copy.deepcopy(base_team[which])
                    continue
                expr = str(obj.get(which, "") or "").strip()
                if not expr and isinstance(compiled_spec, dict):
                    expr = str(compiled_spec.get(which, "") or "").strip()
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
        claimed_schema = _sanitize_architecture_schema_claim(obj.get("architecture_schema", {}))
        if claimed_schema:
            out["_claimed_architecture_schema"] = copy.deepcopy(claimed_schema)
        if mechanism_schema:
            out["_claimed_mechanism_schema"] = copy.deepcopy(mechanism_schema)
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

        base_ratio = max(1.0, float(self.cfg.get("llm_single_tree_max_ratio", 1.5)))
        escape_key = str(getattr(self, "_current_escape_key", "") or "")
        size_streak = int(self.single_tree_size_escape_streaks.get(escape_key, 0)) if escape_key else 0
        ratio_bonus = max(0.0, float(self.cfg.get("llm_single_tree_size_ratio_bonus", 0.0)))
        if target == "update":
            ratio_bonus += max(0.0, float(self.cfg.get("llm_single_tree_update_ratio_bonus", 0.35)))
        ratio_bonus += min(0.75, 0.20 * float(size_streak))
        max_ratio = max(1.0, base_ratio + ratio_bonus)
        base_size = max(1, len(base_team[target]))
        new_size = len(new_tree)
        max_allowed = max(base_size + 3 + int(size_streak), int(math.ceil(base_size * max_ratio)))
        relaxed_hard_cap = max_allowed + max(6, int(math.ceil(base_size * 0.75)))
        if new_size > max_allowed:
            if not (target == "update" and new_size <= relaxed_hard_cap):
                return None, [f"single_tree_too_large:{target}:{new_size}>{max_allowed}"]

        out = {
            "init_dex": copy.deepcopy(base_team["init_dex"]),
            "update": copy.deepcopy(base_team["update"]),
            "query": copy.deepcopy(base_team["query"]),
        }
        out[target] = new_tree
        claimed_schema = _sanitize_architecture_schema_claim(obj.get("architecture_schema", {}))
        if claimed_schema:
            out["_claimed_architecture_schema"] = copy.deepcopy(claimed_schema)
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

    def validate_team_candidate(self, team, evaluator, existing_canon=None, pset_map=None, duplicate_blocklist=None):
        reasons = []
        warns = []
        if pset_map is not None:
            try:
                team = _semantic_repair_team_with_evaluator(evaluator, team, pset_map)
            except Exception:
                pass
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

        qreasons = list(self._validate_query_tree_details(evaluator, team["query"]))
        for rr in qreasons:
            srr = str(rr)
            if srr == "query_dynamic_path_invalid":
                warns.append("query_dynamic_path_soft_invalid")
                continue
            reasons.append(srr)
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
        claimed_schema = _sanitize_architecture_schema_claim(team.get("_claimed_architecture_schema", {})) if isinstance(team, dict) else {}
        claimed_mechanism = _sanitize_mechanism_schema_claim(team.get("_claimed_mechanism_schema", {})) if isinstance(team, dict) else {}
        if claimed_schema:
            candidate_meta["claimed_architecture_schema"] = copy.deepcopy(claimed_schema)
        if claimed_mechanism:
            candidate_meta["claimed_mechanism_schema"] = copy.deepcopy(claimed_mechanism)
        sem_reasons, sem_warns = _semantic_materialization_checks(
            evaluator,
            team,
            candidate_meta=candidate_meta,
            claimed_schema=claimed_schema,
        )
        reasons.extend(list(sem_reasons))
        warns.extend(list(sem_warns))
        if reasons:
            return {"ok": False, "reasons": sorted(set(reasons)), "warnings": sorted(set(warns))}

        key_v2 = tuple(candidate_meta.get("key_v2", key)) if isinstance(candidate_meta.get("key_v2", key), (list, tuple)) else key
        repair_dup_key = tuple(candidate_meta.get("repair_dup_key", key)) if isinstance(candidate_meta.get("repair_dup_key", key), (list, tuple)) else key
        if isinstance(duplicate_blocklist, dict):
            schema_motif = f"{str(candidate_meta.get('schema_hash', '') or '').strip()}|{str(candidate_meta.get('motif_key', '') or '').strip()}"
            family_tag_local = str(candidate_meta.get("family_tag", "") or "").strip()
            mechanism_cluster_local = str(candidate_meta.get("mechanism_cluster", "") or "").strip()
            mechanism_family_local = str((candidate_meta.get("mechanism_schema", {}) or {}).get("mechanism_family", "") or "").strip()
            if _key_token(key_v2) in set(str(x) for x in duplicate_blocklist.get("key_v2", set())) or _key_token(key) in set(str(x) for x in duplicate_blocklist.get("key_v2", set())):
                return {"ok": False, "reasons": ["duplicate_recent_basin_team"], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta, "key": key, "key_v2": key_v2, "repair_dup_key": repair_dup_key}
            if _key_token(repair_dup_key) in set(str(x) for x in duplicate_blocklist.get("repair_dup_keys", set())):
                return {"ok": False, "reasons": ["duplicate_recent_basin_structure"], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta, "key": key, "key_v2": key_v2, "repair_dup_key": repair_dup_key}
            if schema_motif and int(duplicate_blocklist.get("schema_motif", Counter()).get(schema_motif, 0)) > 0:
                return {"ok": False, "reasons": ["duplicate_recent_schema_motif"], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta, "key": key, "key_v2": key_v2, "repair_dup_key": repair_dup_key}
            if family_tag_local and int(duplicate_blocklist.get("family_tags", Counter()).get(family_tag_local, 0)) >= 2:
                return {"ok": False, "reasons": ["duplicate_recent_family_cooldown"], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta, "key": key, "key_v2": key_v2, "repair_dup_key": repair_dup_key}
            if mechanism_cluster_local and int(duplicate_blocklist.get("mechanism_clusters", Counter()).get(mechanism_cluster_local, 0)) >= 2:
                return {"ok": False, "reasons": ["duplicate_recent_mechanism_cluster"], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta, "key": key, "key_v2": key_v2, "repair_dup_key": repair_dup_key}
            if mechanism_family_local and int(duplicate_blocklist.get("mechanism_families", Counter()).get(mechanism_family_local, 0)) >= 3 and float(candidate_meta.get("mechanism_distance_from_cms", 0.0)) < 5.0:
                return {"ok": False, "reasons": ["duplicate_recent_mechanism_family"], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta, "key": key, "key_v2": key_v2, "repair_dup_key": repair_dup_key}
        numeric_risk = _numeric_risk_probe_with_evaluator(evaluator, team, cfg=getattr(self, "cfg", {}), phase="llm_validate")
        if bool(numeric_risk.get("block", False)):
            return {"ok": False, "reasons": [str(numeric_risk.get("reason", "numeric_risk_reject"))], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta, "key": key, "key_v2": key_v2, "repair_dup_key": repair_dup_key, "numeric_risk": numeric_risk}
        if bool(numeric_risk.get("warn", False)) and str(numeric_risk.get("warning", "")).strip():
            warns.append(str(numeric_risk.get("warning", "")).strip())
        if existing_canon is not None:
            if key in existing_canon or key_v2 in existing_canon:
                return {"ok": False, "reasons": ["duplicate_canonical_team"], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta, "key": key, "key_v2": key_v2, "repair_dup_key": repair_dup_key}
            if repair_dup_key in existing_canon:
                return {"ok": False, "reasons": ["duplicate_structural_team"], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta, "key": key, "key_v2": key_v2, "repair_dup_key": repair_dup_key}

        try:
            fit, err, case_vec = evaluator.evaluate_individual(
                team["init_dex"], team["update"], team["query"], return_case_vec=True
            )
        except Exception as e:
            return {"ok": False, "reasons": [f"evaluate_failed:{e}"], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta}
        try:
            validate_err_cap = min(float(getattr(self, "cfg", {}).get("llm_validate_absolute_err_cap", 8.0e8)), max(8.0e6, float(getattr(self, "cfg", {}).get("llm_validate_err_mult_cap", 6000.0)) * max(1.0, float(getattr(evaluator, "E0", 1.0) or 1.0))))
        except Exception:
            validate_err_cap = 5.0e8
        if float(err) > float(validate_err_cap):
            return {"ok": False, "reasons": [f"llm_validate_err_hardcap:{float(err):.2f}>{float(validate_err_cap):.2f}"], "warnings": sorted(set(warns)), "candidate_meta": candidate_meta, "key": key, "key_v2": key_v2, "repair_dup_key": repair_dup_key}
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
            "numeric_risk": numeric_risk,
        }

    def prepare_phase_candidates(self, phase, gp_ctx, evaluator, base_team, existing_canon, limit, extra_prompt_hints=None, family_guidance=None, force_single_tree=None, force_single_tree_target=None, target_funcs_override=None, candidate_channel="", adaptive_reason="", duplicate_blocklist=None):
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
        escape_channel = str(candidate_channel or ("repair" if self._prefer_single_tree_for_phase(phase, report=report) else "novelty"))
        escape_key = "|".join([
            str(phase),
            str(escape_channel),
            str((family_guidance or {}).get("target_family_tag", "")),
            str(force_single_tree_target or self._default_single_tree_target(phase, report)),
        ])
        self._current_escape_key = escape_key
        prior_escape = int(self.duplicate_escape_streaks.get(escape_key, 0))
        prior_size_escape = int(self.single_tree_size_escape_streaks.get(escape_key, 0))
        if prior_escape > 0 or prior_size_escape > 0:
            report["_force_single_tree"] = False
            if prior_escape > 0:
                report["_duplicate_escape_mode"] = "schema_conditioned_rewrite" if prior_escape == 1 else ("dual_axis_escape" if prior_escape == 2 else "full_team_family_jump")
            else:
                report["_duplicate_escape_mode"] = "single_tree_size_escape"
            if prior_escape >= 2 and str(candidate_channel or "").strip().lower() != "repair":
                report["_forbid_single_tree"] = True
            if prior_size_escape >= 1:
                report["_forbid_single_tree"] = True
            report["target_funcs"] = sorted(_parse_llm_target_funcs("init_dex,update,query"))
            report["prompt_hints"] = _merge_prompt_hint_dicts(
                report.get("prompt_hints", {}),
                {
                    "hard_avoid": [
                        "Do not return a canonically equivalent team.",
                        "Do not stay inside the same family basin with only cosmetic constant edits.",
                        "Do not expand the update tree only by adding local nested branches.",
                    ],
                    "prefer": [
                        "Escape the current duplicate basin by changing at least one high-level schema axis.",
                        "Prefer coordinated init_dex+update changes over another tiny local rewrite.",
                        "When size pressure is high, use a compact full-team rewrite instead of a giant single-tree patch.",
                    ],
                    "repair_focus": [
                        "Use existing primitives only, but make one architecture-level jump that changes family or motif identity.",
                        "Keep update compact; prefer semantic role changes over deeper nested syntax.",
                    ],
                },
            )
        if str(candidate_channel or "").strip().lower() in {"novelty", "irregular_novelty", "innovation", "innovative_novelty"}:
            report["_force_single_tree"] = False
            report["_forbid_single_tree"] = True
            report["target_funcs"] = ["init_dex", "update", "query"]
            report["prompt_hints"] = _merge_prompt_hint_dicts(
                report.get("prompt_hints", {}),
                {
                    "hard_avoid": [
                        "Do not return a single-tree patch for novelty. Propose a coordinated mechanism-level team.",
                        "Do not merely rephrase the current cms-like family with constant edits.",
                    ],
                    "prefer": [
                        "Novelty should be mechanism-first: role split, handoff contract, state contract, query contract.",
                        "Change at least two coordinated parts when proposing a new mechanism.",
                    ],
                },
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
                if str(candidate_channel or "").strip().lower() in {"novelty", "irregular_novelty", "innovation", "innovative_novelty"}:
                    _record_failed("materialize_single_tree", ["single_tree_disallowed_for_novelty"])
                    continue
                team, perr = self.materialize_single_tree(obj, gp_ctx["pset_map"], base_team=base_team)
                if perr:
                    _record_failed("materialize_single_tree", list(perr))
                    continue
                n_materialized += 1
                chk = self.validate_team_candidate(team, evaluator, existing_canon=seen, pset_map=gp_ctx["pset_map"], duplicate_blocklist=duplicate_blocklist)
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
                    "mechanism_schema": copy.deepcopy(chk.get("candidate_meta", {}).get("mechanism_schema", {})),
                    "mechanism_key": str(chk.get("candidate_meta", {}).get("mechanism_key", "")),
                    "mechanism_cluster": str(chk.get("candidate_meta", {}).get("mechanism_cluster", "")),
                    "mechanism_distance_from_cms": float(chk.get("candidate_meta", {}).get("mechanism_distance_from_cms", 0.0)),
                    "numeric_risk": copy.deepcopy(chk.get("numeric_risk", {})),
                })
                if len(out) >= max(1, int(limit)):
                    return

            for obj in parsed_teams:
                team, perr = self.materialize_team(obj, gp_ctx["pset_map"], base_team=base_team)
                if perr:
                    _record_failed("materialize", list(perr))
                    continue
                n_materialized += 1
                chk = self.validate_team_candidate(team, evaluator, existing_canon=seen, pset_map=gp_ctx["pset_map"], duplicate_blocklist=duplicate_blocklist)
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
                    "mechanism_schema": copy.deepcopy(chk.get("candidate_meta", {}).get("mechanism_schema", {})),
                    "mechanism_key": str(chk.get("candidate_meta", {}).get("mechanism_key", "")),
                    "mechanism_cluster": str(chk.get("candidate_meta", {}).get("mechanism_cluster", "")),
                    "mechanism_distance_from_cms": float(chk.get("candidate_meta", {}).get("mechanism_distance_from_cms", 0.0)),
                    "numeric_risk": copy.deepcopy(chk.get("numeric_risk", {})),
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

        duplicate_dominated_last_round = False
        while len(out) < max(1, int(limit)) and repair_rounds < max_repair_rounds and failed_records:
            repair_rounds += 1
            duplicate_dominated = _duplicate_fail_dominates_failed_records(failed_records)
            size_dominated = _materialize_fail_dominates_failed_records(failed_records, token="single_tree_too_large:update")
            duplicate_dominated_last_round = duplicate_dominated_last_round or duplicate_dominated
            if duplicate_dominated or size_dominated:
                if duplicate_dominated:
                    self.duplicate_escape_streaks[escape_key] = int(self.duplicate_escape_streaks.get(escape_key, 0)) + 1
                if size_dominated:
                    self.single_tree_size_escape_streaks[escape_key] = int(self.single_tree_size_escape_streaks.get(escape_key, 0)) + 1
                streak = int(self.duplicate_escape_streaks.get(escape_key, 0))
                size_streak = int(self.single_tree_size_escape_streaks.get(escape_key, 0))
                report["_force_single_tree"] = False
                if (streak >= 2 and str(candidate_channel or "").strip().lower() != "repair") or size_streak >= 1:
                    report["_forbid_single_tree"] = True
                if duplicate_dominated:
                    report["_duplicate_escape_mode"] = "schema_conditioned_rewrite" if streak == 1 else ("dual_axis_escape" if streak == 2 else "full_team_family_jump")
                elif size_dominated:
                    report["_duplicate_escape_mode"] = "single_tree_size_escape"
                report["target_funcs"] = sorted(_parse_llm_target_funcs("init_dex,update,query"))
                extra_hard_avoid = [
                    "Do not fall back to a canonically equivalent team.",
                    "Do not repeat the same family with only cosmetic constant rewrites.",
                    "Do not satisfy update repair by just growing a much larger local subtree.",
                ]
                extra_prefer = [
                    "Escape the current canonical basin by changing a high-level schema axis.",
                    "Prefer coordinated init_dex+update changes over another tiny single-tree repair.",
                    "Use compact rewrites that keep update semantics clear.",
                ]
                extra_focus = [
                    "Make one principal architecture/schema rewrite and then materialize it with existing primitives only.",
                ]
                if streak >= 2:
                    extra_hard_avoid.append("Avoid staying inside the same hybrid/read-before-write basin.")
                    extra_prefer.append("Change family identity, not just constants or one local branch.")
                    extra_focus.append("Perform a dual-axis rewrite touching init_dex plus update or query together.")
                if streak >= 3:
                    extra_hard_avoid.append("Do not preserve the same motif key unless semantics clearly change.")
                    extra_prefer.append("Prefer an underexplored frontier schema even if it is less locally similar to the seed team.")
                    extra_focus.append("Full-team rewrite is preferred over another near-duplicate local edit.")
                if size_streak > 0:
                    extra_hard_avoid.append("Avoid overgrown update-only edits that exceed the single-tree size budget.")
                    extra_prefer.append("If update needs a bigger change, switch to compact full-team or dual-axis materialization.")
                    extra_focus.append("Keep update compact; express novelty by role/layout changes instead of deeper nested syntax.")
                report["prompt_hints"] = _merge_prompt_hint_dicts(
                    report.get("prompt_hints", {}),
                    {
                        "hard_avoid": extra_hard_avoid,
                        "prefer": extra_prefer,
                        "repair_focus": extra_focus,
                    },
                )
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
        if len(out) > 0 and not duplicate_dominated_last_round:
            if escape_key in self.duplicate_escape_streaks:
                self.duplicate_escape_streaks[escape_key] = max(0, int(self.duplicate_escape_streaks.get(escape_key, 0)) - 1)
                if self.duplicate_escape_streaks[escape_key] <= 0:
                    self.duplicate_escape_streaks.pop(escape_key, None)
            if escape_key in self.single_tree_size_escape_streaks:
                self.single_tree_size_escape_streaks[escape_key] = max(0, int(self.single_tree_size_escape_streaks.get(escape_key, 0)) - 1)
                if self.single_tree_size_escape_streaks[escape_key] <= 0:
                    self.single_tree_size_escape_streaks.pop(escape_key, None)
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
            single_tree_mode=self._prefer_single_tree_for_phase(phase, report=report),
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
    mechanism_schema = None
    if isinstance(team, dict):
        mechanism_schema = team.get("mechanism_schema") or team.get("_claimed_mechanism_schema")
        if not mechanism_schema:
            try:
                mechanism_schema = asdict(_infer_mechanism_schema_from_team(team))
            except Exception:
                mechanism_schema = None
    return {
        "init_dex": str(team["init_dex"]),
        "update": str(team["update"]),
        "query": str(team["query"]),
        "rationale": str(rationale or ""),
        "source": str(source or ""),
        "family_tag": fam_tag,
        "family_parts": copy.deepcopy(fam_parts) if isinstance(fam_parts, dict) else None,
        "mechanism_schema": copy.deepcopy(mechanism_schema) if isinstance(mechanism_schema, dict) else None,
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
    mech = _sanitize_mechanism_schema_claim(spec.get("mechanism_schema", {}))
    if mech:
        team["mechanism_schema"] = copy.deepcopy(mech)
        team["_claimed_mechanism_schema"] = copy.deepcopy(mech)
    return team

# expose private helpers needed by sibling modules via import *
__all__ = [name for name in dir() if not name.startswith('__')]
