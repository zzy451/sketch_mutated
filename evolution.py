try:
    from .common import *
    from .evaluator import CMSketchEvaluator
    from .helpers import *
    from .llm_engine import *
except ImportError:
    from common import *
    from evaluator import CMSketchEvaluator
    from helpers import *
    from llm_engine import *


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
                    "mechanism_key": str(source_meta.get("mechanism_key", "")),
                    "mechanism_cluster": str(source_meta.get("mechanism_cluster", "")),
                    "mechanism_distance_from_cms": float(source_meta.get("mechanism_distance_from_cms", 0.0)),
                    "architecture_schema": copy.deepcopy(source_meta.get("architecture_schema", {})),
                    "motif_signature": copy.deepcopy(source_meta.get("motif_signature", {})),
                    "mechanism_schema": copy.deepcopy(source_meta.get("mechanism_schema", {})),
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
    island_profile = _get_island_profile(cfg, int(state.get("island_idx", 0)))
    family_hist = _family_histogram_from_state(state)
    family_guidance = _augment_family_guidance_with_frontier(
        _build_family_guidance(island_profile, family_hist, available_specs=specs, failure_buckets=[]),
        state=state,
        profile=island_profile,
    )
    if len(specs) > 1:
        specs = sorted(
            specs,
            key=lambda sp: (
                float(sp.get("err", 1e18) or 1e18),
                -float(sp.get("fit", 0.0) or 0.0),
                -float(sp.get("mechanism_distance_from_cms", 0.0) or 0.0),
            ) + _pick_spec_family_injection_sort_key(
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
        spec_target_funcs = _parse_llm_target_funcs(
            spec.get("target_funcs_override", target_funcs_override if target_funcs_override is not None else cfg.get("llm_target_funcs", "update,query"))
        )
        ok_seed, seed_reason = _seed_spec_allowed_for_state(spec, state, cfg=cfg)
        if not ok_seed:
            continue
        try:
            cand_team = _deserialize_team_spec(spec, gp_ctx["pset_map"])
            base_team = {
                "init_dex": state["pops"]["init_dex"][idx],
                "update": state["pops"]["update"][idx],
                "query": state["pops"]["query"][idx],
            }
            team = _compose_team_with_targets(base_team, cand_team, spec_target_funcs)
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
                source_meta={"phase": "seed", "source": spec.get("source", "offline_json"), "rationale": spec.get("rationale", ""), "family_tag": seed_meta.get("family_tag", spec.get("family_tag", "")), "architecture_schema": copy.deepcopy(seed_meta.get("architecture_schema", {})), "motif_signature": copy.deepcopy(seed_meta.get("motif_signature", {})), "mechanism_schema": copy.deepcopy(seed_meta.get("mechanism_schema", {})), "schema_hash": str(seed_meta.get("schema_hash", "")), "motif_key": str(seed_meta.get("motif_key", "")), "mechanism_key": str(seed_meta.get("mechanism_key", "")), "mechanism_cluster": str(seed_meta.get("mechanism_cluster", "")), "mechanism_distance_from_cms": float(seed_meta.get("mechanism_distance_from_cms", 0.0)), "arch_type": str(seed_meta.get("arch_type", "regular"))},
                fec_key=fec_key,
            )
            print(f"[LLM_SEED_APPLY] idx={idx} fit={float(fit):.6f} err={float(err):.6f}", flush=True)
            inserted += 1
        except Exception:
            continue
    return state, inserted


def _inject_llm_immigrants_with_engine(state, cfg, gp_ctx, llm_engine, candidate_specs, success_budget, target_funcs_override=None):
    if success_budget <= 0:
        return state, 0, []
    evaluator = _make_evaluator_from_cfg(cfg)
    if cfg.get("hard_case_replay", False):
        evaluator.import_hard_case_state(state.get("hard_case_state"))
    _age_novelty_incubator_state(state)
    duplicate_blocklist = _duplicate_blocklist_snapshot(state)

    target_funcs = _parse_llm_target_funcs(target_funcs_override if target_funcs_override is not None else cfg.get("llm_target_funcs", "update,query"))
    island_profile = _get_island_profile(cfg, int(state.get("island_idx", 0)))
    family_hist = _family_histogram_from_state(state)
    dominant_family_cooldown = _dominant_family_cooldown_snapshot(state, family_hist=family_hist, topk=int(cfg.get("llm_novelty_dominant_topk", 3)), min_count=int(cfg.get("llm_novelty_dominant_min_count", 2)))
    failure_hints = _collect_recent_failure_hints_from_states([state])
    family_guidance = _augment_family_guidance_with_frontier(
        _build_family_guidance(
            island_profile,
            family_hist,
            available_specs=candidate_specs,
            failure_buckets=failure_hints.get("failure_buckets", []),
            prefer_irregular=False,
        ),
        state=state,
        profile=island_profile,
    )
    mechanism_hist = _mechanism_histogram_from_state(state)
    irregular_family_guidance = _augment_family_guidance_with_frontier(
        _build_family_guidance(
            island_profile,
            family_hist,
            available_specs=candidate_specs,
            failure_buckets=failure_hints.get("failure_buckets", []),
            prefer_irregular=True,
        ),
        state=state,
        profile=island_profile,
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
                mechanism_hist=mechanism_hist,
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

    for spec in list(candidate_specs or []):
        channel = _llm_candidate_channel(spec)
        novelty_subtype = _llm_candidate_subtype(spec)
        if channel == "novelty":
            novelty_stats["proposed"] += 1
            if str(island_profile.get("name", "")) == "baseline":
                _record_llm_novelty_reject(state, "baseline_island_skip")
                llm_engine.logger.info("novelty candidate rejected", island=int(state.get("island_idx", 0)), reason="baseline_island_skip")
                continue
        if ptr >= len(repl_idx):
            ptr = 0
        idx = repl_idx[ptr] if repl_idx else 0
        ptr += 1
        try:
            cand_team = _deserialize_team_spec(spec, gp_ctx["pset_map"])
            if isinstance(spec.get("architecture_schema", {}), dict) and spec.get("architecture_schema"):
                cand_team["_claimed_architecture_schema"] = copy.deepcopy(spec.get("architecture_schema", {}))
            base_team = {
                "init_dex": state["pops"]["init_dex"][idx],
                "update": state["pops"]["update"][idx],
                "query": state["pops"]["query"][idx],
            }
            team = _compose_team_with_targets(base_team, cand_team, target_funcs)
            if isinstance(cand_team.get("_claimed_architecture_schema", {}), dict) and cand_team.get("_claimed_architecture_schema"):
                team["_claimed_architecture_schema"] = copy.deepcopy(cand_team.get("_claimed_architecture_schema", {}))
            chk = llm_engine.validate_team_candidate(team, evaluator, existing_canon=None, pset_map=gp_ctx["pset_map"], duplicate_blocklist=duplicate_blocklist)
            if not chk.get("ok", False):
                bad_reasons = [str(x) for x in list(chk.get("reasons", []))]
                if any(rr.startswith("duplicate_") for rr in bad_reasons):
                    _register_duplicate_reject_in_state(state, candidate_meta=chk.get("candidate_meta", {}), chk=chk, family_tag=str(spec.get("family_tag", "")))
                    duplicate_blocklist = _duplicate_blocklist_snapshot(state)
                if channel == "novelty":
                    rej_reason = "validate_fail:" + "|".join([str(x) for x in bad_reasons[:3]])
                    _record_llm_novelty_reject(state, rej_reason)
                    llm_engine.logger.info(
                        "novelty candidate rejected",
                        island=int(state.get("island_idx", 0)),
                        family_tag=str(spec.get("family_tag", "")),
                        reason=rej_reason,
                    )
                continue

            candidate_meta_local = dict(chk.get("candidate_meta", {}) or {})
            novelty_subtype, novelty_auto_reason = _auto_promote_novelty_subtype(chk["team"], candidate_meta=candidate_meta_local, requested=novelty_subtype)
            if novelty_auto_reason:
                spec["novelty_auto_reason"] = str(novelty_auto_reason)
            motif_terms = _compute_motif_score_terms(
                chk["team"],
                state,
                profile=island_profile,
                novelty_subtype=novelty_subtype,
                candidate_meta=candidate_meta_local,
            )
            guidance_local = irregular_family_guidance if novelty_subtype == "irregular" else family_guidance
            frontier_bonus_local, frontier_hits_local = _frontier_bonus_from_candidate_meta(candidate_meta=candidate_meta_local, guidance=guidance_local)
            mechanism_terms = _mechanism_score_terms_from_candidate_meta(candidate_meta_local)
            mechanism_frontier_bonus_local, mechanism_frontier_hits_local = _mechanism_frontier_bonus_from_candidate_meta(candidate_meta=candidate_meta_local, guidance=guidance_local)
            cooldown_reason = ""
            if channel != "repair":
                cooldown_reason = _dominant_family_cooldown_reason(
                    candidate_meta=candidate_meta_local,
                    cooldown=dominant_family_cooldown,
                    motif_distance_score=float(motif_terms.get("motif_distance_score", 0.0)),
                    frontier_bonus=float(frontier_bonus_local),
                    mechanism_frontier_bonus=float(mechanism_frontier_bonus_local),
                    cfg=cfg,
                    novelty_subtype=novelty_subtype,
                )
                if cooldown_reason:
                    _record_llm_novelty_reject(state, cooldown_reason)
                    llm_engine.logger.info(
                        "novelty candidate rejected",
                        island=int(state.get("island_idx", 0)),
                        family_tag=str(candidate_meta_local.get("family_tag", _team_family_tag(chk["team"]))),
                        reason=cooldown_reason,
                    )
                    continue
            numeric_risk = _numeric_risk_probe_with_evaluator(evaluator, chk["team"], cfg=cfg, phase="llm_novelty" if channel != "repair" else "llm_repair")
            if bool(numeric_risk.get("block", False)):
                rej_reason = str(numeric_risk.get("reason", "numeric_risk_reject"))
                if channel == "novelty":
                    _record_llm_novelty_reject(state, rej_reason)
                    llm_engine.logger.info(
                        "novelty candidate rejected",
                        island=int(state.get("island_idx", 0)),
                        family_tag=str(candidate_meta_local.get("family_tag", _team_family_tag(chk["team"]))),
                        reason=rej_reason,
                    )
                continue

            if channel == "repair":
                target_fit_local, target_err_local = state.get("fits", [])[idx] if idx < len(state.get("fits", [])) else (0.0, float("inf"))
                repair_ok, repair_reason = _repair_candidate_quality_gate(
                    chk.get("fit", 0.0),
                    chk.get("err", 1e18),
                    island_best_fit,
                    island_best_err,
                    target_fit=target_fit_local,
                    target_err=target_err_local,
                    cfg=cfg,
                )
                if not repair_ok:
                    continue
                replace_individual_in_state(
                    state,
                    idx,
                    chk["team"],
                    (chk["fit"], chk["err"], chk["case_vec"]),
                    source_meta={
                        "phase": "stagnation",
                        "source": spec.get("source", "offline_json"),
                        "rationale": spec.get("rationale", ""),
                        "family_tag": str(candidate_meta_local.get("family_tag", spec.get("family_tag", ""))),
                        "channel": "repair",
                        "adaptive_reason": str(spec.get("adaptive_reason", "")),
                        "architecture_schema": copy.deepcopy(candidate_meta_local.get("architecture_schema", {})),
                        "motif_signature": copy.deepcopy(candidate_meta_local.get("motif_signature", {})),
                        "mechanism_schema": copy.deepcopy(candidate_meta_local.get("mechanism_schema", {})),
                        "schema_hash": str(candidate_meta_local.get("schema_hash", "")),
                        "motif_key": str(candidate_meta_local.get("motif_key", "")),
                        "mechanism_key": str(candidate_meta_local.get("mechanism_key", "")),
                        "mechanism_cluster": str(candidate_meta_local.get("mechanism_cluster", "")),
                        "mechanism_distance_from_cms": float(candidate_meta_local.get("mechanism_distance_from_cms", 0.0)),
                        "arch_type": str(candidate_meta_local.get("arch_type", "regular")),
                    },
                    fec_key=chk["key"],
                )
                print(f"[LLM_IMMIGRANT_APPLY] channel=repair idx={idx} fit={float(chk['fit']):.6f} err={float(chk['err']):.6f}", flush=True)
                inserted += 1
                accepted_specs.append(spec)
                if inserted >= int(success_budget):
                    break
                continue

            novelty_stats["validated"] += 1
            mech_ok, mech_metrics, mech_reasons = _mechanism_gate(
                candidate_meta=candidate_meta_local,
                guidance=guidance_local,
                cfg=cfg,
                novelty_subtype=novelty_subtype,
            )
            if not mech_ok:
                rej_reason = str(mech_reasons[0] if mech_reasons else "mechanism_gate_fail")
                _record_llm_novelty_reject(state, rej_reason)
                llm_engine.logger.info(
                    "novelty candidate rejected",
                    island=int(state.get("island_idx", 0)),
                    family_tag=str(_team_family_tag(chk["team"])),
                    reason=rej_reason,
                    mechanism_family=str(mech_metrics.get("mechanism_family", "cms_like")),
                    mechanism_distance=float(mech_metrics.get("mechanism_distance_from_cms", 0.0)),
                )
                continue
            mech_sat_ok, mech_sat_metrics, mech_sat_reasons = _mechanism_saturation_gate(
                candidate_meta=candidate_meta_local,
                state=state,
                cfg=cfg,
                novelty_subtype=novelty_subtype,
            )
            if not mech_sat_ok:
                rej_reason = str(mech_sat_reasons[0] if mech_sat_reasons else "mechanism_saturation_fail")
                _record_llm_novelty_reject(state, rej_reason)
                llm_engine.logger.info(
                    "novelty candidate rejected",
                    island=int(state.get("island_idx", 0)),
                    family_tag=str(_team_family_tag(chk["team"])),
                    reason=rej_reason,
                    mechanism_cluster=str(mech_sat_metrics.get("mechanism_cluster", "")),
                )
                continue
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
                candidate_meta=candidate_meta_local,
                motif_distance_score=float(motif_terms.get("motif_distance_score", 0.0)),
            )
            family_override_used = False
            if not fam_ok:
                if bool(mech_metrics.get("family_override", False)):
                    family_override_used = True
                else:
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
                candidate_meta=candidate_meta_local,
                state=state,
                cfg=cfg,
                novelty_subtype=novelty_subtype,
                mechanism_frontier_bonus=float(mechanism_frontier_bonus_local),
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

            materialization_strength = _materialization_strength_from_candidate_meta(candidate_meta_local)
            qual_ok, qual_reason = _novelty_quality_gate(
                chk["fit"],
                chk["err"],
                island_best_fit,
                island_best_err,
                cfg,
                stage="phase1",
                novelty_subtype=novelty_subtype,
                materialization_strength=materialization_strength,
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
            if novelty_holdout_ev is not None:
                try:
                    _, holdout_lite_err = novelty_holdout_ev.evaluate_individual(
                        chk["team"]["init_dex"],
                        chk["team"]["update"],
                        chk["team"]["query"],
                    )
                except Exception:
                    holdout_lite_err = None
            if holdout_lite_err is not None:
                novelty_stats["pass_holdout"] += 1

            failure_summary = _team_failure_bucket_summary(evaluator, chk["team"])
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
            score_info["motif_distance_score"] = float(motif_terms.get("motif_distance_score", 0.0))
            score_info["motif_frequency"] = int(motif_terms.get("motif_frequency", 0))
            score_info["arch_type"] = str(motif_terms.get("arch_type", candidate_meta_local.get("arch_type", "regular")))
            score_info["mechanism_distance_from_cms"] = float(mech_metrics.get("mechanism_distance_from_cms", mechanism_terms.get("mechanism_distance_from_cms", 0.0)))
            score_info["mechanism_cluster"] = str(mechanism_terms.get("mechanism_cluster", candidate_meta_local.get("mechanism_cluster", "")))
            score_info["mechanism_family"] = str(mechanism_terms.get("mechanism_family", "cms_like"))
            score_info["mechanism_frontier_bonus"] = float(mechanism_frontier_bonus_local)
            score_info["lane_role_entropy"] = float(mechanism_terms.get("lane_role_entropy", 0.0))
            score_info["cms_similarity"] = float(mechanism_terms.get("cms_similarity", 0.0))
            score_info["mechanism_frequency"] = int(mechanism_terms.get("mechanism_frequency", 0))
            score_info["score"] = float(score_info.get("score", 0.0)) + 5.0 * float(motif_terms.get("motif_distance_score", 0.0)) + float(motif_terms.get("innovation_bonus", 0.0))
            score_info["innovation_bonus"] = float(score_info.get("innovation_bonus", 0.0)) + float(motif_terms.get("innovation_bonus", 0.0)) + float(mechanism_terms.get("mechanism_rarity_bonus", 0.0))
            if float(mechanism_terms.get("lane_role_entropy", 0.0)) > 0.15:
                score_info["score"] += 3.5 * float(mechanism_terms.get("lane_role_entropy", 0.0))
            if str(mechanism_terms.get("mechanism_family", "cms_like")) != "cms_like":
                score_info["score"] += 2.5
            score_info["frontier_bonus"] = float(frontier_bonus_local)
            score_info["numeric_risk_score"] = float(numeric_risk.get("heuristic_score", 0.0))
            score_info.setdefault("innovation_reasons", [])
            score_info["innovation_reasons"] = list(score_info.get("innovation_reasons", [])) + [f"motif_freq={int(motif_terms.get('motif_frequency', 0))}", f"arch_type={str(motif_terms.get('arch_type', candidate_meta_local.get('arch_type', 'regular')))}", f"mechanism_family={str(mechanism_terms.get('mechanism_family', 'cms_like'))}", f"mech_dist={float(score_info.get('mechanism_distance_from_cms', 0.0)):.2f}", f"cms_sim={float(mechanism_terms.get('cms_similarity', 0.0)):.2f}"]
            score_info["score"] -= float(cfg.get("llm_novelty_numeric_risk_penalty_scale", 9.0)) * float(numeric_risk.get("heuristic_score", 0.0))
            score_info["score"] += float(cfg.get("llm_novelty_mechanism_score_scale", 9.0)) * float(score_info.get("mechanism_distance_from_cms", 0.0))
            score_info["score"] += float(cfg.get("llm_novelty_mechanism_entropy_score_scale", 4.0)) * float(mechanism_terms.get("lane_role_entropy", 0.0))
            score_info["score"] += float(cfg.get("llm_novelty_mechanism_frontier_score_scale", 6.0)) * float(mechanism_frontier_bonus_local)
            score_info["score"] += 1.5 * float(mechanism_terms.get("mechanism_rarity_bonus", 0.0))
            score_info["score"] -= float(cfg.get("llm_novelty_cms_like_penalty_scale", 10.0)) * float(mech_metrics.get("cms_like_penalty", 0.0))
            score_info["score"] -= float(cfg.get("llm_novelty_cms_similarity_penalty_scale", 5.0)) * float(mechanism_terms.get("cms_similarity", 0.0))
            if float(frontier_bonus_local) > 0.0:
                score_info["score"] += float(cfg.get("llm_novelty_frontier_score_scale", 4.0)) * float(frontier_bonus_local)
                score_info["innovation_reasons"].append(f"frontier_bonus={float(frontier_bonus_local):.2f}")
            if float(mechanism_frontier_bonus_local) > 0.0:
                score_info["innovation_reasons"].append(f"mechanism_frontier_bonus={float(mechanism_frontier_bonus_local):.2f}")
            if float(numeric_risk.get("heuristic_score", 0.0)) > 0.0:
                score_info["innovation_reasons"].append(f"numeric_risk={float(numeric_risk.get('heuristic_score', 0.0)):.2f}")
            if family_override_used:
                score_info["innovation_reasons"].append("family_gate_overridden_by_mechanism")
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

            min_score = float(cfg.get("llm_novelty_phase1_min_score_irregular", 4.0) if novelty_subtype == "irregular" else cfg.get("llm_novelty_phase1_min_score", 6.0))
            if float(score_info.get("score", -1e18)) < float(min_score):
                rej_reason = f"novelty_phase1_score_too_low:{float(score_info.get('score', 0.0)):.2f}<{float(min_score):.2f}"
                _record_llm_novelty_reject(state, rej_reason)
                llm_engine.logger.info(
                    "novelty candidate rejected",
                    island=int(state.get("island_idx", 0)),
                    family_tag=str(sat_metrics.get("tag", _team_family_tag(chk["team"]))),
                    reason=rej_reason,
                    novelty_score=float(score_info.get("score", 0.0)),
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
            spec["holdout_lite_reason"] = "phase1_holdout_record_only" if holdout_lite_err is not None else "phase1_holdout_skip"
            spec["novelty_score"] = float(score_info.get("score", 0.0))
            spec["innovation_bonus"] = float(score_info.get("innovation_bonus", 0.0))
            spec["innovation_reasons"] = list(score_info.get("innovation_reasons", []))
            spec["failure_buckets"] = list(score_info.get("failure_buckets", []))
            spec["exact_family_count"] = int(sat_metrics.get("exact_count", 0))
            spec["component_family_count"] = int(sat_metrics.get("component_count", 0))
            spec["novelty_subtype"] = str(novelty_subtype)
            spec["motif_distance_score"] = float(score_info.get("motif_distance_score", 0.0))
            spec["motif_frequency"] = int(score_info.get("motif_frequency", 0))
            spec["frontier_bonus"] = float(score_info.get("frontier_bonus", 0.0))
            spec["numeric_risk_score"] = float(score_info.get("numeric_risk_score", 0.0))
            spec["arch_type"] = str(score_info.get("arch_type", spec.get("arch_type", "regular")))
            spec.setdefault("architecture_schema", copy.deepcopy(candidate_meta_local.get("architecture_schema", {})))
            spec.setdefault("motif_signature", copy.deepcopy(candidate_meta_local.get("motif_signature", {})))
            spec.setdefault("mechanism_schema", copy.deepcopy(candidate_meta_local.get("mechanism_schema", {})))
            spec.setdefault("schema_hash", str(candidate_meta_local.get("schema_hash", "")))
            spec.setdefault("motif_key", str(candidate_meta_local.get("motif_key", "")))
            spec.setdefault("mechanism_key", str(candidate_meta_local.get("mechanism_key", "")))
            spec.setdefault("mechanism_cluster", str(candidate_meta_local.get("mechanism_cluster", "")))
            spec["mechanism_distance_from_cms"] = float(candidate_meta_local.get("mechanism_distance_from_cms", mech_metrics.get("mechanism_distance_from_cms", 0.0)))
            spec["materialization_strength"] = float(materialization_strength)

            _incubate_novelty_candidate(
                state,
                chk["team"],
                spec,
                chk,
                candidate_meta_local,
                score_info,
                holdout_lite_err=holdout_lite_err,
            )
            llm_engine.logger.info(
                "novelty candidate incubated",
                island=int(state.get("island_idx", 0)),
                family_tag=str(_team_family_tag(chk["team"])),
                novelty_score=float(spec.get("novelty_score", 0.0)),
                novelty_subtype=str(spec.get("novelty_subtype", "stable")),
                holdout_lite_err=spec.get("holdout_lite_err", None),
                arch_type=str(spec.get("arch_type", "regular")),
                materialization_strength=float(spec.get("materialization_strength", 0.0)),
            )
        except Exception as e:
            if channel == "novelty":
                _record_llm_novelty_reject(state, f"inject_exception:{e}")
            continue

    remaining_budget = max(0, int(success_budget) - int(inserted))
    promoted = 0
    if remaining_budget > 0:
        state, promoted, _ = _promote_from_novelty_incubator(
            state,
            cfg,
            gp_ctx,
            evaluator,
            island_best_fit,
            island_best_err,
            success_budget=remaining_budget,
        )
        inserted += int(promoted)

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



def _mechanism_novelty_value(team, fit, err):
    try:
        meta = asdict(_candidate_meta_from_team(type("_Tmp", (), {"_canonical_tree_str": lambda self, t: str(t), "_canonical_triplet_key": lambda self, a, b, c: (str(a), str(b), str(c))})(), team))
    except Exception:
        meta = {}
    mech_terms = _mechanism_score_terms_from_candidate_meta(meta)
    mech_dist = float(meta.get("mechanism_distance_from_cms", 0.0))
    mech_family = str((meta.get("mechanism_schema", {}) or {}).get("mechanism_family", "cms_like"))
    arch_type = str(meta.get("arch_type", "regular"))
    val = 42.0 * float(fit)
    val += 9.0 * mech_dist
    val += 4.0 * float(mech_terms.get("lane_role_entropy", 0.0))
    if mech_family != "cms_like":
        val += 5.0
    if arch_type != "regular":
        val += 3.0
    if float(err) <= 1000.0:
        val += 4.0
    elif float(err) > 50000.0:
        val -= 6.0
    val -= 0.00004 * min(float(err), 2.0e7)
    return float(val), meta

def _refresh_island_mechanism_best(state):
    fits = state["fits"]
    pops = state["pops"]
    pop_size = len(fits)
    best = None
    for i in range(pop_size):
        fit, err = fits[i]
        team = {
            "init_dex": copy.deepcopy(pops["init_dex"][i]),
            "update": copy.deepcopy(pops["update"][i]),
            "query": copy.deepcopy(pops["query"][i]),
        }
        score, meta = _mechanism_novelty_value(team, fit, err)
        cur = (score, float(fit), -float(err), i, team, meta)
        if best is None or cur[:3] > best[:3]:
            best = cur
    if best is None:
        return -1, -1e18, 0.0, 1e18, None, {}
    return int(best[3]), float(best[0]), float(best[1]), float(-best[2]), best[4], best[5]


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

    # Sort by holdout_err, but treat differences within a relative tolerance as ties
    # and break ties by stage1_fit. This prevents a noise-level holdout difference
    # (e.g. 0 vs 1) from overriding a meaningfully better stage1 candidate.
    _holdout_errs = [float(d["holdout_err"]) for d in scored]
    _best_holdout = min(_holdout_errs)
    _holdout_tol_rel = 0.10  # within 10% of best holdout is considered a tie
    _holdout_tol_abs = 5.0   # or within 5 absolute AAE units
    def _sort_key(d):
        h = float(d["holdout_err"])
        tol = max(_holdout_tol_abs, abs(_best_holdout) * _holdout_tol_rel)
        holdout_bucket = 0 if h <= _best_holdout + tol else 1
        return (holdout_bucket, -float(d["stage1_fit"]), float(d["stage1_err"]))
    scored.sort(key=_sort_key)
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
    init_source_counts = {
        "init_dex": Counter(),
        "update": Counter(),
        "query": Counter(),
    }

    pops = {'init_dex': [], 'update': [], 'query': []}
    for which in ("init_dex", "update", "query"):
        for _ in range(pop_size):
            ind, src = _init_individual_from_ctx(
                gp_ctx,
                which,
                p_skeleton=init_p_skeleton,
                p_seed=init_p_seed,
                p_llm_seed=llm_seed_prob,
                allowed_family_labels=fam_allowed.get(which, set()),
                return_source=True,
            )
            pops[which].append(ind)
            init_source_counts[which][str(src)] += 1
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
        f"init_src_init={dict(init_source_counts['init_dex'])} "
        f"init_src_update={dict(init_source_counts['update'])} "
        f"init_src_query={dict(init_source_counts['query'])} "
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
        'init_source_counts': {k: dict(v) for k, v in init_source_counts.items()},
                "hard_case_state": evaluator.export_hard_case_state() if cfg.get("hard_case_replay", False) else {"version": 0, **evaluator._empty_hard_case_state()},
        "scored_hard_case_version": 0,
        "recent_diag_history": [copy.deepcopy(dbg)],
        "llm_novelty_stats": _empty_llm_novelty_stats(),
        "innovation_archive": [],
        "novelty_incubator": _empty_novelty_incubator_state(),
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
                child = _semantic_repair_team_with_evaluator(evaluator, child, gp_ctx["pset_map"])
            except Exception:
                pass

            numeric_child = _numeric_risk_probe_with_evaluator(evaluator, child, cfg=cfg, phase="gp_child")
            if bool(numeric_child.get("block", False)):
                err = max(float(numeric_child.get("probe_err", 0.0) or 0.0), float(cfg.get("gp_numeric_risk_fail_err", 5.0e7)))
                fit = float(evaluator._norm_fitness(err))
                case_vec = (float(evaluator.lexicase_default_bad),) * int(evaluator.lexicase_total_cases)
            else:
                try:
                    fit, err, case_vec = evaluator.evaluate_individual(
                        child['init_dex'],
                        child['update'],
                        child['query'],
                        return_case_vec=True,
                    )
                    if bool(numeric_child.get("warn", False)) or float(numeric_child.get("heuristic_score", 0.0)) > 0.0:
                        err = float(err) + float(cfg.get("gp_numeric_risk_penalty_weight", 250.0)) * float(numeric_child.get("heuristic_score", 0.0))
                        fit = float(evaluator._norm_fitness(err))
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


def _build_mechanism_feedback_migrant(src, isl, cfg=None):
    cfg = cfg if isinstance(cfg, dict) else {}
    if not bool(cfg.get("llm_mechanism_feedback_enable", True)):
        return None
    idx, mech_score, mech_fit, mech_err, mech_team, mech_meta = _refresh_island_mechanism_best(src)
    if idx < 0 or mech_team is None:
        return None
    _, best_fit, best_err, _ = _refresh_island_best(src)
    min_score = float(cfg.get("llm_mechanism_feedback_min_score", 70.0))
    err_rel = float(cfg.get("llm_mechanism_feedback_err_rel_mult", 10.0))
    err_floor = float(cfg.get("llm_mechanism_feedback_err_floor", 1200.0))
    err_cap_abs = float(cfg.get("llm_mechanism_feedback_abs_err_cap", 60000.0))
    err_cap = min(err_cap_abs, max(err_floor, max(1.0, float(best_err)) * err_rel))
    if float(mech_score) < min_score:
        return None
    if float(mech_err) > err_cap:
        return None
    src_case_vecs = src.get("case_vecs", [()] * len(src.get("fits", [])))
    src_fec_keys = list(src.get("fec_keys", []))
    fec_key = src_fec_keys[idx] if idx < len(src_fec_keys) else None
    if not fec_key or str(fec_key).startswith("__fallback__"):
        cluster = str((mech_meta or {}).get("mechanism_cluster", "") or "")
        key_hint = str((mech_meta or {}).get("mechanism_key", "") or "")
        fec_key = f"__mechanism__:{cluster or key_hint or isl}:{idx}"
    return {
        "team": {
            "init_dex": copy.deepcopy(mech_team["init_dex"]),
            "update": copy.deepcopy(mech_team["update"]),
            "query": copy.deepcopy(mech_team["query"]),
        },
        "fit": float(mech_fit),
        "err": float(mech_err),
        "case_vec": tuple(float(x) for x in (src_case_vecs[idx] if idx < len(src_case_vecs) else ())),
        "fec_key": fec_key,
        "source_meta": {
            "phase": "migration_mechanism_feedback",
            "source_island": int(isl),
            "mechanism_score": float(mech_score),
            "mechanism_cluster": str((mech_meta or {}).get("mechanism_cluster", "")),
            "mechanism_family": str(((mech_meta or {}).get("mechanism_schema", {}) or {}).get("mechanism_family", "cms_like")),
            "mechanism_distance_from_cms": float((mech_meta or {}).get("mechanism_distance_from_cms", 0.0)),
        },
    }


def _migrate_island_states(island_states, mig_k: int, cfg=None):
    islands = len(island_states)
    if islands <= 1:
        return island_states

    cfg = cfg if isinstance(cfg, dict) else {}
    k = max(1, int(mig_k))
    mech_extra = max(0, int(cfg.get("llm_mechanism_feedback_extra_migrants", 1)))

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
        ordinary_tokens = set()
        for j in mig_idx:
            migrant = {
                "team": {
                    "init_dex": copy.deepcopy(src['pops']['init_dex'][j]),
                    "update": copy.deepcopy(src['pops']['update'][j]),
                    "query": copy.deepcopy(src['pops']['query'][j]),
                },
                "fit": float(src['fits'][j][0]),
                "err": float(src['fits'][j][1]),
                "case_vec": tuple(float(x) for x in src_case_vecs[j]),
                "fec_key": src_fec_keys[j],
                "source_meta": {
                    "phase": "migration",
                    "source_island": int(isl),
                    "fec_key": src_fec_keys[j],
                },
            }
            migrants.append(migrant)
            ordinary_tokens.add(_key_token(migrant["team"]))

        if mech_extra > 0:
            mech_migrant = _build_mechanism_feedback_migrant(src, isl, cfg=cfg)
            if mech_migrant is not None and _key_token(mech_migrant.get("team", {})) not in ordinary_tokens:
                migrants.append(mech_migrant)

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
                source_meta=copy.deepcopy(mig.get("source_meta", {
                    "phase": "migration",
                    "source_island": int(isl),
                    "fec_key": mig["fec_key"],
                })),
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
    gp_ctx_main = _populate_llm_seed_bank_from_cfg(gp_ctx_main, cfg)
    llm_logger = LLMRunLogger(cfg.get("llm_log_path", ""))
    llm_ref = PrimitiveSpecReference(gp_ctx_main["pset_map"], cfg, llm_logger)
    llm_engine = LLMProposalEngine(cfg, llm_ref, llm_logger)
    main_eval_for_llm = _make_evaluator_from_cfg(cfg)
    gp_ctx_main = _filter_llm_team_bank_with_evaluator(gp_ctx_main, main_eval_for_llm, cfg)

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
                family_guidance = _augment_family_guidance_with_frontier(
                    _build_family_guidance(island_profile, global_seed_hist, available_specs=[], failure_buckets=[]),
                    state={"innovation_archive": [], "llm_meta": [], "pops": {}, "fits": [], "novelty_incubator": _empty_novelty_incubator_state()},
                    profile=island_profile,
                )
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
            seed_candidates, seed_dropped = _filter_llm_seed_candidates(seed_candidates, cfg=cfg)
            for rec in seed_dropped:
                c = rec.get("candidate", {}) if isinstance(rec, dict) else {}
                llm_logger.info(
                    "seed candidate dropped",
                    source=str(c.get("source", "offline_json")),
                    fitness=float(c.get("fit", 0.0)),
                    error=float(c.get("err", 0.0)),
                    family_tag=str((c.get("team", {}) or {}).get("family_tag", "")),
                    reason=str(rec.get("reason", "seed_filter")),
                )
            for i, c in enumerate(seed_candidates):
                llm_logger.info(
                    "seed candidate accepted",
                    idx=int(i),
                    source=str(c.get("source", "offline_json")),
                    fitness=float(c.get("fit", 0.0)),
                    error=float(c.get("err", 0.0)),
                    family_tag=str(c["team"].get("family_tag", "")),
                )
            if not seed_candidates:
                local_team_bank = list(gp_ctx_main.get("llm_team_bank", []))
                for spec in local_team_bank[:seed_limit]:
                    seed_candidates.append({
                        "team": {
                            "init_dex": copy.deepcopy(spec["init_dex"]),
                            "update": copy.deepcopy(spec["update"]),
                            "query": copy.deepcopy(spec["query"]),
                            "family_tag": str(spec.get("family_tag", _team_family_tag(spec))),
                            "family_parts": copy.deepcopy(spec.get("family_parts", _team_family_parts(spec))),
                        },
                        "fit": float(spec.get("fitness", 0.0)),
                        "err": float(spec.get("error", 2_000_000_000.0)),
                        "case_vec": tuple(float(x) for x in spec.get("case_vec", ())),
                        "rationale": "local_llm_team_bank",
                        "source": "local_llm_team_bank",
                    })
                if local_team_bank:
                    llm_logger.info("team seed fallback activated", source="local_llm_team_bank", total=len(seed_candidates))
            cfg["llm_seed_specs"] = []
            for c in seed_candidates:
                spec = _serialize_team_spec(
                    {**c["team"], "family_tag": _team_family_tag(c["team"]), "family_parts": _team_family_parts(c["team"])},
                    rationale=c.get("rationale", ""),
                    source=c.get("source", "offline_json"),
                )
                spec["fit"] = float(c.get("fit", 0.0))
                spec["err"] = float(c.get("err", 1e18))
                spec["case_vec"] = tuple(float(x) for x in c.get("case_vec", ()))
                spec["target_funcs_override"] = "init_dex,update,query"
                cfg["llm_seed_specs"].append(spec)
            llm_logger.info("team seed candidate bank prepared", total=len(cfg["llm_seed_specs"]))
    except Exception as e:
        llm_logger.warn("seed candidate prepare failed", error=str(e))

    no_improve_chunks = 0

    best_fitness = -float('inf')
    best_error = float('inf')
    best_team = None
    best_mechanism_score = -float('inf')
    best_mechanism_fit = 0.0
    best_mechanism_err = float('inf')
    best_mechanism_team = None
    best_mechanism_meta = {}

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
        best_mechanism_score = -float('inf')
        best_mechanism_fit = 0.0
        best_mechanism_err = float('inf')
        best_mechanism_team = None
        best_mechanism_meta = {}

        for st in island_states:
            _, st_best_fit, st_best_err, st_best_team = _refresh_island_best(st)
            st["best_fitness"] = float(st_best_fit)
            st["best_error"] = float(st_best_err)
            _, st_mech_score, st_mech_fit, st_mech_err, st_mech_team, st_mech_meta = _refresh_island_mechanism_best(st)
            st["best_mechanism_score"] = float(st_mech_score)
            if (st_best_fit > best_fitness) or (st_best_fit == best_fitness and st_best_err < best_error):
                best_fitness = float(st_best_fit)
                best_error = float(st_best_err)
                best_team = st_best_team
            if (st_mech_score > best_mechanism_score) or (st_mech_score == best_mechanism_score and st_mech_err < best_mechanism_err):
                best_mechanism_score = float(st_mech_score)
                best_mechanism_fit = float(st_mech_fit)
                best_mechanism_err = float(st_mech_err)
                best_mechanism_team = st_mech_team
                best_mechanism_meta = copy.deepcopy(st_mech_meta)

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
            mechanism_improved = False
            for r in results:
                if (r['best_fitness'] > best_fitness) or (r['best_fitness'] == best_fitness and r['best_error'] < best_error):
                    best_fitness = float(r['best_fitness'])
                    best_error = float(r['best_error'])
                    best_team = r['best_team']
                    improved = True
                st = r.get('state', {}) if isinstance(r, dict) else {}
                try:
                    _, st_mech_score, st_mech_fit, st_mech_err, st_mech_team, st_mech_meta = _refresh_island_mechanism_best(st)
                except Exception:
                    st_mech_score, st_mech_fit, st_mech_err, st_mech_team, st_mech_meta = -1e18, 0.0, 1e18, None, {}
                if (st_mech_score > best_mechanism_score) or (st_mech_score == best_mechanism_score and st_mech_err < best_mechanism_err):
                    best_mechanism_score = float(st_mech_score)
                    best_mechanism_fit = float(st_mech_fit)
                    best_mechanism_err = float(st_mech_err)
                    best_mechanism_team = st_mech_team
                    best_mechanism_meta = copy.deepcopy(st_mech_meta)
                    mechanism_improved = True

            print(
                f"平均fitness: {avg_fit:.6f} (平均AAE={avg_err:.6f})，最佳fitness: {best_fitness:.6f} (AAE={best_error:.6f}) | 机制最佳score: {best_mechanism_score:.3f} (AAE={best_mechanism_err:.6f})",
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
            if mechanism_improved and best_mechanism_team is not None:
                print(f"发现新的机制最佳个体: mechanism_score={best_mechanism_score:.3f} (fitness={best_mechanism_fit:.6f}, AAE={best_mechanism_err:.6f})", flush=True)
                try:
                    tmp_ev = _make_evaluator_from_cfg(cfg)
                    print("当前机制最佳个体表达式：")
                    print("init_dex: ", tmp_ev._canonical_tree_str(best_mechanism_team['init_dex']))
                    print("update: ", tmp_ev._canonical_tree_str(best_mechanism_team['update']))
                    print("query: ", tmp_ev._canonical_tree_str(best_mechanism_team['query']))
                    print("mechanism_schema:", json.dumps(best_mechanism_meta.get('mechanism_schema', {}), ensure_ascii=False))
                    print("mechanism_distance_from_cms:", float(best_mechanism_meta.get('mechanism_distance_from_cms', 0.0)))
                except Exception:
                    pass

            if islands > 1 and gens_done < int(generations) and int(mig_period) > 0:
                print(f"[MIGRATE] after_gen={gens_done} k={mig_k} islands={islands}", flush=True)
                island_states = _migrate_island_states(island_states, mig_k=mig_k, cfg=cfg)

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
                    repair_frac = min(0.60, max(0.15, float(cfg.get("llm_stagnation_repair_fraction", 0.34))))
                    repair_limit = max(1, int(math.floor(cand_limit * repair_frac)))
                    irregular_limit = 0
                    if bool(cfg.get("llm_irregular_enable", True)) and cand_limit >= 3:
                        irregular_limit = max(1, int(math.ceil(cand_limit * float(cfg.get("llm_irregular_fraction", 0.25)))))
                    if repair_limit + irregular_limit >= cand_limit:
                        repair_limit = max(1, cand_limit - irregular_limit - 1)
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
                        family_guidance = _augment_family_guidance_with_frontier(
                            _build_family_guidance(
                                island_profile_cur,
                                state_family_hist if state_family_hist.get("total", 0) > 0 else global_family_hist,
                                available_specs=cfg.get("llm_seed_specs", []),
                                failure_buckets=merged_hints.get("failure_buckets", []),
                                prefer_irregular=False,
                            ),
                            state=st,
                            profile=island_profile_cur,
                        )
                        irregular_profile = copy.deepcopy(island_profile_cur)
                        if isinstance(irregular_profile, dict):
                            ir_map = dict(irregular_profile.get("innovation_family_labels", {}))
                            if ir_map:
                                irregular_profile["allowed_family_labels"] = ir_map
                        irregular_family_guidance = _augment_family_guidance_with_frontier(
                            _build_family_guidance(
                                irregular_profile,
                                state_family_hist if state_family_hist.get("total", 0) > 0 else global_family_hist,
                                available_specs=cfg.get("llm_seed_specs", []),
                                failure_buckets=merged_hints.get("failure_buckets", []),
                                prefer_irregular=True,
                            ),
                            state=st,
                            profile=irregular_profile,
                        )
                        repair_allowed_targets = _parse_llm_target_funcs(cfg.get("llm_target_funcs", "update,query"))
                        adaptive_target, adaptive_reason = _adaptive_single_tree_target_from_hints(
                            island_profile_cur,
                            merged_hints,
                            allowed_targets=repair_allowed_targets,
                            fallback=str(cfg.get("llm_single_tree_target", "update")),
                        )
                        duplicate_blocklist = _duplicate_blocklist_snapshot(st)
                        repair_force_single_tree = _should_force_single_tree_repair_from_hints(
                            merged_hints,
                            duplicate_blocklist=duplicate_blocklist,
                            cfg=cfg,
                        )
                        repair_target_funcs_override = cfg.get("llm_target_funcs", "update,query") if repair_force_single_tree else "init_dex,update,query"
                        repair_phase = llm_engine.prepare_phase_candidates(
                            phase="stagnation",
                            gp_ctx=gp_ctx_main,
                            evaluator=main_eval_for_llm,
                            base_team=base_team_for_llm,
                            existing_canon=phase_seen,
                            limit=min(per_profile_repair, cand_limit - len(phase_candidates)),
                            extra_prompt_hints=merged_hints,
                            family_guidance=family_guidance,
                            force_single_tree=repair_force_single_tree,
                            force_single_tree_target=adaptive_target if repair_force_single_tree else None,
                            target_funcs_override=repair_target_funcs_override,
                            candidate_channel="repair",
                            adaptive_reason=adaptive_reason,
                            duplicate_blocklist=duplicate_blocklist,
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
                                duplicate_blocklist=duplicate_blocklist,
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
                                    duplicate_blocklist=duplicate_blocklist,
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
                        spec["target_funcs_override"] = cfg.get("llm_target_funcs", "update,query") if (spec["channel"] == "repair" and spec.get("edit_mode", "team") == "single_tree") else "init_dex,update,query"
                        spec["adaptive_reason"] = str(c.get("adaptive_reason", ""))
                        spec["architecture_schema"] = copy.deepcopy(c.get("architecture_schema", {}))
                        spec["motif_signature"] = copy.deepcopy(c.get("motif_signature", {}))
                        spec["mechanism_schema"] = copy.deepcopy(c.get("mechanism_schema", spec.get("mechanism_schema", {})))
                        spec["schema_hash"] = str(c.get("schema_hash", ""))
                        spec["motif_key"] = str(c.get("motif_key", ""))
                        spec["mechanism_key"] = str(c.get("mechanism_key", ""))
                        spec["mechanism_cluster"] = str(c.get("mechanism_cluster", ""))
                        spec["mechanism_distance_from_cms"] = float(c.get("mechanism_distance_from_cms", 0.0))
                        spec["arch_type"] = str(c.get("arch_type", "regular"))
                        spec["novelty_subtype"] = str(c.get("novelty_subtype", spec.get("novelty_subtype", "stable")))
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
                        leftover_repair_budget = max(0, int(repair_budget) - int(repair_injected))
                        island_states, novelty_injected = _apply_channel(island_states, novelty_specs, novelty_budget + leftover_repair_budget, "init_dex,update,query", "novelty")
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
                        f"novelty_incubated={int(novelty_summary.get('incubated', 0))} "
                        f"novelty_promoted={int(novelty_summary.get('promoted', 0))} "
                        f"novelty_reject_top3={top_novelty_rejects}",
                        flush=True,
                    )
                except Exception as e:
                    import traceback as _tb
                    print(f"[LLM_IMMIGRANT_TRIGGER_SKIP] after_gen={gens_done} reason={e}", flush=True)
                    print(f"[LLM_IMMIGRANT_TRIGGER_SKIP_TB] {_tb.format_exc()}", flush=True)
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

# expose private helpers needed by cli.py via import *
__all__ = [name for name in dir() if not name.startswith('__')]
