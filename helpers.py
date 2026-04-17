try:
    from .common import *
    from .evaluator import CMSketchEvaluator
except ImportError:
    from common import *
    from evaluator import CMSketchEvaluator


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

    mode = str(cfg.get("llm_mode", "none") or "none").strip().lower()
    enabled = bool(cfg.get("llm_enable", False))

    if (not enabled) or (mode not in {"seeds", "both"}):
        return gp_ctx

    expr_bank = _build_local_llm_seed_exprs()
    team_bank = _build_local_llm_seed_teams() + _build_local_mechanism_seed_teams()
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
class MechanismSchema:
    lane_roles: list = field(default_factory=lambda: ["replica", "replica", "replica"])
    lane_relations: list = field(default_factory=list)
    state_contract: str = "none"
    query_contract: str = "simple_reduce"
    replication_budget: int = 3
    novelty_axes: list = field(default_factory=list)
    mechanism_family: str = "cms_like"
    mechanism_note: str = "cms-like replica counter sketch"


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
    mechanism_schema: dict = field(default_factory=dict)
    schema_hash: str = ""
    motif_key: str = ""
    mechanism_key: str = ""
    mechanism_cluster: str = ""
    mechanism_distance_from_cms: float = 0.0
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


def _mechanism_schema_hash(schema):
    try:
        payload = schema if isinstance(schema, dict) else asdict(schema)
    except Exception:
        payload = dict(schema or {}) if isinstance(schema, dict) else {"mechanism_family": "cms_like"}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _mechanism_cluster_key(schema):
    try:
        sc = schema if isinstance(schema, dict) else asdict(schema)
    except Exception:
        sc = dict(schema or {}) if isinstance(schema, dict) else {}
    roles = [str(x) for x in list(sc.get("lane_roles", []))[:3]]
    payload = {
        "mechanism_family": str(sc.get("mechanism_family", "cms_like") or "cms_like"),
        "state_contract": str(sc.get("state_contract", "none") or "none"),
        "query_contract": str(sc.get("query_contract", "simple_reduce") or "simple_reduce"),
        "lane_roles": roles,
        "replication_budget": int(sc.get("replication_budget", 3) or 3),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _sanitize_mechanism_schema_claim(schema):
    if not isinstance(schema, dict):
        return {}
    allowed_family = {"cms_like", "adaptive_count", "scout_rescue", "witness_gated", "layered_promotion", "delegate_fallback", "overflow_delegate"}
    allowed_state = {"none", "witness_gate", "overflow_witness", "delegate_flag", "sidecar_flag"}
    allowed_query = {"simple_reduce", "min_reduce", "median_reduce", "avg_reduce", "state_prefers_rescue", "trust_witness", "fallback_if_witness"}
    allowed_roles = {"replica", "main", "scout", "witness", "rescue", "fallback", "delegate", "audit", "promotion"}
    out = {}
    fam = str(schema.get("mechanism_family", "") or "").strip()
    if fam in allowed_family:
        out["mechanism_family"] = fam
    st = str(schema.get("state_contract", "") or "").strip()
    if st in allowed_state:
        out["state_contract"] = st
    qc = str(schema.get("query_contract", "") or "").strip()
    if qc in allowed_query:
        out["query_contract"] = qc
    try:
        rb = int(schema.get("replication_budget", 3))
        out["replication_budget"] = max(0, min(3, rb))
    except Exception:
        pass
    roles = []
    for x in list(schema.get("lane_roles", []))[:3]:
        sx = str(x or "").strip()
        if sx in allowed_roles:
            roles.append(sx)
    if roles:
        out["lane_roles"] = roles
    rels = []
    for x in list(schema.get("lane_relations", []))[:6]:
        sx = str(x or "").strip()
        if sx:
            rels.append(sx[:64])
    if rels:
        out["lane_relations"] = rels
    axes = []
    for x in list(schema.get("novelty_axes", []))[:8]:
        sx = str(x or "").strip()
        if sx:
            axes.append(sx[:32])
    if axes:
        out["novelty_axes"] = axes
    note = str(schema.get("mechanism_note", "") or "").strip()
    if note:
        out["mechanism_note"] = note[:160]
    return out


def _infer_mechanism_schema_from_team(team, architecture_schema=None, motif_signature=None):
    arch = architecture_schema if isinstance(architecture_schema, dict) else asdict(_infer_architecture_schema_from_team(team))
    sig = motif_signature if isinstance(motif_signature, dict) else asdict(_extract_motif_signature(team, schema=arch))
    init_fam = str(sig.get("init_family", "symmetric-init") or "symmetric-init")
    upd_fam = str(sig.get("update_family", "triple-write") or "triple-write")
    qry_fam = str(sig.get("query_family", "min-query") or "min-query")
    arch_type = str(arch.get("arch_type", "regular") or "regular")

    lane_roles = ["replica", "replica", "replica"]
    lane_relations = []
    state_contract = "none"
    query_contract = "simple_reduce"
    novelty_axes = []
    mechanism_family = "cms_like"
    mechanism_note = "cms-like replica counter sketch"
    replication_budget = 3

    if qry_fam == "min-query":
        query_contract = "min_reduce"
    elif qry_fam == "median-query":
        query_contract = "median_reduce"
    elif qry_fam == "avg-query":
        query_contract = "avg_reduce"
    elif qry_fam == "state-gated-query":
        query_contract = "state_prefers_rescue"
        novelty_axes.append("gated_query")
    elif qry_fam in {"fallback-query", "mixed-aggregator-query"}:
        query_contract = "fallback_if_witness"
        novelty_axes.append("conditional_fusion")

    if upd_fam == "read-before-write":
        lane_roles = ["scout", "main", "replica"]
        lane_relations.append("lane0->lane1:read_before_write")
        novelty_axes.append("role_split")
        replication_budget = 2
        mechanism_family = "adaptive_count"
        mechanism_note = "one scout lane informs the main estimator before full replica writing"
    elif upd_fam == "conditional-write":
        lane_roles = ["main", "promotion", "rescue"]
        lane_relations.append("lane0->lane1:on_threshold")
        lane_relations.append("lane1->lane2:on_condition")
        novelty_axes.extend(["conditional_write", "handoff"])
        replication_budget = 1
        mechanism_family = "layered_promotion"
        mechanism_note = "writes are promoted across lanes instead of replicated symmetrically"
    elif upd_fam in {"stateful-write", "rescue-write"}:
        lane_roles = ["main", "witness", "rescue"]
        lane_relations.append("lane1->lane2:on_state")
        novelty_axes.extend(["state_handoff", "role_split"])
        replication_budget = 1
        mechanism_family = "witness_gated"
        mechanism_note = "one lane stores evidence/state that governs rescue behavior"

    if arch_type in {"diamond", "overflow"}:
        if lane_roles == ["replica", "replica", "replica"]:
            lane_roles = ["main", "witness", "rescue"]
            replication_budget = 1
        state_contract = "overflow_witness" if bool(sig.get("uses_overflow_state", False)) else "witness_gate"
        if "state_handoff" not in novelty_axes:
            novelty_axes.append("state_handoff")
        mechanism_family = "overflow_delegate"
        lane_relations.append("lane1->lane2:on_overflow")
        mechanism_note = "overflow-like evidence delegates trust to a rescue/witness lane"
    elif arch_type == "pyramid":
        lane_roles = ["main", "promotion", "witness"]
        replication_budget = 1
        lane_relations.append("lane0->lane1:on_promotion")
        lane_relations.append("lane1->lane2:on_confirmation")
        novelty_axes.extend(["layered_layout", "promotion"])
        mechanism_family = "layered_promotion"
        mechanism_note = "lane roles are stratified; upper lanes act as promoted refiners"
    elif arch_type in {"elastic", "hybrid"}:
        if lane_roles == ["replica", "replica", "replica"]:
            lane_roles = ["scout", "main", "fallback"]
            replication_budget = 2
        lane_relations.append("lane0->lane2:delegate_on_disagreement")
        novelty_axes.append("delegation")
        if mechanism_family == "cms_like":
            mechanism_family = "delegate_fallback"
            mechanism_note = "one lane scouts, one estimates, and one serves as delegated fallback"

    if bool(sig.get("uses_overflow_state", False)):
        if state_contract == "none":
            state_contract = "witness_gate"
        if mechanism_family == "cms_like":
            mechanism_family = "witness_gated"
            mechanism_note = "state decides which lane to trust"
    if query_contract in {"state_prefers_rescue", "fallback_if_witness"} and state_contract == "none":
        state_contract = "delegate_flag"

    if init_fam in {"hybrid-slice-init", "layered-init"} and "layout_split" not in novelty_axes:
        novelty_axes.append("layout_split")
    if bool(sig.get("has_handoff", False)) and "handoff" not in novelty_axes:
        novelty_axes.append("handoff")

    if mechanism_family == "cms_like" and len(set(lane_roles)) == 1 and state_contract == "none" and query_contract in {"min_reduce", "median_reduce", "avg_reduce", "simple_reduce"}:
        replication_budget = 3

    return MechanismSchema(
        lane_roles=list(lane_roles),
        lane_relations=list(dict.fromkeys(lane_relations)),
        state_contract=str(state_contract),
        query_contract=str(query_contract),
        replication_budget=int(replication_budget),
        novelty_axes=list(dict.fromkeys(novelty_axes)),
        mechanism_family=str(mechanism_family),
        mechanism_note=str(mechanism_note),
    )


def _mechanism_distance_from_cms(schema):
    try:
        sc = schema if isinstance(schema, dict) else asdict(schema)
    except Exception:
        sc = dict(schema or {}) if isinstance(schema, dict) else {}
    roles = {str(x) for x in list(sc.get("lane_roles", [])) if str(x)}
    dist = 0.0
    if str(sc.get("mechanism_family", "cms_like") or "cms_like") != "cms_like":
        dist += 1.6
    if str(sc.get("state_contract", "none") or "none") != "none":
        dist += 1.6
    if str(sc.get("query_contract", "simple_reduce") or "simple_reduce") not in {"simple_reduce", "min_reduce", "median_reduce", "avg_reduce"}:
        dist += 1.2
    if len(roles) >= 2:
        dist += 1.1
    dist += 0.9 * len([r for r in roles if r not in {"replica", "main"}])
    rels = list(sc.get("lane_relations", []))
    if rels:
        dist += min(2.2, 0.7 * len(rels))
    try:
        rep = int(sc.get("replication_budget", 3) or 3)
    except Exception:
        rep = 3
    if rep < 3:
        dist += 0.7 * float(3 - rep)
    axes = list(sc.get("novelty_axes", []))
    if axes:
        dist += min(1.8, 0.35 * len(axes))
    return float(min(10.0, dist))


def _mechanism_score_terms_from_candidate_meta(candidate_meta=None):
    meta = dict(candidate_meta or {})
    mech = dict(meta.get("mechanism_schema", {}) or {})
    dist = float(meta.get("mechanism_distance_from_cms", _mechanism_distance_from_cms(mech)))
    fam = str(mech.get("mechanism_family", "cms_like") or "cms_like")
    cluster = str(meta.get("mechanism_cluster", _mechanism_cluster_key(mech)))
    roles = [str(x) for x in list(mech.get("lane_roles", [])) if str(x)]
    lane_role_entropy = 0.0
    if roles:
        counts = Counter(roles)
        total = float(sum(counts.values()))
        uniq = max(1, len(counts))
        if total > 0.0 and uniq > 1:
            ent = 0.0
            for c in counts.values():
                p = float(c) / total
                if p > 0.0:
                    ent -= p * math.log(p, 2)
            lane_role_entropy = ent / max(1.0, math.log(float(uniq), 2))
    cms_similarity = max(0.0, 1.0 - (float(dist) / 10.0))
    mechanism_frequency = int(meta.get("mechanism_frequency", 0) or 0)
    mechanism_rarity_bonus = float(meta.get("mechanism_rarity_bonus", max(0.0, 3.0 - min(3, mechanism_frequency))))
    return {
        "mechanism_distance_from_cms": float(dist),
        "mechanism_family": fam,
        "mechanism_cluster": cluster,
        "state_contract": str(mech.get("state_contract", "none") or "none"),
        "query_contract": str(mech.get("query_contract", "simple_reduce") or "simple_reduce"),
        "replication_budget": int(mech.get("replication_budget", 3) or 3),
        "lane_role_entropy": float(lane_role_entropy),
        "cms_similarity": float(cms_similarity),
        "mechanism_frequency": int(mechanism_frequency),
        "mechanism_rarity_bonus": float(mechanism_rarity_bonus),
    }


def _compile_mechanism_schema_to_team_spec(schema, variant=0):
    sc = _sanitize_mechanism_schema_claim(schema)
    roles = [str(x) for x in list(sc.get("lane_roles", [])) if str(x)]
    mech_family = str(sc.get("mechanism_family", "") or "")
    state_contract = str(sc.get("state_contract", "none") or "none")
    query_contract = str(sc.get("query_contract", "simple_reduce") or "simple_reduce")
    variant = int(variant or 0)

    if mech_family == "":
        if state_contract != "none":
            mech_family = "witness_gated"
        elif "promotion" in roles or "rescue" in roles:
            mech_family = "layered_promotion"
        elif "fallback" in roles or "delegate" in roles:
            mech_family = "delegate_fallback"
        elif "scout" in roles:
            mech_family = "adaptive_count"
        else:
            mech_family = "cms_like"

    if mech_family == "cms_like":
        init_expr = "list_3(hash_salt(0,e,1), safe_mod(hash_salt(0,e,1),102), 102, hash_salt(1,e,1), safe_mod(hash_salt(1,e,1),102), 102, hash_salt(2,e,1), safe_mod(hash_salt(2,e,1),102), 102)"
        update_expr = "base(update_count(e,0,1), update_count(e,1,1), update_count(e,2,1))"
        query_expr = "base_sel(0, query_date(e,0), query_date(e,1), query_date(e,2))"
    elif mech_family in {"adaptive_count", "scout_rescue", "delegate_fallback"}:
        init_expr = "list_3(hash_on_slice(0,e,0,8), safe_mod(hash_on_slice(0,e,0,8),102), 102, hash_salt(1,e,1), safe_mod(hash_salt(1,e,17),102), 102, hash_on_slice(2,e,8,16), safe_mod(hash_salt(2,e,31),102), 102)"
        update_expr = "base(write_count(e,0,safe_add(query_count(e,0),1)), write_count(e,1,safe_add(query_count(e,1),1)), updatecount_if(gt(abs_int(safe_sub(query_count(e,0),query_count(e,1))),3),e,2,1))"
        query_expr = "base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2))"
    elif mech_family in {"witness_gated", "overflow_delegate"}:
        init_expr = "list_3(hash_on_slice(0,e,0,8), safe_mod(hash_on_slice(0,e,0,8),102), 102, hash_salt(1,e,1), safe_mod(hash_salt(1,e,17),102), 102, hash_salt(2,e,31), safe_mod(hash_salt(2,e,31),102), 102)"
        update_expr = "base(write_count(e,0,safe_add(query_count(e,0),1)), update_state(e,1,gt(abs_int(safe_sub(query_count(e,0),query_count(e,2))),3)), writecount_if(query_state(e,1),e,2,safe_add(query_count(e,2),1)))"
        if query_contract in {"state_prefers_rescue", "fallback_if_witness", "trust_witness"}:
            query_expr = "base_sel(cnt_rdstate(e,1), query_date(e,0), query_date(e,2), query_date(e,0))"
        else:
            query_expr = "base_sel(0, query_date(e,0), query_date(e,2), query_date(e,0))"
    else:  # layered_promotion or fallback
        init_expr = "list_3(hash_salt(0,str_slice(e,0,8),1), safe_mod(hash_salt(0,str_slice(e,0,8),1),102), 102, hash_salt(1,e,17), safe_mod(hash_salt(1,e,17),102), 102, hash_salt(2,e,31), safe_mod(hash_salt(2,e,31),102), 102)"
        update_expr = "base(update_count(e,0,1), updatecount_if(gt(query_count(e,0),7),e,1,1), writecount_if(gt(query_count(e,1),query_count(e,0)),e,2,safe_add(query_count(e,2),1)))"
        query_expr = "base_sel(0, query_date(e,0), query_date(e,1), query_date(e,2))"

    if variant % 3 == 1 and "base_sel(0" in query_expr:
        query_expr = query_expr.replace("base_sel(0", "base_sel(2", 1)
    elif variant % 3 == 2 and "base_sel(0" in query_expr:
        query_expr = query_expr.replace("base_sel(0", "base_sel(3", 1)

    return {
        "mode": "mechanism_team",
        "mechanism_schema": copy.deepcopy(sc),
        "init_dex": init_expr,
        "update": update_expr,
        "query": query_expr,
        "rationale": str(sc.get("mechanism_note", "mechanism-schema-compiled") or "mechanism-schema-compiled"),
        "source": "mechanism_compiler",
    }


def _build_local_mechanism_seed_teams():
    schema_bank = [
        {
            "mechanism_family": "adaptive_count",
            "lane_roles": ["scout", "main", "fallback"],
            "query_contract": "median_reduce",
            "state_contract": "none",
            "replication_budget": 2,
            "novelty_axes": ["role_split", "delegation"],
            "mechanism_note": "scout lane plus delegated fallback lane",
        },
        {
            "mechanism_family": "witness_gated",
            "lane_roles": ["main", "witness", "rescue"],
            "query_contract": "state_prefers_rescue",
            "state_contract": "witness_gate",
            "replication_budget": 1,
            "novelty_axes": ["state_handoff", "conditional_fusion"],
            "mechanism_note": "witness lane governs rescue lane via state",
        },
        {
            "mechanism_family": "layered_promotion",
            "lane_roles": ["main", "promotion", "witness"],
            "query_contract": "min_reduce",
            "state_contract": "none",
            "replication_budget": 1,
            "novelty_axes": ["promotion", "layout_split"],
            "mechanism_note": "promoted refinement lane plus witness lane",
        },
    ]
    out = []
    seen = set()
    for i, sc in enumerate(schema_bank):
        spec = _compile_mechanism_schema_to_team_spec(sc, variant=i)
        key = (str(spec.get("init_dex", "")), str(spec.get("update", "")), str(spec.get("query", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def _candidate_meta_from_team(evaluator, team):
    schema = _infer_architecture_schema_from_team(team)
    motif = _extract_motif_signature(team, schema=schema)
    schema_dict = asdict(schema)
    motif_dict = asdict(motif)
    mechanism = _infer_mechanism_schema_from_team(team, architecture_schema=schema_dict, motif_signature=motif_dict)
    mechanism_dict = asdict(mechanism)
    init_key = evaluator._canonical_tree_str(team["init_dex"])
    update_key = evaluator._canonical_tree_str(team["update"])
    query_key = evaluator._canonical_tree_str(team["query"])
    schema_hash = _architecture_schema_hash(schema_dict)
    motif_key = _motif_signature_key(motif_dict)
    mechanism_key = _mechanism_schema_hash(mechanism_dict)
    mechanism_cluster = _mechanism_cluster_key(mechanism_dict)
    cms_distance = _mechanism_distance_from_cms(mechanism_dict)
    return CandidateMeta(
        family_tag=str(_team_family_tag(team)),
        key_v1=evaluator._canonical_triplet_key(team["init_dex"], team["update"], team["query"]),
        key_v2=(init_key, update_key, query_key, schema_hash, motif_key, mechanism_key),
        repair_dup_key=(init_key, update_key, schema_hash, motif_key, mechanism_cluster),
        architecture_schema=schema_dict,
        motif_signature=motif_dict,
        mechanism_schema=mechanism_dict,
        schema_hash=str(schema_hash),
        motif_key=str(motif_key),
        mechanism_key=str(mechanism_key),
        mechanism_cluster=str(mechanism_cluster),
        mechanism_distance_from_cms=float(cms_distance),
        arch_type=str(schema_dict.get("arch_type", "regular")),
    )



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
                    "mechanism_schema": copy.deepcopy(meta_dict.get("mechanism_schema", {})),
                    "mechanism_key": str(meta_dict.get("mechanism_key", "")),
                    "mechanism_cluster": str(meta_dict.get("mechanism_cluster", "")),
                    "mechanism_distance_from_cms": float(meta_dict.get("mechanism_distance_from_cms", 0.0)),
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
            "mechanism_schema": copy.deepcopy(meta_dict.get("mechanism_schema", {})),
            "mechanism_key": str(meta_dict.get("mechanism_key", "")),
            "mechanism_cluster": str(meta_dict.get("mechanism_cluster", "")),
            "mechanism_distance_from_cms": float(meta_dict.get("mechanism_distance_from_cms", 0.0)),
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




def _sanitize_architecture_schema_claim(schema):
    if not isinstance(schema, dict):
        return {}
    allowed = {
        "arch_type": {"regular", "hybrid", "pyramid", "diamond", "overflow", "elastic"},
        "handoff_policy": {"none", "overflow_to_sidecar", "layered_correction", "branch_fallback"},
        "query_fusion": {"min", "median", "avg", "base_sel", "state_gated_min", "branch_conditional"},
        "state_usage": {"none", "overflow_state", "branch_state", "mixed"},
        "layout_style": {"regular", "layered", "asymmetric_dual_path", "sidecar_heavy"},
        "primary_innovation_axis": {"", "balanced", "layout_then_update", "update_state_handoff", "layout_sidecar", "existing_primitives_only", "init_dex", "update", "query", "architecture"},
    }
    out = {}
    for k in ("arch_type", "handoff_policy", "query_fusion", "state_usage", "layout_style", "primary_innovation_axis"):
        sv = str(schema.get(k, "") or "").strip()
        if not sv:
            continue
        if k not in allowed or sv in allowed[k]:
            out[k] = sv
    try:
        nb = int(schema.get("num_branches", 3))
        out["num_branches"] = max(1, min(8, nb))
    except Exception:
        pass
    roles = schema.get("branch_roles", [])
    if isinstance(roles, (list, tuple)):
        clean_roles = []
        for x in list(roles)[:8]:
            sx = str(x or "").strip()
            if sx:
                clean_roles.append(sx[:32])
        if clean_roles:
            out["branch_roles"] = clean_roles
    note = str(schema.get("innovation_note", "") or "").strip()
    if note:
        out["innovation_note"] = note[:160]
    return out


def _materialization_strength_from_candidate_meta(candidate_meta):
    meta = dict(candidate_meta or {})
    sig = dict(meta.get("motif_signature", {}) or {})
    strength = 0.0
    for k in (
        "uses_overflow_state", "has_handoff", "has_sidecar_branch", "is_layered",
        "query_fuses_multi_branch", "update_is_conditional", "update_is_read_before_write",
        "update_is_branch_asymmetric", "init_is_asymmetric",
    ):
        if bool(sig.get(k, False)):
            strength += 1.0
    if str(meta.get("arch_type", "regular") or "regular") != "regular":
        strength += 1.0
    return float(strength)


def _safe_gp_expr_for_semantic_repair(which: str, flavor: str = "") -> str:
    which = str(which or "").strip().lower()
    flavor = str(flavor or "").strip().lower()
    if which == "init_dex":
        if flavor == "layered":
            return (
                "list_3(hash_salt(0,e,1), safe_mod(hash_salt(0,str_slice(e,0,8),1),102), 102, "
                "hash_salt(1,e,1), safe_mod(hash_salt(1,e,3),102), 102, "
                "hash_salt(2,e,32), safe_mod(hash_salt(2,e,1),102), 102)"
            )
        return (
            "list_3(hash_salt(0,e,1), safe_mod(hash_salt(0,e,1),102), 102, "
            "hash_salt(1,e,1), safe_mod(hash_salt(1,e,1),102), 102, "
            "hash_salt(2,e,1), safe_mod(hash_salt(2,e,1),102), 102)"
        )
    if which == "update":
        if flavor == "read_before_write":
            return (
                "base(write_count(e,0,safe_add(query_count(e,0),1)), "
                "write_count(e,1,safe_add(query_count(e,1),1)), "
                "write_count(e,2,safe_add(query_count(e,2),1)))"
            )
        return "base(update_count(e,0,1), update_count(e,1,1), update_count(e,2,1))"
    if which == "query":
        if flavor == "median":
            return "base_sel(2, query_date(e,0), query_date(e,1), query_date(e,2))"
        return "base_sel(0, query_date(e,0), query_date(e,1), query_date(e,2))"
    return ""


def _parse_expr_or_none(expr: str, pset):
    expr = str(expr or "").strip()
    if not expr:
        return None
    try:
        return INDIVIDUAL_CLS.from_string(expr, pset)
    except Exception:
        return None


def _semantic_repair_component_with_evaluator(evaluator, which: str, tree, pset_map):
    which = str(which or "").strip().lower()
    if tree is None or which not in {"init_dex", "update", "query"}:
        return tree
    best = tree
    try:
        canon = evaluator._canonical_tree_str(tree)
        reparsed = _parse_expr_or_none(canon, pset_map[which])
        if reparsed is not None:
            best = reparsed
    except Exception:
        pass

    try:
        if which == "init_dex":
            info = evaluator.analyze_init_tree(best)
            ast0 = evaluator._simplify_ast(evaluator._tree_to_ast(best))
            pat0 = evaluator._ast_pattern_summary("init", ast0)
            if (not bool(info.get("root_ok", False))) or int(info.get("hash_calls", 0)) <= 0 or int(pat0.get("nonconst_hash_idx", 0)) > 0:
                fallback = _parse_expr_or_none(_safe_gp_expr_for_semantic_repair("init_dex"), pset_map["init_dex"])
                if fallback is not None:
                    best = fallback
        elif which == "update":
            ast0 = evaluator._simplify_ast(evaluator._tree_to_ast(best))
            eff0 = evaluator._ast_effect_summary(ast0)
            pat0 = evaluator._ast_pattern_summary("update", ast0)
            if int(eff0.get("real_write_calls", 0)) <= 0 or int(pat0.get("bad_write_context", 0)) > 0 or int(pat0.get("nonconst_path_idx", 0)) > 0:
                best_txt = str(best)
                flavor = "read_before_write" if "query_count(" in best_txt else ""
                fallback = _parse_expr_or_none(_safe_gp_expr_for_semantic_repair("update", flavor=flavor), pset_map["update"])
                if fallback is not None:
                    best = fallback
        elif which == "query":
            ast0 = evaluator._simplify_ast(evaluator._tree_to_ast(best))
            eff0 = evaluator._ast_effect_summary(ast0)
            pat0 = evaluator._ast_pattern_summary("query", ast0)
            qinfo = evaluator.analyze_query_tree(best)
            if (not bool(qinfo.get("root_ok", False))) or int(eff0.get("query_date_calls", 0)) <= 0 or int(pat0.get("nonconst_path_idx", 0)) > 0:
                fallback = _parse_expr_or_none(_safe_gp_expr_for_semantic_repair("query", flavor="median"), pset_map["query"])
                if fallback is not None:
                    best = fallback
    except Exception:
        return best
    return best


def _semantic_repair_team_with_evaluator(evaluator, team, pset_map):
    if not isinstance(team, dict):
        return team
    out = {
        "init_dex": _semantic_repair_component_with_evaluator(evaluator, "init_dex", team.get("init_dex"), pset_map),
        "update": _semantic_repair_component_with_evaluator(evaluator, "update", team.get("update"), pset_map),
        "query": _semantic_repair_component_with_evaluator(evaluator, "query", team.get("query"), pset_map),
    }
    if isinstance(team.get("_claimed_architecture_schema", {}), dict) and team.get("_claimed_architecture_schema"):
        out["_claimed_architecture_schema"] = copy.deepcopy(team.get("_claimed_architecture_schema", {}))
    for k, v in list(team.items()):
        if k not in out and k.startswith("_"):
            out[k] = copy.deepcopy(v)
    return out





def _estimate_numeric_risk_from_team(evaluator, team):
    init_txt = _tree_text_for_family(team.get("init_dex"))
    upd_txt = _tree_text_for_family(team.get("update"))
    qry_txt = _tree_text_for_family(team.get("query"))
    score = 0.0
    reasons = []

    def _has_any(txt, toks):
        return any(tok in txt for tok in toks)

    if init_txt.count("safe_div(") > 0:
        score += 3.0
        reasons.append("init_safe_div")
    if qry_txt.count("safe_div(") > 0:
        score += 3.5
        reasons.append("query_safe_div")
    if upd_txt.count("safe_div(") > 0:
        score += 1.5
        reasons.append("update_safe_div")

    if ("safe_mul(" in qry_txt) and _has_any(qry_txt, ["query_date(", "cnt_rdstate(", "median3(", "sum3("]):
        score += 3.0
        reasons.append("query_scaled_read")
    if qry_txt.count("query_date(") >= 2 and _has_any(qry_txt, ["safe_mul(", "sum3(", "median3(", "safe_add("]):
        score += 1.8
        reasons.append("query_multi_read_agg")
    if ("cnt_rdstate(" in qry_txt) and _has_any(qry_txt, ["safe_mul(", "safe_div(", "safe_add(", "sum3(", "median3("]):
        score += 2.4
        reasons.append("state_gate_amplifier")
    if qry_txt.count("query_date(") >= 3 and ("base_sel(3" in qry_txt or "sum3(" in qry_txt):
        score += 1.6
        reasons.append("avg_like_read_amplifier")

    if init_txt.count("safe_mod(") >= 2 and _has_any(init_txt, ["hash_salt(", "hash_on_slice("]):
        score += 1.3
        reasons.append("nested_hash_mod")
    if init_txt.count("lt(") + init_txt.count("gt(") + init_txt.count("eq(") >= 2:
        score += 1.1
        reasons.append("init_comparison_layout")
    if _has_any(init_txt, ["hash_on_slice(", "hash_salt("]) and "safe_div(" in init_txt:
        score += 2.2
        reasons.append("hash_div_layout")
    if init_txt.count("hash_on_slice(") >= 2 and init_txt.count("str_slice(") >= 2:
        score += 1.0
        reasons.append("deep_slice_hash_layout")

    if upd_txt.count("write_count(") + upd_txt.count("update_count(") >= 3 and ("query_count(" in upd_txt) and _has_any(upd_txt, ["safe_mul(", "safe_div(", "safe_add("]):
        score += 2.0
        reasons.append("write_read_feedback")
    if upd_txt.count("update_state(") + upd_txt.count("writestate_if(") > 0 and _has_any(upd_txt, ["safe_mul(", "safe_div("]):
        score += 1.7
        reasons.append("state_write_amplifier")
    if upd_txt.count("writecount_if(") + upd_txt.count("updatecount_if(") >= 2 and upd_txt.count("query_count(") >= 2:
        score += 1.2
        reasons.append("conditional_feedback_loop")

    return {"score": float(score), "reasons": reasons}


def _numeric_risk_phase_profile(evaluator, cfg=None, phase="llm_validate"):
    cfg = cfg if isinstance(cfg, dict) else {}
    phase = str(phase or "llm_validate").strip().lower()
    prefix = "gp" if phase.startswith("gp") else "llm"
    try:
        raw_e0 = float(getattr(evaluator, "E0", 0.0) or 0.0)
    except Exception:
        raw_e0 = 0.0
    try:
        pkts = max(1, int(getattr(evaluator, "pkts", 0) or 0))
    except Exception:
        pkts = 1
    if raw_e0 > 0.0:
        base_e0 = max(1.0, float(raw_e0))
    else:
        # Seed 阶段 E0 还没初始化时，不要把 probe 阈值压得过低。
        # 用与数据规模相关的保守 fallback，避免 LLM 候选在 validate 前门被大面积误杀。
        base_e0 = max(1.0, float(pkts) * 25.0)

    if prefix == "llm":
        # Seed/novelty 阶段不要再把 warning 直接当 reject，避免 LLM 候选在 validate 前门被大量清空。
        default_warn = 2.4
        default_reject = 3.6
        default_hard = 6.4
        default_err_mult = 16.0
        default_err_cap = 1.5e6
        default_probe_err_floor = max(12000.0, float(pkts) * 8.0)
        default_catastrophic_mult = 3.2
        default_catastrophic_err_cap = max(default_err_cap, 4.5e6)
        default_catastrophic_err_floor = max(default_probe_err_floor * 3.0, float(pkts) * 22.0)
        default_penalty_scale = 8.0
        default_warning_is_reject = False
        if phase == "llm_repair":
            default_warn = 2.2
            default_reject = 3.3
        elif phase == "llm_novelty":
            default_warn = 2.5
            default_reject = 3.8
    else:
        default_warn = 2.0
        default_reject = 3.2
        default_hard = 7.2
        default_err_mult = 18.0
        default_err_cap = 5.0e6
        default_probe_err_floor = max(8000.0, float(pkts) * 8.0)
        default_catastrophic_mult = 3.0
        default_catastrophic_err_cap = max(default_err_cap, 2.0e7)
        default_catastrophic_err_floor = max(default_probe_err_floor * 3.0, float(pkts) * 24.0)
        default_penalty_scale = 6.0
        default_warning_is_reject = True

    warn_score = float(cfg.get(f"{prefix}_numeric_risk_warn_score", default_warn))
    reject_score = float(cfg.get(f"{prefix}_numeric_risk_reject_score", default_reject))
    hard_reject_score = float(cfg.get(f"{prefix}_numeric_risk_hard_reject_score", default_hard))
    probe_min_score = float(cfg.get(f"{prefix}_numeric_risk_probe_min_score", warn_score))
    err_mult = float(cfg.get(f"{prefix}_numeric_risk_probe_err_mult", default_err_mult))
    err_cap = float(cfg.get(f"{prefix}_numeric_risk_probe_err_cap", default_err_cap))
    probe_err_floor = float(cfg.get(f"{prefix}_numeric_risk_probe_err_floor", default_probe_err_floor))
    catastrophic_mult = float(cfg.get(f"{prefix}_numeric_risk_catastrophic_mult", default_catastrophic_mult))
    catastrophic_err_cap = float(cfg.get(f"{prefix}_numeric_risk_catastrophic_err_cap", default_catastrophic_err_cap))
    catastrophic_err_floor = float(cfg.get(f"{prefix}_numeric_risk_catastrophic_err_floor", default_catastrophic_err_floor))
    penalty_scale = float(cfg.get(f"{prefix}_numeric_risk_penalty_scale", default_penalty_scale))
    stage_idx = 1 if phase in {"llm_validate", "llm_novelty", "llm_repair", "gp_child"} else 0
    warning_is_reject = bool(cfg.get(f"{prefix}_numeric_risk_warning_is_reject", default_warning_is_reject))
    warning_reject_score = float(cfg.get(f"{prefix}_numeric_risk_warning_reject_score", warn_score))
    probe_threshold = min(float(err_cap), max(float(probe_err_floor), float(err_mult) * float(base_e0)))
    catastrophic_threshold = min(float(catastrophic_err_cap), max(float(catastrophic_err_floor), float(probe_threshold) * float(catastrophic_mult)))
    return {
        "phase": phase,
        "prefix": prefix,
        "stage_idx": int(stage_idx),
        "warn_score": float(warn_score),
        "reject_score": float(reject_score),
        "hard_reject_score": float(hard_reject_score),
        "probe_min_score": float(probe_min_score),
        "probe_threshold": float(probe_threshold),
        "catastrophic_threshold": float(catastrophic_threshold),
        "penalty_scale": float(penalty_scale),
        "warning_is_reject": bool(warning_is_reject),
        "warning_reject_score": float(warning_reject_score),
    }



def _numeric_risk_attach_unified_verdict(out, profile, cfg=None):
    cfg = cfg if isinstance(cfg, dict) else {}
    heur = float(out.get("heuristic_score", 0.0) or 0.0)
    catastrophic = bool(out.get("catastrophic", False))
    hard_reject = bool(out.get("hard_reject", False))
    reject = bool(out.get("reject", False))
    warn = bool(out.get("warn", False))
    if catastrophic or hard_reject:
        verdict = "hard_reject"
        severity = 3
    elif reject:
        verdict = "reject"
        severity = 2
    elif warn:
        verdict = "warn"
        severity = 1
    else:
        verdict = "accept"
        severity = 0
    out["verdict"] = verdict
    out["severity"] = int(severity)
    out["allow"] = bool(verdict in {"accept", "warn"})
    out["block"] = bool(not out["allow"])
    out["runtime_compatible"] = bool(out["allow"])
    if bool(out.get("warn", False)) and not str(out.get("warning", "")).strip():
        out["warning"] = f"numeric_risk:{heur:.2f}"
    prefix = str(profile.get("prefix", "llm") or "llm")
    default_seed_bank_cap = float(profile.get("warning_reject_score", profile.get("warn_score", 0.0)) or 0.0)
    if prefix == "llm":
        default_seed_bank_cap = max(default_seed_bank_cap, 3.10)
    seed_bank_cap = float(cfg.get("llm_seed_keep_max_numeric_risk", default_seed_bank_cap))
    seed_bank_reject = bool(out["block"])
    seed_bank_reason = str(out.get("reason", "")) if seed_bank_reject else ""
    if (not seed_bank_reject) and prefix == "llm" and heur > seed_bank_cap:
        seed_bank_reject = True
        seed_bank_reason = f"seed_numeric_risk_verdict:{heur:.2f}>{seed_bank_cap:.2f}"
    out["seed_bank_threshold"] = float(seed_bank_cap)
    out["seed_bank_reject"] = bool(seed_bank_reject)
    out["seed_bank_allow"] = bool(not seed_bank_reject)
    out["seed_bank_reason"] = str(seed_bank_reason)
    return out
def _numeric_risk_probe_with_evaluator(evaluator, team, cfg=None, phase="llm_validate"):
    cfg = cfg if isinstance(cfg, dict) else {}
    heur = _estimate_numeric_risk_from_team(evaluator, team)
    profile = _numeric_risk_phase_profile(evaluator, cfg=cfg, phase=phase)
    out = {
        "phase": str(profile.get("phase", phase)),
        "prefix": str(profile.get("prefix", "llm")),
        "heuristic_score": float(heur.get("score", 0.0)),
        "heuristic_reasons": list(heur.get("reasons", [])),
        "probe_run": False,
        "probe_err": None,
        "probe_fit": None,
        "catastrophic": False,
        "hard_reject": False,
        "reject": False,
        "warn": False,
        "reason": "",
        "warning": "",
        "penalty": 0.0,
        "stage_idx": int(profile.get("stage_idx", 0)),
        "probe_threshold": float(profile.get("probe_threshold", 0.0)),
        "catastrophic_threshold": float(profile.get("catastrophic_threshold", 0.0)),
    }
    if float(out["heuristic_score"]) <= 0.0:
        out["reason"] = "numeric_risk_ok:0.00"
        return _numeric_risk_attach_unified_verdict(out, profile, cfg=cfg)
    out["penalty"] = float(out["heuristic_score"]) * float(profile.get("penalty_scale", 0.0))
    if float(out["heuristic_score"]) >= float(profile.get("warn_score", 0.0)):
        out["warn"] = True
        out["warning"] = f"numeric_risk:{float(out['heuristic_score']):.2f}"
    if float(out["heuristic_score"]) >= float(profile.get("hard_reject_score", 0.0)):
        out["catastrophic"] = True
        out["hard_reject"] = True
        out["reason"] = f"numeric_risk_hard_reject:{float(out['heuristic_score']):.2f}>={float(profile.get('hard_reject_score', 0.0)):.2f}"
        return _numeric_risk_attach_unified_verdict(out, profile, cfg=cfg)
    if bool(profile.get("warning_is_reject", False)) and bool(out.get("warn", False)) and float(out["heuristic_score"]) >= float(profile.get("warning_reject_score", profile.get("warn_score", 0.0))):
        out["reject"] = True
        out["reason"] = f"numeric_risk_warning_reject:{float(out['heuristic_score']):.2f}>={float(profile.get('warning_reject_score', profile.get('warn_score', 0.0))):.2f}"
        return _numeric_risk_attach_unified_verdict(out, profile, cfg=cfg)
    should_probe = bool(float(out["heuristic_score"]) >= float(profile.get("probe_min_score", 0.0)))
    if should_probe:
        try:
            probe_fit, probe_err = evaluator._evaluate_individual_core(
                team["init_dex"], team["update"], team["query"], stage_idx=int(profile.get("stage_idx", 0)), return_case_vec=False
            )
            out["probe_run"] = True
            out["probe_fit"] = float(probe_fit)
            out["probe_err"] = float(probe_err)
            if float(probe_err) > float(profile.get("catastrophic_threshold", 0.0)):
                out["catastrophic"] = True
                out["reason"] = f"numeric_probe_catastrophic err={float(probe_err):.2f} thr={float(profile.get('catastrophic_threshold', 0.0)):.2f} stage={int(profile.get('stage_idx', 0))}"
                return _numeric_risk_attach_unified_verdict(out, profile, cfg=cfg)
            if float(probe_err) > float(profile.get("probe_threshold", 0.0)):
                out["reject"] = True
                out["reason"] = f"numeric_probe_reject err={float(probe_err):.2f} thr={float(profile.get('probe_threshold', 0.0)):.2f} stage={int(profile.get('stage_idx', 0))}"
                return _numeric_risk_attach_unified_verdict(out, profile, cfg=cfg)
        except Exception as e:
            out["probe_run"] = True
            out["catastrophic"] = True
            out["hard_reject"] = True
            out["reason"] = f"numeric_probe_failed:{type(e).__name__}"
            return _numeric_risk_attach_unified_verdict(out, profile, cfg=cfg)
    if float(out["heuristic_score"]) >= float(profile.get("reject_score", 0.0)):
        out["reject"] = True
        out["reason"] = f"numeric_risk_reject:{float(out['heuristic_score']):.2f}>={float(profile.get('reject_score', 0.0)):.2f}"
        return _numeric_risk_attach_unified_verdict(out, profile, cfg=cfg)
    out["reason"] = f"numeric_risk_ok:{float(out['heuristic_score']):.2f}"
    return _numeric_risk_attach_unified_verdict(out, profile, cfg=cfg)

def _seed_candidate_error_cap(best_err, cfg=None):
    cfg = cfg if isinstance(cfg, dict) else {}
    best_err = max(1.0, float(best_err))
    rel_mult = float(cfg.get("llm_seed_keep_err_rel_mult", 8.0))
    floor = float(cfg.get("llm_seed_keep_err_floor", 250000.0))
    abs_cap = float(cfg.get("llm_seed_keep_abs_err_cap", 2.0e6))
    return float(min(abs_cap, max(floor, best_err * rel_mult)))


def _filter_llm_seed_candidates(seed_candidates, cfg=None):
    cfg = cfg if isinstance(cfg, dict) else {}
    cands = list(seed_candidates or [])
    if not cands:
        return [], []
    cands = sorted(
        cands,
        key=lambda c: (
            float(c.get("err", 1e18) or 1e18),
            -float(c.get("fit", 0.0) or 0.0),
            -float((c.get("candidate_meta", {}) or {}).get("mechanism_distance_from_cms", c.get("mechanism_distance_from_cms", 0.0)) or 0.0),
        ),
    )
    best_err = max(1.0, float(cands[0].get("err", 1e18) or 1e18))
    err_cap = _seed_candidate_error_cap(best_err, cfg)
    max_risk = float(cfg.get("llm_seed_keep_max_numeric_risk", 3.10))
    min_keep = max(1, int(cfg.get("llm_seed_keep_min", 1)))
    kept = []
    dropped = []
    kept_ids = set()
    for idx, cand in enumerate(cands):
        err = float(cand.get("err", 1e18) or 1e18)
        nr = dict(cand.get("numeric_risk", {}) or {})
        risk = float(nr.get("heuristic_score", 0.0) or 0.0)
        reason = ""
        if idx > 0 and err > err_cap:
            reason = f"seed_err_cap:{err:.2f}>{err_cap:.2f}"
        elif idx > 0 and bool(nr.get("seed_bank_reject", False)):
            reason = str(nr.get("seed_bank_reason", "") or f"seed_numeric_risk_cap:{risk:.2f}>{max_risk:.2f}")
        elif idx > 0 and ("seed_bank_reject" not in nr) and risk > max_risk:
            reason = f"seed_numeric_risk_cap:{risk:.2f}>{max_risk:.2f}"
        if reason:
            dropped.append({"candidate": cand, "reason": reason})
            continue
        kept.append(cand)
        kept_ids.add(id(cand))
    if len(kept) < min_keep:
        for cand in cands:
            if id(cand) in kept_ids:
                continue
            kept.append(cand)
            kept_ids.add(id(cand))
            if len(kept) >= min_keep:
                break
    return kept, dropped


def _seed_spec_allowed_for_state(spec, state, cfg=None):
    cfg = cfg if isinstance(cfg, dict) else {}
    fits = list(state.get("fits", [])) if isinstance(state, dict) else []
    errs = []
    for rec in fits:
        try:
            errs.append(float(rec[1]))
        except Exception:
            continue
    if not errs:
        return True, "seed_state_skip"
    best_err = max(1.0, min(errs))
    rel_mult = float(cfg.get("llm_seed_keep_err_rel_mult", 8.0))
    abs_cap = float(cfg.get("llm_seed_keep_abs_err_cap", 2.0e6))
    err_cap = float(min(abs_cap, best_err * rel_mult))
    spec_err = float((spec or {}).get("err", 1e18) or 1e18)
    if spec_err > err_cap:
        return False, f"seed_inject_err_cap:{spec_err:.2f}>{err_cap:.2f}"
    return True, "seed_inject_ok"


def _mechanism_diversity_override(candidate_meta=None, mechanism_hist=None, recent=None, cfg=None, novelty_subtype="stable", frontier_bonus=0.0, mechanism_frontier_bonus=0.0):
    meta = dict(candidate_meta or {})
    mechanism_hist = mechanism_hist if isinstance(mechanism_hist, dict) else {"cluster": Counter(), "family": Counter()}
    recent = recent if isinstance(recent, dict) else {}
    mech_terms = _mechanism_score_terms_from_candidate_meta(meta)
    cluster = str(mech_terms.get("mechanism_cluster", "") or "")
    mech_family = str(mech_terms.get("mechanism_family", "cms_like") or "cms_like")
    dist = float(mech_terms.get("mechanism_distance_from_cms", 0.0) or 0.0)
    entropy = float(mech_terms.get("lane_role_entropy", 0.0) or 0.0)
    cluster_count = int(mechanism_hist.get("cluster", Counter()).get(cluster, 0)) if cluster else 0
    family_count = int(mechanism_hist.get("family", Counter()).get(mech_family, 0)) if mech_family else 0
    recent_cluster_count = int(recent.get("mechanism_clusters", Counter()).get(cluster, 0)) if cluster else 0
    recent_family_count = int(recent.get("mechanism_families", Counter()).get(mech_family, 0)) if mech_family else 0
    cfg = cfg if isinstance(cfg, dict) else {}
    novelty_subtype = str(novelty_subtype or "stable").strip().lower()
    base_dist = float(cfg.get("llm_mechanism_diversity_escape_distance_irregular", 2.8) if novelty_subtype == "irregular" else cfg.get("llm_mechanism_diversity_escape_distance", 3.2))
    base_entropy = float(cfg.get("llm_mechanism_diversity_escape_entropy", 0.08))
    frontier_req = float(cfg.get("llm_mechanism_diversity_escape_frontier_bonus", 0.35))
    combined_frontier = float(frontier_bonus) + 1.35 * float(mechanism_frontier_bonus)
    effective_dist = float(dist) + 0.45 * float(entropy) + 0.70 * max(0.0, combined_frontier)
    if cluster and cluster_count == 0 and recent_cluster_count == 0 and effective_dist >= max(2.4, base_dist - 0.5):
        return True
    if cluster and cluster_count <= 1 and recent_cluster_count == 0 and family_count <= 1 and effective_dist >= base_dist and (entropy >= base_entropy or combined_frontier >= frontier_req):
        return True
    if cluster and cluster_count <= 1 and recent_cluster_count == 0 and effective_dist >= (base_dist + 0.35) and combined_frontier >= max(0.20, frontier_req - 0.15):
        return True
    if mech_family and family_count <= 1 and recent_family_count == 0 and effective_dist >= (base_dist + 0.10):
        return True
    if novelty_subtype == "irregular" and cluster and cluster_count <= 1 and recent_cluster_count == 0 and effective_dist >= max(2.3, base_dist - 0.6):
        return True
    return False

def _repair_candidate_quality_gate(chk_fit, chk_err, island_best_fit, island_best_err, target_fit=None, target_err=None, cfg=None):
    cfg = cfg if isinstance(cfg, dict) else {}
    chk_fit = float(chk_fit)
    chk_err = float(chk_err)
    island_best_fit = float(island_best_fit)
    island_best_err = max(1.0, float(island_best_err))
    try:
        target_fit = float(target_fit)
    except Exception:
        target_fit = 0.0
    try:
        target_err = float(target_err)
    except Exception:
        target_err = float('inf')
    rel_best = float(cfg.get("llm_repair_max_err_rel_to_best", 8.0))
    floor = float(cfg.get("llm_repair_err_floor", 1200.0))
    abs_cap = float(cfg.get("llm_repair_abs_err_cap", 20000.0))
    target_improve_ratio = float(cfg.get("llm_repair_target_improve_ratio", 0.10))
    err_cap = min(abs_cap, max(floor, island_best_err * rel_best))
    if chk_err > err_cap:
        return False, f"repair_err_too_far_from_best:{chk_err:.2f}>{err_cap:.2f}"
    if math.isfinite(target_err) and target_err < 1.0e18:
        target_cap = float(target_err) * (1.0 - target_improve_ratio)
        if chk_err >= target_cap and chk_fit <= target_fit:
            return False, f"repair_not_better_than_target:{chk_err:.2f}>={target_cap:.2f}"
    return True, "repair_quality_ok"


def _should_force_single_tree_repair_from_hints(hints, duplicate_blocklist=None, cfg=None):
    cfg = cfg if isinstance(cfg, dict) else {}
    policy = str(cfg.get("llm_repair_single_tree_policy", "auto") or "auto").strip().lower()
    if policy in {"always", "true", "single_tree", "force"}:
        return True
    if policy in {"never", "false", "team", "full_team"}:
        return False
    dominant = _dominant_failure_bucket_from_hints(hints)
    local_fix_buckets = {"bad_write_ctx", "real_write_zero", "query_date_zero", "nonconst_path"}
    if dominant in local_fix_buckets:
        return True
    block = duplicate_blocklist if isinstance(duplicate_blocklist, dict) else {}
    repair_dup_n = len(list(block.get("repair_dup_keys", [])))
    family_dup_n = int(sum(dict(block.get("family_tags", Counter())).values()))
    mech_dup_n = int(sum(dict(block.get("mechanism_clusters", Counter())).values()))
    if repair_dup_n >= 2 or family_dup_n >= 4 or mech_dup_n >= 2:
        return False
    return bool(cfg.get("llm_repair_single_tree_default", False))

def _dominant_family_cooldown_snapshot(state, family_hist=None, topk=3, min_count=2):
    hist = family_hist if isinstance(family_hist, dict) else _family_histogram_from_state(state)
    dominant = Counter()
    for fam, cnt in list(hist.get("exact", Counter()).most_common(max(1, int(topk)))):
        if int(cnt) >= int(min_count):
            dominant[str(fam)] += int(cnt)
    recent = _recent_promoted_cooldown_snapshot(state if isinstance(state, dict) else {})
    for fam, cnt in dict(recent.get("family_tags", Counter())).items():
        if int(cnt) > 0:
            dominant[str(fam)] += int(cnt)
    mech_hist = _mechanism_histogram_from_state(state if isinstance(state, dict) else {})
    return {
        "family_tags": dominant,
        "recent": recent,
        "mechanism_cluster": Counter(dict(mech_hist.get("cluster", Counter()))),
        "mechanism_family": Counter(dict(mech_hist.get("family", Counter()))),
    }


def _dominant_family_cooldown_reason(candidate_meta=None, cooldown=None, motif_distance_score=0.0, frontier_bonus=0.0, mechanism_frontier_bonus=0.0, cfg=None, novelty_subtype="stable"):
    meta = dict(candidate_meta or {})
    fam = str(meta.get("family_tag", "") or "").strip()
    if not fam:
        return ""
    cooldown = cooldown if isinstance(cooldown, dict) else {"family_tags": Counter(), "recent": {}, "mechanism_cluster": Counter(), "mechanism_family": Counter()}
    fam_cnt = int(cooldown.get("family_tags", Counter()).get(fam, 0))
    recent_obj = (cooldown.get("recent", {}) or {}) if isinstance(cooldown, dict) else {}
    recent_cnt = int(recent_obj.get("family_tags", Counter()).get(fam, 0))
    cfg = cfg if isinstance(cfg, dict) else {}
    novelty_subtype = str(novelty_subtype or "stable").strip().lower()
    max_recent = int(cfg.get("llm_novelty_dominant_recent_cap", 2))
    dominant_min = int(cfg.get("llm_novelty_dominant_min_count", 3))
    min_motif_escape = float(cfg.get("llm_novelty_dominant_escape_min_motif_distance", 2.6 if novelty_subtype == "irregular" else 1.8))
    min_frontier_escape = float(cfg.get("llm_novelty_dominant_escape_min_frontier_bonus", 0.55))
    mechanism_hist = {
        "cluster": Counter(dict(cooldown.get("mechanism_cluster", Counter()))),
        "family": Counter(dict(cooldown.get("mechanism_family", Counter()))),
    }
    if _mechanism_diversity_override(
        candidate_meta=meta,
        mechanism_hist=mechanism_hist,
        recent=recent_obj,
        cfg=cfg,
        novelty_subtype=novelty_subtype,
        frontier_bonus=frontier_bonus,
        mechanism_frontier_bonus=mechanism_frontier_bonus,
    ):
        return ""
    combined_frontier = float(frontier_bonus) + 1.25 * float(mechanism_frontier_bonus)
    if recent_cnt >= max_recent and float(motif_distance_score) < float(min_motif_escape) and combined_frontier < float(min_frontier_escape):
        return f"novelty_dominant_family_cooldown:{fam}:recent={recent_cnt}"
    if fam_cnt >= dominant_min and float(motif_distance_score) < float(min_motif_escape) and combined_frontier < float(min_frontier_escape):
        return f"novelty_dominant_family_cooldown:{fam}:count={fam_cnt}"
    return ""

def _frontier_bonus_from_candidate_meta(candidate_meta=None, guidance=None):
    meta = dict(candidate_meta or {})
    schema = dict(meta.get("architecture_schema", {}) or {})
    frontier = guidance.get("schema_frontier", {}) if isinstance(guidance, dict) else {}
    if not isinstance(frontier, dict):
        return 0.0, []
    bonus = 0.0
    hits = []
    for axis_info in list(frontier.get("axes", [])):
        axis = str(axis_info.get("axis", "") or "")
        under = [str(x) for x in list(axis_info.get("underexplored", [])) if str(x)]
        if not axis or not under:
            continue
        val = str(schema.get(axis, "") or "")
        if val and val in under:
            rank = under.index(val)
            inc = max(0.15, 0.50 - 0.12 * float(rank))
            bonus += inc
            hits.append(f"{axis}={val}")
    return float(bonus), hits


def _auto_promote_novelty_subtype(team, candidate_meta=None, requested="stable"):
    req = str(requested or "stable").strip().lower()
    if req == "irregular":
        return "irregular", "requested_irregular"
    meta = dict(candidate_meta or {})
    arch_type = str(meta.get("arch_type", meta.get("architecture_schema", {}).get("arch_type", "regular")) or "regular")
    strength = _materialization_strength_from_candidate_meta(meta)
    fam_tag = str(meta.get("family_tag", "") or (_team_family_tag(team) if isinstance(team, dict) and all(k in team for k in ("init_dex", "update", "query")) else ""))
    if arch_type != "regular" and strength >= 5.5:
        return "irregular", f"auto_irregular:{arch_type}:strength={strength:.1f}"
    if _is_irregular_family_tag(fam_tag) and strength >= 5.0:
        return "irregular", f"auto_irregular_family:{fam_tag}"
    return req, ""


def _key_token(obj) -> str:
    if obj is None:
        return ""
    if isinstance(obj, (list, tuple)):
        try:
            return json.dumps(list(obj), ensure_ascii=False, sort_keys=False)
        except Exception:
            return repr(tuple(obj))
    if isinstance(obj, dict):
        try:
            return json.dumps(obj, ensure_ascii=False, sort_keys=True)
        except Exception:
            return repr(obj)
    return str(obj)


def _push_recent_window_value(lst, value, cap=16):
    if not isinstance(lst, list):
        lst = []
    value = str(value or "").strip()
    if not value:
        return lst
    lst.append(value)
    cap = max(1, int(cap))
    if len(lst) > cap:
        del lst[:-cap]
    return lst



def _ensure_recent_duplicate_basin_state(state):
    if not isinstance(state, dict):
        return {"repair_dup_keys": [], "key_v2": [], "schema_motif": [], "family_tags": [], "mechanism_clusters": [], "mechanism_families": []}
    cur = state.get("recent_duplicate_basin")
    if not isinstance(cur, dict):
        cur = {"repair_dup_keys": [], "key_v2": [], "schema_motif": [], "family_tags": [], "mechanism_clusters": [], "mechanism_families": []}
        state["recent_duplicate_basin"] = cur
    for k in ("repair_dup_keys", "key_v2", "schema_motif", "family_tags", "mechanism_clusters", "mechanism_families"):
        if not isinstance(cur.get(k), list):
            cur[k] = []
    return cur


def _duplicate_blocklist_snapshot(state):
    basin = _ensure_recent_duplicate_basin_state(state if isinstance(state, dict) else {})
    return {
        "repair_dup_keys": set(str(x) for x in list(basin.get("repair_dup_keys", [])) if str(x)),
        "key_v2": set(str(x) for x in list(basin.get("key_v2", [])) if str(x)),
        "schema_motif": Counter(str(x) for x in list(basin.get("schema_motif", [])) if str(x)),
        "family_tags": Counter(str(x) for x in list(basin.get("family_tags", [])) if str(x)),
        "mechanism_clusters": Counter(str(x) for x in list(basin.get("mechanism_clusters", [])) if str(x)),
        "mechanism_families": Counter(str(x) for x in list(basin.get("mechanism_families", [])) if str(x)),
    }


def _register_duplicate_reject_in_state(state, candidate_meta=None, chk=None, family_tag=""):
    if not isinstance(state, dict):
        return None
    basin = _ensure_recent_duplicate_basin_state(state)
    meta = dict(candidate_meta or {})
    if not meta and isinstance(chk, dict):
        meta = dict(chk.get("candidate_meta", {}) or {})
    fam = str(family_tag or meta.get("family_tag", "") or "").strip()
    repair_dup_key = meta.get("repair_dup_key", None)
    if repair_dup_key is None and isinstance(chk, dict):
        repair_dup_key = chk.get("repair_dup_key", None)
    key_v2 = meta.get("key_v2", None)
    if key_v2 is None and isinstance(chk, dict):
        key_v2 = chk.get("key_v2", None)
    schema_hash = str(meta.get("schema_hash", "") or "").strip()
    motif_key = str(meta.get("motif_key", "") or "").strip()
    mechanism_cluster = str(meta.get("mechanism_cluster", "") or "").strip()
    mechanism_family = str((meta.get("mechanism_schema", {}) or {}).get("mechanism_family", meta.get("mechanism_family", "")) or "").strip()
    basin["repair_dup_keys"] = _push_recent_window_value(basin.get("repair_dup_keys", []), _key_token(repair_dup_key), cap=20)
    basin["key_v2"] = _push_recent_window_value(basin.get("key_v2", []), _key_token(key_v2), cap=20)
    if schema_hash or motif_key:
        basin["schema_motif"] = _push_recent_window_value(basin.get("schema_motif", []), f"{schema_hash}|{motif_key}", cap=24)
    if fam:
        basin["family_tags"] = _push_recent_window_value(basin.get("family_tags", []), fam, cap=24)
    if mechanism_cluster:
        basin["mechanism_clusters"] = _push_recent_window_value(basin.get("mechanism_clusters", []), mechanism_cluster, cap=24)
    if mechanism_family:
        basin["mechanism_families"] = _push_recent_window_value(basin.get("mechanism_families", []), mechanism_family, cap=24)
    return basin

def _ensure_recent_promoted_cooldown_state(state):
    if not isinstance(state, dict):
        return {"family_tags": [], "motif_keys": [], "schema_hashes": [], "mechanism_clusters": [], "mechanism_families": []}
    cur = state.get("recent_promoted_structure")
    if not isinstance(cur, dict):
        cur = {"family_tags": [], "motif_keys": [], "schema_hashes": [], "mechanism_clusters": [], "mechanism_families": []}
        state["recent_promoted_structure"] = cur
    for k in ("family_tags", "motif_keys", "schema_hashes", "mechanism_clusters", "mechanism_families"):
        if not isinstance(cur.get(k), list):
            cur[k] = []
    return cur


def _recent_promoted_cooldown_snapshot(state):
    cur = _ensure_recent_promoted_cooldown_state(state if isinstance(state, dict) else {})
    return {
        "family_tags": Counter(str(x) for x in list(cur.get("family_tags", [])) if str(x)),
        "motif_keys": Counter(str(x) for x in list(cur.get("motif_keys", [])) if str(x)),
        "schema_hashes": Counter(str(x) for x in list(cur.get("schema_hashes", [])) if str(x)),
        "mechanism_clusters": Counter(str(x) for x in list(cur.get("mechanism_clusters", [])) if str(x)),
        "mechanism_families": Counter(str(x) for x in list(cur.get("mechanism_families", [])) if str(x)),
    }


def _record_recent_promoted_structure(state, candidate_meta=None):
    if not isinstance(state, dict):
        return None
    cur = _ensure_recent_promoted_cooldown_state(state)
    meta = dict(candidate_meta or {})
    fam = str(meta.get("family_tag", "") or "").strip()
    motif = str(meta.get("motif_key", "") or "").strip()
    schema = str(meta.get("schema_hash", "") or "").strip()
    mech_cluster = str(meta.get("mechanism_cluster", "") or "").strip()
    mech_schema = dict(meta.get("mechanism_schema", {}) or {})
    mech_family = str(mech_schema.get("mechanism_family", meta.get("mechanism_family", "")) or "").strip()
    if fam:
        cur["family_tags"] = _push_recent_window_value(cur.get("family_tags", []), fam, cap=10)
    if motif:
        cur["motif_keys"] = _push_recent_window_value(cur.get("motif_keys", []), motif, cap=14)
    if schema:
        cur["schema_hashes"] = _push_recent_window_value(cur.get("schema_hashes", []), schema, cap=14)
    if mech_cluster:
        cur["mechanism_clusters"] = _push_recent_window_value(cur.get("mechanism_clusters", []), mech_cluster, cap=14)
    if mech_family:
        cur["mechanism_families"] = _push_recent_window_value(cur.get("mechanism_families", []), mech_family, cap=12)
    return cur


def _semantic_materialization_checks(evaluator, team, candidate_meta=None, claimed_schema=None):
    reasons = []
    warns = []
    meta = dict(candidate_meta or {})
    actual_schema = dict(meta.get("architecture_schema", {}) or {})
    actual_sig = dict(meta.get("motif_signature", {}) or {})
    claim = _sanitize_architecture_schema_claim(claimed_schema)
    effective_schema = dict(actual_schema)
    for k, v in claim.items():
        if k not in effective_schema and v not in (None, "", [], {}):
            effective_schema[k] = v

    try:
        init_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["init_dex"]))
        upd_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["update"]))
        qry_ast = evaluator._simplify_ast(evaluator._tree_to_ast(team["query"]))
        init_pat = evaluator._ast_pattern_summary("init", init_ast)
        upd_pat = evaluator._ast_pattern_summary("update", upd_ast)
        qry_pat = evaluator._ast_pattern_summary("query", qry_ast)
        upd_eff = evaluator._ast_effect_summary(upd_ast)
        qry_eff = evaluator._ast_effect_summary(qry_ast)
    except Exception as e:
        return [f"semantic_materialize_ast_failed:{type(e).__name__}"], warns

    real_write_calls = int(upd_eff.get("real_write_calls", 0))
    cond_write_calls = int(upd_eff.get("conditional_write_calls", 0))
    query_date_calls = int(qry_eff.get("query_date_calls", 0))
    if real_write_calls <= 0:
        reasons.append("update_missing_real_write_primitive")
    if real_write_calls <= 0 and cond_write_calls > 0:
        reasons.append("update_missing_real_write_anchor")
    if query_date_calls <= 0:
        reasons.append("query_missing_query_date")
    if int(upd_pat.get("bad_write_context", 0)) > 0:
        reasons.append("update_bad_write_context")

    handoff_policy = str(effective_schema.get("handoff_policy", "none") or "none")
    state_usage = str(effective_schema.get("state_usage", "none") or "none")
    layout_style = str(effective_schema.get("layout_style", "regular") or "regular")
    query_fusion = str(effective_schema.get("query_fusion", "base_sel") or "base_sel")
    arch_type = str(effective_schema.get("arch_type", meta.get("arch_type", "regular")) or "regular")

    if state_usage != "none" and not bool(actual_sig.get("uses_overflow_state", False)):
        reasons.append("state_usage_not_materialized")
    if handoff_policy != "none" and not bool(actual_sig.get("has_handoff", False)):
        reasons.append("handoff_not_materialized")
    if layout_style == "layered" and not bool(actual_sig.get("is_layered", False)):
        reasons.append("layered_layout_not_materialized")
    if layout_style in {"asymmetric_dual_path", "sidecar_heavy"} and not bool(actual_sig.get("init_is_asymmetric", False) or actual_sig.get("has_sidecar_branch", False)):
        reasons.append("asymmetric_layout_not_materialized")
    if query_fusion in {"state_gated_min", "branch_conditional"} and not bool(actual_sig.get("query_fuses_multi_branch", False) or actual_sig.get("uses_overflow_state", False)):
        reasons.append("query_fusion_not_materialized")
    if arch_type != "regular":
        reasons.extend(_validate_irregular_candidate_meta({
            "architecture_schema": effective_schema,
            "motif_signature": actual_sig,
            "arch_type": arch_type,
        }))

    if claim:
        for k in ("arch_type", "handoff_policy", "state_usage", "layout_style", "query_fusion"):
            cv = str(claim.get(k, "") or "")
            av = str(actual_schema.get(k, "") or "")
            if cv and av and cv != av:
                warns.append(f"schema_claim_mismatch:{k}:{cv}!={av}")

    return sorted(set(reasons)), sorted(set(warns))


def _empty_novelty_incubator_state():
    return {
        "records": [],
        "version": 0,
        "last_incubated": [],
        "last_promoted": [],
    }


def _ensure_novelty_incubator_state(state):
    cur = state.get("novelty_incubator") if isinstance(state, dict) else None
    if not isinstance(cur, dict):
        cur = _empty_novelty_incubator_state()
        if isinstance(state, dict):
            state["novelty_incubator"] = cur
    if not isinstance(cur.get("records"), list):
        cur["records"] = []
    if not isinstance(cur.get("last_incubated"), list):
        cur["last_incubated"] = []
    if not isinstance(cur.get("last_promoted"), list):
        cur["last_promoted"] = []
    cur["version"] = int(cur.get("version", 0))
    return cur


def _age_novelty_incubator_state(state):
    inc = _ensure_novelty_incubator_state(state)
    for rec in inc.get("records", []):
        rec["age"] = int(rec.get("age", 0)) + 1
    return inc


def _incubator_structural_key_from_meta(candidate_meta, fallback_key=None):
    meta = dict(candidate_meta or {})
    repair_dup_key = meta.get("repair_dup_key", fallback_key)
    if isinstance(repair_dup_key, tuple):
        repair_dup_key = list(repair_dup_key)
    elif not isinstance(repair_dup_key, list):
        repair_dup_key = [str(repair_dup_key)] if repair_dup_key is not None else []
    payload = {
        "schema_hash": str(meta.get("schema_hash", "")),
        "motif_key": str(meta.get("motif_key", "")),
        "repair_dup_key": repair_dup_key,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _incubator_record_score(rec):
    score = float(rec.get("novelty_score", 0.0))
    score += 4.0 * min(4, int(rec.get("survival_count", 1)))
    score += 1.5 * min(6, int(rec.get("age", 0)))
    score += 0.75 * float(rec.get("materialization_strength", 0.0))
    score -= 0.00002 * min(float(rec.get("err", 1e18)), 2.0e7)
    if rec.get("holdout_lite_err", None) is not None:
        score -= 0.00003 * min(float(rec.get("holdout_lite_err", 1e18)), 2.0e7)
    if str(rec.get("novelty_subtype", "stable")) == "irregular":
        score += 2.0
    if str(rec.get("arch_type", "regular")) != "regular":
        score += 1.5
    score += 2.0 * float(rec.get("mechanism_distance_from_cms", 0.0))
    return float(score)


def _trim_novelty_incubator_records(records, cap, family_cap=2, motif_cap=3, relaxed_fraction=0.5):
    records = list(records or [])
    cap = max(1, int(cap))
    family_cap = max(1, int(family_cap))
    motif_cap = max(1, int(motif_cap))
    relaxed_fraction = min(1.0, max(0.0, float(relaxed_fraction)))
    ordered = sorted(
        records,
        key=lambda d: (-_incubator_record_score(d), float(d.get("err", 1e18)), str(d.get("structural_key", ""))),
    )

    def _take_pass(relaxed=False):
        fam_counts = Counter()
        motif_counts = Counter()
        chosen = []
        for rec in ordered:
            fam = str(rec.get("family_tag", ""))
            motif = str(rec.get("motif_key", ""))
            if (not relaxed) and fam and fam_counts[fam] >= family_cap:
                continue
            if (not relaxed) and motif and motif_counts[motif] >= motif_cap:
                continue
            chosen.append(rec)
            if fam:
                fam_counts[fam] += 1
            if motif:
                motif_counts[motif] += 1
            if len(chosen) >= cap:
                break
        return chosen

    kept = _take_pass(relaxed=False)
    relaxed_need = max(1, int(round(cap * relaxed_fraction)))
    if len(kept) >= min(cap, relaxed_need):
        return kept[:cap]

    relaxed = _take_pass(relaxed=True)
    return relaxed[:cap]


def _incubate_novelty_candidate(state, team, spec, chk, candidate_meta, score_info, holdout_lite_err=None):
    try:
        from llm_engine import _serialize_team_spec as _sts
    except ImportError:
        try:
            from .llm_engine import _serialize_team_spec as _sts
        except ImportError:
            _sts = None
    if _sts is None:
        # fallback: inline minimal serialization
        def _sts(t, rationale="", source="offline_json"):
            return {"init_dex": str(t.get("init_dex", "")), "update": str(t.get("update", "")), "query": str(t.get("query", "")), "rationale": rationale, "source": source}
    inc = _ensure_novelty_incubator_state(state)
    novelty_stats = _ensure_llm_novelty_stats_in_state(state)
    candidate_meta = dict(candidate_meta or {})
    struct_key = _incubator_structural_key_from_meta(candidate_meta, fallback_key=chk.get("repair_dup_key", chk.get("key")))
    rec = {
        "team_spec": _sts(
            {**team, "family_tag": _team_family_tag(team), "family_parts": _team_family_parts(team)},
            rationale=str(spec.get("rationale", "") or ""),
            source=str(spec.get("source", "") or "offline_json"),
        ),
        "family_tag": str(candidate_meta.get("family_tag", _team_family_tag(team))),
        "architecture_schema": copy.deepcopy(candidate_meta.get("architecture_schema", {})),
        "motif_signature": copy.deepcopy(candidate_meta.get("motif_signature", {})),
        "mechanism_schema": copy.deepcopy(candidate_meta.get("mechanism_schema", {})),
        "schema_hash": str(candidate_meta.get("schema_hash", "")),
        "motif_key": str(candidate_meta.get("motif_key", "")),
        "mechanism_key": str(candidate_meta.get("mechanism_key", "")),
        "mechanism_cluster": str(candidate_meta.get("mechanism_cluster", "")),
        "mechanism_distance_from_cms": float(candidate_meta.get("mechanism_distance_from_cms", 0.0)),
        "arch_type": str(candidate_meta.get("arch_type", "regular")),
        "repair_dup_key": copy.deepcopy(candidate_meta.get("repair_dup_key", chk.get("repair_dup_key", chk.get("key", "")))),
        "structural_key": struct_key,
        "fit": float(chk.get("fit", 0.0)),
        "err": float(chk.get("err", 1e18)),
        "case_vec": tuple(float(x) for x in chk.get("case_vec", ()) or ()),
        "holdout_lite_err": None if holdout_lite_err is None else float(holdout_lite_err),
        "novelty_score": float(score_info.get("score", 0.0)),
        "innovation_bonus": float(score_info.get("innovation_bonus", 0.0)),
        "motif_distance_score": float(score_info.get("motif_distance_score", 0.0)),
        "motif_frequency": int(score_info.get("motif_frequency", 0)),
        "family_match_score": float(spec.get("family_match_score", score_info.get("target_match_score", 0.0))),
        "family_distance_score": float(spec.get("family_distance_score", score_info.get("mainstream_distance_score", 0.0))),
        "failure_buckets": list(score_info.get("failure_buckets", [])),
        "novelty_subtype": str(spec.get("novelty_subtype", "stable")),
        "materialization_strength": float(_materialization_strength_from_candidate_meta(candidate_meta)),
        "survival_count": 1,
        "age": 0,
        "source": str(spec.get("source", "offline_json")),
        "rationale": str(spec.get("rationale", "") or ""),
        "channel": str(spec.get("channel", "novelty") or "novelty"),
    }
    kept = []
    updated = False
    for old in list(inc.get("records", [])):
        if str(old.get("structural_key", "")) == struct_key:
            merged = dict(old)
            merged["age"] = 0
            merged["survival_count"] = int(old.get("survival_count", 1)) + 1
            if float(rec.get("novelty_score", -1e18)) >= float(old.get("novelty_score", -1e18)) or float(rec.get("err", 1e18)) <= float(old.get("err", 1e18)):
                for k, v in rec.items():
                    if k not in {"survival_count", "age"}:
                        merged[k] = copy.deepcopy(v)
                merged["survival_count"] = max(int(merged.get("survival_count", 1)), int(old.get("survival_count", 1)) + 1)
                merged["age"] = 0
            kept.append(merged)
            updated = True
        else:
            kept.append(old)
    if not updated:
        kept.append(rec)
    cap = max(4, int(state.get("structure_keep_quota", 2)) * max(4, int(state.get("llm_novelty_incubator_capacity", 8))))
    family_cap = max(1, int(state.get("llm_novelty_incubator_family_cap", 1)))
    motif_cap = max(1, int(state.get("llm_novelty_incubator_motif_cap", 2)))
    kept = _trim_novelty_incubator_records(
        kept,
        cap=cap,
        family_cap=family_cap,
        motif_cap=motif_cap,
        relaxed_fraction=0.6,
    )
    inc["records"] = kept
    inc["version"] = int(inc.get("version", 0)) + 1
    inc["last_incubated"] = [struct_key]
    novelty_stats["incubated"] = int(novelty_stats.get("incubated", 0)) + 1
    return rec


def _promote_from_novelty_incubator(state, cfg, gp_ctx, evaluator, island_best_fit, island_best_err, success_budget=1):
    try:
        from llm_engine import _deserialize_team_spec as _dts
    except ImportError:
        try:
            from .llm_engine import _deserialize_team_spec as _dts
        except ImportError:
            _dts = None
    inc = _ensure_novelty_incubator_state(state)
    novelty_stats = _ensure_llm_novelty_stats_in_state(state)
    records = list(inc.get("records", []))
    if success_budget <= 0 or not records:
        return state, 0, []
    repl_idx = _rank_replacement_targets(state)
    if not repl_idx:
        return state, 0, []

    promote_candidates = []
    keep_records = []
    accepted = []
    min_survival_default = int(cfg.get("llm_novelty_incubator_min_survival", 2))
    min_survival_irregular = int(cfg.get("llm_novelty_incubator_min_survival_irregular", 1))
    min_promote_default = float(cfg.get("llm_novelty_incubator_min_promote_score", 14.0))
    min_promote_irregular = float(cfg.get("llm_novelty_incubator_min_promote_score_irregular", 11.0))
    max_age = int(cfg.get("llm_novelty_incubator_max_age", 4))
    family_hist = _family_histogram_from_state(state)
    promote_exact_cap = max(1, int(cfg.get("llm_novelty_promote_exact_cap", 2)))
    promote_round_family_cap = max(1, int(cfg.get("llm_novelty_promote_round_family_cap", 1)))
    promote_round_motif_cap = max(1, int(cfg.get("llm_novelty_promote_round_motif_cap", 1)))
    promote_round_mech_cluster_cap = max(1, int(cfg.get("llm_novelty_promote_round_mechanism_cluster_cap", 1)))
    promote_round_mech_family_cap = max(1, int(cfg.get("llm_novelty_promote_round_mechanism_family_cap", 1)))
    recent_promoted = _recent_promoted_cooldown_snapshot(state)
    recent_family_cap = max(1, int(cfg.get("llm_novelty_recent_family_cap", 2)))
    recent_motif_cap = max(1, int(cfg.get("llm_novelty_recent_motif_cap", 2)))
    recent_schema_cap = max(1, int(cfg.get("llm_novelty_recent_schema_cap", 2)))
    recent_mech_cluster_cap = max(1, int(cfg.get("llm_novelty_recent_mechanism_cluster_cap", 2)))
    recent_mech_family_cap = max(1, int(cfg.get("llm_novelty_recent_mechanism_family_cap", 2)))

    for rec in sorted(records, key=lambda d: (-_incubator_record_score(d), float(d.get("err", 1e18)), str(d.get("structural_key", "")))):
        novelty_subtype = str(rec.get("novelty_subtype", "stable"))
        min_survival = min_survival_irregular if novelty_subtype == "irregular" else min_survival_default
        min_promote = min_promote_irregular if novelty_subtype == "irregular" else min_promote_default
        promo_score = _incubator_record_score(rec)
        qual_ok, qual_reason = _novelty_quality_gate(
            rec.get("fit", 0.0),
            rec.get("err", 1e18),
            island_best_fit,
            island_best_err,
            cfg,
            stage="phase2",
            novelty_subtype=novelty_subtype,
            materialization_strength=rec.get("materialization_strength", 0.0),
        )
        hold_ok = True
        hold_reason = "phase2_holdout_skip"
        holdout_lite_err = rec.get("holdout_lite_err", None)
        if holdout_lite_err is not None:
            hold_ok, hold_reason = _novelty_holdout_gate(holdout_lite_err, rec.get("err", 1e18), island_best_err, cfg)
        if int(rec.get("age", 0)) > max_age and (not qual_ok or not hold_ok):
            continue
        if not qual_ok or not hold_ok:
            keep_records.append(rec)
            continue
        if int(rec.get("survival_count", 1)) < min_survival and float(promo_score) < float(min_promote + 3.0):
            keep_records.append(rec)
            continue
        if float(promo_score) < float(min_promote):
            keep_records.append(rec)
            continue
        family_tag = str(rec.get("family_tag", ""))
        motif_key = str(rec.get("motif_key", ""))
        schema_hash = str(rec.get("schema_hash", ""))
        mechanism_cluster = str(rec.get("mechanism_cluster", ""))
        mechanism_family = str((rec.get("mechanism_schema", {}) or {}).get("mechanism_family", "") or "")
        pop_exact = int(family_hist.get("exact", Counter()).get(family_tag, 0))
        if pop_exact >= promote_exact_cap and novelty_subtype != "irregular" and float(promo_score) < float(min_promote + 10.0):
            keep_records.append(rec)
            continue
        if family_tag and int(recent_promoted.get("family_tags", Counter()).get(family_tag, 0)) >= recent_family_cap and float(promo_score) < float(min_promote + 12.0):
            keep_records.append(rec)
            continue
        if motif_key and int(recent_promoted.get("motif_keys", Counter()).get(motif_key, 0)) >= recent_motif_cap and float(promo_score) < float(min_promote + 12.0):
            keep_records.append(rec)
            continue
        if schema_hash and int(recent_promoted.get("schema_hashes", Counter()).get(schema_hash, 0)) >= recent_schema_cap and float(promo_score) < float(min_promote + 10.0):
            keep_records.append(rec)
            continue
        if mechanism_cluster and int(recent_promoted.get("mechanism_clusters", Counter()).get(mechanism_cluster, 0)) >= recent_mech_cluster_cap and float(promo_score) < float(min_promote + 10.0):
            keep_records.append(rec)
            continue
        if mechanism_family and int(recent_promoted.get("mechanism_families", Counter()).get(mechanism_family, 0)) >= recent_mech_family_cap and float(promo_score) < float(min_promote + 10.0):
            keep_records.append(rec)
            continue
        promote_candidates.append((rec, float(promo_score), str(qual_reason), str(hold_reason)))

    inserted = 0
    used_structural = set()
    used_families = Counter()
    used_motifs = Counter()
    used_mechanism_clusters = Counter()
    used_mechanism_families = Counter()
    for rank, (rec, promo_score, qual_reason, hold_reason) in enumerate(promote_candidates[:max(0, int(success_budget * 3))]):
        if inserted >= int(success_budget):
            keep_records.append(rec)
            continue
        if rank >= len(repl_idx):
            keep_records.append(rec)
            continue
        struct_key = str(rec.get("structural_key", ""))
        family_tag = str(rec.get("family_tag", ""))
        motif_key = str(rec.get("motif_key", ""))
        mechanism_cluster = str(rec.get("mechanism_cluster", ""))
        mechanism_family = str((rec.get("mechanism_schema", {}) or {}).get("mechanism_family", "") or "")
        if struct_key in used_structural:
            keep_records.append(rec)
            continue
        if family_tag and used_families[family_tag] >= promote_round_family_cap:
            keep_records.append(rec)
            continue
        if motif_key and used_motifs[motif_key] >= promote_round_motif_cap:
            keep_records.append(rec)
            continue
        if mechanism_cluster and used_mechanism_clusters[mechanism_cluster] >= promote_round_mech_cluster_cap:
            keep_records.append(rec)
            continue
        if mechanism_family and used_mechanism_families[mechanism_family] >= promote_round_mech_family_cap:
            keep_records.append(rec)
            continue
        try:
            team = _dts(rec.get("team_spec", {}), gp_ctx["pset_map"])
        except Exception:
            keep_records.append(rec)
            continue
        idx = repl_idx[inserted]
        source_meta = {
            "phase": "stagnation",
            "source": str(rec.get("source", "incubator")),
            "rationale": str(rec.get("rationale", "")),
            "family_tag": str(rec.get("family_tag", _team_family_tag(team))),
            "channel": "novelty",
            "novelty_stage": "incubator_promote",
            "novelty_subtype": str(rec.get("novelty_subtype", "stable")),
            "novelty_score": float(rec.get("novelty_score", 0.0)),
            "holdout_lite_err": rec.get("holdout_lite_err", None),
            "family_match": None,
            "family_distance": None,
            "family_match_score": float(rec.get("family_match_score", 0.0)),
            "family_distance_score": float(rec.get("family_distance_score", 0.0)),
            "distance_near_miss_used": False,
            "innovation_bonus": float(rec.get("innovation_bonus", 0.0)),
            "architecture_schema": copy.deepcopy(rec.get("architecture_schema", {})),
            "motif_signature": copy.deepcopy(rec.get("motif_signature", {})),
            "mechanism_schema": copy.deepcopy(rec.get("mechanism_schema", {})),
            "schema_hash": str(rec.get("schema_hash", "")),
            "motif_key": str(rec.get("motif_key", "")),
            "mechanism_key": str(rec.get("mechanism_key", "")),
            "mechanism_cluster": str(rec.get("mechanism_cluster", "")),
            "mechanism_distance_from_cms": float(rec.get("mechanism_distance_from_cms", 0.0)),
            "arch_type": str(rec.get("arch_type", "regular")),
            "motif_distance_score": float(rec.get("motif_distance_score", 0.0)),
            "motif_frequency": int(rec.get("motif_frequency", 0)),
            "promotion_score": float(promo_score),
            "quality_gate_reason": str(qual_reason),
            "holdout_lite_reason": str(hold_reason),
            "survival_count": int(rec.get("survival_count", 1)),
            "materialization_strength": float(rec.get("materialization_strength", 0.0)),
        }
        replace_individual_in_state(
            state,
            idx,
            team,
            (float(rec.get("fit", 0.0)), float(rec.get("err", 1e18)), tuple(float(x) for x in (rec.get("case_vec", ()) or (state["case_vecs"][idx] if idx < len(state.get("case_vecs", [])) else ()) ))),
            source_meta=source_meta,
            fec_key=_safe_triplet_fec_key(evaluator, team["init_dex"], team["update"], team["query"]),
        )
        used_structural.add(struct_key)
        if family_tag:
            used_families[family_tag] += 1
        if motif_key:
            used_motifs[motif_key] += 1
        if mechanism_cluster:
            used_mechanism_clusters[mechanism_cluster] += 1
        if mechanism_family:
            used_mechanism_families[mechanism_family] += 1
        novelty_stats["promoted"] = int(novelty_stats.get("promoted", 0)) + 1
        novelty_stats["injected"] = int(novelty_stats.get("injected", 0)) + 1
        novelty_stats["accepted_families"][str(rec.get("family_tag", _team_family_tag(team)))] += 1
        _record_recent_promoted_structure(state, {
            "family_tag": str(rec.get("family_tag", _team_family_tag(team))),
            "motif_key": str(rec.get("motif_key", "")),
            "schema_hash": str(rec.get("schema_hash", "")),
            "mechanism_cluster": str(rec.get("mechanism_cluster", "")),
            "mechanism_schema": copy.deepcopy(rec.get("mechanism_schema", {})),
        })
        inserted += 1
        accepted.append(rec)

    promoted_keys = {str(rec.get("structural_key", "")) for rec in accepted}
    remaining_records = []
    for rec in keep_records + [x[0] for x in promote_candidates]:
        if str(rec.get("structural_key", "")) not in promoted_keys:
            remaining_records.append(rec)
    cap = max(4, int(state.get("structure_keep_quota", 2)) * max(4, int(state.get("llm_novelty_incubator_capacity", 8))))
    inc["records"] = _trim_novelty_incubator_records(
        remaining_records,
        cap=cap,
        family_cap=max(1, int(state.get("llm_novelty_incubator_family_cap", 2))),
        motif_cap=max(1, int(state.get("llm_novelty_incubator_motif_cap", 3))),
        relaxed_fraction=0.6,
    )
    inc["version"] = int(inc.get("version", 0)) + (1 if inserted > 0 else 0)
    inc["last_promoted"] = [str(rec.get("structural_key", "")) for rec in accepted]
    return state, inserted, accepted


def _schema_histogram_from_state(state):
    axes = {
        "arch_type": Counter(),
        "layout_style": Counter(),
        "handoff_policy": Counter(),
        "state_usage": Counter(),
        "query_fusion": Counter(),
        "primary_innovation_axis": Counter(),
    }
    if not isinstance(state, dict):
        return axes

    def consume_schema(schema):
        if not isinstance(schema, dict):
            return
        for axis in list(axes.keys()):
            val = str(schema.get(axis, "") or "").strip()
            if val:
                axes[axis][val] += 1

    for rec in list(state.get("innovation_archive", []) or []):
        if isinstance(rec, dict):
            consume_schema(rec.get("architecture_schema", {}))
    llm_meta = state.get("llm_meta", [])
    if isinstance(llm_meta, list):
        for rec in llm_meta:
            if isinstance(rec, dict):
                consume_schema(rec.get("architecture_schema", {}))
    pops = state.get("pops", {}) if isinstance(state.get("pops", {}), dict) else {}
    fits = list(state.get("fits", [])) if isinstance(state.get("fits", []), list) else []
    for i in range(len(fits)):
        try:
            team = {"init_dex": pops["init_dex"][i], "update": pops["update"][i], "query": pops["query"][i]}
            consume_schema(asdict(_infer_architecture_schema_from_team(team)))
        except Exception:
            continue
    return axes


def _profile_schema_axis_defaults(profile_name: str):
    prof = str(profile_name or "").strip().lower()
    if prof == "init_explore":
        return {
            "arch_type": ["hybrid", "pyramid", "elastic", "regular"],
            "layout_style": ["layered", "asymmetric_dual_path", "sidecar_heavy", "regular"],
            "handoff_policy": ["layered_correction", "branch_fallback", "none"],
            "state_usage": ["none", "overflow_state", "mixed"],
            "query_fusion": ["median", "branch_conditional", "base_sel", "min"],
            "primary_innovation_axis": ["layout_then_update", "layout_sidecar", "init_dex", "balanced"],
        }
    if prof == "update_explore":
        return {
            "arch_type": ["overflow", "diamond", "hybrid", "regular"],
            "layout_style": ["asymmetric_dual_path", "sidecar_heavy", "regular", "layered"],
            "handoff_policy": ["overflow_to_sidecar", "branch_fallback", "layered_correction", "none"],
            "state_usage": ["overflow_state", "mixed", "none"],
            "query_fusion": ["branch_conditional", "state_gated_min", "median", "base_sel"],
            "primary_innovation_axis": ["update_state_handoff", "update", "architecture", "balanced"],
        }
    if prof == "irregular_architecture":
        return {
            "arch_type": ["overflow", "diamond", "pyramid", "elastic", "hybrid", "regular"],
            "layout_style": ["layered", "sidecar_heavy", "asymmetric_dual_path", "regular"],
            "handoff_policy": ["overflow_to_sidecar", "layered_correction", "branch_fallback", "none"],
            "state_usage": ["overflow_state", "mixed", "branch_state", "none"],
            "query_fusion": ["branch_conditional", "state_gated_min", "median", "base_sel", "min"],
            "primary_innovation_axis": ["architecture", "update_state_handoff", "layout_sidecar", "existing_primitives_only", "balanced"],
        }
    return {
        "arch_type": ["regular", "hybrid", "overflow", "diamond"],
        "layout_style": ["regular", "asymmetric_dual_path", "layered"],
        "handoff_policy": ["none", "branch_fallback", "overflow_to_sidecar"],
        "state_usage": ["none", "overflow_state", "mixed"],
        "query_fusion": ["base_sel", "median", "branch_conditional", "min"],
        "primary_innovation_axis": ["balanced", "update", "init_dex", "architecture"],
    }


def _discover_structure_frontier(state, profile=None, topk=3):
    profile_name = str((profile or {}).get("name", "") or "") if isinstance(profile, dict) else str(profile or "")
    axes_hist = _schema_histogram_from_state(state)
    defaults = _profile_schema_axis_defaults(profile_name)
    frontier_axes = []
    for axis, values in defaults.items():
        ctr = axes_hist.get(axis, Counter())
        ranked = sorted(list(values), key=lambda v: (int(ctr.get(v, 0)), 0 if str(v) != "regular" else 1, values.index(v)))
        frontier_axes.append({
            "axis": str(axis),
            "underexplored": ranked[:max(1, int(topk))],
            "saturated": [str(k) for k, _ in ctr.most_common(max(1, int(topk)))],
        })
    rare_examples = []
    for rec in sorted(list(state.get("innovation_archive", []) or []), key=lambda d: (float(d.get("err", 1e18)), str(d.get("schema_hash", "")))):
        schema = dict(rec.get("architecture_schema", {}) or {})
        if schema and str(schema.get("arch_type", "regular")) != "regular":
            rare_examples.append(schema)
        if len(rare_examples) >= max(1, int(topk)):
            break
    incubator = _ensure_novelty_incubator_state(state)
    incubator_top = []
    for rec in sorted(list(incubator.get("records", [])), key=lambda d: (-_incubator_record_score(d), float(d.get("err", 1e18))))[:max(1, int(topk))]:
        incubator_top.append({
            "family_tag": str(rec.get("family_tag", "")),
            "arch_type": str(rec.get("arch_type", "regular")),
            "score": float(rec.get("novelty_score", 0.0)),
            "survival_count": int(rec.get("survival_count", 1)),
            "age": int(rec.get("age", 0)),
        })
    return {
        "profile": str(profile_name),
        "axes": frontier_axes,
        "rare_examples": rare_examples,
        "incubator_size": int(len(incubator.get("records", []))),
        "incubator_top": incubator_top,
    }


def _mechanism_histogram_from_state(state):
    hist = {
        "cluster": Counter(),
        "family": Counter(),
        "state_contract": Counter(),
        "query_contract": Counter(),
    }
    if not isinstance(state, dict):
        return hist
    def consume_meta(meta):
        if not isinstance(meta, dict):
            return
        mech = dict(meta.get("mechanism_schema", {}) or {})
        if not mech and all(k in meta for k in ("init_dex", "update", "query")):
            try:
                mech = asdict(_infer_mechanism_schema_from_team(meta))
            except Exception:
                mech = {}
        if not mech:
            return
        cluster = str(meta.get("mechanism_cluster", "") or _mechanism_cluster_key(mech))
        fam = str(mech.get("mechanism_family", "cms_like") or "cms_like")
        st = str(mech.get("state_contract", "none") or "none")
        qc = str(mech.get("query_contract", "simple_reduce") or "simple_reduce")
        hist["cluster"][cluster] += 1
        hist["family"][fam] += 1
        hist["state_contract"][st] += 1
        hist["query_contract"][qc] += 1
    for rec in list(state.get("innovation_archive", []) or []):
        if isinstance(rec, dict):
            consume_meta(rec)
    llm_meta = state.get("llm_meta", [])
    if isinstance(llm_meta, list):
        for rec in llm_meta:
            if isinstance(rec, dict):
                consume_meta(rec)
    pops = state.get("pops", {}) if isinstance(state.get("pops", {}), dict) else {}
    fits = list(state.get("fits", [])) if isinstance(state.get("fits", []), list) else []
    for i in range(len(fits)):
        try:
            team = {"init_dex": pops["init_dex"][i], "update": pops["update"][i], "query": pops["query"][i]}
            consume_meta({
                "mechanism_schema": asdict(_infer_mechanism_schema_from_team(team)),
                "mechanism_cluster": _mechanism_cluster_key(asdict(_infer_mechanism_schema_from_team(team))),
            })
        except Exception:
            continue
    return hist


def _discover_mechanism_frontier(state, topk=3):
    hist = _mechanism_histogram_from_state(state)
    default_fams = ["cms_like", "adaptive_count", "scout_rescue", "witness_gated", "layered_promotion", "delegate_fallback", "overflow_delegate"]
    default_state = ["none", "witness_gate", "overflow_witness", "delegate_flag", "sidecar_flag"]
    default_query = ["simple_reduce", "min_reduce", "median_reduce", "avg_reduce", "state_prefers_rescue", "trust_witness", "fallback_if_witness"]
    fam_under = sorted(default_fams, key=lambda x: (int(hist["family"].get(x, 0)), 0 if x != "cms_like" else 1, default_fams.index(x)))
    state_under = sorted(default_state, key=lambda x: (int(hist["state_contract"].get(x, 0)), 0 if x != "none" else 1, default_state.index(x)))
    query_under = sorted(default_query, key=lambda x: (int(hist["query_contract"].get(x, 0)), 0 if x not in {"simple_reduce", "min_reduce", "median_reduce", "avg_reduce"} else 1, default_query.index(x)))
    return {
        "mechanism_family_underexplored": fam_under[:max(1, int(topk))],
        "state_contract_underexplored": state_under[:max(1, int(topk))],
        "query_contract_underexplored": query_under[:max(1, int(topk))],
        "dominant_mechanism_families": [str(k) for k, _ in hist["family"].most_common(max(1, int(topk)))],
        "dominant_mechanism_clusters": [str(k) for k, _ in hist["cluster"].most_common(max(1, int(topk)))],
    }


def _mechanism_frontier_bonus_from_candidate_meta(candidate_meta=None, guidance=None):
    meta = dict(candidate_meta or {})
    mech = dict(meta.get("mechanism_schema", {}) or {})
    frontier = guidance.get("mechanism_frontier", {}) if isinstance(guidance, dict) else {}
    if not isinstance(frontier, dict):
        return 0.0, []
    hits = []
    bonus = 0.0
    fam = str(mech.get("mechanism_family", "") or "")
    if fam and fam in set(str(x) for x in list(frontier.get("mechanism_family_underexplored", []))):
        bonus += 0.8
        hits.append(f"mechanism_family={fam}")
    st = str(mech.get("state_contract", "") or "")
    if st and st in set(str(x) for x in list(frontier.get("state_contract_underexplored", []))):
        bonus += 0.5
        hits.append(f"state_contract={st}")
    qc = str(mech.get("query_contract", "") or "")
    if qc and qc in set(str(x) for x in list(frontier.get("query_contract_underexplored", []))):
        bonus += 0.5
        hits.append(f"query_contract={qc}")
    return float(bonus), hits


def _augment_family_guidance_with_frontier(guidance, state=None, profile=None):
    out = copy.deepcopy(guidance) if isinstance(guidance, dict) else {}
    if isinstance(state, dict):
        frontier = _discover_structure_frontier(state, profile=profile or out.get("profile_name", ""))
        out["schema_frontier"] = frontier
        mechanism_frontier = _discover_mechanism_frontier(state)
        out["mechanism_frontier"] = mechanism_frontier
        proposal_focus = list(out.get("proposal_focus", []))
        for axis_info in list(frontier.get("axes", []))[:3]:
            axis = str(axis_info.get("axis", ""))
            under = list(axis_info.get("underexplored", []))
            if axis and under:
                line = f"Self-discovered frontier / {axis}: prefer underexplored values {', '.join(str(x) for x in under[:3])}."
                if line not in proposal_focus:
                    proposal_focus.append(line)
        if int(frontier.get("incubator_size", 0)) > 0:
            line = f"Innovation incubator currently holds {int(frontier.get('incubator_size', 0))} structure candidates; prefer compatible but non-duplicate schema moves."
            if line not in proposal_focus:
                proposal_focus.append(line)
        mfam = list(out.get("mechanism_frontier", {}).get("mechanism_family_underexplored", [])) if isinstance(out.get("mechanism_frontier", {}), dict) else []
        if mfam:
            line = f"Mechanism frontier: prefer underexplored mechanism families {', '.join(str(x) for x in mfam[:3])}."
            if line not in proposal_focus:
                proposal_focus.append(line)
        out["proposal_focus"] = proposal_focus
    return out


def _duplicate_fail_dominates_failed_records(failed_records, min_hits=2):
    dup = 0
    total = 0
    for rec in list(failed_records or []):
        reasons = [str(x) for x in list(rec.get("reasons", []))]
        if not reasons:
            continue
        total += 1
        if any(r in {"duplicate_canonical_team", "duplicate_structural_team"} for r in reasons):
            dup += 1
    if dup >= max(1, int(min_hits)):
        return True
    if total > 0 and dup >= max(1, int(math.ceil(total * 0.5))):
        return True
    return False



def _materialize_fail_dominates_failed_records(failed_records, token="single_tree_too_large:update", min_hits=2):
    token = str(token or "").strip()
    cnt = 0
    total = 0
    for rec in list(failed_records or []):
        stage = str(rec.get("stage", "") or "")
        reasons = [str(x) for x in list(rec.get("reasons", []))]
        if not reasons:
            continue
        if "materialize" not in stage:
            continue
        total += 1
        if any(token in rr for rr in reasons):
            cnt += 1
    if cnt >= max(1, int(min_hits)):
        return True
    if total > 0 and cnt >= max(1, int(math.ceil(total * 0.5))):
        return True
    return False
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
            mechanism = asdict(_infer_mechanism_schema_from_team(team, architecture_schema=asdict(schema), motif_signature=asdict(sig)))
            meta = {
                "family_tag": _team_family_tag(team),
                "arch_type": str(asdict(schema).get("arch_type", "regular")),
                "schema_hash": _architecture_schema_hash(asdict(schema)),
                "motif_key": _motif_signature_key(asdict(sig)),
                "mechanism_key": _mechanism_schema_hash(mechanism),
                "mechanism_cluster": _mechanism_cluster_key(mechanism),
                "mechanism_distance_from_cms": _mechanism_distance_from_cms(mechanism),
                "architecture_schema": asdict(schema),
                "motif_signature": asdict(sig),
                "mechanism_schema": mechanism,
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
    try:
        from llm_engine import _deserialize_team_spec as _dts
    except ImportError:
        try:
            from .llm_engine import _deserialize_team_spec as _dts
        except ImportError:
            _dts = None
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
            team = _dts(spec, pset_map)
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


def _pick_llm_candidate_spec_sort_key(spec, profile, family_hist=None, pset_map=None, target_parts=None, mechanism_hist=None):
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
    mechanism_hist = mechanism_hist if isinstance(mechanism_hist, dict) else {"cluster": Counter(), "family": Counter()}
    mech_schema = _sanitize_mechanism_schema_claim((spec or {}).get("mechanism_schema", {}))
    if mech_schema:
        mech_cluster = _mechanism_cluster_key(mech_schema)
        mech_dist = _mechanism_distance_from_cms(mech_schema)
        mech_terms = _mechanism_score_terms_from_candidate_meta({
            "mechanism_schema": mech_schema,
            "mechanism_cluster": mech_cluster,
            "mechanism_distance_from_cms": mech_dist,
        })
        mech_family = str(mech_terms.get("mechanism_family", "cms_like") or "cms_like")
    else:
        mech_cluster = ""
        mech_dist = 0.0
        mech_family = "cms_like"
        mech_terms = {"lane_role_entropy": 0.0}
    novelty_subtype = _llm_candidate_subtype(spec)
    if channel == "novelty":
        subtype_rank = 0 if novelty_subtype == "irregular" else 1
        cluster_count = int(mechanism_hist.get("cluster", Counter()).get(str(mech_cluster), 0)) if mech_cluster else 0
        family_count = int(mechanism_hist.get("family", Counter()).get(str(mech_family), 0)) if mech_family else 0
        return (
            cluster_count,
            family_count,
            -float(mech_dist),
            -float(mech_terms.get("lane_role_entropy", 0.0)),
            subtype_rank,
        ) + fam_key + (err, -fit, str((spec or {}).get("source", "")), str(mech_cluster))
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


def _mechanism_saturation_histogram(state):
    return _mechanism_histogram_from_state(state)



def _mechanism_gate(candidate_meta=None, guidance=None, cfg=None, novelty_subtype="stable"):
    meta = dict(candidate_meta or {})
    mech_terms = _mechanism_score_terms_from_candidate_meta(meta)
    mech = dict(meta.get("mechanism_schema", {}) or {})
    mech_dist = float(mech_terms.get("mechanism_distance_from_cms", 0.0))
    frontier_bonus, frontier_hits = _mechanism_frontier_bonus_from_candidate_meta(candidate_meta=meta, guidance=guidance)
    cfg = cfg if isinstance(cfg, dict) else {}
    novelty_subtype = str(novelty_subtype or "stable")
    min_dist = float(cfg.get("llm_mechanism_min_distance_irregular", 1.8) if novelty_subtype == "irregular" else cfg.get("llm_mechanism_min_distance", 2.8))
    strong_override = float(cfg.get("llm_mechanism_family_override_distance", 4.4))
    reasons = []
    cms_like_penalty = 0.0
    mech_family = str(mech_terms.get("mechanism_family", "cms_like"))
    if mech_family == "cms_like":
        cms_like_penalty += 4.0
    if int(mech_terms.get("replication_budget", 3)) >= 3:
        cms_like_penalty += 1.2
    if float(mech_terms.get("lane_role_entropy", 0.0)) <= 0.05:
        cms_like_penalty += 1.0
    effective_mech_dist = float(mech_dist + frontier_bonus + 0.45 * float(mech_terms.get("lane_role_entropy", 0.0)))
    if effective_mech_dist < float(min_dist):
        reasons.append(f"mechanism_distance_too_low:{float(mech_dist):.2f}<{float(min_dist):.2f}")
    info = {
        "mechanism_distance_from_cms": float(mech_dist),
        "effective_mechanism_distance": float(effective_mech_dist),
        "mechanism_family": mech_family,
        "mechanism_cluster": str(mech_terms.get("mechanism_cluster", "")),
        "state_contract": str(mech_terms.get("state_contract", "none")),
        "query_contract": str(mech_terms.get("query_contract", "simple_reduce")),
        "frontier_bonus": float(frontier_bonus),
        "frontier_hits": list(frontier_hits),
        "cms_like_penalty": float(cms_like_penalty),
        "family_override": bool(float(effective_mech_dist) >= float(strong_override)),
    }
    return (len(reasons) == 0), info, reasons


def _mechanism_saturation_gate(candidate_meta=None, state=None, cfg=None, novelty_subtype="stable"):
    meta = dict(candidate_meta or {})
    hist = _mechanism_saturation_histogram(state if isinstance(state, dict) else {})
    recent = _recent_promoted_cooldown_snapshot(state if isinstance(state, dict) else {})
    mech_terms = _mechanism_score_terms_from_candidate_meta(meta)
    cluster = str(mech_terms.get("mechanism_cluster", ""))
    mech_family = str(mech_terms.get("mechanism_family", "cms_like"))
    dist = float(mech_terms.get("mechanism_distance_from_cms", 0.0))
    entropy = float(mech_terms.get("lane_role_entropy", 0.0))
    cfg = cfg if isinstance(cfg, dict) else {}
    novelty_subtype = str(novelty_subtype or "stable")
    exact_cap = int(cfg.get("llm_mechanism_cluster_exact_cap_irregular", 2) if novelty_subtype == "irregular" else cfg.get("llm_mechanism_cluster_exact_cap", 1))
    family_cap = int(cfg.get("llm_mechanism_family_cap_irregular", 4) if novelty_subtype == "irregular" else cfg.get("llm_mechanism_family_cap", 3))
    recent_cap = int(cfg.get("llm_mechanism_recent_cluster_cap", 1))
    override_dist = float(cfg.get("llm_mechanism_cluster_override_distance", 5.0))
    reasons = []
    effective_override = float(override_dist + 0.6 * entropy)
    if cluster and int(hist.get("cluster", Counter()).get(cluster, 0)) >= exact_cap and float(dist) < float(effective_override):
        reasons.append(f"novelty_mechanism_cluster_saturated:{int(hist.get('cluster', Counter()).get(cluster, 0))}>={exact_cap}")
    if mech_family and int(hist.get("family", Counter()).get(mech_family, 0)) >= family_cap and float(dist) < float(effective_override + 0.4):
        reasons.append(f"novelty_mechanism_family_saturated:{mech_family}:{int(hist.get('family', Counter()).get(mech_family, 0))}>={family_cap}")
    if cluster and int(recent.get("mechanism_clusters", Counter()).get(cluster, 0)) >= recent_cap and float(dist) < float(effective_override + 0.8):
        reasons.append(f"novelty_mechanism_recent_cooldown:{cluster}:recent={int(recent.get('mechanism_clusters', Counter()).get(cluster, 0))}")
    info = {
        "mechanism_cluster": cluster,
        "mechanism_family": mech_family,
        "cluster_count": int(hist.get("cluster", Counter()).get(cluster, 0)),
        "family_count": int(hist.get("family", Counter()).get(mech_family, 0)),
        "recent_cluster_count": int(recent.get("mechanism_clusters", Counter()).get(cluster, 0)),
    }
    return (len(reasons) == 0), info, reasons

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


def _novelty_family_gate(team_or_spec, family_hist, target_parts, pset_map=None, min_match=2, min_distance=1, profile=None, guidance=None, novelty_subtype="stable", candidate_meta=None, motif_distance_score=0.0):
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
    mechanism_distance = float(dict(candidate_meta or {}).get("mechanism_distance_from_cms", 0.0)) if isinstance(candidate_meta, dict) or candidate_meta is not None else 0.0
    if mechanism_distance >= 4.0:
        min_match_score = max(0.5, min_match_score - 1.1)
        min_distance_score = max(0.2, min_distance_score - 0.7)
        near_miss_distance_margin += 0.25
    elif mechanism_distance >= 2.5:
        min_match_score = max(0.5, min_match_score - 0.4)
        min_distance_score = max(0.2, min_distance_score - 0.2)
    target_match_score = float(metrics.get("target_match_score", 0.0))
    mainstream_distance_score = float(metrics.get("mainstream_distance_score", 0.0))
    frontier_bonus, frontier_hits = _frontier_bonus_from_candidate_meta(candidate_meta=candidate_meta, guidance=guidance)
    materialization_strength = _materialization_strength_from_candidate_meta(candidate_meta or {})
    materialization_bonus = 0.0
    if materialization_strength >= 6.0:
        materialization_bonus = 0.70
    elif materialization_strength >= 4.0:
        materialization_bonus = 0.40
    elif materialization_strength >= 2.0:
        materialization_bonus = 0.20
    motif_bonus = min(0.85, max(0.0, 0.18 * float(motif_distance_score)))
    effective_distance_score = float(mainstream_distance_score + frontier_bonus + materialization_bonus + motif_bonus + 0.35 * mechanism_distance)
    effective_match_score = float(target_match_score + 0.35 * frontier_bonus + 0.50 * materialization_bonus + 0.15 * mechanism_distance)
    distance_near_miss_used = False
    match_near_miss_used = False
    if float(target_match_score) <= 0.0:
        reasons.append(f"novelty_family_match_zero:{float(target_match_score):.2f}")
    elif float(target_match_score) < float(min_match_score):
        if (not missing) and float(effective_match_score) >= float(min_match_score):
            match_near_miss_used = True
        else:
            reasons.append(f"novelty_family_match_score_too_low:{float(target_match_score):.2f}<{float(min_match_score):.2f}")
    if float(mainstream_distance_score) < float(min_distance_score):
        if (not missing) and float(target_match_score) > 0.0 and float(effective_distance_score + near_miss_distance_margin) >= float(min_distance_score):
            distance_near_miss_used = True
        else:
            reasons.append(f"novelty_family_distance_score_too_low:{float(mainstream_distance_score):.2f}<{float(min_distance_score):.2f}")
    metrics["distance_near_miss_used"] = bool(distance_near_miss_used)
    metrics["match_near_miss_used"] = bool(match_near_miss_used)
    metrics["near_miss_distance_margin"] = float(near_miss_distance_margin)
    metrics["effective_min_match_score"] = float(min_match_score)
    metrics["effective_min_distance_score"] = float(min_distance_score)
    metrics["frontier_bonus"] = float(frontier_bonus)
    metrics["frontier_hits"] = list(frontier_hits)
    metrics["materialization_bonus"] = float(materialization_bonus)
    metrics["motif_distance_bonus"] = float(motif_bonus)
    metrics["effective_distance_score"] = float(effective_distance_score)
    metrics["effective_match_score"] = float(effective_match_score)
    metrics["materialization_strength"] = float(materialization_strength)
    return (len(reasons) == 0), metrics, reasons


def _novelty_quality_gate(chk_fit, chk_err, island_best_fit, island_best_err, cfg, stage="phase1", novelty_subtype="stable", materialization_strength=0.0):
    chk_fit = float(chk_fit)
    chk_err = float(chk_err)
    island_best_fit = float(island_best_fit)
    island_best_err = max(1.0, float(island_best_err))
    stage = str(stage or "phase1").strip().lower()
    novelty_subtype = str(novelty_subtype or "stable").strip().lower()
    irregular = novelty_subtype == "irregular"
    materialization_strength = max(0.0, float(materialization_strength))

    if stage == "phase2":
        err_mult = max(1.0, float(cfg.get("llm_novelty_phase2_err_mult_irregular", 6.0) if irregular else cfg.get("llm_novelty_phase2_err_mult", 4.5)))
        err_cap = max(1.0, float(cfg.get("llm_novelty_phase2_err_cap_irregular", 6000.0) if irregular else cfg.get("llm_novelty_phase2_err_cap", 4000.0)))
        fit_delta = max(0.0, float(cfg.get("llm_novelty_phase2_fit_delta_irregular", 0.06) if irregular else cfg.get("llm_novelty_phase2_fit_delta", 0.04)))
        err_threshold = min(err_cap, err_mult * island_best_err)
        if chk_err <= err_threshold:
            return True, f"phase2_err_gate:{chk_err:.2f}<={err_threshold:.2f}"
        if chk_fit >= island_best_fit - fit_delta:
            return True, f"phase2_fit_gate:{chk_fit:.6f}>={island_best_fit - fit_delta:.6f}"
        return False, f"novelty_phase2_quality_fail err={chk_err:.2f} thr={err_threshold:.2f} fit={chk_fit:.6f} ref={island_best_fit:.6f}"

    err_mult = max(1.0, float(cfg.get("llm_novelty_phase1_err_mult_irregular", 24.0) if irregular else cfg.get("llm_novelty_phase1_err_mult", 18.0)))
    err_cap = max(1.0, float(cfg.get("llm_novelty_phase1_err_cap_irregular", 120000.0) if irregular else cfg.get("llm_novelty_phase1_err_cap", 80000.0)))
    catastrophic_mult = max(err_mult, float(cfg.get("llm_novelty_phase1_catastrophic_mult_irregular", 220.0) if irregular else cfg.get("llm_novelty_phase1_catastrophic_mult", 160.0)))
    catastrophic_cap = max(err_cap, float(cfg.get("llm_novelty_phase1_catastrophic_cap_irregular", 2.0e7) if irregular else cfg.get("llm_novelty_phase1_catastrophic_cap", 1.0e7)))
    fit_delta = max(0.0, float(cfg.get("llm_novelty_phase1_fit_delta_irregular", 0.22) if irregular else cfg.get("llm_novelty_phase1_fit_delta", 0.18)))

    err_threshold = min(err_cap, err_mult * island_best_err)
    catastrophic_threshold = min(catastrophic_cap, catastrophic_mult * island_best_err)
    if chk_err <= err_threshold:
        return True, f"phase1_err_gate:{chk_err:.2f}<={err_threshold:.2f}"
    if chk_err > catastrophic_threshold:
        return False, f"novelty_phase1_catastrophic_fail err={chk_err:.2f} thr={catastrophic_threshold:.2f}"
    if chk_fit >= island_best_fit - fit_delta:
        return True, f"phase1_fit_gate:{chk_fit:.6f}>={island_best_fit - fit_delta:.6f}"
    if materialization_strength >= float(cfg.get("llm_novelty_phase1_materialization_strength", 3.0)):
        return True, f"phase1_materialized_gate strength={materialization_strength:.2f}"
    return True, f"phase1_keepalive_gate err={chk_err:.2f} cat_thr={catastrophic_threshold:.2f}"


def _empty_llm_novelty_stats():
    return {
        "proposed": 0,
        "validated": 0,
        "pass_family": 0,
        "pass_saturation": 0,
        "pass_quality": 0,
        "pass_holdout": 0,
        "pass_score": 0,
        "incubated": 0,
        "promoted": 0,
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
    for k in ("proposed", "validated", "pass_family", "pass_saturation", "pass_quality", "pass_holdout", "pass_score", "incubated", "promoted", "injected"):
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
        for k in ("proposed", "validated", "pass_family", "pass_saturation", "pass_quality", "pass_holdout", "pass_score", "incubated", "promoted", "injected"):
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


def _novelty_saturation_gate(team_or_spec, family_hist, pset_map=None, exact_cap=1, component_cap=12, min_distance_if_saturated=2, candidate_meta=None, state=None, cfg=None, novelty_subtype="stable", mechanism_frontier_bonus=0.0):
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
    meta = dict(candidate_meta or {})
    if not meta:
        mech = {}
        if isinstance(team_or_spec, dict):
            mech = _sanitize_mechanism_schema_claim(team_or_spec.get("mechanism_schema", {}) or team_or_spec.get("_claimed_mechanism_schema", {}))
        if mech:
            meta = {
                "mechanism_schema": mech,
                "mechanism_cluster": _mechanism_cluster_key(mech),
                "mechanism_distance_from_cms": _mechanism_distance_from_cms(mech),
            }
    mech_terms = _mechanism_score_terms_from_candidate_meta(meta)
    mech_hist = _mechanism_histogram_from_state(state if isinstance(state, dict) else {})
    recent = _recent_promoted_cooldown_snapshot(state if isinstance(state, dict) else {})
    cluster = str(mech_terms.get("mechanism_cluster", "") or "")
    mech_family = str(mech_terms.get("mechanism_family", "cms_like") or "cms_like")
    mech_dist = float(mech_terms.get("mechanism_distance_from_cms", 0.0) or 0.0)
    entropy = float(mech_terms.get("lane_role_entropy", 0.0) or 0.0)
    cluster_count = int(mech_hist.get("cluster", Counter()).get(cluster, 0)) if cluster else 0
    recent_cluster_count = int(recent.get("mechanism_clusters", Counter()).get(cluster, 0)) if cluster else 0
    cfg = cfg if isinstance(cfg, dict) else {}
    novelty_subtype = str(novelty_subtype or "stable")
    override_dist = float(cfg.get("llm_novelty_family_mechanism_escape_distance_irregular", 3.0) if novelty_subtype == "irregular" else cfg.get("llm_novelty_family_mechanism_escape_distance", 3.8))
    override_entropy = float(cfg.get("llm_novelty_family_mechanism_escape_entropy", 0.16))
    effective_override = float(mech_dist) + 0.45 * float(entropy) + 0.85 * float(mechanism_frontier_bonus)
    family_override = bool(cluster and cluster_count == 0 and recent_cluster_count == 0 and effective_override >= max(2.6, override_dist - 0.5))
    if not family_override:
        family_override = bool(cluster and mech_dist >= override_dist and entropy >= override_entropy and cluster_count <= 1 and recent_cluster_count <= 0)
    if not family_override:
        family_override = _mechanism_diversity_override(
            candidate_meta=meta,
            mechanism_hist=mech_hist,
            recent=recent,
            cfg=cfg,
            novelty_subtype=novelty_subtype,
            frontier_bonus=0.0,
            mechanism_frontier_bonus=mechanism_frontier_bonus,
        )
    reasons = []
    if exact_count >= int(exact_cap) and not family_override:
        reasons.append(f"novelty_exact_family_saturated:{exact_count}>={int(exact_cap)}")
    if component_count >= int(component_cap) and int(mainstream_distance) < int(min_distance_if_saturated) and not family_override:
        reasons.append(f"novelty_component_saturation:{component_count}>={int(component_cap)}")
    return (len(reasons) == 0), {
        "tag": str(tag),
        "parts": cand_parts,
        "mainstream_parts": mainstream,
        "mainstream_distance": int(mainstream_distance),
        "exact_count": int(exact_count),
        "component_count": int(component_count),
        "mechanism_cluster": cluster,
        "mechanism_family": mech_family,
        "mechanism_distance_from_cms": float(mech_dist),
        "lane_role_entropy": float(entropy),
        "mechanism_cluster_count": int(cluster_count),
        "recent_mechanism_cluster_count": int(recent_cluster_count),
        "family_override": bool(family_override),
        "mechanism_frontier_bonus": float(mechanism_frontier_bonus),
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




def _init_individual_from_ctx(ctx, which, p_skeleton=0.70, p_seed=0.20, p_llm_seed=0.0, allowed_family_labels=None, return_source=False):
    p_skeleton, p_seed, p_llm_seed = _normalize_init_probs(
        p_skeleton=p_skeleton,
        p_seed=p_seed,
        p_llm_seed=p_llm_seed,
    )

    r = random.random()
    source = "random"
    out = None
    if r < p_skeleton:
        source = "skeleton"
        cand = _sample_component_from_bank(ctx, which, "skeleton_bank", allowed_family_labels)
        if cand is not None:
            out = cand
        else:
            out = _skeleton_individual_from_ctx(ctx, which)
    elif r < p_skeleton + p_seed:
        source = "seed"
        cand = _sample_component_from_bank(ctx, which, "seed_bank", allowed_family_labels)
        if cand is not None:
            out = cand
        else:
            out = _seeded_individual_from_ctx(ctx, which)
    elif r < p_skeleton + p_seed + p_llm_seed:
        source = "llm_seed"
        cand = _sample_component_from_bank(ctx, which, "llm_seed_bank", allowed_family_labels)
        if cand is not None:
            out = cand
        else:
            out = _llm_seeded_individual_from_ctx(ctx, which)
    else:
        out = ctx["toolboxes"][which].individual()
    if return_source:
        return out, source
    return out








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

# _safe_triplet_fec_key is defined here so helpers can use it without importing evolution.py
def _safe_triplet_fec_key(evaluator, init_dex_tree, update_tree, query_tree):
    try:
        return evaluator._canonical_triplet_key(init_dex_tree, update_tree, query_tree)
    except Exception:
        return "RAW::" + repr((str(init_dex_tree), str(update_tree), str(query_tree)))


# replace_individual_in_state is defined here so that helpers functions like
# _promote_from_novelty_incubator can call it without importing evolution.py.
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


# _replacement_score and _rank_replacement_targets are defined here so that
# helpers functions like _promote_from_novelty_incubator can call them without
# creating a circular import with evolution.py.
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


# expose private helpers needed by sibling modules via import *
__all__ = [name for name in dir() if not name.startswith('__')]
