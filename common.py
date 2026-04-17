"""Shared imports, primitive registry, and DEAP bootstrap for the modularized CMS sketch search code."""

import sys
import copy
from dataclasses import dataclass, asdict, field
from collections import Counter

sys.path.append("/home/xgr/zhangzy")

from deap import gp, creator, base, tools
import random
import ast
import math
import statistics
import subprocess
import os
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _import_primitive_modules():
    """Import primitive-set modules in both package and flat-file layouts."""
    try:
        from . import init_dex_language as init_dex_language_module
        from . import update_language as update_language_module
        from . import query_language as query_language_module
    except ImportError:
        import init_dex_language as init_dex_language_module
        import update_language as update_language_module
        import query_language as query_language_module
    return init_dex_language_module, update_language_module, query_language_module


_init_dex_language_module, _update_language_module, _query_language_module = _import_primitive_modules()

init_dex_pset = _init_dex_language_module.pset
pset_update = _update_language_module.pset_update
query_pset = _query_language_module.pset

PRIMITIVE_MODULES = {
    "init_dex": _init_dex_language_module,
    "update": _update_language_module,
    "query": _query_language_module,
}
PRIMITIVE_PSETS = {
    "init_dex": init_dex_pset,
    "update": pset_update,
    "query": query_pset,
}
PRIMITIVE_FILENAMES = {
    "init_dex": "init_dex_language.py",
    "update": "update_language.py",
    "query": "query_language.py",
}


def _default_primitive_reference_paths(base_dir: str = ""):
    """Return best-effort primitive reference files for LLM/runtime validation.

    Search order:
    1. explicit base_dir if given
    2. the current file's directory
    3. the imported module's own __file__ path
    4. current working directory
    """
    roots = []
    if base_dir:
        roots.append(os.path.abspath(str(base_dir)))
    roots.append(_THIS_DIR)
    for mod in PRIMITIVE_MODULES.values():
        mod_path = os.path.abspath(getattr(mod, "__file__", "") or "")
        if mod_path:
            roots.append(os.path.dirname(mod_path))
    roots.append(os.getcwd())

    paths = {}
    for key, filename in PRIMITIVE_FILENAMES.items():
        chosen = ""
        for root in roots:
            cand = os.path.join(root, filename)
            if os.path.exists(cand):
                chosen = os.path.abspath(cand)
                break
        if not chosen:
            mod_file = getattr(PRIMITIVE_MODULES[key], "__file__", "") or ""
            chosen = os.path.abspath(mod_file) if mod_file else os.path.join(_THIS_DIR, filename)
        paths[key] = chosen
    return paths


PRIMITIVE_REFERENCE_PATHS = _default_primitive_reference_paths()


def _primitive_constant_snapshot():
    fields = (
        "rows_per_matrix",
        "cols_per_matrix",
        "planes",
        "COUNTER_BITS",
        "MAX_COUNTER",
        "INF",
    )
    snap = {}
    for key, mod in PRIMITIVE_MODULES.items():
        snap[key] = {name: getattr(mod, name, None) for name in fields}
    return snap


PRIMITIVE_CONSTANT_SNAPSHOT = _primitive_constant_snapshot()


def _init_layout_protocol_snapshot():
    sample = _init_dex_language_module.list_3(11, 21, 102, 31, 41, 102, 51, 61, 102)
    tuple_arity = [len(v) if isinstance(v, (tuple, list)) else None for v in sample]
    lane_planes = [int(v[2]) if isinstance(v, (tuple, list)) and len(v) > 2 else None for v in sample]
    return {
        "root": "list_3",
        "tuple_arity": tuple_arity,
        "lane_planes": lane_planes,
        "uses_triplet_protocol": (tuple_arity == [3, 3, 3] and lane_planes == [0, 1, 2]),
    }


INIT_LAYOUT_PROTOCOL = _init_layout_protocol_snapshot()


# Shared constants prefer the init/query modules as the source of truth.
ROWS_PER_MATRIX = int(getattr(_init_dex_language_module, "rows_per_matrix", 102) or 102)
COLS_PER_MATRIX = int(getattr(_init_dex_language_module, "cols_per_matrix", 102) or 102)
PLANES = int(getattr(_init_dex_language_module, "planes", 3) or 3)
COUNTER_BITS = int(getattr(_init_dex_language_module, "COUNTER_BITS", 32) or 32)
MAX_COUNTER = int(getattr(_query_language_module, "MAX_COUNTER", (1 << COUNTER_BITS) - 1) or ((1 << COUNTER_BITS) - 1))
INF = int(getattr(_query_language_module, "INF", MAX_COUNTER + 1) or (MAX_COUNTER + 1))


def _primitive_consistency_report():
    snapshot = _primitive_constant_snapshot()
    baseline = snapshot.get("init_dex", {})
    mismatches = {}
    for name in ("rows_per_matrix", "cols_per_matrix", "planes", "COUNTER_BITS", "MAX_COUNTER"):
        values = {key: snap.get(name) for key, snap in snapshot.items()}
        uniq = {repr(v) for v in values.values()}
        if len(uniq) > 1:
            mismatches[name] = values
    return {
        "ok": len(mismatches) == 0,
        "baseline": baseline,
        "snapshot": snapshot,
        "mismatches": mismatches,
        "reference_paths": dict(PRIMITIVE_REFERENCE_PATHS),
        "init_layout_protocol": dict(INIT_LAYOUT_PROTOCOL),
    }


PRIMITIVE_CONSISTENCY_REPORT = _primitive_consistency_report()


def _apply_proxy_mode_to_stream_path(path: str, dataset_mode: str, proxy_mode: str) -> str:
    """给 stage1_fixed_stream 派生按 proxy_mode 区分的固定流文件名。"""
    path = str(path or "")
    if not path:
        return ""
    if str(dataset_mode) != "proxy":
        return path
    root, ext = os.path.splitext(path)
    if not ext:
        ext = ".npy"
    return f"{root}_{proxy_mode}{ext}"


def set_seed(seed: int):
    """Set Python RNG seed for reproducibility (this project uses `random` only)."""
    random.seed(int(seed) & 0xFFFFFFFF)


if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

if "Individual" not in creator.__dict__:
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    INDIVIDUAL_CLS = creator.Individual
else:
    try:
        if getattr(creator.Individual, "fitness", None) is not None and tuple(creator.Individual.fitness.weights) == (1.0,):
            INDIVIDUAL_CLS = creator.Individual
        else:
            if "IndividualMax" not in creator.__dict__:
                creator.create("IndividualMax", gp.PrimitiveTree, fitness=creator.FitnessMax)
            INDIVIDUAL_CLS = creator.IndividualMax
    except Exception:
        if "IndividualMax" not in creator.__dict__:
            creator.create("IndividualMax", gp.PrimitiveTree, fitness=creator.FitnessMax)
        INDIVIDUAL_CLS = creator.IndividualMax

# expose private helpers needed by cli.py via import *
__all__ = [name for name in dir() if not name.startswith('__')]
