# -*- coding: utf-8 -*-
"""
rm_oneclick_protocol_run3.py
============================================================
把“三模型训练/复用 + 统一评估&可视化”串起来的一键脚本。

三模型：
  - FNO : rm_train_fno_protocol_allinone.py
  - UNet: rm_train_unet_protocol_allinone.py
  - GCGO: rm_train_gcgo_protocol_allinone.py

新增：
  - 自动生成 splits（若 splits/*.txt 不存在）
  - dataset_root 解析更鲁棒：优先按 rm_total_config.py 所在目录解析相对路径，
    不再强行用 CWD，避免你从别的目录运行导致 “buildings=0/targets=0” 的问题。
"""

from __future__ import annotations

import os
import json
import sys
import subprocess
import time
import gc
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Set

import torch

import rm_total_config as rtc
from rm_protocol_viz_compare3 import compare3


# =============================================================================
# CONFIG
# =============================================================================
CFG: Dict[str, Any] = {
    # Per-model action: reuse | train | reuse_if_exists
    "ACTION": {
        "fno": "reuse_if_exists",
        "unet": "reuse_if_exists",
        "gcgo": "reuse_if_exists",
    },

    # Optional: manually set ckpt paths to force reuse (highest priority)
    "CKPT_OVERRIDE": {
        "fno": "",
        "unet": "",
        "gcgo": "",
    },

    # If you want to isolate runs by model:
    # out_dir_base: default TOTAL_CFG["train"]["out_dir"] (e.g. ./rm_runs)
    "OUT_DIR_BASE": None,     # e.g. "./rm_runs_protocol"
    "RUN_NAME_BASE": None,    # e.g. "exp"

    # Compare report output
    "COMPARE_OUT_DIR": "./rm_compare3_runs",
    "COMPARE_RUN_NAME": "compare3_protocol",
    "EVAL_SPLIT": "test",

    # Runtime
    "DEVICE": "cuda",

    # viz CFG passthrough (optional)
    "N_EXAMPLES": 12,
# GCGO highlights (extra visualizations to showcase physical+geometric advantages)
"GCGO_HIGHLIGHTS": {
    "ENABLE": True,      # set False to skip highlights
    "NUM_SAMPLES": 24,   # how many samples to aggregate radial curves
    "TOPK": 3,           # how many "best improvement" samples to render as highlight_*.png
    "N_BINS": 20,        # radial distance bins
    "SEED": 123,
},
}


# ============================== DDP LAUNCH SWITCH ==============================
# Use torchrun by default. Switch to "python" for quick single-process debugging.
RUN_MODE = os.environ.get("RM_RUN_MODE", "torchrun").strip().lower()  # "torchrun" | "python"
NPROC_PER_NODE = int(os.environ.get("RM_NPROC_PER_NODE", "4"))

# Safety guard: do NOT launch this oneclick with torchrun.
# This script orchestrates per-model torchrun subprocesses itself.
if int(os.environ.get("WORLD_SIZE", "1")) > 1 or int(os.environ.get("RANK", "0")) != 0:
    raise RuntimeError(
        "Do NOT run rm_oneclick_protocol_run3.py with torchrun. "
        "Run it with python, and it will spawn torchrun subprocesses for each model. "
        "Example: `python rm_oneclick_protocol_run3.py`."
    )


# =============================================================================
# Helpers
# =============================================================================

def script_dir() -> Path:
    return Path(__file__).resolve().parent

def _as_path(p: str | Path) -> Path:
    pp = Path(str(p)).expanduser()
    # 如果是绝对路径，直接返回；如果是相对路径，基于脚本所在目录解析
    return pp if pp.is_absolute() else (script_dir() / pp)


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _resolve_out_run_base(total_cfg: Dict[str, Any]) -> Tuple[Path, str]:
    train_cfg = total_cfg.get("train", {})
    out_base = CFG.get("OUT_DIR_BASE") or train_cfg.get("out_dir", "./rm_runs")
    run_base = CFG.get("RUN_NAME_BASE") or train_cfg.get("run_name", "exp")
    return _as_path(out_base), str(run_base)


def _looks_like_dataset_root(root: Path, sp: Dict[str, Any]) -> bool:
    """Check whether root contains the expected subdirs (target/buildings/antenna)."""
    target = root / str(sp.get("target_subdir", ""))
    bld = root / str(sp.get("buildings_subdir", ""))
    ant = root / str(sp.get("antenna_subdir", ""))
    return target.exists() and bld.exists() and ant.exists()


def _resolve_dataset_root(total_cfg: Dict[str, Any]) -> Path:
    """
    Resolve dataset_root robustly:
    - If sp["dataset_root"] is absolute: use it.
    - Else: resolve relative to rm_total_config.py directory (NOT CWD),
      because the dataset usually lives alongside the scripts.
    - If that doesn't look valid, try oneclick script dir, then CWD.
    """
    proto = total_cfg.get("protocol", {})
    sp = proto.get("split", {})
    dr_raw = str(sp.get("dataset_root", "."))
    dr_path = _as_path(dr_raw)

    candidates: List[Path] = []

    # 1) absolute path
    if dr_path.is_absolute():
        candidates.append(dr_path.resolve())
    else:
        # resolve relative to rm_total_config.py
        cfg_dir = Path(rtc.__file__).resolve().parent
        candidates.append((cfg_dir / dr_path).resolve())
        # resolve relative to this script
        script_dir = Path(__file__).resolve().parent
        candidates.append((script_dir / dr_path).resolve())
        # finally, CWD
        candidates.append((Path.cwd().resolve() / dr_path).resolve())

    # de-dup
    uniq: List[Path] = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)

    for c in uniq:
        if _looks_like_dataset_root(c, sp):
            return c

    # If nothing matches strictly, still return first candidate but provide hint via exception from split maker
    return uniq[0] if uniq else Path.cwd().resolve()


def _patch_dataset_root_abs(total_cfg: Dict[str, Any]) -> Path:
    proto = total_cfg.get("protocol", {})
    sp = proto.get("split", {})
    root = _resolve_dataset_root(total_cfg)
    sp["dataset_root"] = str(root)
    proto["split"] = sp
    total_cfg["protocol"] = proto
    return root


def _find_latest_dir(prefix_parent: Path, prefix: str) -> Optional[Path]:
    if not prefix_parent.is_dir():
        return None
    cands = []
    for p in prefix_parent.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _latest_ckpt_fno_or_unet(out_dir_model: Path, run_name_base: str) -> Optional[Path]:
    # training scripts create: out_dir / f"{run_name}_{ts}" / ckpts / best.pt
    latest = _find_latest_dir(out_dir_model, prefix=f"{run_name_base}_")
    if latest is None:
        return None
    ckpt = latest / "ckpts" / "best.pt"
    return ckpt if ckpt.is_file() else None


def _latest_ckpt_gcgo(out_dir_model: Path, run_name_base: str) -> Optional[Path]:
    # gcgo script creates: out_dir / run_name / ckpt_best.pt
    # we allow run_name to have timestamp suffix
    latest = _find_latest_dir(out_dir_model, prefix=f"{run_name_base}_")
    if latest is None:
        # maybe exact
        exact = out_dir_model / run_name_base
        ckpt = exact / "ckpt_best.pt"
        return ckpt if ckpt.is_file() else None
    ckpt = latest / "ckpt_best.pt"
    return ckpt if ckpt.is_file() else None


def _dump_total_cfg_json(patched_total_cfg: Dict[str, Any], tag: str, tmp_dir: Path) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / f"total_cfg_{tag}.json"
    p.write_text(json.dumps(patched_total_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def _run_train_subprocess(script_name: str, total_cfg_json: Path) -> None:
    script_path = (Path(__file__).resolve().parent / script_name).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Train script not found: {script_path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_path.parent) + os.pathsep + env.get("PYTHONPATH", "")

    if RUN_MODE == "python":
        cmd = [sys.executable, str(script_path), "--total_cfg_json", str(total_cfg_json)]
    elif RUN_MODE == "torchrun":
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={NPROC_PER_NODE}",
            str(script_path),
            "--total_cfg_json",
            str(total_cfg_json),
        ]
    else:
        raise ValueError(f"RUN_MODE must be 'torchrun' or 'python', got: {RUN_MODE!r}")

    print("[Launch]", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def _train_fno(patched_total_cfg: Dict[str, Any], tmp_cfg_dir: Path) -> Path:
    total_cfg_json = _dump_total_cfg_json(patched_total_cfg, tag="fno", tmp_dir=tmp_cfg_dir)
    _run_train_subprocess("rm_train_fno_protocol_allinone.py", total_cfg_json)

    out_dir = _as_path(patched_total_cfg["train"]["out_dir"])
    run_name = str(patched_total_cfg["train"]["run_name"])
    ckpt = _latest_ckpt_fno_or_unet(out_dir, run_name)
    if ckpt is None:
        raise RuntimeError("FNO training finished but best.pt not found.")
    return ckpt


def _train_unet(patched_total_cfg: Dict[str, Any], tmp_cfg_dir: Path) -> Path:
    total_cfg_json = _dump_total_cfg_json(patched_total_cfg, tag="unet", tmp_dir=tmp_cfg_dir)
    _run_train_subprocess("rm_train_unet_protocol_allinone.py", total_cfg_json)

    out_dir = _as_path(patched_total_cfg["train"]["out_dir"])
    run_name = str(patched_total_cfg["train"]["run_name"])
    ckpt = _latest_ckpt_fno_or_unet(out_dir, run_name)
    if ckpt is None:
        raise RuntimeError("UNet training finished but best.pt not found.")
    return ckpt


def _train_gcgo(patched_total_cfg: Dict[str, Any], tmp_cfg_dir: Path) -> Path:
    total_cfg_json = _dump_total_cfg_json(patched_total_cfg, tag="gcgo", tmp_dir=tmp_cfg_dir)
    _run_train_subprocess("rm_train_gcgo_protocol_allinone.py", total_cfg_json)

    out_dir = _as_path(patched_total_cfg["train"]["out_dir"])
    run_name = str(patched_total_cfg["train"]["run_name"])
    ckpt = _latest_ckpt_gcgo(out_dir, run_name)
    if ckpt is None:
        raise RuntimeError("GCGO training finished but ckpt_best.pt not found.")
    return ckpt


def _gc_collect_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Splits: auto generation (map-level)
# =============================================================================

def _stem_map_from_target(stem: str) -> Optional[str]:
    # expected: mapId_txId  (map may contain underscores; tx is last token)
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    return "_".join(parts[:-1])


def _discover_maps_buildings(dataset_root: Path, buildings_subdir: str) -> Set[str]:
    d = dataset_root / buildings_subdir
    if not d.is_dir():
        return set()
    return {p.stem for p in d.glob("*.png")}


def _discover_maps_antenna(dataset_root: Path, antenna_subdir: str) -> Set[str]:
    d = dataset_root / antenna_subdir
    if not d.is_dir():
        return set()
    return {p.stem for p in d.glob("*.json")}


def _discover_maps_targets(dataset_root: Path, target_subdir: str) -> Set[str]:
    d = dataset_root / target_subdir
    if not d.is_dir():
        return set()
    maps: Set[str] = set()
    for p in list(d.glob("*.png")) + list(d.glob("*.npy")):
        m = _stem_map_from_target(p.stem)
        if m:
            maps.add(m)
    return maps


def _write_list(path: Path, items: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(str(it) + "\n")


def _make_splits(dataset_root: Path, sp: Dict[str, Any]) -> None:
    buildings_subdir = str(sp.get("buildings_subdir", "png/buildings_complete"))
    antenna_subdir = str(sp.get("antenna_subdir", "antenna"))
    target_subdir = str(sp.get("target_subdir", "gain/DPM"))

    maps_b = _discover_maps_buildings(dataset_root, buildings_subdir)
    maps_a = _discover_maps_antenna(dataset_root, antenna_subdir)
    maps_t = _discover_maps_targets(dataset_root, target_subdir)

    inter = maps_b & maps_a & maps_t
    if not inter:
        print("[Splits] WARNING: could not intersect buildings/antenna/targets; fallback to any non-empty source.")
        print(f"         buildings={len(maps_b)}, antenna={len(maps_a)}, targets={len(maps_t)}")
        # fallback priority: buildings -> targets -> antenna
        if maps_b:
            inter = maps_b
        elif maps_t:
            inter = maps_t
        elif maps_a:
            inter = maps_a
        else:
            raise RuntimeError(
                f"[Splits] no maps discovered. dataset_root={dataset_root}\n"
                f"  Checked:\n"
                f"    - {dataset_root/buildings_subdir}\n"
                f"    - {dataset_root/antenna_subdir}\n"
                f"    - {dataset_root/target_subdir}\n"
                f"  Hint: set protocol.split.dataset_root to the dataset folder, or run from that folder."
            )

    maps = sorted(list(inter))

    max_maps = sp.get("max_maps", None)
    if max_maps is not None:
        try:
            max_maps = int(max_maps)
        except Exception:
            max_maps = None
    map_select = str(sp.get("map_select", "first")).lower()
    seed = int(sp.get("split_seed", sp.get("seed", 123)))
    rng = random.Random(seed)

    if max_maps is not None and len(maps) > max_maps:
        if map_select == "random":
            rng.shuffle(maps)
            maps = maps[:max_maps]
            maps.sort()
        else:
            maps = maps[:max_maps]

    val_ratio = float(sp.get("val_map_ratio", 0.15))
    test_ratio = float(sp.get("test_map_ratio", 0.0))
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError(f"[Splits] invalid ratios: val={val_ratio}, test={test_ratio}")

    n = len(maps)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    # ensure at least 1 train if possible
    if n - n_test - n_val <= 0 and n >= 3:
        n_val = max(1, min(n_val, n-2))
        n_test = max(1, min(n_test, n-1-n_val))

    idx = list(range(n))
    rng.shuffle(idx)
    test_idx = set(idx[:n_test])
    val_idx = set(idx[n_test:n_test+n_val])

    test_maps = [maps[i] for i in range(n) if i in test_idx]
    val_maps = [maps[i] for i in range(n) if i in val_idx]
    train_maps = [maps[i] for i in range(n) if i not in test_idx and i not in val_idx]

    # write
    train_path = dataset_root / str(sp.get("train_list_path", "splits/train_maps.txt"))
    val_path = dataset_root / str(sp.get("val_list_path", "splits/val_maps.txt"))
    test_path = dataset_root / str(sp.get("test_list_path", "splits/test_maps.txt"))

    _write_list(train_path, train_maps)
    _write_list(val_path, val_maps)
    _write_list(test_path, test_maps)

    meta = {
        "dataset_root": str(dataset_root),
        "seed": seed,
        "val_map_ratio": val_ratio,
        "test_map_ratio": test_ratio,
        "max_maps": max_maps,
        "map_select": map_select,
        "counts": {"train": len(train_maps), "val": len(val_maps), "test": len(test_maps), "all": n},
        "sources": {"buildings": len(maps_b), "antenna": len(maps_a), "targets": len(maps_t), "used": len(inter)},
        "ts": _now_ts(),
    }
    meta_path = train_path.parent / "split_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[Splits] generated:")
    print(f"  train={len(train_maps)}, val={len(val_maps)}, test={len(test_maps)}")
    print(f"  paths: {train_path} | {val_path} | {test_path}")


def _ensure_splits(total_cfg: Dict[str, Any]) -> None:
    proto = total_cfg.get("protocol", {})
    sp = proto.get("split", {})

    if not bool(sp.get("auto_make_splits", True)):
        return

    dataset_root = Path(str(sp.get("dataset_root", "."))).resolve()
    train_path = dataset_root / str(sp.get("train_list_path", "splits/train_maps.txt"))
    val_path = dataset_root / str(sp.get("val_list_path", "splits/val_maps.txt"))
    test_path = dataset_root / str(sp.get("test_list_path", "splits/test_maps.txt"))

    force = bool(sp.get("force_remake_splits", False))
    if (train_path.is_file() and val_path.is_file() and test_path.is_file()) and not force:
        print("[Splits] found existing splits, skip.")
        return

    _make_splits(dataset_root, sp)


# =============================================================================
# Main
# =============================================================================

def main():
    base_total = deepcopy(rtc.TOTAL_CFG)
    tmp_cfg_dir = (Path(__file__).resolve().parent / "tmp_cfgs").resolve()


    # Resolve dataset root robustly & auto generate splits if needed
    dataset_root = _patch_dataset_root_abs(base_total)
    _ensure_splits(base_total)

    out_base, run_base = _resolve_out_run_base(base_total)

    # separate per-model out dirs (recommended)
    out_fno = out_base / "fno"
    out_unet = out_base / "unet"
    out_gcgo = out_base / "gcgo"

    # baseline run_name prefixes
    rn_fno = f"{run_base}_fno"
    rn_unet = f"{run_base}_unet"
    rn_gcgo = f"{run_base}_gcgo"

    actions = CFG.get("ACTION", {})
    override = CFG.get("CKPT_OVERRIDE", {})

    ckpts: Dict[str, Path] = {}

    # ---------------- FNO ----------------
    if str(override.get("fno", "")).strip():
        ckpts["fno"] = _as_path(override["fno"])
    else:
        act = str(actions.get("fno", "reuse_if_exists")).lower().strip()
        ck = _latest_ckpt_fno_or_unet(out_fno, rn_fno)
        if act == "reuse":
            if ck is None:
                raise FileNotFoundError(f"FNO reuse requested but no checkpoint found under: {out_fno}")
            ckpts["fno"] = ck
        elif act == "train" or (act == "reuse_if_exists" and ck is None):
            total = deepcopy(base_total)
            total["train"]["out_dir"] = str(out_fno)
            total["train"]["run_name"] = rn_fno
            total["train"]["device"] = str(CFG.get("DEVICE", total["train"].get("device", "cuda")))
            # Ensure explicit model_fno exists (avoid relying on implicit defaults)
            if "model_fno" not in total:
                if "model" in total and isinstance(total.get("model"), dict):
                    total["model_fno"] = deepcopy(total["model"])
                else:
                    import rm_train_fno_protocol_allinone as mod
                    total["model_fno"] = deepcopy(mod.DEFAULT_TOTAL_CFG.get("model", {}))
            # Backward-compat alias
            if total.get("model") is None:
                total["model"] = deepcopy(total["model_fno"])
            ckpts["fno"] = _train_fno(total, tmp_cfg_dir)
        else:
            ckpts["fno"] = ck

    _gc_collect_cuda()

    # ---------------- UNet ----------------
    if str(override.get("unet", "")).strip():
        ckpts["unet"] = _as_path(override["unet"])
    else:
        act = str(actions.get("unet", "reuse_if_exists")).lower().strip()
        ck = _latest_ckpt_fno_or_unet(out_unet, rn_unet)
        if act == "reuse":
            if ck is None:
                raise FileNotFoundError(f"UNet reuse requested but no checkpoint found under: {out_unet}")
            ckpts["unet"] = ck
        elif act == "train" or (act == "reuse_if_exists" and ck is None):
            total = deepcopy(base_total)
            total["train"]["out_dir"] = str(out_unet)
            total["train"]["run_name"] = rn_unet
            total["train"]["device"] = str(CFG.get("DEVICE", total["train"].get("device", "cuda")))
            # Ensure explicit model_fno exists (avoid relying on implicit defaults)
            if "model_fno" not in total:
                if "model" in total and isinstance(total.get("model"), dict):
                    total["model_fno"] = deepcopy(total["model"])
                else:
                    import rm_train_fno_protocol_allinone as mod
                    total["model_fno"] = deepcopy(mod.DEFAULT_TOTAL_CFG.get("model", {}))
            # Backward-compat alias
            if total.get("model") is None:
                total["model"] = deepcopy(total["model_fno"])
            if "model_unet" not in total:
                import rm_train_unet_protocol_allinone as mod
                total["model_unet"] = deepcopy(mod.DEFAULT_TOTAL_CFG.get("model_unet", {}))
            ckpts["unet"] = _train_unet(total, tmp_cfg_dir)
        else:
            ckpts["unet"] = ck

    _gc_collect_cuda()

    # ---------------- GCGO ----------------
    if str(override.get("gcgo", "")).strip():
        ckpts["gcgo"] = _as_path(override["gcgo"])
    else:
        act = str(actions.get("gcgo", "reuse_if_exists")).lower().strip()
        ck = _latest_ckpt_gcgo(out_gcgo, rn_gcgo)
        if act == "reuse":
            if ck is None:
                raise FileNotFoundError(f"GCGO reuse requested but no checkpoint found under: {out_gcgo}")
            ckpts["gcgo"] = ck
        elif act == "train" or (act == "reuse_if_exists" and ck is None):
            total = deepcopy(base_total)
            total["train"]["out_dir"] = str(out_gcgo)
            total["train"]["run_name"] = f"{rn_gcgo}_{_now_ts()}"
            total["train"]["device"] = str(CFG.get("DEVICE", total["train"].get("device", "cuda")))
            # Ensure explicit model_fno exists (avoid relying on implicit defaults)
            if "model_fno" not in total:
                if "model" in total and isinstance(total.get("model"), dict):
                    total["model_fno"] = deepcopy(total["model"])
                else:
                    import rm_train_fno_protocol_allinone as mod
                    total["model_fno"] = deepcopy(mod.DEFAULT_TOTAL_CFG.get("model", {}))
            # Backward-compat alias
            if total.get("model") is None:
                total["model"] = deepcopy(total["model_fno"])
            if "model_gcgo" not in total:
                import rm_train_gcgo_protocol_allinone as mod
                total["model_gcgo"] = deepcopy(mod.DEFAULT_TOTAL_CFG.get("model_gcgo", {}))
            ckpts["gcgo"] = _train_gcgo(total, tmp_cfg_dir)
        else:
            ckpts["gcgo"] = ck

    _gc_collect_cuda()

    # ---------------- Compare & Viz ----------------
    split = str(CFG.get("EVAL_SPLIT", "val"))
    device = str(CFG.get("DEVICE", "cuda"))

    # sync some viz config
    import rm_protocol_viz_compare3 as viz
    viz.CFG["N_EXAMPLES"] = int(CFG.get("N_EXAMPLES", viz.CFG.get("N_EXAMPLES", 12)))
    viz.CFG["SPLIT"] = split
    viz.CFG["DEVICE"] = device

    compare_out = _as_path(CFG.get("COMPARE_OUT_DIR", "./rm_compare3_runs"))
    compare_run = str(CFG.get("COMPARE_RUN_NAME", "compare3_protocol"))

    run_dir = compare3(
        ckpt_fno=str(ckpts["fno"]),
        ckpt_unet=str(ckpts["unet"]),
        ckpt_gcgo=str(ckpts["gcgo"]),
        out_dir=str(compare_out),
        run_name=compare_run,
        split=split,
        device=device,
    )
    # ---- GCGO highlights (radial MAE curve + improvement maps + 1D cross-sections) ----
    hcfg = CFG.get("GCGO_HIGHLIGHTS", {}) if isinstance(CFG.get("GCGO_HIGHLIGHTS", {}), dict) else {}
    if bool(hcfg.get("ENABLE", True)):
        try:
            from rm_viz_gcgo_highlights import run_gcgo_highlights
            # Skip safely if ckpts are missing
            _need = ["fno", "unet", "gcgo"]
            _missing = [k for k in _need if (k not in ckpts or not Path(str(ckpts[k])).exists())]
            if _missing:
                print(f"[Highlights] missing ckpt(s) {_missing}, skip.")
            else:
                highlight_out = Path(str(run_dir)) / "gcgo_highlights"
                run_gcgo_highlights(
                    ckpt_fno=str(ckpts["fno"]),
                    ckpt_unet=str(ckpts["unet"]),
                    ckpt_gcgo=str(ckpts["gcgo"]),
                    split=split,
                    out_dir=str(highlight_out),
                    num_samples=int(hcfg.get("NUM_SAMPLES", 24)),
                    topk=int(hcfg.get("TOPK", 3)),
                    n_bins=int(hcfg.get("N_BINS", 20)),
                    seed=int(hcfg.get("SEED", 123)),
                    device=device,
                )
                print(f"[Highlights] saved to: {highlight_out}")
        except Exception as e:
            print(f"[Warn] GCGO highlights failed (skip): {repr(e)}")
    
    
if __name__ == "__main__":
    main()
