# -*- coding: utf-8 -*-
"""
rm_protocol_dataset.py
============================================================
把“数据 Dataset / 通道生成 / 布局(layout)”从训练脚本中彻底剥离出来的【新模块】。

设计目标
- 输入协议（P0+）由 protocol["input"]["layout"] 决定，顺序强约束（可 strict_assert）。
- 三种模型（FNO / UNet / GCGO）吃同一份公开输入 x：
    * FNO/UNet: 直接吃 x (B,C,H,W)
    * GCGO: 从同一份 x 切分角色 (f_src + geom_feat)，不引入额外信息
- 目标 y 的 transform / clip / norm 也由 protocol["target_norm"] 统一管理（同口径评估）。

你只需要在训练脚本里做三件事：
1) proto = TOTAL_CFG["protocol"]
2) ds = RadiomapProtocolDataset(proto, split="train"/"val"/"test")
3) batch = adapter.make_model_inputs(model_type, x, meta)  # 可选

------------------------------------------------------------
【默认适配你的工程目录结构】
dataset_root/
  gain/DPM/                  # target_subdir: 目标场（png 或 npy）
  png/buildings_complete/     # buildings_subdir: 建筑mask（map_id.png）
  antenna/                    # antenna_subdir: 发射机坐标（map_id.json, list of [x,y] or [y,x]）
  splits/*.json               # train/val/test 列表（由你的 rm_make_splits.py 产生）
  stats/*.json                # x/y 统计量（可选；不提供则按 protocol 的 norm 策略降级）

------------------------------------------------------------
【协议字段依赖（核心）】
protocol["input"]:
  layout: ["building","tx_gaussian","tx_invdist","dy","dx"]  # P0+ (你当前的主对比协议)
  construct:
    image_size, tx_gaussian_sigma, tx_invdist_eps, coord_norm
  x_norm: per-channel normalization rule (可选)

protocol["target_norm"]:
  y_space_train / y_space_eval_main（一般都用 fno_target）
  y_transform: "none" | "log1p" | "clip" | "log1p+clip"
  y_clip_range: [lo,hi] or None
  y_stats_path: 用于 global minmax / zscore 的统计量文件（可选）
  x_stats_path: 用于 x 通道 zscore 的统计量文件（可选）

protocol["split"]:
  dataset_root, target_subdir, buildings_subdir, antenna_subdir
  train/val/test list path (json/txt)

============================================================
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy

import json
import re
import math

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


# ============================================================
# 0) Exceptions
# ============================================================

class ProtocolError(RuntimeError):
    """协议字段缺失/不一致/非法时抛出。"""
    pass


# ============================================================
# 1) File utils & parsing
# ============================================================

_FILENAME_PATTERNS = [
    r'^(\d+)_([0-9]+)$',           # 99_0
    r'^map(\d+)_([0-9]+)$',        # map99_0
    r'^(\d+)-tx([0-9]+)$',         # 99-tx0
    r'^(\d+)_tx([0-9]+)$',         # 99_tx0
    r'^(\d+)_([0-9]{1,3})\D*$',    # 99_000something
]

def parse_map_tx_from_stem(stem: str) -> Tuple[str, Optional[int]]:
    """
    从文件名 stem 解析 (map_id, tx_id)。
    返回 tx_id 可能为 None（解析失败时）。
    """
    s = stem.lower()
    for pat in _FILENAME_PATTERNS:
        m = re.match(pat, s)
        if m:
            return m.group(1), int(m.group(2))
    parts = re.split(r'[_\-]', s)
    return parts[0], None


def load_gray(path: Path) -> np.ndarray:
    """load image as float32 gray (H,W)"""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)


def load_field(path: Path, npy_channel: int = 0) -> np.ndarray:
    """
    支持：
      - .png/.jpg: 灰度图
      - .npy: (H,W) 或 (C,H,W) 或 (H,W,C) -> select channel
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
        if arr.ndim == 2:
            out = arr
        elif arr.ndim == 3:
            # (C,H,W)
            if arr.shape[0] <= 16 and arr.shape[1] > 32 and arr.shape[2] > 32:
                c = int(np.clip(npy_channel, 0, arr.shape[0]-1))
                out = arr[c]
            # (H,W,C)
            elif arr.shape[2] <= 16 and arr.shape[0] > 32 and arr.shape[1] > 32:
                c = int(np.clip(npy_channel, 0, arr.shape[2]-1))
                out = arr[..., c]
            else:
                out = arr.squeeze()
        else:
            out = arr.squeeze()
        if out.ndim != 2:
            out = out.squeeze()
        return out.astype(np.float32)

    # image
    return load_gray(path).astype(np.float32)


def safe_int(s: str) -> int:
    m = re.sub(r"\D", "", str(s))
    return int(m) if m else 0


# ============================================================
# 2) Protocol access helpers
# Sentinel for required fields
_NOT_IMPLEMENTED = object()


# ============================================================

def _get(d: Dict[str, Any], path: str, default=_NOT_IMPLEMENTED):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            if default is _NOT_IMPLEMENTED:
                raise ProtocolError(f"Missing protocol field: {path}")
            return default
        cur = cur[k]
    return cur


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def load_json_if_exists(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    p = _as_path(path)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


# ============================================================
# 3) Channel builders (公开通道 P0+)
# ============================================================

def build_tx_gaussian(h: int, w: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    """tx_gaussian: exp(-d^2/(2*sigma^2)) ∈ (0,1]"""
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * float(sigma) ** 2 + 1e-12))
    return g.astype(np.float32)


def build_tx_invdist(h: int, w: int, cy: float, cx: float, eps: float) -> np.ndarray:
    """
    tx_invdist: 1/(d+eps) -> normalize to [0,1] by dividing max (at TX).
    说明：这是真·inverse-distance 的数值版本（和你旧代码里 (1-d_norm)^p 不同）。
    """
    yy = np.arange(h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    inv = 1.0 / (d + float(eps))
    inv = inv / (float(inv.max()) + 1e-12)   # -> [0,1]
    return inv.astype(np.float32)


def build_coord_yy_xx(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    yy/xx ∈ [-1,1]（按像素坐标线性归一化）
    """
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    yy = np.repeat(y, w, axis=1)
    xx = np.repeat(x, h, axis=0)
    return yy, xx


def build_rel_dx_dy(h: int, w: int, cy: float, cx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    dy = yy - ty, dx = xx - tx，其中 ty/tx 也在 [-1,1]。
    注意 dy/dx 的理论范围是 [-2,2]（因为两个[-1,1]相减）。
    """
    yy, xx = build_coord_yy_xx(h, w)
    ty = (float(cy) / (h - 1 + 1e-12)) * 2.0 - 1.0
    tx = (float(cx) / (w - 1 + 1e-12)) * 2.0 - 1.0
    dy = (yy - float(ty)).astype(np.float32)
    dx = (xx - float(tx)).astype(np.float32)
    return dy, dx


# ============================================================
# 4) Normalization rules (per-channel)
# ============================================================

def _apply_log1p(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn = float(x.min())
    if mn < 0:
        x = x - mn
    return np.log1p(x + float(eps)).astype(np.float32)


def apply_x_norm(name: str,
                 arr: np.ndarray,
                 rule: str,
                 x_stats: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    rule examples:
      - "none"
      - "zscore"
      - "log1p+zscore"
      - "minmax"  (needs stats[name].min/max)
    x_stats format建议：
      { "tx_invdist": {"mean":..., "std":..., "min":..., "max":...}, ... }
    """
    r = (rule or "none").lower()
    out = arr.astype(np.float32)

    if r == "none":
        return out

    if "log1p" in r:
        out = _apply_log1p(out)

    if "zscore" in r:
        if not x_stats or name not in x_stats:
            # 降级：不做 zscore（保证可跑），但显式提示更安全
            return out
        mu = float(x_stats[name].get("mean", 0.0))
        sd = float(x_stats[name].get("std", 1.0))
        return ((out - mu) / (sd + 1e-6)).astype(np.float32)

    if r == "minmax":
        if not x_stats or name not in x_stats:
            return out
        lo = float(x_stats[name].get("min", float(out.min())))
        hi = float(x_stats[name].get("max", float(out.max())))
        return ((out - lo) / (hi - lo + 1e-6)).astype(np.float32)

    raise ProtocolError(f"Unknown x_norm rule: {rule}")


def apply_y_transform_and_norm(y_raw: np.ndarray,
                               target_cfg: Dict[str, Any],
                               y_stats: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    target_cfg 来自 protocol["target_norm"]。

    约定：
      - y_transform: "none" | "log1p" | "clip" | "log1p+clip"
      - y_clip_range: [lo,hi] 或 None
      - y_norm: (可选) "none" | "global_minmax" | "zscore"
        * 如果未给 y_norm，我们默认：
            - 有 y_clip_range => global_minmax
            - 否则 => none
    """
    y = y_raw.astype(np.float32)

    y_transform = str(target_cfg.get("y_transform", "none")).lower()
    clip_range = target_cfg.get("y_clip_range", None)

    if "log1p" in y_transform:
        y = _apply_log1p(y)

    if "clip" in y_transform:
        if clip_range is None:
            # 如果协议要求 clip 但没给 range，尝试从 y_stats 里找
            if y_stats and "clip" in y_stats:
                clip_range = y_stats["clip"]
            else:
                raise ProtocolError("y_transform contains 'clip' but y_clip_range is None and y_stats has no 'clip'.")
        lo, hi = float(clip_range[0]), float(clip_range[1])
        y = np.clip(y, lo, hi).astype(np.float32)

    # norm mode
    y_norm = str(target_cfg.get("y_norm", "")).lower().strip()
    if not y_norm:
        y_norm = "global_minmax" if clip_range is not None else "none"

    if y_norm == "none":
        return y.astype(np.float32)

    if y_norm == "global_minmax":
        if clip_range is None:
            # fallback to y_stats
            if y_stats and ("min" in y_stats and "max" in y_stats):
                lo, hi = float(y_stats["min"]), float(y_stats["max"])
            else:
                lo, hi = float(y.min()), float(y.max())
        else:
            lo, hi = float(clip_range[0]), float(clip_range[1])
        return ((y - lo) / (hi - lo + 1e-6)).astype(np.float32)

    if y_norm == "zscore":
        if not y_stats:
            return y.astype(np.float32)
        mu = float(y_stats.get("mean", 0.0))
        sd = float(y_stats.get("std", 1.0))
        return ((y - mu) / (sd + 1e-6)).astype(np.float32)

    raise ProtocolError(f"Unknown y_norm: {y_norm}")


    
# ============================================================
# 4.5) Offline / deterministic target stats (Must-Fix for fair comparison)
# ============================================================

def _proto_for_clip_stats(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Return a protocol copy configured to yield target in 'pre-clip, pre-norm' space.

    - Keep log1p if present in y_transform
    - Disable clip and norm
    """
    p = deepcopy(protocol)
    tn = dict(_get(p, "target_norm", {}))
    y_transform = str(tn.get("y_transform", "none")).lower()
    tn["y_transform"] = "log1p" if ("log1p" in y_transform) else "none"
    tn["y_clip_range"] = None
    tn["y_norm"] = "none"
    p["target_norm"] = tn
    return p


def ensure_target_stats(protocol: Dict[str, Any],
                        *,
                        split: str = "train",
                        percentiles: Tuple[float, float] = (1.0, 99.0),
                        sample_per_image: int = 8192,
                        max_total_samples: int = 2_000_000,
                        seed: int = 123,
                        force_recompute: bool = False,
                        write_clip_into_protocol: bool = True) -> Dict[str, Any]:
    """Deterministically compute and persist y_stats (and y_clip_range) for fair model comparison.

    Why:
      - Prevent normalization drift across runs/models.
      - Provide a single, fixed y_clip_range and stats file to enable metric inversion.

    Behavior:
      - If protocol['target_norm']['y_stats_path'] exists and file exists, load and reuse.
      - Otherwise compute stats from the chosen split sequentially (no DataLoader shuffle).
      - Sampling is deterministic given (seed) and the deterministic order of dataset indices.

    Returns:
      The loaded/computed y_stats dict.
    """
    import numpy as np
    from pathlib import Path

    # Resolve dataset root
    ds_root = Path(str(_get(protocol, "split.dataset_root", default=_get(protocol, "dataset.dataset_root", default="."))))
    tn = dict(_get(protocol, "target_norm", {}))

    y_stats_rel = tn.get("y_stats_path", None)
    if not y_stats_rel:
        # default stats path under dataset root
        y_stats_rel = "stats/y_stats.json"
        tn["y_stats_path"] = y_stats_rel
        protocol["target_norm"] = tn

    y_stats_path = (ds_root / str(y_stats_rel)).resolve()
    y_stats_path.parent.mkdir(parents=True, exist_ok=True)

    if (not force_recompute) and y_stats_path.is_file():
        y_stats = json.loads(y_stats_path.read_text(encoding="utf-8"))
        # optionally write clip into protocol
        if write_clip_into_protocol:
            clip = y_stats.get("clip", None)
            if clip is not None and tn.get("y_clip_range", None) is None:
                tn["y_clip_range"] = [float(clip[0]), float(clip[1])]
                protocol["target_norm"] = tn
        return y_stats

    # Build a dataset that yields target in pre-clip, pre-norm space
    p_stats = _proto_for_clip_stats(protocol)
    ds = RadiomapProtocolDataset(p_stats, split=str(split), strict_layout=False)

    rng = np.random.default_rng(int(seed))
    values = []
    total = 0

    for i in range(len(ds)):
        # sequential, deterministic
        _x, y, _m = ds[i]
        yv = np.asarray(y, dtype=np.float32).reshape(-1)
        if yv.size == 0:
            continue
        k = int(min(int(sample_per_image), yv.size))
        if k <= 0:
            continue
        idx = rng.integers(0, yv.size, size=(k,), dtype=np.int64)
        take = yv[idx]
        values.append(take)
        total += take.size

        # memory guard: downsample deterministically when too many points
        if total >= int(max_total_samples) * 2:
            cat = np.concatenate(values, axis=0)
            if cat.size > int(max_total_samples):
                sel = rng.choice(cat.size, size=(int(max_total_samples),), replace=False)
                cat = cat[sel]
            values = [cat.astype(np.float32)]
            total = int(values[0].size)

    if not values:
        y_stats = {
            "clip": [0.0, 1.0],
            "min": 0.0,
            "max": 1.0,
            "mean": 0.0,
            "std": 1.0,
            "percentiles": [float(percentiles[0]), float(percentiles[1])],
            "n_samples": 0,
        }
    else:
        allv = np.concatenate(values, axis=0).astype(np.float32)
        if allv.size > int(max_total_samples):
            sel = rng.choice(allv.size, size=(int(max_total_samples),), replace=False)
            allv = allv[sel]
        lo_p, hi_p = float(percentiles[0]), float(percentiles[1])
        lo = float(np.percentile(allv, lo_p))
        hi = float(np.percentile(allv, hi_p))
        if abs(hi - lo) < 1e-8:
            hi = lo + 1e-6
        y_stats = {
            "clip": [lo, hi],
            "min": lo,
            "max": hi,
            "mean": float(allv.mean()),
            "std": float(allv.std(ddof=0) + 1e-12),
            "percentiles": [lo_p, hi_p],
            "n_samples": int(allv.size),
            "seed": int(seed),
            "sample_per_image": int(sample_per_image),
            "max_total_samples": int(max_total_samples),
            "y_transform_preclip": str(_get(p_stats, "target_norm.y_transform", default="none")),
        }

    y_stats_path.write_text(json.dumps(y_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    if write_clip_into_protocol:
        # Only write into protocol if it requires clip and clip range missing
        tn2 = dict(_get(protocol, "target_norm", {}))
        y_transform = str(tn2.get("y_transform", "none")).lower()
        if ("clip" in y_transform) and (tn2.get("y_clip_range", None) is None):
            tn2["y_clip_range"] = [float(y_stats["clip"][0]), float(y_stats["clip"][1])]
            protocol["target_norm"] = tn2

    return y_stats


# (removed stray duplicated line from a previous patch)



# ============================================================
# 5\) TX coord resolver
# ============================================================

def detect_tx_center_from_field(y: np.ndarray) -> Tuple[float, float]:
    """
    简单可靠的 fallback：取 argmax。
    如果你需要更稳的连通域中心法，可以在这里替换（保持接口不变）。
    """
    if y.ndim != 2:
        y = y.squeeze()
    h, w = y.shape
    idx = int(np.argmax(y))
    cy, cx = divmod(idx, w)
    return float(cy), float(cx)


@dataclass
class AntennaConfig:
    use_antenna: bool = True              # 是否使用 antenna json
    antenna_subdir: str = "antenna"       # antenna 目录
    antenna_order: str = "xy"             # json 里的坐标顺序：xy 或 yx
    antenna_index_base: int = 0           # tx_id 从 0 还是从 1 开始（按你的数据集）
    cache_path: str = "tx_analysis_results/tx_coords_cache.json"  # tx缓存位置（相对dataset_root）


class TxCoordResolver:
    def __init__(self, dataset_root: Union[str, Path], ant_cfg: AntennaConfig):
        self.root = _as_path(dataset_root)
        self.cfg = ant_cfg
        self.ant_dir = self.root / ant_cfg.antenna_subdir
        self.cache_path = self.root / ant_cfg.cache_path

        self._antenna_cache: Dict[str, List[List[float]]] = {}
        self._tx_cache: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self._load_cache()

    def _load_cache(self):
        try:
            if self.cache_path.is_file():
                obj = json.loads(self.cache_path.read_text(encoding="utf-8"))
                for mid, d in obj.items():
                    self._tx_cache.setdefault(str(mid), {})
                    for txid, yx in d.items():
                        self._tx_cache[str(mid)][str(txid)] = (float(yx[0]), float(yx[1]))
        except Exception:
            pass

    def save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        out = {mid: {txid: [float(yx[0]), float(yx[1])] for txid, yx in d.items()}
               for mid, d in self._tx_cache.items()}
        self.cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    def _load_antenna_list(self, map_id: str) -> List[List[float]]:
        mid = str(map_id)
        if mid in self._antenna_cache:
            return self._antenna_cache[mid]
        p = self.ant_dir / f"{mid}.json"
        if not p.is_file():
            raise FileNotFoundError(f"Antenna json not found: {p}")
        arr = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(arr, list) or len(arr) == 0:
            raise ValueError(f"Invalid antenna json: {p}")
        self._antenna_cache[mid] = arr
        return arr

    def _antenna_tx_yx(self, map_id: str, tx_id: int) -> Tuple[float, float]:
        ant = self._load_antenna_list(map_id)
        idx = int(tx_id) - int(self.cfg.antenna_index_base)
        if idx < 0 or idx >= len(ant):
            raise IndexError(f"tx_id={tx_id} out of range for antenna/{map_id}.json (len={len(ant)})")
        a, b = ant[idx][0], ant[idx][1]
        if str(self.cfg.antenna_order).lower() == "xy":
            x, y = float(a), float(b)
            return float(y), float(x)
        y, x = float(a), float(b)
        return float(y), float(x)

    def resolve(self, map_id: str, tx_id: int, y_raw: np.ndarray) -> Tuple[float, float]:
        mid = str(map_id)
        tid = str(int(tx_id))
        if mid in self._tx_cache and tid in self._tx_cache[mid]:
            return self._tx_cache[mid][tid]

        if bool(self.cfg.use_antenna):
            cy, cx = self._antenna_tx_yx(map_id, tx_id)
            self._tx_cache.setdefault(mid, {})[tid] = (cy, cx)
            return cy, cx

        cy, cx = detect_tx_center_from_field(y_raw)
        self._tx_cache.setdefault(mid, {})[tid] = (cy, cx)
        return cy, cx


# ============================================================
# 6) Split loader
# ============================================================

def _load_split_list(path: Union[str, Path]) -> List[Any]:
    """
    支持：
      - .json: list (map_id list 或 sample_id list)
      - .txt : 每行一个 id
    """
    p = _as_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Split list not found: {p}")
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError(f"Split json must be a list: {p}")
        return obj
    # txt
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and (not ln.startswith("#"))]


def _is_map_list(items: List[Any]) -> bool:
    """
    经验判断：如果元素看起来是纯map_id（不含 tx 分隔符），认为是 map list。
    sample list 的形式允许： "mapid_txid" / [map_id, tx_id] / {"map_id":...,"tx_id":...}
    """
    if not items:
        return True
    x = items[0]
    if isinstance(x, (list, tuple, dict)):
        return False
    s = str(x)
    return ("tx" not in s.lower()) and ("_" not in s) and ("-" not in s)


def _normalize_sample_id(map_id: str, tx_id: int) -> str:
    return f"{str(map_id)}_tx{int(tx_id)}"


# ============================================================
# 7) Dataset (protocol-driven)
# ============================================================

@dataclass
class DataPaths:
    dataset_root: str
    target_subdir: str = "gain/DPM"
    buildings_subdir: str = "png/buildings_complete"
    los_subdir: str = "los_wallcount"  # map_id_tx_id.npy (预计算 LoS/wallcount)
    buildings_ext: str = ".png"   # map_id + .png
    target_npy_channel: int = 0

    building_threshold: int = 127  # 建筑二值化阈值
    building_invert: bool = False  # 是否反转（数据集定义不同才开）


class RadiomapProtocolDataset(Dataset):
    """
    一个“协议驱动”的 Dataset：layout 决定通道顺序与内容定义。

    返回：
      x: (C,H,W) float32
      y: (1,H,W) float32
      meta: dict (map_id, tx_id, tx_y, tx_x, path, baseline_idx, split)

    注意：这里不做 augmentation（你想做增强也建议另外独立模块实现，避免耦合）。
    """

    def __init__(self,
                 protocol: Dict[str, Any],
                 split: str,
                 *,
                 strict_layout: bool = True):
        super().__init__()
        split = str(split).lower().strip()
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train|val|test")
        self.split = split
        self.proto = protocol

        # ---- split / paths ----
        sp = _get(protocol, "split", default=_get(protocol, "dataset", {}))
        self.paths = DataPaths(
            dataset_root=str(_get(sp, "dataset_root")),
            target_subdir=str(sp.get("target_subdir", "gain/DPM")),
            buildings_subdir=str(sp.get("buildings_subdir", "png/buildings_complete")),
            target_npy_channel=int(sp.get("target_npy_channel", 0)),
            building_threshold=int(sp.get("building_threshold", 127)),
            building_invert=bool(sp.get("building_invert", False)),
        )
        self.root = _as_path(self.paths.dataset_root)
        self.tgt_dir = self.root / self.paths.target_subdir
        self.bld_dir = self.root / self.paths.buildings_subdir
        if not self.tgt_dir.is_dir():
            raise FileNotFoundError(f"Target dir not found: {self.tgt_dir}")
        if not self.bld_dir.is_dir():
            raise FileNotFoundError(f"Buildings dir not found: {self.bld_dir}")

        # ---- input proto ----
        self.layout: List[str] = list(_get(protocol, "input.layout"))
        construct = dict(_get(protocol, "input.construct", {}))
        self.sigma = float(construct.get("tx_gaussian_sigma", 3.0))
        self.invdist_eps = float(construct.get("tx_invdist_eps", 1e-3))
        self.coord_norm = str(construct.get("coord_norm", "normalize_to_-1_1_by_image_size"))

        # ---- LoS / wallcount precomputed channel (optional) ----
        # When layout includes "los_wallcount", load: <dataset_root>/<los_subdir>/<map_id>_<tx_id>.npy
        self.los_subdir = str(construct.get("los_subdir", getattr(self.paths, "los_subdir", "los_wallcount")))
        self.los_dir = self.root / self.los_subdir
        self.los_clip_max = float(construct.get("los_clip_max", 20.0))
        self.los_scale_div = float(construct.get("los_scale_div", 20.0))
        if "los_wallcount" in self.layout:
            if not self.los_dir.is_dir():
                raise FileNotFoundError(f"LoS dir not found but required by layout: {self.los_dir}")
            if self.los_scale_div <= 0:
                raise ValueError(f"los_scale_div must be >0, got {self.los_scale_div}")

        # x_norm
        self.x_norm_cfg: Dict[str, str] = dict(_get(protocol, "input.x_norm", default=_get(protocol, "target_norm.x_norm", {})))

        # ---- target proto ----
        self.target_cfg: Dict[str, Any] = dict(_get(protocol, "target_norm", {}))

        # y/x stats path: allow None (meaning "no stats")
        y_stats_path = self.target_cfg.get("y_stats_path", None)
        x_stats_path = self.target_cfg.get("x_stats_path", None)
        self.y_stats = load_json_if_exists(self.root / str(y_stats_path)) if y_stats_path else None
        self.x_stats = load_json_if_exists(self.root / str(x_stats_path)) if x_stats_path else None

        # ---- tx resolver ----
        ant = AntennaConfig(
            use_antenna=bool(sp.get("use_antenna", True)),
            antenna_subdir=str(sp.get("antenna_subdir", "antenna")),
            antenna_order=str(sp.get("antenna_order", "xy")),
            antenna_index_base=int(sp.get("antenna_index_base", 0)),
            cache_path=str(sp.get("tx_cache_path", "tx_analysis_results/tx_coords_cache.json")),
        )
        self.tx_resolver = TxCoordResolver(self.root, ant)

        # ---- build sample index ----
        self.samples = self._build_samples()
        if strict_layout:
            self._assert_layout_supported()

        # cache building per map
        self._bld_cache: Dict[str, np.ndarray] = {}
        # cache coords per size
        self._coord_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    # ----------------------------
    # sample index
    # ----------------------------
    def _build_samples(self) -> List[Tuple[str, int, Path]]:
        sp = _get(self.proto, "split", {})
        train_list = sp.get("train_list_path", None)
        val_list = sp.get("val_list_path", None)
        test_list = sp.get("test_list_path", None)
        path_map = {"train": train_list, "val": val_list, "test": test_list}
        list_path = path_map.get(self.split, None)
        if not list_path:
            raise ProtocolError(f"split.{self.split}_list_path is required in protocol['split'].")

        keep_items = _load_split_list(self.root / str(list_path))
        is_map_list = _is_map_list(keep_items)

        keep_maps: Optional[set] = None
        keep_samples: Optional[set] = None

        if is_map_list:
            keep_maps = set([str(x) for x in keep_items])
        else:
            ks = set()
            for it in keep_items:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    ks.add(_normalize_sample_id(str(it[0]), int(it[1])))
                elif isinstance(it, dict):
                    ks.add(_normalize_sample_id(str(it["map_id"]), int(it["tx_id"])))
                else:
                    s = str(it)
                    # accept "map_txid" or "map_tx123"
                    if "_tx" in s:
                        ks.add(s)
                    else:
                        # fallback: try parse
                        mid, tid = parse_map_tx_from_stem(s)
                        if tid is not None:
                            ks.add(_normalize_sample_id(mid, tid))
            keep_samples = ks

        # scan target dir
        files = [p for p in self.tgt_dir.glob("*") if p.suffix.lower() in (".png", ".npy")]
        out: List[Tuple[str, int, Path]] = []
        for p in files:
            map_id, tx_id = parse_map_tx_from_stem(p.stem)
            if tx_id is None:
                continue
            if keep_maps is not None:
                if str(map_id) not in keep_maps:
                    continue
            if keep_samples is not None:
                sid = _normalize_sample_id(map_id, tx_id)
                if sid not in keep_samples:
                    continue
            out.append((str(map_id), int(tx_id), p))

        if not out:
            raise RuntimeError(f"No samples found under split={self.split}. Check split lists and naming.")
        out.sort(key=lambda t: (safe_int(t[0]), t[1]))
        return out

    # ----------------------------
    # building
    # ----------------------------
    def _load_building(self, map_id: str) -> np.ndarray:
        if map_id in self._bld_cache:
            return self._bld_cache[map_id]
        p = self.bld_dir / f"{map_id}.png"
        if not p.is_file():
            raise FileNotFoundError(f"Building mask not found: {p}")
        arr = load_gray(p).astype(np.float32)
        if not self.paths.building_invert:
            b = (arr > float(self.paths.building_threshold)).astype(np.float32)
        else:
            b = (arr < float(self.paths.building_threshold)).astype(np.float32)
        self._bld_cache[map_id] = b
        return b

    def _load_los_wallcount(self, map_id: str, tx_id: int, h: int, w: int) -> np.ndarray:
        """Load precomputed LoS/wallcount map from .npy and apply light normalization.

        Expected path: <dataset_root>/<los_subdir>/<map_id>_<tx_id>.npy
        Expected shape: (H, W)
        Processing:
          - clip to [0, los_clip_max]
          - divide by los_scale_div (default makes it roughly in [0,1])
        """
        p = self.los_dir / f"{map_id}_{tx_id}.npy"
        if not p.is_file():
            raise FileNotFoundError(f"LoS/wallcount file not found: {p}")
        arr = np.load(p).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"LoS/wallcount must be 2D (H,W), got shape={arr.shape} from {p}")
        if arr.shape[0] != h or arr.shape[1] != w:
            raise ValueError(f"LoS/wallcount shape mismatch: expected {(h,w)}, got {arr.shape} from {p}")
        # light processing
        if self.los_clip_max is not None:
            arr = np.clip(arr, 0.0, float(self.los_clip_max))
        arr = arr / float(self.los_scale_div)
        return arr


    def _coords(self, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (h, w)
        if key not in self._coord_cache:
            self._coord_cache[key] = build_coord_yy_xx(h, w)
        return self._coord_cache[key]

    # ----------------------------
    # layout & channel build
    # ----------------------------
    def _assert_layout_supported(self):
        supported = {"building", "tx_gaussian", "tx_invdist", "dy", "dx", "los_wallcount"}
        unknown = [c for c in self.layout if c not in supported]
        if unknown:
            raise ProtocolError(
                f"layout contains unknown channels: {unknown}\n"
                f"Supported now: {sorted(list(supported))}\n"
                f"（如果你要扩展 P1/P2：在本模块新增 channel builder 并注册即可）"
            )

    def _build_x(self, map_id: str, tx_id: int, y_raw: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        # base fields
        bld = self._load_building(map_id)
        h, w = bld.shape

        cy, cx = self.tx_resolver.resolve(map_id, tx_id, y_raw)
        # channel maps
        ch_map: Dict[str, np.ndarray] = {}

        # building
        ch_map["building"] = bld

        # tx maps
        ch_map["tx_gaussian"] = build_tx_gaussian(h, w, cy, cx, sigma=self.sigma)
        ch_map["tx_invdist"] = build_tx_invdist(h, w, cy, cx, eps=self.invdist_eps)

        # precomputed LoS / wallcount (optional)
        if "los_wallcount" in self.layout:
            ch_map["los_wallcount"] = self._load_los_wallcount(map_id, tx_id, h, w)

        # coords
        dy, dx = build_rel_dx_dy(h, w, cy, cx)
        ch_map["dy"] = dy
        ch_map["dx"] = dx

        # stack in protocol order + per-channel norm
        x_list: List[np.ndarray] = []
        for name in self.layout:
            arr = ch_map[name].astype(np.float32)
            rule = self.x_norm_cfg.get(name, "none")
            arr = apply_x_norm(name, arr, rule, x_stats=self.x_stats)
            x_list.append(arr)

        x = np.stack(x_list, axis=0).astype(np.float32)

        meta = {
            "map_id": str(map_id),
            "tx_id": int(tx_id),
            "tx_y": float(cy),
            "tx_x": float(cx),
            "baseline_idx": int(self.layout.index("tx_invdist")) if "tx_invdist" in self.layout else -1,
            "split": self.split,
        }
        return x, meta

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        map_id, tx_id, tgt_path = self.samples[idx]
        y_raw = load_field(tgt_path, npy_channel=self.paths.target_npy_channel)
        if y_raw.ndim != 2:
            y_raw = y_raw.squeeze()
        y_raw = y_raw.astype(np.float32)

        x, meta = self._build_x(map_id, tx_id, y_raw)
        y = apply_y_transform_and_norm(y_raw, self.target_cfg, y_stats=self.y_stats)[None, ...].astype(np.float32)

        meta["path"] = str(tgt_path)

        return torch.from_numpy(x), torch.from_numpy(y), meta


# ============================================================
# 8) Model input adapters (三模型同一份 x 的不同吃法)
# ============================================================

class ModelInputAdapter:
    """
    把同一份公开输入 x (B,C,H,W) 转成不同模型需要的输入。
    注意：这不是“私有信息”，只是对 x 的切片/分组。
    """
    def __init__(self, protocol: Dict[str, Any]):
        self.proto = protocol
        self.layout = list(_get(protocol, "input.layout"))
        # Unified roles field for GCGO (preferred): protocol["input"]["roles"]["gcgo"]
        # Backward-compatible with older configs that used a legacy key under protocol["input"].
        roles = _get(protocol, "input.roles.gcgo", None)
        legacy_key = "gcgo_" "roles"  # legacy compatibility key
        if roles is None:
            roles = _get(protocol, f"input.{legacy_key}", None)
        if roles is None:
            roles = {"source_channel":"tx_gaussian", "geom_channels":["building","tx_invdist","dy","dx"]}
        roles = dict(roles)
        self.gcgo_source_name = str(roles.get("source_channel", "tx_gaussian"))
        self.gcgo_geom_names = list(roles.get("geom_channels", ["building", "tx_invdist", "dy", "dx"]))

        # UNet allowed input channels (preferred): protocol['input']['roles']['unet']
        unet_roles = _get(protocol, 'input.roles.unet', None)
        if unet_roles is None:
            # Fallback: keep official RadioUNet default inputs
            unet_roles = ['building', 'tx_gaussian']
        self.unet_channel_names = list(unet_roles)

        self._idx = {n: i for i, n in enumerate(self.layout)}

        if self.gcgo_source_name not in self._idx:
            raise ProtocolError(f"GCGO source_channel '{self.gcgo_source_name}' not in layout={self.layout}")
        for n in self.gcgo_geom_names:
            if n not in self._idx:
                raise ProtocolError(f"GCGO geom_channel '{n}' not in layout={self.layout}")

        # Validate UNet channels
        for n in self.unet_channel_names:
            if n not in self._idx:
                raise ProtocolError(f"UNet channel '{n}' not in layout={self.layout}")

    def make_model_inputs(self,
                          model_type: str,
                          x: torch.Tensor,
                          meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None) -> Any:
        """
        model_type: "fno" | "unet" | "gcgo"
        x: (B,C,H,W) or (C,H,W)
        返回：
          - fno: x 原样返回
          - unet: 只返回 protocol.input.roles.unet 指定的子通道
          - gcgo: dict {"geom": geom_feat, "f_src": f_src, "meta": meta}
        """
        mt = str(model_type).lower().strip()
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if mt == "fno":
            return x

        if mt == "unet":
            # Slice only the channels UNet is allowed to see (default: building + tx_gaussian).
            idxs = [self._idx[n] for n in self.unet_channel_names]
            x_u = x[:, idxs, :, :]
            return x_u


        if mt == "gcgo":
            src_idx = self._idx[self.gcgo_source_name]
            geom_idxs = [self._idx[n] for n in self.gcgo_geom_names]
            f_src = x[:, src_idx:src_idx+1, ...]          # (B,1,H,W)
            geom = x[:, geom_idxs, ...]                   # (B,Cg,H,W)
            return {"geom": geom, "f_src": f_src, "meta": meta}

        raise ValueError("model_type must be fno|unet|gcgo")


# ============================================================
# 9) Optional: stats fitting helpers (离线用，不建议训练时每次自动扫)
# ============================================================

def fit_channel_stats(dataset: Dataset,
                      channel_names: List[str],
                      sample_limit: Optional[int] = None,
                      eps: float = 1e-6) -> Dict[str, Dict[str, float]]:
    """
    对 dataset 输出的 x 做简单统计（mean/std/min/max），用于 zscore/minmax。
    注意：这是离线工具函数，建议对 train split 跑一次写到 stats/x_stats.json。
    """
    n = len(dataset) if sample_limit is None else min(len(dataset), int(sample_limit))
    sums = {c: 0.0 for c in channel_names}
    sqs  = {c: 0.0 for c in channel_names}
    mins = {c: float("inf") for c in channel_names}
    maxs = {c: float("-inf") for c in channel_names}
    cnt  = 0

    for i in range(n):
        x, _, _ = dataset[i]
        x = x.numpy()  # (C,H,W)
        for ci, c in enumerate(channel_names):
            arr = x[ci].astype(np.float64)
            sums[c] += float(arr.mean())
            sqs[c]  += float((arr**2).mean())
            mins[c]  = min(mins[c], float(arr.min()))
            maxs[c]  = max(maxs[c], float(arr.max()))
        cnt += 1

    out: Dict[str, Dict[str, float]] = {}
    for c in channel_names:
        mu = sums[c] / max(1, cnt)
        m2 = sqs[c] / max(1, cnt)
        var = max(0.0, m2 - mu*mu)
        sd = math.sqrt(var)
        out[c] = {"mean": float(mu), "std": float(sd + eps), "min": float(mins[c]), "max": float(maxs[c])}
    return out


def fit_target_clip(dataset: Dataset,
                    percentiles: Tuple[float, float] = (1.0, 99.0),
                    sample_limit: Optional[int] = None) -> Dict[str, Any]:
    """
    拟合 y 的 clip 范围（percentile clip），用于 global_minmax。
    返回：{"clip":[lo,hi], "min":lo, "max":hi}
    """
    n = len(dataset) if sample_limit is None else min(len(dataset), int(sample_limit))
    buf = []
    for i in range(n):
        _, y, _ = dataset[i]
        arr = y.numpy().reshape(-1).astype(np.float32)
        # 采样一部分点加速
        if arr.size > 4000:
            idx = np.random.choice(arr.size, size=4000, replace=False)
            arr = arr[idx]
        buf.append(arr)
    allv = np.concatenate(buf, axis=0)
    lo = float(np.percentile(allv, float(percentiles[0])))
    hi = float(np.percentile(allv, float(percentiles[1])))
    return {"clip": [lo, hi], "min": lo, "max": hi}