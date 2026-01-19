# -*- coding: utf-8 -*-
"""
rm_protocol_eval.py
============================================================
统一评估模块（Protocol-driven Evaluator）

你现在已经确定：
- 输入协议 P0+（通道布局、通道含义、开关）
- 评估协议 EvalProtocol（compare_space、mask_buildings、boundary_width、high_q、dist_bins 等）

本文件把“评估口径”彻底抽离出来，作为所有模型（FNO/UNet/GCGO）的统一出口：
- 模型输出 pred -> 统一 evaluator -> 返回标准 metrics dict
- 训练期：用于 val 监控与 save_best_by
- 最终期：用于出报告（含分位数、距离分桶、可选可视化输出）

原则
- evaluator 只依赖：pred, gt, x(至少包含 building 通道), protocol
- 三模型对比时，必须使用同一套 evaluator（避免口径漂移）
- evaluator 内部不做“额外信息注入”：只基于 x/gt/pred 派生 mask

============================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import json

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from rm_protocol_dataset import load_json_if_exists



# ============================================================
# 0) Exceptions
# ============================================================

class EvalProtocolError(RuntimeError):
    """评估协议不合法/必要字段缺失时抛出。"""
    pass


# ============================================================
# 1) Small helpers
# ============================================================

def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def ensure_4d(x: torch.Tensor) -> torch.Tensor:
    """(H,W)->(1,1,H,W), (C,H,W)->(1,C,H,W), (B,C,H,W)->same"""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"Tensor must be 2/3/4D, got shape={tuple(x.shape)}")


def ensure_1chw(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape is (B,1,H,W)"""
    x = ensure_4d(x)
    if x.shape[1] != 1:
        raise ValueError(f"Expect channel=1, got {x.shape}")
    return x


def resize_like(pred: torch.Tensor, gt: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Resize pred to match gt spatial size if needed."""
    pred = ensure_4d(pred)
    gt = ensure_4d(gt)
    if pred.shape[-2:] == gt.shape[-2:]:
        return pred
    return F.interpolate(pred, size=gt.shape[-2:], mode=mode, align_corners=False)


def torch_nanmean(x: torch.Tensor) -> torch.Tensor:
    """torch.mean but ignore NaNs (safe fallback)."""
    if torch.isnan(x).any():
        return torch.nanmean(x)
    return x.mean()


def sample_tensor_1d(x: torch.Tensor, max_points: int = 4096) -> torch.Tensor:
    """
    从 1D tensor 里最多采样 max_points 个点（用于 quantile / percentiles 加速）。
    """
    x = x.flatten()
    n = x.numel()
    if n <= max_points:
        return x
    idx = torch.randint(low=0, high=n, size=(max_points,), device=x.device)
    return x[idx]


# ============================================================
# 2) Mask builders (domain / boundary / high-value)
# ============================================================

def build_domain_mask(building: torch.Tensor, mask_buildings: bool) -> torch.Tensor:
    """
    building: (B,1,H,W) 取值 0/1（1=建筑）
    返回 mask_domain: (B,1,H,W) bool
      - mask_buildings=True  -> outdoor-only (building==0)
      - mask_buildings=False -> all pixels
    """
    building = ensure_1chw(building)
    if mask_buildings:
        return (building <= 0.5)
    return torch.ones_like(building, dtype=torch.bool)


def _dilate(binary: torch.Tensor, radius: int) -> torch.Tensor:
    """binary (B,1,H,W) float/bool -> float (0/1) dilation using max_pool"""
    if radius <= 0:
        return binary.float()
    x = binary.float()
    k = 2 * radius + 1
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=radius)


def _erode(binary: torch.Tensor, radius: int) -> torch.Tensor:
    """binary erosion using max_pool on inverted"""
    if radius <= 0:
        return binary.float()
    x = binary.float()
    inv = 1.0 - x
    k = 2 * radius + 1
    inv_dil = F.max_pool2d(inv, kernel_size=k, stride=1, padding=radius)
    return 1.0 - inv_dil


def build_boundary_ring(building: torch.Tensor,
                        width: int,
                        *,
                        outdoor_only: bool = True) -> torch.Tensor:
    """
    边界环：通过 dilation/erosion 的差得到“边界附近区域”。
    - width=2 表示半径2的环（厚度与实现有关，稳定即可）
    - outdoor_only=True：只保留 outdoor 部分（与主指标一致）
    返回：(B,1,H,W) bool
    """
    building = ensure_1chw(building)
    w = int(max(0, width))
    dil = _dilate(building > 0.5, w)
    ero = _erode(building > 0.5, w)
    ring = (dil - ero) > 1e-6
    if outdoor_only:
        ring = ring & (building <= 0.5)
    return ring


def make_boundary_ring_mask(building: torch.Tensor, width: int) -> torch.Tensor:
    """Alias for backward compatibility.

    Returns a boundary ring around buildings (both sides). The caller can AND with domain_mask
    (e.g., outdoor-only) to match the evaluation protocol.
    """
    return build_boundary_ring(building, width, outdoor_only=False)


def make_high_value_mask(gt: torch.Tensor, q: float = 0.9) -> torch.Tensor:
    """High-value mask by per-sample quantile on GT."""
    if gt.dim() == 3:
        gt_ = gt.unsqueeze(1)
    else:
        gt_ = gt
    B, C, H, W = gt_.shape
    flat = gt_.reshape(B, -1)
    thr = torch.quantile(flat, q, dim=1, keepdim=True)  # (B,1)
    return (flat >= thr).reshape(B, C, H, W)

def error_percentiles(abs_err: torch.Tensor,
                      domain_mask: torch.Tensor,
                      percentiles=(50, 90, 99),
                      sample_per_image: int = 800) -> dict:
    """Approximate abs-error percentiles over a boolean mask."""
    if abs_err.dim() == 3:
        abs_err = abs_err.unsqueeze(1)
    if domain_mask.dim() == 3:
        domain_mask = domain_mask.unsqueeze(1)

    B = abs_err.shape[0]
    out = {}
    for p in percentiles:
        vals = []
        for b in range(B):
            v = abs_err[b][domain_mask[b]].flatten()
            if v.numel() == 0:
                continue
            if sample_per_image and v.numel() > sample_per_image:
                idx = torch.randint(0, v.numel(), (sample_per_image,), device=v.device)
                v = v[idx]
            vals.append(v)
        if not vals:
            out[f"abs_p{int(p)}"] = float("nan")
            continue
        vv = torch.cat(vals, dim=0)
        out[f"abs_p{int(p)}"] = float(torch.quantile(vv, float(p) / 100.0).item())
    return out


def build_high_value_mask(gt: torch.Tensor,
                          domain_mask: torch.Tensor,
                          q: float,
                          *,
                          max_points: int = 4096) -> torch.Tensor:
    """
    高值区域：对每张图在 domain_mask 内取 gt 的 q 分位数阈值，
    mask_high = gt >= thr 且在 domain_mask 内
    返回：(B,1,H,W) bool
    """
    gt = ensure_1chw(gt)
    domain_mask = ensure_1chw(domain_mask).bool()
    q = float(q)
    if not (0.0 < q < 1.0):
        raise EvalProtocolError(f"high_q must be in (0,1), got {q}")

    B = gt.shape[0]
    out = torch.zeros_like(gt, dtype=torch.bool)
    for b in range(B):
        vals = gt[b][domain_mask[b]]
        if vals.numel() == 0:
            continue
        vals_s = sample_tensor_1d(vals, max_points=max_points)
        thr = torch.quantile(vals_s, q)
        out[b] = (gt[b] >= thr) & domain_mask[b]
    return out


# ============================================================
# 3) Metrics (MAE/RMSE + optional percentiles & dist bins)
# ============================================================

def masked_mae(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = ensure_1chw(pred)
    gt = ensure_1chw(gt)
    mask = ensure_1chw(mask).bool()
    diff = (pred - gt).abs()
    vals = diff[mask]
    if vals.numel() == 0:
        return torch.tensor(float("nan"), device=pred.device)
    return vals.mean()


def masked_rmse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = ensure_1chw(pred)
    gt = ensure_1chw(gt)
    mask = ensure_1chw(mask).bool()
    diff2 = (pred - gt) ** 2
    vals = diff2[mask]
    if vals.numel() == 0:
        return torch.tensor(float("nan"), device=pred.device)
    return torch.sqrt(vals.mean() + 1e-12)


def masked_abs_percentiles(pred: torch.Tensor,
                           gt: torch.Tensor,
                           mask: torch.Tensor,
                           percentiles: Sequence[float],
                           *,
                           sample_per_image: int = 800,
                           max_points: int = 200000) -> Dict[str, float]:
    """
    统计 abs error 的分位数（p50/p90/p99），用于长尾分析。
    - 默认每张图采样 sample_per_image 个点加速
    返回：{"abs_p50":..., "abs_p90":..., ...}（按 batch 合并）
    """
    pred = ensure_1chw(pred)
    gt = ensure_1chw(gt)
    mask = ensure_1chw(mask).bool()
    B = pred.shape[0]
    all_samples = []
    for b in range(B):
        diff = (pred[b] - gt[b]).abs()
        vals = diff[mask[b]]
        if vals.numel() == 0:
            continue
        if sample_per_image and vals.numel() > sample_per_image:
            idx = torch.randint(0, vals.numel(), size=(sample_per_image,), device=vals.device)
            vals = vals[idx]
        all_samples.append(vals)
    if not all_samples:
        return {f"abs_p{int(p)}": float("nan") for p in percentiles}
    x = torch.cat(all_samples, dim=0)
    if x.numel() > max_points:
        idx = torch.randint(0, x.numel(), size=(max_points,), device=x.device)
        x = x[idx]
    out = {}
    for p in percentiles:
        out[f"abs_p{int(p)}"] = float(torch.quantile(x, float(p)/100.0).item())
    return out


def dist_to_building_bins(building: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    """
    计算“到最近建筑像素的距离”（像素单位）并返回 distance map (H,W) float32。

    注意：需要 scipy（distance_transform_edt）。如果环境没有 scipy，会抛 ImportError。
    """
    from scipy.ndimage import distance_transform_edt  # noqa

    b = (building > 0.5).astype(np.uint8)  # 1=building
    outdoor = (1 - b).astype(np.uint8)     # 1=outdoor, 0=building
    # distance_transform_edt：对非零元素计算到最近零元素距离
    # 这里 outdoor==1 是非零，building==0 是零 => 得到 outdoor 到 building 的距离
    dist = distance_transform_edt(outdoor).astype(np.float32)
    return dist


def masked_dist_bin_metrics(pred: torch.Tensor,
                            gt: torch.Tensor,
                            building: torch.Tensor,
                            domain_mask: torch.Tensor,
                            bins: Sequence[float]) -> Dict[str, float]:
    """Distance-bin metrics with *pixel-weighted* (micro) aggregation.

    For each bin [lo, hi):
      - dist_mae_<lo>_<hi>_sum / _count   (abs error sum + pixel count)
      - dist_rmse_<lo>_<hi>_sq_sum / _count (squared error sum + pixel count)

    Also returns the derived:
      - dist_mae_<lo>_<hi>
      - dist_rmse_<lo>_<hi>

    Notes:
      - Only pixels inside domain_mask are counted (typically outdoor-only).
      - This avoids the "average of averages" bias.
    """
    pred = ensure_1chw(pred)
    gt = ensure_1chw(gt)
    building = ensure_1chw(building)
    domain_mask = ensure_1chw(domain_mask).bool()

    bins = list(bins)
    if len(bins) < 2:
        return {}

    B = pred.shape[0]
    abs_err = (pred - gt).abs()
    sq_err = (pred - gt) ** 2

    # accumulator
    abs_sum = {}
    sq_sum = {}
    cnt_sum = {}

    for b in range(B):
        bld_np = building[b, 0].detach().cpu().numpy().astype('float32')
        try:
            dist = dist_to_building_bins(bld_np, bins=bins)  # (H,W)
        except Exception:
            # scipy missing or other failure => skip dist bins
            return {}

        dist_t = torch.from_numpy(dist).to(pred.device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        for i in range(len(bins) - 1):
            lo = float(bins[i])
            hi = float(bins[i+1])
            m = (dist_t >= lo) & (dist_t < hi) & domain_mask[b:b+1]
            c = float(m.sum().item())
            if c <= 0:
                continue

            a = float(abs_err[b:b+1][m].sum().item())
            s = float(sq_err[b:b+1][m].sum().item())

            hi_lab = (int(hi) if hi < 1e8 else 'inf')
            key_mae = f"dist_mae_{int(lo)}_{hi_lab}"
            key_rmse = f"dist_rmse_{int(lo)}_{hi_lab}"

            abs_sum[key_mae] = abs_sum.get(key_mae, 0.0) + a
            sq_sum[key_rmse] = sq_sum.get(key_rmse, 0.0) + s
            cnt_sum[key_mae] = cnt_sum.get(key_mae, 0.0) + c
            cnt_sum[key_rmse] = cnt_sum.get(key_rmse, 0.0) + c

    out = {}
    for k, a in abs_sum.items():
        c = cnt_sum.get(k, 0.0)
        out[f"{k}_sum"] = float(a)
        out[f"{k}_count"] = float(c)
        out[k] = float(a / max(1e-12, c))

    for k, s in sq_sum.items():
        c = cnt_sum.get(k, 0.0)
        out[f"{k}_sq_sum"] = float(s)
        out[f"{k}_count"] = float(c)
        out[k] = float(math.sqrt(s / max(1e-12, c)))

    return out


# ============================================================
# 4) Composite score (save_best_by)
# ============================================================

def composite_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    组合分数：sum(w_i * metric_i) —— 越小越好（默认用于误差类指标）。
    例如 weights={"rmse":1.0, "boundary_rmse":0.5, "high_rmse":0.5}
    """
    s = 0.0
    for k, w in weights.items():
        if k not in metrics or metrics[k] is None:
            continue
        v = metrics[k]
        if isinstance(v, torch.Tensor):
            v = float(v.item())
        s += float(w) * float(v)
    return float(s)


# ============================================================
# 5) Evaluator class (single source of truth)
# ============================================================

@dataclass
class EvalSettings:
    compare_space: str = "fno_target"  # "fno_target" | "raw"（目前 evaluator 默认不做 invert，仅用于记录/断言）
    mask_buildings: bool = True
    boundary_width: int = 2
    high_q: float = 0.90
    resize_pred_to_gt: bool = True
    resize_mode: str = "bilinear"
    post_clamp01: bool = False

    # extra report
    percentiles_enabled: bool = True
    sample_per_image: int = 800
    percentiles: Tuple[int, ...] = (50, 90, 99)

    dist_bins_enabled: bool = True
    dist_bins: Tuple[float, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 1e9)


class ProtocolEvaluator:
    """Unified evaluator.

    Key fixes implemented:
      1) Metrics can be computed in *physical space* by inverting target transforms.
      2) Distance-bin metrics use micro-averaging (sum-over-count).
    """

    def __init__(self,
                 protocol: Dict[str, Any],
                 *,
                 metrics: Optional[List[str]] = None,
                 composite_weights: Optional[Dict[str, float]] = None,
                 return_composite: Optional[bool] = None,
                 percentiles_enabled: Optional[bool] = None,
                 dist_bins_enabled: Optional[bool] = None):
        self.proto = protocol

        layout = _get(protocol, "input.layout", None)
        if not layout:
            raise EvalProtocolError("protocol missing input.layout")
        self.layout: List[str] = list(layout)
        if "building" not in self.layout:
            raise EvalProtocolError("input.layout must contain 'building' channel for masking.")
        self.building_idx = self.layout.index("building")

        ev = _get(protocol, "eval", {}) or {}
        self.cfg = EvalSettings(
            compare_space=str(ev.get("compare_space", "fno_target")),
            mask_buildings=bool(ev.get("mask_buildings", True)),
            boundary_width=int(ev.get("boundary_width", 2)),
            high_q=float(ev.get("high_q", 0.90)),
            resize_pred_to_gt=bool(ev.get("resize_pred_to_gt", True)),
            resize_mode=str(ev.get("resize_mode", "bilinear")),
            post_clamp01=bool(ev.get("post_clamp01", False)),
            percentiles_enabled=bool(_get(ev, "error_percentiles.enabled", True)),
            sample_per_image=int(_get(ev, "error_percentiles.sample_per_image", 800)),
            percentiles=tuple(int(x) for x in _get(ev, "error_percentiles.percentiles", [50, 90, 99])),
            dist_bins_enabled=bool(_get(ev, "dist_bins.enabled", True)),
            dist_bins=tuple(float(x) for x in _get(ev, "dist_bins.bins", [0, 1, 2, 4, 8, 16, 32, 64, 1e9])),
        )

        # metric space: "physical" (invert target transform) or "normalized" (old behavior)
        self.metrics_space = str(ev.get("metrics_space", "physical")).lower().strip()
        self.composite_weights = dict(ev.get("composite_weights", {"rmse": 1.0, "boundary_rmse": 0.5, "high_rmse": 0.5}))

        # Override weights if provided
        if composite_weights is not None:
            self.composite_weights = dict(composite_weights)

        # Metrics subset (default: protocol.eval.eval_metrics if present; else full 3-metric set)
        if metrics is None:
            metrics = list(ev.get("eval_metrics", ["rmse", "boundary_rmse", "high_rmse"]))
        self.metrics_to_compute = [str(m).strip() for m in metrics]
        allowed = {"rmse", "boundary_rmse", "high_rmse"}
        for mname in self.metrics_to_compute:
            if mname not in allowed:
                raise EvalProtocolError(f"Unsupported metric '{mname}'. Supported: {sorted(allowed)}")

        # Whether to return composite (training may use it; test often disables it)
        if return_composite is None:
            return_composite = bool(ev.get("return_composite", True))
        self.return_composite = bool(return_composite)

        # Optional overrides for extra diagnostics
        if percentiles_enabled is not None:
            self.cfg.percentiles_enabled = bool(percentiles_enabled)
        if dist_bins_enabled is not None:
            self.cfg.dist_bins_enabled = bool(dist_bins_enabled)


        # target inversion config
        self.root = Path(str(_get(protocol, "split.dataset_root", default=".")))
        self.target_cfg: Dict[str, Any] = dict(_get(protocol, "target_norm", {}))

        y_stats_path = self.target_cfg.get("y_stats_path", None)
        self.y_stats = load_json_if_exists(self.root / str(y_stats_path)) if y_stats_path else None

        clip = self.target_cfg.get("y_clip_range", None)
        if clip is None and self.y_stats is not None:
            clip = self.y_stats.get("clip", None)
        self.clip_range = clip

        if self.metrics_space == "physical":
            # For physical inversion we need clip range for global_minmax.
            y_norm = str(self.target_cfg.get("y_norm", "none")).lower().strip()
            y_transform = str(self.target_cfg.get("y_transform", "none")).lower().strip()
            if y_norm == "global_minmax" and self.clip_range is None and not (self.y_stats and ("min" in self.y_stats and "max" in self.y_stats)):
                raise EvalProtocolError("metrics_space='physical' requires y_clip_range or y_stats min/max for global_minmax inversion")
            if "log1p" in y_transform and y_norm == "zscore" and not self.y_stats:
                raise EvalProtocolError("metrics_space='physical' requires y_stats (mean/std) for zscore inversion")

    # ---------------- inversion utilities ----------------
    def invert_target_tensor(self, y: torch.Tensor) -> torch.Tensor:
        """Invert dataset target transform (norm + optional log1p) back to physical space.

        If metrics_space != 'physical', you should not call this.
        """
        y = ensure_1chw(y).float()

        y_norm = str(self.target_cfg.get("y_norm", "none")).lower().strip()
        y_transform = str(self.target_cfg.get("y_transform", "none")).lower().strip()

        yy = y

        if y_norm == "global_minmax":
            if self.clip_range is None:
                # fallback: stats min/max
                if self.y_stats and ("min" in self.y_stats and "max" in self.y_stats):
                    lo, hi = float(self.y_stats["min"]), float(self.y_stats["max"])
                else:
                    raise RuntimeError("invert_target_tensor: missing clip_range/minmax stats")
            else:
                lo, hi = float(self.clip_range[0]), float(self.clip_range[1])
            yy = yy * (hi - lo) + lo

        elif y_norm == "zscore":
            if not self.y_stats:
                raise RuntimeError("invert_target_tensor: zscore requires y_stats")
            mu = float(self.y_stats.get("mean", 0.0))
            sd = float(self.y_stats.get("std", 1.0))
            yy = yy * sd + mu

        elif y_norm in ("none", ""):
            yy = yy
        else:
            # unknown norm -> keep as-is
            yy = yy

        # clip in transform-space can't be undone; but denorm should already be within clip.
        if "clip" in y_transform and self.clip_range is not None:
            lo, hi = float(self.clip_range[0]), float(self.clip_range[1])
            yy = yy.clamp(lo, hi)

        if "log1p" in y_transform:
            yy = torch.expm1(torch.clamp(yy, min=0.0))

        return yy

    def _maybe_metric_space(self, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.metrics_space == "physical":
            return self.invert_target_tensor(pred), self.invert_target_tensor(gt)
        return pred, gt

    # ---------------- existing helpers ----------------
    def _extract_building(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_4d(x)
        if x.shape[1] <= self.building_idx:
            raise ValueError(f"x has {x.shape[1]} channels but building_idx={self.building_idx}")
        return x[:, self.building_idx:self.building_idx+1]


    def _prepare(self, pred: torch.Tensor, gt: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = ensure_1chw(pred)
        gt = ensure_1chw(gt)
        x = ensure_4d(x)

        # optional resize
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = resize_like(pred, gt, mode=self.cfg.resize_mode)

        if self.cfg.post_clamp01 and self.metrics_space != "physical":
            # only meaningful in normalized space
            pred = pred.clamp(0.0, 1.0)

        building = self._extract_building(x)
        building = (building > 0.5).float()
        return pred, gt, building

    @torch.no_grad()
    def compute_batch(self,
                      pred: torch.Tensor,
                      gt: torch.Tensor,
                      x: torch.Tensor,
                      *,
                      want_percentiles: Optional[bool] = None,
                      want_dist_bins: Optional[bool] = None) -> Dict[str, float]:
        """Compute metrics for a batch (protocol-driven).

        By default the evaluator can compute:
          - rmse            : over domain (outdoor if mask_buildings=True)
          - boundary_rmse   : over boundary ring (intersection with domain)
          - high_rmse       : over top-q high-value region (intersection with domain)

        Which metrics are actually computed is controlled by `self.metrics_to_compute`.
        Extra diagnostics (percentiles / distance bins) are controlled by self.cfg flags,
        and can be overridden by want_percentiles / want_dist_bins.
        """
        pred = ensure_1chw(pred)
        gt = ensure_1chw(gt)

        # x can be either a (B,1,H,W) building mask OR the full input tensor (B,C,H,W).
        # For masking/boundary computation we only need the building channel.
        x4 = ensure_4d(x)
        if x4.shape[1] == 1:
            x_build = x4
        else:
            # Select building channel using protocol layout.
            if getattr(self, "building_idx", None) is None:
                raise EvalProtocolError("Cannot select building channel: evaluator missing building_idx")
            if self.building_idx < 0 or self.building_idx >= x4.shape[1]:
                raise EvalProtocolError(
                    f"Building channel index out of range: building_idx={self.building_idx}, x.shape={tuple(x4.shape)}"
                )
            x_build = x4[:, self.building_idx:self.building_idx + 1]
        x = ensure_1chw(x_build)

        # Align pred resolution to gt if needed
        if self.cfg.resize_pred_to_gt and (pred.shape[-2:] != gt.shape[-2:]):
            pred = torch.nn.functional.interpolate(
                pred, size=gt.shape[-2:], mode=self.cfg.resize_mode, align_corners=False
            )

        # Post clamp in normalized space (optional)
        if self.cfg.post_clamp01:
            pred = pred.clamp(0.0, 1.0)

        # Build masks from x (layout includes building)
        building = x  # x is already (B,1,H,W) building mask
        domain_mask = torch.ones_like(building, dtype=torch.bool)
        if self.cfg.mask_buildings:
            domain_mask = (building <= 0.5)

        # Work in the requested metrics space
        pred_m, gt_m = self._maybe_metric_space(pred, gt)
        diff = pred_m - gt_m
        sq_err = diff ** 2

        def _sq_sum_cnt(mask: torch.Tensor):
            mask = ensure_1chw(mask).bool()
            c = float(mask.sum().item())
            if c <= 0:
                return 0.0, 0.0
            s = float(sq_err[mask].sum().item())
            return s, c

        out: Dict[str, float] = {}

        # rmse
        if "rmse" in self.metrics_to_compute:
            s, c = _sq_sum_cnt(domain_mask)
            out["rmse_sq_sum"] = s
            out["rmse_count"] = c
            out["rmse"] = float(math.sqrt(s / max(1e-12, c))) if c > 0 else float('nan')

        # boundary rmse
        if "boundary_rmse" in self.metrics_to_compute:
            boundary_mask = make_boundary_ring_mask(building, width=self.cfg.boundary_width) & domain_mask
            s, c = _sq_sum_cnt(boundary_mask)
            out["boundary_rmse_sq_sum"] = s
            out["boundary_rmse_count"] = c
            out["boundary_rmse"] = float(math.sqrt(s / max(1e-12, c))) if c > 0 else float('nan')

        # high rmse
        if "high_rmse" in self.metrics_to_compute:
            high_mask = make_high_value_mask(gt_m, q=self.cfg.high_q) & domain_mask
            s, c = _sq_sum_cnt(high_mask)
            out["high_rmse_sq_sum"] = s
            out["high_rmse_count"] = c
            out["high_rmse"] = float(math.sqrt(s / max(1e-12, c))) if c > 0 else float('nan')

        # Extra diagnostics
        wp = self.cfg.percentiles_enabled if want_percentiles is None else bool(want_percentiles)
        if wp:
            # Note: percentiles are computed on absolute error in physical space
            abs_err = diff.abs()
            pct = error_percentiles(abs_err, domain_mask, self.cfg.percentiles, sample_per_image=self.cfg.sample_per_image)
            out.update(pct)

        wd = self.cfg.dist_bins_enabled if want_dist_bins is None else bool(want_dist_bins)
        if wd:
            d = masked_dist_bin_metrics(pred_m, gt_m, building, domain_mask, bins=self.cfg.dist_bins)
            out.update(d)

        # Composite score (optional)
        if self.return_composite:
            out["composite"] = composite_score(out, self.composite_weights)

        return out

    @staticmethod
    def reduce_epoch(batch_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate batch metrics.

        - For *_sum/*_sq_sum/*_count keys: sums across batches.
        - For other scalar keys: arithmetic mean across batches (ignoring NaN).
        - Recompute derived mae/rmse (and dist bin mae/rmse) using global sums/counts.
        """
        if not batch_metrics:
            return {}

        out: Dict[str, float] = {}
        sum_keys = {}
        mean_keys = {}

        # aggregate
        for m in batch_metrics:
            for k, v in m.items():
                if v is None:
                    continue
                if k.endswith('_sum') or k.endswith('_sq_sum') or k.endswith('_count'):
                    sum_keys[k] = sum_keys.get(k, 0.0) + float(v)
                else:
                    mean_keys.setdefault(k, []).append(float(v))

        # write sums
        out.update({k: float(v) for k, v in sum_keys.items()})

        # write means (这里会自动包含 'composite' 的平均值)
        for k, vs in mean_keys.items():
            vv = [x for x in vs if x == x]  # drop NaN
            out[k] = float(sum(vv) / max(1, len(vv))) if vv else float('nan')

        # derived helpers
        def _derive_rmse(name: str, sq_sum_k: str, cnt_k: str):
            s = out.get(sq_sum_k, None)
            c = out.get(cnt_k, None)
            if s is None or c is None or c <= 0:
                out[name] = float('nan')
                return
            out[name] = float(math.sqrt(float(s) / max(1e-12, float(c))))

        # main metrics
        _derive_rmse('rmse', 'rmse_sq_sum', 'rmse_count')
        _derive_rmse('boundary_rmse', 'boundary_rmse_sq_sum', 'boundary_rmse_count')
        _derive_rmse('high_rmse', 'high_rmse_sq_sum', 'high_rmse_count')

        # derive dist bins
        for k in list(out.keys()):
            if k.startswith('dist_mae_') and k.endswith('_sum'):
                base = k[:-4]
                c = out.get(base + '_count', 0.0)
                out[base] = float(out[k] / max(1e-12, c)) if c > 0 else float('nan')
            if k.startswith('dist_rmse_') and k.endswith('_sq_sum'):
                base = k[:-7]
                c = out.get(base + '_count', 0.0)
                out[base] = float(math.sqrt(out[k] / max(1e-12, c))) if c > 0 else float('nan')

        # [已删除] 静态方法无法访问 self.composite_weights，直接使用上面 mean_keys 计算出的平均 composite 即可
        # out['composite'] = composite_score(out, self.composite_weights) 
        
        return out