# -*- coding: utf-8 -*-
"""
rm_train_fno_protocol.py
============================================================
【彻底重构版】FNO 训练逻辑（Protocol-first）

你已经确定：
- 数据/通道：rm_protocol_dataset.py
- 评估：rm_protocol_eval.py

本脚本只做三件事：
1) 读取 TOTAL_CFG（或使用默认示例）
2) 构建 DataLoader（严格对齐 input.layout / target_norm）
3) 训练 FNO + 使用 ProtocolEvaluator 做统一评估与 best 选择

强调：不再依赖 rm_core / 旧训练脚本的任何函数。

------------------------------------------------------------
运行：
  python rm_train_fno_protocol.py

如果你有自己的总配置文件（推荐）：
  在同目录放一个 rm_total_config.py，里面定义 TOTAL_CFG = {...}
  本脚本会自动 import 并覆盖默认示例。

============================================================
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


import argparse
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import rm_ddp
import torch

# ---------------- AMP compatibility (torch 1.x / 2.x) ----------------
import contextlib
def make_grad_scaler(device_type: str = 'cuda', enabled: bool = True):
    """Create a GradScaler compatible with both torch>=2 (torch.amp) and torch<=1.x (torch.cuda.amp)."""
    try:
        # torch>=2.0
        return torch.amp.GradScaler(device_type, enabled=enabled)  # type: ignore[attr-defined]
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)

def autocast_ctx(device_type: str, dtype: torch.dtype, enabled: bool):
    """Autocast context compatible across torch versions."""
    if not enabled:
        return contextlib.nullcontext()
    if hasattr(torch, 'autocast'):
        try:
            return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)  # type: ignore[arg-type]
        except TypeError:
            return torch.autocast(device_type=device_type, enabled=enabled)  # type: ignore[misc]
    try:
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=enabled)
# ---------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from rm_protocol_dataset import RadiomapProtocolDataset, ensure_target_stats
from rm_protocol_eval import ProtocolEvaluator, build_boundary_ring, build_high_value_mask





# ============================================================
# FNO Model (merged from rm_models_fno.py)
# ============================================================

import contextlib
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _no_autocast_ctx(x: torch.Tensor):
    """
    关掉 autocast 的上下文（保证 FFT 相关运算在 float32/complex64 做）。
    """
    if not x.is_cuda:
        return contextlib.nullcontext()
    try:
        return torch.autocast(device_type="cuda", enabled=False)
    except Exception:
        return torch.cuda.amp.autocast(enabled=False)


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution (rfft2 -> complex multiply -> irfft2)
    - 权重使用 float32 参数（w1r/w1i/w2r/w2i）
    - 前向时 FFT 强制 float32 计算，输出 float32
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)

        scale = 0.02
        self.w1r = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.float32))
        self.w1i = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.float32))
        self.w2r = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.float32))
        self.w2i = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.float32))

    @staticmethod
    def _cmul2d(x_ft: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        a = x_ft.real
        b = x_ft.imag
        out_real = torch.einsum("bixy,ioxy->boxy", a, wr) - torch.einsum("bixy,ioxy->boxy", b, wi)
        out_imag = torch.einsum("bixy,ioxy->boxy", a, wi) + torch.einsum("bixy,ioxy->boxy", b, wr)
        return torch.complex(out_real, out_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)

        with _no_autocast_ctx(x):
            x_ft = torch.fft.rfft2(x.float(), dim=(-2, -1))  # complex64
            out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, device=x.device, dtype=torch.complex64)

            out_ft[:, :, :m1, :m2] = self._cmul2d(
                x_ft[:, :, :m1, :m2],
                self.w1r[:, :, :m1, :m2],
                self.w1i[:, :, :m1, :m2],
            )
            out_ft[:, :, -m1:, :m2] = self._cmul2d(
                x_ft[:, :, -m1:, :m2],
                self.w2r[:, :, :m1, :m2],
                self.w2i[:, :, :m1, :m2],
            )

            x_out = torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))  # float32
        return x_out


@dataclass
class FNO2dConfig:
    in_channels: int
    out_channels: int = 1
    width: int = 96
    modes1: int = 24
    modes2: int = 24
    n_layers: int = 4

    residual_learning: bool = True
    baseline_index: int = -1
    residual_clamp01: bool = True


class FNO2d(nn.Module):
    """
    标准 FNO2d：
      x -> 1x1 conv -> [spec + 1x1] * L -> 1x1 -> 1x1 -> out
    """
    def __init__(self, cfg: FNO2dConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = int(cfg.in_channels)
        self.out_channels = int(cfg.out_channels)
        self.residual_learning = bool(cfg.residual_learning)
        self.baseline_index = int(cfg.baseline_index)
        self.residual_clamp01 = bool(cfg.residual_clamp01)

        width = int(cfg.width)
        self.fc0 = nn.Conv2d(self.in_channels, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, cfg.modes1, cfg.modes2) for _ in range(int(cfg.n_layers))])
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(int(cfg.n_layers))])
        self.fc1 = nn.Conv2d(width, width, 1)
        self.fc2 = nn.Conv2d(width, self.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = None
        if self.residual_learning and self.baseline_index >= 0 and self.baseline_index < x.shape[1]:
            base = x[:, self.baseline_index:self.baseline_index + 1].float()

        h = self.fc0(x)
        for spec, w in zip(self.spec, self.w):
            xs = spec(h)                 # float32
            xw = w(h)                    # may be fp16/bf16 under autocast
            h = F.gelu(xs + xw.float())  # keep float32 state
        h = F.gelu(self.fc1(h).float())
        out = self.fc2(h.float())

        if base is not None:
            out = out + base
            if self.residual_clamp01:
                out = out.clamp(0.0, 1.0)
        return out


# ============================================================
# 0) Default TOTAL_CFG (示例；你会用自己的 rm_total_config.py 覆盖)
# ============================================================

DEFAULT_TOTAL_CFG: Dict[str, Any] = {
    "protocol": {
        "split": {
            "dataset_root": ".",
            "target_subdir": "gain/DPM",
            "buildings_subdir": "png/buildings_complete",
            "antenna_subdir": "antenna",
            "npy_channel": 0,
            "train_list_path": "splits/train_maps.txt",
            "val_list_path": "splits/val_maps.txt",
            "test_list_path": "splits/test_maps.txt",
            "max_tx_per_map": 80,
            "map_select": "first",
        },
        "input": {
            # ===== 主对比协议 P0+（5 通道）=====
            "layout": ["building", "tx_gaussian", "tx_invdist", "dy", "dx"],
            "construct": {
                "image_size": [256, 256],
                "tx_gaussian_sigma": 2.0,
                "tx_invdist_eps": 1.0,
            },
            "x_norm": {
                # 说明：这里默认不做 zscore（否则要提供 stats 文件）
                "mode": "none"
            }
        },
        "target_norm": {
            "y_space_train": "fno_target",
            "y_space_eval_main": "fno_target",
            "y_transform": "log1p+clip",     # none | log1p | clip | log1p+clip
            "y_clip_range": None,           # 若为 None 且 transform 需要 clip，则在训练启动时自动估计
            "y_norm": "global_minmax",      # none | global_minmax | zscore
        },
        "eval": {
            "compare_space": "fno_target",
            "mask_buildings": True,
            "boundary_width": 3,
            "high_q": 0.90,
            "post_clamp01": False,
            "resize_pred_to_gt": True,
            "resize_mode": "bilinear",
            "composite_weights": {"rmse": 1.0, "boundary_rmse": 0.5, "high_rmse": 0.5},
            "error_percentiles": {"enabled": True, "percentiles": [50, 90, 99], "sample_per_image": 800},
            "dist_bins": {"enabled": True, "bins": [0, 1, 2, 4, 8, 16, 32, 64, 1e9]},
        },
    },

    "train": {
        # ---------- runtime ----------
        "device": "cuda",
        "seed": 123,
        "epochs": 60,
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,

        # ---------- optimization ----------
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "amp": True,
        "amp_dtype": "fp16",          # fp16 | bf16
        "grad_accum": 1,
        "grad_clip": 0.0,

        # ---------- loss ----------
        "loss_type": "huber",         # l1 | mse | huber
        "huber_delta": 0.05,
        "loss_mask_buildings": True,  # loss 只在 outdoor 计算
        "loss_weight_boundary": 0.0,  # 建议先 0 做 aligned baseline
        "boundary_width": 3,
        "loss_weight_high": 0.0,
        "high_q": 0.90,
        "tv_weight": 0.0,
        "radial_weight": 0.0,
        "radial_corr_target": 0.2,

        # ---------- save / eval ----------
        "out_dir": "./rm_runs_fno_protocol",
        "run_name": "fno_protocol",
        "save_every": 1,
        "save_best_by": "composite",  # composite | rmse | mae | boundary_rmse | high_rmse
        "log_every_steps": 50,
        "resume_ckpt": None,          # 断点续训 ckpt 路径
    },

    "model": {
        "modes": 24,
        "width": 96,
        "layers": 4,
        "residual_learning": False,
        "baseline_index": "auto",   # auto -> 用 input.layout 里 tx_invdist 的位置
        "residual_clamp01": True,
    },
}


def load_total_cfg(total_cfg_json: Optional[str] = None) -> Dict[str, Any]:
    """Load TOTAL_CFG.

    Priority:
      1) CLI: --total_cfg_json
      2) env: RM_TOTAL_CFG_JSON / TOTAL_CFG_JSON
      3) rm_total_config.py: TOTAL_CFG
      4) DEFAULT_TOTAL_CFG
    """
    json_path = total_cfg_json or os.environ.get("RM_TOTAL_CFG_JSON") or os.environ.get("TOTAL_CFG_JSON")
    if json_path:
        p = resolve_path(str(json_path))
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[Config] Failed to load json: {p} | {e}")
        else:
            print(f"[Config] total_cfg_json not found: {p}")

    try:
        from rm_total_config import TOTAL_CFG  # type: ignore
        return deepcopy(TOTAL_CFG)
    except Exception:
        return deepcopy(DEFAULT_TOTAL_CFG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_cfg_json", type=str, default=None, help="Path to a TOTAL_CFG json snapshot.")
    return parser.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    import random, numpy as np
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(p: str) -> Path:
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (script_dir() / pp)


# ============================================================
# 2) Auto target clip estimation (when y_transform contains 'clip' and y_clip_range is None)
# ============================================================

@torch.no_grad()
def estimate_target_clip_percentiles(proto: Dict[str, Any],
                                     percentiles: Tuple[float, float] = (1.0, 99.0),
                                     max_batches: int = 200,
                                     pixels_per_batch: int = 8192,
                                     seed: int = 123) -> Tuple[float, float]:
    """
    在“未 clip/未 norm”的 target space 上，估计 (lo,hi)：
      - 先按 y_transform（仅 log1p / none）做变换
      - 再对像素采样做分位数
    """
    import numpy as np

    # 复制协议，禁用 clip/norm，避免循环依赖
    p = deepcopy(proto)
    tn = p.get("target_norm", {})
    y_transform = str(tn.get("y_transform", "none")).lower()
    # 只保留 log 部分
    if "log1p" in y_transform:
        xform = "log1p"
    else:
        xform = "none"
    tn["y_transform"] = xform
    tn["y_norm"] = "none"
    tn["y_clip_range"] = None
    p["target_norm"] = tn

    ds = RadiomapProtocolDataset(p, split="train", strict_layout=True)
    # 这里不依赖 train_cfg（proto 内通常不包含训练超参），clip 估计用一个小 batch 即可
    bs = 8
    loader = DataLoader(ds, batch_size=min(16, bs), shuffle=True, num_workers=0)

    rng = np.random.default_rng(seed)
    samples = []

    lo_p, hi_p = float(percentiles[0]), float(percentiles[1])

    for bi, (_x, y, _m) in enumerate(loader):
        if bi >= int(max_batches):
            break
        y = y.float().view(-1)
        if y.numel() == 0:
            continue
        k = min(int(pixels_per_batch), y.numel())
        idx = torch.from_numpy(rng.integers(0, y.numel(), size=(k,), dtype=np.int64))
        take = y[idx].detach().cpu().numpy().astype(np.float32)
        samples.append(take)

    if not samples:
        return 0.0, 1.0

    allv = np.concatenate(samples, axis=0)
    lo = float(np.percentile(allv, lo_p))
    hi = float(np.percentile(allv, hi_p))
    if abs(hi - lo) < 1e-8:
        hi = lo + 1e-6
    return lo, hi


def _get(d: Dict[str, Any], path: str, fallback: Any = None, *, strict: bool = False) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            if strict:
                raise KeyError(path)
            return fallback
        cur = cur[k]
    return cur


# ============================================================
# 3) Loss functions (对齐旧逻辑，但不依赖旧代码)
# ============================================================

def morph_boundary_ring(building: torch.Tensor, width: int) -> torch.Tensor:
    """
    building: (B,1,H,W) float {0,1} (1=building)
    return ring: (B,1,H,W) float {0,1}
    """
    width = max(1, int(width))
    k = 2 * width + 1
    dil = F.max_pool2d(building, kernel_size=k, stride=1, padding=width)
    ero = 1.0 - F.max_pool2d(1.0 - building, kernel_size=k, stride=1, padding=width)
    ring = (dil - ero).clamp(0.0, 1.0)
    return (ring > 0.0).float()


def pearson_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    a,b,mask: (B,1,H,W) -> returns (B,) corr
    """
    B = a.shape[0]
    m = mask.view(B, -1)
    aa = a.view(B, -1)
    bb = b.view(B, -1)

    wsum = m.sum(dim=1, keepdim=True).clamp_min(1e-8)
    am = (aa * m).sum(dim=1, keepdim=True) / wsum
    bm = (bb * m).sum(dim=1, keepdim=True) / wsum

    ac = (aa - am) * m
    bc = (bb - bm) * m

    cov = (ac * bc).sum(dim=1) / wsum.squeeze(1).clamp_min(1e-8)
    va = (ac ** 2).sum(dim=1) / wsum.squeeze(1).clamp_min(1e-8)
    vb = (bc ** 2).sum(dim=1) / wsum.squeeze(1).clamp_min(1e-8)

    corr = cov / (va.sqrt() * vb.sqrt() + 1e-8)
    return corr


def tv_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/mask: (B,1,H,W)
    """
    dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs()
    dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs()
    my = mask[:, :, 1:, :]
    mx = mask[:, :, :, 1:]
    tv = (dy * my).sum() / (my.sum() + 1e-8) + (dx * mx).sum() / (mx.sum() + 1e-8)
    return tv


def weighted_supervised_loss(pred: torch.Tensor,
                             target: torch.Tensor,
                             building: torch.Tensor,
                             *,
                             loss_type: str,
                             huber_delta: float,
                             mask_buildings: bool,
                             w_boundary: float,
                             boundary_width: int,
                             w_high: float,
                             high_q: float) -> torch.Tensor:
    """
    对齐旧脚本：loss_map * weight_map / sum(weight_map)
    [Fix] High-mask 逻辑对齐 ProtocolEvaluator：在 domain_mask 范围内算分位数（domain-aware quantile）
    """
    assert target.shape[1] == 1, "Expect single-channel target"
    w = torch.ones_like(target)

    bld = building.clamp(0.0, 1.0)

    # 1) Boundary ring
    if w_boundary and w_boundary > 0:
        ring = morph_boundary_ring(bld, width=boundary_width)  # float (B,1,H,W)
        w = w + float(w_boundary) * ring

    # 2) Domain mask（outdoor 或全图）
    if mask_buildings:
        domain_mask = (1.0 - bld).clamp(0.0, 1.0)
    else:
        domain_mask = torch.ones_like(target)

    # 3) High-value region（domain-aware）
    if w_high and w_high > 0:
        with torch.no_grad():
            high_mask = build_high_value_mask(target, domain_mask, q=high_q).float()  # (B,1,H,W)
        w = w + float(w_high) * high_mask

    # 4) Apply domain to total weights
    if mask_buildings:
        w = w * domain_mask

    # 5) Loss map
    diff = pred - target
    lt = str(loss_type).lower()
    if lt == "l1":
        loss_map = diff.abs()
    elif lt == "mse":
        loss_map = diff ** 2
    else:
        absd = diff.abs()
        delta = float(huber_delta)
        loss_map = torch.where(absd <= delta, 0.5 * diff ** 2, delta * (absd - 0.5 * delta))

    # 6) Weighted mean
    return (loss_map * w).sum() / (w.sum() + 1e-8)


# ============================================================
# 4) Train / Eval loops
# ============================================================

def build_loaders(
    proto: Dict[str, Any],
    train_cfg: Dict[str, Any],
    *,
    is_ddp: bool = False,
    rank0_only_val: bool = True,
    worker_init_fn=None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DistributedSampler]]:
    ds_train = RadiomapProtocolDataset(proto, split="train", strict_layout=True)

    ds_val = None
    if (not rank0_only_val) or rm_ddp.is_master():
        ds_val = RadiomapProtocolDataset(proto, split="val", strict_layout=True)

    bs = int(train_cfg.get("batch_size", 8))   # per-GPU
    nw = int(train_cfg.get("num_workers", 4))
    pin = bool(train_cfg.get("pin_memory", True))

    train_sampler = None
    shuffle = True
    if is_ddp:
        train_sampler = DistributedSampler(ds_train, shuffle=True, drop_last=True)
        shuffle = False

    train_loader = DataLoader(
        ds_train,
        batch_size=bs,
        shuffle=shuffle,
        sampler=train_sampler,
        drop_last=True,
        num_workers=nw,
        pin_memory=pin,
        worker_init_fn=worker_init_fn or seed_worker,
        persistent_workers=(nw > 0),
    )

    val_loader = None
    if ds_val is not None:
        val_loader = DataLoader(
            ds_val,
            batch_size=bs,
            shuffle=False,
            drop_last=False,
            num_workers=nw,
            pin_memory=pin,
            worker_init_fn=worker_init_fn or seed_worker,
            persistent_workers=(nw > 0),
        )

    return train_loader, val_loader, train_sampler
def build_model(proto: Dict[str, Any], model_cfg: Dict[str, Any]) -> FNO2d:
    layout = list(_get(proto, "input.layout", fallback=[]))
    in_ch = len(layout)

    baseline_index = model_cfg.get("baseline_index", "auto")
    if baseline_index == "auto":
        baseline_index = layout.index("tx_invdist") if "tx_invdist" in layout else -1

    cfg = FNO2dConfig(
        in_channels=in_ch,
        out_channels=1,
        width=int(model_cfg.get("width", 96)),
        modes1=int(model_cfg.get("modes", 24)),
        modes2=int(model_cfg.get("modes", 24)),
        n_layers=int(model_cfg.get("layers", 4)),
        residual_learning=bool(model_cfg.get("residual_learning", False)),
        baseline_index=int(baseline_index),
        residual_clamp01=bool(model_cfg.get("residual_clamp01", True)),
    )
    return FNO2d(cfg)


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optim: torch.optim.Optimizer,
                    device: torch.device,
                    *,
                    building_idx: int,
                    amp: bool,
                    amp_dtype: str,
                    scaler: Optional[torch.cuda.amp.GradScaler],
                    grad_accum: int,
                    grad_clip: float,
                    loss_cfg: Dict[str, Any]) -> float:
    model.train()
    grad_accum = max(1, int(grad_accum))
    optim.zero_grad(set_to_none=True)

    use_amp = bool(amp and device.type == "cuda")
    use_fp16_scaler = bool(use_amp and scaler is not None and getattr(scaler, "is_enabled", lambda: False)())
    dtype = torch.float16 if str(amp_dtype).lower() == "fp16" else torch.bfloat16

    total = 0.0
    n = 0

    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc="train", leave=False)

    for step, (x, y, meta) in enumerate(it, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        building = x[:, building_idx:building_idx+1].float()
        building = (building > 0.5).float()

        with autocast_ctx(device.type, dtype=dtype, enabled=use_amp):
            pred = model(x)

            sup = weighted_supervised_loss(
                pred, y, building,
                loss_type=str(loss_cfg.get("loss_type", "huber")),
                huber_delta=float(loss_cfg.get("huber_delta", 0.05)),
                mask_buildings=bool(loss_cfg.get("loss_mask_buildings", True)),
                w_boundary=float(loss_cfg.get("loss_weight_boundary", 0.0)),
                boundary_width=int(loss_cfg.get("boundary_width", 3)),
                w_high=float(loss_cfg.get("loss_weight_high", 0.0)),
                high_q=float(loss_cfg.get("high_q", 0.90)),
            )

            loss = sup

            tv_w = float(loss_cfg.get("tv_weight", 0.0))
            if tv_w and tv_w > 0:
                free = (1.0 - building).clamp(0.0, 1.0)
                loss = loss + tv_w * tv_loss(pred, free)

            radial_w = float(loss_cfg.get("radial_weight", 0.0))
            if radial_w and radial_w > 0:
                # baseline invdist channel index: prefer meta["baseline_idx"] else search layout
                base_idx = -1
                try:
                    bi = meta.get("baseline_idx", -1)
                    if isinstance(bi, (list, tuple)):
                        base_idx = int(bi[0])
                    elif torch.is_tensor(bi):
                        base_idx = int(bi[0].item())
                    else:
                        base_idx = int(bi)
                except Exception:
                    base_idx = -1

                if 0 <= base_idx < x.shape[1]:
                    inv = x[:, base_idx:base_idx+1].float()
                    domain = (1.0 - building).clamp(0.0, 1.0) if bool(loss_cfg.get("loss_mask_buildings", True)) else torch.ones_like(building)
                    corr = pearson_corr(pred, inv, domain)
                    target_corr = float(loss_cfg.get("radial_corr_target", 0.2))
                    pen = F.relu(target_corr - corr).mean()
                    loss = loss + radial_w * pen

            loss = loss / float(grad_accum)

        if use_fp16_scaler:
            scaler.scale(loss).backward()  # type: ignore
            if step % grad_accum == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optim)  # type: ignore
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                scaler.step(optim)  # type: ignore
                scaler.update()     # type: ignore
                optim.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if step % grad_accum == 0:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optim.step()
                optim.zero_grad(set_to_none=True)

        total += float(loss.item()) * grad_accum
        n += 1

    # flush remaining
    if len(loader) % grad_accum != 0:
        if use_fp16_scaler:
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)  # type: ignore
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            scaler.step(optim)  # type: ignore
            scaler.update()     # type: ignore
        else:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optim.step()
        optim.zero_grad(set_to_none=True)

    return total / max(1, n)


@torch.no_grad()
def eval_one_epoch(model: nn.Module,
                   loader: DataLoader,
                   evaluator: ProtocolEvaluator,
                   device: torch.device) -> Dict[str, float]:
    model.eval()
    batch_metrics: List[Dict[str, float]] = []

    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc="val", leave=False)

    for x, y, meta in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        m = evaluator.compute_batch(pred, y, x)
        batch_metrics.append(m)

    return ProtocolEvaluator.reduce_epoch(batch_metrics)


# ============================================================
# 5) Checkpoint I/O
# ============================================================

def save_ckpt(path: Path,
              *,
              epoch: int,
              model: nn.Module,
              optim: torch.optim.Optimizer,
              scaler: Optional[torch.cuda.amp.GradScaler],
              total_cfg: Dict[str, Any],
              model_cfg: Dict[str, Any],
              metrics: Dict[str, float],
              history: List[Dict[str, Any]]):
    obj = {
        "epoch": int(epoch),
        "total_cfg": total_cfg,
        "model_cfg": model_cfg,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
        "metrics": metrics,
        "history": history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def try_load_ckpt(path: Optional[str], device: torch.device):
    if not path:
        return None
    p = resolve_path(path)
    if not p.is_file():
        return None
    obj = torch.load(str(p), map_location=device)
    return obj


# ============================================================
# 6) Main
# ============================================================

def main():
    # -----------------------------
    # DDP setup (torchrun compatible)
    # -----------------------------
    ddp_info = rm_ddp.setup_ddp(verbose=rm_ddp.is_master())
    is_ddp = bool(getattr(ddp_info, "is_ddp", False))
    local_rank = int(getattr(ddp_info, "local_rank", 0))
    ddp_device = getattr(ddp_info, "device", None)

    # tqdm: only rank0 shows
    global tqdm
    if is_ddp and (not rm_ddp.is_master()):
        tqdm = None

    args = parse_args()
    total = load_total_cfg(args.total_cfg_json)
    proto = deepcopy(total["protocol"])
    train_cfg = total["train"]
    model_cfg = total.get("model_fno", None) or total.get("model", {})

    # device
    if is_ddp:
        if ddp_device is None:
            raise RuntimeError("[DDP] setup_ddp did not return a device")
        device = ddp_device
    else:
        device = get_device(str(train_cfg.get("device", "cuda")))

    # seed: keep torch init consistent across ranks first
    base_seed = int(train_cfg.get("seed", 123))
    rm_ddp.seed_everything(base_seed, torch_same_across_ranks=True, add_rank_to_numpy_python=False)

    # ---- resolve dataset_root absolute (avoid CWD surprises) ----
    ds_root = resolve_path(str(_get(proto, "split.dataset_root", fallback=".")))
    proto["split"]["dataset_root"] = str(ds_root)

    # ---- AutoClip: only master computes then broadcast ----
    tn = proto.get("target_norm", {}) or {}
    y_transform = str(tn.get("y_transform", "none")).lower()
    needs_clip = ("clip" in y_transform)

    if needs_clip and tn.get("y_clip_range", None) is None:
        clip_range = None
        if rm_ddp.is_master():
            lo, hi = estimate_target_clip_percentiles(proto, percentiles=(1.0, 99.0), seed=base_seed)
            clip_range = [float(lo), float(hi)]
            print(f"[AutoClip] y_clip_range inferred as [{lo:.6g}, {hi:.6g}] (percentile 1~99)")
        if is_ddp:
            clip_range = rm_ddp.broadcast_object(clip_range, src=0)
        tn["y_clip_range"] = clip_range
        proto["target_norm"] = tn

    # now offset python/numpy RNG across ranks
    rm_ddp.seed_everything(base_seed, torch_same_across_ranks=True, add_rank_to_numpy_python=True)

    # worker init
    worker_init_fn = rm_ddp.build_worker_init_fn(base_seed)

    # ---- output dirs (rank0 decides, broadcast) ----
    out_dir = resolve_path(str(train_cfg.get("out_dir", "./rm_runs_fno_protocol")))
    run_name = str(train_cfg.get("run_name", "fno_protocol")).strip() or "fno_protocol"

    run_dir_str = None
    if rm_ddp.is_master():
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        run_dir_str = str(out_dir / f"{run_name}_{ts}")
    if is_ddp:
        run_dir_str = rm_ddp.broadcast_object(run_dir_str, src=0)
    assert run_dir_str is not None
    run_dir = Path(run_dir_str)
    ckpt_dir = run_dir / "ckpts"

    if rm_ddp.is_master():
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "total_cfg.json").write_text(json.dumps(total, indent=2, ensure_ascii=False), encoding="utf-8")

    rm_ddp.barrier()

    # ---- loaders ----
    train_loader, val_loader, train_sampler = build_loaders(
        proto,
        train_cfg,
        is_ddp=is_ddp,
        rank0_only_val=True,
        worker_init_fn=worker_init_fn,
    )

    # ---- evaluator (rank0 only) ----
    evaluator = ProtocolEvaluator(proto) if rm_ddp.is_master() else None

    # ---- model ----
    model = build_model(proto, model_cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # ---- optim / amp ----
    optim = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    use_amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    amp_dtype = str(train_cfg.get("amp_dtype", "fp16")).lower()
    scaler = make_grad_scaler('cuda', enabled=bool(use_amp and amp_dtype == "fp16"))

    # ---- resume ----
    history: List[Dict[str, Any]] = []
    start_epoch = 1
    best_val = float("inf")
    best_epoch = -1

    resume_obj = try_load_ckpt(train_cfg.get("resume_ckpt", None), device=device)
    if resume_obj is not None:
        try:
            state_model = resume_obj.get("model", resume_obj.get("model_state", None))
            state_optim = resume_obj.get("optim", resume_obj.get("optim_state", None))
            if state_model is not None:
                (model.module if is_ddp else model).load_state_dict(state_model, strict=True)
            if state_optim is not None:
                optim.load_state_dict(state_optim)
            sc = resume_obj.get("scaler", resume_obj.get("scaler_state", None))
            if scaler is not None and sc is not None:
                scaler.load_state_dict(sc)
            start_epoch = int(resume_obj.get("epoch", 0)) + 1
            history = list(resume_obj.get("history", []))
            if rm_ddp.is_master():
                print(f"[Resume] loaded {train_cfg.get('resume_ckpt')} -> start_epoch={start_epoch}")
        except Exception as e:
            if rm_ddp.is_master():
                print(f"[Resume] failed to load ckpt: {e}")

    # ---- training ----
    layout = list(_get(proto, "input.layout", fallback=[]))
    building_idx = layout.index("building")

    epochs = int(train_cfg.get("epochs", 60))
    save_every = int(train_cfg.get("save_every", 1))
    save_best_by = str(train_cfg.get("save_best_by", "composite")).lower().strip()

    if rm_ddp.is_master():
        bs = int(train_cfg.get("batch_size", 8))
        ws = rm_ddp.get_world_size()
        print(f"[Start] FNO | DDP={is_ddp} | world_size={ws} | bs_per_gpu={bs} | global_bs={bs*ws}")

    for ep in range(start_epoch, epochs + 1):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(ep)

        tr_loss = train_one_epoch(
            model,
            train_loader,
            optim,
            device,
            building_idx=building_idx,
            amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            grad_accum=int(train_cfg.get("grad_accum", 1)),
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            loss_cfg=train_cfg,
        )

        # reduce train loss for consistent logging
        if is_ddp:
            tr_loss_t = torch.tensor([float(tr_loss)], device=device, dtype=torch.float32)
            tr_loss = float(rm_ddp.all_reduce_mean(tr_loss_t)[0].item())

        rm_ddp.barrier()

        # val + io: rank0 only
        if rm_ddp.is_master():
            assert evaluator is not None
            assert val_loader is not None

            val_metrics = eval_one_epoch(model.module if is_ddp else model, val_loader, evaluator, device)
            val_key = float(val_metrics.get(save_best_by, float("inf")))

            rec = {"epoch": ep, "train_loss": float(tr_loss)}
            rec.update({str(k): float(v) for k, v in val_metrics.items()})
            history.append(rec)

            print(f"[epoch {ep:03d}] train_loss={tr_loss:.6f} | " +
                  ", ".join([f"{k}={float(v):.6f}" for k, v in val_metrics.items()]))

            # save periodic
            if save_every > 0 and (ep % save_every == 0):
                save_ckpt(
                    ckpt_dir / f"epoch_{ep:03d}.pt",
                    epoch=ep,
                    model=(model.module if is_ddp else model),
                    optim=optim,
                    scaler=scaler,
                    total_cfg=total,
                    model_cfg=model_cfg,
                    metrics=val_metrics,
                    history=history[-200:],
                )

            # save best
            is_better = (val_key < best_val)
            if is_better:
                best_val = float(val_key)
                best_epoch = int(ep)
                save_ckpt(
                    ckpt_dir / "best.pt",
                    epoch=ep,
                    model=(model.module if is_ddp else model),
                    optim=optim,
                    scaler=scaler,
                    total_cfg=total,
                    model_cfg=model_cfg,
                    metrics=val_metrics,
                    history=history,
                )
                print(f"[best] saved: {ckpt_dir / 'best.pt'} | {save_best_by}={best_val:.6f}")

            # write history
            (run_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
            (run_dir / "best.json").write_text(json.dumps({"epoch": best_epoch, save_best_by: best_val}, indent=2), encoding="utf-8")

        rm_ddp.barrier()

    rm_ddp.cleanup_ddp()


if __name__ == "__main__":
    main()