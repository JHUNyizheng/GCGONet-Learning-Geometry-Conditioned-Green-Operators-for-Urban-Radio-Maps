# -*- coding: utf-8 -*-
"""
rm_train_unet_protocol_allinone.py
============================================================
【彻底重构版】UNet 训练逻辑（Protocol-first, All-in-one）

依赖：
- rm_protocol_dataset.py   -> 统一输入/通道/目标变换
- rm_protocol_eval.py      -> 统一评估出口（MAE/RMSE/Boundary/High/Percentiles/DistBins/Composite）

特点：
- 不在旧代码上打补丁；训练/评估/保存逻辑完全重写
- 与 FNO 的 protocol/评估/输出结构一致，便于三模型公平对比
- loss 可选：outdoor mask、boundary/high 加权、TV 正则、radial（可关）
- AMP + grad_accum + grad_clip
- 自动估计 y_clip_range（当 y_transform 含 clip 且未提供时）

运行：
  python rm_train_unet_protocol_allinone.py

配置：
- 优先加载 rm_total_config.py 中的 TOTAL_CFG
- 若没有该文件/导入失败，则使用脚本内 DEFAULT_TOTAL_CFG（示例）

============================================================
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
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
try:
    from rm_unet_official import RadioUNetOfficial
except Exception:
    RadioUNetOfficial = None  # type: ignore
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = None
from rm_protocol_dataset import RadiomapProtocolDataset, ModelInputAdapter
from rm_protocol_eval import ProtocolEvaluator, build_boundary_ring, build_high_value_mask


# ============================================================
# 0) Utils
# ============================================================

def _get(d: Dict[str, Any], path: str, fallback: Any = None, *, strict: bool = False) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            if strict:
                raise KeyError(path)
            return fallback
        cur = cur[k]
    return cur


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(p: str) -> Path:
    pp = Path(p).expanduser()
    return pp if pp.is_absolute() else (script_dir() / pp)


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


# ============================================================
# 1) Default TOTAL_CFG (示例；建议用 rm_total_config.py 覆盖)
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
            "layout": ["building", "tx_gaussian", "tx_invdist", "dy", "dx"],
            "construct": {
                "image_size": [256, 256],
                "tx_gaussian_sigma": 2.0,
                "tx_invdist_eps": 1.0,
            },
            "x_norm": {"mode": "none", "x_stats_path": None},
        },
        "target_norm": {
            "y_space_train": "fno_target",
            "y_space_eval_main": "fno_target",
            "y_transform": "log1p+clip",
            "y_clip_range": None,          # None -> 自动估计
            "y_norm": "global_minmax",
            "y_stats_path": None,
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
        "device": "cuda",
        "seed": 123,
        "epochs": 80,
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "amp": True,
        "amp_dtype": "fp16",     # fp16 | bf16
        "grad_accum": 1,
        "grad_clip": 0.0,

        # loss
        "loss_type": "huber",    # l1 | mse | huber
        "huber_delta": 0.05,
        "loss_mask_buildings": True,
        "loss_weight_boundary": 0.0,
        "boundary_width": 3,
        "loss_weight_high": 0.0,
        "high_q": 0.90,
        "tv_weight": 0.0,

        # save
        "out_dir": "./rm_runs_unet_protocol",
        "run_name": "unet_protocol",
        "save_every": 1,
        "save_best_by": "composite",  # composite | rmse | mae | boundary_rmse | high_rmse
        "resume_ckpt": None,
    },

    # UNet 专属
    "model_unet": {
        "base_channels": 48,      # 初始通道数
        "depth": 4,               # 下采样次数（4 -> 1/16）
        "norm": "bn",             # bn | gn | none
        "gn_groups": 8,
        "dropout": 0.0,
        "use_transpose": False,   # False -> bilinear upsample + conv
        "act": "gelu",            # relu | gelu | silu
        "out_clamp01": False,     # 如果你强制 y∈[0,1]，可开
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


def estimate_target_clip_percentiles(proto: Dict[str, Any],
                                     percentiles: Tuple[float, float] = (1.0, 99.0),
                                     max_batches: int = 200,
                                     pixels_per_batch: int = 8192,
                                     seed: int = 123) -> Tuple[float, float]:
    """
    在“未 clip/未 norm”的 target space 上估计 (lo,hi)：
    - 若 y_transform 含 log1p，则保留 log1p；否则 none
    - 禁用 clip 和 norm
    """
    import numpy as np

    p = deepcopy(proto)
    tn = p.get("target_norm", {}) or {}
    y_transform = str(tn.get("y_transform", "none")).lower()
    tn["y_transform"] = "log1p" if ("log1p" in y_transform) else "none"
    tn["y_clip_range"] = None
    tn["y_norm"] = "none"
    p["target_norm"] = tn

    ds = RadiomapProtocolDataset(p, split="train", strict_layout=True)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

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


# ============================================================
# 3) Loss helpers (与 FNO 脚本保持一致口径)
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


def tv_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
    outdoor = (bld < 0.5).float()

    # 1) Boundary ring
    if w_boundary and w_boundary > 0:
        ring = build_boundary_ring(bld, width=boundary_width)  # float (B,1,H,W)
        if mask_buildings:
            ring = ring * outdoor
        w = w + float(w_boundary) * ring

    # 2) Domain mask（outdoor 或全图）
    if mask_buildings:
        domain_mask = outdoor
    else:
        domain_mask = torch.ones_like(target)

    # 3) High-value region（domain-aware）
    if w_high and w_high > 0:
        with torch.no_grad():
            high_mask = build_high_value_mask(target, domain_mask, q=high_q).float()
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
# 4) UNet model (All-in-one)
# ============================================================

def _act(name: str) -> nn.Module:
    n = str(name).lower()
    if n == "relu":
        return nn.ReLU(inplace=True)
    if n == "silu":
        return nn.SiLU(inplace=True)
    return nn.GELU()


def _norm(kind: str, ch: int, gn_groups: int = 8) -> nn.Module:
    k = str(kind).lower()
    if k == "bn":
        return nn.BatchNorm2d(ch)
    if k == "gn":
        g = max(1, min(int(gn_groups), ch))
        return nn.GroupNorm(num_groups=g, num_channels=ch)
    return nn.Identity()


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str, gn_groups: int, dropout: float, act: str):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.n1 = _norm(norm, out_ch, gn_groups)
        self.a1 = _act(act)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.n2 = _norm(norm, out_ch, gn_groups)
        self.a2 = _act(act)
        self.dp = nn.Dropout2d(p=float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.n1(self.c1(x)))
        x = self.dp(x)
        x = self.a2(self.n2(self.c2(x)))
        return x


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str, gn_groups: int, dropout: float, act: str):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch, norm, gn_groups, dropout, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str, gn_groups: int, dropout: float, act: str, use_transpose: bool):
        super().__init__()
        self.use_transpose = bool(use_transpose)
        if self.use_transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            up_out = in_ch // 2
        else:
            self.up = nn.Identity()
            up_out = in_ch

        self.block = ConvBlock(up_out + skip_ch, out_ch, norm, gn_groups, dropout, act)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.use_transpose:
            x = self.up(x)
        else:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        # pad if needed
        diff_y = skip.size(-2) - x.size(-2)
        diff_x = skip.size(-1) - x.size(-1)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.block(x)


@dataclass
class UNetConfig:
    in_channels: int
    base_channels: int = 48
    depth: int = 4
    norm: str = "bn"     # bn|gn|none
    gn_groups: int = 8
    dropout: float = 0.0
    use_transpose: bool = False
    act: str = "gelu"
    out_channels: int = 1
    out_clamp01: bool = False


class UNet2d(nn.Module):
    def __init__(self, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        c0 = int(cfg.base_channels)
        d = int(cfg.depth)

        self.inc = ConvBlock(cfg.in_channels, c0, cfg.norm, cfg.gn_groups, cfg.dropout, cfg.act)

        # encoder
        downs = []
        chs = [c0]
        for i in range(d):
            downs.append(Down(chs[-1], chs[-1] * 2, cfg.norm, cfg.gn_groups, cfg.dropout, cfg.act))
            chs.append(chs[-1] * 2)
        self.downs = nn.ModuleList(downs)

        # bottleneck (extra conv block)
        self.mid = ConvBlock(chs[-1], chs[-1], cfg.norm, cfg.gn_groups, cfg.dropout, cfg.act)

        # decoder
        ups = []
        for i in range(d):
            in_ch = chs[-1]
            skip_ch = chs[-2]
            out_ch = skip_ch
            ups.append(Up(in_ch, skip_ch, out_ch, cfg.norm, cfg.gn_groups, cfg.dropout, cfg.act, cfg.use_transpose))
            chs.pop()  # consume
        self.ups = nn.ModuleList(ups)

        self.outc = nn.Conv2d(c0, cfg.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.mid(x)

        # last skip corresponds to deepest feature; we don't use it as skip for first up (common UNet uses pre-down skips)
        # We'll pop last (deepest) and use remaining for ups.
        skips.pop()  # remove deepest
        for up in self.ups:
            skip = skips.pop()
            x = up(x, skip)
        out = self.outc(x)
        if self.cfg.out_clamp01:
            out = out.clamp(0.0, 1.0)
        return out


# ============================================================
# 5) Data / Model builders
# ============================================================

def build_loaders(
    proto: Dict[str, Any],
    train_cfg: Dict[str, Any],
    *,
    is_ddp: bool = False,
    rank0_only_val: bool = True,
    worker_init_fn=None,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DistributedSampler]]:
    """DDP-friendly dataloaders.

    - train: DistributedSampler (shuffle handled by sampler), drop_last=True
    - val: rank0-only by default (keeps metrics behavior identical to single-gpu)
    """
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
def build_model(proto: Dict[str, Any], model_cfg: Dict[str, Any]) -> nn.Module:
    """Build UNet.

    当前工程默认对比：FNO vs Official RadioUNet vs GCGO。
    - official：严格复刻 RadioUNet（单 U），输入必须是 2 通道（building + tx_gaussian）。

    你也可以通过 model_unet.type 显式切换（预留接口），但若不是 official 将直接报错，避免口径漂移。
    """
    model_type = str(model_cfg.get("type", "official")).lower().strip()

    roles = (proto.get("input", {}) or {}).get("roles", {}) or {}
    unet_roles = roles.get("unet", ["building", "tx_gaussian"]) or ["building", "tx_gaussian"]
    unet_roles = list(unet_roles)

    # 严格保证与官方一致：通道必须且只能是这两个，并且顺序固定
    if unet_roles != ["building", "tx_gaussian"]:
        raise ValueError(
            f"Official UNet requires roles.unet exactly ['building','tx_gaussian'], got {unet_roles}"
        )

    inputs = int(model_cfg.get("inputs", 2))
    if inputs != 2:
        raise ValueError(f"Official UNet requires inputs=2, got {inputs}")

    if model_type != "official":
        raise ValueError(
            f"This project is locked to official UNet baseline for fair comparison; got model_unet.type={model_type!r}"
        )

    if RadioUNetOfficial is None:
        raise ImportError("Cannot import RadioUNetOfficial from rm_unet_official.py")

    return RadioUNetOfficial(inputs=2)



# ============================================================
# 6) Train / Eval loops
# ============================================================

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optim: torch.optim.Optimizer,
                    device: torch.device,
                    adapter: ModelInputAdapter,
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
    dtype = torch.float16 if str(amp_dtype).lower() == "fp16" else torch.bfloat16
    use_fp16_scaler = bool(use_amp and scaler is not None and getattr(scaler, "is_enabled", lambda: False)())

    total = 0.0
    n = 0

    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc="train", leave=False)

    for step, (x, y, _meta) in enumerate(it, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        building = x[:, building_idx:building_idx+1].float()
        building = (building > 0.5).float()

        x_u = adapter.make_model_inputs("unet", x, _meta)

        with autocast_ctx(device.type, dtype=dtype, enabled=use_amp):
            pred = model(x_u)

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
                free = (1.0 - building).clamp(0.0, 1.0) if bool(loss_cfg.get("loss_mask_buildings", True)) else torch.ones_like(building)
                loss = loss + tv_w * tv_loss(pred, free)

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
                   device: torch.device,
                   adapter: ModelInputAdapter) -> Dict[str, float]:
    model.eval()
    batch_metrics: List[Dict[str, float]] = []

    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc="val", leave=False)

    for x, y, _meta in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_u = adapter.make_model_inputs("unet", x, _meta)
        pred = model(x_u)
        m = evaluator.compute_batch(pred, y, x)
        batch_metrics.append(m)

    return evaluator.reduce_epoch(batch_metrics)


# ============================================================
# 7) Checkpoint I/O
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
    return torch.load(str(p), map_location=device)


# ============================================================
# 8) Main
# ============================================================

def main():
    ddp_info = rm_ddp.setup_ddp(verbose=rm_ddp.is_master())
    is_ddp = bool(getattr(ddp_info, "is_ddp", False))
    local_rank = int(getattr(ddp_info, "local_rank", 0))
    ddp_device = getattr(ddp_info, "device", None)

    global tqdm
    if is_ddp and (not rm_ddp.is_master()):
        tqdm = None

    args = parse_args()
    total = load_total_cfg(args.total_cfg_json)
    proto = deepcopy(total["protocol"])
    train_cfg = total["train"]
    model_cfg = total.get("model_unet", {})

    # device
    if is_ddp:
        if ddp_device is None:
            raise RuntimeError("[DDP] setup_ddp did not return a device")
        device = ddp_device
    else:
        device = get_device(str(train_cfg.get("device", "cuda")))

    base_seed = int(train_cfg.get("seed", 123))
    rm_ddp.seed_everything(base_seed, torch_same_across_ranks=True, add_rank_to_numpy_python=False)

    # dataset_root absolute
    ds_root = resolve_path(str(_get(proto, "split.dataset_root", fallback=".")))
    proto["split"]["dataset_root"] = str(ds_root)

    # AutoClip (rank0-only)
    tn = proto.get("target_norm", {}) or {}
    y_transform = str(tn.get("y_transform", "none")).lower()
    needs_clip = ("clip" in y_transform)
    if needs_clip and tn.get("y_clip_range", None) is None:
        pct = tn.get("y_clip_percentiles", (1.0, 99.0))
        try:
            pct = (float(pct[0]), float(pct[1])) if isinstance(pct, (list, tuple)) and len(pct) == 2 else (1.0, 99.0)
        except Exception:
            pct = (1.0, 99.0)

        clip_range = None
        if rm_ddp.is_master():
            lo, hi = estimate_target_clip_percentiles(proto, percentiles=pct, seed=base_seed)
            clip_range = [float(lo), float(hi)]
            print(f"[AutoClip] y_clip_range inferred as [{lo:.6g}, {hi:.6g}] (percentile {pct[0]}~{pct[1]})")
        if is_ddp:
            clip_range = rm_ddp.broadcast_object(clip_range, src=0)

        tn["y_clip_range"] = clip_range
        proto["target_norm"] = tn

    rm_ddp.seed_everything(base_seed, torch_same_across_ranks=True, add_rank_to_numpy_python=True)

    # save-friendly total cfg
    total_save = deepcopy(total)
    total_save["protocol"] = proto

    worker_init_fn = rm_ddp.build_worker_init_fn(base_seed)
    train_loader, val_loader, train_sampler = build_loaders(
        proto,
        train_cfg,
        is_ddp=is_ddp,
        rank0_only_val=True,
        worker_init_fn=worker_init_fn,
    )
    adapter = ModelInputAdapter(proto)


    evaluator = ProtocolEvaluator(proto) if rm_ddp.is_master() else None

    model = build_model(proto, model_cfg).to(device)
    if is_ddp:
        # Do NOT convert to SyncBatchNorm (rank0-only val would hang).
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optim = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    use_amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    amp_dtype = str(train_cfg.get("amp_dtype", "fp16"))
    scaler = make_grad_scaler('cuda', enabled=bool(use_amp and amp_dtype.lower() == "fp16"))

    # output dir (rank0 decides, broadcast to others)
    out_dir = resolve_path(str(train_cfg.get("out_dir", "./rm_runs_unet_protocol")))
    run_name = str(train_cfg.get("run_name", "unet_protocol")).strip() or "unet_protocol"

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
        (run_dir / "total_cfg.json").write_text(json.dumps(total_save, indent=2, ensure_ascii=False), encoding="utf-8")

    rm_ddp.barrier()

    # resume
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

    layout = list(_get(proto, "input.layout", fallback=[]))
    building_idx = layout.index("building")

    epochs = int(train_cfg.get("epochs", 60))
    save_every = int(train_cfg.get("save_every", 1))
    save_best_by = str(train_cfg.get("save_best_by", "rmse")).lower().strip() or "rmse"

    if rm_ddp.is_master():
        bs = int(train_cfg.get("batch_size", 8))
        ws = rm_ddp.get_world_size()
        print(f"[Start] UNet | DDP={is_ddp} | world_size={ws} | bs_per_gpu={bs} | global_bs={bs*ws}")

    for ep in range(start_epoch, epochs + 1):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(ep)

        tr_loss = train_one_epoch(
            model,
            train_loader,
            optim,
            device,
            adapter,
            building_idx=building_idx,
            amp=use_amp,
            amp_dtype=str(amp_dtype),
            scaler=scaler,
            grad_accum=int(train_cfg.get("grad_accum", 1)),
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            loss_cfg=train_cfg,
        )

        if is_ddp:
            tr_loss_t = torch.tensor([float(tr_loss)], device=device, dtype=torch.float32)
            tr_loss = float(rm_ddp.all_reduce_mean(tr_loss_t)[0].item())

        rm_ddp.barrier()

        if rm_ddp.is_master():
            assert evaluator is not None
            assert val_loader is not None

            val_metrics = eval_one_epoch(model.module if is_ddp else model, val_loader, evaluator, device, adapter)
            val_key = float(val_metrics.get(save_best_by, float("inf")))

            rec = {"epoch": ep, "train_loss": float(tr_loss)}
            rec.update({str(k): float(v) for k, v in val_metrics.items()})
            history.append(rec)

            print(f"[epoch {ep:03d}] train_loss={tr_loss:.6f} | " +
                  ", ".join([f"{k}={float(v):.6f}" for k, v in val_metrics.items()]))

            if save_every > 0 and (ep % save_every == 0):
                save_ckpt(
                    ckpt_dir / f"epoch_{ep:03d}.pt",
                    epoch=ep,
                    model=(model.module if is_ddp else model),
                    optim=optim,
                    scaler=scaler,
                    total_cfg=total_save,
                    model_cfg=model_cfg,
                    metrics=val_metrics,
                    history=history[-200:],
                )

            if val_key < best_val:
                best_val = float(val_key)
                best_epoch = int(ep)
                save_ckpt(
                    ckpt_dir / "best.pt",
                    epoch=ep,
                    model=(model.module if is_ddp else model),
                    optim=optim,
                    scaler=scaler,
                    total_cfg=total_save,
                    model_cfg=model_cfg,
                    metrics=val_metrics,
                    history=history,
                )
                print(f"[best] saved: {ckpt_dir / 'best.pt'} | {save_best_by}={best_val:.6f}")

            (run_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
            (run_dir / "best.json").write_text(json.dumps({"epoch": best_epoch, save_best_by: best_val}, indent=2), encoding="utf-8")

        rm_ddp.barrier()

    rm_ddp.cleanup_ddp()


if __name__ == "__main__":
    main()