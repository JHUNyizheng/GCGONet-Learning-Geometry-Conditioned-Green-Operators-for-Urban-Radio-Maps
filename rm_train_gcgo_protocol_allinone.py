# -*- coding: utf-8 -*-
"""
rm_train_gcgo_protocol_allinone.py
============================================================
【彻底重构版】GCGO / GeoGreen-Op 训练逻辑（Protocol-first, All-in-one）

你要求：
- 不在旧代码上打补丁：训练/评估/保存逻辑彻底重写
- 严格对齐你已经确定的：
    * 输入协议（rm_protocol_dataset.py -> RadiomapProtocolDataset + ModelInputAdapter）
    * 统一评估出口（rm_protocol_eval.py -> ProtocolEvaluator）
- 重点：仔细实现 GCGO / GeoGreen-Op 的“精髓”：
    * Geometry Encoder：从 (geom + f_src) 提取 local embedding h(x) 与 global latent z_Ω
    * GeoGreen Operator Blocks：频域 Green 核（geometry-conditioned spectral kernel） + 空间低秩非平稳修正（low-rank correction）
    * 多层残差堆叠（L blocks）+ 1×1 mixing + dropout，保持稳定训练（float32 FFT + AMP 兼容）

说明（非常重要，避免信息泄漏）：
- 本脚本的 GCGO 只使用“公开输入 x”做切片/分组，不引入任何额外私有信息。
- f_src 与 geom_feat 都来自同一份 x（由 ModelInputAdapter 决定）。

依赖文件（同目录）：
- rm_total_config.py
- rm_protocol_dataset.py
- rm_protocol_eval.py
"""

from __future__ import annotations

import os
import json
import math
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import argparse
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

from rm_protocol_dataset import RadiomapProtocolDataset, ModelInputAdapter, ensure_target_stats
from rm_protocol_eval import ProtocolEvaluator


THIS_DIR = Path(__file__).resolve().parent


# ============================================================
# 0) Default TOTAL_CFG（当 rm_total_config.py 不存在时兜底）
# ============================================================

DEFAULT_TOTAL_CFG: Dict[str, Any] = {
    "protocol": {
        "split": {
            "dataset_root": ".",
            "target_subdir": "gain/DPM",
            "buildings_subdir": "png/buildings_complete",
            "antenna_subdir": "antenna",
            "split_subdir": "splits",
            "npy_channel": 0,
            "train_list_path": "splits/train_maps.txt",
            "val_list_path": "splits/val_maps.txt",
            "test_list_path": "splits/test_maps.txt",
            "max_tx_per_map": 80,
            "map_select": "first",
        },
        "input": {
            # 你主协议 P0+（示例，实际以 rm_total_config 为准）
            "layout": ["building", "tx_gaussian", "tx_invdist", "dy", "dx"],
            "norm": {
                "x_norm": "none",        # none | global_zscore | global_minmax
                "x_stats_path": None,
            },
            "roles": {
                # GCGO 如何从 x 切片（仍然是公开信息）
                "gcgo": {
                    "source_channel": "tx_gaussian",
                    "geom_channels": ["building", "tx_invdist", "dy", "dx"],
                }
            }
        },
        "target_norm": {
            "y_transform": "none",   # none | log1p
            "y_clip_range": None,    # None -> 不 clip
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
            "error_percentiles": {"enabled": True, "qs": [50, 75, 90, 95, 99]},
            "dist_bins": {"enabled": True, "bins": [0, 2, 4, 8, 16, 32, 64]},
        },
    },

    "train": {
        "device": "cuda",
        "seed": 123,
        "epochs": 30,
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
        "out_dir": "./rm_runs_gcgo_protocol",
        "run_name": "gcgo_protocol",
        "save_every": 1,
        "save_best_by": "composite",  # composite | rmse | mae | boundary_rmse | high_rmse
        "resume_ckpt": None,
        "log_every_steps": 50,
    },

    # 注意：这里给 GCGO 单独一个 model_gcgo 子配置（若 rm_total_config.py 未提供）
    "model_gcgo": {
        # Encoder
        "h_dim": 128,             # local embedding h(x) 通道数
        "z_dim": 128,             # global latent z_Ω 维度
        "enc_base": 64,

        # Field feature width
        "width": 96,             # 频域/特征空间 u 的通道数（base_ch）

        # GeoGreen blocks
        "L": 6,                  # Green operator blocks 数
        "modes": 32,             # FFT 低频保留模态数（每个维度）
        "rank": 32,               # 低秩修正 rank R
        "dropout": 0.0,

        # Geometry-conditioned spectral kernel (paper-style)
        "k_embed_freqs": [1, 2, 4, 8],  # 频率坐标 embedding 的频段
        "kernel_hidden": 256,    # MLP_K 隐层宽度
        "kernel_clamp": 2.0,     # 对生成的 kernel 做 clamp（稳定训练）

        # Low-rank correction
        "corr_scale": 0.5,
        "corr_inner": "u_proj",   # f_src | u_mean | u_proj

        # Optional source metadata embedding（默认关闭；你的当前 meta 没有 band）
        "use_band_emb": False,
        "n_bands": 1,
        "band_emb_dim": 8,
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


def resolve_path(p: Union[str, Path]) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _get(d, path, fallback=None):
    """Safely get a nested field from a dict using dot-separated keys.

    Supports integer list indices if a path component is a digit.
    """
    cur = d
    for key in str(path).split("."):
        if isinstance(cur, dict):
            if key in cur:
                cur = cur[key]
                continue
            return fallback
        if isinstance(cur, (list, tuple)) and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(cur):
                cur = cur[idx]
                continue
            return fallback
        return fallback
    return cur



def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_device(device_str: str) -> torch.device:
    d = str(device_str).lower().strip()
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(d)
    return torch.device("cpu")


def get_device(device_str: str) -> torch.device:
    """Compat helper: some older scripts used `get_device` instead of `as_device`."""
    return as_device(device_str)


def build_model(proto: Dict[str, Any], model_cfg: Dict[str, Any]) -> nn.Module:
    """
    Build GeoGreen-Op (GCGO) model (Radiomap2).

    Notes:
      - GeoGreenOp uses L (number of Green-operator blocks) rather than "depth".
      - Backward-compatible aliases:
          depth -> L
          corr_rank -> rank
          K / Kx / Ky -> modes
          band_dim -> band_emb_dim
    """
    roles = _get(proto, "input.roles.gcgo", fallback={}) or {}
    geom_channels = roles.get("geom_channels", [])
    source_channel = roles.get("source_channel", "tx_gaussian")

    Cg = int(len(geom_channels))
    Cs = int(len(source_channel)) if isinstance(source_channel, (list, tuple)) else 1

    # hyperparams
    h_dim = int(model_cfg.get("h_dim", 64))
    z_dim = int(model_cfg.get("z_dim", 64))
    enc_base = int(model_cfg.get("enc_base", 32))
    width = int(model_cfg.get("width", 48))

    L = int(model_cfg.get("L", model_cfg.get("depth", 3)))

    # spectral modes (single int)
    modes = int(model_cfg.get("modes",
                              model_cfg.get("K",
                                            model_cfg.get("Kx",
                                                          model_cfg.get("Ky", 24)))))

    rank = int(model_cfg.get("rank", model_cfg.get("corr_rank", 8)))
    dropout = float(model_cfg.get("dropout", 0.1))

    k_embed_freqs = model_cfg.get("k_embed_freqs", (1, 2, 4, 8))
    kernel_hidden = int(model_cfg.get("kernel_hidden", 128))
    kernel_clamp = float(model_cfg.get("kernel_clamp", 2.0))

    corr_scale = float(model_cfg.get("corr_scale", 0.1))
    corr_inner = str(model_cfg.get("corr_inner", "f_src"))

    use_band_emb = bool(model_cfg.get("use_band_emb", False))
    n_bands = int(model_cfg.get("n_bands", 1))
    band_emb_dim = int(model_cfg.get("band_emb_dim", model_cfg.get("band_dim", 8)))

    return GeoGreenOp(
        Cg=Cg,
        Cs=Cs,
        h_dim=h_dim,
        z_dim=z_dim,
        enc_base=enc_base,
        width=width,
        L=L,
        modes=modes,
        rank=rank,
        dropout=dropout,
        k_embed_freqs=list(k_embed_freqs),
        kernel_hidden=kernel_hidden,
        kernel_clamp=kernel_clamp,
        corr_scale=corr_scale,
        corr_inner=corr_inner,
        use_band_emb=use_band_emb,
        n_bands=n_bands,
        band_emb_dim=band_emb_dim,
    )


# ============================================================
# 2) GeoGreen-Op (GCGO) model
# ============================================================

class SimpleUNetEncoder(nn.Module):
    """
    Shallow U-Net-ish encoder (paper: "shallow U-Net geometry encoder")：
    - 输出 local h(x) : (B, h_dim, H, W)
    - 输出 global z_Ω : (B, z_dim)
    """
    def __init__(self, in_ch: int, base: int = 32, z_dim: int = 64, h_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.GELU(),
        )
        self.down = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, base * 2),
            nn.GELU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.GELU(),
        )
        self.up = nn.ConvTranspose2d(base * 2, h_dim, 4, stride=2, padding=1)
        self.h_dim = int(h_dim)
        self.z_mlp = nn.Sequential(
            nn.Linear(h_dim, 128),
            nn.GELU(),
            nn.Linear(128, z_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv1(x)
        x2 = self.conv2(self.down(x1))
        h = self.up(x2)  # (B,h_dim,H,W)

        pooled = h.mean(dim=(-2, -1))          # (B,h_dim)
        z = self.z_mlp(pooled)                 # (B,z_dim)
        return h, z


def _build_k_embedding(Ky: int, Kx: int, H: int, W: int, freqs: List[int], device: torch.device) -> torch.Tensor:
    """
    频域坐标 embedding φ(k)：
    - 输入：Ky,Kx 为保留的低频模态数量
    - 输出：phi shape (Ky,Kx,phi_dim)
    设计：
      phi = [ky_norm, kx_norm,
             sin(2π f ky_norm), cos(2π f ky_norm),
             sin(2π f kx_norm), cos(2π f kx_norm) ... for f in freqs]
    """
    ky = torch.arange(Ky, device=device, dtype=torch.float32) / max(1.0, float(H))
    kx = torch.arange(Kx, device=device, dtype=torch.float32) / max(1.0, float(W))
    ky_grid, kx_grid = torch.meshgrid(ky, kx, indexing="ij")  # (Ky,Kx)

    feats = [ky_grid, kx_grid]
    two_pi = 2.0 * math.pi
    for f in freqs:
        w = float(f) * two_pi
        feats.append(torch.sin(w * ky_grid))
        feats.append(torch.cos(w * ky_grid))
        feats.append(torch.sin(w * kx_grid))
        feats.append(torch.cos(w * kx_grid))
    return torch.stack(feats, dim=-1)  # (Ky,Kx,phi_dim)


class GeoSpectralCore(nn.Module):
    """
    Geometry-conditioned Green kernel in frequency domain (paper-style):

      \hat{K}_θ(k | z_Ω) = MLP_K([φ(k), z_Ω])  -> complex weights per (k, channel)
      \hat{u}_core(k) = \hat{K}_θ(k | z_Ω) ⊙ \hat{u}(k)

    关键点：
    - FFT/ iFFT 在 float32 & autocast disabled，避免 AMP NaN
    - kernel 由 z_Ω 条件化（每个样本不同），更贴近论文的“geometry-conditioned kernel”
    """
    def __init__(self,
                 channels: int,
                 z_dim: int,
                 modes: int,
                 *,
                 k_embed_freqs: List[int],
                 kernel_hidden: int = 128,
                 kernel_clamp: float = 2.0):
        super().__init__()
        self.channels = int(channels)
        self.z_dim = int(z_dim)
        self.modes = int(modes)
        self.k_embed_freqs = list(k_embed_freqs)
        self.kernel_clamp = float(kernel_clamp)
        self.kernel_hidden = int(kernel_hidden)

        # 【修复】直接在 init 中计算维度并初始化 MLP，确保 .to(device) 生效
        # phi 包含: ky, kx (2个) + 每个频率的 sin/cos (4 * len)
        phi_dim = 2 + 4 * len(self.k_embed_freqs)
        in_dim = int(phi_dim + self.z_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.kernel_hidden),
            nn.GELU(),
            nn.Linear(self.kernel_hidden, 2 * self.channels),
        )

        # cache: (H,W,Ky,Kx) -> phi tensor
        self._phi_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

    def _get_phi(self, H: int, W: int, Ky: int, Kx: int, device: torch.device) -> torch.Tensor:
        key = (H, W, Ky, Kx)
        phi = self._phi_cache.get(key, None)
        if phi is None or phi.device != device:
            phi = _build_k_embedding(Ky, Kx, H, W, self.k_embed_freqs, device=device)  # float32
            self._phi_cache[key] = phi
        return phi

    def forward(self, u: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        u: (B,C,H,W)
        z: (B,z_dim)
        returns: u_core (B,C,H,W)
        """
        B, C, H, W = u.shape
        Ky = min(self.modes, H)
        Kx = min(self.modes, W // 2 + 1)  # rfft2 width
        if Ky <= 0 or Kx <= 0:
            return torch.zeros_like(u)

        device = u.device
        phi = self._get_phi(H, W, Ky, Kx, device=device)  # (Ky,Kx,phi_dim)

        # Build per-sample kernel weights for each (ky,kx)
        # phi: (Ky,Kx,D) -> (1,Ky,Kx,D) -> (B,Ky,Kx,D)
        phi_b = phi.unsqueeze(0).expand(B, -1, -1, -1)
        z_b = z.view(B, 1, 1, -1).expand(B, Ky, Kx, -1)
        inp = torch.cat([phi_b, z_b], dim=-1)  # (B,Ky,Kx,D+z)

        # Important: MLP in float32 for stability
        if device.type == "cuda":
            mlp_ctx = autocast_ctx("cuda", dtype=torch.float16, enabled=False)
        else:
            mlp_ctx = _nullcontext()
        with mlp_ctx:
            # 【修复】使用 self.mlp (在 init 中已初始化)
            w = self.mlp(inp.float())  # (B,Ky,Kx,2C) float32

        w = torch.clamp(w, -self.kernel_clamp, self.kernel_clamp)
        w = w.view(B, Ky, Kx, 2, C).permute(0, 4, 1, 2, 3).contiguous()  # (B,C,Ky,Kx,2)
        w_complex = torch.complex(w[..., 0].float(), w[..., 1].float())  # (B,C,Ky,Kx)  # avoid ComplexHalf warning

        # FFT path in float32, no autocast
        if device.type == "cuda":
            autocast_off = autocast_ctx("cuda", dtype=torch.float16, enabled=False)
        else:
            autocast_off = _nullcontext()

        with autocast_off:
            u32 = u.float()
            U = torch.fft.rfft2(u32, norm="ortho")  # complex64
            U_out = torch.zeros_like(U)
            U_low = U[:, :, :Ky, :Kx] * w_complex  # broadcast match
            U_out[:, :, :Ky, :Kx] = U_low
            u_core32 = torch.fft.irfft2(U_out, s=(H, W), norm="ortho")  # float32

        u_core = u_core32.to(dtype=u.dtype)
        u_core = torch.nan_to_num(u_core, nan=0.0, posinf=0.0, neginf=0.0)
        return u_core
    
class LowRankCorrection(nn.Module):
    """
    Non-stationary low-rank correction (paper: "spatial low-rank correction"):
      Δu(x) = Σ_r a_r(x) * Σ_y b_r(y) * s(y)

    其中 s(y) 默认为 f_src(y)（与 notebook 版本一致，更稳定），也可选用 u 的某种投影。
    """
    def __init__(self, h_dim: int, z_dim: int, rank: int = 8, corr_scale: float = 0.1):
        super().__init__()
        self.rank = int(rank)
        self.corr_scale = float(corr_scale)
        self.mlp_a = nn.Sequential(
            nn.Conv2d(h_dim + z_dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, self.rank, 1),
        )
        self.mlp_b = nn.Sequential(
            nn.Conv2d(h_dim + z_dim, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, self.rank, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        h: (B,h_dim,H,W)
        z: (B,z_dim)
        s: (B,1,H,W)  用于 inner product 的标量场
        returns: corr (B,1,H,W)
        """
        B, _, H, W = h.shape
        z_img = z.view(B, -1, 1, 1).expand(B, z.shape[1], H, W)
        hz = torch.cat([h, z_img], dim=1)
        a = self.mlp_a(hz)  # (B,R,H,W)
        b = self.mlp_b(hz)  # (B,R,H,W)

        # normalize for stability
        a = (a - a.mean(dim=(-2, -1), keepdim=True)) / (a.std(dim=(-2, -1), keepdim=True) + 1e-6)
        b = (b - b.mean(dim=(-2, -1), keepdim=True)) / (b.std(dim=(-2, -1), keepdim=True) + 1e-6)

        device = h.device
        if device.type == "cuda":
            autocast_off = autocast_ctx("cuda", dtype=torch.float16, enabled=False)
        else:
            autocast_off = _nullcontext()

        with autocast_off:
            b32 = b.float()
            s32 = s.float()
            # 修改后：改为 mean，并放大一点 clamp 范围以适应 mean 的数值特性
            # inner = (b32 * s32).mean(dim=(-2, -1)) * math.sqrt(H * W)
            inner = (b32 * s32).mean(dim=(-2, -1))  
            inner = torch.clamp(inner, -50.0, 50.0)

        corr = (a * inner[..., None, None]).sum(dim=1, keepdim=True)  # (B,1,H,W)
        corr = self.corr_scale * corr
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return corr


class _nullcontext:
    def __enter__(self):  # pragma: no cover
        return None
    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        return False


class GeoGreenOp(nn.Module):
    """
    GeoGreen-Op / GCGO（适配你的协议）：

    输入：
      geom: (B, Cg, H, W)   # geometry features (from x)
      f_src: (B, 1, H, W)  # source map (from x)
      meta:  仅用于可选 band embedding（默认关闭，meta 不提供也没关系）

    输出：
      pred: (B,1,H,W)
    """
    def __init__(self,
                 Cg: int,
                 Cs: int = 1,
                 *,
                 h_dim: int = 64,
                 z_dim: int = 64,
                 enc_base: int = 32,
                 width: int = 48,
                 L: int = 3,
                 modes: int = 24,
                 rank: int = 8,
                 dropout: float = 0.1,
                 k_embed_freqs: List[int] = (1, 2, 4, 8),
                 kernel_hidden: int = 128,
                 kernel_clamp: float = 2.0,
                 corr_scale: float = 0.1,
                 corr_inner: str = "f_src",
                 use_band_emb: bool = False,
                 n_bands: int = 1,
                 band_emb_dim: int = 8):
        super().__init__()
        self.Cg = int(Cg)
        self.Cs = int(max(1, Cs))
        self.h_dim = int(h_dim)
        self.z_dim = int(z_dim)
        self.width = int(width)
        self.L = int(L)
        self.modes = int(modes)
        self.rank = int(rank)
        self.dropout = float(dropout)
        self.corr_inner = str(corr_inner).lower().strip()

        self.use_band_emb = bool(use_band_emb)
        self.n_bands = int(max(1, n_bands))
        self.band_emb_dim = int(band_emb_dim) if self.use_band_emb else 0
        if self.use_band_emb:
            self.band_emb = nn.Embedding(self.n_bands, self.band_emb_dim)
        else:
            self.band_emb = None

        # Encoder input channels = geom (Cg) + source map (Cs) + optional band embedding.
        in_ch = self.Cg + self.Cs + self.band_emb_dim
        self.enc = SimpleUNetEncoder(in_ch, base=int(enc_base), z_dim=self.z_dim, h_dim=self.h_dim)
        self.proj_in = nn.Conv2d(self.h_dim, self.width, 1)

        self.blocks = nn.ModuleList()
        for _ in range(self.L):
            blk = nn.ModuleDict({
                "spec": GeoSpectralCore(self.width, self.z_dim, self.modes,
                                        k_embed_freqs=list(k_embed_freqs),
                                        kernel_hidden=int(kernel_hidden),
                                        kernel_clamp=float(kernel_clamp)),
                "corr": LowRankCorrection(self.h_dim, self.z_dim, rank=self.rank, corr_scale=float(corr_scale)),
                "mix": nn.Conv2d(self.width, self.width, 1),
                "drop": nn.Dropout2d(self.dropout),
            })
            self.blocks.append(blk)

        self.readout = nn.Sequential(
            nn.Conv2d(self.width, self.width, 1),
            nn.GELU(),
            nn.Conv2d(self.width, 1, 1),
        )

        # optional u->scalar projection for corr_inner = u_proj
        self.u_proj = nn.Conv2d(self.width, 1, 1)

    def _meta_to_band(self, meta: Any, B: int, device: torch.device) -> torch.Tensor:
        """
        meta 来自 DataLoader collate，通常是 list[dict]。
        若 meta 不包含 band，则返回全 0。
        """
        if not self.use_band_emb:
            return torch.zeros((B,), dtype=torch.long, device=device)

        if isinstance(meta, list) and len(meta) == B:
            vals = []
            for m in meta:
                try:
                    vals.append(int(m.get("band", m.get("band_id", 0))))
                except Exception:
                    vals.append(0)
            band = torch.tensor(vals, dtype=torch.long, device=device)
            band = torch.clamp(band, 0, self.n_bands - 1)
            return band

        if isinstance(meta, dict):
            b = int(meta.get("band", meta.get("band_id", 0)))
            b = max(0, min(self.n_bands - 1, b))
            return torch.full((B,), b, dtype=torch.long, device=device)

        return torch.zeros((B,), dtype=torch.long, device=device)

    def forward(self, geom: torch.Tensor, f_src: torch.Tensor, meta: Any = None) -> torch.Tensor:
        B, _, H, W = geom.shape

        if self.use_band_emb and self.band_emb is not None:
            band = self._meta_to_band(meta, B, geom.device)
            be = self.band_emb(band).view(B, -1, 1, 1).expand(B, -1, H, W)
            x_in = torch.cat([geom, f_src, be], dim=1)
        else:
            x_in = torch.cat([geom, f_src], dim=1)

        h, z = self.enc(x_in)      # h: (B,h_dim,H,W), z:(B,z_dim)
        u = self.proj_in(h)        # (B,width,H,W)

        for blk in self.blocks:
            u_core = blk["spec"](u, z)

            # correction inner source
            if self.corr_inner == "u_mean":
                s = u.mean(dim=1, keepdim=True)
            elif self.corr_inner == "u_proj":
                s = self.u_proj(u)
            else:
                s = f_src
                # LowRankCorrection expects a single-channel signal. If multiple
                # source channels are provided, collapse them to a scalar field.
                if s.dim() == 4 and s.shape[1] != 1:
                    s = s.mean(dim=1, keepdim=True)

            corr = blk["corr"](h, z, s)       # (B,1,H,W)
            corr_c = corr.expand_as(u)        # lift

            u = u + blk["drop"](F.gelu(blk["mix"](u_core + corr_c)))

        out = self.readout(u)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


# ============================================================
# 3) Loss
# ============================================================

def _huber(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=float(delta), reduction="none")


def compute_loss(pred: torch.Tensor,
                 y: torch.Tensor,
                 x: torch.Tensor,
                 *,
                 building_idx: int,
                 loss_cfg: Dict[str, Any]) -> torch.Tensor:
    """
    pred,y: (B,1,H,W)
    x: (B,C,H,W)  (需要 building 通道用于室外 mask / boundary ring)
    """
    loss_type = str(loss_cfg.get("loss_type", "huber")).lower().strip()
    huber_delta = float(loss_cfg.get("huber_delta", 0.05))

    mask_buildings = bool(loss_cfg.get("loss_mask_buildings", True))
    bw = int(loss_cfg.get("boundary_width", 3))
    w_boundary = float(loss_cfg.get("loss_weight_boundary", 0.0))
    high_q = float(loss_cfg.get("high_q", 0.90))
    w_high = float(loss_cfg.get("loss_weight_high", 0.0))
    tv_w = float(loss_cfg.get("tv_weight", 0.0))

    # outdoor mask: building==1 表示建筑（室内/阻挡）
    bld = x[:, building_idx:building_idx+1, ...]
    outdoor = (bld < 0.5).float()

    if loss_type == "mse":
        per = (pred - y) ** 2
    elif loss_type == "l1":
        per = (pred - y).abs()
    else:
        per = _huber(pred, y, delta=huber_delta)

    # 主 loss（室外）
    if mask_buildings:
        base = (per * outdoor).sum() / (outdoor.sum() + 1e-6)
    else:
        base = per.mean()

    # boundary ring 加权（沿用 evaluator 的 ring 构造逻辑）
    if w_boundary > 0:
        from rm_protocol_eval import build_boundary_ring
        ring = build_boundary_ring(bld, width=bw)  # (B,1,H,W), float
        boundary = (per * ring).sum() / (ring.sum() + 1e-6)
    else:
        boundary = pred.new_tensor(0.0)

    # high-value region 加权
    if w_high > 0:
        from rm_protocol_eval import build_high_value_mask
        
        # [修复] 构造必填参数 domain_mask
        if mask_buildings:
            # outdoor 已经是 (B,1,H,W) 的 0/1 float，函数内部会转 bool
            d_mask = outdoor 
        else:
            d_mask = torch.ones_like(y)

        # [修复] 传入 d_mask
        high_mask = build_high_value_mask(y, d_mask, q=high_q)  # 返回 bool Tensor
        high_mask = high_mask.float()  # 转回 float 用于计算 loss

        high_loss = (per * high_mask).sum() / (high_mask.sum() + 1e-6)
    else:
        high_loss = pred.new_tensor(0.0)

    # TV regularization（让预测更平滑，可关）
    if tv_w > 0:
        dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean()
        dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
        tv = dy + dx
    else:
        tv = pred.new_tensor(0.0)

    return base + w_boundary * boundary + w_high * high_loss + tv_w * tv


# ============================================================
# 4) Train / Eval
# ============================================================

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    adapter: ModelInputAdapter,
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

    is_master = rm_ddp.is_master()
    it = loader
    if is_master and (tqdm is not None):
        it = tqdm(loader, desc="train", leave=False)

    for step, (x, y, meta) in enumerate(it, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        inp = adapter.make_model_inputs("gcgo", x, meta)
        geom = inp["geom"].to(device, non_blocking=True)
        f_src = inp["f_src"].to(device, non_blocking=True)
        meta_inp = inp.get("meta", meta)

        if use_amp:
            with autocast_ctx("cuda", dtype=dtype, enabled=True):
                pred = model(geom, f_src, meta_inp)
                loss = compute_loss(pred, y, x, building_idx=building_idx, loss_cfg=loss_cfg) / grad_accum
        else:
            pred = model(geom, f_src, meta_inp)
            loss = compute_loss(pred, y, x, building_idx=building_idx, loss_cfg=loss_cfg) / grad_accum

        if use_fp16_scaler:
            assert scaler is not None
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum == 0:
            if grad_clip and grad_clip > 0:
                if use_fp16_scaler:
                    scaler.unscale_(optim)  # type: ignore
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))

            if use_fp16_scaler:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)

        total += float(loss.detach().item()) * grad_accum
        n += 1

        log_every = int(loss_cfg.get("log_every_steps", 0) or 0)
        if is_master and (tqdm is None) and log_every > 0 and (step % log_every == 0):
            print(f"[train] step={step} loss={total/max(1,n):.6f}")

    return total / max(1, n)


@torch.no_grad()
def eval_one_epoch(model: nn.Module,
                   loader: DataLoader,
                   adapter: ModelInputAdapter,
                   evaluator: ProtocolEvaluator,
                   device: torch.device) -> Dict[str, float]:
    model.eval()
    batch_metrics: List[Dict[str, float]] = []

    is_master = rm_ddp.is_master()
    it = loader
    if is_master and (tqdm is not None):
        it = tqdm(loader, desc="val", leave=False)

    for x, y, meta in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        inp = adapter.make_model_inputs("gcgo", x, meta)
        geom = inp["geom"].to(device, non_blocking=True)
        f_src = inp["f_src"].to(device, non_blocking=True)
        meta_inp = inp.get("meta", meta)

        pred = model(geom, f_src, meta_inp)
        m = evaluator.compute_batch(pred, y, x)
        batch_metrics.append(m)

    return evaluator.reduce_epoch(batch_metrics)


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
              history: List[Dict[str, Any]],
              best_val: float = float('inf'),
              best_epoch: int = -1):
    obj = {
        "epoch": int(epoch),
        "total_cfg": total_cfg,
        "model_cfg": model_cfg,
        "model": model.state_dict(),  
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "history": history,
        "best_val": float(best_val) if best_val is not None else None,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(path))


def try_load_ckpt(path: Optional[str], device: torch.device):
    """Load checkpoint dict if path exists, else return None."""
    if not path:
        return None
    p = resolve_path(path)
    if not p.is_file():
        return None
    return torch.load(str(p), map_location=device)



def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    # Handle 'module.' prefix mismatch between DDP and non-DDP checkpoints.
    if not state:
        return state
    keys = list(state.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def _get_ckpt_field(ckpt: Dict[str, Any], *names: str):
    for n in names:
        if n in ckpt:
            return ckpt[n]
    return None


# ============================================================
# 6) Main
# ============================================================

def main():
    ddp_info = rm_ddp.setup_ddp(verbose=rm_ddp.is_master())
    is_ddp = bool(getattr(ddp_info, "is_ddp", False))
    local_rank = int(getattr(ddp_info, "local_rank", 0))
    ddp_device = getattr(ddp_info, "device", None)

    global tqdm
    if is_ddp and (not rm_ddp.is_master()):
        tqdm = None
        # Silence repeated per-rank ComplexHalf warning spam in DDP
        warnings.filterwarnings("ignore", message=r"ComplexHalf support is experimental.*")

    args = parse_args()
    total = load_total_cfg(args.total_cfg_json)
    proto = deepcopy(total["protocol"])
    train_cfg = total["train"]
    model_cfg = total.get("model_gcgo", {})

    # device
    if is_ddp:
        if ddp_device is None:
            raise RuntimeError("[DDP] setup_ddp did not return a device")
        device = ddp_device
    else:
        device = get_device(str(train_cfg.get("device", "cuda")))

    seed = int(train_cfg.get("seed", 123))
    rm_ddp.seed_everything(seed, torch_same_across_ranks=True, add_rank_to_numpy_python=False)

    # dataset_root absolute
    ds_root = resolve_path(str(_get(proto, "split.dataset_root", fallback=".")))
    proto["split"]["dataset_root"] = str(ds_root)

    # target stats (rank0 computes once -> broadcast proto so all ranks share y_stats_path)
    if rm_ddp.is_master():
        ensure_target_stats(proto, seed=seed)
    rm_ddp.barrier()
    if is_ddp:
        proto = rm_ddp.broadcast_object(proto, src=0)

    # now offset python/numpy RNG across ranks
    rm_ddp.seed_everything(seed, torch_same_across_ranks=True, add_rank_to_numpy_python=True)
    worker_init_fn = rm_ddp.build_worker_init_fn(seed)

    # Adapter slices/repackages the shared public input x according to
    # protocol.input.roles (GCGO expects {geom, f_src, meta}).
    adapter = ModelInputAdapter(proto)

    # --- datasets / loaders ---
    ds_train = RadiomapProtocolDataset(proto, split="train", strict_layout=True)
    ds_val = None
    if rm_ddp.is_master():
        ds_val = RadiomapProtocolDataset(proto, split="val", strict_layout=True)

    bs = int(train_cfg.get("batch_size", 4))   # per-GPU
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
        num_workers=nw,
        pin_memory=pin,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=(nw > 0),
    )

    val_loader = None
    if ds_val is not None:
        val_loader = DataLoader(
            ds_val,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            persistent_workers=(nw > 0),
        )

    evaluator = ProtocolEvaluator(proto)
    building_idx = int(getattr(evaluator, 'building_idx', _get(proto, 'input.layout', fallback=[]).index('building')))


    # --- output dir ---
    out_dir = resolve_path(str(train_cfg.get("out_dir", "./rm_runs/gcgo")))
    run_name = str(train_cfg.get("run_name", "exp_gcgo")).strip() or "exp_gcgo"

    run_dir_str = None
    if rm_ddp.is_master():
        run_dir_str = str(out_dir / run_name)
    if is_ddp:
        run_dir_str = rm_ddp.broadcast_object(run_dir_str, src=0)
    assert run_dir_str is not None
    run_dir = Path(run_dir_str)
    ckpt_path = run_dir / "ckpt_best.pt"

    if rm_ddp.is_master():
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "total_cfg.json").write_text(json.dumps(total, indent=2, ensure_ascii=False), encoding="utf-8")

    rm_ddp.barrier()

    # --- model / optim ---
    model = build_model(proto, model_cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optim = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    use_amp = bool(train_cfg.get("amp", True) and device.type == "cuda")
    amp_dtype = str(train_cfg.get("amp_dtype", "fp16")).lower()
    scaler = make_grad_scaler('cuda', enabled=bool(use_amp and amp_dtype == "fp16"))

    # --- resume (key-robust) ---
    history: List[Dict[str, Any]] = []
    start_epoch = 1
    best_val = float("inf")
    best_epoch = -1

    resume_obj = try_load_ckpt(train_cfg.get("resume_ckpt", None), device=device)
    if resume_obj is not None:
        try:
            state_model = _get_ckpt_field(resume_obj, "model", "model_state", "state_dict", "net")
            if isinstance(state_model, dict):
                state_model = _strip_module_prefix(state_model)
                (model.module if is_ddp else model).load_state_dict(state_model, strict=True)

            state_optim = _get_ckpt_field(resume_obj, "optim", "optim_state", "optimizer", "optimizer_state_dict")
            if state_optim is not None:
                optim.load_state_dict(state_optim)

            sc = _get_ckpt_field(resume_obj, "scaler", "scaler_state")
            if scaler is not None and sc is not None:
                scaler.load_state_dict(sc)

            start_epoch = int(resume_obj.get("epoch", 0)) + 1
            history = list(resume_obj.get("history", []))
            best_val = float(resume_obj.get("best_val", best_val))
            best_epoch = int(resume_obj.get("best_epoch", best_epoch))

            if rm_ddp.is_master():
                print(f"[Resume] loaded {train_cfg.get('resume_ckpt')} -> start_epoch={start_epoch}")
        except Exception as e:
            if rm_ddp.is_master():
                print(f"[Resume] failed to load ckpt: {e}")

    # --- train ---
    epochs = int(train_cfg.get("epochs", 80))

    if rm_ddp.is_master():
        ws = rm_ddp.get_world_size()
        print(f"[Start] GCGO | DDP={is_ddp} | world_size={ws} | bs_per_gpu={bs} | global_bs={bs*ws}")

    for ep in range(start_epoch, epochs + 1):
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(ep)

        tr_loss = train_one_epoch(
            model,
            train_loader,
            adapter,
            optim,
            device,
            amp=use_amp,
            amp_dtype=amp_dtype,
            scaler=scaler,
            grad_accum=int(train_cfg.get("grad_accum", 1)),
            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
            building_idx=building_idx,
            loss_cfg=train_cfg,
        )

        if is_ddp:
            tr_loss_t = torch.tensor([float(tr_loss)], device=device, dtype=torch.float32)
            tr_loss = float(rm_ddp.all_reduce_mean(tr_loss_t)[0].item())

        rm_ddp.barrier()

        if rm_ddp.is_master():
            assert evaluator is not None
            assert val_loader is not None

            val_metrics = eval_one_epoch(model.module if is_ddp else model, val_loader, adapter, evaluator, device)
            val_key = float(val_metrics.get("composite", val_metrics.get("rmse", float("inf"))))

            rec = {"epoch": ep, "train_loss": float(tr_loss)}
            rec.update({str(k): float(v) for k, v in val_metrics.items()})
            history.append(rec)

            print(f"[epoch {ep:03d}] train_loss={tr_loss:.6f} | " +
                  ", ".join([f"{k}={float(v):.6f}" for k, v in val_metrics.items()]))

            if val_key < best_val:
                best_val = float(val_key)
                best_epoch = int(ep)
                save_ckpt(
                    ckpt_path,
                    epoch=ep,
                    model=(model.module if is_ddp else model),
                    optim=optim,
                    scaler=scaler,
                    total_cfg=total,
                    model_cfg=model_cfg,
                    metrics=val_metrics,
                    history=history,
                    best_val=best_val,
                    best_epoch=best_epoch,
                )
                print(f"[best] saved: {ckpt_path} | composite={best_val:.6f}")

            (run_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        rm_ddp.barrier()

    rm_ddp.cleanup_ddp()


if __name__ == "__main__":
    main()