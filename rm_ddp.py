# -*- coding: utf-8 -*-
"""
rm_ddp.py
============================================================
DDP (Distributed Data Parallel) helper utilities for RadioMap2.

Design goals
- Minimal-intrusion: training scripts can stay clean.
- Dual-mode: works in both single-process (python) and multi-process (torchrun).
- Safe defaults: Rank-0-only IO is easy; synchronization helpers included.
- Reproducibility: seed helpers for DDP + DataLoader workers.

Typical usage in a training script:
------------------------------------------------------------
import rm_ddp

ddp = rm_ddp.setup_ddp()   # returns DDPInfo
rm_ddp.seed_everything(base_seed, torch_same_across_ranks=True)
worker_init_fn = rm_ddp.build_worker_init_fn(base_seed)

# Only rank0 prints / writes:
if rm_ddp.is_master():
    print("hello")

# In DDP: use DistributedSampler, call sampler.set_epoch(epoch)
# Add rm_ddp.barrier() around rank0-only validation / ckpt saving.
------------------------------------------------------------
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional, Callable, List

import numpy as np
import torch
import torch.distributed as dist


# -----------------------------
# Basic distributed state
# -----------------------------
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank(default: int = 0) -> int:
    # torchrun sets LOCAL_RANK; legacy launchers may set it too.
    v = os.environ.get("LOCAL_RANK", None)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def is_master() -> bool:
    """Rank-0 process (the only process that should write logs/ckpts)."""
    return get_rank() == 0


def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()


def cleanup_ddp() -> None:
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


# -----------------------------
# Setup / info
# -----------------------------
@dataclass(frozen=True)
class DDPInfo:
    is_ddp: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str


def _infer_backend() -> str:
    # Prefer NCCL for CUDA, else GLOO.
    return "nccl" if torch.cuda.is_available() else "gloo"


def setup_ddp(
    backend: Optional[str] = None,
    init_method: str = "env://",
    timeout_sec: int = 1800,
    verbose: bool = True,
) -> DDPInfo:
    """
    Auto-detect torchrun environment and initialize process group if needed.

    Returns:
        DDPInfo(is_ddp, rank, world_size, local_rank, device, backend)

    Notes:
    - torchrun typically provides: RANK, WORLD_SIZE, LOCAL_RANK.
    - If WORLD_SIZE <= 1, we treat it as single-process mode.
    """
    # Detect environment
    env_rank = os.environ.get("RANK", None)
    env_world = os.environ.get("WORLD_SIZE", None)

    wants_ddp = (env_rank is not None) and (env_world is not None)
    world_size = int(env_world) if (wants_ddp and str(env_world).isdigit()) else 1

    backend_ = backend or _infer_backend()

    if wants_ddp and world_size > 1:
        local_rank = get_local_rank(0)

        # Set device first (NCCL needs correct CUDA device per process)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        if not is_dist_avail_and_initialized():
            dist.init_process_group(
                backend=backend_,
                init_method=init_method,
                timeout=timedelta(seconds=timeout_sec),
            )

        rank = get_rank()
        world_size = get_world_size()

        if verbose and rank == 0:
            print(f"[DDP] initialized | backend={backend_} | world_size={world_size}")

        return DDPInfo(
            is_ddp=True,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            backend=backend_,
        )

    # Single-process mode
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        backend_ = "nccl"  # informational; not initialized
    else:
        device = torch.device("cpu")
        backend_ = "gloo"

    if verbose:
        print(f"[DDP] not detected | single-process on {device}")

    return DDPInfo(
        is_ddp=False,
        rank=0,
        world_size=1,
        local_rank=0,
        device=device,
        backend=backend_,
    )


# -----------------------------
# Seeding helpers
# -----------------------------
def seed_everything(
    seed: int,
    *,
    torch_same_across_ranks: bool = True,
    add_rank_to_numpy_python: bool = True,
    deterministic: bool = False,
) -> None:
    """
    Seed python/random/numpy/torch for reproducibility.

    Recommended default:
    - torch_same_across_ranks=True: model init weights consistent across ranks.
    - add_rank_to_numpy_python=True: data augmentation randomness differs per rank.
    - deterministic=False: keep performance; set True only when you *need* bitwise reproducibility.

    If you prefer "two-phase seeding":
    - call once (torch_same_across_ranks=True) before building model
    - then call torch_seed_with_rank(seed) after wrapping DDP, if desired
    """
    rank = get_rank()

    # Python + numpy seeds
    base = seed + (rank if add_rank_to_numpy_python else 0)
    random.seed(base)
    np.random.seed(base)

    # Torch seeds
    torch_seed = seed if torch_same_across_ranks else (seed + rank)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    if deterministic:
        # Deterministic settings may reduce speed significantly.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def torch_seed_with_rank(seed: int) -> None:
    """
    Optional helper for "phase-2" seeding:
    after model initialization, set torch seed to seed+rank so dropout differs per rank.
    """
    rank = get_rank()
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def build_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """
    Returns a DataLoader worker_init_fn that produces unique, reproducible seeds
    across:
      - ranks
      - workers
    """
    rank = get_rank()

    def _init_fn(worker_id: int) -> None:
        # Large stride to avoid collisions across ranks.
        worker_seed = int(base_seed) + rank * 1000 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn


# -----------------------------
# Communication helpers (optional but handy)
# -----------------------------
def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast a picklable python object from src to all ranks.
    In single-process mode, returns obj unchanged.
    """
    if not is_dist_avail_and_initialized():
        return obj
    obj_list: List[Any] = [obj if get_rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    """All-reduce mean for a tensor scalar (or tensor). No-op in single-process."""
    if not is_dist_avail_and_initialized():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y = y / float(get_world_size())
    return y


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """All-reduce sum for a tensor. No-op in single-process."""
    if not is_dist_avail_and_initialized():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def all_reduce_scalar_dict(d: Dict[str, float], op: str = "sum") -> Dict[str, float]:
    """
    All-reduce a dict of python floats via torch tensors.
    op: 'sum' or 'mean'
    """
    if not is_dist_avail_and_initialized():
        return dict(d)

    keys = sorted(d.keys())
    t = torch.tensor([float(d[k]) for k in keys], device=torch.device("cuda", get_local_rank(0)) if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    if op.lower() == "mean":
        t = t / float(get_world_size())
    elif op.lower() != "sum":
        raise ValueError(f"Unsupported op: {op}")

    out = {k: float(v) for k, v in zip(keys, t.tolist())}
    return out
