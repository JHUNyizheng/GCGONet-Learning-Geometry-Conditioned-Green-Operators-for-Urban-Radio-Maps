# GeoGreen-Op for Radio Map Estimation (GCGO)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#requirements)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](#requirements)
[![Reproducible](https://img.shields.io/badge/Reproducible-Protocol--First-success.svg)](#reproducibility)
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](#license)

A **protocol-first** baseline codebase for radio-map prediction with unified **dataset / normalization / evaluation** across:
- **FNO** (Fourier Neural Operator)
- **UNet** (official baseline: strict 2-channel input)
- **GeoGreen-Op (GCGO)** (geometry-conditioned Green-operator style model)

Abstract：High-fidelity radio coverage estimation in dense urban areas is a significant challenge, largely because signal propagation is dictated by complex city geometry. While recent deep learning models—from CNNs to Transformers—have shown promise in predicting radio environment maps (REMs), they often treat wave propagation as a generic image-regression task, neglecting its fundamental physical structure. In this paper, we introduce a geometry-conditioned Green-operator framework that explicitly models how signal fields arise from city layouts. Instead of learning an implicit mapping, our model parameterizes the underlying Green functions as a geometry-conditioned kernel. We implement this using a neural operator architecture that combines a geometry-aware spectral core with a non-stationary low-rank correction. Evaluated on the URBAN-RM benchmark, our approach demonstrates superior generalization across diverse urban morphologies, outperforming standard FNO, Geo-FNO, and Transformer-based baselines. By bridging data-driven models with physics-inspired operators, we provide both state-of-the-art performance and interpretable diagnostics for wireless digital twins.

The core idea is to describe an experiment once in `TOTAL_CFG["protocol"]` (split, input layout, target transforms, evaluator settings), then reuse the exact same protocol across all models.

---

## Table of Contents
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Layout](#dataset-layout)
- [Configuration: TOTAL_CFG](#configuration-total_cfg)
- [Quick Start (One-Click)](#quick-start-one-click)
- [Train Individually](#train-individually)
- [Evaluation & Metrics](#evaluation--metrics)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Key Features
- **Protocol-first pipeline**: the protocol determines split, channels, normalization, and evaluation.
- **Fair comparison**: a single `ProtocolEvaluator` computes metrics for all models in the same space.
- **DDP-ready training**: training scripts support `torchrun` multi-GPU.
- **Robust dataset_root resolution**: one-click runner resolves relative paths against `rm_total_config.py` directory to avoid running from the wrong CWD.
- **Optional LoS auxiliary channel**: you can add an LoS / wall-count map as an additional input channel (if your dataset provides it).

---

## Repository Structure
Main entry points:
- `rm_total_config.py` — experiment config (`TOTAL_CFG`) including protocol + train + model sections
- `rm_oneclick_protocol_run3.py` — one-click orchestration: train/reuse 3 models + compare
- `rm_protocol_dataset.py` — `RadiomapProtocolDataset` and `ModelInputAdapter`
- `rm_protocol_eval.py` — `ProtocolEvaluator` (RMSE/boundary/high/percentiles/dist-bins)
- `rm_train_fno_protocol_allinone.py` — FNO training script
- `rm_train_unet_protocol_allinone.py` — UNet training script
- `rm_train_gcgo_protocol_allinone.py` — GCGO training script
- `rm_unet_official.py` — official UNet baseline (strict 2-channel input)
- `rm_ddp.py` — DDP utilities

Optional (expected in full repo if you enable comparison plots):
- `rm_protocol_viz_compare3.py` — provides `compare3()` used by `rm_oneclick_protocol_run3.py`

---

## Requirements
- Python **3.9+**
- PyTorch **2.x** (CUDA recommended)
- Common packages: `numpy`, `Pillow`, `tqdm` (recommended)

---

## Installation
Example setup:

```bash
conda create -n radiomap2 python=3.10 -y
conda activate radiomap2

# Install the correct PyTorch build for your CUDA
# https://pytorch.org/get-started/locally/

pip install numpy pillow tqdm
```

---

## Dataset Layout
The dataset is organized by **map** and **TX instance**.

Typical items:
- Target map (DPM / RSRP): `mapId_txId.png` (or `.npy`)
- Building mask: `mapId.png`
- Antenna metadata: `mapId.json`
- Optional LoS map: `mapId_txId.npy`

Typical directory layout:

```text
DATASET_ROOT/
  gain/DPM/                 # target_subdir: {mapId}_{txId}.png or .npy
  png/buildings_complete/   # buildings_subdir: {mapId}.png
  antenna/                  # antenna_subdir: {mapId}.json
  los_wallcount/            # (optional) los_subdir: {mapId}_{txId}.npy
  splits/
    train_maps.txt
    val_maps.txt
    test_maps.txt
```

Splits are **map-level** lists (one map id per line). For each map, the dataset may sample up to `max_tx_per_map` TX instances.

---

## Configuration: TOTAL_CFG
All experiments are described by `TOTAL_CFG` (recommended: edit `rm_total_config.py`).

### 1) `TOTAL_CFG["protocol"]`
- `split`: dataset_root, subdirs, split lists, and sampling options
- `input.layout`: ordered list of input channels
- `input.roles`: how GCGO slices public `x` into `{geom, f_src}`
- `target_norm`: target transform and normalization
- `eval`: evaluator settings (masking, boundary ring, high-value mask, dist bins, composite weights)

### 2) `TOTAL_CFG["train"]`
- output dir, run name, seed, epochs, batch size, amp, etc.

### 3) `TOTAL_CFG["model_fno"] / ["model_unet"] / ["model_gcgo"]`
- model-specific hyperparameters

### UNet channel constraint (important)
The official UNet baseline is intentionally strict: it expects **exactly 2 input channels**. A common choice is:

```python
input.layout = ["building", "tx_gaussian"]
```

If you include extra channels (e.g., `tx_invdist`, `dx/dy`, `los_wallcount`), use FNO/GCGO, or create a separate UNet variant that supports more channels.

---

## Quick Start (One-Click)
`rm_oneclick_protocol_run3.py` orchestrates:
- per-model **train / reuse / reuse_if_exists**
- optional **auto split generation** (if split files do not exist)
- unified **compare3** visualization/evaluation

### Run

```bash
python rm_oneclick_protocol_run3.py
```

### Configure actions
Open `rm_oneclick_protocol_run3.py` and edit the `CFG` dict:
- `CFG["ACTION"]["fno"|"unet"|"gcgo"] = "train" | "reuse" | "reuse_if_exists"`
- optionally set `CFG["CKPT_OVERRIDE"][...]` to force specific checkpoints

### Multi-GPU behavior
The one-click script must be run with **python (single process)**. It will spawn `torchrun` subprocesses per model.

Environment variables:

```bash
# launch mode for the per-model subprocess
export RM_RUN_MODE=torchrun   # torchrun | python

# number of GPUs used by each per-model torchrun
export RM_NPROC_PER_NODE=4

python rm_oneclick_protocol_run3.py
```

---

## Train Individually
All three training scripts accept a protocol snapshot JSON via:
- CLI: `--total_cfg_json /path/to/total_cfg.json`
- ENV: `RM_TOTAL_CFG_JSON=/path/to/total_cfg.json` (or `TOTAL_CFG_JSON`)
- fallback: import `TOTAL_CFG` from `rm_total_config.py`

### FNO

```bash
torchrun --nproc_per_node=4 rm_train_fno_protocol_allinone.py \
  --total_cfg_json /path/to/total_cfg.json
```

### UNet

```bash
torchrun --nproc_per_node=4 rm_train_unet_protocol_allinone.py \
  --total_cfg_json /path/to/total_cfg.json
```

### GCGO

```bash
torchrun --nproc_per_node=4 rm_train_gcgo_protocol_allinone.py \
  --total_cfg_json /path/to/total_cfg.json
```

Single GPU / CPU debugging:

```bash
python rm_train_fno_protocol_allinone.py --total_cfg_json /path/to/total_cfg.json
```

---

## Evaluation & Metrics
`ProtocolEvaluator` supports consistent evaluation across models.

Common metrics:
- **RMSE**: root mean squared error
- **Boundary RMSE**: error on a ring around building boundaries
- **High-value RMSE**: error on the high-quantile region (e.g., top q)
- **Composite score**: weighted sum used for best-checkpoint selection

Diagnostics:
- error **percentiles**
- **distance-bin** MAE (radial profiling)

Masking:
- `mask_buildings=True` evaluates only **outdoor** areas (derived from the building channel)

---

## Outputs
Each training run writes to `TOTAL_CFG["train"]["out_dir"]`.

Typical layout:

```text
OUT_DIR/
  run_name_YYYYMMDD_HHMMSS/
    ckpts/
      best.pt
      epoch_XXX.pt
    history.json
    best.json
    total_cfg.snapshot.json
```

Notes:
- FNO/UNet: best checkpoint is typically `ckpts/best.pt`
- GCGO: best checkpoint is typically `ckpt_best.pt` under the run folder

---

## Reproducibility
Recommended checklist:
1. Keep a frozen `total_cfg.snapshot.json` for every run.
2. Fix `train.seed`.
3. Record your exact command and environment (CUDA/PyTorch versions).

---

## Troubleshooting
- **buildings=0 / targets=0**: your `dataset_root` or subdir names do not match the filesystem. Double-check `TOTAL_CFG["protocol"]["split"]`.
- **Running from the wrong directory**: the one-click runner resolves dataset_root relative to `rm_total_config.py`, but standalone scripts may resolve relative paths differently. Prefer absolute paths or provide a JSON snapshot.
- **UNet input mismatch**: official UNet requires 2 input channels.
- **DDP hangs**: ensure you use `torchrun` (not `python`) for multi-GPU training scripts.

---

## Citation
If you use this code in academic work, please cite your paper or this repository. Example:

```bibtex
@misc{radiomapGCGO,
  title        = {Learning Geometry-Conditioned Green Operators for Urban Radio Maps},
  author       = {Yi Zheng},
  year         = {2026},
  howpublished = {Under Review},
}
```

---

## License
Add a `LICENSE` file (MIT / Apache-2.0 recommended if compatible with dependencies).
