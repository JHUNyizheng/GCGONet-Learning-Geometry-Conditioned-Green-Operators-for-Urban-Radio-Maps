# -*- coding: utf-8 -*-
"""
rm_total_config.py
============================================================
【总配置模板】（Python 版，带中文注释）
- 你可以在这个文件里统一管理：协议(输入/目标/评估/保存) + 训练超参 + 模型超参
- 任何训练脚本（FNO / UNet / GCGO）都只 import 这个 TOTAL_CFG

用法：
  from rm_total_config import TOTAL_CFG

============================================================
"""

TOTAL_CFG = {
    # ============================================================
    # 1) Protocol：公开信息口径（对比协议）
    # ============================================================
    "protocol": {
        "split": {
            # 数据集根目录（相对路径会按“训练脚本所在目录”解析）
            "dataset_root": ".",

            # ===== 自动划分 splits（oneclick 可选启用）=====
            # 若 splits/*.txt 不存在，rm_oneclick_protocol_run3.py 会按以下参数自动生成
            "auto_make_splits": True,
            "force_remake_splits": False,   # True 会覆盖已有 splits
            "split_seed": 123,              # 划分随机种子
            "val_map_ratio": 0.1,
            "test_map_ratio": 0.1,
            "max_maps": 700,         # 调试用：仅使用前 N 张地图（None 表示全用）
            # 目标场目录（RSRP / gain）——每个样本一张：mapId_txId.png 或 .npy
            "target_subdir": "gain/DPM",

            # 建筑物 mask（每张地图一张）：mapId.png；像素 1=building,0=free（若反了就用 input.building_invert）
            "buildings_subdir": "png/buildings_complete",

            # 天线/发射机轨迹（每张地图一个 json）：提供 tx 坐标序列（用于生成 tx_gaussian / tx_invdist / dy/dx）
            "antenna_subdir": "antenna",

            # 如果 target 是 .npy 且为多通道，可以指定取第几通道
            "npy_channel": 0,

            # split 列表：可以是 txt（map_id 一行一个）或 json（样本列表）
            "train_list_path": "splits/train_maps.txt",
            "val_list_path": "splits/val_maps.txt",
            "test_list_path": "splits/test_maps.txt",

            # 每张 map 最多用多少个 tx（None 表示全用）
            "max_tx_per_map": 80,

            # map_select: "first" | "random" —— 在 max_maps 截断时如何挑地图
            "map_select": "first",
        },

        "input": {
            # ===== 输入协议（主对比：P0+ 5 通道）=====
            # 说明：layout 的顺序就是 x 的通道顺序；严格一致才能对齐实验
            "layout": ["building", "tx_gaussian", "tx_invdist", "dy", "dx", "los_wallcount"],

            # ===== 输入通道角色（同一份 x 的切片分组）=====
            # 目的：保证三模型公平（不引入私货），同时允许不同模型使用不同子集通道。
            "roles": {
                # GCGO: 从 layout 里切出源项图和几何图
                "gcgo": {"source_channel": "tx_gaussian", "geom_channels": ["building", "tx_invdist", "dy", "dx", "los_wallcount"]},
                # UNet: 只用两个通道（严格对齐官方 RadioUNet 输入）
                "unet": ["building", "tx_gaussian"],
            },

            "construct": {
                # 输入分辨率 (H,W)
                "image_size": [256, 256],

                # tx_gaussian 的 sigma（像素单位）
                "tx_gaussian_sigma": 2.0,

                # tx_invdist 的 eps（避免 1/0）
                "tx_invdist_eps": 1.0,

                # LoS / wallcount 预计算通道（当 layout 包含 "los_wallcount" 时启用）
                # 文件路径: <dataset_root>/<los_subdir>/<map_id>_<tx_id>.npy
                "los_subdir": "los_wallcount",
                # 轻量归一化：clip 后除以 scale_div
                "los_clip_max": 20.0,
                "los_scale_div": 20.0,
            },

            "x_norm": {
                # x 的归一化方式：
                # - "none": 不做
                # - "zscore": 需要 x_stats_path（每通道 mean/std）
                "mode": "none",
                "x_stats_path": None,
            },
        },

        "target_norm": {
            # 统一说明：模型训练/评估都在同一个 y-space 下对齐
            # fno_target: 你旧脚本里的目标空间（通常是对 gain/rsrp 做 log/clip/minmax）
            "y_space_train": "fno_target",
            "y_space_eval_main": "fno_target",

            # y 的变换：none | log1p | clip | log1p+clip
            "y_transform": "log1p+clip",

            # clip 的范围 [lo, hi]，None 表示训练脚本会自动用 train set 估计（分位数法）
            "y_clip_range": [9.999995427278918e-07, 5.181783676147461],

            # y 的归一化：none | global_minmax | zscore
            "y_norm": "global_minmax",

            # y 的统计量（当 y_norm 需要 stats 时）
            "y_stats_path": None,
        },

        "eval": {
            # 评估在哪个空间对齐（通常与 y_space_eval_main 一致）
            "compare_space": "fno_target",

            # 是否 mask 掉建筑内部（True = 只算 outdoor）
            "mask_buildings": True,

            # boundary ring 宽度（像素），用于 boundary_rmse/mae
            "boundary_width": 3,

            # high 区域分位数（0~1），用于 high_rmse/mae
            "high_q": 0.90,

            # eval 时是否把 pred clamp 到 [0,1]
            "post_clamp01": False,

            # 预测/GT 分辨率不一致时的处理
            "resize_pred_to_gt": True,
            "resize_mode": "bilinear",

            # 用于 best 选择的综合指标：composite = Σ w_i * metric_i
            "composite_weights": {"rmse": 1.0, "boundary_rmse": 0.5, "high_rmse": 0.5},
            # 评估指标口径（训练/验证/测试统一）：rmse + boundary_rmse + high_rmse
            "eval_metrics": ["rmse", "boundary_rmse", "high_rmse"],

            # 误差分位数（用于“尾部误差/极端误差”诊断）
            "error_percentiles": {"enabled": True, "percentiles": [50, 90, 99], "sample_per_image": 800},

            # 按距 TX 的距离分桶统计误差（用于诊断“光圈误差/径向规律”）
            "dist_bins": {"enabled": True, "bins": [0, 1, 2, 4, 8, 16, 32, 64, 1e9]},
        },
    },

    # ============================================================
    # 2) Train：训练超参（与模型无关）
    # ============================================================
    "train": {
        "device": "cuda",
        "seed": 123,

        "epochs": 60,
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,

        "lr": 1e-4,
        "weight_decay": 1e-4,

        # AMP：fp16 / bf16
        "amp": True,
        "amp_dtype": "fp16",

        # 梯度累计 & 裁剪
        "grad_accum": 1,
        "grad_clip": 1.0,

        # loss 配置（对齐旧脚本）
        "loss_type": "huber",          # l1 | mse | huber
        "huber_delta": 0.05,
        "loss_mask_buildings": True,   # loss 只在 outdoor 计算
        "loss_weight_boundary": 0.5,
        "boundary_width": 3,
        "loss_weight_high": 0.5,
        "high_q": 0.90,
        "tv_weight": 0.0,
        "radial_weight": 0.0,
        "radial_corr_target": 0.2,

        # 输出目录与保存策略
        "out_dir": "./rm_runs",
        "run_name": "exp",
        "save_every": 0,

        # best 选择口径：composite / rmse / mae / boundary_rmse / high_rmse ...
        "save_best_by": "composite",

        # 断点续训 ckpt 路径（None 表示不启用）
        "resume_ckpt": None,
    },

    # ============================================================
    # 3) Model：三个模型的显式配置（避免“隐式默认值”歧义）
    # ============================================================

    # --- FNO ---
    # 说明：为了兼容旧脚本/旧 checkpoint，这里同时保留 "model" (旧 key) 和 "model_fno" (新 key)。
    "model_fno": {
        # FNO 频域模态数（modes1/modes2）
        "modes": 64,

        # 宽度（隐层通道数）
        "width": 96,

        # SpectralConv block 数
        "layers": 4,

        # residual learning：是否 out = base(tx_invdist) + f(x)
        "residual_learning": True,

        # baseline_index: "auto" 表示从 input.layout 里找 tx_invdist 的通道位置
        "baseline_index": "auto",

        # residual 后是否 clamp 到 [0,1]
        "residual_clamp01": True,
    },

    # 兼容旧 key：保持与 model_fno 完全一致
    "model": None,

    # --- UNet ---
    # 官方 RadioUNet baseline（严格：仅 2 通道 building+tx_gaussian）
    "model_unet": {
        "type": "official",   # official | modern（当前工程默认 official）
        "inputs": 2,           # 必须为 2（由 roles.unet 严格校验）
    },

    # --- GCGO (GeoGreen-Op) ---
    "model_gcgo": {
        # Encoder
        "h_dim": 128,             # local embedding h(x)
        "z_dim": 128,             # global latent z_Ω
        "enc_base": 64,

        # Field feature width
        "width": 96,

        # GeoGreen blocks
        "L": 6,
        "modes": 32,
        "rank": 32,
        "dropout": 0.0,

        # Geometry-conditioned spectral kernel
        "k_embed_freqs": [1, 2, 4, 8],
        "kernel_hidden": 256,
        "kernel_clamp": 2.0,

        # Low-rank correction
        "corr_scale": 0.5,
        "corr_inner": "u_proj",   # f_src | u_mean | u_proj

        # Optional band embedding（默认关闭）
        "use_band_emb": False,
        "n_bands": 1,
        "band_emb_dim": 8,
    },
}

# ---- Fill compatibility alias ----
# Keep TOTAL_CFG["model"] as alias to model_fno for backward-compat.
if TOTAL_CFG.get("model") is None:
    TOTAL_CFG["model"] = TOTAL_CFG.get("model_fno", {})
