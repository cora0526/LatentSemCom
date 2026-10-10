from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image


# ============================================================
# Import your training file
# ============================================================
# The script tries these module names in order.
# Put this evaluation file in the SAME folder as your v3 training file.
TRAIN_MODULE_CANDIDATES = [
    "train_gan_pretrain_then_joint_mu_vaegan_label_cond_siso_ddpm_v3_fixed",
    "train_gan_pretrain_then_joint_mu_vaegan_label_cond_siso_ddpm_v3",
]


def import_training_module():
    last_error = None
    for name in TRAIN_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(name)
            print(f"Imported training module: {name}.py")
            return module
        except ModuleNotFoundError as exc:
            last_error = exc
    raise ModuleNotFoundError(
        "Could not import the v3 training file. Put this script in the same folder as "
        f"one of: {TRAIN_MODULE_CANDIDATES}"
    ) from last_error


M = import_training_module()


# ============================================================
# User settings
# ============================================================
HISTORY_CSV = Path(
    "./save_data/Deep_WGANGP_VAE_DDPM/"
    "ddpm_label_cond_siso_random_alpha_training_history.csv"
)

OUTPUT_ROOT = Path(
    "./save_data/Deep_WGANGP_VAE_DDPM/random_alpha_model_test"
)

# Fixed channel SNR for alpha sweep.
FIXED_SNR_DB = 10.0

# alpha sweep: 0.0, 0.1, ..., 2.0
ALPHAS = [round(float(x), 2) for x in np.arange(0.0, 2.01, 0.1)]

# Number of paired images used for quantitative evaluation.
EVAL_COUNT = 128
BATCH_SIZE = 32

# Save image comparisons only at these interference strengths.
QUALITATIVE_ALPHAS = {0.0, 0.5, 1.0, 1.5, 2.0}
QUALITATIVE_COUNT = 8

# Multi-interferer robustness sweep.
# The trained DDPM saw one interferer during training; K>1 therefore tests
# generalization to a harder, out-of-distribution interference condition.
MULTI_INTERFERER_COUNTS = [1, 2, 3, 4, 5]
MULTI_INTERFERER_ALPHAS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
MULTI_EVAL_COUNT = 64

# Muted, color-blind-friendly palette (Okabe-Ito inspired).
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#6F6F6F",
}

K_COLORS = [
    COLORS["blue"],
    COLORS["green"],
    COLORS["orange"],
    COLORS["vermillion"],
    COLORS["purple"],
]
K_MARKERS = ["o", "s", "^", "D", "v"]
K_LINESTYLES = ["-", "--", "-.", ":", "-"]


# def style_axis(ax) -> None:
#     """Light paper-style formatting shared by all plots."""
#     ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

def style_axis(ax) -> None:
    """Light paper-style formatting shared by all plots."""
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)

    # Keep all four borders
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)


# ============================================================
# Utilities
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_01(x: torch.Tensor) -> torch.Tensor:
    return ((x.detach().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred_01 = to_01(pred)
    target_01 = to_01(target)
    mse = (
        (pred_01 - target_01)
        .pow(2)
        .flatten(start_dim=1)
        .mean(dim=1)
        .clamp_min(1e-12)
    )
    return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred_np = to_01(pred).permute(0, 2, 3, 1).cpu().numpy()
    target_np = to_01(target).permute(0, 2, 3, 1).cpu().numpy()

    values = []
    for p, t in zip(pred_np, target_np):
        values.append(
            structural_similarity(
                t,
                p,
                data_range=1.0,
                channel_axis=-1,
            )
        )
    return np.asarray(values, dtype=np.float64)


def save_rows(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_grid(
    path: Path,
    real: torch.Tensor,
    no_sep: torch.Tensor,
    ddpm: torch.Tensor,
    count: int,
) -> None:
    count = min(count, real.size(0))
    grid_tensor = torch.cat(
        [real[:count], no_sep[:count], ddpm[:count]],
        dim=0,
    )
    grid = make_grid(
        grid_tensor,
        nrow=count,
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(grid, str(path))
    path.with_suffix(".txt").write_text(
        "Row 1: Real target\n"
        "Row 2: No separation\n"
        "Row 3: Random-alpha DDPM\n",
        encoding="utf-8",
    )


# ============================================================
# Part A: plot DDPM training CSV
# ============================================================
def plot_training_history(csv_path: Path, output_dir: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path.resolve()}")

    ensure_dir(output_dir)
    df = pd.read_csv(csv_path)

    required = [
        "epoch",
        "total_loss",
        "eps_eurosat",
        "eps_deepglobe",
        "x0_eurosat",
        "x0_deepglobe",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(
            f"CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    epoch = df["epoch"].to_numpy()

    # 1. Total loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epoch,
        df["total_loss"],
        color=COLORS["blue"],
        linewidth=2.0,
        marker="o",
        markersize=3,
        markevery=max(1, len(epoch) // 20),
        label="Total Loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    # ax.set_title("DDPM Training Total Loss")
    style_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ddpm_total_loss.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ddpm_total_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Epsilon-prediction loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epoch,
        df["eps_eurosat"],
        color=COLORS["blue"],
        linewidth=1.9,
        label="EuroSAT",
    )
    ax.plot(
        epoch,
        df["eps_deepglobe"],
        color=COLORS["orange"],
        linewidth=1.9,
        label="DeepGlobe",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    # ax.set_title("DDPM Noise-Prediction Loss")
    style_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ddpm_epsilon_loss.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ddpm_epsilon_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. x0 latent reconstruction loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epoch,
        df["x0_eurosat"],
        color=COLORS["blue"],
        linewidth=1.9,
        label="EuroSAT",
    )
    ax.plot(
        epoch,
        df["x0_deepglobe"],
        color=COLORS["orange"],
        linewidth=1.9,
        label="DeepGlobe",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Latent MSE")
    # ax.set_title("DDPM Clean-Latent Reconstruction Loss")
    style_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ddpm_x0_loss.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ddpm_x0_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    # 4. Mean random alpha used during training
    alpha_columns = ["alpha_eurosat_mean", "alpha_deepglobe_mean"]
    if all(column in df.columns for column in alpha_columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            epoch,
            df["alpha_eurosat_mean"],
            color=COLORS["blue"],
            linewidth=1.9,
            label="EuroSAT target",
        )
        ax.plot(
            epoch,
            df["alpha_deepglobe_mean"],
            color=COLORS["orange"],
            linewidth=1.9,
            label="DeepGlobe target",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"Mean interference amplitude $\alpha$")
        # ax.set_title("Random Interference Strength During DDPM Training")
        style_axis(ax)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / "ddpm_alpha_mean.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "ddpm_alpha_mean.pdf", bbox_inches="tight")
        plt.close(fig)

    # 5. Log-scale loss
    fig, ax = plt.subplots(figsize=(8, 5))
    safe_loss = np.maximum(df["total_loss"].to_numpy(dtype=float), 1e-12)
    ax.semilogy(
        epoch,
        safe_loss,
        color=COLORS["blue"],
        linewidth=2.0,
        label="Total Loss",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    # ax.set_title("DDPM Training Total Loss — Log Scale")
    style_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "ddpm_total_loss_log.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ddpm_total_loss_log.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Training-history plots saved to: {output_dir.resolve()}")
    print(f"CSV rows: {len(df)}")
    if len(df) > 0:
        print(f"Latest epoch: {int(df.iloc[-1]['epoch'])}")
        print(f"Latest total loss: {float(df.iloc[-1]['total_loss']):.6f}")
        best_idx = int(df["total_loss"].astype(float).idxmin())
        print(
            "Best recorded total loss: "
            f"{float(df.loc[best_idx, 'total_loss']):.6f} "
            f"at epoch {int(df.loc[best_idx, 'epoch'])}"
        )


# ============================================================
# Fixed evaluation data
# ============================================================
def build_fixed_eval_loaders(cfg, eval_count: int = EVAL_COUNT):
    eurosat, deepglobe = M.build_domain_datasets(cfg)
    count = min(eval_count, len(eurosat), len(deepglobe))

    subset_e = Subset(eurosat, list(range(count)))
    subset_d = Subset(deepglobe, list(range(count)))

    loader_e = DataLoader(
        subset_e,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    loader_d = DataLoader(
        subset_d,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return loader_e, loader_d, count


# ============================================================
# Part B: alpha sweep evaluation
# ============================================================
@torch.no_grad()
def evaluate_one_alpha(
    cfg,
    device,
    encoder,
    generator,
    ddpm,
    schedule,
    loader_e,
    loader_d,
    alpha: float,
    qualitative_dir: Path,
) -> List[dict]:
    result: Dict[str, Dict[str, list]] = {
        "eurosat": {
            "psnr_clean": [],
            "psnr_no_sep": [],
            "psnr_ddpm": [],
            "ssim_clean": [],
            "ssim_no_sep": [],
            "ssim_ddpm": [],
            "latent_mse": [],
            "latent_cosine": [],
        },
        "deepglobe": {
            "psnr_clean": [],
            "psnr_no_sep": [],
            "psnr_ddpm": [],
            "ssim_clean": [],
            "ssim_no_sep": [],
            "ssim_ddpm": [],
            "latent_mse": [],
            "latent_cosine": [],
        },
    }

    qualitative_saved = False

    for batch_index, (batch_e, batch_d) in enumerate(zip(loader_e, loader_d)):
        image_e, _ = batch_e
        image_d, _ = batch_d
        image_e = image_e.to(device)
        image_d = image_d.to(device)

        batch = min(image_e.size(0), image_d.size(0))
        image_e = image_e[:batch]
        image_d = image_d[:batch]

        mu_e, _ = encoder(image_e)
        mu_d, _ = encoder(image_d)
        z_e = M.apply_global_latent_scale(mu_e, cfg)
        z_d = M.apply_global_latent_scale(mu_d, cfg)

        labels_e = torch.full(
            (batch,), M.EUROSAT_LABEL, device=device, dtype=torch.long
        )
        labels_d = torch.full(
            (batch,), M.DEEPGLOBE_LABEL, device=device, dtype=torch.long
        )

        # Clean reconstruction ceiling.
        clean_recon_e = generator(mu_e, labels_e)
        clean_recon_d = generator(mu_d, labels_d)

        alpha_tensor = torch.full(
            (batch,), float(alpha), device=device, dtype=z_e.dtype
        )

        # ----------------------------------------------------
        # EuroSAT target: y_E = z_E + alpha*z_D + n
        # ----------------------------------------------------
        torch.manual_seed(cfg.seed + int(round(alpha * 1000)) + 10000 + batch_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                cfg.seed + int(round(alpha * 1000)) + 10000 + batch_index
            )

        obs_e, _ = M.target_centered_siso_observation(
            z_target=z_e,
            z_interference=z_d,
            alpha=alpha_tensor,
            snr_db=FIXED_SNR_DB,
        )

        no_sep_mu_e = M.invert_global_latent_scale(obs_e, cfg)
        no_sep_recon_e = generator(no_sep_mu_e, labels_e)

        # Fixed sampling seed across alpha values for a cleaner comparison.
        torch.manual_seed(cfg.seed + 500000 + batch_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + 500000 + batch_index)
        init_noise_e = torch.randn(batch, cfg.z_dim, device=device)

        ddpm_scaled_e = M.sample_conditional_ddpm(
            model=ddpm,
            schedule=schedule,
            observation=obs_e,
            labels=labels_e,
            z_dim=cfg.z_dim,
            initial_noise=init_noise_e,
        )
        ddpm_mu_e = M.invert_global_latent_scale(ddpm_scaled_e, cfg)
        ddpm_recon_e = generator(ddpm_mu_e, labels_e)

        # ----------------------------------------------------
        # DeepGlobe target: y_D = z_D + alpha*z_E + n
        # ----------------------------------------------------
        torch.manual_seed(cfg.seed + int(round(alpha * 1000)) + 20000 + batch_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                cfg.seed + int(round(alpha * 1000)) + 20000 + batch_index
            )

        obs_d, _ = M.target_centered_siso_observation(
            z_target=z_d,
            z_interference=z_e,
            alpha=alpha_tensor,
            snr_db=FIXED_SNR_DB,
        )

        no_sep_mu_d = M.invert_global_latent_scale(obs_d, cfg)
        no_sep_recon_d = generator(no_sep_mu_d, labels_d)

        torch.manual_seed(cfg.seed + 600000 + batch_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + 600000 + batch_index)
        init_noise_d = torch.randn(batch, cfg.z_dim, device=device)

        ddpm_scaled_d = M.sample_conditional_ddpm(
            model=ddpm,
            schedule=schedule,
            observation=obs_d,
            labels=labels_d,
            z_dim=cfg.z_dim,
            initial_noise=init_noise_d,
        )
        ddpm_mu_d = M.invert_global_latent_scale(ddpm_scaled_d, cfg)
        ddpm_recon_d = generator(ddpm_mu_d, labels_d)

        # Metrics — EuroSAT
        result["eurosat"]["psnr_clean"].extend(
            batch_psnr(clean_recon_e, image_e).tolist()
        )
        result["eurosat"]["psnr_no_sep"].extend(
            batch_psnr(no_sep_recon_e, image_e).tolist()
        )
        result["eurosat"]["psnr_ddpm"].extend(
            batch_psnr(ddpm_recon_e, image_e).tolist()
        )
        result["eurosat"]["ssim_clean"].extend(
            batch_ssim(clean_recon_e, image_e).tolist()
        )
        result["eurosat"]["ssim_no_sep"].extend(
            batch_ssim(no_sep_recon_e, image_e).tolist()
        )
        result["eurosat"]["ssim_ddpm"].extend(
            batch_ssim(ddpm_recon_e, image_e).tolist()
        )
        result["eurosat"]["latent_mse"].extend(
            (ddpm_scaled_e - z_e).pow(2).mean(dim=1).cpu().tolist()
        )
        result["eurosat"]["latent_cosine"].extend(
            F.cosine_similarity(ddpm_scaled_e, z_e, dim=1).cpu().tolist()
        )

        # Metrics — DeepGlobe
        result["deepglobe"]["psnr_clean"].extend(
            batch_psnr(clean_recon_d, image_d).tolist()
        )
        result["deepglobe"]["psnr_no_sep"].extend(
            batch_psnr(no_sep_recon_d, image_d).tolist()
        )
        result["deepglobe"]["psnr_ddpm"].extend(
            batch_psnr(ddpm_recon_d, image_d).tolist()
        )
        result["deepglobe"]["ssim_clean"].extend(
            batch_ssim(clean_recon_d, image_d).tolist()
        )
        result["deepglobe"]["ssim_no_sep"].extend(
            batch_ssim(no_sep_recon_d, image_d).tolist()
        )
        result["deepglobe"]["ssim_ddpm"].extend(
            batch_ssim(ddpm_recon_d, image_d).tolist()
        )
        result["deepglobe"]["latent_mse"].extend(
            (ddpm_scaled_d - z_d).pow(2).mean(dim=1).cpu().tolist()
        )
        result["deepglobe"]["latent_cosine"].extend(
            F.cosine_similarity(ddpm_scaled_d, z_d, dim=1).cpu().tolist()
        )

        if alpha in QUALITATIVE_ALPHAS and not qualitative_saved:
            ensure_dir(qualitative_dir)
            save_comparison_grid(
                qualitative_dir / f"eurosat_target_alpha_{alpha:.2f}.png",
                image_e,
                no_sep_recon_e,
                ddpm_recon_e,
                QUALITATIVE_COUNT,
            )
            save_comparison_grid(
                qualitative_dir / f"deepglobe_target_alpha_{alpha:.2f}.png",
                image_d,
                no_sep_recon_d,
                ddpm_recon_d,
                QUALITATIVE_COUNT,
            )
            qualitative_saved = True

    rows = []
    for domain in ("eurosat", "deepglobe"):
        data = result[domain]
        rows.append(
            {
                "target": domain,
                "alpha": float(alpha),
                "fixed_snr_db": float(FIXED_SNR_DB),
                "num_samples": len(data["psnr_ddpm"]),
                "psnr_clean": float(np.mean(data["psnr_clean"])),
                "psnr_no_separation": float(np.mean(data["psnr_no_sep"])),
                "psnr_ddpm": float(np.mean(data["psnr_ddpm"])),
                "ssim_clean": float(np.mean(data["ssim_clean"])),
                "ssim_no_separation": float(np.mean(data["ssim_no_sep"])),
                "ssim_ddpm": float(np.mean(data["ssim_ddpm"])),
                "latent_mse_ddpm": float(np.mean(data["latent_mse"])),
                "latent_cosine_ddpm": float(np.mean(data["latent_cosine"])),
            }
        )
    return rows


# ============================================================
# Evaluation plotting
# ============================================================
def plot_domain_metrics(rows: List[dict], domain: str, output_dir: Path) -> None:
    selected = sorted(
        [row for row in rows if row["target"] == domain],
        key=lambda x: x["alpha"],
    )
    x = [row["alpha"] for row in selected]

    # PSNR
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        [r["psnr_clean"] for r in selected],
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.8,
        marker="s",
        markersize=4,
        label="Clean Latent",
    )
    ax.plot(
        x,
        [r["psnr_no_separation"] for r in selected],
        color=COLORS["gray"],
        linestyle=":",
        linewidth=2.0,
        marker="D",
        markersize=4,
        label="No Separation",
    )
    ax.plot(
        x,
        [r["psnr_ddpm"] for r in selected],
        color=COLORS["blue"],
        linestyle="-",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Random-$\\alpha$ DDPM",
    )
    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("PSNR (dB)")
    # ax.set_title(f"{domain.capitalize()} PSNR vs Interference Strength")
    style_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{domain}_PSNR_vs_alpha.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{domain}_PSNR_vs_alpha.pdf", bbox_inches="tight")
    plt.close(fig)

    # SSIM
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        [r["ssim_clean"] for r in selected],
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.8,
        marker="s",
        markersize=4,
        label="Clean Latent",
    )
    ax.plot(
        x,
        [r["ssim_no_separation"] for r in selected],
        color=COLORS["gray"],
        linestyle=":",
        linewidth=2.0,
        marker="D",
        markersize=4,
        label="No Separation",
    )
    ax.plot(
        x,
        [r["ssim_ddpm"] for r in selected],
        color=COLORS["blue"],
        linestyle="-",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Random-$\\alpha$ DDPM",
    )
    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("SSIM")
    # ax.set_title(f"{domain.capitalize()} SSIM vs Interference Strength")
    style_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{domain}_SSIM_vs_alpha.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{domain}_SSIM_vs_alpha.pdf", bbox_inches="tight")
    plt.close(fig)

    # Latent MSE
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        [r["latent_mse_ddpm"] for r in selected],
        color=COLORS["vermillion"],
        linewidth=2.1,
        marker="o",
        markersize=4,
    )
    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("Latent MSE")
    # ax.set_title(f"{domain.capitalize()} Latent MSE vs Interference Strength")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{domain}_latent_MSE_vs_alpha.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{domain}_latent_MSE_vs_alpha.pdf", bbox_inches="tight")
    plt.close(fig)

    # Latent cosine similarity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        x,
        [r["latent_cosine_ddpm"] for r in selected],
        color=COLORS["purple"],
        linewidth=2.1,
        marker="o",
        markersize=4,
    )
    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("Latent Cosine Similarity")
    # ax.set_title(f"{domain.capitalize()} Latent Cosine Similarity vs Interference Strength")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output_dir / f"{domain}_latent_cosine_vs_alpha.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{domain}_latent_cosine_vs_alpha.pdf", bbox_inches="tight")
    plt.close(fig)


def make_interference_sum(z_interference: torch.Tensor, num_interferers: int) -> torch.Tensor:
    """Construct sum_{k=1}^K z_k using deterministic circular shifts.

    The first interferer is the original batch. Additional interferers are
    different samples from the same interference-domain batch obtained by
    circular shifts. This keeps the experiment reproducible and avoids loading
    K separate datasets.
    """
    if num_interferers < 1:
        raise ValueError("num_interferers must be >= 1")

    batch = z_interference.size(0)
    if batch < num_interferers:
        raise ValueError(
            f"Batch size ({batch}) must be >= number of interferers ({num_interferers})."
        )

    pieces = [torch.roll(z_interference, shifts=i, dims=0) for i in range(num_interferers)]
    return torch.stack(pieces, dim=0).sum(dim=0)


@torch.no_grad()
def evaluate_multi_interferer_setting(
    cfg,
    device,
    encoder,
    generator,
    ddpm,
    schedule,
    loader_e,
    loader_d,
    alpha: float,
    num_interferers: int,
) -> List[dict]:
    """Evaluate K-interferer robustness.

    EuroSAT target:
        y_E = z_E + alpha * sum_{k=1}^K z_{D,k} + n

    DeepGlobe target:
        y_D = z_D + alpha * sum_{k=1}^K z_{E,k} + n
    """
    result = {
        "eurosat": {"psnr": [], "ssim": [], "latent_mse": [], "latent_cosine": []},
        "deepglobe": {"psnr": [], "ssim": [], "latent_mse": [], "latent_cosine": []},
    }

    for batch_index, (batch_e, batch_d) in enumerate(zip(loader_e, loader_d)):
        image_e, _ = batch_e
        image_d, _ = batch_d
        image_e = image_e.to(device)
        image_d = image_d.to(device)

        batch = min(image_e.size(0), image_d.size(0))
        image_e = image_e[:batch]
        image_d = image_d[:batch]

        mu_e, _ = encoder(image_e)
        mu_d, _ = encoder(image_d)
        z_e = M.apply_global_latent_scale(mu_e, cfg)
        z_d = M.apply_global_latent_scale(mu_d, cfg)

        labels_e = torch.full(
            (batch,), M.EUROSAT_LABEL, device=device, dtype=torch.long
        )
        labels_d = torch.full(
            (batch,), M.DEEPGLOBE_LABEL, device=device, dtype=torch.long
        )

        # Build K interferers from the opposite domain.
        z_d_sum = make_interference_sum(z_d, num_interferers)
        z_e_sum = make_interference_sum(z_e, num_interferers)

        alpha_tensor = torch.full(
            (batch,), float(alpha), device=device, dtype=z_e.dtype
        )

        # Same base Gaussian channel-noise realization across alpha/K conditions.
        torch.manual_seed(cfg.seed + 710000 + batch_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + 710000 + batch_index)
        obs_e, _ = M.target_centered_siso_observation(
            z_target=z_e,
            z_interference=z_d_sum,
            alpha=alpha_tensor,
            snr_db=FIXED_SNR_DB,
        )

        torch.manual_seed(cfg.seed + 720000 + batch_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + 720000 + batch_index)
        init_noise_e = torch.randn(batch, cfg.z_dim, device=device)
        ddpm_scaled_e = M.sample_conditional_ddpm(
            model=ddpm,
            schedule=schedule,
            observation=obs_e,
            labels=labels_e,
            z_dim=cfg.z_dim,
            initial_noise=init_noise_e,
        )
        ddpm_mu_e = M.invert_global_latent_scale(ddpm_scaled_e, cfg)
        ddpm_recon_e = generator(ddpm_mu_e, labels_e)

        torch.manual_seed(cfg.seed + 730000 + batch_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + 730000 + batch_index)
        obs_d, _ = M.target_centered_siso_observation(
            z_target=z_d,
            z_interference=z_e_sum,
            alpha=alpha_tensor,
            snr_db=FIXED_SNR_DB,
        )

        torch.manual_seed(cfg.seed + 740000 + batch_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed + 740000 + batch_index)
        init_noise_d = torch.randn(batch, cfg.z_dim, device=device)
        ddpm_scaled_d = M.sample_conditional_ddpm(
            model=ddpm,
            schedule=schedule,
            observation=obs_d,
            labels=labels_d,
            z_dim=cfg.z_dim,
            initial_noise=init_noise_d,
        )
        ddpm_mu_d = M.invert_global_latent_scale(ddpm_scaled_d, cfg)
        ddpm_recon_d = generator(ddpm_mu_d, labels_d)

        result["eurosat"]["psnr"].extend(batch_psnr(ddpm_recon_e, image_e).tolist())
        result["eurosat"]["ssim"].extend(batch_ssim(ddpm_recon_e, image_e).tolist())
        result["eurosat"]["latent_mse"].extend(
            (ddpm_scaled_e - z_e).pow(2).mean(dim=1).cpu().tolist()
        )
        result["eurosat"]["latent_cosine"].extend(
            F.cosine_similarity(ddpm_scaled_e, z_e, dim=1).cpu().tolist()
        )

        result["deepglobe"]["psnr"].extend(batch_psnr(ddpm_recon_d, image_d).tolist())
        result["deepglobe"]["ssim"].extend(batch_ssim(ddpm_recon_d, image_d).tolist())
        result["deepglobe"]["latent_mse"].extend(
            (ddpm_scaled_d - z_d).pow(2).mean(dim=1).cpu().tolist()
        )
        result["deepglobe"]["latent_cosine"].extend(
            F.cosine_similarity(ddpm_scaled_d, z_d, dim=1).cpu().tolist()
        )

    rows = []
    for domain in ("eurosat", "deepglobe"):
        data = result[domain]
        rows.append(
            {
                "target": domain,
                "num_interferers": int(num_interferers),
                "alpha": float(alpha),
                "fixed_snr_db": float(FIXED_SNR_DB),
                "num_samples": len(data["psnr"]),
                "psnr_ddpm": float(np.mean(data["psnr"])),
                "ssim_ddpm": float(np.mean(data["ssim"])),
                "latent_mse_ddpm": float(np.mean(data["latent_mse"])),
                "latent_cosine_ddpm": float(np.mean(data["latent_cosine"])),
            }
        )
    return rows


def plot_multi_interferer_metric(
    rows: List[dict],
    domain: str,
    metric: str,
    ylabel: str,
    output_dir: Path,
) -> None:
    """Plot one paper-style figure with K=1..5 as separate curves."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for index, k in enumerate(MULTI_INTERFERER_COUNTS):
        selected = sorted(
            [
                row
                for row in rows
                if row["target"] == domain and int(row["num_interferers"]) == k
            ],
            key=lambda x: x["alpha"],
        )
        if not selected:
            continue

        ax.plot(
            [row["alpha"] for row in selected],
            [row[metric] for row in selected],
            color=K_COLORS[index % len(K_COLORS)],
            marker=K_MARKERS[index % len(K_MARKERS)],
            linestyle=K_LINESTYLES[index % len(K_LINESTYLES)],
            linewidth=2.0,
            markersize=5,
            label=f"{k} interferer" if k == 1 else f"{k} interferers",
        )

    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel(ylabel)
    # ax.set_title(f"{domain.capitalize()} — Different Numbers of Interferers")
    style_axis(ax)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    stem = f"{domain}_{metric}_vs_alpha_multi_interferer"
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_multi_interferer_test() -> None:
    cfg = M.Config()
    device = M.resolve_device(cfg.device)
    M.set_seed(cfg.seed)

    output_dir = OUTPUT_ROOT / f"multi_interferer_sweep_snr{FIXED_SNR_DB:g}"
    ensure_dir(output_dir)

    print("=" * 78)
    print("MULTI-INTERFERER ROBUSTNESS TEST")
    print("=" * 78)
    print(f"Device                : {device}")
    print(f"Fixed SNR             : {FIXED_SNR_DB} dB")
    print(f"Interferer counts     : {MULTI_INTERFERER_COUNTS}")
    print(f"Alpha values          : {MULTI_INTERFERER_ALPHAS}")
    print(f"Evaluation pairs      : {MULTI_EVAL_COUNT}")
    print("Channel model          : y = z_target + alpha * sum(z_interferer,k) + n")

    encoder = M.load_encoder(cfg, device)
    generator, _ = M.load_gan_models(cfg, device)
    ddpm, schedule = M.load_ddpm(cfg, device)

    encoder.eval()
    generator.eval()
    ddpm.eval()
    M.set_requires_grad(encoder, False)
    M.set_requires_grad(generator, False)
    M.set_requires_grad(ddpm, False)

    loader_e, loader_d, count = build_fixed_eval_loaders(cfg, MULTI_EVAL_COUNT)
    print(f"Actual evaluation pairs: {count}")

    all_rows: List[dict] = []
    for k in MULTI_INTERFERER_COUNTS:
        for alpha in MULTI_INTERFERER_ALPHAS:
            rows = evaluate_multi_interferer_setting(
                cfg=cfg,
                device=device,
                encoder=encoder,
                generator=generator,
                ddpm=ddpm,
                schedule=schedule,
                loader_e=loader_e,
                loader_d=loader_d,
                alpha=alpha,
                num_interferers=k,
            )
            all_rows.extend(rows)

            print(f"\nK={k}, alpha={alpha:.2f}")
            for row in rows:
                print(
                    f"  {row['target']:10s} | "
                    f"PSNR={row['psnr_ddpm']:.3f} dB | "
                    f"SSIM={row['ssim_ddpm']:.4f} | "
                    f"latentMSE={row['latent_mse_ddpm']:.6f} | "
                    f"cos={row['latent_cosine_ddpm']:.4f}"
                )

    metrics_csv = output_dir / "multi_interferer_random_alpha_ddpm_metrics.csv"
    save_rows(metrics_csv, all_rows)

    # Primary figure requested: PSNR vs alpha, one curve for each K.
    plot_multi_interferer_metric(
        all_rows,
        "eurosat",
        "psnr_ddpm",
        "PSNR (dB)",
        output_dir,
    )
    plot_multi_interferer_metric(
        all_rows,
        "deepglobe",
        "psnr_ddpm",
        "PSNR (dB)",
        output_dir,
    )

    # Also save SSIM versions for thesis/appendix use.
    plot_multi_interferer_metric(
        all_rows,
        "eurosat",
        "ssim_ddpm",
        "SSIM",
        output_dir,
    )
    plot_multi_interferer_metric(
        all_rows,
        "deepglobe",
        "ssim_ddpm",
        "SSIM",
        output_dir,
    )

    print("\n" + "=" * 78)
    print("MULTI-INTERFERER TEST COMPLETED")
    print(f"Metrics CSV : {metrics_csv.resolve()}")
    print(f"Plots       : {output_dir.resolve()}")


@torch.no_grad()
def run_model_test() -> None:
    cfg = M.Config()
    device = M.resolve_device(cfg.device)
    M.set_seed(cfg.seed)

    output_dir = OUTPUT_ROOT / f"alpha_sweep_snr{FIXED_SNR_DB:g}"
    qualitative_dir = output_dir / "qualitative"
    ensure_dir(output_dir)

    print("=" * 78)
    print("RANDOM-ALPHA DDPM MODEL TEST")
    print("=" * 78)
    print(f"Device                : {device}")
    print(f"Fixed SNR             : {FIXED_SNR_DB} dB")
    print(f"Alpha values          : {ALPHAS}")
    print(f"Encoder checkpoint    : {cfg.encoder_best_checkpoint}")
    print(f"Generator checkpoint  : {cfg.gan_best_checkpoint}")
    print(f"DDPM checkpoint       : {cfg.ddpm_best_checkpoint}")

    # Make checkpoint failures obvious before evaluation starts.
    for label, path in [
        ("Encoder", Path(cfg.encoder_best_checkpoint)),
        ("Generator", Path(cfg.gan_best_checkpoint)),
        ("DDPM", Path(cfg.ddpm_best_checkpoint)),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} checkpoint not found: {path.resolve()}")

    encoder = M.load_encoder(cfg, device)
    generator, _ = M.load_gan_models(cfg, device)
    ddpm, schedule = M.load_ddpm(cfg, device)

    encoder.eval()
    generator.eval()
    ddpm.eval()
    M.set_requires_grad(encoder, False)
    M.set_requires_grad(generator, False)
    M.set_requires_grad(ddpm, False)

    loader_e, loader_d, count = build_fixed_eval_loaders(cfg)
    print(f"Evaluation pairs       : {count}")

    all_rows: List[dict] = []
    for alpha in ALPHAS:
        rows = evaluate_one_alpha(
            cfg=cfg,
            device=device,
            encoder=encoder,
            generator=generator,
            ddpm=ddpm,
            schedule=schedule,
            loader_e=loader_e,
            loader_d=loader_d,
            alpha=alpha,
            qualitative_dir=qualitative_dir,
        )
        all_rows.extend(rows)

        print(f"\nalpha = {alpha:.2f}")
        for row in rows:
            print(
                f"  {row['target']:10s} | "
                f"Clean={row['psnr_clean']:.3f} dB | "
                f"NoSep={row['psnr_no_separation']:.3f} dB | "
                f"DDPM={row['psnr_ddpm']:.3f} dB | "
                f"SSIM={row['ssim_ddpm']:.4f} | "
                f"latentMSE={row['latent_mse_ddpm']:.6f} | "
                f"cos={row['latent_cosine_ddpm']:.4f}"
            )

    metrics_csv = output_dir / "random_alpha_ddpm_alpha_sweep_metrics.csv"
    save_rows(metrics_csv, all_rows)
    plot_domain_metrics(all_rows, "eurosat", output_dir)
    plot_domain_metrics(all_rows, "deepglobe", output_dir)

    print("\n" + "=" * 78)
    print("MODEL TEST COMPLETED")
    print(f"Metrics CSV : {metrics_csv.resolve()}")
    print(f"Plots       : {output_dir.resolve()}")
    print(f"Images      : {qualitative_dir.resolve()}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "plot", "eval", "multi"],
        default="all",
        help=(
            "all = training plots + single-interferer alpha sweep + multi-interferer sweep; "
            "plot = training CSV only; eval = single-interferer model test only; "
            "multi = K=1..5 interferer robustness test only"
        ),
    )
    parser.add_argument(
        "--history-csv",
        type=str,
        default=str(HISTORY_CSV),
        help="Path to ddpm_label_cond_siso_random_alpha_training_history.csv",
    )
    args = parser.parse_args()

    ensure_dir(OUTPUT_ROOT)

    if args.mode in {"all", "plot"}:
        plot_training_history(
            Path(args.history_csv),
            OUTPUT_ROOT / "training_history_plots",
        )

    if args.mode in {"all", "eval"}:
        run_model_test()

    if args.mode in {"all", "multi"}:
        run_multi_interferer_test()


if __name__ == "__main__":
    main()
