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
    ax.plot(epoch, df["total_loss"], marker="o", markersize=3, label="Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DDPM Training Total Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ddpm_total_loss.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ddpm_total_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Epsilon-prediction loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch, df["eps_eurosat"], marker="o", markersize=3, label="EuroSAT")
    ax.plot(epoch, df["eps_deepglobe"], marker="s", markersize=3, label="DeepGlobe")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("DDPM Noise-Prediction Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ddpm_epsilon_loss.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ddpm_epsilon_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. x0 latent reconstruction loss
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch, df["x0_eurosat"], marker="o", markersize=3, label="EuroSAT")
    ax.plot(epoch, df["x0_deepglobe"], marker="s", markersize=3, label="DeepGlobe")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Latent MSE")
    ax.set_title("DDPM Clean-Latent Reconstruction Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
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
            marker="o",
            markersize=3,
            label="EuroSAT target",
        )
        ax.plot(
            epoch,
            df["alpha_deepglobe_mean"],
            marker="s",
            markersize=3,
            label="DeepGlobe target",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"Mean interference amplitude $\alpha$")
        ax.set_title("Random Interference Strength During DDPM Training")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "ddpm_alpha_mean.png", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "ddpm_alpha_mean.pdf", bbox_inches="tight")
        plt.close(fig)

    # 5. Log-scale loss is useful if late-stage differences are small.
    fig, ax = plt.subplots(figsize=(8, 5))
    safe_loss = np.maximum(df["total_loss"].to_numpy(dtype=float), 1e-12)
    ax.semilogy(epoch, safe_loss, marker="o", markersize=3, label="Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("DDPM Training Total Loss — Log Scale")
    ax.grid(True, alpha=0.3)
    ax.legend()
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
def build_fixed_eval_loaders(cfg):
    eurosat, deepglobe = M.build_domain_datasets(cfg)
    count = min(EVAL_COUNT, len(eurosat), len(deepglobe))

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
    ax.plot(x, [r["psnr_clean"] for r in selected], linestyle="--", marker="s", label="Clean Latent")
    ax.plot(x, [r["psnr_no_separation"] for r in selected], linestyle=":", marker="D", label="No Separation")
    ax.plot(x, [r["psnr_ddpm"] for r in selected], linestyle="-", marker="o", label="Random-alpha DDPM")
    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{domain.capitalize()} PSNR vs Interference Strength")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{domain}_PSNR_vs_alpha.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{domain}_PSNR_vs_alpha.pdf", bbox_inches="tight")
    plt.close(fig)

    # SSIM
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["ssim_clean"] for r in selected], linestyle="--", marker="s", label="Clean Latent")
    ax.plot(x, [r["ssim_no_separation"] for r in selected], linestyle=":", marker="D", label="No Separation")
    ax.plot(x, [r["ssim_ddpm"] for r in selected], linestyle="-", marker="o", label="Random-alpha DDPM")
    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("SSIM")
    ax.set_title(f"{domain.capitalize()} SSIM vs Interference Strength")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{domain}_SSIM_vs_alpha.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{domain}_SSIM_vs_alpha.pdf", bbox_inches="tight")
    plt.close(fig)

    # Latent MSE
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["latent_mse_ddpm"] for r in selected], marker="o")
    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("Latent MSE")
    ax.set_title(f"{domain.capitalize()} Latent MSE vs Interference Strength")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{domain}_latent_MSE_vs_alpha.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{domain}_latent_MSE_vs_alpha.pdf", bbox_inches="tight")
    plt.close(fig)

    # Latent cosine similarity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["latent_cosine_ddpm"] for r in selected], marker="o")
    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("Latent Cosine Similarity")
    ax.set_title(f"{domain.capitalize()} Latent Cosine Similarity vs Interference Strength")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{domain}_latent_cosine_vs_alpha.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{domain}_latent_cosine_vs_alpha.pdf", bbox_inches="tight")
    plt.close(fig)


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
        choices=["all", "plot", "eval"],
        default="all",
        help="all = plot training CSV + evaluate model; plot = CSV only; eval = model test only",
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


if __name__ == "__main__":
    main()
