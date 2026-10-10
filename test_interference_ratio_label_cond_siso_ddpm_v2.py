
"""
Interference-ratio sweep for the trained VAE-GAN + label-conditioned SISO DDPM.

System:
    z_target = c * mu_target
    z_interf = c * mu_interference

    y = z_target + alpha * z_interf + n

For each alpha, the SAME label-conditioned DDPM is asked to recover the target latent.

Outputs:
    - CSV metrics for EuroSAT-target and DeepGlobe-target
    - PSNR vs interference-ratio plots
    - SSIM vs interference-ratio plots
    - latent MSE / cosine vs interference-ratio plots
    - qualitative grids for selected alpha values

Important:
    This script DOES NOT retrain anything.
    It loads:
        deep_wgangp_vae_encoder_best.pth
        deep_wgangp_satellite_best.pth
        deep_label_cond_siso_ddpm_best.pth

Put this file in the same folder as:
    train_gan_pretrain_then_joint_mu_vaegan_label_cond_siso_ddpm_eval_fixed.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_gan_pretrain_then_joint_mu_vaegan_label_cond_siso_ddpm_v2 as M


# ============================================================
# Experiment settings
# ============================================================

SAVE_ROOT = Path(
    "./save_data/Deep_WGANGP_VAE_DDPM/interference_ratio_sweep"
)

# Fixed channel noise SNR for this experiment.
FIXED_SNR_DB = 10.0

# alpha controls interference strength:
#   y = z_target + alpha * z_interference + n
#
# 0.0 -> no interference
# 1.0 -> equal target/interference amplitude
# 2.0 -> interference amplitude twice the target
ALPHAS = [
    round(x, 2)
    for x in np.arange(0.0, 2.01, 0.1)
]

EVAL_COUNT = 128
BATCH_SIZE = 32

# Qualitative figures only for these alpha values.
QUALITATIVE_ALPHAS = {
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
}

QUALITATIVE_COUNT = 8


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def to_01(x: torch.Tensor) -> torch.Tensor:
    return (
        (x.detach().float() + 1.0) / 2.0
    ).clamp(0.0, 1.0)


def batch_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    pred_01 = to_01(pred)
    target_01 = to_01(target)

    mse = (
        (pred_01 - target_01)
        .pow(2)
        .flatten(start_dim=1)
        .mean(dim=1)
        .clamp_min(1e-12)
    )

    return (
        10.0
        * torch.log10(
            1.0 / mse
        )
    ).cpu().numpy()


def batch_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    pred_np = (
        to_01(pred)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )

    target_np = (
        to_01(target)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )

    values = []

    for p, t in zip(
        pred_np,
        target_np,
    ):
        values.append(
            structural_similarity(
                t,
                p,
                data_range=1.0,
                channel_axis=-1,
            )
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def save_rows(
    path: Path,
    rows: List[dict],
) -> None:
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_comparison_grid(
    path: Path,
    real: torch.Tensor,
    no_sep: torch.Tensor,
    ddpm: torch.Tensor,
    count: int,
) -> None:
    count = min(
        count,
        real.size(0),
    )

    grid_tensor = torch.cat(
        [
            real[:count],
            no_sep[:count],
            ddpm[:count],
        ],
        dim=0,
    )

    grid = make_grid(
        grid_tensor,
        nrow=count,
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )

    save_image(
        grid,
        str(path),
    )

    path.with_suffix(".txt").write_text(
        "\n".join(
            [
                "Row 1: Real target",
                "Row 2: No separation",
                "Row 3: Label-conditioned DDPM",
            ]
        ),
        encoding="utf-8",
    )


def add_awgn(
    clean_signal: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """
    AWGN using the power of the complete received clean mixture.

        y_clean = z_target + alpha*z_interference

    SNR is defined relative to y_clean.
    """
    if math.isinf(
        float(snr_db)
    ):
        return clean_signal

    signal_power = (
        clean_signal.pow(2)
        .mean(
            dim=1,
            keepdim=True,
        )
        .clamp_min(1e-12)
    )

    noise_power = (
        signal_power
        / (
            10.0
            ** (
                float(snr_db)
                / 10.0
            )
        )
    )

    noise = (
        torch.randn_like(
            clean_signal
        )
        * torch.sqrt(
            noise_power
            + 1e-12
        )
    )

    return clean_signal + noise


# ============================================================
# Fixed evaluation dataset
# ============================================================

def build_fixed_eval_loaders(
    cfg: M.Config,
):
    eurosat, deepglobe = (
        M.build_domain_datasets(
            cfg
        )
    )

    count = min(
        EVAL_COUNT,
        len(eurosat),
        len(deepglobe),
    )

    subset_e = Subset(
        eurosat,
        list(range(count)),
    )

    subset_d = Subset(
        deepglobe,
        list(range(count)),
    )

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

    return (
        loader_e,
        loader_d,
        count,
    )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_one_alpha(
    cfg: M.Config,
    device: torch.device,
    encoder,
    generator,
    ddpm,
    schedule,
    loader_e,
    loader_d,
    alpha: float,
) -> List[dict]:

    result = {
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

    for batch_index, (
        batch_e,
        batch_d,
    ) in enumerate(
        zip(
            loader_e,
            loader_d,
        )
    ):
        image_e, _ = batch_e
        image_d, _ = batch_d

        image_e = image_e.to(
            device
        )

        image_d = image_d.to(
            device
        )

        batch = min(
            image_e.size(0),
            image_d.size(0),
        )

        image_e = image_e[:batch]
        image_d = image_d[:batch]

        # ----------------------------------------------------
        # Frozen VAE encoder.
        # ----------------------------------------------------
        mu_e, _ = encoder(
            image_e
        )

        mu_d, _ = encoder(
            image_d
        )

        z_e = M.apply_global_latent_scale(
            mu_e,
            cfg,
        )

        z_d = M.apply_global_latent_scale(
            mu_d,
            cfg,
        )

        labels_e = torch.full(
            (batch,),
            M.EUROSAT_LABEL,
            device=device,
            dtype=torch.long,
        )

        labels_d = torch.full(
            (batch,),
            M.DEEPGLOBE_LABEL,
            device=device,
            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Clean reconstruction ceiling.
        # ----------------------------------------------------
        clean_recon_e = generator(
            mu_e,
            labels_e,
        )

        clean_recon_d = generator(
            mu_d,
            labels_d,
        )

        # ====================================================
        # EuroSAT target:
        #
        # y_E = z_E + alpha*z_D + n
        # ====================================================
        clean_mix_e = (
            z_e
            + float(alpha)
            * z_d
        )

        # Deterministic noise realization per alpha/batch.
        seed_e = (
            int(cfg.seed)
            + int(
                round(
                    alpha * 1000
                )
            )
            + 10000
            + batch_index
        )

        torch.manual_seed(
            seed_e
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed_e
            )

        obs_e = add_awgn(
            clean_mix_e,
            FIXED_SNR_DB,
        )

        # No-separation baseline:
        # directly decode received mixture as requested target.
        no_sep_mu_e = (
            M.invert_global_latent_scale(
                obs_e,
                cfg,
            )
        )

        no_sep_recon_e = generator(
            no_sep_mu_e,
            labels_e,
        )

        # Fixed initial Gaussian seed makes alpha comparison cleaner.
        sample_seed_e = (
            int(cfg.seed)
            + 500000
            + batch_index
        )

        torch.manual_seed(
            sample_seed_e
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                sample_seed_e
            )

        init_noise_e = torch.randn(
            batch,
            cfg.z_dim,
            device=device,
        )

        ddpm_scaled_e = (
            M.sample_conditional_ddpm(
                model=ddpm,
                schedule=schedule,
                observation=obs_e,
                labels=labels_e,
                z_dim=cfg.z_dim,
                initial_noise=init_noise_e,
            )
        )

        ddpm_mu_e = (
            M.invert_global_latent_scale(
                ddpm_scaled_e,
                cfg,
            )
        )

        ddpm_recon_e = generator(
            ddpm_mu_e,
            labels_e,
        )

        # ====================================================
        # DeepGlobe target:
        #
        # y_D = z_D + alpha*z_E + n
        # ====================================================
        clean_mix_d = (
            z_d
            + float(alpha)
            * z_e
        )

        seed_d = (
            int(cfg.seed)
            + int(
                round(
                    alpha * 1000
                )
            )
            + 20000
            + batch_index
        )

        torch.manual_seed(
            seed_d
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed_d
            )

        obs_d = add_awgn(
            clean_mix_d,
            FIXED_SNR_DB,
        )

        no_sep_mu_d = (
            M.invert_global_latent_scale(
                obs_d,
                cfg,
            )
        )

        no_sep_recon_d = generator(
            no_sep_mu_d,
            labels_d,
        )

        sample_seed_d = (
            int(cfg.seed)
            + 600000
            + batch_index
        )

        torch.manual_seed(
            sample_seed_d
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                sample_seed_d
            )

        init_noise_d = torch.randn(
            batch,
            cfg.z_dim,
            device=device,
        )

        ddpm_scaled_d = (
            M.sample_conditional_ddpm(
                model=ddpm,
                schedule=schedule,
                observation=obs_d,
                labels=labels_d,
                z_dim=cfg.z_dim,
                initial_noise=init_noise_d,
            )
        )

        ddpm_mu_d = (
            M.invert_global_latent_scale(
                ddpm_scaled_d,
                cfg,
            )
        )

        ddpm_recon_d = generator(
            ddpm_mu_d,
            labels_d,
        )

        # ----------------------------------------------------
        # Metrics — EuroSAT.
        # ----------------------------------------------------
        result["eurosat"][
            "psnr_clean"
        ].extend(
            batch_psnr(
                clean_recon_e,
                image_e,
            ).tolist()
        )

        result["eurosat"][
            "psnr_no_sep"
        ].extend(
            batch_psnr(
                no_sep_recon_e,
                image_e,
            ).tolist()
        )

        result["eurosat"][
            "psnr_ddpm"
        ].extend(
            batch_psnr(
                ddpm_recon_e,
                image_e,
            ).tolist()
        )

        result["eurosat"][
            "ssim_clean"
        ].extend(
            batch_ssim(
                clean_recon_e,
                image_e,
            ).tolist()
        )

        result["eurosat"][
            "ssim_no_sep"
        ].extend(
            batch_ssim(
                no_sep_recon_e,
                image_e,
            ).tolist()
        )

        result["eurosat"][
            "ssim_ddpm"
        ].extend(
            batch_ssim(
                ddpm_recon_e,
                image_e,
            ).tolist()
        )

        result["eurosat"][
            "latent_mse"
        ].extend(
            (
                (ddpm_scaled_e - z_e)
                .pow(2)
                .mean(dim=1)
                .cpu()
                .tolist()
            )
        )

        result["eurosat"][
            "latent_cosine"
        ].extend(
            F.cosine_similarity(
                ddpm_scaled_e,
                z_e,
                dim=1,
            )
            .cpu()
            .tolist()
        )

        # ----------------------------------------------------
        # Metrics — DeepGlobe.
        # ----------------------------------------------------
        result["deepglobe"][
            "psnr_clean"
        ].extend(
            batch_psnr(
                clean_recon_d,
                image_d,
            ).tolist()
        )

        result["deepglobe"][
            "psnr_no_sep"
        ].extend(
            batch_psnr(
                no_sep_recon_d,
                image_d,
            ).tolist()
        )

        result["deepglobe"][
            "psnr_ddpm"
        ].extend(
            batch_psnr(
                ddpm_recon_d,
                image_d,
            ).tolist()
        )

        result["deepglobe"][
            "ssim_clean"
        ].extend(
            batch_ssim(
                clean_recon_d,
                image_d,
            ).tolist()
        )

        result["deepglobe"][
            "ssim_no_sep"
        ].extend(
            batch_ssim(
                no_sep_recon_d,
                image_d,
            ).tolist()
        )

        result["deepglobe"][
            "ssim_ddpm"
        ].extend(
            batch_ssim(
                ddpm_recon_d,
                image_d,
            ).tolist()
        )

        result["deepglobe"][
            "latent_mse"
        ].extend(
            (
                (ddpm_scaled_d - z_d)
                .pow(2)
                .mean(dim=1)
                .cpu()
                .tolist()
            )
        )

        result["deepglobe"][
            "latent_cosine"
        ].extend(
            F.cosine_similarity(
                ddpm_scaled_d,
                z_d,
                dim=1,
            )
            .cpu()
            .tolist()
        )

        # ----------------------------------------------------
        # Qualitative.
        # ----------------------------------------------------
        if (
            alpha in QUALITATIVE_ALPHAS
            and not qualitative_saved
        ):
            save_comparison_grid(
                SAVE_ROOT
                / (
                    f"qualitative_eurosat_target_"
                    f"alpha_{alpha:.2f}.png"
                ),
                image_e,
                no_sep_recon_e,
                ddpm_recon_e,
                QUALITATIVE_COUNT,
            )

            save_comparison_grid(
                SAVE_ROOT
                / (
                    f"qualitative_deepglobe_target_"
                    f"alpha_{alpha:.2f}.png"
                ),
                image_d,
                no_sep_recon_d,
                ddpm_recon_d,
                QUALITATIVE_COUNT,
            )

            qualitative_saved = True

    rows = []

    for domain in (
        "eurosat",
        "deepglobe",
    ):
        data = result[domain]

        row = {
            "target":
                domain,

            "alpha":
                float(alpha),

            "fixed_snr_db":
                float(
                    FIXED_SNR_DB
                ),

            "psnr_clean":
                float(
                    np.mean(
                        data[
                            "psnr_clean"
                        ]
                    )
                ),

            "psnr_no_separation":
                float(
                    np.mean(
                        data[
                            "psnr_no_sep"
                        ]
                    )
                ),

            "psnr_ddpm":
                float(
                    np.mean(
                        data[
                            "psnr_ddpm"
                        ]
                    )
                ),

            "ssim_clean":
                float(
                    np.mean(
                        data[
                            "ssim_clean"
                        ]
                    )
                ),

            "ssim_no_separation":
                float(
                    np.mean(
                        data[
                            "ssim_no_sep"
                        ]
                    )
                ),

            "ssim_ddpm":
                float(
                    np.mean(
                        data[
                            "ssim_ddpm"
                        ]
                    )
                ),

            "latent_mse_ddpm":
                float(
                    np.mean(
                        data[
                            "latent_mse"
                        ]
                    )
                ),

            "latent_cosine_ddpm":
                float(
                    np.mean(
                        data[
                            "latent_cosine"
                        ]
                    )
                ),
        }

        rows.append(
            row
        )

    return rows


# ============================================================
# Plotting
# ============================================================

def plot_domain(
    rows: List[dict],
    domain: str,
) -> None:
    selected = [
        row
        for row in rows
        if row["target"] == domain
    ]

    selected = sorted(
        selected,
        key=lambda x: x["alpha"],
    )

    x = [
        row["alpha"]
        for row in selected
    ]

    # PSNR
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        x,
        [
            row["psnr_clean"]
            for row in selected
        ],
        linestyle="--",
        marker="s",
        label="Clean Latent",
    )

    ax.plot(
        x,
        [
            row[
                "psnr_no_separation"
            ]
            for row in selected
        ],
        linestyle=":",
        marker="D",
        label="No Separation",
    )

    ax.plot(
        x,
        [
            row["psnr_ddpm"]
            for row in selected
        ],
        linestyle="-",
        marker="o",
        label="Proposed DDPM",
    )

    ax.set_xlabel(
        r"Interference amplitude $\alpha$"
    )

    ax.set_ylabel(
        "PSNR (dB)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SAVE_ROOT
        / f"{domain}_PSNR_vs_interference_alpha.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        SAVE_ROOT
        / f"{domain}_PSNR_vs_interference_alpha.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # SSIM
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        x,
        [
            row["ssim_clean"]
            for row in selected
        ],
        linestyle="--",
        marker="s",
        label="Clean Latent",
    )

    ax.plot(
        x,
        [
            row[
                "ssim_no_separation"
            ]
            for row in selected
        ],
        linestyle=":",
        marker="D",
        label="No Separation",
    )

    ax.plot(
        x,
        [
            row["ssim_ddpm"]
            for row in selected
        ],
        linestyle="-",
        marker="o",
        label="Proposed DDPM",
    )

    ax.set_xlabel(
        r"Interference amplitude $\alpha$"
    )

    ax.set_ylabel(
        "SSIM"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SAVE_ROOT
        / f"{domain}_SSIM_vs_interference_alpha.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        SAVE_ROOT
        / f"{domain}_SSIM_vs_interference_alpha.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Latent MSE.
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        x,
        [
            row[
                "latent_mse_ddpm"
            ]
            for row in selected
        ],
        marker="o",
    )

    ax.set_xlabel(
        r"Interference amplitude $\alpha$"
    )

    ax.set_ylabel(
        "Latent MSE"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        SAVE_ROOT
        / f"{domain}_latent_MSE_vs_interference_alpha.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        SAVE_ROOT
        / f"{domain}_latent_MSE_vs_interference_alpha.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Latent cosine.
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        x,
        [
            row[
                "latent_cosine_ddpm"
            ]
            for row in selected
        ],
        marker="o",
    )

    ax.set_xlabel(
        r"Interference amplitude $\alpha$"
    )

    ax.set_ylabel(
        "Latent cosine similarity"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        SAVE_ROOT
        / f"{domain}_latent_cosine_vs_interference_alpha.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        SAVE_ROOT
        / f"{domain}_latent_cosine_vs_interference_alpha.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_dir(
        SAVE_ROOT
    )

    cfg = M.Config()

    # This experiment uses the exact same calibrated scale as DDPM training.
    device = M.resolve_device(
        cfg.device
    )

    M.set_seed(
        cfg.seed
    )

    print(
        "Device:",
        device,
    )

    print(
        f"Fixed SNR: {FIXED_SNR_DB} dB"
    )

    print(
        "Interference model:"
    )

    print(
        "  EuroSAT target   : y = z_E + alpha*z_D + n"
    )

    print(
        "  DeepGlobe target : y = z_D + alpha*z_E + n"
    )

    print(
        "alpha values:",
        ALPHAS,
    )

    # --------------------------------------------------------
    # Load exactly the same frozen semantic backbone used by eval.
    # --------------------------------------------------------
    encoder = M.load_encoder(
        cfg,
        device,
    )

    generator, _ = M.load_gan_models(
        cfg,
        device,
    )

    ddpm, schedule = M.load_ddpm(
        cfg,
        device,
    )

    encoder.eval()
    generator.eval()
    ddpm.eval()

    M.set_requires_grad(
        encoder,
        False,
    )

    M.set_requires_grad(
        generator,
        False,
    )

    M.set_requires_grad(
        ddpm,
        False,
    )

    print()
    print(
        "Loaded:"
    )

    print(
        "  Encoder  :",
        cfg.encoder_best_checkpoint,
    )

    print(
        "  Generator:",
        cfg.gan_best_checkpoint,
    )

    print(
        "  DDPM     :",
        cfg.ddpm_best_checkpoint,
    )

    print(
        "  global c :",
        cfg.global_latent_scale,
    )

    loader_e, loader_d, count = (
        build_fixed_eval_loaders(
            cfg
        )
    )

    print(
        "Evaluation pairs:",
        count,
    )

    all_rows = []

    for alpha in ALPHAS:
        print()
        print(
            "=" * 70
        )

        print(
            f"alpha = {alpha:.2f}"
        )

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
        )

        all_rows.extend(
            rows
        )

        for row in rows:
            print(
                f"{row['target']:10s} | "
                f"Clean={row['psnr_clean']:.3f} dB | "
                f"NoSep={row['psnr_no_separation']:.3f} dB | "
                f"DDPM={row['psnr_ddpm']:.3f} dB | "
                f"latentMSE={row['latent_mse_ddpm']:.5f} | "
                f"cos={row['latent_cosine_ddpm']:.4f}"
            )

    save_rows(
        SAVE_ROOT
        / "interference_ratio_metrics.csv",
        all_rows,
    )

    plot_domain(
        all_rows,
        "eurosat",
    )

    plot_domain(
        all_rows,
        "deepglobe",
    )

    print()
    print(
        "=" * 70
    )

    print(
        "Completed."
    )

    print(
        "Saved to:",
        SAVE_ROOT.resolve(),
    )

    print(
        "Main CSV:",
        (
            SAVE_ROOT
            / "interference_ratio_metrics.csv"
        ).resolve(),
    )


if __name__ == "__main__":
    main()
