"""
Final pre-DDPM diagnostic for the NEW fixed-global-scaling pipeline.

Pipeline to validate
--------------------
    image
      -> Encoder
      -> raw mu
      -> fixed global scale: z_tx = c * mu
      -> 2x2 MIMO + AWGN
      -> DDPM conditioned on the full two-Rx observation
      -> recovered scaled latent z_hat
      -> inverse global scale: mu_hat = z_hat / c
      -> Generator

Why this script exists
----------------------
The original per-sample normalization:

    z = mu / sqrt(mean(mu^2))

was found to cause a very large reconstruction drop because the Generator
was trained on raw mu rather than per-sample unit-power mu.

This script validates the fixed-global-scaling replacement before launching
the long DDPM training.

Checks
------
1. Checkpoint consistency.
2. Estimate one fixed global latent scale c from a larger TRAINING calibration set.
3. Verify scale -> inverse-scale is numerically lossless.
4. Verify 2x2 MIMO + ZF + inverse-scale approaches the raw-mu reconstruction ceiling.
5. Verify diffusion schedule tensor/math consistency using the NEW scaled latent target.
6. Train a short GLOBAL-SCALE DDPM smoke test from scratch.
7. Run short DDIM inference with that smoke model and compare:
       Raw-mu clean reconstruction
       Global-scale ZF
       Global-scale DDPM
   at selected SNRs.

Nothing here overwrites your original checkpoints.
All outputs go to:

    ./save_data/Deep_WGANGP_VAE_DDPM_GLOBAL_PRECHECK

Expected setup
--------------
Put this file in the SAME folder as:

    train_gan_pretrain_then_joint_mu_vaegan_ddpm.py

Then run:

    python pre_ddpm_global_scale_final_check.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid, save_image


# ============================================================
# Import original training script
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import train_gan_pretrain_then_joint_mu_vaegan_ddpm as M
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import train_gan_pretrain_then_joint_mu_vaegan_ddpm.py.\n"
        "Put this file in the SAME folder as the original training script."
    ) from exc


# ============================================================
# User settings
# ============================================================

SAVE_ROOT = Path(
    "./save_data/Deep_WGANGP_VAE_DDPM_GLOBAL_PRECHECK"
)

# Larger calibration set used only to estimate the fixed global scale.
# Equal number of samples are drawn from both domains.
GLOBAL_SCALE_CALIBRATION_PER_DOMAIN = 5000
GLOBAL_SCALE_CALIBRATION_BATCH_SIZE = 64

# Fixed diagnostic pairs used for reconstruction / channel evaluation.
CHECK_COUNT = 256
CHECK_BATCH_SIZE = 32

# Channel sanity SNRs.
CHECK_SNRS = (0.0, 10.0, 20.0, 30.0)

# Global-scale DDPM smoke training.
RUN_GLOBAL_DDPM_SMOKE = True
SMOKE_STEPS = 500
SMOKE_BATCH_SIZE = 32
SMOKE_LOG_EVERY = 20

# Short DDIM evaluation after smoke training.
RUN_SMOKE_DDIM_EVAL = True
SMOKE_DDIM_EVAL_SNRS = (10.0, 20.0)
SMOKE_DDIM_STEPS = 50
SMOKE_DDIM_EVAL_COUNT = 64

# Number of qualitative samples shown in each row.
QUALITATIVE_COUNT = 8


# ============================================================
# Generic utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: List[dict]) -> None:
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
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


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
        * torch.log10(1.0 / mse)
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

    for p, t in zip(pred_np, target_np):
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


def save_comparison_grid(
    path: Path,
    real: torch.Tensor,
    outputs: List[Tuple[str, torch.Tensor]],
    count: int,
) -> None:
    count = min(
        count,
        real.size(0),
    )

    rows = [real[:count]]
    labels = ["Real"]

    for name, tensor in outputs:
        rows.append(
            tensor[:count]
        )
        labels.append(name)

    grid = make_grid(
        torch.cat(
            rows,
            dim=0,
        ),
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
            f"Row {idx + 1}: {label}"
            for idx, label in enumerate(labels)
        ),
        encoding="utf-8",
    )


def state_dict_max_abs_diff(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
) -> float:
    if set(a.keys()) != set(b.keys()):
        return float("inf")

    max_diff = 0.0

    for key in a:
        ta = a[key].detach().float().cpu()
        tb = b[key].detach().float().cpu()

        if ta.shape != tb.shape:
            return float("inf")

        max_diff = max(
            max_diff,
            (ta - tb).abs().max().item(),
        )

    return float(max_diff)


# ============================================================
# Fixed dataset helpers
# ============================================================

def build_fixed_diagnostic_loaders(
    cfg,
):
    eurosat, deepglobe = M.build_domain_datasets(cfg)

    count = min(
        CHECK_COUNT,
        len(eurosat),
        len(deepglobe),
    )

    if count <= 0:
        raise RuntimeError(
            "No diagnostic samples are available."
        )

    # Deterministic fixed subset.
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
        batch_size=min(
            CHECK_BATCH_SIZE,
            count,
        ),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    loader_d = DataLoader(
        subset_d,
        batch_size=min(
            CHECK_BATCH_SIZE,
            count,
        ),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return (
        loader_e,
        loader_d,
        count,
    )


def build_global_scale_calibration_loaders(
    cfg,
):
    """
    Use TRAINING datasets only.

    A deterministic random subset is chosen independently from each domain.
    The same number of samples is used from EuroSAT and DeepGlobe to avoid
    one domain dominating the shared global power estimate.
    """
    eurosat, deepglobe = M.build_domain_datasets(cfg)

    per_domain = min(
        GLOBAL_SCALE_CALIBRATION_PER_DOMAIN,
        len(eurosat),
        len(deepglobe),
    )

    if per_domain <= 0:
        raise RuntimeError(
            "No samples available for global-scale calibration."
        )

    generator = torch.Generator().manual_seed(
        int(cfg.seed) + 12345
    )

    idx_e = torch.randperm(
        len(eurosat),
        generator=generator,
    )[:per_domain].tolist()

    idx_d = torch.randperm(
        len(deepglobe),
        generator=generator,
    )[:per_domain].tolist()

    loader_e = DataLoader(
        Subset(
            eurosat,
            idx_e,
        ),
        batch_size=GLOBAL_SCALE_CALIBRATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    loader_d = DataLoader(
        Subset(
            deepglobe,
            idx_d,
        ),
        batch_size=GLOBAL_SCALE_CALIBRATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return (
        loader_e,
        loader_d,
        per_domain,
    )


# ============================================================
# Check 1 — checkpoint consistency
# ============================================================

def check_checkpoint_consistency(
    cfg,
) -> dict:
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 1 — CHECKPOINT CONSISTENCY\n"
        + "=" * 90
    )

    required = {
        "joint_best":
            cfg.joint_best_checkpoint,

        "encoder_best":
            cfg.encoder_best_checkpoint,

        "gan_best":
            cfg.gan_best_checkpoint,
    }

    inventory = []

    for name, path in required.items():
        exists = os.path.exists(path)

        epoch = None

        if exists:
            ckpt = torch.load(
                path,
                map_location="cpu",
            )

            if isinstance(
                ckpt,
                dict,
            ):
                epoch = ckpt.get(
                    "epoch",
                    None,
                )

        inventory.append(
            {
                "checkpoint": name,
                "path": path,
                "exists": exists,
                "epoch": epoch,
            }
        )

        print(
            f"{name:15s} | "
            f"exists={exists} | "
            f"epoch={epoch} | "
            f"{path}"
        )

    write_rows(
        SAVE_ROOT
        / "01_checkpoint_inventory.csv",
        inventory,
    )

    for item in inventory:
        if not item["exists"]:
            raise FileNotFoundError(
                f"Missing required checkpoint: "
                f"{item['checkpoint']} -> {item['path']}"
            )

    joint = torch.load(
        cfg.joint_best_checkpoint,
        map_location="cpu",
    )

    encoder_ckpt = torch.load(
        cfg.encoder_best_checkpoint,
        map_location="cpu",
    )

    gan_ckpt = torch.load(
        cfg.gan_best_checkpoint,
        map_location="cpu",
    )

    encoder_state = (
        encoder_ckpt["encoder"]
        if "encoder" in encoder_ckpt
        else encoder_ckpt
    )

    generator_state = (
        gan_ckpt["generator"]
        if "generator" in gan_ckpt
        else gan_ckpt
    )

    encoder_diff = state_dict_max_abs_diff(
        joint["encoder"],
        encoder_state,
    )

    generator_diff = state_dict_max_abs_diff(
        joint["generator"],
        generator_state,
    )

    result = {
        "joint_best_epoch":
            joint.get("epoch"),

        "encoder_best_epoch":
            encoder_ckpt.get("epoch"),

        "gan_best_epoch":
            gan_ckpt.get("epoch"),

        "encoder_vs_joint_max_abs_diff":
            encoder_diff,

        "generator_vs_joint_max_abs_diff":
            generator_diff,
    }

    print()
    print(
        "Encoder max |diff|   :",
        encoder_diff,
    )
    print(
        "Generator max |diff| :",
        generator_diff,
    )

    if encoder_diff == 0.0 and generator_diff == 0.0:
        print(
            "[PASS] Encoder and Generator exactly match joint-best checkpoint."
        )
    else:
        print(
            "[WARNING] Separate best checkpoints do not exactly match joint-best."
        )

    write_rows(
        SAVE_ROOT
        / "01_checkpoint_consistency.csv",
        [result],
    )

    return result


# ============================================================
# Check 2 — estimate ONE fixed global scale from training data
# ============================================================

@torch.no_grad()
def estimate_and_save_global_scale(
    cfg,
    device: torch.device,
    encoder,
):
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 2 — ESTIMATE FIXED GLOBAL LATENT SCALE\n"
        + "=" * 90
    )

    (
        loader_e,
        loader_d,
        per_domain,
    ) = build_global_scale_calibration_loaders(
        cfg
    )

    total_sq_e = 0.0
    total_numel_e = 0

    total_sq_d = 0.0
    total_numel_d = 0

    for batch, _ in loader_e:
        batch = batch.to(
            device,
            non_blocking=True,
        )

        mu, _ = encoder(batch)

        total_sq_e += float(
            mu.pow(2).sum().item()
        )

        total_numel_e += mu.numel()

    for batch, _ in loader_d:
        batch = batch.to(
            device,
            non_blocking=True,
        )

        mu, _ = encoder(batch)

        total_sq_d += float(
            mu.pow(2).sum().item()
        )

        total_numel_d += mu.numel()

    power_e = (
        total_sq_e
        / total_numel_e
    )

    power_d = (
        total_sq_d
        / total_numel_d
    )

    # Balanced shared power estimate.
    # Because the same number of images and same z_dim are used from
    # each domain, this is equivalent to averaging all latent elements.
    global_power = (
        total_sq_e + total_sq_d
    ) / (
        total_numel_e + total_numel_d
    )

    global_scale = (
        1.0
        / math.sqrt(
            global_power + 1e-12
        )
    )

    result = {
        "calibration_samples_per_domain":
            per_domain,

        "eurosat_latent_power":
            float(power_e),

        "deepglobe_latent_power":
            float(power_d),

        "global_latent_power":
            float(global_power),

        "global_scale":
            float(global_scale),

        "inverse_global_scale":
            float(1.0 / global_scale),
    }

    print(
        f"Calibration samples/domain : {per_domain}"
    )

    print(
        f"EuroSAT latent power       : {power_e:.8f}"
    )

    print(
        f"DeepGlobe latent power     : {power_d:.8f}"
    )

    print(
        f"Shared global power        : {global_power:.8f}"
    )

    print(
        f"FIXED global scale c       : {global_scale:.8f}"
    )

    print(
        f"Inverse scale 1/c          : {1.0/global_scale:.8f}"
    )

    write_rows(
        SAVE_ROOT
        / "02_global_scale_calibration.csv",
        [result],
    )

    with open(
        SAVE_ROOT
        / "global_latent_scale.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    return result


# ============================================================
# Fixed-global-scale channel
# ============================================================

def mimo_2x2_global_scale_observation(
    z_e_scaled: torch.Tensor,
    z_d_scaled: torch.Tensor,
    snr_db: torch.Tensor | float,
    cfg,
):
    """
    IMPORTANT:
    This channel does NOT perform any internal per-sample normalization.

    Inputs must already be globally scaled:
        z_scaled = c * mu
    """

    if z_e_scaled.shape != z_d_scaled.shape:
        raise ValueError(
            f"Latent shapes do not match: "
            f"{z_e_scaled.shape} vs {z_d_scaled.shape}"
        )

    streams = torch.stack(
        [
            z_e_scaled,
            z_d_scaled,
        ],
        dim=1,
    )

    h = M.get_channel_matrix(
        cfg,
        streams.device,
        streams.dtype,
    )

    clean_y = torch.einsum(
        "ij,bjz->biz",
        h,
        streams,
    )

    if not torch.is_tensor(
        snr_db
    ):
        snr_db = torch.tensor(
            float(snr_db),
            device=streams.device,
            dtype=streams.dtype,
        )

    snr_db = snr_db.to(
        device=streams.device,
        dtype=streams.dtype,
    )

    if snr_db.ndim == 0:
        snr_db = snr_db.expand(
            streams.size(0)
        )

    signal_power = (
        clean_y.pow(2)
        .mean(
            dim=(1, 2),
            keepdim=True,
        )
    )

    noise_power = (
        signal_power
        / torch.pow(
            10.0,
            snr_db.view(
                -1,
                1,
                1,
            )
            / 10.0,
        )
    )

    noisy_y = (
        clean_y
        + torch.randn_like(
            clean_y
        )
        * torch.sqrt(
            noise_power
            + 1e-12
        )
    )

    h_inv = torch.linalg.inv(h)

    zf = torch.einsum(
        "ij,bjz->biz",
        h_inv,
        noisy_y,
    )

    observation = noisy_y.flatten(
        start_dim=1
    )

    return (
        observation,
        zf[:, 0],
        zf[:, 1],
    )


# ============================================================
# Check 3 — global scale roundtrip
# ============================================================

@torch.no_grad()
def check_global_scale_roundtrip(
    cfg,
    device: torch.device,
    loader_e,
    loader_d,
    encoder,
    generator,
    global_scale: float,
):
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 3 — GLOBAL SCALE -> INVERSE SCALE ROUNDTRIP\n"
        + "=" * 90
    )

    result_lists: Dict[
        str,
        List[float],
    ] = {}

    latent_mse_e = []
    latent_mse_d = []

    qualitative_done = False

    c = float(
        global_scale
    )

    for batch_e, batch_d in zip(
        loader_e,
        loader_d,
    ):
        image_e, _ = batch_e
        image_d, _ = batch_d

        image_e = image_e.to(device)
        image_d = image_d.to(device)

        mu_e, _ = encoder(image_e)
        mu_d, _ = encoder(image_d)

        labels_e = torch.full(
            (mu_e.size(0),),
            M.EUROSAT_LABEL,
            device=device,
            dtype=torch.long,
        )

        labels_d = torch.full(
            (mu_d.size(0),),
            M.DEEPGLOBE_LABEL,
            device=device,
            dtype=torch.long,
        )

        raw_e = generator(
            mu_e,
            labels_e,
        )

        raw_d = generator(
            mu_d,
            labels_d,
        )

        z_e_scaled = mu_e * c
        z_d_scaled = mu_d * c

        mu_e_restored = (
            z_e_scaled / c
        )

        mu_d_restored = (
            z_d_scaled / c
        )

        restored_e = generator(
            mu_e_restored,
            labels_e,
        )

        restored_d = generator(
            mu_d_restored,
            labels_d,
        )

        latent_mse_e.append(
            F.mse_loss(
                mu_e_restored,
                mu_e,
            ).item()
        )

        latent_mse_d.append(
            F.mse_loss(
                mu_d_restored,
                mu_d,
            ).item()
        )

        metrics = {
            "eurosat_raw_psnr":
                batch_psnr(
                    raw_e,
                    image_e,
                ),

            "eurosat_roundtrip_psnr":
                batch_psnr(
                    restored_e,
                    image_e,
                ),

            "eurosat_raw_ssim":
                batch_ssim(
                    raw_e,
                    image_e,
                ),

            "eurosat_roundtrip_ssim":
                batch_ssim(
                    restored_e,
                    image_e,
                ),

            "deepglobe_raw_psnr":
                batch_psnr(
                    raw_d,
                    image_d,
                ),

            "deepglobe_roundtrip_psnr":
                batch_psnr(
                    restored_d,
                    image_d,
                ),

            "deepglobe_raw_ssim":
                batch_ssim(
                    raw_d,
                    image_d,
                ),

            "deepglobe_roundtrip_ssim":
                batch_ssim(
                    restored_d,
                    image_d,
                ),
        }

        for key, values in metrics.items():
            result_lists.setdefault(
                key,
                [],
            ).extend(
                np.asarray(
                    values
                )
                .reshape(-1)
                .tolist()
            )

        if not qualitative_done:
            save_comparison_grid(
                SAVE_ROOT
                / "03_eurosat_global_roundtrip.png",
                image_e,
                [
                    (
                        "Raw mu",
                        raw_e,
                    ),
                    (
                        "c*mu -> /c",
                        restored_e,
                    ),
                ],
                QUALITATIVE_COUNT,
            )

            save_comparison_grid(
                SAVE_ROOT
                / "03_deepglobe_global_roundtrip.png",
                image_d,
                [
                    (
                        "Raw mu",
                        raw_d,
                    ),
                    (
                        "c*mu -> /c",
                        restored_d,
                    ),
                ],
                QUALITATIVE_COUNT,
            )

            qualitative_done = True

    result = {
        "global_scale":
            c,

        "eurosat_latent_roundtrip_mse":
            float(
                np.mean(
                    latent_mse_e
                )
            ),

        "deepglobe_latent_roundtrip_mse":
            float(
                np.mean(
                    latent_mse_d
                )
            ),
    }

    for key, values in result_lists.items():
        result[
            key + "_mean"
        ] = float(
            np.mean(values)
        )

        result[
            key + "_std"
        ] = float(
            np.std(values)
        )

    result[
        "eurosat_psnr_drop"
    ] = (
        result["eurosat_raw_psnr_mean"]
        - result["eurosat_roundtrip_psnr_mean"]
    )

    result[
        "deepglobe_psnr_drop"
    ] = (
        result["deepglobe_raw_psnr_mean"]
        - result["deepglobe_roundtrip_psnr_mean"]
    )

    print(
        f"EuroSAT raw={result['eurosat_raw_psnr_mean']:.3f} dB | "
        f"roundtrip={result['eurosat_roundtrip_psnr_mean']:.3f} dB | "
        f"drop={result['eurosat_psnr_drop']:.6f} dB"
    )

    print(
        f"DeepGlobe raw={result['deepglobe_raw_psnr_mean']:.3f} dB | "
        f"roundtrip={result['deepglobe_roundtrip_psnr_mean']:.3f} dB | "
        f"drop={result['deepglobe_psnr_drop']:.6f} dB"
    )

    write_rows(
        SAVE_ROOT
        / "03_global_roundtrip_metrics.csv",
        [result],
    )

    return result


# ============================================================
# Check 4 — global scale MIMO + ZF + inverse scale
# ============================================================

@torch.no_grad()
def check_global_scale_mimo_zf(
    cfg,
    device: torch.device,
    loader_e,
    loader_d,
    encoder,
    generator,
    global_scale: float,
):
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 4 — GLOBAL SCALE -> MIMO -> ZF -> INVERSE SCALE\n"
        + "=" * 90
    )

    rows = []
    qualitative_saved = set()

    c = float(
        global_scale
    )

    for snr_db in CHECK_SNRS:
        raw_psnr_e = []
        raw_psnr_d = []

        zf_psnr_e = []
        zf_psnr_d = []

        zf_ssim_e = []
        zf_ssim_d = []

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

            image_e = image_e.to(device)
            image_d = image_d.to(device)

            mu_e, _ = encoder(image_e)
            mu_d, _ = encoder(image_d)

            labels_e = torch.full(
                (mu_e.size(0),),
                M.EUROSAT_LABEL,
                device=device,
                dtype=torch.long,
            )

            labels_d = torch.full(
                (mu_d.size(0),),
                M.DEEPGLOBE_LABEL,
                device=device,
                dtype=torch.long,
            )

            raw_e = generator(
                mu_e,
                labels_e,
            )

            raw_d = generator(
                mu_d,
                labels_d,
            )

            z_e_scaled = mu_e * c
            z_d_scaled = mu_d * c

            seed_here = (
                int(cfg.seed)
                + int(
                    float(snr_db)
                    * 1000
                )
                + batch_index
                + 100000
            )

            torch.manual_seed(
                seed_here
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    seed_here
                )

            (
                observation,
                zf_scaled_e,
                zf_scaled_d,
            ) = (
                mimo_2x2_global_scale_observation(
                    z_e_scaled,
                    z_d_scaled,
                    snr_db,
                    cfg,
                )
            )

            zf_mu_e = (
                zf_scaled_e / c
            )

            zf_mu_d = (
                zf_scaled_d / c
            )

            zf_recon_e = generator(
                zf_mu_e,
                labels_e,
            )

            zf_recon_d = generator(
                zf_mu_d,
                labels_d,
            )

            raw_psnr_e.extend(
                batch_psnr(
                    raw_e,
                    image_e,
                ).tolist()
            )

            raw_psnr_d.extend(
                batch_psnr(
                    raw_d,
                    image_d,
                ).tolist()
            )

            zf_psnr_e.extend(
                batch_psnr(
                    zf_recon_e,
                    image_e,
                ).tolist()
            )

            zf_psnr_d.extend(
                batch_psnr(
                    zf_recon_d,
                    image_d,
                ).tolist()
            )

            zf_ssim_e.extend(
                batch_ssim(
                    zf_recon_e,
                    image_e,
                ).tolist()
            )

            zf_ssim_d.extend(
                batch_ssim(
                    zf_recon_d,
                    image_d,
                ).tolist()
            )

            if snr_db not in qualitative_saved:
                save_comparison_grid(
                    SAVE_ROOT
                    / f"04_eurosat_global_zf_snr{snr_db:g}.png",
                    image_e,
                    [
                        (
                            "Raw clean",
                            raw_e,
                        ),
                        (
                            f"Global ZF @ {snr_db:g} dB",
                            zf_recon_e,
                        ),
                    ],
                    QUALITATIVE_COUNT,
                )

                save_comparison_grid(
                    SAVE_ROOT
                    / f"04_deepglobe_global_zf_snr{snr_db:g}.png",
                    image_d,
                    [
                        (
                            "Raw clean",
                            raw_d,
                        ),
                        (
                            f"Global ZF @ {snr_db:g} dB",
                            zf_recon_d,
                        ),
                    ],
                    QUALITATIVE_COUNT,
                )

                qualitative_saved.add(
                    snr_db
                )

        row = {
            "snr_db":
                float(snr_db),

            "eurosat_raw_psnr":
                float(
                    np.mean(
                        raw_psnr_e
                    )
                ),

            "eurosat_zf_psnr":
                float(
                    np.mean(
                        zf_psnr_e
                    )
                ),

            "eurosat_zf_ssim":
                float(
                    np.mean(
                        zf_ssim_e
                    )
                ),

            "deepglobe_raw_psnr":
                float(
                    np.mean(
                        raw_psnr_d
                    )
                ),

            "deepglobe_zf_psnr":
                float(
                    np.mean(
                        zf_psnr_d
                    )
                ),

            "deepglobe_zf_ssim":
                float(
                    np.mean(
                        zf_ssim_d
                    )
                ),
        }

        rows.append(row)

        print(
            f"SNR={snr_db:>4g} dB | "
            f"E raw={row['eurosat_raw_psnr']:.3f}, "
            f"E ZF={row['eurosat_zf_psnr']:.3f} | "
            f"D raw={row['deepglobe_raw_psnr']:.3f}, "
            f"D ZF={row['deepglobe_zf_psnr']:.3f}"
        )

    write_rows(
        SAVE_ROOT
        / "04_global_mimo_zf_metrics.csv",
        rows,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    x = [
        row["snr_db"]
        for row in rows
    ]

    ax.plot(
        x,
        [
            row["eurosat_zf_psnr"]
            for row in rows
        ],
        marker="o",
        label="EuroSAT ZF",
    )

    ax.plot(
        x,
        [
            row["deepglobe_zf_psnr"]
            for row in rows
        ],
        marker="s",
        label="DeepGlobe ZF",
    )

    ax.axhline(
        rows[0][
            "eurosat_raw_psnr"
        ],
        linestyle="--",
        linewidth=1.5,
        label="EuroSAT raw ceiling",
    )

    ax.axhline(
        rows[0][
            "deepglobe_raw_psnr"
        ],
        linestyle=":",
        linewidth=1.5,
        label="DeepGlobe raw ceiling",
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "PSNR (dB)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        SAVE_ROOT
        / "04_global_mimo_zf_psnr_vs_snr.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return rows


# ============================================================
# Check 5 — DDPM tensor/math sanity on GLOBAL-SCALED latent
# ============================================================

@torch.no_grad()
def check_global_ddpm_interface(
    cfg,
    device: torch.device,
    loader_e,
    loader_d,
    encoder,
    global_scale: float,
):
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 5 — GLOBAL-SCALE DDPM INTERFACE / DIFFUSION SANITY\n"
        + "=" * 90
    )

    batch_e, _ = next(
        iter(loader_e)
    )

    batch_d, _ = next(
        iter(loader_d)
    )

    batch_e = batch_e.to(device)
    batch_d = batch_d.to(device)

    batch = min(
        batch_e.size(0),
        batch_d.size(0),
    )

    batch_e = batch_e[:batch]
    batch_d = batch_d[:batch]

    mu_e, _ = encoder(batch_e)
    mu_d, _ = encoder(batch_d)

    c = float(
        global_scale
    )

    z_e = mu_e * c
    z_d = mu_d * c

    observation, _, _ = (
        mimo_2x2_global_scale_observation(
            z_e,
            z_d,
            10.0,
            cfg,
        )
    )

    ddpm = M.ConditionalLatentDDPM(
        cfg
    ).to(device)

    schedule = M.DiffusionSchedule(
        cfg.diffusion_steps
    ).to(device)

    t = torch.randint(
        0,
        min(
            cfg.train_t_max,
            cfg.diffusion_steps,
        ),
        (batch,),
        device=device,
        dtype=torch.long,
    )

    true_noise = torch.randn_like(
        z_e
    )

    z_t = schedule.q_sample(
        z_e,
        t,
        true_noise,
    )

    labels_e = torch.full(
        (batch,),
        M.EUROSAT_LABEL,
        device=device,
        dtype=torch.long,
    )

    pred_eps = ddpm(
        z_t,
        t,
        labels_e,
        observation,
    )

    recovered_x0 = (
        schedule.predict_x0(
            z_t,
            t,
            true_noise,
        )
    )

    perfect_mse = F.mse_loss(
        recovered_x0,
        z_e,
    ).item()

    result = {
        "z_shape":
            str(
                tuple(
                    z_e.shape
                )
            ),

        "observation_shape":
            str(
                tuple(
                    observation.shape
                )
            ),

        "ddpm_output_shape":
            str(
                tuple(
                    pred_eps.shape
                )
            ),

        "perfect_eps_x0_mse":
            perfect_mse,

        "pred_eps_finite":
            bool(
                torch.isfinite(
                    pred_eps
                ).all().item()
            ),
    }

    print(
        "Scaled z shape       :",
        tuple(z_e.shape),
    )

    print(
        "Observation shape    :",
        tuple(observation.shape),
    )

    print(
        "DDPM output shape    :",
        tuple(pred_eps.shape),
    )

    print(
        "Perfect-eps x0 MSE   :",
        f"{perfect_mse:.8e}",
    )

    write_rows(
        SAVE_ROOT
        / "05_global_ddpm_interface.csv",
        [result],
    )

    return result


# ============================================================
# Deterministic DDIM sampler using original DDPM model/schedule
# ============================================================

@torch.no_grad()
def ddim_sample_global(
    model,
    schedule,
    observation: torch.Tensor,
    labels: torch.Tensor,
    z_dim: int,
    sample_steps: int,
    seed: int,
):
    """
    Deterministic DDIM (eta=0), same mathematical form as the
    original training script, but with explicit seed control.
    """

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    device = observation.device
    batch_size = observation.size(0)

    x = torch.randn(
        batch_size,
        z_dim,
        device=device,
    )

    times = torch.linspace(
        schedule.steps - 1,
        0,
        steps=sample_steps,
        device=device,
    ).long()

    for index, time_value in enumerate(
        times
    ):
        t = torch.full(
            (batch_size,),
            int(
                time_value.item()
            ),
            device=device,
            dtype=torch.long,
        )

        eps = model(
            x,
            t,
            labels,
            observation,
        )

        x0 = schedule.predict_x0(
            x,
            t,
            eps,
        )

        if index == len(times) - 1:
            x = x0
            break

        next_t = times[
            index + 1
        ]

        alpha_bar_next = (
            schedule.alpha_bar[
                next_t
            ]
            .view(1, 1)
        )

        x = (
            torch.sqrt(
                alpha_bar_next
            )
            * x0
            + torch.sqrt(
                1.0
                - alpha_bar_next
            )
            * eps
        )

    return x


# ============================================================
# Check 6 — GLOBAL-SCALE DDPM smoke training
# ============================================================

def train_global_ddpm_smoke(
    cfg,
    device: torch.device,
    encoder,
    global_scale: float,
):
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 6 — GLOBAL-SCALE DDPM SMOKE TRAINING\n"
        + "=" * 90
    )

    (
        euro_loader,
        deep_loader,
        _,
        _,
    ) = M.build_paired_loaders(
        cfg,
        SMOKE_BATCH_SIZE,
    )

    euro_iter = M.infinite_loader(
        euro_loader
    )

    deep_iter = M.infinite_loader(
        deep_loader
    )

    ddpm = M.ConditionalLatentDDPM(
        cfg
    ).to(device)

    schedule = M.DiffusionSchedule(
        cfg.diffusion_steps
    ).to(device)

    optimizer = torch.optim.Adam(
        ddpm.parameters(),
        lr=cfg.lr_ddpm,
    )

    c = float(
        global_scale
    )

    rows = []

    ddpm.train()

    for step in range(
        1,
        SMOKE_STEPS + 1,
    ):
        batch_e, _ = next(
            euro_iter
        )

        batch_d, _ = next(
            deep_iter
        )

        batch_e = batch_e.to(
            device,
            non_blocking=True,
        )

        batch_d = batch_d.to(
            device,
            non_blocking=True,
        )

        batch = min(
            batch_e.size(0),
            batch_d.size(0),
        )

        batch_e = batch_e[:batch]
        batch_d = batch_d[:batch]

        with torch.no_grad():
            mu_e, _ = encoder(
                batch_e
            )

            mu_d, _ = encoder(
                batch_d
            )

            # NEW target distribution.
            z_e = mu_e * c
            z_d = mu_d * c

            snr_db = torch.empty(
                batch,
                device=device,
            ).uniform_(
                cfg.snr_min_db,
                cfg.snr_max_db,
            )

            (
                observation,
                _,
                _,
            ) = (
                mimo_2x2_global_scale_observation(
                    z_e,
                    z_d,
                    snr_db,
                    cfg,
                )
            )

        t = torch.randint(
            0,
            min(
                cfg.train_t_max,
                cfg.diffusion_steps,
            ),
            (batch,),
            device=device,
            dtype=torch.long,
        )

        noise_e = torch.randn_like(
            z_e
        )

        noise_d = torch.randn_like(
            z_d
        )

        zt_e = schedule.q_sample(
            z_e,
            t,
            noise_e,
        )

        zt_d = schedule.q_sample(
            z_d,
            t,
            noise_d,
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

        pred_e = ddpm(
            zt_e,
            t,
            labels_e,
            observation,
        )

        pred_d = ddpm(
            zt_d,
            t,
            labels_d,
            observation,
        )

        pred_x0_e = (
            schedule.predict_x0(
                zt_e,
                t,
                pred_e,
            )
        )

        pred_x0_d = (
            schedule.predict_x0(
                zt_d,
                t,
                pred_d,
            )
        )

        loss_eps_e = F.mse_loss(
            pred_e,
            noise_e,
        )

        loss_eps_d = F.mse_loss(
            pred_d,
            noise_d,
        )

        loss_x0_e = F.smooth_l1_loss(
            pred_x0_e,
            z_e,
        )

        loss_x0_d = F.smooth_l1_loss(
            pred_x0_d,
            z_d,
        )

        loss = (
            0.5
            * (
                loss_eps_e
                + loss_eps_d
            )
            + cfg.lambda_x0
            * 0.5
            * (
                loss_x0_e
                + loss_x0_d
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        grad_norm = (
            torch.nn.utils
            .clip_grad_norm_(
                ddpm.parameters(),
                cfg.grad_clip,
            )
        )

        optimizer.step()

        row = {
            "step":
                step,

            "total_loss":
                float(
                    loss.item()
                ),

            "eps_eurosat":
                float(
                    loss_eps_e.item()
                ),

            "eps_deepglobe":
                float(
                    loss_eps_d.item()
                ),

            "x0_eurosat":
                float(
                    loss_x0_e.item()
                ),

            "x0_deepglobe":
                float(
                    loss_x0_d.item()
                ),

            "grad_norm":
                float(
                    grad_norm
                ),
        }

        rows.append(row)

        if (
            step == 1
            or step % SMOKE_LOG_EVERY == 0
            or step == SMOKE_STEPS
        ):
            print(
                f"Step {step:4d}/{SMOKE_STEPS} | "
                f"Total={row['total_loss']:.4f} | "
                f"epsE={row['eps_eurosat']:.4f} | "
                f"epsD={row['eps_deepglobe']:.4f} | "
                f"x0E={row['x0_eurosat']:.4f} | "
                f"x0D={row['x0_deepglobe']:.4f}"
            )

    write_rows(
        SAVE_ROOT
        / "06_global_ddpm_smoke_history.csv",
        rows,
    )

    losses = np.asarray(
        [
            row["total_loss"]
            for row in rows
        ],
        dtype=np.float64,
    )

    window = min(
        20,
        len(losses),
    )

    if window > 1:
        kernel = (
            np.ones(window)
            / window
        )

        smoothed = np.convolve(
            losses,
            kernel,
            mode="valid",
        )

        smooth_x = np.arange(
            window,
            len(losses) + 1,
        )
    else:
        smoothed = losses
        smooth_x = np.arange(
            1,
            len(losses) + 1,
        )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        np.arange(
            1,
            len(losses) + 1,
        ),
        losses,
        alpha=0.35,
        label="Raw total loss",
    )

    ax.plot(
        smooth_x,
        smoothed,
        linewidth=2,
        label=f"{window}-step moving average",
    )

    ax.set_xlabel(
        "Optimization step"
    )

    ax.set_ylabel(
        "Global-scale DDPM loss"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SAVE_ROOT
        / "06_global_ddpm_smoke_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    compare_window = min(
        50,
        max(
            1,
            len(losses) // 5,
        ),
    )

    first_mean = float(
        losses[
            :compare_window
        ].mean()
    )

    last_mean = float(
        losses[
            -compare_window:
        ].mean()
    )

    summary = {
        "steps":
            len(losses),

        "first_window_size":
            compare_window,

        "first_window_mean_loss":
            first_mean,

        "last_window_mean_loss":
            last_mean,

        "relative_change":
            (
                (
                    last_mean
                    - first_mean
                )
                / max(
                    abs(first_mean),
                    1e-12,
                )
            ),
    }

    write_rows(
        SAVE_ROOT
        / "06_global_ddpm_smoke_summary.csv",
        [summary],
    )

    # Save the smoke model ONLY inside diagnostic folder.
    torch.save(
        {
            "ddpm":
                ddpm.state_dict(),

            "config":
                vars(cfg),

            "global_scale":
                c,

            "smoke_steps":
                SMOKE_STEPS,
        },
        SAVE_ROOT
        / "global_ddpm_smoke_model.pth",
    )

    print()
    print(
        f"First-window mean loss: "
        f"{first_mean:.6f}"
    )

    print(
        f"Last-window mean loss : "
        f"{last_mean:.6f}"
    )

    if last_mean < first_mean:
        print(
            "[PASS] Global-scale DDPM smoke loss decreased overall."
        )
    else:
        print(
            "[WARNING] Global-scale DDPM smoke loss did not decrease overall."
        )

    ddpm.eval()

    return (
        ddpm,
        schedule,
        summary,
    )


# ============================================================
# Check 7 — short DDIM reconstruction after smoke training
# ============================================================

@torch.no_grad()
def evaluate_smoke_ddim(
    cfg,
    device: torch.device,
    encoder,
    generator,
    ddpm,
    schedule,
    global_scale: float,
):
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 7 — SHORT DDIM RECONSTRUCTION WITH GLOBAL SCALE\n"
        + "=" * 90
    )

    eurosat, deepglobe = (
        M.build_domain_datasets(
            cfg
        )
    )

    count = min(
        SMOKE_DDIM_EVAL_COUNT,
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
        batch_size=count,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    loader_d = DataLoader(
        subset_d,
        batch_size=count,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    image_e, _ = next(
        iter(loader_e)
    )

    image_d, _ = next(
        iter(loader_d)
    )

    image_e = image_e.to(device)
    image_d = image_d.to(device)

    mu_e, _ = encoder(
        image_e
    )

    mu_d, _ = encoder(
        image_d
    )

    labels_e = torch.full(
        (count,),
        M.EUROSAT_LABEL,
        device=device,
        dtype=torch.long,
    )

    labels_d = torch.full(
        (count,),
        M.DEEPGLOBE_LABEL,
        device=device,
        dtype=torch.long,
    )

    clean_e = generator(
        mu_e,
        labels_e,
    )

    clean_d = generator(
        mu_d,
        labels_d,
    )

    c = float(
        global_scale
    )

    z_e = mu_e * c
    z_d = mu_d * c

    rows = []

    for snr_index, snr_db in enumerate(
        SMOKE_DDIM_EVAL_SNRS
    ):
        channel_seed = (
            int(cfg.seed)
            + 700000
            + snr_index
        )

        torch.manual_seed(
            channel_seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                channel_seed
            )

        (
            observation,
            zf_scaled_e,
            zf_scaled_d,
        ) = (
            mimo_2x2_global_scale_observation(
                z_e,
                z_d,
                snr_db,
                cfg,
            )
        )

        # ZF branch.
        zf_mu_e = (
            zf_scaled_e / c
        )

        zf_mu_d = (
            zf_scaled_d / c
        )

        zf_recon_e = generator(
            zf_mu_e,
            labels_e,
        )

        zf_recon_d = generator(
            zf_mu_d,
            labels_d,
        )

        # DDPM branch.
        ddpm_scaled_e = (
            ddim_sample_global(
                model=ddpm,
                schedule=schedule,
                observation=observation,
                labels=labels_e,
                z_dim=cfg.z_dim,
                sample_steps=SMOKE_DDIM_STEPS,
                seed=(
                    int(cfg.seed)
                    + 800000
                    + snr_index * 10
                ),
            )
        )

        ddpm_scaled_d = (
            ddim_sample_global(
                model=ddpm,
                schedule=schedule,
                observation=observation,
                labels=labels_d,
                z_dim=cfg.z_dim,
                sample_steps=SMOKE_DDIM_STEPS,
                seed=(
                    int(cfg.seed)
                    + 800001
                    + snr_index * 10
                ),
            )
        )

        ddpm_mu_e = (
            ddpm_scaled_e / c
        )

        ddpm_mu_d = (
            ddpm_scaled_d / c
        )

        ddpm_recon_e = generator(
            ddpm_mu_e,
            labels_e,
        )

        ddpm_recon_d = generator(
            ddpm_mu_d,
            labels_d,
        )

        row = {
            "snr_db":
                float(snr_db),

            "eurosat_clean_psnr":
                float(
                    np.mean(
                        batch_psnr(
                            clean_e,
                            image_e,
                        )
                    )
                ),

            "eurosat_zf_psnr":
                float(
                    np.mean(
                        batch_psnr(
                            zf_recon_e,
                            image_e,
                        )
                    )
                ),

            "eurosat_ddpm_psnr":
                float(
                    np.mean(
                        batch_psnr(
                            ddpm_recon_e,
                            image_e,
                        )
                    )
                ),

            "eurosat_clean_ssim":
                float(
                    np.mean(
                        batch_ssim(
                            clean_e,
                            image_e,
                        )
                    )
                ),

            "eurosat_zf_ssim":
                float(
                    np.mean(
                        batch_ssim(
                            zf_recon_e,
                            image_e,
                        )
                    )
                ),

            "eurosat_ddpm_ssim":
                float(
                    np.mean(
                        batch_ssim(
                            ddpm_recon_e,
                            image_e,
                        )
                    )
                ),

            "deepglobe_clean_psnr":
                float(
                    np.mean(
                        batch_psnr(
                            clean_d,
                            image_d,
                        )
                    )
                ),

            "deepglobe_zf_psnr":
                float(
                    np.mean(
                        batch_psnr(
                            zf_recon_d,
                            image_d,
                        )
                    )
                ),

            "deepglobe_ddpm_psnr":
                float(
                    np.mean(
                        batch_psnr(
                            ddpm_recon_d,
                            image_d,
                        )
                    )
                ),

            "deepglobe_clean_ssim":
                float(
                    np.mean(
                        batch_ssim(
                            clean_d,
                            image_d,
                        )
                    )
                ),

            "deepglobe_zf_ssim":
                float(
                    np.mean(
                        batch_ssim(
                            zf_recon_d,
                            image_d,
                        )
                    )
                ),

            "deepglobe_ddpm_ssim":
                float(
                    np.mean(
                        batch_ssim(
                            ddpm_recon_d,
                            image_d,
                        )
                    )
                ),
        }

        rows.append(row)

        print(
            f"SNR={snr_db:g} dB | "
            f"EuroSAT: clean={row['eurosat_clean_psnr']:.3f}, "
            f"ZF={row['eurosat_zf_psnr']:.3f}, "
            f"DDPM={row['eurosat_ddpm_psnr']:.3f}"
        )

        print(
            f"              DeepGlobe: clean={row['deepglobe_clean_psnr']:.3f}, "
            f"ZF={row['deepglobe_zf_psnr']:.3f}, "
            f"DDPM={row['deepglobe_ddpm_psnr']:.3f}"
        )

        save_comparison_grid(
            SAVE_ROOT
            / f"07_eurosat_smoke_ddim_snr{snr_db:g}.png",
            image_e,
            [
                (
                    "Raw clean",
                    clean_e,
                ),
                (
                    "Global ZF",
                    zf_recon_e,
                ),
                (
                    "Global DDPM smoke",
                    ddpm_recon_e,
                ),
            ],
            QUALITATIVE_COUNT,
        )

        save_comparison_grid(
            SAVE_ROOT
            / f"07_deepglobe_smoke_ddim_snr{snr_db:g}.png",
            image_d,
            [
                (
                    "Raw clean",
                    clean_d,
                ),
                (
                    "Global ZF",
                    zf_recon_d,
                ),
                (
                    "Global DDPM smoke",
                    ddpm_recon_d,
                ),
            ],
            QUALITATIVE_COUNT,
        )

    write_rows(
        SAVE_ROOT
        / "07_smoke_ddim_metrics.csv",
        rows,
    )

    # Plot PSNR for quick visual check.
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    x = [
        row["snr_db"]
        for row in rows
    ]

    ax.plot(
        x,
        [
            row["eurosat_zf_psnr"]
            for row in rows
        ],
        marker="o",
        label="EuroSAT ZF",
    )

    ax.plot(
        x,
        [
            row["eurosat_ddpm_psnr"]
            for row in rows
        ],
        marker="o",
        linestyle="--",
        label="EuroSAT DDPM smoke",
    )

    ax.plot(
        x,
        [
            row["deepglobe_zf_psnr"]
            for row in rows
        ],
        marker="s",
        label="DeepGlobe ZF",
    )

    ax.plot(
        x,
        [
            row["deepglobe_ddpm_psnr"]
            for row in rows
        ],
        marker="s",
        linestyle="--",
        label="DeepGlobe DDPM smoke",
    )

    ax.set_xlabel(
        "SNR (dB)"
    )

    ax.set_ylabel(
        "PSNR (dB)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        SAVE_ROOT
        / "07_smoke_ddim_psnr.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return rows


# ============================================================
# Final report
# ============================================================

def write_final_report(
    checkpoint_result: dict,
    scale_result: dict,
    roundtrip_result: dict,
    mimo_rows: List[dict],
    interface_result: dict,
    smoke_result: dict | None,
    ddim_rows: List[dict] | None,
):
    lines = []

    lines.append(
        "FINAL GLOBAL-SCALE PRE-DDPM CHECK"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")
    lines.append(
        "1. CHECKPOINT"
    )

    lines.append(
        f"Joint best epoch: "
        f"{checkpoint_result['joint_best_epoch']}"
    )

    lines.append(
        f"Encoder diff vs joint: "
        f"{checkpoint_result['encoder_vs_joint_max_abs_diff']}"
    )

    lines.append(
        f"Generator diff vs joint: "
        f"{checkpoint_result['generator_vs_joint_max_abs_diff']}"
    )

    lines.append("")
    lines.append(
        "2. GLOBAL SCALE"
    )

    lines.append(
        f"Calibration samples/domain: "
        f"{scale_result['calibration_samples_per_domain']}"
    )

    lines.append(
        f"EuroSAT latent power: "
        f"{scale_result['eurosat_latent_power']:.8f}"
    )

    lines.append(
        f"DeepGlobe latent power: "
        f"{scale_result['deepglobe_latent_power']:.8f}"
    )

    lines.append(
        f"Shared global power: "
        f"{scale_result['global_latent_power']:.8f}"
    )

    lines.append(
        f"Global scale c: "
        f"{scale_result['global_scale']:.8f}"
    )

    lines.append("")
    lines.append(
        "3. ROUNDTRIP"
    )

    lines.append(
        f"EuroSAT PSNR drop: "
        f"{roundtrip_result['eurosat_psnr_drop']:.6f} dB"
    )

    lines.append(
        f"DeepGlobe PSNR drop: "
        f"{roundtrip_result['deepglobe_psnr_drop']:.6f} dB"
    )

    lines.append("")
    lines.append(
        "4. GLOBAL MIMO + ZF"
    )

    for row in mimo_rows:
        lines.append(
            f"SNR {row['snr_db']:g}: "
            f"E ZF={row['eurosat_zf_psnr']:.3f}, "
            f"D ZF={row['deepglobe_zf_psnr']:.3f}"
        )

    lines.append("")
    lines.append(
        "5. DDPM INTERFACE"
    )

    lines.append(
        f"Perfect-eps x0 MSE: "
        f"{interface_result['perfect_eps_x0_mse']:.8e}"
    )

    if smoke_result is not None:
        lines.append("")
        lines.append(
            "6. GLOBAL DDPM SMOKE"
        )

        lines.append(
            f"First-window loss: "
            f"{smoke_result['first_window_mean_loss']:.6f}"
        )

        lines.append(
            f"Last-window loss: "
            f"{smoke_result['last_window_mean_loss']:.6f}"
        )

    if ddim_rows is not None:
        lines.append("")
        lines.append(
            "7. SMOKE DDIM EVALUATION"
        )

        for row in ddim_rows:
            lines.append(
                f"SNR {row['snr_db']:g}: "
                f"E clean/ZF/DDPM="
                f"{row['eurosat_clean_psnr']:.3f}/"
                f"{row['eurosat_zf_psnr']:.3f}/"
                f"{row['eurosat_ddpm_psnr']:.3f}; "
                f"D clean/ZF/DDPM="
                f"{row['deepglobe_clean_psnr']:.3f}/"
                f"{row['deepglobe_zf_psnr']:.3f}/"
                f"{row['deepglobe_ddpm_psnr']:.3f}"
            )

    lines.append("")
    lines.append(
        "INTERPRETATION"
    )

    lines.append(
        "- Global roundtrip should have approximately 0 dB PSNR loss."
    )

    lines.append(
        "- High-SNR ZF should approach the raw-mu reconstruction ceiling."
    )

    lines.append(
        "- Global-scale DDPM smoke loss should decrease overall."
    )

    lines.append(
        "- A 500-step smoke DDPM is NOT expected to beat a strong ZF baseline."
    )

    lines.append(
        "- The DDIM check only verifies that the end-to-end DDPM inference path "
        "does not collapse and produces finite/reasonable outputs."
    )

    (
        SAVE_ROOT
        / "FINAL_GLOBAL_PRECHECK_REPORT.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(
        SAVE_ROOT
    )

    cfg = M.Config()

    device = M.resolve_device(
        cfg.device
    )

    set_seed(
        int(cfg.seed)
    )

    print(
        "Device:",
        device,
    )

    print(
        "Output folder:",
        SAVE_ROOT.resolve(),
    )

    print(
        "Original checkpoints are NOT overwritten."
    )

    # --------------------------------------------------------
    # Check 1
    # --------------------------------------------------------
    checkpoint_result = (
        check_checkpoint_consistency(
            cfg
        )
    )

    # Load frozen backbone exactly once.
    encoder = M.load_encoder(
        cfg,
        device,
    )

    generator, _ = M.load_gan_models(
        cfg,
        device,
    )

    encoder.eval()
    generator.eval()

    M.set_requires_grad(
        encoder,
        False,
    )

    M.set_requires_grad(
        generator,
        False,
    )

    # --------------------------------------------------------
    # Check 2 — global scale calibration
    # --------------------------------------------------------
    scale_result = (
        estimate_and_save_global_scale(
            cfg,
            device,
            encoder,
        )
    )

    global_scale = float(
        scale_result[
            "global_scale"
        ]
    )

    # Fixed diagnostic pair set.
    (
        loader_e,
        loader_d,
        count,
    ) = (
        build_fixed_diagnostic_loaders(
            cfg
        )
    )

    print(
        "\nFixed diagnostic pairs:",
        count,
    )

    # --------------------------------------------------------
    # Check 3
    # --------------------------------------------------------
    roundtrip_result = (
        check_global_scale_roundtrip(
            cfg,
            device,
            loader_e,
            loader_d,
            encoder,
            generator,
            global_scale,
        )
    )

    # --------------------------------------------------------
    # Check 4
    # --------------------------------------------------------
    mimo_rows = (
        check_global_scale_mimo_zf(
            cfg,
            device,
            loader_e,
            loader_d,
            encoder,
            generator,
            global_scale,
        )
    )

    # --------------------------------------------------------
    # Check 5
    # --------------------------------------------------------
    interface_result = (
        check_global_ddpm_interface(
            cfg,
            device,
            loader_e,
            loader_d,
            encoder,
            global_scale,
        )
    )

    # --------------------------------------------------------
    # Check 6
    # --------------------------------------------------------
    ddpm = None
    schedule = None
    smoke_result = None

    if RUN_GLOBAL_DDPM_SMOKE:
        (
            ddpm,
            schedule,
            smoke_result,
        ) = train_global_ddpm_smoke(
            cfg,
            device,
            encoder,
            global_scale,
        )

    # --------------------------------------------------------
    # Check 7
    # --------------------------------------------------------
    ddim_rows = None

    if (
        RUN_SMOKE_DDIM_EVAL
        and ddpm is not None
        and schedule is not None
    ):
        ddim_rows = (
            evaluate_smoke_ddim(
                cfg,
                device,
                encoder,
                generator,
                ddpm,
                schedule,
                global_scale,
            )
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    write_final_report(
        checkpoint_result,
        scale_result,
        roundtrip_result,
        mimo_rows,
        interface_result,
        smoke_result,
        ddim_rows,
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL GLOBAL-SCALE PRECHECK COMPLETED"
    )

    print(
        "=" * 90
    )

    print()
    print(
        "Most important outputs:"
    )

    print(
        "  global_latent_scale.json"
    )

    print(
        "  03_global_roundtrip_metrics.csv"
    )

    print(
        "  04_global_mimo_zf_metrics.csv"
    )

    print(
        "  05_global_ddpm_interface.csv"
    )

    if RUN_GLOBAL_DDPM_SMOKE:
        print(
            "  06_global_ddpm_smoke_history.csv"
        )

        print(
            "  06_global_ddpm_smoke_loss.png"
        )

    if RUN_SMOKE_DDIM_EVAL:
        print(
            "  07_smoke_ddim_metrics.csv"
        )

        print(
            "  07_smoke_ddim_psnr.png"
        )

    print(
        "  FINAL_GLOBAL_PRECHECK_REPORT.txt"
    )


if __name__ == "__main__":
    main()
