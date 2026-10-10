"""
Pre-DDPM full diagnostic for:
Conditional GAN pretraining -> Joint mu-VAE-GAN -> 2x2 MIMO -> Latent DDPM

Purpose
-------
Run this BEFORE the long DDPM training.

This script checks:
1. Which checkpoints are actually being loaded, and whether the separately
   exported Encoder / Generator match the joint-best checkpoint.
2. Raw encoder-mu reconstruction vs. power-normalized-mu reconstruction.
3. Latent statistics before / after power normalization.
4. 2x2 MIMO + ZF reconstruction at SNR = 0, 10, 20, 30 dB.
5. DDPM tensor/interface consistency and diffusion schedule sanity.
6. Fixed-global-scale roundtrip and MIMO/ZF checks.
7. Optional short DDPM smoke training to verify that losses can decrease.

Nothing in the original training folder is overwritten.
All diagnostic outputs are saved to a SEPARATE folder:

    ./save_data/Deep_WGANGP_VAE_DDPM_PRECHECK

Expected setup
--------------
Put this file in the same folder as:

    train_gan_pretrain_then_joint_mu_vaegan_ddpm.py

Then run:

    python pre_ddpm_full_diagnostic.py

If your training script has another filename, change TRAINING_SCRIPT below.
"""

from __future__ import annotations

import csv
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image


# ============================================================
# User settings
# ============================================================

# Completely separate output folder.
DIAGNOSTIC_ROOT = Path(
    "./save_data/Deep_WGANGP_VAE_DDPM_PRECHECK"
)

# Number of paired images used for reconstruction/channel checks.
CHECK_COUNT = 256

# The SNRs to sanity-check before DDPM training.
CHECK_SNRS = (0.0, 10.0, 20.0, 30.0)

# Short DDPM training test.
RUN_DDPM_SMOKE_TEST = True
SMOKE_STEPS = 200
SMOKE_BATCH_SIZE = 16
SMOKE_LOG_EVERY = 10

# Number of samples shown in qualitative grids.
QUALITATIVE_COUNT = 8


# ============================================================
# Import your original training script
# ============================================================

# This diagnostic script is intended to be placed in the SAME folder as:
#     train_gan_pretrain_then_joint_mu_vaegan_ddpm.py
#
# Use a normal Python import instead of importlib dynamic execution.
# This avoids the Python 3.11 @dataclass/sys.modules issue entirely.

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import train_gan_pretrain_then_joint_mu_vaegan_ddpm as M
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import 'train_gan_pretrain_then_joint_mu_vaegan_ddpm.py'.\n"
        "Please put pre_ddpm_full_diagnostic_v2.py in the SAME folder as "
        "train_gan_pretrain_then_joint_mu_vaegan_ddpm.py, then run it again."
    ) from exc


# ============================================================
# Small utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: List[dict]):
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

    psnr = 10.0 * torch.log10(1.0 / mse)

    return psnr.cpu().numpy()


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

    return np.asarray(values, dtype=np.float64)


def mean_std(values) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)

    return (
        float(arr.mean()),
        float(arr.std()),
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

        diff = (
            ta - tb
        ).abs().max().item()

        max_diff = max(
            max_diff,
            diff,
        )

    return float(max_diff)


def checkpoint_epoch(path: str):
    if not os.path.exists(path):
        return None

    obj = torch.load(
        path,
        map_location="cpu",
    )

    if isinstance(obj, dict):
        return obj.get(
            "epoch",
            None,
        )

    return None


def save_comparison_grid(
    path: Path,
    real: torch.Tensor,
    outputs: List[Tuple[str, torch.Tensor]],
    count: int,
):
    """
    The row labels are written to a companion TXT file.
    Image rows:
        Real
        output 1
        output 2
        ...
    """

    count = min(
        count,
        real.size(0),
    )

    rows = [
        real[:count]
    ]

    labels = [
        "Real"
    ]

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

    label_path = path.with_suffix(".txt")

    label_path.write_text(
        "\n".join(
            f"Row {i + 1}: {name}"
            for i, name in enumerate(labels)
        ),
        encoding="utf-8",
    )


# ============================================================
# Check 1 — checkpoint consistency
# ============================================================

def check_checkpoints(
    cfg,
    device: torch.device,
) -> dict:
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 1 — CHECKPOINT CONSISTENCY\n"
        + "=" * 90
    )

    paths = {
        "joint_best": cfg.joint_best_checkpoint,
        "joint_latest": cfg.joint_checkpoint,
        "encoder_best": cfg.encoder_best_checkpoint,
        "encoder_latest": cfg.encoder_checkpoint,
        "gan_best": cfg.gan_best_checkpoint,
        "gan_latest": cfg.gan_checkpoint,
    }

    rows = []

    for name, path in paths.items():
        exists = os.path.exists(path)
        epoch = checkpoint_epoch(path)

        rows.append(
            {
                "checkpoint": name,
                "path": path,
                "exists": exists,
                "epoch": epoch,
            }
        )

        print(
            f"{name:15s} | "
            f"exists={str(exists):5s} | "
            f"epoch={epoch} | "
            f"{path}"
        )

    write_rows(
        DIAGNOSTIC_ROOT
        / "01_checkpoint_inventory.csv",
        rows,
    )

    if not os.path.exists(
        cfg.joint_best_checkpoint
    ):
        raise FileNotFoundError(
            "joint_best checkpoint does not exist. "
            "Do not start DDPM yet."
        )

    if not os.path.exists(
        cfg.encoder_best_checkpoint
    ):
        raise FileNotFoundError(
            "encoder_best checkpoint does not exist. "
            "Do not start DDPM yet."
        )

    if not os.path.exists(
        cfg.gan_best_checkpoint
    ):
        raise FileNotFoundError(
            "gan_best checkpoint does not exist. "
            "Do not start DDPM yet."
        )

    # Load joint-best directly.
    joint_ckpt = torch.load(
        cfg.joint_best_checkpoint,
        map_location="cpu",
    )

    # Load separately exported best encoder / GAN.
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
        joint_ckpt["encoder"],
        encoder_state,
    )

    generator_diff = state_dict_max_abs_diff(
        joint_ckpt["generator"],
        generator_state,
    )

    result = {
        "joint_best_epoch":
            joint_ckpt.get("epoch"),
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
        "Joint best epoch      :",
        result["joint_best_epoch"],
    )
    print(
        "Encoder best epoch    :",
        result["encoder_best_epoch"],
    )
    print(
        "Generator best epoch  :",
        result["gan_best_epoch"],
    )
    print(
        "Encoder max |diff|    :",
        encoder_diff,
    )
    print(
        "Generator max |diff|  :",
        generator_diff,
    )

    if encoder_diff == 0.0:
        print(
            "[PASS] Separate best encoder "
            "matches joint-best encoder exactly."
        )
    else:
        print(
            "[WARNING] Separate best encoder DOES NOT "
            "exactly match joint-best encoder."
        )

    if generator_diff == 0.0:
        print(
            "[PASS] Separate best generator "
            "matches joint-best generator exactly."
        )
    else:
        print(
            "[WARNING] Separate best generator DOES NOT "
            "exactly match joint-best generator."
        )

    write_rows(
        DIAGNOSTIC_ROOT
        / "01_checkpoint_consistency.csv",
        [result],
    )

    return result


# ============================================================
# Dataset loader used for all following checks
# ============================================================

def build_fixed_paired_loader(
    cfg,
):
    eurosat, deepglobe = (
        M.build_domain_datasets(cfg)
    )

    count = min(
        CHECK_COUNT,
        len(eurosat),
        len(deepglobe),
    )

    if count <= 0:
        raise RuntimeError(
            "No paired samples are available."
        )

    # Fixed deterministic subset for every diagnostic.
    idx_e = list(
        range(count)
    )

    idx_d = list(
        range(count)
    )

    subset_e = torch.utils.data.Subset(
        eurosat,
        idx_e,
    )

    subset_d = torch.utils.data.Subset(
        deepglobe,
        idx_d,
    )

    loader_e = DataLoader(
        subset_e,
        batch_size=min(32, count),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    loader_d = DataLoader(
        subset_d,
        batch_size=min(32, count),
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
# Check 2 — raw mu vs normalized mu
# ============================================================

@torch.no_grad()
def check_latent_normalization(
    cfg,
    device: torch.device,
    loader_e,
    loader_d,
):
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 2 — RAW MU vs POWER-NORMALIZED MU\n"
        + "=" * 90
    )

    # Load exactly the same path that the original DDPM stage uses.
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

    result_lists: Dict[str, List[float]] = {}

    latent_rows = []

    qualitative_done = False

    for batch_index, (
        batch_e,
        batch_d,
    ) in enumerate(
        zip(loader_e, loader_d)
    ):
        image_e, _ = batch_e
        image_d, _ = batch_d

        image_e = image_e.to(device)
        image_d = image_d.to(device)

        z_e, _ = encoder(image_e)
        z_d, _ = encoder(image_d)

        zn_e = M.normalize_latent_power(
            z_e
        )
        zn_d = M.normalize_latent_power(
            z_d
        )

        raw_e = generator(
            z_e,
            torch.full(
                (z_e.size(0),),
                M.EUROSAT_LABEL,
                dtype=torch.long,
                device=device,
            ),
        )

        norm_e = generator(
            zn_e,
            torch.full(
                (z_e.size(0),),
                M.EUROSAT_LABEL,
                dtype=torch.long,
                device=device,
            ),
        )

        raw_d = generator(
            z_d,
            torch.full(
                (z_d.size(0),),
                M.DEEPGLOBE_LABEL,
                dtype=torch.long,
                device=device,
            ),
        )

        norm_d = generator(
            zn_d,
            torch.full(
                (z_d.size(0),),
                M.DEEPGLOBE_LABEL,
                dtype=torch.long,
                device=device,
            ),
        )

        metrics = {
            "eurosat_raw_psnr":
                batch_psnr(raw_e, image_e),
            "eurosat_norm_psnr":
                batch_psnr(norm_e, image_e),
            "eurosat_raw_ssim":
                batch_ssim(raw_e, image_e),
            "eurosat_norm_ssim":
                batch_ssim(norm_e, image_e),

            "deepglobe_raw_psnr":
                batch_psnr(raw_d, image_d),
            "deepglobe_norm_psnr":
                batch_psnr(norm_d, image_d),
            "deepglobe_raw_ssim":
                batch_ssim(raw_d, image_d),
            "deepglobe_norm_ssim":
                batch_ssim(norm_d, image_d),
        }

        for key, values in metrics.items():
            result_lists.setdefault(
                key,
                [],
            ).extend(
                np.asarray(values)
                .reshape(-1)
                .tolist()
            )

        # Latent statistics.
        for name, z, zn in (
            ("EuroSAT", z_e, zn_e),
            ("DeepGlobe", z_d, zn_d),
        ):
            p_raw = (
                z.pow(2)
                .mean(dim=1)
                .detach()
                .cpu()
                .numpy()
            )

            p_norm = (
                zn.pow(2)
                .mean(dim=1)
                .detach()
                .cpu()
                .numpy()
            )

            latent_rows.append(
                {
                    "dataset": name,
                    "batch": batch_index,
                    "raw_mu_mean":
                        float(z.mean().item()),
                    "raw_mu_std":
                        float(z.std().item()),
                    "raw_power_mean":
                        float(p_raw.mean()),
                    "normalized_mu_mean":
                        float(zn.mean().item()),
                    "normalized_mu_std":
                        float(zn.std().item()),
                    "normalized_power_mean":
                        float(p_norm.mean()),
                }
            )

        if not qualitative_done:
            save_comparison_grid(
                DIAGNOSTIC_ROOT
                / "02_eurosat_raw_vs_normalized.png",
                image_e,
                [
                    ("Raw mu", raw_e),
                    ("Normalized mu", norm_e),
                ],
                QUALITATIVE_COUNT,
            )

            save_comparison_grid(
                DIAGNOSTIC_ROOT
                / "02_deepglobe_raw_vs_normalized.png",
                image_d,
                [
                    ("Raw mu", raw_d),
                    ("Normalized mu", norm_d),
                ],
                QUALITATIVE_COUNT,
            )

            qualitative_done = True

    summary = {}

    for key, values in result_lists.items():
        m, s = mean_std(values)

        summary[key + "_mean"] = m
        summary[key + "_std"] = s

    summary[
        "eurosat_psnr_drop_due_to_normalization"
    ] = (
        summary["eurosat_raw_psnr_mean"]
        - summary["eurosat_norm_psnr_mean"]
    )

    summary[
        "deepglobe_psnr_drop_due_to_normalization"
    ] = (
        summary["deepglobe_raw_psnr_mean"]
        - summary["deepglobe_norm_psnr_mean"]
    )

    print(
        "EuroSAT raw PSNR : "
        f"{summary['eurosat_raw_psnr_mean']:.3f} dB"
    )
    print(
        "EuroSAT norm PSNR: "
        f"{summary['eurosat_norm_psnr_mean']:.3f} dB"
    )
    print(
        "EuroSAT PSNR drop: "
        f"{summary['eurosat_psnr_drop_due_to_normalization']:.3f} dB"
    )

    print()

    print(
        "DeepGlobe raw PSNR : "
        f"{summary['deepglobe_raw_psnr_mean']:.3f} dB"
    )
    print(
        "DeepGlobe norm PSNR: "
        f"{summary['deepglobe_norm_psnr_mean']:.3f} dB"
    )
    print(
        "DeepGlobe PSNR drop: "
        f"{summary['deepglobe_psnr_drop_due_to_normalization']:.3f} dB"
    )

    write_rows(
        DIAGNOSTIC_ROOT
        / "02_normalization_metrics.csv",
        [summary],
    )

    write_rows(
        DIAGNOSTIC_ROOT
        / "02_latent_statistics.csv",
        latent_rows,
    )

    return (
        encoder,
        generator,
        summary,
    )


# ============================================================
# Check 2B — fixed global latent scaling + inverse scaling
# ============================================================

@torch.no_grad()
def estimate_global_latent_scale(
    encoder,
    device: torch.device,
    loader_e,
    loader_d,
):
    """
    Estimate ONE fixed scalar c from the diagnostic calibration set:

        P_global = E[mu^2]
        c = 1 / sqrt(P_global)

    This is intentionally different from per-sample normalization.
    Every sample uses the same c.

    For the final paper/training pipeline, estimate c once from a
    sufficiently large TRAINING calibration set and then freeze it.
    """
    total_sq = 0.0
    total_numel = 0

    for batch_e, batch_d in zip(loader_e, loader_d):
        image_e, _ = batch_e
        image_d, _ = batch_d

        image_e = image_e.to(device)
        image_d = image_d.to(device)

        mu_e, _ = encoder(image_e)
        mu_d, _ = encoder(image_d)

        total_sq += float(mu_e.pow(2).sum().item())
        total_sq += float(mu_d.pow(2).sum().item())
        total_numel += mu_e.numel() + mu_d.numel()

    if total_numel <= 0:
        raise RuntimeError("No latent elements were available to estimate global scale.")

    global_power = total_sq / total_numel
    global_scale = 1.0 / math.sqrt(global_power + 1e-12)

    return float(global_power), float(global_scale)


@torch.no_grad()
def check_global_scale_roundtrip(
    cfg,
    device: torch.device,
    loader_e,
    loader_d,
    encoder,
    generator,
):
    """
    Verify the proposed fixed-global-scaling interface:

        mu_raw
          -> z_tx = c * mu_raw
          -> receiver-side inverse scaling = z_tx / c
          -> Generator

    If implemented correctly, the reconstructed image should be almost
    identical to Generator(mu_raw), because c is an exactly invertible,
    fixed scalar.

    This is the key pre-check before changing DDPM training to use
    fixed global scaling instead of per-sample normalization.
    """
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 2B — FIXED GLOBAL SCALE + INVERSE SCALE ROUNDTRIP\n"
        + "=" * 90
    )

    global_power, global_scale = estimate_global_latent_scale(
        encoder=encoder,
        device=device,
        loader_e=loader_e,
        loader_d=loader_d,
    )

    print(f"Estimated shared latent power P_global = {global_power:.8f}")
    print(f"Fixed global scale c = 1/sqrt(P_global) = {global_scale:.8f}")

    result_lists: Dict[str, List[float]] = {}
    latent_mse_e = []
    latent_mse_d = []
    qualitative_done = False

    for batch_e, batch_d in zip(loader_e, loader_d):
        image_e, _ = batch_e
        image_d, _ = batch_d

        image_e = image_e.to(device)
        image_d = image_d.to(device)

        mu_e, _ = encoder(image_e)
        mu_d, _ = encoder(image_d)

        labels_e = torch.full(
            (mu_e.size(0),),
            M.EUROSAT_LABEL,
            dtype=torch.long,
            device=device,
        )

        labels_d = torch.full(
            (mu_d.size(0),),
            M.DEEPGLOBE_LABEL,
            dtype=torch.long,
            device=device,
        )

        # Raw reconstruction ceiling.
        raw_e = generator(mu_e, labels_e)
        raw_d = generator(mu_d, labels_d)

        # Fixed global transmitter scaling.
        ztx_e = mu_e * global_scale
        ztx_d = mu_d * global_scale

        # Exact receiver-side inverse scaling.
        recovered_mu_e = ztx_e / global_scale
        recovered_mu_d = ztx_d / global_scale

        restored_e = generator(recovered_mu_e, labels_e)
        restored_d = generator(recovered_mu_d, labels_d)

        latent_mse_e.append(
            F.mse_loss(recovered_mu_e, mu_e).item()
        )
        latent_mse_d.append(
            F.mse_loss(recovered_mu_d, mu_d).item()
        )

        metrics = {
            "eurosat_raw_psnr": batch_psnr(raw_e, image_e),
            "eurosat_global_roundtrip_psnr": batch_psnr(restored_e, image_e),
            "eurosat_raw_ssim": batch_ssim(raw_e, image_e),
            "eurosat_global_roundtrip_ssim": batch_ssim(restored_e, image_e),

            "deepglobe_raw_psnr": batch_psnr(raw_d, image_d),
            "deepglobe_global_roundtrip_psnr": batch_psnr(restored_d, image_d),
            "deepglobe_raw_ssim": batch_ssim(raw_d, image_d),
            "deepglobe_global_roundtrip_ssim": batch_ssim(restored_d, image_d),
        }

        for key, values in metrics.items():
            result_lists.setdefault(key, []).extend(
                np.asarray(values).reshape(-1).tolist()
            )

        if not qualitative_done:
            save_comparison_grid(
                DIAGNOSTIC_ROOT / "02B_eurosat_global_scale_roundtrip.png",
                image_e,
                [
                    ("Raw mu reconstruction", raw_e),
                    ("Global scale -> inverse scale", restored_e),
                ],
                QUALITATIVE_COUNT,
            )

            save_comparison_grid(
                DIAGNOSTIC_ROOT / "02B_deepglobe_global_scale_roundtrip.png",
                image_d,
                [
                    ("Raw mu reconstruction", raw_d),
                    ("Global scale -> inverse scale", restored_d),
                ],
                QUALITATIVE_COUNT,
            )

            qualitative_done = True

    summary = {
        "global_power": global_power,
        "global_scale": global_scale,
        "eurosat_latent_roundtrip_mse": float(np.mean(latent_mse_e)),
        "deepglobe_latent_roundtrip_mse": float(np.mean(latent_mse_d)),
    }

    for key, values in result_lists.items():
        summary[key + "_mean"] = float(np.mean(values))
        summary[key + "_std"] = float(np.std(values))

    summary["eurosat_psnr_drop_after_global_roundtrip"] = (
        summary["eurosat_raw_psnr_mean"]
        - summary["eurosat_global_roundtrip_psnr_mean"]
    )

    summary["deepglobe_psnr_drop_after_global_roundtrip"] = (
        summary["deepglobe_raw_psnr_mean"]
        - summary["deepglobe_global_roundtrip_psnr_mean"]
    )

    print(
        f"EuroSAT raw PSNR      : {summary['eurosat_raw_psnr_mean']:.3f} dB"
    )
    print(
        f"EuroSAT roundtrip PSNR: {summary['eurosat_global_roundtrip_psnr_mean']:.3f} dB"
    )
    print(
        "EuroSAT roundtrip drop: "
        f"{summary['eurosat_psnr_drop_after_global_roundtrip']:.6f} dB"
    )

    print()

    print(
        f"DeepGlobe raw PSNR      : {summary['deepglobe_raw_psnr_mean']:.3f} dB"
    )
    print(
        f"DeepGlobe roundtrip PSNR: {summary['deepglobe_global_roundtrip_psnr_mean']:.3f} dB"
    )
    print(
        "DeepGlobe roundtrip drop: "
        f"{summary['deepglobe_psnr_drop_after_global_roundtrip']:.6f} dB"
    )

    write_rows(
        DIAGNOSTIC_ROOT / "02B_global_scale_roundtrip_metrics.csv",
        [summary],
    )

    # PASS criterion: the exact scale/unscale operation should be numerically
    # lossless and PSNR should remain essentially identical.
    if (
        abs(summary["eurosat_psnr_drop_after_global_roundtrip"]) < 1e-4
        and abs(summary["deepglobe_psnr_drop_after_global_roundtrip"]) < 1e-4
    ):
        print(
            "[PASS] Fixed global scaling is exactly invertible before the channel."
        )
    else:
        print(
            "[WARNING] Global scale roundtrip changed reconstruction unexpectedly."
        )

    return summary


# ============================================================
# Check 2C — fixed-global-scale 2x2 MIMO + ZF + inverse scaling
# ============================================================

def mimo_2x2_observation_no_internal_normalization(
    z_e: torch.Tensor,
    z_g: torch.Tensor,
    snr_db: torch.Tensor | float,
    cfg,
):
    """
    Same 2x2 MIMO model as the training script, EXCEPT this function
    does NOT call per-sample normalize_latent_power() internally.

    Inputs are assumed to already be transmitter latents:

        z_tx = c * mu_raw

    This is required for a true fixed-global-scaling experiment.
    """
    if z_e.shape != z_g.shape:
        raise ValueError(
            f"Latent shapes must match, got {z_e.shape} and {z_g.shape}"
        )

    streams = torch.stack([z_e, z_g], dim=1)  # [B, 2, Z]
    h = M.get_channel_matrix(cfg, streams.device, streams.dtype)

    clean_y = torch.einsum(
        "ij,bjz->biz",
        h,
        streams,
    )

    if not torch.is_tensor(snr_db):
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

    signal_power = clean_y.pow(2).mean(
        dim=(1, 2),
        keepdim=True,
    )

    noise_power = signal_power / torch.pow(
        10.0,
        snr_db.view(-1, 1, 1) / 10.0,
    )

    noisy_y = clean_y + torch.randn_like(clean_y) * torch.sqrt(
        noise_power + 1e-12
    )

    h_inv = torch.linalg.inv(h)

    zf = torch.einsum(
        "ij,bjz->biz",
        h_inv,
        noisy_y,
    )

    observation = noisy_y.flatten(start_dim=1)

    return observation, zf[:, 0], zf[:, 1]


@torch.no_grad()
def check_global_scale_mimo_zf(
    cfg,
    device: torch.device,
    loader_e,
    loader_d,
    encoder,
    generator,
    global_scale_summary: dict,
):
    """
    Full fixed-global-scaling sanity path:

        raw mu
          -> multiply by fixed global c
          -> 2x2 MIMO + AWGN
          -> ZF
          -> divide by fixed global c
          -> Generator

    At high SNR, ZF reconstruction should approach the RAW-mu
    reconstruction ceiling, not the poor per-sample-normalized ceiling.
    """
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 2C — GLOBAL SCALE -> MIMO -> ZF -> INVERSE SCALE\n"
        + "=" * 90
    )

    c = float(
        global_scale_summary["global_scale"]
    )

    rows = []
    qualitative_saved = set()

    for snr_db in CHECK_SNRS:
        clean_e_psnr = []
        clean_d_psnr = []
        zf_e_psnr = []
        zf_d_psnr = []
        zf_e_ssim = []
        zf_d_ssim = []

        for batch_index, (
            batch_e,
            batch_d,
        ) in enumerate(
            zip(loader_e, loader_d)
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
                dtype=torch.long,
                device=device,
            )

            labels_d = torch.full(
                (mu_d.size(0),),
                M.DEEPGLOBE_LABEL,
                dtype=torch.long,
                device=device,
            )

            raw_clean_e = generator(mu_e, labels_e)
            raw_clean_d = generator(mu_d, labels_d)

            # Transmitter fixed global scaling.
            ztx_e = mu_e * c
            ztx_d = mu_d * c

            seed_here = (
                int(cfg.seed)
                + 500000
                + int(snr_db * 1000)
                + batch_index
            )

            torch.manual_seed(seed_here)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_here)

            observation, zf_scaled_e, zf_scaled_d = (
                mimo_2x2_observation_no_internal_normalization(
                    ztx_e,
                    ztx_d,
                    snr_db,
                    cfg,
                )
            )

            # Receiver restores the original latent scale.
            zf_mu_e = zf_scaled_e / c
            zf_mu_d = zf_scaled_d / c

            zf_recon_e = generator(zf_mu_e, labels_e)
            zf_recon_d = generator(zf_mu_d, labels_d)

            clean_e_psnr.extend(
                batch_psnr(raw_clean_e, image_e).tolist()
            )

            clean_d_psnr.extend(
                batch_psnr(raw_clean_d, image_d).tolist()
            )

            zf_e_psnr.extend(
                batch_psnr(zf_recon_e, image_e).tolist()
            )

            zf_d_psnr.extend(
                batch_psnr(zf_recon_d, image_d).tolist()
            )

            zf_e_ssim.extend(
                batch_ssim(zf_recon_e, image_e).tolist()
            )

            zf_d_ssim.extend(
                batch_ssim(zf_recon_d, image_d).tolist()
            )

            if snr_db not in qualitative_saved:
                save_comparison_grid(
                    DIAGNOSTIC_ROOT
                    / f"02C_eurosat_global_zf_snr{snr_db:g}.png",
                    image_e,
                    [
                        ("Raw clean", raw_clean_e),
                        (f"Global-scale ZF @ {snr_db:g} dB", zf_recon_e),
                    ],
                    QUALITATIVE_COUNT,
                )

                save_comparison_grid(
                    DIAGNOSTIC_ROOT
                    / f"02C_deepglobe_global_zf_snr{snr_db:g}.png",
                    image_d,
                    [
                        ("Raw clean", raw_clean_d),
                        (f"Global-scale ZF @ {snr_db:g} dB", zf_recon_d),
                    ],
                    QUALITATIVE_COUNT,
                )

                qualitative_saved.add(snr_db)

        row = {
            "snr_db": float(snr_db),
            "global_scale": c,

            "eurosat_raw_clean_psnr": float(np.mean(clean_e_psnr)),
            "eurosat_global_zf_psnr": float(np.mean(zf_e_psnr)),
            "eurosat_global_zf_ssim": float(np.mean(zf_e_ssim)),

            "deepglobe_raw_clean_psnr": float(np.mean(clean_d_psnr)),
            "deepglobe_global_zf_psnr": float(np.mean(zf_d_psnr)),
            "deepglobe_global_zf_ssim": float(np.mean(zf_d_ssim)),
        }

        rows.append(row)

        print(
            f"SNR={snr_db:>4g} dB | "
            f"E raw={row['eurosat_raw_clean_psnr']:.3f}, "
            f"E global-ZF={row['eurosat_global_zf_psnr']:.3f} | "
            f"D raw={row['deepglobe_raw_clean_psnr']:.3f}, "
            f"D global-ZF={row['deepglobe_global_zf_psnr']:.3f}"
        )

    write_rows(
        DIAGNOSTIC_ROOT / "02C_global_scale_mimo_zf_metrics.csv",
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
            row["eurosat_global_zf_psnr"]
            for row in rows
        ],
        marker="o",
        label="EuroSAT global-scale ZF",
    )

    ax.plot(
        x,
        [
            row["deepglobe_global_zf_psnr"]
            for row in rows
        ],
        marker="s",
        label="DeepGlobe global-scale ZF",
    )

    ax.axhline(
        rows[0]["eurosat_raw_clean_psnr"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="EuroSAT raw-mu ceiling",
    )

    ax.axhline(
        rows[0]["deepglobe_raw_clean_psnr"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="DeepGlobe raw-mu ceiling",
    )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()

    fig.savefig(
        DIAGNOSTIC_ROOT
        / "02C_global_scale_mimo_zf_psnr_vs_snr.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # High-SNR gap is the most important validation number.
    high = rows[-1]

    high["eurosat_high_snr_gap_to_raw"] = (
        high["eurosat_raw_clean_psnr"]
        - high["eurosat_global_zf_psnr"]
    )

    high["deepglobe_high_snr_gap_to_raw"] = (
        high["deepglobe_raw_clean_psnr"]
        - high["deepglobe_global_zf_psnr"]
    )

    print()
    print(
        "30 dB gap to raw-mu ceiling | "
        f"EuroSAT={high['eurosat_high_snr_gap_to_raw']:.3f} dB | "
        f"DeepGlobe={high['deepglobe_high_snr_gap_to_raw']:.3f} dB"
    )

    return rows



# ============================================================
# Check 3 — MIMO / ZF sanity
# ============================================================

@torch.no_grad()
def check_mimo_zf(
    cfg,
    device: torch.device,
    loader_e,
    loader_d,
    encoder,
    generator,
):
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 3 — 2x2 MIMO + ZF SANITY\n"
        + "=" * 90
    )

    rows = []

    qualitative_saved = set()

    for snr_db in CHECK_SNRS:
        psnr_e_all = []
        psnr_d_all = []
        ssim_e_all = []
        ssim_d_all = []

        clean_psnr_e_all = []
        clean_psnr_d_all = []

        for batch_index, (
            batch_e,
            batch_d,
        ) in enumerate(
            zip(loader_e, loader_d)
        ):
            image_e, _ = batch_e
            image_d, _ = batch_d

            image_e = image_e.to(device)
            image_d = image_d.to(device)

            z_e, _ = encoder(image_e)
            z_d, _ = encoder(image_d)

            # Exactly match the original DDPM training path.
            z_e = M.normalize_latent_power(
                z_e
            )
            z_d = M.normalize_latent_power(
                z_d
            )

            labels_e = torch.full(
                (z_e.size(0),),
                M.EUROSAT_LABEL,
                device=device,
                dtype=torch.long,
            )

            labels_d = torch.full(
                (z_d.size(0),),
                M.DEEPGLOBE_LABEL,
                device=device,
                dtype=torch.long,
            )

            # Reproducible channel noise for this SNR/batch.
            seed_here = (
                int(cfg.seed)
                + int(snr_db * 1000)
                + batch_index
            )

            torch.manual_seed(
                seed_here
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    seed_here
                )

            observation, zf_e, zf_d = (
                M.mimo_2x2_observation(
                    z_e,
                    z_d,
                    snr_db,
                    cfg,
                )
            )

            clean_e = generator(
                z_e,
                labels_e,
            )

            clean_d = generator(
                z_d,
                labels_d,
            )

            zf_recon_e = generator(
                zf_e,
                labels_e,
            )

            zf_recon_d = generator(
                zf_d,
                labels_d,
            )

            clean_psnr_e_all.extend(
                batch_psnr(
                    clean_e,
                    image_e,
                ).tolist()
            )

            clean_psnr_d_all.extend(
                batch_psnr(
                    clean_d,
                    image_d,
                ).tolist()
            )

            psnr_e_all.extend(
                batch_psnr(
                    zf_recon_e,
                    image_e,
                ).tolist()
            )

            psnr_d_all.extend(
                batch_psnr(
                    zf_recon_d,
                    image_d,
                ).tolist()
            )

            ssim_e_all.extend(
                batch_ssim(
                    zf_recon_e,
                    image_e,
                ).tolist()
            )

            ssim_d_all.extend(
                batch_ssim(
                    zf_recon_d,
                    image_d,
                ).tolist()
            )

            if snr_db not in qualitative_saved:
                save_comparison_grid(
                    DIAGNOSTIC_ROOT
                    / f"03_eurosat_zf_snr{snr_db:g}.png",
                    image_e,
                    [
                        ("Normalized clean", clean_e),
                        (f"ZF @ {snr_db:g} dB", zf_recon_e),
                    ],
                    QUALITATIVE_COUNT,
                )

                save_comparison_grid(
                    DIAGNOSTIC_ROOT
                    / f"03_deepglobe_zf_snr{snr_db:g}.png",
                    image_d,
                    [
                        ("Normalized clean", clean_d),
                        (f"ZF @ {snr_db:g} dB", zf_recon_d),
                    ],
                    QUALITATIVE_COUNT,
                )

                qualitative_saved.add(
                    snr_db
                )

        row = {
            "snr_db":
                float(snr_db),

            "eurosat_clean_psnr":
                float(np.mean(clean_psnr_e_all)),

            "eurosat_zf_psnr":
                float(np.mean(psnr_e_all)),

            "eurosat_zf_ssim":
                float(np.mean(ssim_e_all)),

            "deepglobe_clean_psnr":
                float(np.mean(clean_psnr_d_all)),

            "deepglobe_zf_psnr":
                float(np.mean(psnr_d_all)),

            "deepglobe_zf_ssim":
                float(np.mean(ssim_d_all)),
        }

        rows.append(row)

        print(
            f"SNR={snr_db:>4g} dB | "
            f"E clean={row['eurosat_clean_psnr']:.3f}, "
            f"E ZF={row['eurosat_zf_psnr']:.3f} | "
            f"D clean={row['deepglobe_clean_psnr']:.3f}, "
            f"D ZF={row['deepglobe_zf_psnr']:.3f}"
        )

    write_rows(
        DIAGNOSTIC_ROOT
        / "03_mimo_zf_metrics.csv",
        rows,
    )

    # Plot PSNR sanity curve.
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

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        DIAGNOSTIC_ROOT
        / "03_mimo_zf_psnr_vs_snr.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return rows


# ============================================================
# Check 4 — DDPM interface / diffusion math sanity
# ============================================================

@torch.no_grad()
def check_ddpm_interface(
    cfg,
    device: torch.device,
    loader_e,
    loader_d,
    encoder,
):
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 4 — DDPM INTERFACE / DIFFUSION SANITY\n"
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

    z_e, _ = encoder(batch_e)
    z_d, _ = encoder(batch_d)

    z_e = M.normalize_latent_power(
        z_e
    )

    z_d = M.normalize_latent_power(
        z_d
    )

    observation, _, _ = (
        M.mimo_2x2_observation(
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

    assert z_e.shape == (
        batch,
        cfg.z_dim,
    )

    assert observation.shape == (
        batch,
        2 * cfg.z_dim,
    )

    assert z_t.shape == z_e.shape
    assert pred_eps.shape == z_e.shape

    if not torch.isfinite(
        pred_eps
    ).all():
        raise RuntimeError(
            "DDPM forward output contains NaN/Inf."
        )

    # Perfect-noise reconstruction test:
    # if the schedule formulas are mutually consistent,
    # predict_x0(z_t, t, true_noise) should reconstruct z_e.
    recovered_x0 = schedule.predict_x0(
        z_t,
        t,
        true_noise,
    )

    schedule_mse = F.mse_loss(
        recovered_x0,
        z_e,
    ).item()

    result = {
        "batch":
            batch,

        "z_dim":
            cfg.z_dim,

        "observation_dim":
            int(
                observation.shape[1]
            ),

        "ddpm_output_dim":
            int(
                pred_eps.shape[1]
            ),

        "schedule_perfect_noise_x0_mse":
            schedule_mse,

        "pred_eps_finite":
            bool(
                torch.isfinite(
                    pred_eps
                ).all().item()
            ),
    }

    print(
        "z target shape      :",
        tuple(z_e.shape),
    )

    print(
        "observation shape   :",
        tuple(observation.shape),
    )

    print(
        "DDPM output shape   :",
        tuple(pred_eps.shape),
    )

    print(
        "schedule perfect-eps x0 MSE:",
        f"{schedule_mse:.8e}",
    )

    if schedule_mse < 1e-8:
        print(
            "[PASS] q_sample() and predict_x0() "
            "are numerically consistent."
        )
    else:
        print(
            "[WARNING] Diffusion schedule inverse "
            "consistency error is larger than expected."
        )

    write_rows(
        DIAGNOSTIC_ROOT
        / "04_ddpm_interface_sanity.csv",
        [result],
    )

    return result


# ============================================================
# Check 5 — short DDPM smoke training
# ============================================================

def run_ddpm_smoke_test(
    cfg,
    device: torch.device,
):
    print(
        "\n"
        + "=" * 90
        + "\nCHECK 5 — SHORT DDPM SMOKE TRAINING\n"
        + "=" * 90
    )

    euro_loader, deep_loader, _, _ = (
        M.build_paired_loaders(
            cfg,
            SMOKE_BATCH_SIZE,
        )
    )

    euro_iter = M.infinite_loader(
        euro_loader
    )

    deep_iter = M.infinite_loader(
        deep_loader
    )

    encoder = M.load_encoder(
        cfg,
        device,
    )

    encoder.eval()

    M.set_requires_grad(
        encoder,
        False,
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

    rows = []

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
            z_e, _ = encoder(
                batch_e
            )

            z_d, _ = encoder(
                batch_d
            )

            z_e = M.normalize_latent_power(
                z_e
            )

            z_d = M.normalize_latent_power(
                z_d
            )

            snr_db = torch.empty(
                batch,
                device=device,
            ).uniform_(
                cfg.snr_min_db,
                cfg.snr_max_db,
            )

            observation, _, _ = (
                M.mimo_2x2_observation(
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
            "step": step,
            "total_loss":
                float(loss.item()),
            "eps_eurosat":
                float(loss_eps_e.item()),
            "eps_deepglobe":
                float(loss_eps_d.item()),
            "x0_eurosat":
                float(loss_x0_e.item()),
            "x0_deepglobe":
                float(loss_x0_d.item()),
            "grad_norm":
                float(grad_norm),
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
        DIAGNOSTIC_ROOT
        / "05_ddpm_smoke_history.csv",
        rows,
    )

    # Moving average only for diagnostic visualization.
    window = min(
        20,
        len(rows),
    )

    losses = np.asarray(
        [
            row["total_loss"]
            for row in rows
        ],
        dtype=np.float64,
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

        smoothed_x = np.arange(
            window,
            len(losses) + 1,
        )
    else:
        smoothed = losses
        smoothed_x = np.arange(
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
        smoothed_x,
        smoothed,
        linewidth=2,
        label=f"{window}-step moving average",
    )

    ax.set_xlabel(
        "Optimization step"
    )

    ax.set_ylabel(
        "DDPM loss"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        DIAGNOSTIC_ROOT
        / "05_ddpm_smoke_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Simple first-vs-last-window summary.
    compare_window = min(
        25,
        max(
            1,
            len(losses) // 4,
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
                (last_mean - first_mean)
                / max(
                    abs(first_mean),
                    1e-12,
                )
            ),
    }

    write_rows(
        DIAGNOSTIC_ROOT
        / "05_ddpm_smoke_summary.csv",
        [summary],
    )

    print()
    print(
        "First-window mean loss:",
        f"{first_mean:.6f}",
    )

    print(
        "Last-window mean loss :",
        f"{last_mean:.6f}",
    )

    if last_mean < first_mean:
        print(
            "[PASS] Short smoke-test loss decreased overall."
        )
    else:
        print(
            "[WARNING] Short smoke-test loss did not decrease overall. "
            "Inspect the CSV/curve before launching full DDPM training."
        )

    return summary


# ============================================================
# Final report
# ============================================================

def make_text_report(
    checkpoint_result: dict,
    normalization_result: dict,
    global_scale_result: dict,
    global_scale_mimo_rows: List[dict],
    mimo_rows: List[dict],
    ddpm_interface_result: dict,
    smoke_result: dict | None,
):
    lines = []

    lines.append(
        "PRE-DDPM DIAGNOSTIC SUMMARY"
    )

    lines.append(
        "=" * 80
    )

    lines.append("")
    lines.append(
        "1. CHECKPOINT CONSISTENCY"
    )

    lines.append(
        f"Joint best epoch: "
        f"{checkpoint_result['joint_best_epoch']}"
    )

    lines.append(
        f"Encoder best epoch: "
        f"{checkpoint_result['encoder_best_epoch']}"
    )

    lines.append(
        f"Generator best epoch: "
        f"{checkpoint_result['gan_best_epoch']}"
    )

    lines.append(
        "Encoder vs joint max abs diff: "
        f"{checkpoint_result['encoder_vs_joint_max_abs_diff']}"
    )

    lines.append(
        "Generator vs joint max abs diff: "
        f"{checkpoint_result['generator_vs_joint_max_abs_diff']}"
    )

    lines.append("")
    lines.append(
        "2. NORMALIZATION EFFECT"
    )

    lines.append(
        "EuroSAT raw->normalized PSNR drop: "
        f"{normalization_result['eurosat_psnr_drop_due_to_normalization']:.3f} dB"
    )

    lines.append(
        "DeepGlobe raw->normalized PSNR drop: "
        f"{normalization_result['deepglobe_psnr_drop_due_to_normalization']:.3f} dB"
    )

    lines.append("")
    lines.append(
        "2B. FIXED GLOBAL SCALE ROUNDTRIP"
    )

    lines.append(
        "Shared global power: "
        f"{global_scale_result['global_power']:.8f}"
    )

    lines.append(
        "Shared global scale c: "
        f"{global_scale_result['global_scale']:.8f}"
    )

    lines.append(
        "EuroSAT global roundtrip PSNR drop: "
        f"{global_scale_result['eurosat_psnr_drop_after_global_roundtrip']:.6f} dB"
    )

    lines.append(
        "DeepGlobe global roundtrip PSNR drop: "
        f"{global_scale_result['deepglobe_psnr_drop_after_global_roundtrip']:.6f} dB"
    )

    lines.append("")
    lines.append(
        "2C. FIXED GLOBAL SCALE MIMO / ZF"
    )

    for row in global_scale_mimo_rows:
        lines.append(
            f"SNR {row['snr_db']:g} dB: "
            f"EuroSAT global-ZF={row['eurosat_global_zf_psnr']:.3f} dB, "
            f"DeepGlobe global-ZF={row['deepglobe_global_zf_psnr']:.3f} dB"
        )

    lines.append("")
    lines.append(
        "3. ORIGINAL PER-SAMPLE NORMALIZED MIMO / ZF"
    )

    for row in mimo_rows:
        lines.append(
            f"SNR {row['snr_db']:g} dB: "
            f"EuroSAT ZF={row['eurosat_zf_psnr']:.3f} dB, "
            f"DeepGlobe ZF={row['deepglobe_zf_psnr']:.3f} dB"
        )

    lines.append("")
    lines.append(
        "4. DDPM INTERFACE"
    )

    lines.append(
        "Perfect-noise schedule x0 MSE: "
        f"{ddpm_interface_result['schedule_perfect_noise_x0_mse']:.8e}"
    )

    lines.append(
        "Predicted epsilon finite: "
        f"{ddpm_interface_result['pred_eps_finite']}"
    )

    if smoke_result is not None:
        lines.append("")
        lines.append(
            "5. DDPM SMOKE TEST"
        )

        lines.append(
            "First-window mean loss: "
            f"{smoke_result['first_window_mean_loss']:.6f}"
        )

        lines.append(
            "Last-window mean loss: "
            f"{smoke_result['last_window_mean_loss']:.6f}"
        )

    lines.append("")
    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "Do not launch the full DDPM training only because this script "
        "finishes without an exception. Inspect the normalization and "
        "ZF images/metrics. The most important warning sign is a large "
        "reconstruction drop after latent power normalization."
    )

    (
        DIAGNOSTIC_ROOT
        / "PRE_DDPM_DIAGNOSTIC_REPORT.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(
        DIAGNOSTIC_ROOT
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
        "Diagnostic output folder:",
        DIAGNOSTIC_ROOT.resolve(),
    )

    print(
        "Original training outputs are NOT overwritten."
    )

    # --------------------------------------------------------
    # Check 1
    # --------------------------------------------------------
    checkpoint_result = (
        check_checkpoints(
            cfg,
            device,
        )
    )

    # Fixed images used for all reconstruction/channel checks.
    (
        loader_e,
        loader_d,
        count,
    ) = build_fixed_paired_loader(
        cfg
    )

    print(
        "\nFixed diagnostic pair count:",
        count,
    )

    # --------------------------------------------------------
    # Check 2
    # --------------------------------------------------------
    (
        encoder,
        generator,
        normalization_result,
    ) = check_latent_normalization(
        cfg,
        device,
        loader_e,
        loader_d,
    )

    # --------------------------------------------------------
    # Check 2B — fixed global scaling roundtrip
    # --------------------------------------------------------
    global_scale_result = check_global_scale_roundtrip(
        cfg,
        device,
        loader_e,
        loader_d,
        encoder,
        generator,
    )

    # --------------------------------------------------------
    # Check 2C — fixed global scaling through MIMO + ZF
    # --------------------------------------------------------
    global_scale_mimo_rows = check_global_scale_mimo_zf(
        cfg,
        device,
        loader_e,
        loader_d,
        encoder,
        generator,
        global_scale_result,
    )

    # --------------------------------------------------------
    # Check 3
    # --------------------------------------------------------
    mimo_rows = check_mimo_zf(
        cfg,
        device,
        loader_e,
        loader_d,
        encoder,
        generator,
    )

    # --------------------------------------------------------
    # Check 4
    # --------------------------------------------------------
    ddpm_interface_result = (
        check_ddpm_interface(
            cfg,
            device,
            loader_e,
            loader_d,
            encoder,
        )
    )

    # --------------------------------------------------------
    # Check 5
    # --------------------------------------------------------
    smoke_result = None

    if RUN_DDPM_SMOKE_TEST:
        smoke_result = (
            run_ddpm_smoke_test(
                cfg,
                device,
            )
        )

    # --------------------------------------------------------
    # Final summary text
    # --------------------------------------------------------
    make_text_report(
        checkpoint_result,
        normalization_result,
        global_scale_result,
        global_scale_mimo_rows,
        mimo_rows,
        ddpm_interface_result,
        smoke_result,
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PRE-DDPM DIAGNOSTIC COMPLETED"
    )

    print(
        "=" * 90
    )

    print(
        "Check the folder:"
    )

    print(
        DIAGNOSTIC_ROOT.resolve()
    )

    print()
    print(
        "Most important files:"
    )

    print(
        "  PRE_DDPM_DIAGNOSTIC_REPORT.txt"
    )

    print(
        "  01_checkpoint_consistency.csv"
    )

    print(
        "  02_normalization_metrics.csv"
    )

    print(
        "  02_eurosat_raw_vs_normalized.png"
    )

    print(
        "  02_deepglobe_raw_vs_normalized.png"
    )

    print(
        "  02B_global_scale_roundtrip_metrics.csv"
    )

    print(
        "  02C_global_scale_mimo_zf_metrics.csv"
    )

    print(
        "  02C_global_scale_mimo_zf_psnr_vs_snr.png"
    )

    print(
        "  03_mimo_zf_metrics.csv"
    )

    print(
        "  03_mimo_zf_psnr_vs_snr.png"
    )

    print(
        "  04_ddpm_interface_sanity.csv"
    )

    if RUN_DDPM_SMOKE_TEST:
        print(
            "  05_ddpm_smoke_history.csv"
        )

        print(
            "  05_ddpm_smoke_loss.png"
        )


if __name__ == "__main__":
    main()
