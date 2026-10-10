"""
DeepGlobe single-source Rayleigh + AWGN evaluation for four separator systems.

Four evaluated systems
----------------------
1. VAE-DDPM-VAE
2. VAE-MLP-VAE
3. VAE-DDPM-GAN
4. VAE-MLP-GAN

Pipeline
--------
DeepGlobe image
    -> corresponding encoder mean mu
    -> per-sample latent power normalization
    -> identical Rayleigh fading + identical AWGN realization
    -> perfect-CSI equalization (optional)
    -> corresponding separator
    -> corresponding decoder/generator
    -> image and latent metrics

Outputs
-------
- original_8x8.png
- VAE_DDPM_VAE_8x8.png
- VAE_MLP_VAE_8x8.png
- VAE_DDPM_GAN_8x8.png
- VAE_MLP_GAN_8x8.png
- comparison_5rows_x_8cols.png
- per_image_metrics.csv
- summary_metrics.csv

Notes
-----
- DeepGlobe uses label 1.
- DDPM and MLP separators return (EuroSAT latent, DeepGlobe latent).
- This is a single-source ablation. The separators were originally designed
  for recovering sources from a mixed latent signal.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as skimage_ssim
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from model_shared_vae import load_encoder_decoder
from model_shared_vaegan import load_encoder_generator
from model_seperator import (
    DiffusionSchedule,
    load_vae_ddpm_gan,
    load_vae_ddpm_vae,
    load_vae_mlp_gan,
    load_vae_mlp_vae,
    sample_ddpm,
)


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    deepglobe_root: str = (
        r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    )

    # Image models
    vae_checkpoint: str = (
        "./save_model/shared_vae_satellite_latest.pth"
    )
    vaegan_checkpoint: str = (
        "./save_model/vae_gan_from_scratch_best.pth"
    )

    # Separator checkpoints
    vae_ddpm_checkpoint: str = (
        "./save_model/DDPM_vae_latent_separator_latest.pth"
    )
    vae_mlp_checkpoint: str = (
        "./save_model/MLP_vae_latent_separator_best.pth"
    )
    vaegan_ddpm_checkpoint: str = (
        "./save_model/"
        "DDPM_vaegan_scratch_latent_separator_best.pth"
    )
    vaegan_mlp_checkpoint: str = (
        "./save_model/"
        "MLP_vaegan_scratch_latent_separator_best.pth"
    )

    output_dir: str = (
        "./save_data/deepglobe_four_separator_eval"
    )

    z_dim: int = 256
    num_classes: int = 2
    label_dim: int = 64
    diffusion_steps: int = 400

    num_images: int = 64
    batch_size: int = 64
    num_workers: int = 0

    snr_db: float = 10.0
    equalizer: str = "perfect_csi"  # perfect_csi | none
    min_channel_gain: float = 0.10

    sampling_method: str = "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    clip_x0: Optional[float] = None

    seed: int = 42
    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Try to calculate LPIPS when the package is installed.
    use_lpips: bool = True


# ============================================================
# Dataset
# ============================================================

class DeepGlobeDataset(Dataset):
    EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff", ".webp",
    }

    def __init__(self, root: str, transform=None) -> None:
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(
                f"DeepGlobe directory does not exist:\n{self.root}"
            )

        self.files = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.EXTENSIONS
        )

        if not self.files:
            raise RuntimeError(
                f"No DeepGlobe images found under:\n{self.root}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Tensor, str]:
        path = self.files[index]

        with Image.open(path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, str(path)


# ============================================================
# Model bundle
# ============================================================

@dataclass
class MethodBundle:
    name: str
    encoder: nn.Module
    decoder: nn.Module
    mode: str

    ddpm: Optional[nn.Module] = None
    schedule: Optional[DiffusionSchedule] = None
    mlp: Optional[nn.Module] = None


# ============================================================
# Reproducibility and validation
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_paths(config: Config) -> None:
    paths = {
        "DeepGlobe root": config.deepglobe_root,
        "VAE checkpoint": config.vae_checkpoint,
        "VAE-GAN checkpoint": config.vaegan_checkpoint,
        "VAE DDPM checkpoint": config.vae_ddpm_checkpoint,
        "VAE MLP checkpoint": config.vae_mlp_checkpoint,
        "VAE-GAN DDPM checkpoint":
            config.vaegan_ddpm_checkpoint,
        "VAE-GAN MLP checkpoint":
            config.vaegan_mlp_checkpoint,
    }

    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if not Path(path).exists()
    ]

    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            "The following required paths do not exist:\n"
            f"{joined}\n\n"
            "Run this script from the project root or correct Config."
        )


# ============================================================
# Loading all four systems
# ============================================================

def load_all_methods(
    config: Config,
    device: torch.device,
) -> List[MethodBundle]:

    # The red and green systems use the same VAE checkpoint.
    vae_encoder, vae_decoder = load_encoder_decoder(
        checkpoint=config.vae_checkpoint,
        device=device,
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
        strict=True,
        freeze=True,
    )

    # The purple and blue systems use the same VAE-GAN checkpoint.
    vaegan_encoder, vaegan_generator = load_encoder_generator(
        checkpoint=config.vaegan_checkpoint,
        device=device,
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
        strict=True,
        freeze=True,
    )

    vae_ddpm, vae_schedule = load_vae_ddpm_vae(
        checkpoint=config.vae_ddpm_checkpoint,
        device=device,
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
        diffusion_steps=config.diffusion_steps,
        strict=True,
        freeze=True,
    )

    vae_mlp = load_vae_mlp_vae(
        checkpoint=config.vae_mlp_checkpoint,
        device=device,
        z_dim=config.z_dim,
        strict=True,
        freeze=True,
    )

    vaegan_ddpm, vaegan_schedule = load_vae_ddpm_gan(
        checkpoint=config.vaegan_ddpm_checkpoint,
        device=device,
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
        diffusion_steps=config.diffusion_steps,
        strict=True,
        freeze=True,
    )

    vaegan_mlp = load_vae_mlp_gan(
        checkpoint=config.vaegan_mlp_checkpoint,
        device=device,
        z_dim=config.z_dim,
        strict=True,
        freeze=True,
    )

    return [
        MethodBundle(
            name="VAE-DDPM-VAE",
            encoder=vae_encoder,
            decoder=vae_decoder,
            mode="ddpm",
            ddpm=vae_ddpm,
            schedule=vae_schedule,
        ),
        MethodBundle(
            name="VAE-MLP-VAE",
            encoder=vae_encoder,
            decoder=vae_decoder,
            mode="mlp",
            mlp=vae_mlp,
        ),
        MethodBundle(
            name="VAE-DDPM-GAN",
            encoder=vaegan_encoder,
            decoder=vaegan_generator,
            mode="ddpm",
            ddpm=vaegan_ddpm,
            schedule=vaegan_schedule,
        ),
        MethodBundle(
            name="VAE-MLP-GAN",
            encoder=vaegan_encoder,
            decoder=vaegan_generator,
            mode="mlp",
            mlp=vaegan_mlp,
        ),
    ]


# ============================================================
# Channel
# ============================================================

def power_normalize(
    latent: Tensor,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """
    Normalize every sample to unit average latent power.

    Returns
    -------
    normalized: [B, D]
    rms:        [B, 1]
    """
    rms = torch.sqrt(
        latent.pow(2).mean(dim=1, keepdim=True) + eps
    )
    return latent / rms, rms


def generate_channel_realization(
    batch_size: int,
    z_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    min_channel_gain: float,
) -> Tuple[Tensor, Tensor]:
    """
    Generate one shared channel realization for every method.

    h is a Rayleigh envelope with E[h^2] approximately 1.
    noise_unit is standard Gaussian and is reused by every method.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    real = torch.randn(
        batch_size,
        1,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    imag = torch.randn(
        batch_size,
        1,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    h = torch.sqrt(
        (real.pow(2) + imag.pow(2)) / 2.0
    ).clamp_min(min_channel_gain)

    noise_unit = torch.randn(
        batch_size,
        z_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    return h, noise_unit


def apply_rayleigh_awgn(
    latent_normalized: Tensor,
    h: Tensor,
    noise_unit: Tensor,
    snr_db: float,
    equalizer: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Channel
        y = h*z + n

    AWGN power is computed independently for every sample.

    perfect_csi
        z_rx = y/h

    none
        z_rx = y
    """
    faded = h * latent_normalized

    if math.isinf(float(snr_db)):
        noise_std = torch.zeros(
            faded.size(0),
            1,
            device=faded.device,
            dtype=faded.dtype,
        )
        received = faded
    else:
        signal_power = faded.pow(2).mean(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)

        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        noise_std = torch.sqrt(signal_power / snr_linear)
        received = faded + noise_std * noise_unit

    equalizer = equalizer.lower()

    if equalizer == "perfect_csi":
        channel_output = received / h
    elif equalizer == "none":
        channel_output = received
    else:
        raise ValueError(
            "equalizer must be 'perfect_csi' or 'none'"
        )

    return channel_output, received, noise_std


# ============================================================
# Separator inference
# ============================================================

@torch.no_grad()
def separate_deepglobe(
    bundle: MethodBundle,
    received_latent: Tensor,
    config: Config,
    method_index: int,
) -> Tensor:
    """
    Both DDPM and MLP return:
        EuroSAT latent, DeepGlobe latent

    Only the DeepGlobe latent is used.
    """
    if bundle.mode == "mlp":
        if bundle.mlp is None:
            raise RuntimeError(
                f"{bundle.name} has no MLP separator."
            )

        _, predicted_deepglobe = bundle.mlp(received_latent)
        return predicted_deepglobe

    if bundle.mode == "ddpm":
        if bundle.ddpm is None or bundle.schedule is None:
            raise RuntimeError(
                f"{bundle.name} has no DDPM/schedule."
            )

        labels_deepglobe = torch.ones(
            received_latent.size(0),
            dtype=torch.long,
            device=received_latent.device,
        )

        predicted_deepglobe = sample_ddpm(
            model=bundle.ddpm,
            schedule=bundle.schedule,
            cond=received_latent,
            labels=labels_deepglobe,
            seed=config.seed + 1000 + method_index,
            sampling_method=config.sampling_method,
            ddim_steps=config.ddim_steps,
            ddim_eta=config.ddim_eta,
            clip_x0=config.clip_x0,
        )

        return predicted_deepglobe

    raise ValueError(
        f"Unknown separator mode: {bundle.mode}"
    )


# ============================================================
# Metrics
# ============================================================

def to_01(images: Tensor) -> Tensor:
    return (
        (images.detach().float() + 1.0) / 2.0
    ).clamp(0.0, 1.0)


def image_psnr(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    prediction = to_01(prediction)
    target = to_01(target)

    mse = (
        prediction - target
    ).pow(2).flatten(1).mean(dim=1)

    return 10.0 * torch.log10(
        1.0 / mse.clamp_min(1e-12)
    )


def image_ssim(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    prediction_np = (
        to_01(prediction)
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

    for pred, real in zip(prediction_np, target_np):
        values.append(
            skimage_ssim(
                real,
                pred,
                data_range=1.0,
                channel_axis=-1,
            )
        )

    return torch.tensor(values, dtype=torch.float32)


def image_mae(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    prediction = to_01(prediction)
    target = to_01(target)

    return (
        prediction - target
    ).abs().flatten(1).mean(dim=1)


def image_mse(
    prediction: Tensor,
    target: Tensor,
) -> Tensor:
    prediction = to_01(prediction)
    target = to_01(target)

    return (
        prediction - target
    ).pow(2).flatten(1).mean(dim=1)


def compute_latent_metrics(
    prediction: Tensor,
    target: Tensor,
) -> Dict[str, Tensor]:
    difference = prediction - target

    mse = difference.pow(2).mean(dim=1)
    mae = difference.abs().mean(dim=1)

    nmse = (
        difference.pow(2).sum(dim=1)
        / target.pow(2).sum(dim=1).clamp_min(1e-12)
    )

    cosine = F.cosine_similarity(
        prediction,
        target,
        dim=1,
    )

    return {
        "latent_mse": mse,
        "latent_mae": mae,
        "latent_nmse": nmse,
        "latent_cosine": cosine,
    }


def calculate_lpips(
    model,
    prediction: Tensor,
    target: Tensor,
) -> Optional[Tensor]:
    if model is None:
        return None

    with torch.no_grad():
        values = model(target, prediction)

    return values.flatten().detach().cpu()


def summarize_tensor(values: Tensor) -> Tuple[float, float]:
    values = values.detach().float().cpu()

    return (
        values.mean().item(),
        values.std(unbiased=False).item(),
    )


# ============================================================
# Evaluate one method
# ============================================================

@torch.no_grad()
def evaluate_method(
    bundle: MethodBundle,
    images_cpu: Tensor,
    h: Tensor,
    noise_unit: Tensor,
    config: Config,
    method_index: int,
    lpips_model,
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, float]]:
    device = torch.device(config.device)
    images = images_cpu.to(device, non_blocking=True)

    labels = torch.ones(
        images.size(0),
        dtype=torch.long,
        device=device,
    )

    mu, logvar = bundle.encoder(images)
    clean_latent = mu

    normalized_latent, latent_rms = power_normalize(
        clean_latent
    )

    equalized_latent, raw_received, noise_std = (
        apply_rayleigh_awgn(
            latent_normalized=normalized_latent,
            h=h,
            noise_unit=noise_unit,
            snr_db=config.snr_db,
            equalizer=config.equalizer,
        )
    )

    predicted_normalized = separate_deepglobe(
        bundle=bundle,
        received_latent=equalized_latent,
        config=config,
        method_index=method_index,
    )

    predicted_latent = predicted_normalized * latent_rms
    reconstructed = bundle.decoder(predicted_latent, labels)

    # Oracle reconstruction isolates encoder/decoder quality from separator quality.
    oracle_reconstruction = bundle.decoder(clean_latent, labels)

    metrics: Dict[str, Tensor] = {
        "psnr_real": image_psnr(reconstructed, images),
        "ssim_real": image_ssim(reconstructed, images),
        "image_mae_real": image_mae(reconstructed, images),
        "image_mse_real": image_mse(reconstructed, images),

        "psnr_oracle": image_psnr(
            reconstructed,
            oracle_reconstruction,
        ),
        "ssim_oracle": image_ssim(
            reconstructed,
            oracle_reconstruction,
        ),

        "oracle_psnr_real": image_psnr(
            oracle_reconstruction,
            images,
        ),
        "oracle_ssim_real": image_ssim(
            oracle_reconstruction,
            images,
        ),
    }

    metrics.update(
        compute_latent_metrics(
            prediction=predicted_latent,
            target=clean_latent,
        )
    )

    lpips_real = calculate_lpips(
        lpips_model,
        reconstructed,
        images,
    )

    if lpips_real is not None:
        metrics["lpips_real"] = lpips_real

    diagnostics = {
        "channel_gain_mean": h.mean().item(),
        "channel_gain_min": h.min().item(),
        "channel_gain_max": h.max().item(),
        "noise_std_mean": noise_std.mean().item(),
        "received_power_mean":
            raw_received.pow(2).mean().item(),
        "clean_latent_power_mean":
            clean_latent.pow(2).mean().item(),
        "predicted_latent_power_mean":
            predicted_latent.pow(2).mean().item(),
        "posterior_std_mean":
            torch.exp(0.5 * logvar).mean().item(),
    }

    print("\n" + "=" * 76)
    print(bundle.name)

    for metric_name, values in metrics.items():
        mean, std = summarize_tensor(values)
        print(
            f"{metric_name:>24s}: "
            f"{mean:.6f} ± {std:.6f}"
        )

    for name, value in diagnostics.items():
        print(f"{name:>24s}: {value:.6f}")

    return reconstructed, metrics, diagnostics


# ============================================================
# Save images
# ============================================================

def safe_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_")


def save_8x8_grid(
    images: Tensor,
    path: Path,
) -> None:
    images = to_01(images).cpu()

    grid = make_grid(
        images,
        nrow=8,
        padding=2,
    )

    save_image(grid, path)


def save_five_row_comparison(
    originals: Tensor,
    reconstructions: Dict[str, Tensor],
    path: Path,
) -> None:
    """
    For each block of 8 images:
        row 1: Original
        row 2: VAE-DDPM-VAE
        row 3: VAE-MLP-VAE
        row 4: VAE-DDPM-GAN
        row 5: VAE-MLP-GAN

    With 64 inputs, there are 8 repeated five-row blocks.
    """
    order = (
        "VAE-DDPM-VAE",
        "VAE-MLP-VAE",
        "VAE-DDPM-GAN",
        "VAE-MLP-GAN",
    )

    originals = to_01(originals).cpu()

    recon_cpu = {
        name: to_01(reconstructions[name]).cpu()
        for name in order
    }

    rows: List[Tensor] = []

    for start in range(0, originals.size(0), 8):
        end = start + 8

        rows.append(originals[start:end])

        for name in order:
            rows.append(recon_cpu[name][start:end])

    sheet = torch.cat(rows, dim=0)

    grid = make_grid(
        sheet,
        nrow=8,
        padding=2,
    )

    save_image(grid, path)


# ============================================================
# Save CSV files
# ============================================================

def save_per_image_csv(
    path: Path,
    file_paths: List[str],
    metrics_by_method: Dict[str, Dict[str, Tensor]],
) -> None:
    fieldnames = ["index", "file"]

    for method_name, metrics in metrics_by_method.items():
        method_key = safe_name(method_name)

        for metric_name in metrics:
            fieldnames.append(
                f"{method_key}_{metric_name}"
            )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for index, image_path in enumerate(file_paths):
            row = {
                "index": index,
                "file": image_path,
            }

            for method_name, metrics in metrics_by_method.items():
                method_key = safe_name(method_name)

                for metric_name, values in metrics.items():
                    row[
                        f"{method_key}_{metric_name}"
                    ] = float(
                        values[index].detach().cpu()
                    )

            writer.writerow(row)


def save_summary_csv(
    path: Path,
    metrics_by_method: Dict[str, Dict[str, Tensor]],
    diagnostics_by_method: Dict[str, Dict[str, float]],
    config: Config,
) -> None:
    rows: List[Dict[str, object]] = []

    for method_name, metrics in metrics_by_method.items():
        row: Dict[str, object] = {
            "method": method_name,
            "snr_db": config.snr_db,
            "channel": "Rayleigh + AWGN",
            "equalizer": config.equalizer,
            "num_images": config.num_images,
        }

        for metric_name, values in metrics.items():
            mean, std = summarize_tensor(values)
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_std"] = std

        row.update(diagnostics_by_method[method_name])
        rows.append(row)

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    config = Config()
    seed_everything(config.seed)

    device = torch.device(config.device)

    print(f"Device: {device}")
    print("Methods:")
    print("  1. VAE-DDPM-VAE")
    print("  2. VAE-MLP-VAE")
    print("  3. VAE-DDPM-GAN")
    print("  4. VAE-MLP-GAN")
    print(f"SNR: {config.snr_db} dB")
    print(f"Equalizer: {config.equalizer}")

    validate_paths(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    dataset = DeepGlobeDataset(
        root=config.deepglobe_root,
        transform=transform,
    )

    if len(dataset) < config.num_images:
        raise RuntimeError(
            f"Requested {config.num_images} images, "
            f"but only {len(dataset)} were found."
        )

    # Fixed random subset rather than always taking adjacent images.
    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    )[:config.num_images].tolist()

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    images, file_paths = next(iter(loader))

    if images.size(0) != config.num_images:
        raise RuntimeError(
            "This script expects all 64 images in one batch. "
            "Set batch_size equal to num_images."
        )

    print("\nLoading all four systems...")
    methods = load_all_methods(config, device)

    # One identical fading/noise realization is reused by all methods.
    h, noise_unit = generate_channel_realization(
        batch_size=images.size(0),
        z_dim=config.z_dim,
        device=device,
        dtype=torch.float32,
        seed=config.seed + 12345,
        min_channel_gain=config.min_channel_gain,
    )

    lpips_model = None

    if config.use_lpips:
        try:
            import lpips

            lpips_model = lpips.LPIPS(
                net="alex"
            ).to(device).eval()

            for parameter in lpips_model.parameters():
                parameter.requires_grad = False

            print("LPIPS enabled.")
        except ImportError:
            print(
                "LPIPS is not installed; LPIPS will be skipped.\n"
                "Install it with: pip install lpips"
            )

    reconstructions: Dict[str, Tensor] = {}
    metrics_by_method: Dict[str, Dict[str, Tensor]] = {}
    diagnostics_by_method: Dict[str, Dict[str, float]] = {}

    for method_index, bundle in enumerate(methods):
        reconstruction, metrics, diagnostics = evaluate_method(
            bundle=bundle,
            images_cpu=images,
            h=h,
            noise_unit=noise_unit,
            config=config,
            method_index=method_index,
            lpips_model=lpips_model,
        )

        reconstructions[bundle.name] = reconstruction
        metrics_by_method[bundle.name] = metrics
        diagnostics_by_method[bundle.name] = diagnostics

    save_8x8_grid(
        images,
        output_dir / "original_8x8.png",
    )

    for method_name, reconstruction in reconstructions.items():
        save_8x8_grid(
            reconstruction,
            output_dir / (
                f"{safe_name(method_name)}_"
                f"SNR{config.snr_db:g}_8x8.png"
            ),
        )

    save_five_row_comparison(
        originals=images,
        reconstructions=reconstructions,
        path=output_dir / (
            f"comparison_5rows_x_8cols_"
            f"SNR{config.snr_db:g}.png"
        ),
    )

    save_per_image_csv(
        path=output_dir / "per_image_metrics.csv",
        file_paths=list(file_paths),
        metrics_by_method=metrics_by_method,
    )

    save_summary_csv(
        path=output_dir / "summary_metrics.csv",
        metrics_by_method=metrics_by_method,
        diagnostics_by_method=diagnostics_by_method,
        config=config,
    )

    print("\n" + "=" * 76)
    print("Evaluation completed.")
    print(f"Results saved under:\n{output_dir.resolve()}")
    print("\nOutput files:")
    print("  original_8x8.png")
    print(
        f"  VAE_DDPM_VAE_SNR{config.snr_db:g}_8x8.png"
    )
    print(
        f"  VAE_MLP_VAE_SNR{config.snr_db:g}_8x8.png"
    )
    print(
        f"  VAE_DDPM_GAN_SNR{config.snr_db:g}_8x8.png"
    )
    print(
        f"  VAE_MLP_GAN_SNR{config.snr_db:g}_8x8.png"
    )
    print(
        f"  comparison_5rows_x_8cols_"
        f"SNR{config.snr_db:g}.png"
    )
    print("  per_image_metrics.csv")
    print("  summary_metrics.csv")


if __name__ == "__main__":
    main()
