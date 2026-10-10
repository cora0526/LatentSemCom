from __future__ import annotations

"""
DeepGlobe heterogeneous latent-mixture evaluation.

Source 1:
    image_1 -> VAE encoder -> z1

Source 2:
    image_2 -> VAE-GAN encoder -> z2

Channel:
    received = z1_norm + z2_norm + noise

Recovery:
    VAE DDPM / VAE MLP       recover z1 -> VAE decoder
    VAE-GAN DDPM / VAE-GAN MLP recover z2 -> GAN generator

Important:
    This is a heterogeneous cross-latent-space experiment. It is only a
    matched test if the separators were trained with the same heterogeneous
    mixture definition. Otherwise, treat it as an out-of-distribution ablation.
"""

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as skimage_ssim
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
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
    # Use a true unseen split here when testing generalization.
    deepglobe_root: str = (
        r"D:/Python/SemCom/saved_data/deepglobe_64/test"
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
        "./save_model/DDPM_vaegan_scratch_latent_separator_best.pth"
    )
    vaegan_mlp_checkpoint: str = (
        "./save_model/MLP_vaegan_scratch_latent_separator_best.pth"
    )

    output_dir: str = (
        "./save_data/deepglobe_heterogeneous_mixture_no_h_snr_sweep"
    )

    z_dim: int = 256
    num_classes: int = 2
    label_dim: int = 64
    diffusion_steps: int = 400

    num_pairs: int = 64
    batch_size: int = 16
    num_workers: int = 0

    snr_values: Tuple[float, ...] = (
        0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0
    )
    qualitative_snrs: Tuple[float, ...] = (
        0.0, 10.0, 20.0, 30.0
    )
    qualitative_count: int = 8

    # Noise power is computed from the total mixture signal power.
    # received = z1_norm + z2_norm + noise

    sampling_method: str = "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    clip_x0: Optional[float] = None

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_lpips: bool = True
    figure_dpi: int = 300
    save_pdf: bool = True
    show_std_band: bool = True


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


class PairedDeepGlobeDataset(Dataset):
    """Return two different fixed DeepGlobe images for each pair."""

    def __init__(
        self,
        base_dataset: DeepGlobeDataset,
        num_pairs: int,
        seed: int,
    ) -> None:
        if len(base_dataset) < 2:
            raise RuntimeError("At least two images are required.")
        if num_pairs > len(base_dataset):
            raise RuntimeError(
                f"Requested {num_pairs} pairs, but only "
                f"{len(base_dataset)} images are available."
            )

        self.base_dataset = base_dataset

        generator = torch.Generator().manual_seed(seed)
        first = torch.randperm(
            len(base_dataset), generator=generator
        )[:num_pairs]

        # A second independently permuted list; repair accidental identical pairs.
        second = torch.randperm(
            len(base_dataset), generator=generator
        )[:num_pairs]

        for i in range(num_pairs):
            if second[i].item() == first[i].item():
                swap_index = (i + 1) % num_pairs
                second[i], second[swap_index] = (
                    second[swap_index].clone(), second[i].clone()
                )

        self.first_indices = first.tolist()
        self.second_indices = second.tolist()

    def __len__(self) -> int:
        return len(self.first_indices)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Tensor, Tensor, str, str]:
        image_1, path_1 = self.base_dataset[self.first_indices[index]]
        image_2, path_2 = self.base_dataset[self.second_indices[index]]
        return image_1, image_2, path_1, path_2


# ============================================================
# Model containers
# ============================================================

@dataclass
class LoadedModels:
    vae_encoder: nn.Module
    vae_decoder: nn.Module
    vaegan_encoder: nn.Module
    vaegan_generator: nn.Module

    vae_ddpm: nn.Module
    vae_ddpm_schedule: DiffusionSchedule
    vae_mlp: nn.Module

    vaegan_ddpm: nn.Module
    vaegan_ddpm_schedule: DiffusionSchedule
    vaegan_mlp: nn.Module


METHOD_ORDER = (
    "VAE-DDPM-VAE",
    "VAE-MLP-VAE",
    "VAEGAN-DDPM-GAN",
    "VAEGAN-MLP-GAN",
)

METHOD_TARGET = {
    "VAE-DDPM-VAE": "z1",
    "VAE-MLP-VAE": "z1",
    "VAEGAN-DDPM-GAN": "z2",
    "VAEGAN-MLP-GAN": "z2",
}


# ============================================================
# General utilities
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
        "VAE-GAN DDPM checkpoint": config.vaegan_ddpm_checkpoint,
        "VAE-GAN MLP checkpoint": config.vaegan_mlp_checkpoint,
    }

    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if not Path(path).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "The following required paths do not exist:\n"
            + "\n".join(missing)
            + "\n\nRun this script from the project root "
              "or correct Config."
        )


def to_01(images: Tensor) -> Tensor:
    return ((images.detach().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def power_normalize(
    latent: Tensor,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    rms = torch.sqrt(
        latent.pow(2).mean(dim=1, keepdim=True) + eps
    )
    return latent / rms, rms


# ============================================================
# Model loading
# ============================================================

def load_models(
    config: Config,
    device: torch.device,
) -> LoadedModels:
    vae_encoder, vae_decoder = load_encoder_decoder(
        checkpoint=config.vae_checkpoint,
        device=device,
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
        strict=True,
        freeze=True,
    )

    vaegan_encoder, vaegan_generator = load_encoder_generator(
        checkpoint=config.vaegan_checkpoint,
        device=device,
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
        strict=True,
        freeze=True,
    )

    vae_ddpm, vae_ddpm_schedule = load_vae_ddpm_vae(
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

    vaegan_ddpm, vaegan_ddpm_schedule = load_vae_ddpm_gan(
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

    return LoadedModels(
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        vaegan_encoder=vaegan_encoder,
        vaegan_generator=vaegan_generator,
        vae_ddpm=vae_ddpm,
        vae_ddpm_schedule=vae_ddpm_schedule,
        vae_mlp=vae_mlp,
        vaegan_ddpm=vaegan_ddpm,
        vaegan_ddpm_schedule=vaegan_ddpm_schedule,
        vaegan_mlp=vaegan_mlp,
    )


# ============================================================
# Heterogeneous AWGN mixture channel
# ============================================================

def generate_noise_realization(batch_size: int,z_dim: int,device: torch.device,dtype: torch.dtype,seed: int) -> Tensor:
    """Generate one standard-Gaussian noise realization."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    return torch.randn(batch_size,z_dim,generator=generator,device=device,dtype=dtype)


def mix_heterogeneous_latents(z1_normalized: Tensor,z2_normalized: Tensor,noise_unit: Tensor,snr_db: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
    received = z1_norm + z2_norm + noise

    SNR is defined relative to the total clean mixture power.
    """
    clean_mixture = z1_normalized + z2_normalized

    if math.isinf(float(snr_db)):
        noise_std = torch.zeros(clean_mixture.size(0),1,device=clean_mixture.device,dtype=clean_mixture.dtype)
        received = clean_mixture
    else:
        mixture_power = clean_mixture.pow(2).mean(dim=1,keepdim=True).clamp_min(1e-12)

        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        noise_std = torch.sqrt(mixture_power / snr_linear)
        received = clean_mixture + noise_std * noise_unit

    return received, clean_mixture, noise_std


# ============================================================
# Separation
# ============================================================

@torch.no_grad()
def separate_vae_ddpm(models: LoadedModels,received: Tensor,labels: Tensor,config: Config,seed: int) -> Tensor:
    return sample_ddpm(
        model=models.vae_ddpm,
        schedule=models.vae_ddpm_schedule,
        cond=received,
        labels=labels,
        seed=seed,
        sampling_method=config.sampling_method,
        ddim_steps=config.ddim_steps,
        ddim_eta=config.ddim_eta,
        clip_x0=config.clip_x0,
    )


@torch.no_grad()
def separate_vaegan_ddpm(
    models: LoadedModels,
    received: Tensor,
    labels: Tensor,
    config: Config,
    seed: int,
) -> Tensor:
    return sample_ddpm(
        model=models.vaegan_ddpm,
        schedule=models.vaegan_ddpm_schedule,
        cond=received,
        labels=labels,
        seed=seed,
        sampling_method=config.sampling_method,
        ddim_steps=config.ddim_steps,
        ddim_eta=config.ddim_eta,
        clip_x0=config.clip_x0,
    )


@torch.no_grad()
def separate_vae_mlp(
    models: LoadedModels,
    received: Tensor,
) -> Tensor:
    # Existing separator convention: second output is DeepGlobe.
    _, deepglobe = models.vae_mlp(received)
    return deepglobe


@torch.no_grad()
def separate_vaegan_mlp(models: LoadedModels,received: Tensor) -> Tensor:
    # Existing separator convention: second output is DeepGlobe.
    _, deepglobe = models.vaegan_mlp(received)
    return deepglobe


# ============================================================
# Metrics
# ============================================================

def image_psnr(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = to_01(prediction)
    target = to_01(target)

    mse = (prediction - target).pow(2).flatten(1).mean(dim=1)

    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def image_ssim(prediction: Tensor, target: Tensor) -> Tensor:
    prediction_np = (to_01(prediction).permute(0, 2, 3, 1).cpu().numpy())
    target_np = (to_01(target).permute(0, 2, 3, 1).cpu().numpy())

    values = [skimage_ssim(real,pred,data_range=1.0,channel_axis=-1)for pred, real in zip(prediction_np, target_np)]

    return torch.tensor(values, dtype=torch.float32)


def image_mae(prediction: Tensor, target: Tensor) -> Tensor:
    return (to_01(prediction) - to_01(target)).abs().flatten(1).mean(dim=1)
def image_mse(prediction: Tensor, target: Tensor) -> Tensor:
    return (to_01(prediction) - to_01(target)).pow(2).flatten(1).mean(dim=1)
def compute_latent_metrics(prediction: Tensor,target: Tensor) -> Dict[str, Tensor]:
    difference = prediction - target

    return {
        "latent_mse": difference.pow(2).mean(dim=1),
        "latent_mae": difference.abs().mean(dim=1),
        "latent_nmse": (
            difference.pow(2).sum(dim=1)
            / target.pow(2).sum(dim=1).clamp_min(1e-12)
        ),
        "latent_cosine": F.cosine_similarity(
            prediction,
            target,
            dim=1,
        ),
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


# ============================================================
# Evaluation of one heterogeneous batch
# ============================================================

@torch.no_grad()
def evaluate_mixture_batch(
    models: LoadedModels,
    image_1: Tensor,
    image_2: Tensor,
    noise_unit: Tensor,
    snr_db: float,
    config: Config,
    base_sample_seed: int,
    lpips_model,
) -> Dict[str, object]:
    batch_size = image_1.size(0)
    labels = torch.ones(
        batch_size,
        dtype=torch.long,
        device=image_1.device,
    )

    # Source 1 uses VAE latent space.
    mu_1, logvar_1 = models.vae_encoder(image_1)
    z1 = mu_1

    # Source 2 uses VAE-GAN latent space.
    mu_2, logvar_2 = models.vaegan_encoder(image_2)
    z2 = mu_2

    z1_norm, rms_1 = power_normalize(z1)
    z2_norm, rms_2 = power_normalize(z2)

    received, clean_mixture, noise_std = mix_heterogeneous_latents(
        z1_normalized=z1_norm,
        z2_normalized=z2_norm,
        noise_unit=noise_unit,
        snr_db=snr_db,
    )

    # All four separators receive exactly the same heterogeneous mixture.
    pred_z1_norm_ddpm = separate_vae_ddpm(
        models=models,
        received=received,
        labels=labels,
        config=config,
        seed=base_sample_seed,
    )
    pred_z1_norm_mlp = separate_vae_mlp(
        models=models,
        received=received,
    )

    pred_z2_norm_ddpm = separate_vaegan_ddpm(
        models=models,
        received=received,
        labels=labels,
        config=config,
        seed=base_sample_seed,
    )
    pred_z2_norm_mlp = separate_vaegan_mlp(
        models=models,
        received=received,
    )

    # Restore each source with its own original RMS.
    pred_z1_ddpm = pred_z1_norm_ddpm * rms_1
    pred_z1_mlp = pred_z1_norm_mlp * rms_1
    pred_z2_ddpm = pred_z2_norm_ddpm * rms_2
    pred_z2_mlp = pred_z2_norm_mlp * rms_2

    reconstructions = {
        "VAE-DDPM-VAE": models.vae_decoder(
            pred_z1_ddpm, labels
        ),
        "VAE-MLP-VAE": models.vae_decoder(
            pred_z1_mlp, labels
        ),
        "VAEGAN-DDPM-GAN": models.vaegan_generator(
            pred_z2_ddpm, labels
        ),
        "VAEGAN-MLP-GAN": models.vaegan_generator(
            pred_z2_mlp, labels
        ),
    }

    predicted_latents = {
        "VAE-DDPM-VAE": pred_z1_ddpm,
        "VAE-MLP-VAE": pred_z1_mlp,
        "VAEGAN-DDPM-GAN": pred_z2_ddpm,
        "VAEGAN-MLP-GAN": pred_z2_mlp,
    }

    target_images = {
        "VAE-DDPM-VAE": image_1,
        "VAE-MLP-VAE": image_1,
        "VAEGAN-DDPM-GAN": image_2,
        "VAEGAN-MLP-GAN": image_2,
    }

    target_latents = {
        "VAE-DDPM-VAE": z1,
        "VAE-MLP-VAE": z1,
        "VAEGAN-DDPM-GAN": z2,
        "VAEGAN-MLP-GAN": z2,
    }

    oracle_reconstructions = {
        "z1": models.vae_decoder(z1, labels),
        "z2": models.vaegan_generator(z2, labels),
    }

    metrics_by_method: Dict[str, Dict[str, Tensor]] = {}

    for method in METHOD_ORDER:
        reconstruction = reconstructions[method]
        target_image = target_images[method]
        predicted_latent = predicted_latents[method]
        target_latent = target_latents[method]
        oracle = oracle_reconstructions[METHOD_TARGET[method]]

        method_metrics: Dict[str, Tensor] = {
            "psnr_real": image_psnr(
                reconstruction, target_image
            ),
            "ssim_real": image_ssim(
                reconstruction, target_image
            ),
            "image_mae_real": image_mae(
                reconstruction, target_image
            ),
            "image_mse_real": image_mse(
                reconstruction, target_image
            ),
            "psnr_oracle": image_psnr(
                reconstruction, oracle
            ),
            "ssim_oracle": image_ssim(
                reconstruction, oracle
            ),
            "oracle_psnr_real": image_psnr(
                oracle, target_image
            ),
            "oracle_ssim_real": image_ssim(
                oracle, target_image
            ),
        }

        method_metrics.update(
            compute_latent_metrics(
                prediction=predicted_latent,
                target=target_latent,
            )
        )

        lpips_real = calculate_lpips(
            lpips_model,
            reconstruction,
            target_image,
        )
        if lpips_real is not None:
            method_metrics["lpips_real"] = lpips_real

        metrics_by_method[method] = method_metrics

    diagnostics = {
        "noise_std_mean": noise_std.mean().item(),
        "clean_mixture_power_mean": (
            clean_mixture.pow(2).mean().item()
        ),
        "received_power_mean": received.pow(2).mean().item(),
        "z1_power_mean": z1.pow(2).mean().item(),
        "z2_power_mean": z2.pow(2).mean().item(),
        "posterior_std_1_mean": (
            torch.exp(0.5 * logvar_1).mean().item()
        ),
        "posterior_std_2_mean": (
            torch.exp(0.5 * logvar_2).mean().item()
        ),
    }

    return {
        "reconstructions": reconstructions,
        "predicted_latents": predicted_latents,
        "target_latents": target_latents,
        "oracle_reconstructions": oracle_reconstructions,
        "metrics_by_method": metrics_by_method,
        "received": received,
        "clean_mixture": clean_mixture,
        "diagnostics": diagnostics,
    }


# ============================================================
# Accumulator and CSV
# ============================================================

class MetricAccumulator:
    def __init__(self) -> None:
        self.values: Dict[str, List[float]] = {}

    def add(self, name: str, values: Tensor) -> None:
        array = (
            values.detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        self.values.setdefault(name, []).extend(array.tolist())

    def mean(self, name: str) -> float:
        values = self.values.get(name, [])
        return float(np.mean(values)) if values else float("nan")

    def std(self, name: str) -> float:
        values = self.values.get(name, [])
        return float(np.std(values)) if values else float("nan")

    def count(self) -> int:
        if not self.values:
            return 0
        return len(next(iter(self.values.values())))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames = sorted(
        {key for row in rows for key in row.keys()}
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_row(
    method_name: str,
    snr_db: float,
    accumulator: MetricAccumulator,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "method": method_name,
        "target_source": METHOD_TARGET[method_name],
        "snr_db": snr_db,
        "num_images": accumulator.count(),
        "channel": "z1_norm + z2_norm + AWGN",
    }

    for metric_name in accumulator.values:
        row[f"{metric_name}_mean"] = accumulator.mean(
            metric_name
        )
        row[f"{metric_name}_std"] = accumulator.std(
            metric_name
        )

    return row


# ============================================================
# Qualitative recovery grid
# ============================================================

def save_recovery_grid(
    image_1: Tensor,
    image_2: Tensor,
    reconstructions: Dict[str, Tensor],
    path: Path,
    count: int,
) -> None:
    """
    Rows:
        1. Source 1 original (VAE target)
        2. VAE-DDPM-VAE recovery
        3. VAE-MLP-VAE recovery
        4. Source 2 original (VAE-GAN target)
        5. VAEGAN-DDPM-GAN recovery
        6. VAEGAN-MLP-GAN recovery
    """
    count = min(count, image_1.size(0), image_2.size(0))

    rows = [
        image_1[:count].detach().cpu(),
        reconstructions["VAE-DDPM-VAE"][:count].detach().cpu(),
        reconstructions["VAE-MLP-VAE"][:count].detach().cpu(),
        image_2[:count].detach().cpu(),
        reconstructions["VAEGAN-DDPM-GAN"][:count].detach().cpu(),
        reconstructions["VAEGAN-MLP-GAN"][:count].detach().cpu(),
    ]

    grid = make_grid(
        torch.cat(rows, dim=0),
        nrow=count,
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )

    save_image(grid, path)


# ============================================================
# Plots
# ============================================================

PLOT_SPECS = {
    "PSNR": (
        "psnr_real_mean",
        "psnr_real_std",
        "PSNR (dB)",
        "PSNR vs SNR",
        True,
    ),
    "SSIM": (
        "ssim_real_mean",
        "ssim_real_std",
        "SSIM",
        "SSIM vs SNR",
        True,
    ),
    "LPIPS": (
        "lpips_real_mean",
        "lpips_real_std",
        "LPIPS",
        "LPIPS vs SNR",
        False,
    ),
    "Latent_NMSE": (
        "latent_nmse_mean",
        "latent_nmse_std",
        "Latent NMSE",
        "Latent NMSE vs SNR",
        False,
    ),
    "Latent_Cosine": (
        "latent_cosine_mean",
        "latent_cosine_std",
        "Latent cosine similarity",
        "Latent cosine similarity vs SNR",
        True,
    ),
}


def save_plot(
    figure: plt.Figure,
    output_stem: Path,
    config: Config,
) -> None:
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=config.figure_dpi,
        bbox_inches="tight",
    )

    if config.save_pdf:
        figure.savefig(
            output_stem.with_suffix(".pdf"),
            bbox_inches="tight",
        )

    plt.close(figure)


def plot_metric_vs_snr(
    summary_rows: List[Dict[str, object]],
    metric_key: str,
    mean_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    higher_is_better: bool,
    output_dir: Path,
    config: Config,
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    plotted = False

    for method in METHOD_ORDER:
        selected = sorted(
            [
                row
                for row in summary_rows
                if row["method"] == method
                and mean_column in row
                and np.isfinite(float(row[mean_column]))
            ],
            key=lambda row: float(row["snr_db"]),
        )

        if not selected:
            continue

        x = np.asarray(
            [float(row["snr_db"]) for row in selected],
            dtype=float,
        )
        y = np.asarray(
            [float(row[mean_column]) for row in selected],
            dtype=float,
        )

        line = axis.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            markersize=6.5,
            label=method,
        )[0]

        if (
            config.show_std_band
            and all(std_column in row for row in selected)
        ):
            std = np.asarray(
                [float(row[std_column]) for row in selected],
                dtype=float,
            )
            axis.fill_between(
                x,
                y - std,
                y + std,
                alpha=0.14,
                color=line.get_color(),
            )

        plotted = True

    if not plotted:
        plt.close(figure)
        print(f"Skipping {metric_key}: no data.")
        return

    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=9)
    axis.text(
        0.99,
        0.02,
        "Higher is better" if higher_is_better else "Lower is better",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    figure.tight_layout()
    save_plot(
        figure,
        output_dir / f"{metric_key}_vs_SNR",
        config,
    )


def plot_all_main_metrics(
    summary_rows: List[Dict[str, object]],
    output_dir: Path,
    config: Config,
) -> None:
    for metric_key, (
        mean_column,
        std_column,
        ylabel,
        title,
        higher_is_better,
    ) in PLOT_SPECS.items():
        plot_metric_vs_snr(
            summary_rows=summary_rows,
            metric_key=metric_key,
            mean_column=mean_column,
            std_column=std_column,
            ylabel=ylabel,
            title=title,
            higher_is_better=higher_is_better,
            output_dir=output_dir,
            config=config,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    config = Config()
    seed_everything(config.seed)

    device = torch.device(config.device)
    validate_paths(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {config.deepglobe_root}")
    print(f"SNR values: {config.snr_values}")
    print("Channel: received = z1_norm + z2_norm + noise")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    base_dataset = DeepGlobeDataset(
        root=config.deepglobe_root,
        transform=transform,
    )
    paired_dataset = PairedDeepGlobeDataset(
        base_dataset=base_dataset,
        num_pairs=config.num_pairs,
        seed=config.seed,
    )

    loader = DataLoader(
        paired_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("\nLoading encoders, decoders, and separators...")
    models = load_models(config, device)

    lpips_model = None
    if config.use_lpips:
        try:
            import lpips

            lpips_model = lpips.LPIPS(net="alex").to(device).eval()
            for parameter in lpips_model.parameters():
                parameter.requires_grad = False
            print("LPIPS enabled.")
        except ImportError:
            print(
                "LPIPS is not installed; LPIPS will be skipped.\n"
                "Install with: pip install lpips"
            )

    summary_rows: List[Dict[str, object]] = []
    long_rows: List[Dict[str, object]] = []
    qualitative_targets = {
        float(value) for value in config.qualitative_snrs
    }

    for snr_index, snr_db in enumerate(config.snr_values):
        print("\n" + "=" * 88)
        print(f"Evaluating SNR = {snr_db:g} dB")

        accumulators = {
            method: MetricAccumulator()
            for method in METHOD_ORDER
        }
        qualitative_saved = False

        for batch_index, batch in enumerate(loader):
            image_1, image_2, path_1, path_2 = batch
            image_1 = image_1.to(device, non_blocking=True)
            image_2 = image_2.to(device, non_blocking=True)

            current_batch = image_1.size(0)
            channel_seed = (
                config.seed
                + snr_index * 100_000
                + batch_index * 1000
            )

            noise_unit = generate_noise_realization(
                batch_size=current_batch,
                z_dim=config.z_dim,
                device=device,
                dtype=image_1.dtype,
                seed=channel_seed,
            )

            # The two DDPM methods use the same seed so differences are not
            # caused merely by different initial sampling noise.
            sample_seed = (
                config.seed
                + snr_index * 1_000_000
                + batch_index * 10_000
            )

            result = evaluate_mixture_batch(
                models=models,
                image_1=image_1,
                image_2=image_2,
                noise_unit=noise_unit,
                snr_db=snr_db,
                config=config,
                base_sample_seed=sample_seed,
                lpips_model=lpips_model,
            )

            metrics_by_method = result["metrics_by_method"]

            for method in METHOD_ORDER:
                metrics = metrics_by_method[method]
                accumulator = accumulators[method]

                for metric_name, values in metrics.items():
                    accumulator.add(metric_name, values)

                metrics_cpu = {
                    name: values.detach().cpu()
                    for name, values in metrics.items()
                }

                for image_index in range(current_batch):
                    row: Dict[str, object] = {
                        "method": method,
                        "target_source": METHOD_TARGET[method],
                        "snr_db": snr_db,
                        "batch_index": batch_index,
                        "image_index": image_index,
                        "source_1_file": path_1[image_index],
                        "source_2_file": path_2[image_index],
                    }

                    for metric_name, values in metrics_cpu.items():
                        row[metric_name] = float(
                            values[image_index].item()
                        )

                    long_rows.append(row)

            if (
                not qualitative_saved
                and float(snr_db) in qualitative_targets
            ):
                save_recovery_grid(
                    image_1=image_1,
                    image_2=image_2,
                    reconstructions=result["reconstructions"],
                    path=output_dir / (
                        f"recovery_SNR{snr_db:g}.png"
                    ),
                    count=config.qualitative_count,
                )
                qualitative_saved = True

            print(
                f"SNR={snr_db:g} | "
                f"batch {batch_index + 1}/{len(loader)} complete"
            )

        for method in METHOD_ORDER:
            row = build_summary_row(
                method_name=method,
                snr_db=snr_db,
                accumulator=accumulators[method],
            )
            summary_rows.append(row)

            print(
                f"{method}: "
                f"PSNR={row.get('psnr_real_mean', float('nan')):.3f}, "
                f"SSIM={row.get('ssim_real_mean', float('nan')):.4f}, "
                f"NMSE={row.get('latent_nmse_mean', float('nan')):.6f}, "
                f"Cos={row.get('latent_cosine_mean', float('nan')):.4f}"
            )

        # Partial save after every SNR.
        write_csv(output_dir / "metrics_long.csv", long_rows)
        write_csv(output_dir / "summary_metrics.csv", summary_rows)

    plot_all_main_metrics(
        summary_rows=summary_rows,
        output_dir=output_dir,
        config=config,
    )

    print("\n" + "=" * 88)
    print("Evaluation completed.")
    print(f"Results saved to:\n{output_dir.resolve()}")
    print("\nMain outputs:")
    print("  metrics_long.csv")
    print("  summary_metrics.csv")
    print("  PSNR_vs_SNR.png/.pdf")
    print("  SSIM_vs_SNR.png/.pdf")
    print("  LPIPS_vs_SNR.png/.pdf")
    print("  Latent_NMSE_vs_SNR.png/.pdf")
    print("  Latent_Cosine_vs_SNR.png/.pdf")
    print("  recovery_SNR0.png")
    print("  recovery_SNR10.png")
    print("  recovery_SNR20.png")
    print("  recovery_SNR30.png")


if __name__ == "__main__":
    main()
