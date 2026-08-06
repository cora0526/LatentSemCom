
"""
compare_vaegan_latent_noise.py

Compare latent-representation robustness of:

1. Conditional VAE-GAN trained with WGAN-GP
   checkpoint:
       ./save_model/joint_vae_gan_satellite_latest.pth

2. Conditional VAE-GAN trained with R3GAN
   checkpoint:
       ./save_model/vae_gan_r3gan_same_arch_best.pth

The comparison uses deterministic latent means mu:

    image -> encoder -> mu -> power normalization -> AWGN\n    -> restore original RMS -> generator -> reconstruction

It evaluates:
    - Image PSNR
    - Image SSIM
    - Image L1
    - Image MSE
    - Output drift relative to the clean reconstruction
    - Latent MSE
    - Latent cosine similarity
    - Power-normalized latent L2 norm
    - Noise L2 norm
    - Relative noise norm ||n|| / ||z||
    - Normalized latent distortion ||z_noisy-z||^2 / ||z||^2

Important:
    The WGAN-GP generator is imported from your existing model.py because its
    architecture must exactly match the pretrained checkpoint.

Required packages:
    pip install torch torchvision pandas matplotlib scikit-image pillow

Expected project files:
    model.py
    ./save_model/joint_vae_gan_satellite_latest.pth
    ./save_model/vae_gan_r3gan_same_arch_best.pth
"""

import csv
import glob
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

# This is the generator used by train_VAE_encoder_GAN_condition_c.py.
from model import Generator_Satellite_Conditional


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    wgangp_checkpoint: str = (
        "./save_model/joint_vae_gan_satellite_latest.pth"
    )
    r3gan_checkpoint: str = (
        "./save_model/vae_gan_r3gan_same_arch_best.pth"
    )

    save_root: str = "./save_data/VAEGAN_latent_noise_comparison_power_normalized"

    z_dim: int = 256
    label_dim: int = 64
    num_classes: int = 2

    batch_size: int = 64
    num_workers: int = 4

    # Number of test images used from each domain.
    max_test_images_per_domain: int = 1000

    # Use a deterministic held-out tail of each dataset.
    # This avoids evaluating only the first images used during training.
    use_dataset_tail: bool = True

    snr_values: Tuple[float, ...] = (
        0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0
    )

    # Repeat AWGN generation to reduce Monte-Carlo variation.
    noise_repeats: int = 3

    # Number of images saved in qualitative comparison grids.
    qualitative_count: int = 8

    # SNRs included in qualitative image grids.
    qualitative_snrs: Tuple[float, ...] = (
        0.0, 10.0, 20.0, 30.0
    )


EUROSAT_LABEL = 0
DEEPGLOBE_LABEL = 1


# ============================================================
# Dataset definitions
# ============================================================
class SatelliteDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform=None,
        domain_label: int = DEEPGLOBE_LABEL,
    ):
        self.files = sorted(
            glob.glob(os.path.join(root, "*_sat_*.jpg"))
        )
        self.transform = transform
        self.domain_label = int(domain_label)

        if not self.files:
            raise RuntimeError(
                f"No DeepGlobe images found in:\n{root}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image = Image.open(self.files[index]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.domain_label


class DomainLabelDataset(Dataset):
    def __init__(self, dataset: Dataset, domain_label: int):
        self.dataset = dataset
        self.domain_label = int(domain_label)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return image, self.domain_label


def take_test_subset(
    dataset: Dataset,
    max_items: int,
    use_tail: bool,
) -> Dataset:
    count = min(max_items, len(dataset))

    if count <= 0:
        raise RuntimeError("The selected test subset is empty.")

    if use_tail:
        start = len(dataset) - count
        indices = range(start, len(dataset))
    else:
        indices = range(count)

    return Subset(dataset, indices)


# ============================================================
# WGAN-GP VAE encoder
# Must match train_VAE_encoder_GAN_condition_c.py
# ============================================================
class WGANSharedVAEEncoder(nn.Module):
    def __init__(self, z_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(512 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, z_dim)

    def forward(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features(image).flatten(start_dim=1)
        mu = self.fc_mu(feature)
        logvar = torch.clamp(
            self.fc_logvar(feature),
            min=-10.0,
            max=10.0,
        )
        return mu, logvar


# ============================================================
# R3GAN VAE encoder and generator
# Must match train_vaegan_r3gan_same_arch.py
# ============================================================
class R3GANSharedVAEEncoder(nn.Module):
    def __init__(self, z_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_mu = nn.Linear(512 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, z_dim)

    def forward(
        self,
        image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features(image).flatten(start_dim=1)
        mu = self.fc_mu(feature)
        logvar = torch.clamp(
            self.fc_logvar(feature),
            min=-10.0,
            max=10.0,
        )
        return mu, logvar


class R3GANConditionalGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        num_classes: int = 2,
        label_dim: int = 64,
    ):
        super().__init__()

        self.label_embedding = nn.Embedding(
            num_classes,
            label_dim,
        )

        self.fc = nn.Sequential(
            nn.Linear(
                z_dim + label_dim,
                512 * 4 * 4,
            ),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        latent: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        label_feature = self.label_embedding(labels)
        joint = torch.cat(
            [latent, label_feature],
            dim=1,
        )
        feature = self.fc(joint)
        feature = feature.view(
            latent.size(0),
            512,
            4,
            4,
        )
        return self.decoder(feature)


# ============================================================
# General utilities
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: Config) -> torch.device:
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable. Switching to CPU.")
        return torch.device("cpu")

    return torch.device(config.device)


def load_checkpoint(path: str, device: torch.device) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be a dictionary: {path}"
        )

    return checkpoint


def load_models(
    config: Config,
    device: torch.device,
):
    # --------------------------
    # WGAN-GP model
    # --------------------------
    wgan_checkpoint = load_checkpoint(
        config.wgangp_checkpoint,
        device,
    )

    required_wgan_keys = {
        "encoder",
        "generator",
    }
    missing_wgan = required_wgan_keys.difference(
        wgan_checkpoint.keys()
    )

    if missing_wgan:
        raise KeyError(
            "WGAN-GP checkpoint is missing keys: "
            f"{sorted(missing_wgan)}"
        )

    wgan_encoder = WGANSharedVAEEncoder(
        z_dim=config.z_dim,
    ).to(device)

    wgan_generator = Generator_Satellite_Conditional(
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
        out_ch=3,
    ).to(device)

    wgan_encoder.load_state_dict(
        wgan_checkpoint["encoder"],
        strict=True,
    )
    wgan_generator.load_state_dict(
        wgan_checkpoint["generator"],
        strict=True,
    )

    # --------------------------
    # R3GAN model
    # --------------------------
    r3gan_checkpoint = load_checkpoint(
        config.r3gan_checkpoint,
        device,
    )

    required_r3gan_keys = {
        "encoder",
        "generator",
    }
    missing_r3gan = required_r3gan_keys.difference(
        r3gan_checkpoint.keys()
    )

    if missing_r3gan:
        raise KeyError(
            "R3GAN checkpoint is missing keys: "
            f"{sorted(missing_r3gan)}"
        )

    r3gan_encoder = R3GANSharedVAEEncoder(
        z_dim=config.z_dim,
    ).to(device)

    r3gan_generator = R3GANConditionalGenerator(
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
    ).to(device)

    r3gan_encoder.load_state_dict(
        r3gan_checkpoint["encoder"],
        strict=True,
    )
    r3gan_generator.load_state_dict(
        r3gan_checkpoint["generator"],
        strict=True,
    )

    models = {
        "WGAN-GP VAE-GAN": {
            "encoder": wgan_encoder.eval(),
            "generator": wgan_generator.eval(),
        },
        "R3GAN VAE-GAN": {
            "encoder": r3gan_encoder.eval(),
            "generator": r3gan_generator.eval(),
        },
    }

    return models


# ============================================================
# AWGN and metrics
# ============================================================
def power_normalize_latent(
    latent: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize every latent vector to unit average power:

        rms = sqrt(mean(z^2))
        z_tx = z / rms

    Therefore, for every sample:

        mean(z_tx^2) = 1
        ||z_tx||_2 = sqrt(z_dim)

    Returns:
        normalized_latent: latent transmitted through the channel
        latent_rms: original per-sample RMS, shape [B, 1]

    The decoder was trained on the original latent scale. After channel
    transmission, the received normalized latent is therefore multiplied by
    latent_rms before being passed to the decoder.
    """
    latent_rms = torch.sqrt(
        latent.pow(2)
        .mean(dim=1, keepdim=True)
        .clamp_min(eps)
    )
    normalized_latent = latent / latent_rms
    return normalized_latent, latent_rms


def add_awgn_to_unit_power_latent(
    normalized_latent: torch.Tensor,
    snr_db: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add AWGN after power normalization.

    Since mean(z_tx^2) = 1, the noise variance is identical for both models:

        noise_variance = 10^(-SNR_dB / 10)

    Returns:
        received_normalized_latent
        normalized_domain_noise
    """
    if math.isinf(snr_db):
        noise = torch.zeros_like(normalized_latent)
        return normalized_latent.clone(), noise

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_std = math.sqrt(1.0 / snr_linear)
    noise = torch.randn_like(normalized_latent) * noise_std

    return normalized_latent + noise, noise


def transmit_power_normalized_latent(
    latent: torch.Tensor,
    snr_db: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Complete latent-channel operation:

        z_tx = z / rms(z)
        z_rx = z_tx + n
        z_decoder = z_rx * rms(z)

    Returns:
        decoder_latent
        clean_normalized_latent
        noisy_normalized_latent
        normalized_noise
        original_rms

    Restoring the original RMS before decoding is necessary because the
    pretrained generators were not trained to decode unit-power-normalized
    latent vectors directly.
    """
    normalized_latent, latent_rms = power_normalize_latent(latent)
    noisy_normalized_latent, normalized_noise = (
        add_awgn_to_unit_power_latent(
            normalized_latent,
            snr_db,
        )
    )
    decoder_latent = noisy_normalized_latent * latent_rms

    return (
        decoder_latent,
        normalized_latent,
        noisy_normalized_latent,
        normalized_noise,
        latent_rms,
    )


def to_unit_range(image: torch.Tensor) -> torch.Tensor:
    return ((image + 1.0) / 2.0).clamp(0.0, 1.0)


def batch_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = to_unit_range(prediction)
    target = to_unit_range(target)

    mse = (
        (prediction - target)
        .pow(2)
        .flatten(start_dim=1)
        .mean(dim=1)
    )

    return -10.0 * torch.log10(
        mse.clamp_min(1e-12)
    )


def batch_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = to_unit_range(prediction)
    target = to_unit_range(target)

    return (
        (prediction - target)
        .abs()
        .flatten(start_dim=1)
        .mean(dim=1)
    )


def batch_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = to_unit_range(prediction)
    target = to_unit_range(target)

    return (
        (prediction - target)
        .pow(2)
        .flatten(start_dim=1)
        .mean(dim=1)
    )


def batch_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    prediction_np = (
        to_unit_range(prediction)
        .detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )
    target_np = (
        to_unit_range(target)
        .detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )

    values = []

    for pred_image, target_image in zip(
        prediction_np,
        target_np,
    ):
        value = structural_similarity(
            target_image,
            pred_image,
            data_range=1.0,
            channel_axis=-1,
        )
        values.append(float(value))

    return np.asarray(values, dtype=np.float64)


def compute_latent_metrics(
    clean_latent: torch.Tensor,
    noisy_latent: torch.Tensor,
    noise: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    clean_norm = clean_latent.norm(
        p=2,
        dim=1,
    )
    noisy_norm = noisy_latent.norm(
        p=2,
        dim=1,
    )
    noise_norm = noise.norm(
        p=2,
        dim=1,
    )

    latent_mse = (
        (noisy_latent - clean_latent)
        .pow(2)
        .mean(dim=1)
    )

    latent_cosine = F.cosine_similarity(
        noisy_latent,
        clean_latent,
        dim=1,
        eps=1e-12,
    )

    relative_noise = (
        noise_norm
        / clean_norm.clamp_min(1e-12)
    )

    normalized_distortion = (
        (noisy_latent - clean_latent)
        .pow(2)
        .sum(dim=1)
        / clean_latent
        .pow(2)
        .sum(dim=1)
        .clamp_min(1e-12)
    )

    return {
        "latent_mse": latent_mse,
        "latent_cosine": latent_cosine,
        "clean_latent_l2": clean_norm,
        "noisy_latent_l2": noisy_norm,
        "noise_l2": noise_norm,
        "relative_noise_l2": relative_noise,
        "normalized_latent_distortion": normalized_distortion,
    }


def tensor_to_numpy(
    value: torch.Tensor,
) -> np.ndarray:
    return value.detach().cpu().numpy()


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_domain(
    model_name: str,
    encoder: nn.Module,
    generator: nn.Module,
    loader: DataLoader,
    domain_name: str,
    snr_values: Iterable[float],
    noise_repeats: int,
    device: torch.device,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns:
        summary_rows:
            one averaged row per SNR

        sample_rows:
            one row per image, noise repeat, and SNR
    """
    summary_rows: List[Dict] = []
    sample_rows: List[Dict] = []

    for snr_db in snr_values:
        accumulators: Dict[str, List[np.ndarray]] = {
            "psnr": [],
            "ssim": [],
            "l1": [],
            "image_mse": [],
            "clean_recon_psnr": [],
            "clean_recon_ssim": [],
            "clean_recon_l1": [],
            "output_drift_l1": [],
            "output_drift_mse": [],
            "latent_mse": [],
            "latent_cosine": [],
            "clean_latent_l2": [],
            "noisy_latent_l2": [],
            "noise_l2": [],
            "relative_noise_l2": [],
            "normalized_latent_distortion": [],
            "original_latent_rms": [],
        }

        sample_counter = 0

        for repeat_index in range(noise_repeats):
            for images, labels in loader:
                images = images.to(
                    device,
                    non_blocking=True,
                )
                labels = labels.to(
                    device,
                    dtype=torch.long,
                    non_blocking=True,
                )

                # Deterministic representation.
                mu, _ = encoder(images)

                # Clean decoder output acts as the decoder-specific oracle.
                clean_reconstruction = generator(
                    mu,
                    labels,
                )

                (
                    noisy_mu_for_decoder,
                    normalized_mu,
                    noisy_normalized_mu,
                    normalized_noise,
                    original_mu_rms,
                ) = transmit_power_normalized_latent(
                    mu,
                    snr_db,
                )

                noisy_reconstruction = generator(
                    noisy_mu_for_decoder,
                    labels,
                )

                # Latent robustness is measured in the normalized channel
                # domain, where both models have exactly unit average power.
                latent_metrics = compute_latent_metrics(
                    normalized_mu,
                    noisy_normalized_mu,
                    normalized_noise,
                )

                latent_metrics["original_latent_rms"] = (
                    original_mu_rms.squeeze(1)
                )

                values = {
                    "psnr": tensor_to_numpy(
                        batch_psnr(
                            noisy_reconstruction,
                            images,
                        )
                    ),
                    "ssim": batch_ssim(
                        noisy_reconstruction,
                        images,
                    ),
                    "l1": tensor_to_numpy(
                        batch_l1(
                            noisy_reconstruction,
                            images,
                        )
                    ),
                    "image_mse": tensor_to_numpy(
                        batch_mse(
                            noisy_reconstruction,
                            images,
                        )
                    ),
                    "clean_recon_psnr": tensor_to_numpy(
                        batch_psnr(
                            clean_reconstruction,
                            images,
                        )
                    ),
                    "clean_recon_ssim": batch_ssim(
                        clean_reconstruction,
                        images,
                    ),
                    "clean_recon_l1": tensor_to_numpy(
                        batch_l1(
                            clean_reconstruction,
                            images,
                        )
                    ),
                    "output_drift_l1": tensor_to_numpy(
                        batch_l1(
                            noisy_reconstruction,
                            clean_reconstruction,
                        )
                    ),
                    "output_drift_mse": tensor_to_numpy(
                        batch_mse(
                            noisy_reconstruction,
                            clean_reconstruction,
                        )
                    ),
                }

                for key, metric_tensor in latent_metrics.items():
                    values[key] = tensor_to_numpy(
                        metric_tensor
                    )

                for key, array in values.items():
                    accumulators[key].append(array)

                batch_size = images.size(0)

                for local_index in range(batch_size):
                    row = {
                        "model": model_name,
                        "domain": domain_name,
                        "snr_db": float(snr_db),
                        "noise_repeat": repeat_index,
                        "sample_index": sample_counter + local_index,
                    }

                    for key, array in values.items():
                        row[key] = float(
                            array[local_index]
                        )

                    sample_rows.append(row)

                sample_counter += batch_size

        merged = {
            key: np.concatenate(arrays, axis=0)
            for key, arrays in accumulators.items()
        }

        summary_row = {
            "model": model_name,
            "domain": domain_name,
            "snr_db": float(snr_db),
            "num_measurements": int(
                len(merged["psnr"])
            ),
        }

        for metric_name, metric_values in merged.items():
            summary_row[f"{metric_name}_mean"] = float(
                np.mean(metric_values)
            )
            summary_row[f"{metric_name}_std"] = float(
                np.std(metric_values)
            )

        summary_rows.append(summary_row)

        print(
            f"{model_name:20s} | "
            f"{domain_name:10s} | "
            f"SNR={snr_db:>4.1f} dB | "
            f"PSNR={summary_row['psnr_mean']:.3f} | "
            f"SSIM={summary_row['ssim_mean']:.4f} | "
            f"LatCos={summary_row['latent_cosine_mean']:.4f} | "
            f"DriftL1={summary_row['output_drift_l1_mean']:.6f}"
        )

    return summary_rows, sample_rows


# ============================================================
# Overall aggregation
# ============================================================
def add_overall_rows(
    sample_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = [
        column
        for column in sample_dataframe.columns
        if column not in {
            "model",
            "domain",
            "snr_db",
            "noise_repeat",
            "sample_index",
        }
    ]

    overall_rows = []

    grouped = sample_dataframe.groupby(
        ["model", "snr_db"],
        sort=True,
    )

    for (model_name, snr_db), group in grouped:
        row = {
            "model": model_name,
            "domain": "Overall",
            "snr_db": float(snr_db),
            "num_measurements": len(group),
        }

        for metric in metric_columns:
            row[f"{metric}_mean"] = float(
                group[metric].mean()
            )
            row[f"{metric}_std"] = float(
                group[metric].std(ddof=0)
            )

        overall_rows.append(row)

    return pd.DataFrame(overall_rows)


# ============================================================
# Plotting
# ============================================================
def save_metric_plot(
    summary_dataframe: pd.DataFrame,
    metric_column: str,
    ylabel: str,
    filename: str,
    save_root: str,
    domain: str,
) -> None:
    subset = summary_dataframe[
        summary_dataframe["domain"] == domain
    ]

    if subset.empty:
        return

    plt.figure(figsize=(7.2, 5.2))

    for model_name, group in subset.groupby("model"):
        group = group.sort_values("snr_db")

        plt.plot(
            group["snr_db"],
            group[metric_column],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    plt.xlabel("Latent AWGN SNR (dB)")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Latent SNR — {domain}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        save_root,
        filename,
    )
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def generate_all_plots(
    summary_dataframe: pd.DataFrame,
    save_root: str,
) -> None:
    metric_specs = [
        (
            "psnr_mean",
            "Reconstruction PSNR (dB)",
            "psnr",
        ),
        (
            "ssim_mean",
            "Reconstruction SSIM",
            "ssim",
        ),
        (
            "l1_mean",
            "Reconstruction L1",
            "l1",
        ),
        (
            "image_mse_mean",
            "Reconstruction MSE",
            "image_mse",
        ),
        (
            "output_drift_l1_mean",
            "Output Drift L1",
            "output_drift_l1",
        ),
        (
            "output_drift_mse_mean",
            "Output Drift MSE",
            "output_drift_mse",
        ),
        (
            "latent_mse_mean",
            "Latent MSE",
            "latent_mse",
        ),
        (
            "latent_cosine_mean",
            "Latent Cosine Similarity",
            "latent_cosine",
        ),
        (
            "clean_latent_l2_mean",
            "Clean Latent L2 Norm",
            "clean_latent_l2",
        ),
        (
            "noisy_latent_l2_mean",
            "Noisy Latent L2 Norm",
            "noisy_latent_l2",
        ),
        (
            "relative_noise_l2_mean",
            "Relative Noise Norm ||n|| / ||z||",
            "relative_noise_l2",
        ),
        (
            "normalized_latent_distortion_mean",
            "Normalized Latent Distortion",
            "normalized_latent_distortion",
        ),
        (
            "original_latent_rms_mean",
            "Original Latent RMS Before Normalization",
            "original_latent_rms",
        ),
    ]

    domains = sorted(
        summary_dataframe["domain"].unique()
    )

    for domain in domains:
        safe_domain = domain.replace(" ", "_")

        for metric_column, ylabel, stem in metric_specs:
            if metric_column not in summary_dataframe.columns:
                continue

            save_metric_plot(
                summary_dataframe=summary_dataframe,
                metric_column=metric_column,
                ylabel=ylabel,
                filename=f"{stem}_{safe_domain}.png",
                save_root=save_root,
                domain=domain,
            )


# ============================================================
# Qualitative image comparison
# ============================================================
@torch.no_grad()
def save_qualitative_comparison(
    models: Dict,
    dataset: Dataset,
    domain_name: str,
    label_value: int,
    config: Config,
    device: torch.device,
) -> None:
    count = min(
        config.qualitative_count,
        len(dataset),
    )

    images = torch.stack(
        [dataset[index][0] for index in range(count)]
    ).to(device)

    labels = torch.full(
        (count,),
        fill_value=label_value,
        dtype=torch.long,
        device=device,
    )

    # Use the same random standard-normal directions for both models.
    noise_generators = {}

    for snr_db in config.qualitative_snrs:
        generator = torch.Generator(
            device=device.type
        )
        generator.manual_seed(
            config.seed + int(snr_db * 10) + label_value
        )
        noise_generators[snr_db] = generator

    for model_name, model_parts in models.items():
        encoder = model_parts["encoder"]
        decoder = model_parts["generator"]

        mu, _ = encoder(images)
        clean_recon = decoder(mu, labels)

        rows = [
            to_unit_range(images),
            to_unit_range(clean_recon),
        ]

        row_names = [
            "original",
            "clean",
        ]

        normalized_mu, mu_rms = power_normalize_latent(mu)

        for snr_db in config.qualitative_snrs:
            snr_linear = 10.0 ** (snr_db / 10.0)
            noise_std = math.sqrt(1.0 / snr_linear)

            # The same standard-normal directions are used for both models.
            standard_noise = torch.randn(
                mu.shape,
                generator=noise_generators[snr_db],
                device=device,
                dtype=mu.dtype,
            )
            noisy_normalized_mu = (
                normalized_mu
                + standard_noise * noise_std
            )

            # Restore the model-specific original latent scale before decoding.
            noisy_mu_for_decoder = noisy_normalized_mu * mu_rms
            recon = decoder(noisy_mu_for_decoder, labels)

            rows.append(to_unit_range(recon))
            row_names.append(f"snr_{int(snr_db)}")

        grid_tensor = torch.cat(rows, dim=0)

        safe_model = (
            model_name
            .replace(" ", "_")
            .replace("-", "")
        )

        save_image(
            grid_tensor,
            os.path.join(
                config.save_root,
                f"qualitative_{safe_model}_{domain_name}.png",
            ),
            nrow=count,
        )

        with open(
            os.path.join(
                config.save_root,
                f"qualitative_{safe_model}_{domain_name}_rows.txt",
            ),
            "w",
            encoding="utf-8",
        ) as handle:
            for row_index, row_name in enumerate(row_names):
                handle.write(
                    f"Row {row_index + 1}: {row_name}\n"
                )


# ============================================================
# Latent-dimension sensitivity
# ============================================================
@torch.no_grad()
def collect_dimension_statistics(
    models: Dict,
    loaders: Dict[str, DataLoader],
    config: Config,
    device: torch.device,
) -> pd.DataFrame:
    """
    Collect clean latent statistics and local generator sensitivity.

    Local sensitivity is estimated by perturbing each latent dimension
    independently with a small scale relative to that dimension's std.

    To keep runtime manageable, sensitivity is evaluated on one batch
    per domain and model.
    """
    rows = []

    for model_name, model_parts in models.items():
        encoder = model_parts["encoder"]
        generator = model_parts["generator"]

        for domain_name, loader in loaders.items():
            all_mu = []

            for images, _ in loader:
                images = images.to(device)
                mu, _ = encoder(images)
                normalized_mu, _ = power_normalize_latent(mu)
                all_mu.append(normalized_mu.detach().cpu())

            # Heatmaps now describe the unit-power channel representation,
            # not the unequal raw latent scales.
            latent_matrix = torch.cat(all_mu, dim=0)
            latent_mean = latent_matrix.mean(dim=0)
            latent_std = latent_matrix.std(dim=0, unbiased=False)
            latent_abs_mean = latent_matrix.abs().mean(dim=0)

            for dim_index in range(config.z_dim):
                rows.append(
                    {
                        "model": model_name,
                        "domain": domain_name,
                        "dimension": dim_index,
                        "latent_mean": float(
                            latent_mean[dim_index]
                        ),
                        "latent_std": float(
                            latent_std[dim_index]
                        ),
                        "latent_abs_mean": float(
                            latent_abs_mean[dim_index]
                        ),
                    }
                )

    return pd.DataFrame(rows)


def save_dimension_heatmaps(
    dimension_dataframe: pd.DataFrame,
    save_root: str,
) -> None:
    for metric in [
        "latent_mean",
        "latent_std",
        "latent_abs_mean",
    ]:
        pivot = dimension_dataframe.pivot_table(
            index=["model", "domain"],
            columns="dimension",
            values=metric,
        )

        matrix = pivot.to_numpy()

        plt.figure(figsize=(14, 4.8))
        image = plt.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
        )
        plt.colorbar(
            image,
            label=metric.replace("_", " ").title(),
        )
        plt.yticks(
            range(len(pivot.index)),
            [
                f"{model} — {domain}"
                for model, domain in pivot.index
            ],
        )
        plt.xlabel("Latent Dimension")
        plt.ylabel("Model and Domain")
        plt.title(
            f"Per-Dimension {metric.replace('_', ' ').title()}"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                save_root,
                f"heatmap_{metric}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# ============================================================
# Main
# ============================================================
def main() -> None:
    config = Config()
    set_seed(config.seed)

    device = resolve_device(config)
    os.makedirs(
        config.save_root,
        exist_ok=True,
    )

    print(f"Device: {device}")
    print("Loading datasets...")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    eurosat_base = datasets.ImageFolder(
        root=config.eurosat_root,
        transform=transform,
    )
    eurosat_full = DomainLabelDataset(
        eurosat_base,
        EUROSAT_LABEL,
    )

    deepglobe_full = SatelliteDataset(
        root=config.deepglobe_root,
        transform=transform,
        domain_label=DEEPGLOBE_LABEL,
    )

    eurosat_test = take_test_subset(
        eurosat_full,
        config.max_test_images_per_domain,
        config.use_dataset_tail,
    )
    deepglobe_test = take_test_subset(
        deepglobe_full,
        config.max_test_images_per_domain,
        config.use_dataset_tail,
    )

    loaders = {
        "EuroSAT": DataLoader(
            eurosat_test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(config.num_workers > 0),
        ),
        "DeepGlobe": DataLoader(
            deepglobe_test,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(config.num_workers > 0),
        ),
    }

    print(
        f"EuroSAT test images: {len(eurosat_test)}"
    )
    print(
        f"DeepGlobe test images: {len(deepglobe_test)}"
    )

    print("Loading models...")
    models = load_models(
        config,
        device,
    )

    all_summary_rows = []
    all_sample_rows = []

    for model_name, model_parts in models.items():
        for domain_name, loader in loaders.items():
            summary_rows, sample_rows = evaluate_domain(
                model_name=model_name,
                encoder=model_parts["encoder"],
                generator=model_parts["generator"],
                loader=loader,
                domain_name=domain_name,
                snr_values=config.snr_values,
                noise_repeats=config.noise_repeats,
                device=device,
            )

            all_summary_rows.extend(summary_rows)
            all_sample_rows.extend(sample_rows)

    sample_dataframe = pd.DataFrame(
        all_sample_rows
    )
    domain_summary_dataframe = pd.DataFrame(
        all_summary_rows
    )
    overall_summary_dataframe = add_overall_rows(
        sample_dataframe
    )

    summary_dataframe = pd.concat(
        [
            domain_summary_dataframe,
            overall_summary_dataframe,
        ],
        ignore_index=True,
    )

    sample_csv_path = os.path.join(
        config.save_root,
        "per_sample_metrics.csv",
    )
    summary_csv_path = os.path.join(
        config.save_root,
        "summary_metrics.csv",
    )

    sample_dataframe.to_csv(
        sample_csv_path,
        index=False,
    )
    summary_dataframe.to_csv(
        summary_csv_path,
        index=False,
    )

    print("Generating plots...")
    generate_all_plots(
        summary_dataframe,
        config.save_root,
    )

    print("Saving qualitative comparisons...")
    save_qualitative_comparison(
        models=models,
        dataset=eurosat_test,
        domain_name="EuroSAT",
        label_value=EUROSAT_LABEL,
        config=config,
        device=device,
    )
    save_qualitative_comparison(
        models=models,
        dataset=deepglobe_test,
        domain_name="DeepGlobe",
        label_value=DEEPGLOBE_LABEL,
        config=config,
        device=device,
    )

    print("Collecting per-dimension latent statistics...")
    dimension_dataframe = collect_dimension_statistics(
        models=models,
        loaders=loaders,
        config=config,
        device=device,
    )

    dimension_csv_path = os.path.join(
        config.save_root,
        "latent_dimension_statistics.csv",
    )
    dimension_dataframe.to_csv(
        dimension_csv_path,
        index=False,
    )

    save_dimension_heatmaps(
        dimension_dataframe,
        config.save_root,
    )

    print("=" * 80)
    print("Comparison completed.")
    print(f"Output directory:\n{config.save_root}")
    print(f"Summary CSV:\n{summary_csv_path}")
    print(f"Per-sample CSV:\n{sample_csv_path}")
    print(f"Dimension CSV:\n{dimension_csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
