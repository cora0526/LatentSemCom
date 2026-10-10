"""
Conditional GAN pretraining -> Joint Deep-Residual mu-VAE-GAN fine-tuning -> SISO latent MAC -> Label-conditioned Latent DDPM

Target hardware: NVIDIA RTX 4060 8GB
Image size: 64x64 RGB
Domains:
    0 = EuroSAT
    1 = DeepGlobe

Recommended execution order:
    python train_gan_pretrain_then_joint_mu_vaegan_ddpm.py --stage hybrid
    python train_gan_pretrain_then_joint_mu_vaegan_ddpm.py --stage ddpm
    python train_gan_pretrain_then_joint_mu_vaegan_ddpm.py --stage eval

Or run all training stages sequentially:
    python train_gan_pretrain_then_joint_mu_vaegan_ddpm.py --stage all

Important:
1. This file uses one shared deep GAN architecture.
2. The GAN is trained with Conditional WGAN-GP.
3. The Conditional GAN is first pretrained from random Gaussian latent vectors.
4. A deeper ten-block residual VAE Encoder is then initialized, while the pretrained Generator and Critic are jointly fine-tuned with it.
5. Reconstruction uses the deterministic encoder mean mu, while random-prior generation remains active.
6. The DDPM uses the fixed globally-scaled encoder mean z=c*mu as its clean target latent.
7. DDPM training uses target-centered random-interference observations y_k=z_k+alpha*z_j+n.
8. alpha is sampled per sample; the target domain label tells the DDPM which latent to recover.
9. Training and sampling use the same complete 400-step diffusion horizon.
10. The fixed global latent scale is inverted before the recovered latent is passed to the Generator.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ============================================================
# Configuration
# ============================================================


@dataclass
class Config:
    # Paths
    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    save_root: str = "./save_data/Deep_WGANGP_VAE_DDPM"
    model_root: str = "./save_model"

    # Reproducibility / device
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4

    # Data / architecture
    image_size: int = 64
    max_images_per_domain: int = 50_000
    z_dim: int = 256
    label_dim: int = 64
    num_classes: int = 2

    # Deep GAN architecture suitable for RTX 4060 8GB
    # 4x4, 8x8, 16x16, 32x32, 64x64
    g_widths: Tuple[int, ...] = (512, 512, 256, 128, 64)
    g_blocks: Tuple[int, ...] = (1, 1, 2, 2, 2)
    d_widths: Tuple[int, ...] = (64, 128, 256, 512, 512)
    d_blocks: Tuple[int, ...] = (2, 2, 2, 1, 1)
    groups: int = 8

    # GAN pretraining before joint VAE-GAN fine-tuning
    gan_pretrain_epochs: int = 30
    gan_epochs: int = 150
    gan_batch_size: int = 16
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    beta1: float = 0.0
    beta2: float = 0.9
    lambda_gp: float = 10.0
    n_critic: int = 2
    gan_save_every: int = 1

    # Joint VAE-GAN fine-tuning after GAN pretraining
    joint_epochs: int = 150
    joint_batch_size: int = 16
    lr_joint_encoder: float = 1e-4
    lr_joint_generator: float = 5e-5
    lr_joint_critic: float = 1e-4
    lambda_recon_l1: float = 10.0
    lambda_recon_perceptual: float = 1.0
    lambda_recon_adv: float = 0.05
    lambda_random_adv: float = 1.0
    beta_kl_joint_max: float = 1e-4
    joint_kl_warmup_epochs: int = 30
    latent_std_floor: float = 0.05
    joint_save_every: int = 1

    # Legacy separate Encoder training (kept for compatibility)
    encoder_epochs: int = 60
    encoder_batch_size: int = 32
    lr_encoder: float = 1e-4
    beta_kl_max: float = 1e-4
    kl_warmup_epochs: int = 20
    lambda_encoder_adv: float = 0.001
    encoder_save_every: int = 5

    # Channel
    snr_min_db: float = 0.0
    snr_max_db: float = 30.0

    # Fixed GLOBAL latent scaling obtained from the training-set calibration.
    #
    # IMPORTANT:
    #   z_tx = global_latent_scale * mu
    #
    # Do NOT replace this with per-sample unit-power normalization.
    # The Generator was jointly trained on raw encoder mean mu, and the
    # diagnostic showed that per-sample normalization severely changes the
    # latent distribution. At the receiver, recovered channel/DDPM latents
    # are divided by this same fixed scalar before entering the Generator.
    global_latent_scale: float = 0.17011429

    # Fixed 2x2 channel matrix before Frobenius normalization
    h11: float = 1.0
    h12: float = 0.15
    h21: float = 0.10
    h22: float = 0.90

    # DDPM training
    ddpm_epochs: int = 100
    ddpm_batch_size: int = 32
    ddpm_steps_per_epoch: int = 1000
    lr_ddpm: float = 2e-4
    # Label-conditioned SISO-MAC DDPM.
    # IMPORTANT: train and inference use the SAME full horizon 0..399.
    diffusion_steps: int = 400
    train_t_max: int = 400  # kept only for backward-compatible config printing
    time_dim: int = 128
    ddpm_hidden_dim: int = 768
    lambda_x0: float = 0.2
    grad_clip: float = 1.0
    ddpm_save_every: int = 5
    ddpm_clean_epochs: int = 20

    # Random interference-amplitude training.
    # During the first clean-channel curriculum stage:
    #   alpha ~ U(ddpm_alpha_clean_min, ddpm_alpha_clean_max)
    # Afterwards:
    #   alpha ~ U(ddpm_alpha_min, ddpm_alpha_max), SNR ~ U(snr_min_db, snr_max_db)
    # alpha is sampled independently for every sample and target domain.
    ddpm_alpha_clean_min: float = 0.0
    ddpm_alpha_clean_max: float = 1.0
    ddpm_alpha_min: float = 0.0
    ddpm_alpha_max: float = 2.0

    # Evaluation
    eval_snr_db: float = 10.0
    eval_count: int = 64
    # Full reverse chain: 399 -> ... -> 0.
    ddim_steps: int = 400

    @property
    def gan_checkpoint(self) -> str:
        return os.path.join(self.model_root, "deep_wgangp_satellite_latest.pth")

    @property
    def gan_best_checkpoint(self) -> str:
        return os.path.join(self.model_root, "deep_wgangp_satellite_best.pth")

    @property
    def gan_pretrain_checkpoint(self) -> str:
        return os.path.join(self.model_root, "deep_wgangp_pretrain_latest.pth")

    @property
    def gan_pretrain_best_checkpoint(self) -> str:
        return os.path.join(self.model_root, "deep_wgangp_pretrain_best.pth")

    @property
    def encoder_checkpoint(self) -> str:
        return os.path.join(self.model_root, "deep_wgangp_vae_encoder_latest.pth")

    @property
    def encoder_best_checkpoint(self) -> str:
        return os.path.join(self.model_root, "deep_wgangp_vae_encoder_best.pth")

    @property
    def joint_checkpoint(self) -> str:
        return os.path.join(self.model_root, "deep_wgangp_deep_vae_gan_joint_latest.pth")

    @property
    def joint_best_checkpoint(self) -> str:
        return os.path.join(self.model_root, "deep_wgangp_deep_vae_gan_joint_best.pth")

    @property
    def ddpm_checkpoint(self) -> str:
        return os.path.join(self.model_root, "deep_label_cond_siso_ddpm_latest_random_alpha.pth")

    @property
    def ddpm_best_checkpoint(self) -> str:
        return os.path.join(self.model_root, "deep_label_cond_siso_ddpm_best_random_alpha.pth")


EUROSAT_LABEL = 0
DEEPGLOBE_LABEL = 1


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; switching to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def ensure_dirs(cfg: Config) -> None:
    os.makedirs(cfg.save_root, exist_ok=True)
    os.makedirs(cfg.model_root, exist_ok=True)


def append_csv(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_requires_grad(model: nn.Module, value: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(value)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def normalize_latent_power(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Legacy per-sample latent normalization.

    WARNING:
    This function is intentionally NOT used by the new DDPM/channel pipeline.
    It is kept only for backward compatibility with older experiments.
    """
    power = z.pow(2).mean(dim=1, keepdim=True)
    return z / torch.sqrt(power + eps)


def apply_global_latent_scale(z: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Transmitter-side fixed global scaling: z_tx = c * mu."""
    return z * float(cfg.global_latent_scale)


def invert_global_latent_scale(z: torch.Tensor, cfg: Config) -> torch.Tensor:
    """Receiver-side inverse scaling: mu_hat = z_hat / c."""
    c = float(cfg.global_latent_scale)
    if abs(c) < 1e-12:
        raise ValueError("global_latent_scale must be non-zero.")
    return z / c


def infinite_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


# ============================================================
# Datasets
# ============================================================


class SatelliteDataset(Dataset):
    """DeepGlobe patches, with a fixed domain label."""

    def __init__(self, root: str, transform=None, domain_label: int = DEEPGLOBE_LABEL):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform
        self.domain_label = int(domain_label)
        if not self.files:
            raise RuntimeError(f"No DeepGlobe images matched '*_sat_*.jpg' in:\n{root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        with Image.open(self.files[idx]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.domain_label


class DomainLabelDataset(Dataset):
    """Replace ImageFolder's land-cover label with a domain label."""

    def __init__(self, dataset: Dataset, domain_label: int):
        self.dataset = dataset
        self.domain_label = int(domain_label)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return image, self.domain_label


def build_transform(cfg: Config):
    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def build_domain_datasets(cfg: Config) -> Tuple[Dataset, Dataset]:
    transform = build_transform(cfg)

    base_eurosat = datasets.ImageFolder(cfg.eurosat_root, transform=transform)
    base_eurosat = Subset(
        base_eurosat,
        range(min(cfg.max_images_per_domain, len(base_eurosat))),
    )
    eurosat = DomainLabelDataset(base_eurosat, EUROSAT_LABEL)

    deepglobe_base = SatelliteDataset(
        cfg.deepglobe_root,
        transform=transform,
        domain_label=DEEPGLOBE_LABEL,
    )
    deepglobe = Subset(
        deepglobe_base,
        range(min(cfg.max_images_per_domain, len(deepglobe_base))),
    )

    return eurosat, deepglobe


def build_combined_loader(
    cfg: Config,
    batch_size: int,
    shuffle: bool = True,
) -> Tuple[DataLoader, Dataset, Dataset]:
    eurosat, deepglobe = build_domain_datasets(cfg)
    combined = ConcatDataset([eurosat, deepglobe])
    loader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=True,
    )
    return loader, eurosat, deepglobe


def build_paired_loaders(
    cfg: Config,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, Dataset, Dataset]:
    eurosat, deepglobe = build_domain_datasets(cfg)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        drop_last=True,
    )
    return (
        DataLoader(eurosat, **kwargs),
        DataLoader(deepglobe, **kwargs),
        eurosat,
        deepglobe,
    )


# ============================================================
# Deep conditional GAN
# ============================================================


def valid_groups(channels: int, requested_groups: int) -> int:
    groups = min(channels, requested_groups)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class GeneratorResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int, upsample: bool):
        super().__init__()
        self.upsample = upsample
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            groups=valid_groups(in_ch, groups) if in_ch == out_ch else 1,
        )
        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            groups=valid_groups(out_ch, groups),
        )
        self.mix = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.silu(self.norm1(x), inplace=False)
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            residual = F.interpolate(residual, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.conv2(F.silu(self.norm2(h), inplace=False))
        h = self.mix(h)
        residual = self.skip(residual)
        return (h + residual) / math.sqrt(2.0)


class CriticResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int, downsample: bool):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            groups=valid_groups(out_ch, groups),
        )
        self.mix = nn.Conv2d(out_ch, out_ch, kernel_size=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.leaky_relu(x, 0.2, inplace=False))
        h = self.conv2(F.leaky_relu(h, 0.2, inplace=False))
        h = self.mix(h)
        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2)
            residual = F.avg_pool2d(residual, kernel_size=2)
        return (h + residual) / math.sqrt(2.0)


class DeepConditionalGenerator(nn.Module):
    """64x64 conditional residual Generator."""

    def __init__(self, cfg: Config):
        super().__init__()
        if len(cfg.g_widths) != 5 or len(cfg.g_blocks) != 5:
            raise ValueError("g_widths and g_blocks must each contain 5 stages.")

        self.z_dim = cfg.z_dim
        self.label_embedding = nn.Embedding(cfg.num_classes, cfg.label_dim)
        self.fc = nn.Linear(cfg.z_dim + cfg.label_dim, cfg.g_widths[0] * 4 * 4)

        blocks = []
        in_ch = cfg.g_widths[0]
        for stage, (out_ch, num_blocks) in enumerate(zip(cfg.g_widths, cfg.g_blocks)):
            for block_index in range(num_blocks):
                upsample = stage > 0 and block_index == 0
                blocks.append(GeneratorResBlock(in_ch, out_ch, cfg.groups, upsample))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=False),
            nn.Conv2d(in_ch, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_feature = self.label_embedding(labels)
        h = self.fc(torch.cat([z, label_feature], dim=1))
        h = h.view(z.size(0), -1, 4, 4)
        h = self.blocks(h)
        return self.to_rgb(h)


class DeepConditionalCritic(nn.Module):
    """64x64 projection conditional Critic."""

    def __init__(self, cfg: Config):
        super().__init__()
        if len(cfg.d_widths) != 5 or len(cfg.d_blocks) != 5:
            raise ValueError("d_widths and d_blocks must each contain 5 stages.")

        self.from_rgb = nn.Conv2d(3, cfg.d_widths[0], kernel_size=3, padding=1)
        blocks = []
        in_ch = cfg.d_widths[0]
        for stage, (out_ch, num_blocks) in enumerate(zip(cfg.d_widths, cfg.d_blocks)):
            for block_index in range(num_blocks):
                downsample = stage < 4 and block_index == num_blocks - 1
                blocks.append(CriticResBlock(in_ch, out_ch, cfg.groups, downsample))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.unconditional = nn.Linear(in_ch, 1)
        self.label_embedding = nn.Embedding(cfg.num_classes, in_ch)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.from_rgb(x)
        h = self.blocks(h)
        h = F.leaky_relu(h, 0.2, inplace=False)
        return h.sum(dim=(2, 3))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = self.extract_features(x)
        unconditional = self.unconditional(h).squeeze(1)
        projection = (h * self.label_embedding(labels)).sum(dim=1)
        return unconditional + projection


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
    interpolated = alpha * real + (1.0 - alpha) * fake
    interpolated.requires_grad_(True)
    scores = critic(interpolated, labels)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


# ============================================================
# Deep residual shared VAE Encoder
# ============================================================


class EncoderResBlock(nn.Module):
    """Pre-activation residual block used by the VAE encoder."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        groups: int = 8,
        downsample: bool = False,
    ):
        super().__init__()
        self.downsample = downsample
        self.norm1 = nn.GroupNorm(valid_groups(in_ch, groups), in_ch)
        self.norm2 = nn.GroupNorm(valid_groups(out_ch, groups), out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(F.silu(self.norm1(x), inplace=False))
        h = self.conv2(F.silu(self.norm2(h), inplace=False))

        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2)
            residual = F.avg_pool2d(residual, kernel_size=2)

        return (h + residual) / math.sqrt(2.0)


class SharedVAEEncoder(nn.Module):
    """
    Deep residual VAE encoder for 64x64 RGB images.

    Spatial path:
        64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4

    Channel path:
        3 -> 64 -> 128 -> 256 -> 512 -> 512

    The encoder contains ten residual blocks instead of four plain
    convolutional downsampling layers. GroupNorm is used because the joint
    VAE-GAN training batch size may be as small as 8 or 16.
    """

    def __init__(self, z_dim: int = 256, groups: int = 8):
        super().__init__()
        self.stem = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        self.features = nn.Sequential(
            # 64x64, 64 channels
            EncoderResBlock(64, 64, groups=groups, downsample=False),
            EncoderResBlock(64, 128, groups=groups, downsample=True),
            # 32x32, 128 channels
            EncoderResBlock(128, 128, groups=groups, downsample=False),
            EncoderResBlock(128, 256, groups=groups, downsample=True),
            # 16x16, 256 channels
            EncoderResBlock(256, 256, groups=groups, downsample=False),
            EncoderResBlock(256, 512, groups=groups, downsample=True),
            # 8x8, 512 channels
            EncoderResBlock(512, 512, groups=groups, downsample=False),
            EncoderResBlock(512, 512, groups=groups, downsample=True),
            # 4x4, 512 channels
            EncoderResBlock(512, 512, groups=groups, downsample=False),
            EncoderResBlock(512, 512, groups=groups, downsample=False),
        )
        self.final_norm = nn.GroupNorm(valid_groups(512, groups), 512)
        feature_dim = 512 * 4 * 4
        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.features(h)
        h = F.silu(self.final_norm(h), inplace=False).flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()


# ============================================================
# 2x2 MIMO latent channel
# ============================================================


def get_channel_matrix(cfg: Config, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h = torch.tensor(
        [[cfg.h11, cfg.h12], [cfg.h21, cfg.h22]],
        device=device,
        dtype=dtype,
    )
    # Frobenius norm sqrt(2), matching a two-stream normalized channel.
    h = h / torch.linalg.norm(h, ord="fro") * math.sqrt(2.0)
    return h


def mimo_2x2_observation(
    z_e: torch.Tensor,
    z_g: torch.Tensor,
    snr_db: torch.Tensor | float,
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    2x2 MIMO channel for ALREADY globally-scaled latent vectors.

    Inputs:
        z_e, z_g:
            [B, z_dim] transmitter latents, where
                z_e = c * mu_e
                z_g = c * mu_g

        snr_db:
            scalar or [B]

    Returns:
        observation:
            [B, 2*z_dim], noisy two-receive-antenna observation

        zf_e:
            [B, z_dim], ZF estimate in the globally-scaled latent domain

        zf_g:
            [B, z_dim], ZF estimate in the globally-scaled latent domain

    IMPORTANT:
        There is NO per-sample normalization inside this function.
        Scaling must be applied exactly once before entering the channel.
    """
    if z_e.shape != z_g.shape:
        raise ValueError(
            f"Latent shapes must match, got {z_e.shape} and {z_g.shape}"
        )

    streams = torch.stack([z_e, z_g], dim=1)  # [B, 2, Z]
    h = get_channel_matrix(cfg, streams.device, streams.dtype)

    clean_y = torch.einsum("ij,bjz->biz", h, streams)

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
        snr_db = snr_db.expand(streams.size(0))

    signal_power = clean_y.pow(2).mean(
        dim=(1, 2),
        keepdim=True,
    )

    noise_power = signal_power / torch.pow(
        10.0,
        snr_db.view(-1, 1, 1) / 10.0,
    )

    noisy_y = (
        clean_y
        + torch.randn_like(clean_y)
        * torch.sqrt(noise_power + 1e-12)
    )

    h_inv = torch.linalg.inv(h)

    zf = torch.einsum(
        "ij,bjz->biz",
        h_inv,
        noisy_y,
    )

    observation = noisy_y.flatten(start_dim=1)

    return observation, zf[:, 0], zf[:, 1]


def siso_mac_observation(
    z_e: torch.Tensor,
    z_g: torch.Tensor,
    snr_db: torch.Tensor | float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-receiver latent multiple-access channel.

        y = z_E + z_D + n

    Both z_E and z_D must already be in the globally-scaled latent domain:

        z_E = c * mu_E
        z_D = c * mu_D

    Args:
        z_e, z_g:
            [B, z_dim] globally-scaled source latents.

        snr_db:
            Scalar SNR or [B] per-sample SNR in dB.
            math.inf gives the clean superposition.

    Returns:
        observation:
            [B, z_dim] noisy superimposed latent y.

        clean_mix:
            [B, z_dim] clean superposition z_E + z_D.
    """
    if z_e.shape != z_g.shape:
        raise ValueError(
            f"Latent shapes must match, got {z_e.shape} and {z_g.shape}"
        )

    clean_mix = z_e + z_g

    if not torch.is_tensor(snr_db):
        if math.isinf(float(snr_db)):
            return clean_mix, clean_mix

        snr_db = torch.tensor(
            float(snr_db),
            device=clean_mix.device,
            dtype=clean_mix.dtype,
        )

    snr_db = snr_db.to(
        device=clean_mix.device,
        dtype=clean_mix.dtype,
    )

    if snr_db.ndim == 0:
        if torch.isinf(snr_db):
            return clean_mix, clean_mix
        snr_db = snr_db.expand(clean_mix.size(0))

    # Allow a vector that may contain +inf.
    signal_power = clean_mix.pow(2).mean(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-12)

    finite_mask = torch.isfinite(snr_db).view(-1, 1)

    safe_snr = torch.where(
        torch.isfinite(snr_db),
        snr_db,
        torch.full_like(snr_db, 1000.0),
    )

    noise_power = signal_power / torch.pow(
        10.0,
        safe_snr.view(-1, 1) / 10.0,
    )

    noise = torch.randn_like(clean_mix) * torch.sqrt(
        noise_power + 1e-12
    )

    noise = torch.where(
        finite_mask,
        noise,
        torch.zeros_like(noise),
    )

    observation = clean_mix + noise

    return observation, clean_mix



def target_centered_siso_observation(
    z_target: torch.Tensor,
    z_interference: torch.Tensor,
    alpha: torch.Tensor | float,
    snr_db: torch.Tensor | float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Target-centered latent interference channel used for random-alpha DDPM training.

        y = z_target + alpha * z_interference + n

    alpha may be a scalar or a per-sample tensor [B].  AWGN power is computed
    from the complete clean mixture, matching the evaluation sweep definition.
    """
    if z_target.shape != z_interference.shape:
        raise ValueError(
            f"Latent shapes must match, got {z_target.shape} and {z_interference.shape}"
        )

    batch = z_target.size(0)
    device = z_target.device
    dtype = z_target.dtype

    if not torch.is_tensor(alpha):
        alpha = torch.tensor(float(alpha), device=device, dtype=dtype)
    alpha = alpha.to(device=device, dtype=dtype)
    if alpha.ndim == 0:
        alpha = alpha.expand(batch)
    if alpha.ndim != 1 or alpha.size(0) != batch:
        raise ValueError(f"alpha must be scalar or [B], got {tuple(alpha.shape)}")

    clean_mix = z_target + alpha.view(-1, 1) * z_interference

    if not torch.is_tensor(snr_db):
        if math.isinf(float(snr_db)):
            return clean_mix, clean_mix
        snr_db = torch.tensor(float(snr_db), device=device, dtype=dtype)

    snr_db = snr_db.to(device=device, dtype=dtype)
    if snr_db.ndim == 0:
        if torch.isinf(snr_db):
            return clean_mix, clean_mix
        snr_db = snr_db.expand(batch)
    if snr_db.ndim != 1 or snr_db.size(0) != batch:
        raise ValueError(f"snr_db must be scalar or [B], got {tuple(snr_db.shape)}")

    signal_power = clean_mix.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    finite_mask = torch.isfinite(snr_db).view(-1, 1)
    safe_snr = torch.where(
        torch.isfinite(snr_db),
        snr_db,
        torch.full_like(snr_db, 1000.0),
    )
    noise_power = signal_power / torch.pow(10.0, safe_snr.view(-1, 1) / 10.0)
    noise = torch.randn_like(clean_mix) * torch.sqrt(noise_power + 1e-12)
    noise = torch.where(finite_mask, noise, torch.zeros_like(noise))

    return clean_mix + noise, clean_mix


# ============================================================
# Latent DDPM
# ============================================================


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = math.log(10_000) / max(half - 1, 1)
    frequencies = torch.exp(
        -scale * torch.arange(half, device=t.device, dtype=torch.float32)
    )
    angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(F.silu(self.fc1(h)))
        return (x + h) / math.sqrt(2.0)


class ConditionalLatentDDPM(nn.Module):
    """
    Label-conditioned target latent separator.

    The SAME received mixture can be queried with different labels:

        (y, label=EuroSAT)   -> recover z_E
        (y, label=DeepGlobe) -> recover z_D

    During diffusion training the network predicts epsilon for the noisy
    TARGET latent z_target,t conditioned on the received mixture and label.

    Inputs:
        z_t:
            [B, z_dim] noisy target latent at timestep t.

        t:
            [B] diffusion timestep.

        labels:
            [B], 0=EuroSAT, 1=DeepGlobe.

        observation:
            [B, z_dim], received SISO-MAC mixture
                y = z_E + z_D + n.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.time_dim = cfg.time_dim

        self.label_embedding = nn.Embedding(
            cfg.num_classes,
            cfg.label_dim,
        )

        # z_t + received mixture + time embedding + target label embedding.
        input_dim = (
            cfg.z_dim
            + cfg.z_dim
            + cfg.time_dim
            + cfg.label_dim
        )

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, cfg.ddpm_hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            ResidualMLPBlock(cfg.ddpm_hidden_dim),
            ResidualMLPBlock(cfg.ddpm_hidden_dim),
            ResidualMLPBlock(cfg.ddpm_hidden_dim),
            ResidualMLPBlock(cfg.ddpm_hidden_dim),
        )

        self.output = nn.Sequential(
            nn.LayerNorm(cfg.ddpm_hidden_dim),
            nn.SiLU(),
            nn.Linear(
                cfg.ddpm_hidden_dim,
                cfg.z_dim,
            ),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        if observation.ndim != 2 or observation.size(1) != z_t.size(1):
            raise ValueError(
                "SISO-MAC observation must have shape [B, z_dim]. "
                f"Got z_t={tuple(z_t.shape)}, observation={tuple(observation.shape)}."
            )

        t_emb = sinusoidal_time_embedding(
            t,
            self.time_dim,
        ).to(z_t.dtype)

        y_emb = self.label_embedding(labels)

        h = torch.cat(
            [
                z_t,
                observation,
                t_emb,
                y_emb,
            ],
            dim=1,
        )

        h = self.input_proj(h)
        h = self.blocks(h)

        return self.output(h)


class DiffusionSchedule(nn.Module):
    """
    Linear-beta diffusion schedule.

    Training samples t uniformly from ALL steps:
        t in {0, ..., steps-1}

    The full ancestral reverse sampler also starts from steps-1 and reaches 0,
    eliminating the previous train/inference timestep mismatch.
    """

    def __init__(self, steps: int):
        super().__init__()

        beta = torch.linspace(
            1e-4,
            0.02,
            steps,
            dtype=torch.float32,
        )

        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(
            alpha,
            dim=0,
        )

        alpha_bar_prev = F.pad(
            alpha_bar[:-1],
            (1, 0),
            value=1.0,
        )

        posterior_variance = (
            beta
            * (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar)
        )

        self.steps = steps

        self.register_buffer(
            "beta",
            beta,
        )
        self.register_buffer(
            "alpha",
            alpha,
        )
        self.register_buffer(
            "alpha_bar",
            alpha_bar,
        )
        self.register_buffer(
            "alpha_bar_prev",
            alpha_bar_prev,
        )
        self.register_buffer(
            "sqrt_alpha_bar",
            torch.sqrt(alpha_bar),
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_bar",
            torch.sqrt(1.0 - alpha_bar),
        )
        self.register_buffer(
            "posterior_variance",
            torch.clamp(
                posterior_variance,
                min=1e-20,
            ),
        )

    def extract(
        self,
        values: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return values.gather(
            0,
            t,
        ).view(
            -1,
            *([1] * (x.ndim - 1)),
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        a = self.extract(
            self.sqrt_alpha_bar,
            t,
            x0,
        )

        b = self.extract(
            self.sqrt_one_minus_alpha_bar,
            t,
            x0,
        )

        return a * x0 + b * noise

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        a = self.extract(
            self.sqrt_alpha_bar,
            t,
            x_t,
        )

        b = self.extract(
            self.sqrt_one_minus_alpha_bar,
            t,
            x_t,
        )

        return (
            x_t
            - b * eps_pred
        ) / (
            a + 1e-8
        )


@torch.no_grad()
def sample_conditional_ddpm(
    model: ConditionalLatentDDPM,
    schedule: DiffusionSchedule,
    observation: torch.Tensor,
    labels: torch.Tensor,
    z_dim: int,
    initial_noise: Optional[torch.Tensor] = None,
    clip_x0: Optional[float] = None,
) -> torch.Tensor:
    """
    Full ancestral DDPM sampling:

        x_{T-1} ~ N(0, I)
        T-1 -> T-2 -> ... -> 0

    Because training also uses every t in 0..T-1, inference never queries an
    unseen timestep.
    """
    model.eval()

    batch_size = observation.size(0)
    device = observation.device

    if initial_noise is None:
        x = torch.randn(
            batch_size,
            z_dim,
            device=device,
        )
    else:
        x = initial_noise.to(device)

        if x.shape != (
            batch_size,
            z_dim,
        ):
            raise ValueError(
                "initial_noise must have shape "
                f"{(batch_size, z_dim)}, got {tuple(x.shape)}"
            )

    for step in reversed(
        range(
            schedule.steps
        )
    ):
        t = torch.full(
            (batch_size,),
            step,
            device=device,
            dtype=torch.long,
        )

        pred_eps = model(
            x,
            t,
            labels,
            observation,
        )

        pred_x0 = schedule.predict_x0(
            x,
            t,
            pred_eps,
        )

        if clip_x0 is not None:
            pred_x0 = pred_x0.clamp(
                -clip_x0,
                clip_x0,
            )

        beta_t = schedule.extract(
            schedule.beta,
            t,
            x,
        )

        alpha_t = schedule.extract(
            schedule.alpha,
            t,
            x,
        )

        alpha_bar_t = schedule.extract(
            schedule.alpha_bar,
            t,
            x,
        )

        alpha_bar_prev = schedule.extract(
            schedule.alpha_bar_prev,
            t,
            x,
        )

        coef_x0 = (
            beta_t
            * torch.sqrt(
                alpha_bar_prev
            )
            / (
                1.0
                - alpha_bar_t
            )
        )

        coef_xt = (
            (
                1.0
                - alpha_bar_prev
            )
            * torch.sqrt(
                alpha_t
            )
            / (
                1.0
                - alpha_bar_t
            )
        )

        posterior_mean = (
            coef_x0 * pred_x0
            + coef_xt * x
        )

        if step > 0:
            variance = schedule.extract(
                schedule.posterior_variance,
                t,
                x,
            )

            x = (
                posterior_mean
                + torch.sqrt(
                    variance
                )
                * torch.randn_like(x)
            )
        else:
            x = posterior_mean

    return x


@torch.no_grad()
def ddim_sample(
    model: ConditionalLatentDDPM,
    schedule: DiffusionSchedule,
    observation: torch.Tensor,
    labels: torch.Tensor,
    z_dim: int,
    sample_steps: int,
) -> torch.Tensor:
    """
    Compatibility wrapper.

    For this experiment we intentionally use the FULL ancestral DDPM chain
    because the desired recipe is training t=0..399 and sampling 399..0.
    sample_steps is accepted so old call sites remain valid, but the sampler
    always traverses all schedule steps.
    """
    del sample_steps

    return sample_conditional_ddpm(
        model=model,
        schedule=schedule,
        observation=observation,
        labels=labels,
        z_dim=z_dim,
    )


# ============================================================
# Checkpoint helpers
# ============================================================


def save_gan_checkpoint(
    path: str,
    epoch: int,
    generator: nn.Module,
    critic: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    cfg: Config,
    metric: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "config": asdict(cfg),
            "metric": metric,
        },
        path,
    )


def load_gan_models(
    cfg: Config,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> Tuple[DeepConditionalGenerator, DeepConditionalCritic]:
    path = checkpoint_path or cfg.gan_best_checkpoint
    if not os.path.exists(path):
        path = cfg.gan_checkpoint
    if not os.path.exists(path):
        raise FileNotFoundError(f"GAN checkpoint not found: {path}")

    generator = DeepConditionalGenerator(cfg).to(device)
    critic = DeepConditionalCritic(cfg).to(device)
    checkpoint = torch.load(path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    critic.load_state_dict(checkpoint["critic"])
    return generator, critic


def load_joint_models(
    cfg: Config,
    device: torch.device,
) -> Tuple[SharedVAEEncoder, DeepConditionalGenerator, DeepConditionalCritic]:
    path = cfg.joint_best_checkpoint
    if not os.path.exists(path):
        path = cfg.joint_checkpoint
    if not os.path.exists(path):
        raise FileNotFoundError(f"Joint VAE-GAN checkpoint not found: {path}")

    encoder = SharedVAEEncoder(cfg.z_dim).to(device)
    generator = DeepConditionalGenerator(cfg).to(device)
    critic = DeepConditionalCritic(cfg).to(device)
    checkpoint = torch.load(path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    generator.load_state_dict(checkpoint["generator"])
    critic.load_state_dict(checkpoint["critic"])
    return encoder, generator, critic


def load_encoder(cfg: Config, device: torch.device) -> SharedVAEEncoder:
    path = cfg.encoder_best_checkpoint
    if not os.path.exists(path):
        path = cfg.encoder_checkpoint
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
    encoder = SharedVAEEncoder(cfg.z_dim).to(device)
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["encoder"] if "encoder" in checkpoint else checkpoint
    encoder.load_state_dict(state)
    return encoder


def load_ddpm(
    cfg: Config,
    device: torch.device,
) -> Tuple[ConditionalLatentDDPM, DiffusionSchedule]:
    path = cfg.ddpm_best_checkpoint

    if not os.path.exists(path):
        path = cfg.ddpm_checkpoint

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"DDPM checkpoint not found: {path}"
        )

    model = ConditionalLatentDDPM(cfg).to(device)
    schedule = DiffusionSchedule(cfg.diffusion_steps).to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["ddpm"]
    )

    checkpoint_scale = checkpoint.get(
        "global_latent_scale",
        None,
    )

    checkpoint_config = checkpoint.get(
        "config",
        {},
    )

    checkpoint_steps = checkpoint_config.get(
        "diffusion_steps",
        cfg.diffusion_steps,
    )

    if int(checkpoint_steps) != int(cfg.diffusion_steps):
        raise RuntimeError(
            "DDPM checkpoint diffusion_steps does not match Config. "
            f"checkpoint={checkpoint_steps}, config={cfg.diffusion_steps}"
        )

    if checkpoint_scale is not None:
        if not math.isclose(
            float(checkpoint_scale),
            float(cfg.global_latent_scale),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                "DDPM checkpoint global_latent_scale does not match Config. "
                f"checkpoint={checkpoint_scale}, "
                f"config={cfg.global_latent_scale}"
            )

    return model, schedule


# ============================================================
# Stage 1: Conditional GAN pretraining
# ============================================================


def pretrain_gan(cfg: Config, device: torch.device) -> None:
    print("\n========== Stage 1: GAN Pretraining ==========")
    loader, eurosat, deepglobe = build_combined_loader(cfg, cfg.gan_batch_size)

    generator = DeepConditionalGenerator(cfg).to(device)
    critic = DeepConditionalCritic(cfg).to(device)
    generator.apply(init_weights)
    critic.apply(init_weights)

    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2)
    )
    optimizer_d = torch.optim.Adam(
        critic.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2)
    )

    print(f"Device: {device}")
    print(f"EuroSAT: {len(eurosat)}, DeepGlobe: {len(deepglobe)}")
    print(f"Steps per epoch: {len(loader)}")
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Critic parameters: {count_parameters(critic):,}")

    fixed_z = torch.randn(64, cfg.z_dim, device=device)
    fixed_e_labels = torch.full((64,), EUROSAT_LABEL, device=device, dtype=torch.long)
    fixed_g_labels = torch.full((64,), DEEPGLOBE_LABEL, device=device, dtype=torch.long)

    history_path = os.path.join(cfg.save_root, "gan_pretraining_history.csv")
    best_generator_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.gan_pretrain_epochs):
        generator.train()
        critic.train()
        epoch_d = 0.0
        epoch_g = 0.0
        epoch_gp = 0.0
        g_updates = 0

        for real, labels in loader:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            batch = real.size(0)

            # Critic update
            set_requires_grad(critic, True)
            set_requires_grad(generator, False)
            z = torch.randn(batch, cfg.z_dim, device=device)
            with torch.no_grad():
                fake = generator(z, labels)

            d_real = critic(real, labels)
            d_fake = critic(fake, labels)
            gp = gradient_penalty(critic, real, fake, labels)
            loss_d = d_fake.mean() - d_real.mean() + cfg.lambda_gp * gp

            optimizer_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optimizer_d.step()

            epoch_d += loss_d.item()
            epoch_gp += gp.item()

            # Generator update every n_critic steps
            if (global_step + 1) % cfg.n_critic == 0:
                set_requires_grad(critic, False)
                set_requires_grad(generator, True)
                z = torch.randn(batch, cfg.z_dim, device=device)
                fake = generator(z, labels)
                loss_g = -critic(fake, labels).mean()

                optimizer_g.zero_grad(set_to_none=True)
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
                optimizer_g.step()

                epoch_g += loss_g.item()
                g_updates += 1
            else:
                loss_g = torch.zeros((), device=device)

            if global_step % 200 == 0:
                print(
                    f"Epoch {epoch + 1}/{cfg.gan_pretrain_epochs} | Step {global_step} | "
                    f"D={loss_d.item():.4f} | G={loss_g.item():.4f} | GP={gp.item():.4f}"
                )
            global_step += 1

        avg_d = epoch_d / len(loader)
        avg_gp = epoch_gp / len(loader)
        avg_g = epoch_g / max(g_updates, 1)

        append_csv(
            history_path,
            {
                "epoch": epoch + 1,
                "critic_loss": avg_d,
                "generator_loss": avg_g,
                "gradient_penalty": avg_gp,
            },
        )

        save_gan_checkpoint(
            cfg.gan_pretrain_checkpoint,
            epoch + 1,
            generator,
            critic,
            optimizer_g,
            optimizer_d,
            cfg,
            avg_g,
        )

        if avg_g < best_generator_loss:
            best_generator_loss = avg_g
            save_gan_checkpoint(
                cfg.gan_pretrain_best_checkpoint,
                epoch + 1,
                generator,
                critic,
                optimizer_g,
                optimizer_d,
                cfg,
                avg_g,
            )

        if (epoch + 1) % cfg.gan_save_every == 0 or epoch == 0:
            generator.eval()
            with torch.no_grad():
                sample_e = generator(fixed_z, fixed_e_labels)
                sample_g = generator(fixed_z, fixed_g_labels)
            save_image(
                (sample_e + 1.0) / 2.0,
                os.path.join(cfg.save_root, f"pretrain_random_eurosat_ep{epoch + 1:03d}.png"),
                nrow=8,
            )
            save_image(
                (sample_g + 1.0) / 2.0,
                os.path.join(cfg.save_root, f"pretrain_random_deepglobe_ep{epoch + 1:03d}.png"),
                nrow=8,
            )

        print(
            f"[GAN Pretrain] Epoch {epoch + 1}: D={avg_d:.4f}, G={avg_g:.4f}, GP={avg_gp:.4f}"
        )



# ============================================================
# Stage 2: Joint VAE-GAN fine-tuning after GAN pretraining
# ============================================================


def train_joint_vae_gan(cfg: Config, device: torch.device) -> None:
    print("\n========== Stage 2: Joint mu-VAE-GAN Fine-tuning ==========")
    loader, eurosat, deepglobe = build_combined_loader(cfg, cfg.joint_batch_size)

    encoder = SharedVAEEncoder(cfg.z_dim).to(device)
    encoder.apply(init_weights)

    # Start joint training from the GAN-pretrained Generator and Critic.
    pretrain_path = cfg.gan_pretrain_best_checkpoint
    if not os.path.exists(pretrain_path):
        pretrain_path = cfg.gan_pretrain_checkpoint
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(
            "GAN pretraining checkpoint was not found. Run --stage pretrain or "
            "--stage hybrid before --stage joint. Expected: " + pretrain_path
        )

    generator = DeepConditionalGenerator(cfg).to(device)
    critic = DeepConditionalCritic(cfg).to(device)
    pretrain_checkpoint = torch.load(pretrain_path, map_location=device)
    generator.load_state_dict(pretrain_checkpoint["generator"])
    critic.load_state_dict(pretrain_checkpoint["critic"])
    print(
        f"Loaded GAN pretraining checkpoint: {pretrain_path} "
        f"(epoch={pretrain_checkpoint.get('epoch', 'unknown')})"
    )

    optimizer_eg = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": cfg.lr_joint_encoder},
            {"params": generator.parameters(), "lr": cfg.lr_joint_generator},
        ],
        betas=(0.5, 0.999),
    )
    optimizer_d = torch.optim.Adam(
        critic.parameters(),
        lr=cfg.lr_joint_critic,
        betas=(cfg.beta1, cfg.beta2),
    )

    fixed_count = min(32, len(eurosat), len(deepglobe))
    fixed_e = torch.stack([eurosat[i][0] for i in range(fixed_count)]).to(device)
    fixed_g = torch.stack([deepglobe[i][0] for i in range(fixed_count)]).to(device)
    fixed_e_labels = torch.full((fixed_count,), EUROSAT_LABEL, device=device, dtype=torch.long)
    fixed_g_labels = torch.full((fixed_count,), DEEPGLOBE_LABEL, device=device, dtype=torch.long)
    fixed_z = torch.randn(fixed_count, cfg.z_dim, device=device)

    history_path = os.path.join(cfg.save_root, "joint_vae_gan_history.csv")
    best_recon = float("inf")
    global_step = 0

    for epoch in range(cfg.joint_epochs):
        encoder.train()
        generator.train()
        critic.train()
        beta_kl = cfg.beta_kl_joint_max * min(
            1.0, (epoch + 1) / max(cfg.joint_kl_warmup_epochs, 1)
        )

        sums = {k: 0.0 for k in (
            "d", "gp", "eg", "l1", "feat", "kl", "adv_recon", "adv_random", "std"
        )}

        for real, labels in loader:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            batch = real.size(0)

            # ---------------- Critic update ----------------
            set_requires_grad(critic, True)
            set_requires_grad(encoder, False)
            set_requires_grad(generator, False)
            with torch.no_grad():
                mu, logvar = encoder(real)
                # Deterministic reconstruction path: Generator receives encoder mean mu.
                fake_recon = generator(mu, labels)
                z_prior = torch.randn(batch, cfg.z_dim, device=device)
                fake_random = generator(z_prior, labels)

            fake_for_d = torch.cat([fake_recon, fake_random], dim=0)
            labels_for_d = torch.cat([labels, labels], dim=0)
            real_for_gp = torch.cat([real, real], dim=0)
            d_real = critic(real, labels)
            d_fake_recon = critic(fake_recon, labels)
            d_fake_random = critic(fake_random, labels)
            gp = gradient_penalty(critic, real_for_gp, fake_for_d, labels_for_d)
            loss_d = (
                0.5 * (d_fake_recon.mean() + d_fake_random.mean())
                - d_real.mean()
                + cfg.lambda_gp * gp
            )

            optimizer_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optimizer_d.step()

            # ---------------- Encoder + Generator update ----------------
            set_requires_grad(critic, False)
            set_requires_grad(encoder, True)
            set_requires_grad(generator, True)

            mu, logvar = encoder(real)
            # Deterministic semantic latent used by reconstruction, channel, and DDPM.
            reconstruction = generator(mu, labels)
            z_prior = torch.randn(batch, cfg.z_dim, device=device)
            random_fake = generator(z_prior, labels)

            loss_l1 = F.l1_loss(reconstruction, real)
            with torch.no_grad():
                real_features = critic.extract_features(real)
            recon_features = critic.extract_features(reconstruction)
            loss_feature = F.l1_loss(recon_features, real_features)
            loss_kl = kl_divergence(mu, logvar)
            loss_adv_recon = -critic(reconstruction, labels).mean()
            loss_adv_random = -critic(random_fake, labels).mean()

            # Prevent posterior collapse / nearly deterministic variance.
            posterior_std = torch.exp(0.5 * logvar)
            loss_std_floor = F.relu(cfg.latent_std_floor - posterior_std).mean()

            loss_eg = (
                cfg.lambda_recon_l1 * loss_l1
                + cfg.lambda_recon_perceptual * loss_feature
                + beta_kl * loss_kl
                + cfg.lambda_recon_adv * loss_adv_recon
                + cfg.lambda_random_adv * loss_adv_random
                + 0.1 * loss_std_floor
            )

            optimizer_eg.zero_grad(set_to_none=True)
            loss_eg.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
            optimizer_eg.step()

            sums["d"] += loss_d.item()
            sums["gp"] += gp.item()
            sums["eg"] += loss_eg.item()
            sums["l1"] += loss_l1.item()
            sums["feat"] += loss_feature.item()
            sums["kl"] += loss_kl.item()
            sums["adv_recon"] += loss_adv_recon.item()
            sums["adv_random"] += loss_adv_random.item()
            sums["std"] += posterior_std.mean().item()

            if global_step % 200 == 0:
                print(
                    f"Epoch {epoch + 1}/{cfg.joint_epochs} | Step {global_step} | "
                    f"D={loss_d.item():.4f} | EG={loss_eg.item():.4f} | "
                    f"L1={loss_l1.item():.4f} | Feat={loss_feature.item():.4f} | "
                    f"KL={loss_kl.item():.4f} | beta={beta_kl:.2e}"
                )
            global_step += 1

        n = len(loader)
        row = {"epoch": epoch + 1, "beta_kl": beta_kl}
        row.update({k: v / n for k, v in sums.items()})
        append_csv(history_path, row)

        checkpoint = {
            "epoch": epoch + 1,
            "encoder": encoder.state_dict(),
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_eg": optimizer_eg.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "config": asdict(cfg),
            "reconstruction_loss": row["l1"],
        }
        torch.save(checkpoint, cfg.joint_checkpoint)
        if row["l1"] < best_recon:
            best_recon = row["l1"]
            torch.save(checkpoint, cfg.joint_best_checkpoint)

        # Also export separate checkpoints so the original DDPM/eval loaders remain usable.
        torch.save({"epoch": epoch + 1, "encoder": encoder.state_dict(), "config": asdict(cfg)}, cfg.encoder_checkpoint)
        save_gan_checkpoint(
            cfg.gan_checkpoint, epoch + 1, generator, critic,
            optimizer_eg, optimizer_d, cfg, row["l1"]
        )
        if row["l1"] <= best_recon:
            torch.save({"epoch": epoch + 1, "encoder": encoder.state_dict(), "config": asdict(cfg)}, cfg.encoder_best_checkpoint)
            save_gan_checkpoint(
                cfg.gan_best_checkpoint, epoch + 1, generator, critic,
                optimizer_eg, optimizer_d, cfg, row["l1"]
            )

        # Save visual results every epoch.
        # For each comparison image, the first half contains original images
        # and the second half contains their corresponding reconstructions.
        if (epoch + 1) % cfg.joint_save_every == 0:
            encoder.eval()
            generator.eval()
            with torch.no_grad():
                mu_e, _ = encoder(fixed_e)
                mu_g, _ = encoder(fixed_g)

                # Deterministic reconstruction: Generator receives mu directly.
                recon_e = generator(mu_e, fixed_e_labels)
                recon_g = generator(mu_g, fixed_g_labels)

                # Keep random GAN samples for checking generation ability.
                random_e = generator(fixed_z, fixed_e_labels)
                random_g = generator(fixed_z, fixed_g_labels)

            # Top half: originals; bottom half: reconstructions.
            compare_e = torch.cat([fixed_e, recon_e], dim=0)
            compare_g = torch.cat([fixed_g, recon_g], dim=0)

            save_image(
                (compare_e.clamp(-1, 1) + 1.0) / 2.0,
                os.path.join(
                    cfg.save_root,
                    f"joint_compare_original_recon_eurosat_ep{epoch + 1:03d}.png",
                ),
                nrow=8,
            )
            save_image(
                (compare_g.clamp(-1, 1) + 1.0) / 2.0,
                os.path.join(
                    cfg.save_root,
                    f"joint_compare_original_recon_deepglobe_ep{epoch + 1:03d}.png",
                ),
                nrow=8,
            )

            save_image(
                (random_e.clamp(-1, 1) + 1.0) / 2.0,
                os.path.join(cfg.save_root, f"joint_random_eurosat_ep{epoch + 1:03d}.png"),
                nrow=8,
            )
            save_image(
                (random_g.clamp(-1, 1) + 1.0) / 2.0,
                os.path.join(cfg.save_root, f"joint_random_deepglobe_ep{epoch + 1:03d}.png"),
                nrow=8,
            )

            # Return to training mode for the next epoch.
            encoder.train()
            generator.train()

        print(
            f"[Joint] Epoch {epoch + 1}: D={row['d']:.4f}, EG={row['eg']:.4f}, "
            f"L1={row['l1']:.4f}, Feat={row['feat']:.4f}, KL={row['kl']:.4f}, "
            f"posterior_std={row['std']:.4f}"
        )


# ============================================================
# Stage 1: Deep Conditional WGAN-GP
# ============================================================


def train_gan(cfg: Config, device: torch.device) -> None:
    print("\n========== Stage 1: Deep Conditional WGAN-GP ==========")
    loader, eurosat, deepglobe = build_combined_loader(cfg, cfg.gan_batch_size)

    generator = DeepConditionalGenerator(cfg).to(device)
    critic = DeepConditionalCritic(cfg).to(device)
    generator.apply(init_weights)
    critic.apply(init_weights)

    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2)
    )
    optimizer_d = torch.optim.Adam(
        critic.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2)
    )

    print(f"Device: {device}")
    print(f"EuroSAT: {len(eurosat)}, DeepGlobe: {len(deepglobe)}")
    print(f"Steps per epoch: {len(loader)}")
    print(f"Generator parameters: {count_parameters(generator):,}")
    print(f"Critic parameters: {count_parameters(critic):,}")

    fixed_z = torch.randn(64, cfg.z_dim, device=device)
    fixed_e_labels = torch.full((64,), EUROSAT_LABEL, device=device, dtype=torch.long)
    fixed_g_labels = torch.full((64,), DEEPGLOBE_LABEL, device=device, dtype=torch.long)

    history_path = os.path.join(cfg.save_root, "gan_training_history.csv")
    best_generator_loss = float("inf")
    global_step = 0

    for epoch in range(cfg.gan_epochs):
        generator.train()
        critic.train()
        epoch_d = 0.0
        epoch_g = 0.0
        epoch_gp = 0.0
        g_updates = 0

        for real, labels in loader:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            batch = real.size(0)

            # Critic update
            set_requires_grad(critic, True)
            set_requires_grad(generator, False)
            z = torch.randn(batch, cfg.z_dim, device=device)
            with torch.no_grad():
                fake = generator(z, labels)

            d_real = critic(real, labels)
            d_fake = critic(fake, labels)
            gp = gradient_penalty(critic, real, fake, labels)
            loss_d = d_fake.mean() - d_real.mean() + cfg.lambda_gp * gp

            optimizer_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optimizer_d.step()

            epoch_d += loss_d.item()
            epoch_gp += gp.item()

            # Generator update every n_critic steps
            if (global_step + 1) % cfg.n_critic == 0:
                set_requires_grad(critic, False)
                set_requires_grad(generator, True)
                z = torch.randn(batch, cfg.z_dim, device=device)
                fake = generator(z, labels)
                loss_g = -critic(fake, labels).mean()

                optimizer_g.zero_grad(set_to_none=True)
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
                optimizer_g.step()

                epoch_g += loss_g.item()
                g_updates += 1
            else:
                loss_g = torch.zeros((), device=device)

            if global_step % 200 == 0:
                print(
                    f"Epoch {epoch + 1}/{cfg.gan_epochs} | Step {global_step} | "
                    f"D={loss_d.item():.4f} | G={loss_g.item():.4f} | GP={gp.item():.4f}"
                )
            global_step += 1

        avg_d = epoch_d / len(loader)
        avg_gp = epoch_gp / len(loader)
        avg_g = epoch_g / max(g_updates, 1)

        append_csv(
            history_path,
            {
                "epoch": epoch + 1,
                "critic_loss": avg_d,
                "generator_loss": avg_g,
                "gradient_penalty": avg_gp,
            },
        )

        save_gan_checkpoint(
            cfg.gan_checkpoint,
            epoch + 1,
            generator,
            critic,
            optimizer_g,
            optimizer_d,
            cfg,
            avg_g,
        )

        if avg_g < best_generator_loss:
            best_generator_loss = avg_g
            save_gan_checkpoint(
                cfg.gan_best_checkpoint,
                epoch + 1,
                generator,
                critic,
                optimizer_g,
                optimizer_d,
                cfg,
                avg_g,
            )

        if (epoch + 1) % cfg.gan_save_every == 0 or epoch == 0:
            generator.eval()
            with torch.no_grad():
                sample_e = generator(fixed_z, fixed_e_labels)
                sample_g = generator(fixed_z, fixed_g_labels)
            save_image(
                (sample_e + 1.0) / 2.0,
                os.path.join(cfg.save_root, f"gan_random_eurosat_ep{epoch + 1}.png"),
                nrow=8,
            )
            save_image(
                (sample_g + 1.0) / 2.0,
                os.path.join(cfg.save_root, f"gan_random_deepglobe_ep{epoch + 1}.png"),
                nrow=8,
            )

        print(
            f"[GAN] Epoch {epoch + 1}: D={avg_d:.4f}, G={avg_g:.4f}, GP={avg_gp:.4f}"
        )


# ============================================================
# Stage 2: VAE Encoder -> frozen deep Generator
# ============================================================


def train_encoder(cfg: Config, device: torch.device) -> None:
    print("\n========== Stage 2: VAE Encoder -> Frozen Generator ==========")
    loader, eurosat, deepglobe = build_combined_loader(cfg, cfg.encoder_batch_size)
    generator, critic = load_gan_models(cfg, device)
    encoder = SharedVAEEncoder(cfg.z_dim).to(device)
    encoder.apply(init_weights)

    generator.eval()
    critic.eval()
    set_requires_grad(generator, False)
    set_requires_grad(critic, False)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.lr_encoder, betas=(0.9, 0.999))
    history_path = os.path.join(cfg.save_root, "encoder_training_history.csv")
    best_recon = float("inf")

    fixed_count = min(32, len(eurosat), len(deepglobe))
    fixed_e = torch.stack([eurosat[i][0] for i in range(fixed_count)]).to(device)
    fixed_g = torch.stack([deepglobe[i][0] for i in range(fixed_count)]).to(device)
    fixed_e_labels = torch.full((fixed_count,), EUROSAT_LABEL, device=device, dtype=torch.long)
    fixed_g_labels = torch.full((fixed_count,), DEEPGLOBE_LABEL, device=device, dtype=torch.long)

    for epoch in range(cfg.encoder_epochs):
        encoder.train()
        beta_kl = cfg.beta_kl_max * min(
            1.0, (epoch + 1) / max(cfg.kl_warmup_epochs, 1)
        )
        total_sum = recon_sum = kl_sum = adv_sum = 0.0

        for real, labels in loader:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            mu, logvar = encoder(real)
            reconstruction = generator(mu, labels)

            loss_recon = F.l1_loss(reconstruction, real)
            loss_kl = kl_divergence(mu, logvar)
            # Very small realism term; Critic stays frozen.
            loss_adv = -critic(reconstruction, labels).mean()
            loss_total = (
                loss_recon
                + beta_kl * loss_kl
                + cfg.lambda_encoder_adv * loss_adv
            )

            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
            optimizer.step()

            total_sum += loss_total.item()
            recon_sum += loss_recon.item()
            kl_sum += loss_kl.item()
            adv_sum += loss_adv.item()

        n = len(loader)
        avg_total = total_sum / n
        avg_recon = recon_sum / n
        avg_kl = kl_sum / n
        avg_adv = adv_sum / n

        checkpoint = {
            "epoch": epoch + 1,
            "encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "reconstruction_loss": avg_recon,
        }
        torch.save(checkpoint, cfg.encoder_checkpoint)
        if avg_recon < best_recon:
            best_recon = avg_recon
            torch.save(checkpoint, cfg.encoder_best_checkpoint)

        append_csv(
            history_path,
            {
                "epoch": epoch + 1,
                "total_loss": avg_total,
                "reconstruction_loss": avg_recon,
                "kl_loss": avg_kl,
                "adversarial_loss": avg_adv,
                "beta_kl": beta_kl,
            },
        )

        if (epoch + 1) % cfg.encoder_save_every == 0 or epoch == 0:
            encoder.eval()
            with torch.no_grad():
                mu_e, _ = encoder(fixed_e)
                mu_g, _ = encoder(fixed_g)
                recon_e = generator(mu_e, fixed_e_labels)
                recon_g = generator(mu_g, fixed_g_labels)
            save_image(
                (recon_e + 1.0) / 2.0,
                os.path.join(cfg.save_root, f"encoder_recon_eurosat_ep{epoch + 1}.png"),
                nrow=8,
            )
            save_image(
                (recon_g + 1.0) / 2.0,
                os.path.join(cfg.save_root, f"encoder_recon_deepglobe_ep{epoch + 1}.png"),
                nrow=8,
            )

        print(
            f"[Encoder] Epoch {epoch + 1}: Total={avg_total:.4f}, "
            f"Recon={avg_recon:.4f}, KL={avg_kl:.4f}, Adv={avg_adv:.4f}"
        )


# ============================================================
# Stage 3: Channel-conditioned latent DDPM
# ============================================================


def train_ddpm(
    cfg: Config,
    device: torch.device,
) -> None:
    print(
        "\n========== Stage 3: RANDOM-ALPHA SISO-MAC -> LABEL-CONDITIONED DDPM =========="
    )

    print(
        f"Using fixed global latent scale c = {cfg.global_latent_scale:.8f}"
    )

    print(
        f"Diffusion horizon = {cfg.diffusion_steps} steps; "
        f"training t=0..{cfg.diffusion_steps - 1}; "
        f"sampling {cfg.diffusion_steps - 1}..0"
    )

    euro_loader, deep_loader, _, _ = build_paired_loaders(
        cfg,
        cfg.ddpm_batch_size,
    )

    euro_iter = infinite_loader(
        euro_loader
    )

    deep_iter = infinite_loader(
        deep_loader
    )

    # --------------------------------------------------------
    # Freeze the already-trained VAE-GAN backbone.
    # IMPORTANT:
    # Load the separately exported BEST Encoder and Generator checkpoints
    # from the completed/partial joint VAE-GAN run.  This avoids accidentally
    # loading an older, architecture-incompatible joint checkpoint.
    #
    # Expected files:
    #   deep_wgangp_vae_encoder_best.pth
    #   deep_wgangp_satellite_best.pth
    # --------------------------------------------------------
    encoder = load_encoder(
        cfg,
        device,
    )

    generator, _ = load_gan_models(
        cfg,
        device,
    )

    print(
        f"Loaded DDPM backbone Encoder   : {cfg.encoder_best_checkpoint}"
    )
    print(
        f"Loaded DDPM backbone Generator : {cfg.gan_best_checkpoint}"
    )

    encoder.eval()
    generator.eval()

    set_requires_grad(
        encoder,
        False,
    )

    set_requires_grad(
        generator,
        False,
    )

    # --------------------------------------------------------
    # Fresh label-conditioned DDPM.
    # Do NOT resume an older 2-Rx / 1000-step DDPM checkpoint because its
    # input dimension and diffusion definition are different.
    # --------------------------------------------------------
    ddpm = ConditionalLatentDDPM(
        cfg
    ).to(device)

    schedule = DiffusionSchedule(
        cfg.diffusion_steps
    ).to(device)

    optimizer = torch.optim.AdamW(
        ddpm.parameters(),
        lr=cfg.lr_ddpm,
        weight_decay=1e-4,
    )

    history_path = os.path.join(
        cfg.save_root,
        "ddpm_label_cond_siso_random_alpha_training_history.csv",
    )

    best_loss = float("inf")
    global_step = 0

    for epoch in range(
        cfg.ddpm_epochs
    ):
        ddpm.train()

        total_sum = 0.0
        eps_e_sum = 0.0
        eps_g_sum = 0.0
        x0_e_sum = 0.0
        x0_g_sum = 0.0
        alpha_e_sum = 0.0
        alpha_g_sum = 0.0

        for _ in range(
            cfg.ddpm_steps_per_epoch
        ):
            batch_e, _ = next(
                euro_iter
            )

            batch_g, _ = next(
                deep_iter
            )

            batch_e = batch_e.to(
                device,
                non_blocking=True,
            )

            batch_g = batch_g.to(
                device,
                non_blocking=True,
            )

            batch = min(
                batch_e.size(0),
                batch_g.size(0),
            )

            batch_e = batch_e[:batch]
            batch_g = batch_g[:batch]

            with torch.no_grad():
                # ----------------------------------------------------
                # Frozen deterministic semantic latents.
                # Generator was trained on raw mu.
                # Channel/DDPM use one fixed globally-scaled version.
                # ----------------------------------------------------
                mu_e, _ = encoder(
                    batch_e
                )

                mu_g, _ = encoder(
                    batch_g
                )

                z_e = apply_global_latent_scale(
                    mu_e,
                    cfg,
                )

                z_g = apply_global_latent_scale(
                    mu_g,
                    cfg,
                )

                # ----------------------------------------------------
                # RANDOM-ALPHA target-centered observations.
                #
                # EuroSAT target:
                #   y_E = z_E + alpha_E * z_D + n_E
                # DeepGlobe target:
                #   y_D = z_D + alpha_D * z_E + n_D
                #
                # alpha_E and alpha_D are independently sampled for
                # every sample. This matches the interference-ratio
                # sweep definition used during evaluation.
                # ----------------------------------------------------
                if epoch < cfg.ddpm_clean_epochs:
                    train_snr_db: torch.Tensor | float = math.inf
                    alpha_low = cfg.ddpm_alpha_clean_min
                    alpha_high = cfg.ddpm_alpha_clean_max
                else:
                    train_snr_db = torch.empty(
                        batch,
                        device=device,
                    ).uniform_(
                        cfg.snr_min_db,
                        cfg.snr_max_db,
                    )
                    alpha_low = cfg.ddpm_alpha_min
                    alpha_high = cfg.ddpm_alpha_max

                alpha_e = torch.empty(
                    batch, device=device
                ).uniform_(alpha_low, alpha_high)

                alpha_g = torch.empty(
                    batch, device=device
                ).uniform_(alpha_low, alpha_high)

                observation_e, _ = target_centered_siso_observation(
                    z_target=z_e,
                    z_interference=z_g,
                    alpha=alpha_e,
                    snr_db=train_snr_db,
                )

                observation_g, _ = target_centered_siso_observation(
                    z_target=z_g,
                    z_interference=z_e,
                    alpha=alpha_g,
                    snr_db=train_snr_db,
                )

            # --------------------------------------------------------
            # FULL diffusion-horizon training.
            # t_E and t_D are independently sampled from 0..399.
            # --------------------------------------------------------
            t_e = torch.randint(
                0,
                cfg.diffusion_steps,
                (batch,),
                device=device,
                dtype=torch.long,
            )

            t_g = torch.randint(
                0,
                cfg.diffusion_steps,
                (batch,),
                device=device,
                dtype=torch.long,
            )

            noise_e = torch.randn_like(
                z_e
            )

            noise_g = torch.randn_like(
                z_g
            )

            zt_e = schedule.q_sample(
                z_e,
                t_e,
                noise_e,
            )

            zt_g = schedule.q_sample(
                z_g,
                t_g,
                noise_g,
            )

            labels_e = torch.full(
                (batch,),
                EUROSAT_LABEL,
                device=device,
                dtype=torch.long,
            )

            labels_g = torch.full(
                (batch,),
                DEEPGLOBE_LABEL,
                device=device,
                dtype=torch.long,
            )

            # --------------------------------------------------------
            # Target-specific random-alpha observation + target label.
            # --------------------------------------------------------
            pred_e = ddpm(
                zt_e,
                t_e,
                labels_e,
                observation_e,
            )

            pred_g = ddpm(
                zt_g,
                t_g,
                labels_g,
                observation_g,
            )

            loss_eps_e = F.mse_loss(
                pred_e,
                noise_e,
            )

            loss_eps_g = F.mse_loss(
                pred_g,
                noise_g,
            )

            pred_x0_e = schedule.predict_x0(
                zt_e,
                t_e,
                pred_e,
            )

            pred_x0_g = schedule.predict_x0(
                zt_g,
                t_g,
                pred_g,
            )

            loss_x0_e = F.mse_loss(
                pred_x0_e,
                z_e,
            )

            loss_x0_g = F.mse_loss(
                pred_x0_g,
                z_g,
            )

            loss_eps = 0.5 * (
                loss_eps_e
                + loss_eps_g
            )

            loss_x0 = 0.5 * (
                loss_x0_e
                + loss_x0_g
            )

            loss = (
                loss_eps
                + cfg.lambda_x0
                * loss_x0
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                ddpm.parameters(),
                cfg.grad_clip,
            )

            optimizer.step()

            total_sum += loss.item()
            eps_e_sum += loss_eps_e.item()
            eps_g_sum += loss_eps_g.item()
            x0_e_sum += loss_x0_e.item()
            x0_g_sum += loss_x0_g.item()
            alpha_e_sum += alpha_e.mean().item()
            alpha_g_sum += alpha_g.mean().item()

            if global_step % 200 == 0:
                if isinstance(
                    train_snr_db,
                    float,
                ):
                    snr_text = "clean"
                else:
                    snr_text = (
                        f"{train_snr_db.mean().item():.1f} dB(mean)"
                    )

                print(
                    f"Epoch {epoch + 1}/{cfg.ddpm_epochs} | "
                    f"Step {global_step} | "
                    f"SNR={snr_text} | "
                    f"alphaE={alpha_e.mean().item():.3f} | "
                    f"alphaD={alpha_g.mean().item():.3f} | "
                    f"Total={loss.item():.4f} | "
                    f"epsE={loss_eps_e.item():.4f} | "
                    f"epsG={loss_eps_g.item():.4f} | "
                    f"x0E={loss_x0_e.item():.4f} | "
                    f"x0G={loss_x0_g.item():.4f}"
                )

            global_step += 1

        n = cfg.ddpm_steps_per_epoch

        avg_total = total_sum / n

        row = {
            "epoch":
                epoch + 1,

            "total_loss":
                avg_total,

            "eps_eurosat":
                eps_e_sum / n,

            "eps_deepglobe":
                eps_g_sum / n,

            "x0_eurosat":
                x0_e_sum / n,

            "x0_deepglobe":
                x0_g_sum / n,

            "alpha_eurosat_mean":
                alpha_e_sum / n,

            "alpha_deepglobe_mean":
                alpha_g_sum / n,

            "global_latent_scale":
                float(
                    cfg.global_latent_scale
                ),

            "diffusion_steps":
                cfg.diffusion_steps,
        }

        append_csv(
            history_path,
            row,
        )

        checkpoint = {
            "epoch":
                epoch + 1,

            "ddpm":
                ddpm.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "config":
                asdict(cfg),

            "loss":
                avg_total,

            "global_latent_scale":
                float(
                    cfg.global_latent_scale
                ),

            "latent_definition":
                "z_target = global_latent_scale * encoder_mu",

            "channel_definition":
                "y_target = z_target + alpha*z_interference + AWGN",

            "alpha_training_definition":
                (
                    f"clean epochs: U({cfg.ddpm_alpha_clean_min}, {cfg.ddpm_alpha_clean_max}); "
                    f"noisy epochs: U({cfg.ddpm_alpha_min}, {cfg.ddpm_alpha_max}); "
                    "independent per sample and target"
                ),

            "separator_definition":
                "(z_t, t, y, target_label) -> epsilon_target",

            "diffusion_training_range":
                f"0..{cfg.diffusion_steps - 1}",
        }

        torch.save(
            checkpoint,
            cfg.ddpm_checkpoint,
        )

        if avg_total < best_loss:
            best_loss = avg_total

            torch.save(
                checkpoint,
                cfg.ddpm_best_checkpoint,
            )

        print(
            f"[DDPM] Epoch {epoch + 1}: "
            f"Total={avg_total:.4f}, "
            f"epsE={row['eps_eurosat']:.4f}, "
            f"epsG={row['eps_deepglobe']:.4f}, "
            f"x0E={row['x0_eurosat']:.4f}, "
            f"x0G={row['x0_deepglobe']:.4f}, "
            f"alphaE={row['alpha_eurosat_mean']:.3f}, "
            f"alphaD={row['alpha_deepglobe_mean']:.3f}"
        )


# ============================================================
# Evaluation
# ============================================================


@torch.no_grad()
def evaluate_pipeline(
    cfg: Config,
    device: torch.device,
) -> None:
    print(
        "\n========== Evaluation: separate BEST Encoder/GAN -> raw mu -> GLOBAL SCALE -> "
        "SISO-MAC -> LABEL-CONDITIONED DDPM -> inverse scale -> Generator =========="
    )

    print(
        f"Using fixed global latent scale c = {cfg.global_latent_scale:.8f}"
    )

    print(
        f"DDPM train/sample horizon = 0..{cfg.diffusion_steps - 1}"
    )

    euro_loader, deep_loader, _, _ = build_paired_loaders(
        cfg,
        cfg.eval_count,
    )

    images_e, _ = next(
        iter(
            euro_loader
        )
    )

    images_g, _ = next(
        iter(
            deep_loader
        )
    )

    count = min(
        cfg.eval_count,
        images_e.size(0),
        images_g.size(0),
    )

    images_e = images_e[:count].to(
        device
    )

    images_g = images_g[:count].to(
        device
    )

    # --------------------------------------------------------
    # IMPORTANT FOR EVAL:
    # Do NOT load the old/mismatched "joint" checkpoint here.
    #
    # The completed joint VAE-GAN training exported the best Encoder and
    # Generator separately:
    #
    #   deep_wgangp_vae_encoder_best.pth
    #   deep_wgangp_satellite_best.pth
    #
    # These are the checkpoints used for the semantic backbone at evaluation.
    # This avoids the architecture mismatch caused by loading an older
    # deep_wgangp_deep_vae_gan_joint_best.pth checkpoint.
    # --------------------------------------------------------
    encoder = load_encoder(
        cfg,
        device,
    )

    generator, _ = load_gan_models(
        cfg,
        device,
    )

    # Load the already-trained label-conditioned SISO DDPM.
    ddpm, schedule = load_ddpm(
        cfg,
        device,
    )

    print(
        "Loaded evaluation backbone:"
    )
    print(
        f"  Encoder   : {cfg.encoder_best_checkpoint}"
    )
    print(
        f"  Generator : {cfg.gan_best_checkpoint}"
    )
    print(
        f"  DDPM      : {cfg.ddpm_best_checkpoint}"
    )

    encoder.eval()
    generator.eval()
    ddpm.eval()

    # --------------------------------------------------------
    # Raw semantic latents.
    # --------------------------------------------------------
    mu_e, _ = encoder(
        images_e
    )

    mu_g, _ = encoder(
        images_g
    )

    z_e = apply_global_latent_scale(
        mu_e,
        cfg,
    )

    z_g = apply_global_latent_scale(
        mu_g,
        cfg,
    )

    # --------------------------------------------------------
    # Single received mixture shared by BOTH label queries.
    # --------------------------------------------------------
    observation, clean_mix = siso_mac_observation(
        z_e,
        z_g,
        cfg.eval_snr_db,
    )

    labels_e = torch.full(
        (count,),
        EUROSAT_LABEL,
        device=device,
        dtype=torch.long,
    )

    labels_g = torch.full(
        (count,),
        DEEPGLOBE_LABEL,
        device=device,
        dtype=torch.long,
    )

    # Use the SAME initial Gaussian sample for both target queries so that
    # the target label/conditioning, rather than an unrelated initial seed,
    # is the major controlled difference between the two recoveries.
    common_initial_noise = torch.randn(
        count,
        cfg.z_dim,
        device=device,
    )

    ddpm_e_scaled = sample_conditional_ddpm(
        model=ddpm,
        schedule=schedule,
        observation=observation,
        labels=labels_e,
        z_dim=cfg.z_dim,
        initial_noise=common_initial_noise.clone(),
    )

    ddpm_g_scaled = sample_conditional_ddpm(
        model=ddpm,
        schedule=schedule,
        observation=observation,
        labels=labels_g,
        z_dim=cfg.z_dim,
        initial_noise=common_initial_noise.clone(),
    )

    # --------------------------------------------------------
    # Restore the raw-mu scale expected by the Generator.
    # --------------------------------------------------------
    ddpm_e_mu = invert_global_latent_scale(
        ddpm_e_scaled,
        cfg,
    )

    ddpm_g_mu = invert_global_latent_scale(
        ddpm_g_scaled,
        cfg,
    )

    # No-separation baseline:
    # feed the received mixture itself to the requested target decoder.
    no_sep_mu = invert_global_latent_scale(
        observation,
        cfg,
    )

    # Clean ceiling.
    clean_recon_e = generator(
        mu_e,
        labels_e,
    )

    clean_recon_g = generator(
        mu_g,
        labels_g,
    )

    no_sep_recon_e = generator(
        no_sep_mu,
        labels_e,
    )

    no_sep_recon_g = generator(
        no_sep_mu,
        labels_g,
    )

    ddpm_recon_e = generator(
        ddpm_e_mu,
        labels_e,
    )

    ddpm_recon_g = generator(
        ddpm_g_mu,
        labels_g,
    )

    eval_dir = os.path.join(
        cfg.save_root,
        f"eval_label_cond_siso_snr{cfg.eval_snr_db:g}",
    )

    os.makedirs(
        eval_dir,
        exist_ok=True,
    )

    outputs = {
        "real_eurosat.png":
            images_e,

        "real_deepglobe.png":
            images_g,

        "clean_raw_mu_recon_eurosat.png":
            clean_recon_e,

        "clean_raw_mu_recon_deepglobe.png":
            clean_recon_g,

        "no_separation_recon_eurosat.png":
            no_sep_recon_e,

        "no_separation_recon_deepglobe.png":
            no_sep_recon_g,

        "label_ddpm_recon_eurosat.png":
            ddpm_recon_e,

        "label_ddpm_recon_deepglobe.png":
            ddpm_recon_g,
    }

    for name, tensor in outputs.items():
        save_image(
            (
                tensor.clamp(
                    -1,
                    1,
                )
                + 1.0
            )
            / 2.0,
            os.path.join(
                eval_dir,
                name,
            ),
            nrow=8,
        )

    # --------------------------------------------------------
    # Latent-space diagnostics.
    # --------------------------------------------------------
    latent_mse_e = F.mse_loss(
        ddpm_e_scaled,
        z_e,
    ).item()

    latent_mse_g = F.mse_loss(
        ddpm_g_scaled,
        z_g,
    ).item()

    latent_cos_e = F.cosine_similarity(
        ddpm_e_scaled,
        z_e,
        dim=1,
    ).mean().item()

    latent_cos_g = F.cosine_similarity(
        ddpm_g_scaled,
        z_g,
        dim=1,
    ).mean().item()

    # --------------------------------------------------------
    # Image-space diagnostics.
    # --------------------------------------------------------
    l1_clean_e = F.l1_loss(
        clean_recon_e,
        images_e,
    ).item()

    l1_clean_g = F.l1_loss(
        clean_recon_g,
        images_g,
    ).item()

    l1_no_sep_e = F.l1_loss(
        no_sep_recon_e,
        images_e,
    ).item()

    l1_no_sep_g = F.l1_loss(
        no_sep_recon_g,
        images_g,
    ).item()

    l1_ddpm_e = F.l1_loss(
        ddpm_recon_e,
        images_e,
    ).item()

    l1_ddpm_g = F.l1_loss(
        ddpm_recon_g,
        images_g,
    ).item()

    # --------------------------------------------------------
    # Zero-condition sanity test:
    # if output barely changes, model may be relying mainly on label/prior.
    # --------------------------------------------------------
    zero_observation = torch.zeros_like(
        observation
    )

    zero_e_scaled = sample_conditional_ddpm(
        model=ddpm,
        schedule=schedule,
        observation=zero_observation,
        labels=labels_e,
        z_dim=cfg.z_dim,
        initial_noise=common_initial_noise.clone(),
    )

    zero_g_scaled = sample_conditional_ddpm(
        model=ddpm,
        schedule=schedule,
        observation=zero_observation,
        labels=labels_g,
        z_dim=cfg.z_dim,
        initial_noise=common_initial_noise.clone(),
    )

    condition_effect_e = F.mse_loss(
        zero_e_scaled,
        ddpm_e_scaled,
    ).item()

    condition_effect_g = F.mse_loss(
        zero_g_scaled,
        ddpm_g_scaled,
    ).item()

    metrics = {
        "snr_db":
            float(
                cfg.eval_snr_db
            ),

        "global_latent_scale":
            float(
                cfg.global_latent_scale
            ),

        "latent_mse_eurosat":
            latent_mse_e,

        "latent_mse_deepglobe":
            latent_mse_g,

        "latent_cos_eurosat":
            latent_cos_e,

        "latent_cos_deepglobe":
            latent_cos_g,

        "l1_clean_eurosat":
            l1_clean_e,

        "l1_clean_deepglobe":
            l1_clean_g,

        "l1_no_separation_eurosat":
            l1_no_sep_e,

        "l1_no_separation_deepglobe":
            l1_no_sep_g,

        "l1_ddpm_eurosat":
            l1_ddpm_e,

        "l1_ddpm_deepglobe":
            l1_ddpm_g,

        "zero_condition_output_change_eurosat":
            condition_effect_e,

        "zero_condition_output_change_deepglobe":
            condition_effect_g,
    }

    metrics_path = os.path.join(
        eval_dir,
        "metrics.csv",
    )

    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                metrics.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            metrics
        )

    print(
        f"Saved evaluation outputs to: {eval_dir}"
    )

    print(
        f"EuroSAT | latent MSE={latent_mse_e:.5f}, "
        f"cos={latent_cos_e:.4f} | "
        f"L1 clean={l1_clean_e:.4f}, "
        f"no-sep={l1_no_sep_e:.4f}, "
        f"DDPM={l1_ddpm_e:.4f}"
    )

    print(
        f"DeepGlobe | latent MSE={latent_mse_g:.5f}, "
        f"cos={latent_cos_g:.4f} | "
        f"L1 clean={l1_clean_g:.4f}, "
        f"no-sep={l1_no_sep_g:.4f}, "
        f"DDPM={l1_ddpm_g:.4f}"
    )

    print(
        "Zero-condition output change "
        f"| EuroSAT={condition_effect_e:.6f}, "
        f"DeepGlobe={condition_effect_g:.6f}"
    )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("pretrain", "joint", "hybrid", "gan", "encoder", "ddpm", "eval", "all"),
        default="all",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--joint-batch-size", type=int, default=None)
    parser.add_argument("--gan-batch-size", type=int, default=None)
    parser.add_argument("--gan-pretrain-epochs", type=int, default=None)
    parser.add_argument("--joint-epochs", type=int, default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=None)
    parser.add_argument("--ddpm-batch-size", type=int, default=None)
    parser.add_argument("--eval-snr-db", type=float, default=None)
    parser.add_argument(
        "--global-latent-scale",
        type=float,
        default=None,
        help=(
            "Override fixed global latent scale. "
            "Default is the training-set calibrated value in Config."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    if args.device is not None:
        cfg.device = args.device
    if args.joint_batch_size is not None:
        cfg.joint_batch_size = args.joint_batch_size
    if args.gan_batch_size is not None:
        cfg.gan_batch_size = args.gan_batch_size
    if args.gan_pretrain_epochs is not None:
        cfg.gan_pretrain_epochs = args.gan_pretrain_epochs
    if args.joint_epochs is not None:
        cfg.joint_epochs = args.joint_epochs
    if args.encoder_batch_size is not None:
        cfg.encoder_batch_size = args.encoder_batch_size
    if args.ddpm_batch_size is not None:
        cfg.ddpm_batch_size = args.ddpm_batch_size
    if args.eval_snr_db is not None:
        cfg.eval_snr_db = args.eval_snr_db
    if args.global_latent_scale is not None:
        cfg.global_latent_scale = args.global_latent_scale

    ensure_dirs(cfg)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    print("Configuration:")
    print(f"  device={device}")
    print(f"  z_dim={cfg.z_dim}")
    print(f"  G widths={cfg.g_widths}, blocks={cfg.g_blocks}")
    print(f"  D widths={cfg.d_widths}, blocks={cfg.d_blocks}")
    print(f"  global_latent_scale={cfg.global_latent_scale:.8f}")
    print(f"  DDPM mode=label-conditioned SISO latent MAC")
    print(f"  diffusion_steps={cfg.diffusion_steps} (train/sample full horizon)")

    if args.stage in ("pretrain", "hybrid", "all"):
        pretrain_gan(cfg, device)
    if args.stage in ("joint", "hybrid", "all"):
        train_joint_vae_gan(cfg, device)
    if args.stage == "gan":
        train_gan(cfg, device)
    if args.stage == "encoder":
        train_encoder(cfg, device)
    if args.stage in ("ddpm", "all"):
        train_ddpm(cfg, device)
    if args.stage == "eval":
        evaluate_pipeline(cfg, device)


if __name__ == "__main__":
    main()
