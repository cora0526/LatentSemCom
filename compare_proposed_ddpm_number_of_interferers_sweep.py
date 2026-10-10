"""
Interference-strength robustness experiment for the proposed VAE-DDPM-GAN system.

At a fixed desired-signal SNR, sweep the interferer amplitude alpha.
For each target dataset:

    EuroSAT target:  y_E = z_E + alpha*z_D + n_E
    DeepGlobe target: y_D = z_D + alpha*z_E + n_D

Noise variance is referenced to the desired latent power, so changing alpha
changes only interference strength rather than noise power.

Three conditions are compared separately for EuroSAT and DeepGlobe:
1) Clean Latent
2) No Separation
3) Proposed DDPM

Outputs:
    interference_strength_metrics.csv
    EuroSAT/DeepGlobe PSNR and SSIM vs Interference (.png/.pdf)
    EuroSAT/DeepGlobe Recovery Ratio vs Interference (.png/.pdf)
    qualitative_alpha*.png
"""

import csv
import glob
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


# =========================================================
# Configuration
# =========================================================

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    z_dim: int = 256
    label_dim: int = 64
    num_classes: int = 2
    diffusion_steps: int = 400

    # DDIM inference
    sampling_method: str = "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0

    batch_size: int = 16
    num_workers: int = 0
    max_test_pairs: int = 500

    # =====================================================
    # Experiment settings
    # =====================================================

    # Desired-signal SNR is fixed for the entire experiment.
    fixed_snr_db: float = 10.0

    # X-axis: interference amplitude alpha.
    # interference_levels: Tuple[float, ...] = (
    #     0.25,
    #     0.50,
    #     0.75,
    #     1.00,
    #     1.25,
    #     1.50,
    #     2.00,
    # )

    interference_levels: Tuple[float, ...] = tuple(
        round(0.60 + 0.05 * index, 2)
        for index in range(29)
    )

    # Five curves: K = 1, 2, 3, 4, 5 interferers.
    num_interferers: Tuple[int, ...] = (1, 2, 3, 4, 5)

    # IMPORTANT:
    # The same target image is used for every K and alpha.
    # Interferers are different images from the opposite dataset.
    #
    # Each interferer is rescaled so that its RMS latent power matches
    # the target latent power on a per-sample basis.
    # Therefore every interferer contributes the same average power
    # before multiplication by alpha.

    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    scratch_vaegan_checkpoint: str = (
        "./save_model/vae_gan_from_scratch_best.pth"
    )

    scratch_vaegan_ddpm_checkpoint: str = (
        "./save_model/DDPM_vaegan_scratch_latent_separator_best.pth"
    )

    save_root: str = (
        "./save_data/proposed_ddpm_number_of_interferers_sweep"
    )

    # Save example reconstructions for selected alpha values.
    qualitative_alphas: Tuple[float, ...] = (
        0.50,
        1.00,
        1.50,
        2.00,
    )
    qualitative_count: int = 8


# =========================================================
# Dataset
# =========================================================

class SatelliteDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform

        if not self.files:
            raise RuntimeError(
                f"No matching DeepGlobe images found in:\n{root}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        image = Image.open(self.files[index]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


def get_image(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


# =========================================================
# Proposed VAE-GAN
# =========================================================

class SharedVAEEncoder(nn.Module):
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

        feature_dim = 512 * 4 * 4
        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim)

    def forward(self, x: torch.Tensor):
        h = self.features(x).flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 10.0)
        return mu, logvar


class ScratchConditionalGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        num_classes: int = 2,
        label_dim: int = 64,
    ):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, label_dim)

        self.fc = nn.Sequential(
            nn.Linear(z_dim + label_dim, 512 * 4 * 4),
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

    def forward(self, z: torch.Tensor, labels: torch.Tensor):
        label_feature = self.label_embedding(labels)
        h = self.fc(torch.cat([z, label_feature], dim=1))
        h = h.view(z.size(0), 512, 4, 4)
        return self.decoder(h)


# =========================================================
# DDPM separator
# =========================================================

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor):
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)

        frequencies = torch.exp(
            torch.arange(
                half,
                device=timesteps.device,
                dtype=torch.float32,
            )
            * -scale
        )

        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class LatentConditionalDDPM(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        num_classes: int = 2,
        time_dim: int = 128,
        label_dim: int = 64,
        hidden_dim: int = 768,
        num_blocks: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.label_embedding = nn.Embedding(num_classes, label_dim)

        input_dim = z_dim + z_dim + time_dim + label_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(hidden_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        cond: torch.Tensor,
    ):
        t_emb = self.time_embedding(timesteps)
        y_emb = self.label_embedding(labels)

        h = torch.cat(
            [z_t, cond, t_emb, y_emb],
            dim=1,
        )

        return self.output_layer(
            self.blocks(
                self.input_layer(h)
            )
        )


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        timesteps: int = 400,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()

        betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=torch.float32,
        )

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        alpha_bars_prev = F.pad(
            alpha_bars[:-1],
            (1, 0),
            value=1.0,
        )

        posterior_variance = (
            betas
            * (1.0 - alpha_bars_prev)
            / (1.0 - alpha_bars)
        )

        self.timesteps = timesteps

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)

        self.register_buffer(
            "sqrt_recip_alpha_bars",
            torch.sqrt(1.0 / alpha_bars),
        )

        self.register_buffer(
            "sqrt_recipm1_alpha_bars",
            torch.sqrt(1.0 / alpha_bars - 1.0),
        )

        self.register_buffer(
            "posterior_variance",
            torch.clamp(
                posterior_variance,
                min=1e-20,
            ),
        )

    @staticmethod
    def extract(
        values: torch.Tensor,
        t: torch.Tensor,
        x_shape,
    ):
        out = values.gather(0, t)
        return out.reshape(
            t.size(0),
            *(1,) * (len(x_shape) - 1),
        )

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ):
        recip = self.extract(
            self.sqrt_recip_alpha_bars,
            t,
            x_t.shape,
        )

        recipm1 = self.extract(
            self.sqrt_recipm1_alpha_bars,
            t,
            x_t.shape,
        )

        return recip * x_t - recipm1 * eps


# =========================================================
# Utility
# =========================================================

def freeze(model: nn.Module):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def require_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required checkpoint does not exist:\n{path}"
        )


def load_models(config: Config, device: str):
    require_file(config.scratch_vaegan_checkpoint)
    require_file(config.scratch_vaegan_ddpm_checkpoint)

    vaegan_checkpoint = torch.load(
        config.scratch_vaegan_checkpoint,
        map_location=device,
    )

    if (
        "encoder" not in vaegan_checkpoint
        or "generator" not in vaegan_checkpoint
    ):
        raise KeyError(
            "VAE-GAN checkpoint must contain 'encoder' and 'generator'."
        )

    encoder = SharedVAEEncoder(
        z_dim=config.z_dim,
    ).to(device)

    generator = ScratchConditionalGenerator(
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
    ).to(device)

    encoder.load_state_dict(
        vaegan_checkpoint["encoder"],
        strict=True,
    )

    generator.load_state_dict(
        vaegan_checkpoint["generator"],
        strict=True,
    )

    ddpm_checkpoint = torch.load(
        config.scratch_vaegan_ddpm_checkpoint,
        map_location=device,
    )

    ddpm_state = ddpm_checkpoint.get(
        "model",
        ddpm_checkpoint,
    )

    ddpm = LatentConditionalDDPM(
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
    ).to(device)

    ddpm.load_state_dict(
        ddpm_state,
        strict=True,
    )

    saved_steps = ddpm_checkpoint.get(
        "config",
        {},
    ).get(
        "diffusion_steps",
        config.diffusion_steps,
    )

    if int(saved_steps) != config.diffusion_steps:
        raise ValueError(
            f"DDPM checkpoint used diffusion_steps={saved_steps}, "
            f"but Config.diffusion_steps={config.diffusion_steps}."
        )

    schedule = DiffusionSchedule(
        timesteps=config.diffusion_steps,
    ).to(device)

    freeze(encoder)
    freeze(generator)
    freeze(ddpm)

    print(
        "Loaded VAE-GAN epoch =",
        vaegan_checkpoint.get("epoch", "unknown"),
    )
    print(
        "Loaded DDPM epoch =",
        ddpm_checkpoint.get("epoch", "unknown"),
    )

    return encoder, generator, ddpm, schedule


def add_awgn_relative_to_reference(
    signal: torch.Tensor,
    reference: torch.Tensor,
    snr_db: float,
    noise_unit: torch.Tensor,
):
    """Keep desired-signal SNR fixed while alpha changes."""
    if math.isinf(float(snr_db)):
        return signal

    reference_power = (
        reference.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    )
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_std = torch.sqrt(reference_power / snr_linear)
    return signal + noise_unit * noise_std


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    schedule: DiffusionSchedule,
    cond: torch.Tensor,
    labels: torch.Tensor,
    z_dim: int,
    seed: int,
    sampling_method: str = "ddim",
    ddim_steps: int = 50,
    ddim_eta: float = 0.0,
    clip_x0: float | None = None,
):
    model.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = torch.randn(
        cond.size(0),
        z_dim,
        device=cond.device,
    )

    method = sampling_method.lower()

    if method == "ddpm":
        for step in reversed(range(schedule.timesteps)):
            t = torch.full(
                (cond.size(0),),
                step,
                dtype=torch.long,
                device=cond.device,
            )

            pred_eps = model(
                x,
                t,
                labels,
                cond,
            )

            pred_x0 = schedule.predict_x0_from_eps(
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
                schedule.betas,
                t,
                x.shape,
            )

            alpha_t = schedule.extract(
                schedule.alphas,
                t,
                x.shape,
            )

            alpha_bar_t = schedule.extract(
                schedule.alpha_bars,
                t,
                x.shape,
            )

            alpha_bar_prev = schedule.extract(
                schedule.alpha_bars_prev,
                t,
                x.shape,
            )

            coef_x0 = (
                beta_t
                * torch.sqrt(alpha_bar_prev)
                / (1.0 - alpha_bar_t)
            )

            coef_xt = (
                (1.0 - alpha_bar_prev)
                * torch.sqrt(alpha_t)
                / (1.0 - alpha_bar_t)
            )

            mean = coef_x0 * pred_x0 + coef_xt * x

            if step > 0:
                variance = schedule.extract(
                    schedule.posterior_variance,
                    t,
                    x.shape,
                )

                x = (
                    mean
                    + torch.sqrt(variance)
                    * torch.randn_like(x)
                )
            else:
                x = mean

        return x

    if method != "ddim":
        raise ValueError(
            f"Unknown sampling_method: {sampling_method}"
        )

    steps = max(
        1,
        min(
            int(ddim_steps),
            schedule.timesteps,
        ),
    )

    time_sequence = torch.linspace(
        schedule.timesteps - 1,
        0,
        steps,
        device=cond.device,
    ).round().long()

    time_sequence = torch.unique_consecutive(
        time_sequence
    )

    for index, step_tensor in enumerate(time_sequence):
        step = int(step_tensor.item())

        t = torch.full(
            (cond.size(0),),
            step,
            dtype=torch.long,
            device=cond.device,
        )

        pred_eps = model(
            x,
            t,
            labels,
            cond,
        )

        pred_x0 = schedule.predict_x0_from_eps(
            x,
            t,
            pred_eps,
        )

        if clip_x0 is not None:
            pred_x0 = pred_x0.clamp(
                -clip_x0,
                clip_x0,
            )

        alpha_bar_t = schedule.extract(
            schedule.alpha_bars,
            t,
            x.shape,
        )

        if index == len(time_sequence) - 1:
            x = pred_x0
            continue

        prev_step = int(
            time_sequence[index + 1].item()
        )

        t_prev = torch.full(
            (cond.size(0),),
            prev_step,
            dtype=torch.long,
            device=cond.device,
        )

        alpha_bar_prev = schedule.extract(
            schedule.alpha_bars,
            t_prev,
            x.shape,
        )

        sigma = float(ddim_eta) * torch.sqrt(
            torch.clamp(
                (1.0 - alpha_bar_prev)
                / (1.0 - alpha_bar_t)
                * (
                    1.0
                    - alpha_bar_t / alpha_bar_prev
                ),
                min=0.0,
            )
        )

        direction = torch.sqrt(
            torch.clamp(
                1.0
                - alpha_bar_prev
                - sigma.pow(2),
                min=0.0,
            )
        ) * pred_eps

        if float(ddim_eta) > 0.0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            torch.sqrt(alpha_bar_prev) * pred_x0
            + direction
            + sigma * noise
        )

    return x


# =========================================================
# Metrics
# =========================================================

def to_01(images: torch.Tensor):
    return (
        (images.detach().float() + 1.0) / 2.0
    ).clamp(0.0, 1.0)


def batch_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
):
    pred = to_01(pred)
    target = to_01(target)

    mse = (
        (pred - target)
        .pow(2)
        .flatten(1)
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
):
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

    for prediction, reference in zip(
        pred_np,
        target_np,
    ):
        values.append(
            structural_similarity(
                reference,
                prediction,
                data_range=1.0,
                channel_axis=-1,
            )
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


class MetricAccumulator:
    def __init__(self):
        self.values: Dict[str, List[float]] = {}

    def add(self, name: str, values):
        array = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        self.values.setdefault(
            name,
            [],
        ).extend(array.tolist())

    def mean(self, name: str):
        values = self.values.get(name, [])

        if not values:
            return float("nan")

        return float(np.mean(values))

    def count(self):
        if not self.values:
            return 0

        return len(
            next(iter(self.values.values()))
        )


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()


# =========================================================
# Multiple-interferer utilities
# =========================================================

def match_latent_power(
    z_interferer: torch.Tensor,
    z_target: torch.Tensor,
    eps: float = 1e-8,
):
    """
    Rescale each interferer latent so that its per-sample RMS power
    matches the target latent power.

    This keeps the target latent unchanged while making the
    interferers equal-power relative to that target.
    """
    p_i = z_interferer.pow(2).mean(dim=1, keepdim=True).clamp_min(eps)
    p_t = z_target.pow(2).mean(dim=1, keepdim=True).clamp_min(eps)

    scale = torch.sqrt(p_t / p_i)
    return z_interferer * scale


def build_interferer_bank(
    source_latent: torch.Tensor,
    target_latent: torch.Tensor,
    max_interferers: int,
):
    """
    Build K distinct interferers for every target sample using cyclic
    shifts within the current batch.

    Example for batch index j:
        interferer 1 <- source[j]
        interferer 2 <- source[j+1]
        ...
        interferer K <- source[j+K-1]

    The DataLoader uses drop_last=True and batch_size > max_interferers,
    so the bank contains different source images for each target.
    """
    batch_size = source_latent.size(0)

    if batch_size <= max_interferers:
        raise ValueError(
            f"batch_size={batch_size} must be greater than "
            f"max_interferers={max_interferers}."
        )

    bank = []

    for k in range(max_interferers):
        z_i = torch.roll(
            source_latent,
            shifts=-k,
            dims=0,
        )

        z_i = match_latent_power(
            z_i,
            target_latent,
        )

        bank.append(z_i)

    return bank


@torch.no_grad()
def evaluate_target_with_k_interferers(
    generator: nn.Module,
    ddpm: nn.Module,
    schedule: DiffusionSchedule,
    z_target: torch.Tensor,
    target_labels: torch.Tensor,
    interferer_bank: List[torch.Tensor],
    num_interferers: int,
    alpha: float,
    noise_unit: torch.Tensor,
    sample_seed: int,
    config: Config,
):
    """
    Received latent:

        y = z_t + alpha * sum_{k=1}^{K} z_{i,k} + n

    where each z_{i,k} has been power matched to z_t.

    AWGN is referenced to the desired target latent power, so SNR stays
    fixed at config.fixed_snr_db while K and alpha change.
    """
    interference_sum = torch.zeros_like(z_target)

    for k in range(num_interferers):
        interference_sum = interference_sum + interferer_bank[k]

    mixed_signal = (
        z_target
        + float(alpha) * interference_sum
    )

    received = add_awgn_relative_to_reference(
        signal=mixed_signal,
        reference=z_target,
        snr_db=config.fixed_snr_db,
        noise_unit=noise_unit,
    )

    pred_target = sample_ddpm(
        model=ddpm,
        schedule=schedule,
        cond=received,
        labels=target_labels,
        z_dim=config.z_dim,
        seed=sample_seed,
        sampling_method=config.sampling_method,
        ddim_steps=config.ddim_steps,
        ddim_eta=config.ddim_eta,
    )

    reconstruction = generator(
        pred_target,
        target_labels,
    )

    return {
        "received": received,
        "pred_target": pred_target,
        "reconstruction": reconstruction,
    }


# =========================================================
# Output helpers
# =========================================================

def write_csv(
    path: str,
    rows: List[dict],
):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def save_multi_k_qualitative(
    save_path: str,
    real: torch.Tensor,
    clean: torch.Tensor,
    recon_by_k: Dict[int, torch.Tensor],
    count: int,
):
    """
    Rows:
        Real
        Clean latent reconstruction
        K=1
        K=2
        K=3
        K=4
        K=5
    """
    count = min(count, real.size(0))

    rows = [
        real[:count],
        clean[:count],
    ]

    for k in sorted(recon_by_k):
        rows.append(
            recon_by_k[k][:count]
        )

    grid = make_grid(
        torch.cat(rows, dim=0),
        nrow=count,
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )

    save_image(
        grid,
        save_path,
    )


# =========================================================
# Plotting
# =========================================================

K_COLORS = {
    1: "#4C78A8",
    2: "#72B7B2",
    3: "#F2CF5B",
    4: "#E08B65",
    5: "#8C6BB1",
}

K_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
    4: "D",
    5: "P",
}

K_STYLES = {
    1: "-",
    2: "--",
    3: "-.",
    4: ":",
    5: "-",
}


def plot_k_sweep(
    rows: List[dict],
    dataset_name: str,
    metric_name: str,
    ylabel: str,
    config: Config,
):
    """
    One figure per target dataset.

    X-axis:
        interference amplitude alpha

    Five curves:
        K = 1, 2, 3, 4, 5 interferers
    """
    dataset_key = (
        "eurosat"
        if dataset_name == "EuroSAT"
        else "deepglobe"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5.2)
    )

    x = [
        float(row["interference_alpha"])
        for row in rows
    ]

    for k in config.num_interferers:
        field = (
            f"{dataset_key}_"
            f"k{k}_"
            f"{metric_name}"
        )

        y = [
            float(row[field])
            for row in rows
        ]

        ax.plot(
            x,
            y,
            color=K_COLORS[k],
            linestyle=K_STYLES[k],
            marker=K_MARKERS[k],
            markersize=6,
            linewidth=2.1,
            label=f"{k} Interferer" if k == 1 else f"{k} Interferers",
        )

    ax.set_xlabel(
        r"Interference amplitude $\alpha$",
        fontsize=12,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=12,
    )

    ax.set_xticks(x)

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    # Same scale for EuroSAT and DeepGlobe.
    if metric_name == "psnr":
        ax.set_ylim(10.0, 30.5)
        ax.set_yticks(
            [10, 12, 15, 18, 21, 24, 27, 30]
        )

    elif metric_name == "ssim":
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(
            np.arange(0.0, 1.01, 0.2)
        )

    ax.legend(
        fontsize=8.5,
        frameon=True,
        loc="best",
        ncol=1,
    )

    fig.tight_layout()

    base_name = (
        f"{dataset_name}_"
        f"{metric_name.upper()}_"
        f"vs_NumberOfInterferers"
    )

    png_path = os.path.join(
        config.save_root,
        base_name + ".png",
    )

    pdf_path = os.path.join(
        config.save_root,
        base_name + ".pdf",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", png_path)
    print("Saved:", pdf_path)


# =========================================================
# Main
# =========================================================

def main():
    config = Config()

    if (
        config.device == "cuda"
        and not torch.cuda.is_available()
    ):
        print(
            "CUDA unavailable; switching to CPU."
        )
        device = "cpu"
    else:
        device = config.device

    if config.batch_size <= max(config.num_interferers):
        raise ValueError(
            "Config.batch_size must be greater than the maximum "
            "number of interferers."
        )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(
        config.save_root,
        exist_ok=True,
    )

    print("=" * 90)
    print("Multiple-interferer robustness experiment")
    print(f"Fixed desired-signal SNR = {config.fixed_snr_db} dB")
    print("K =", config.num_interferers)
    print("alpha =", config.interference_levels)
    print(
        f"DDPM sampler = {config.sampling_method}, "
        f"steps={config.ddim_steps}, "
        f"eta={config.ddim_eta}"
    )

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    eurosat = datasets.ImageFolder(
        config.eurosat_root,
        transform=transform,
    )

    deepglobe = SatelliteDataset(
        config.deepglobe_root,
        transform=transform,
    )

    pair_count = min(
        config.max_test_pairs,
        len(eurosat),
        len(deepglobe),
    )

    # drop_last=True is used so every batch has enough distinct
    # interferers to create K=1...5.
    usable_pairs = (
        pair_count // config.batch_size
    ) * config.batch_size

    if usable_pairs <= 0:
        raise RuntimeError(
            "Not enough test pairs for one complete batch."
        )

    print(
        f"Requested pairs = {pair_count}; "
        f"used pairs = {usable_pairs}"
    )

    subset_generator = (
        torch.Generator()
        .manual_seed(config.seed)
    )

    e_indices = torch.randperm(
        len(eurosat),
        generator=subset_generator,
    )[:usable_pairs].tolist()

    d_indices = torch.randperm(
        len(deepglobe),
        generator=subset_generator,
    )[:usable_pairs].tolist()

    loader_e = DataLoader(
        Subset(
            eurosat,
            e_indices,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    loader_d = DataLoader(
        Subset(
            deepglobe,
            d_indices,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    (
        encoder,
        generator,
        ddpm,
        schedule,
    ) = load_models(
        config,
        device,
    )

    rows: List[dict] = []

    # =====================================================
    # Sweep alpha
    # =====================================================
    for alpha in config.interference_levels:
        print("=" * 90)
        print(
            f"alpha={alpha:.2f} | "
            f"SNR={config.fixed_snr_db:g} dB"
        )

        # Separate metric lists for every target dataset and K.
        metric_values: Dict[str, List[float]] = {}

        qualitative_saved_e = False
        qualitative_saved_d = False

        for batch_index, (
            batch_e,
            batch_d,
        ) in enumerate(
            zip(loader_e, loader_d)
        ):
            image_e = get_image(
                batch_e
            ).to(
                device,
                non_blocking=True,
            )

            image_d = get_image(
                batch_d
            ).to(
                device,
                non_blocking=True,
            )

            batch_size = image_e.size(0)

            label_e = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=device,
            )

            label_d = torch.ones(
                batch_size,
                dtype=torch.long,
                device=device,
            )

            # -------------------------------------------------
            # Encode the SAME target images for all K and alpha.
            # -------------------------------------------------
            with torch.no_grad():
                z_e, _ = encoder(image_e)
                z_d, _ = encoder(image_d)

                clean_e = generator(
                    z_e,
                    label_e,
                )

                clean_d = generator(
                    z_d,
                    label_d,
                )

            # EuroSAT target gets DeepGlobe interferers.
            bank_for_e = build_interferer_bank(
                source_latent=z_d,
                target_latent=z_e,
                max_interferers=max(config.num_interferers),
            )

            # DeepGlobe target gets EuroSAT interferers.
            bank_for_d = build_interferer_bank(
                source_latent=z_e,
                target_latent=z_d,
                max_interferers=max(config.num_interferers),
            )

            # -------------------------------------------------
            # SAME noise realization for every K and alpha.
            # alpha is intentionally NOT included in the seed.
            # -------------------------------------------------
            noise_seed = (
                config.seed
                + batch_index
            )

            torch.manual_seed(
                noise_seed
            )

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    noise_seed
                )

            noise_unit_e = torch.randn(
                batch_size,
                config.z_dim,
                device=device,
            )

            noise_unit_d = torch.randn(
                batch_size,
                config.z_dim,
                device=device,
            )

            recon_by_k_e: Dict[int, torch.Tensor] = {}
            recon_by_k_d: Dict[int, torch.Tensor] = {}

            # =================================================
            # K = 1 ... 5
            # =================================================
            for k in config.num_interferers:
                # Same DDIM initial seed across K and alpha.
                sample_seed_base = (
                    config.seed
                    + batch_index * 100
                )

                result_e = evaluate_target_with_k_interferers(
                    generator=generator,
                    ddpm=ddpm,
                    schedule=schedule,
                    z_target=z_e,
                    target_labels=label_e,
                    interferer_bank=bank_for_e,
                    num_interferers=k,
                    alpha=alpha,
                    noise_unit=noise_unit_e,
                    sample_seed=sample_seed_base,
                    config=config,
                )

                result_d = evaluate_target_with_k_interferers(
                    generator=generator,
                    ddpm=ddpm,
                    schedule=schedule,
                    z_target=z_d,
                    target_labels=label_d,
                    interferer_bank=bank_for_d,
                    num_interferers=k,
                    alpha=alpha,
                    noise_unit=noise_unit_d,
                    sample_seed=sample_seed_base + 1,
                    config=config,
                )

                recon_e = result_e["reconstruction"]
                recon_d = result_d["reconstruction"]

                recon_by_k_e[k] = recon_e
                recon_by_k_d[k] = recon_d

                # EuroSAT metrics
                metric_values.setdefault(
                    f"eurosat_k{k}_psnr",
                    [],
                ).extend(
                    batch_psnr(
                        recon_e,
                        image_e,
                    ).tolist()
                )

                metric_values.setdefault(
                    f"eurosat_k{k}_ssim",
                    [],
                ).extend(
                    batch_ssim(
                        recon_e,
                        image_e,
                    ).tolist()
                )

                # DeepGlobe metrics
                metric_values.setdefault(
                    f"deepglobe_k{k}_psnr",
                    [],
                ).extend(
                    batch_psnr(
                        recon_d,
                        image_d,
                    ).tolist()
                )

                metric_values.setdefault(
                    f"deepglobe_k{k}_ssim",
                    [],
                ).extend(
                    batch_ssim(
                        recon_d,
                        image_d,
                    ).tolist()
                )

            # -------------------------------------------------
            # Qualitative samples
            # -------------------------------------------------
            selected_alpha = any(
                math.isclose(
                    float(alpha),
                    float(value),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for value in config.qualitative_alphas
            )

            if selected_alpha and batch_index == 0:
                alpha_tag = (
                    f"{alpha:.2f}"
                    .replace(".", "p")
                )

                if not qualitative_saved_e:
                    save_multi_k_qualitative(
                        os.path.join(
                            config.save_root,
                            f"EuroSAT_qualitative_alpha{alpha_tag}.png",
                        ),
                        image_e,
                        clean_e,
                        recon_by_k_e,
                        config.qualitative_count,
                    )
                    qualitative_saved_e = True

                if not qualitative_saved_d:
                    save_multi_k_qualitative(
                        os.path.join(
                            config.save_root,
                            f"DeepGlobe_qualitative_alpha{alpha_tag}.png",
                        ),
                        image_d,
                        clean_d,
                        recon_by_k_d,
                        config.qualitative_count,
                    )
                    qualitative_saved_d = True

            print(
                f"alpha={alpha:>4.2f} | "
                f"batch {batch_index + 1}/{len(loader_e)} complete"
            )

        # =====================================================
        # Build one CSV row for this alpha
        # =====================================================
        row = {
            "interference_alpha": float(alpha),
            "fixed_snr_db": float(config.fixed_snr_db),
            "num_pairs": usable_pairs,
        }

        for dataset_key in (
            "eurosat",
            "deepglobe",
        ):
            for k in config.num_interferers:
                for metric_name in (
                    "psnr",
                    "ssim",
                ):
                    key = (
                        f"{dataset_key}_"
                        f"k{k}_"
                        f"{metric_name}"
                    )

                    values = metric_values.get(
                        key,
                        [],
                    )

                    row[key] = (
                        float(np.mean(values))
                        if values
                        else float("nan")
                    )

        rows.append(row)

        write_csv(
            os.path.join(
                config.save_root,
                "number_of_interferers_metrics.csv",
            ),
            rows,
        )

        print("EuroSAT PSNR:")
        for k in config.num_interferers:
            print(
                f"  K={k}: "
                f"{row[f'eurosat_k{k}_psnr']:.3f} dB"
            )

        print("DeepGlobe PSNR:")
        for k in config.num_interferers:
            print(
                f"  K={k}: "
                f"{row[f'deepglobe_k{k}_psnr']:.3f} dB"
            )

    # =====================================================
    # Final plots
    # =====================================================
    plot_k_sweep(
        rows=rows,
        dataset_name="EuroSAT",
        metric_name="psnr",
        ylabel="PSNR (dB)",
        config=config,
    )

    plot_k_sweep(
        rows=rows,
        dataset_name="DeepGlobe",
        metric_name="psnr",
        ylabel="PSNR (dB)",
        config=config,
    )

    plot_k_sweep(
        rows=rows,
        dataset_name="EuroSAT",
        metric_name="ssim",
        ylabel="SSIM",
        config=config,
    )

    plot_k_sweep(
        rows=rows,
        dataset_name="DeepGlobe",
        metric_name="ssim",
        ylabel="SSIM",
        config=config,
    )

    print("=" * 90)
    print("Experiment completed.")
    print("Results:", config.save_root)
    print(
        "Main model: "
        "y = z_target + alpha * sum(z_interferer,k) + n"
    )
    print(
        "Five curves correspond to K = 1, 2, 3, 4, 5 interferers."
    )
    print(
        "Each interferer is power matched to the target latent, "
        "and desired-signal SNR is fixed at 10 dB."
    )


if __name__ == "__main__":
    main()
