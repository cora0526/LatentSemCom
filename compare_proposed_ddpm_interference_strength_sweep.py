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

    # Same deterministic DDIM setup used in the previous comparison script.
    sampling_method: str = "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0

    batch_size: int = 16
    num_workers: int = 0

    # Increase this for final thesis figures if runtime allows.
    max_test_pairs: int = 500

    fixed_snr_db: float = 20.0
    interference_levels: Tuple[float, ...] = (
        0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0
    )

    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    # Proposed VAE-GAN encoder / conditional GAN generator.
    scratch_vaegan_checkpoint: str = (
        "./save_model/vae_gan_from_scratch_best.pth"
    )

    # Proposed latent DDPM separator.
    scratch_vaegan_ddpm_checkpoint: str = (
        "./save_model/DDPM_vaegan_scratch_latent_separator_best.pth"
    )

    save_root: str = (
        "./save_data/proposed_ddpm_interference_strength_sweep"
    )

    qualitative_alphas: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
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
            t = torch.full((cond.size(0),),step,dtype=torch.long,device=cond.device)

            pred_eps = model(x,t,labels,cond)
            pred_x0 = schedule.predict_x0_from_eps(x,t,pred_eps)

            if clip_x0 is not None:
                pred_x0 = pred_x0.clamp(-clip_x0,clip_x0)

            beta_t = schedule.extract(schedule.betas,t,x.shape)
            alpha_t = schedule.extract(schedule.alphas,t,x.shape)
            alpha_bar_t = schedule.extract(schedule.alpha_bars,t,x.shape)
            alpha_bar_prev = schedule.extract(schedule.alpha_bars_prev,t,x.shape)
            coef_x0 = (beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t))
            coef_xt = ((1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t))

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

    steps = max(1,min(int(ddim_steps),schedule.timesteps))

    time_sequence = torch.linspace(schedule.timesteps - 1,0,steps,device=cond.device).round().long()

    time_sequence = torch.unique_consecutive(time_sequence)

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

    for prediction, reference in zip(pred_np,target_np):
        values.append(structural_similarity(reference,prediction,data_range=1.0,channel_axis=-1))
    return np.asarray(values,dtype=np.float64)


class MetricAccumulator:
    def __init__(self):
        self.values: Dict[str, List[float]] = {}

    def add(self, name: str, values):
        array = np.asarray(values,dtype=np.float64).reshape(-1)

        self.values.setdefault(name,[]).extend(array.tolist())

    def mean(self, name: str):
        values = self.values.get(name, [])

        if not values:
            return float("nan")

        return float(np.mean(values))

    def count(self):
        if not self.values:
            return 0

        return len(next(iter(self.values.values())))


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate_batch(
    encoder: nn.Module,
    generator: nn.Module,
    ddpm: nn.Module,
    schedule: DiffusionSchedule,
    image_e: torch.Tensor,
    image_d: torch.Tensor,
    interference_alpha: float,
    noise_unit_e: torch.Tensor,
    noise_unit_d: torch.Tensor,
    sample_seed: int,
    config: Config,
):
    """
    Target-centric interference diagnostic:
        y_E = z_E + alpha*z_D + n_E
        y_D = z_D + alpha*z_E + n_D
    """
    batch_size = image_e.size(0)
    label_e = torch.zeros(batch_size, dtype=torch.long, device=image_e.device)
    label_d = torch.ones(batch_size, dtype=torch.long, device=image_e.device)

    z_e, _ = encoder(image_e)
    z_d, _ = encoder(image_d)

    clean_e = generator(z_e, label_e)
    clean_d = generator(z_d, label_d)

    alpha = float(interference_alpha)
    signal_e = z_e + alpha * z_d
    signal_d = z_d + alpha * z_e

    received_e = add_awgn_relative_to_reference(signal_e, z_e, config.fixed_snr_db, noise_unit_e)
    received_d = add_awgn_relative_to_reference(signal_d, z_d, config.fixed_snr_db, noise_unit_d)

    no_sep_e = generator(received_e, label_e)
    no_sep_d = generator(received_d, label_d)

    pred_e = sample_ddpm(
        model=ddpm, schedule=schedule, cond=received_e, labels=label_e,
        z_dim=config.z_dim, seed=sample_seed,
        sampling_method=config.sampling_method, ddim_steps=config.ddim_steps,
        ddim_eta=config.ddim_eta,
    )
    pred_d = sample_ddpm(
        model=ddpm, schedule=schedule, cond=received_d, labels=label_d,
        z_dim=config.z_dim, seed=sample_seed + 1,
        sampling_method=config.sampling_method, ddim_steps=config.ddim_steps,
        ddim_eta=config.ddim_eta,
    )

    proposed_e = generator(pred_e, label_e)
    proposed_d = generator(pred_d, label_d)

    return {
        "z_e": z_e, "z_d": z_d,
        "received_e": received_e, "received_d": received_d,
        "clean_e": clean_e, "clean_d": clean_d,
        "no_sep_e": no_sep_e, "no_sep_d": no_sep_d,
        "proposed_e": proposed_e, "proposed_d": proposed_d,
    }


def add_reconstruction_metrics(
    accumulator: MetricAccumulator,
    prefix: str,
    reconstruction: torch.Tensor,
    real: torch.Tensor,
):
    accumulator.add(
        f"{prefix}_psnr",
        batch_psnr(
            reconstruction,
            real,
        ),
    )

    accumulator.add(
        f"{prefix}_ssim",
        batch_ssim(
            reconstruction,
            real,
        ),
    )


# =========================================================
# Output helpers
# =========================================================

def write_csv(path: str, rows: List[dict]):
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


def save_qualitative_grid(
    save_path: str,
    real_e: torch.Tensor,
    real_d: torch.Tensor,
    result: dict,
    count: int,
):
    """
    EuroSAT rows:
        Real / Clean Latent / No Separation / Proposed DDPM

    DeepGlobe rows:
        Real / Clean Latent / No Separation / Proposed DDPM
    """

    count = min(
        count,
        real_e.size(0),
        real_d.size(0),
    )

    rows = [
        real_e[:count],
        result["clean_e"][:count],
        result["no_sep_e"][:count],
        result["proposed_e"][:count],

        real_d[:count],
        result["clean_d"][:count],
        result["no_sep_d"][:count],
        result["proposed_d"][:count],
    ]

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

SCENARIO_COLORS = {
    "Clean Latent": "#6C757D",
    "No Separation": "#D97757",
    "Proposed DDPM": "#6C63B8",
}

SCENARIO_STYLES = {
    "Clean Latent": "--",
    "No Separation": ":",
    "Proposed DDPM": "-",
}

SCENARIO_MARKERS = {
    "Clean Latent": "s",
    "No Separation": "D",
    "Proposed DDPM": "o",
}

SCENARIO_ORDER = (
    "Clean Latent",
    "No Separation",
    "Proposed DDPM",
)


def plot_dataset_metric(
    rows: List[dict],
    dataset_name: str,
    metric_name: str,
    ylabel: str,
    save_root: str,
):
    dataset_key = "eurosat" if dataset_name == "EuroSAT" else "deepglobe"
    field_map = {
        "Clean Latent": f"{dataset_key}_clean_{metric_name}",
        "No Separation": f"{dataset_key}_no_sep_{metric_name}",
        "Proposed DDPM": f"{dataset_key}_proposed_{metric_name}",
    }

    fig, ax = plt.subplots(figsize=(8, 5.2))
    x = [float(row["interference_alpha"]) for row in rows]

    for scenario in SCENARIO_ORDER:
        y = [float(row[field_map[scenario]]) for row in rows]
        ax.plot(
            x, y,
            color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_STYLES[scenario],
            marker=SCENARIO_MARKERS[scenario],
            markersize=6, linewidth=2.2, label=scenario,
        )

    ax.set_xlabel(r"Interference amplitude $\alpha$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)

    if metric_name == "psnr":
        ax.set_ylim(10.0, 30.5)
        ax.set_yticks([10, 12, 15, 18, 21, 24, 27, 30])
    elif metric_name == "ssim":
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))

    ax.legend(fontsize=9, frameon=True, loc="best")
    fig.tight_layout()

    base_name = f"{dataset_name}_{metric_name.upper()}_vs_Interference"
    png_path = os.path.join(save_root, base_name + ".png")
    pdf_path = os.path.join(save_root, base_name + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", png_path)
    print("Saved:", pdf_path)


def safe_recovery_ratio(clean: float, no_sep: float, proposed: float):
    denominator = clean - no_sep
    if abs(denominator) < 1e-12:
        return float("nan")
    return (proposed - no_sep) / denominator


def plot_recovery_ratio(rows: List[dict], dataset_name: str, save_root: str):
    dataset_key = "eurosat" if dataset_name == "EuroSAT" else "deepglobe"
    x = [float(row["interference_alpha"]) for row in rows]
    y = [float(row[f"{dataset_key}_recovery_ratio"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(
        x, y, color=SCENARIO_COLORS["Proposed DDPM"],
        linestyle="-", marker="o", markersize=6, linewidth=2.2,
        label="Recovery Ratio",
    )
    ax.axhline(1.0, linestyle="--", linewidth=1.5, alpha=0.7, label="Full Recovery")
    ax.axhline(0.0, linestyle=":", linewidth=1.5, alpha=0.7, label="No Recovery")
    ax.set_xlabel(r"Interference amplitude $\alpha$", fontsize=12)
    ax.set_ylabel("PSNR Recovery Ratio", fontsize=12)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True, loc="best")
    fig.tight_layout()

    base_name = f"{dataset_name}_RecoveryRatio_vs_Interference"
    png_path = os.path.join(save_root, base_name + ".png")
    pdf_path = os.path.join(save_root, base_name + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", png_path)
    print("Saved:", pdf_path)


# =========================================================
# Main
# =========================================================

def main():
    config = Config()

    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; switching to CPU.")
        device = "cpu"
    else:
        device = config.device

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(config.save_root, exist_ok=True)

    print("=" * 90)
    print("Interference-strength robustness experiment")
    print(f"Fixed desired-signal SNR = {config.fixed_snr_db} dB")
    print("alpha values =", config.interference_levels)
    print(
        f"DDPM sampler: {config.sampling_method}, "
        f"steps={config.ddim_steps}, eta={config.ddim_eta}"
    )

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    eurosat = datasets.ImageFolder(config.eurosat_root, transform=transform)
    deepglobe = SatelliteDataset(config.deepglobe_root, transform=transform)

    pair_count = min(config.max_test_pairs, len(eurosat), len(deepglobe))
    if pair_count <= 0:
        raise RuntimeError("No test pairs are available.")

    subset_generator = torch.Generator().manual_seed(config.seed)
    e_indices = torch.randperm(len(eurosat), generator=subset_generator)[:pair_count].tolist()
    d_indices = torch.randperm(len(deepglobe), generator=subset_generator)[:pair_count].tolist()

    loader_e = DataLoader(
        Subset(eurosat, e_indices), batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device == "cuda"),
    )
    loader_d = DataLoader(
        Subset(deepglobe, d_indices), batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device == "cuda"),
    )

    encoder, generator, ddpm, schedule = load_models(config, device)
    rows: List[dict] = []

    for alpha in config.interference_levels:
        print("=" * 90)
        print(f"Evaluating alpha={alpha:.2f} at fixed SNR={config.fixed_snr_db:g} dB")
        acc = MetricAccumulator()
        qualitative_saved = False

        for batch_index, (batch_e, batch_d) in enumerate(zip(loader_e, loader_d)):
            image_e = get_image(batch_e).to(device, non_blocking=True)
            image_d = get_image(batch_d).to(device, non_blocking=True)
            current = min(image_e.size(0), image_d.size(0))
            image_e = image_e[:current]
            image_d = image_d[:current]

            # Same AWGN realization for every alpha: alpha is the only swept variable.
            noise_seed = config.seed + batch_index
            torch.manual_seed(noise_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(noise_seed)
            noise_unit_e = torch.randn(current, config.z_dim, device=device)
            noise_unit_d = torch.randn(current, config.z_dim, device=device)

            # Same DDIM initialization for every alpha.
            sample_seed = config.seed + batch_index * 10

            result = evaluate_batch(
                encoder=encoder, generator=generator, ddpm=ddpm, schedule=schedule,
                image_e=image_e, image_d=image_d, interference_alpha=alpha,
                noise_unit_e=noise_unit_e, noise_unit_d=noise_unit_d,
                sample_seed=sample_seed, config=config,
            )

            add_reconstruction_metrics(acc, "eurosat_clean", result["clean_e"], image_e)
            add_reconstruction_metrics(acc, "eurosat_no_sep", result["no_sep_e"], image_e)
            add_reconstruction_metrics(acc, "eurosat_proposed", result["proposed_e"], image_e)

            add_reconstruction_metrics(acc, "deepglobe_clean", result["clean_d"], image_d)
            add_reconstruction_metrics(acc, "deepglobe_no_sep", result["no_sep_d"], image_d)
            add_reconstruction_metrics(acc, "deepglobe_proposed", result["proposed_d"], image_d)

            if (
                not qualitative_saved
                and any(math.isclose(float(alpha), float(v), abs_tol=1e-9)
                        for v in config.qualitative_alphas)
            ):
                alpha_tag = f"{alpha:.2f}".replace(".", "p")
                save_qualitative_grid(
                    os.path.join(config.save_root, f"qualitative_alpha{alpha_tag}.png"),
                    image_e, image_d, result, config.qualitative_count,
                )
                qualitative_saved = True

            print(
                f"alpha={alpha:>4.2f} | batch {batch_index + 1}/{len(loader_e)} complete"
            )

        row = {
            "interference_alpha": float(alpha),
            "interference_power_ratio": float(alpha) ** 2,
            "interference_power_db": (
                float("-inf") if float(alpha) == 0.0
                else 20.0 * math.log10(float(alpha))
            ),
            "fixed_snr_db": config.fixed_snr_db,
            "num_pairs": acc.count(),

            "eurosat_clean_psnr": acc.mean("eurosat_clean_psnr"),
            "eurosat_no_sep_psnr": acc.mean("eurosat_no_sep_psnr"),
            "eurosat_proposed_psnr": acc.mean("eurosat_proposed_psnr"),
            "eurosat_clean_ssim": acc.mean("eurosat_clean_ssim"),
            "eurosat_no_sep_ssim": acc.mean("eurosat_no_sep_ssim"),
            "eurosat_proposed_ssim": acc.mean("eurosat_proposed_ssim"),

            "deepglobe_clean_psnr": acc.mean("deepglobe_clean_psnr"),
            "deepglobe_no_sep_psnr": acc.mean("deepglobe_no_sep_psnr"),
            "deepglobe_proposed_psnr": acc.mean("deepglobe_proposed_psnr"),
            "deepglobe_clean_ssim": acc.mean("deepglobe_clean_ssim"),
            "deepglobe_no_sep_ssim": acc.mean("deepglobe_no_sep_ssim"),
            "deepglobe_proposed_ssim": acc.mean("deepglobe_proposed_ssim"),
        }

        row["eurosat_recovery_ratio"] = safe_recovery_ratio(
            row["eurosat_clean_psnr"], row["eurosat_no_sep_psnr"],
            row["eurosat_proposed_psnr"],
        )
        row["deepglobe_recovery_ratio"] = safe_recovery_ratio(
            row["deepglobe_clean_psnr"], row["deepglobe_no_sep_psnr"],
            row["deepglobe_proposed_psnr"],
        )

        rows.append(row)
        write_csv(
            os.path.join(config.save_root, "interference_strength_metrics.csv"),
            rows,
        )

        print(
            f"EuroSAT | Clean={row['eurosat_clean_psnr']:.3f} | "
            f"NoSep={row['eurosat_no_sep_psnr']:.3f} | "
            f"Proposed={row['eurosat_proposed_psnr']:.3f} | "
            f"Recovery={row['eurosat_recovery_ratio']:.3f}"
        )
        print(
            f"DeepGlobe | Clean={row['deepglobe_clean_psnr']:.3f} | "
            f"NoSep={row['deepglobe_no_sep_psnr']:.3f} | "
            f"Proposed={row['deepglobe_proposed_psnr']:.3f} | "
            f"Recovery={row['deepglobe_recovery_ratio']:.3f}"
        )

    plot_dataset_metric(rows, "EuroSAT", "psnr", "PSNR (dB)", config.save_root)
    plot_dataset_metric(rows, "DeepGlobe", "psnr", "PSNR (dB)", config.save_root)
    plot_dataset_metric(rows, "EuroSAT", "ssim", "SSIM", config.save_root)
    plot_dataset_metric(rows, "DeepGlobe", "ssim", "SSIM", config.save_root)
    plot_recovery_ratio(rows, "EuroSAT", config.save_root)
    plot_recovery_ratio(rows, "DeepGlobe", config.save_root)

    print("=" * 90)
    print("Interference-strength experiment completed.")
    print("Results:", config.save_root)
    print(f"Fixed desired-signal SNR = {config.fixed_snr_db} dB")
    print(
        "alpha=0: no interference; alpha=1: equal latent amplitude; "
        "alpha>1: stronger interferer."
    )


if __name__ == "__main__":
    main()
