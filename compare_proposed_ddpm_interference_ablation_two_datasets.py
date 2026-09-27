"""
Ablation experiment for the proposed VAE-DDPM-GAN semantic communication system.

Goal
----
For the SAME paired EuroSAT / DeepGlobe samples and the SAME AWGN realization,
compare three inference conditions separately for each dataset:

1) Clean Latent
   encoder -> clean latent -> conditional GAN generator

2) No Separation
   encoder -> two-user SISO-MAC mixture + AWGN -> conditional GAN generator
   (the received mixture is directly fed to the generator; no separator)

3) Proposed DDPM
   encoder -> two-user SISO-MAC mixture + AWGN -> DDPM separator
   -> conditional GAN generator

The two datasets are NOT averaged together. The script produces:

    interference_ablation_metrics.csv
    EuroSAT_PSNR_vs_SNR.png / .pdf
    DeepGlobe_PSNR_vs_SNR.png / .pdf
    EuroSAT_SSIM_vs_SNR.png / .pdf
    DeepGlobe_SSIM_vs_SNR.png / .pdf
    qualitative_SNR*.png

The experiment is intentionally AWGN-only so that the discussion isolates the
impact of multi-user interference and the benefit of the DDPM separator.
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

    snr_values: Tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30)

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
        "./save_data/proposed_ddpm_interference_ablation_two_datasets2"
    )

    qualitative_snrs: Tuple[float, ...] = (0, 10, 20, 30)
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


def add_awgn(
    signal: torch.Tensor,
    snr_db: float,
    noise_unit: torch.Tensor,
):
    if math.isinf(float(snr_db)):
        return signal

    power = (
        signal
        .pow(2)
        .mean(dim=1, keepdim=True)
        .clamp_min(1e-12)
    )

    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_std = torch.sqrt(power / snr_linear)

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
def evaluate_batch(
    encoder: nn.Module,
    generator: nn.Module,
    ddpm: nn.Module,
    schedule: DiffusionSchedule,
    image_e: torch.Tensor,
    image_d: torch.Tensor,
    snr_db: float,
    noise_unit: torch.Tensor,
    sample_seed: int,
    config: Config,
):
    """
    E = EuroSAT (label 0)
    D = DeepGlobe (label 1)
    """

    batch_size = image_e.size(0)

    label_e = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=image_e.device,
    )

    label_d = torch.ones(
        batch_size,
        dtype=torch.long,
        device=image_e.device,
    )

    # Encoder output uses the latent mean exactly as in the previous test code.
    z_e, _ = encoder(image_e)
    z_d, _ = encoder(image_d)

    # -----------------------------------------------------
    # 1) Clean latent reconstruction
    # -----------------------------------------------------
    clean_e = generator(z_e, label_e)
    clean_d = generator(z_d, label_d)

    # -----------------------------------------------------
    # Shared two-user SISO-MAC observation
    # z_rx = z_E + z_D + n
    # -----------------------------------------------------
    clean_mix = z_e + z_d

    received = add_awgn(
        clean_mix,
        snr_db,
        noise_unit,
    )

    # -----------------------------------------------------
    # 2) No separator
    # Directly feed the mixed received latent to the
    # conditional generator.
    # -----------------------------------------------------
    no_sep_e = generator(received, label_e)
    no_sep_d = generator(received, label_d)

    # -----------------------------------------------------
    # 3) Proposed DDPM separator
    # Same received mixture; only separator differs.
    # -----------------------------------------------------
    pred_e = sample_ddpm(
        model=ddpm,
        schedule=schedule,
        cond=received,
        labels=label_e,
        z_dim=config.z_dim,
        seed=sample_seed,
        sampling_method=config.sampling_method,
        ddim_steps=config.ddim_steps,
        ddim_eta=config.ddim_eta,
    )

    pred_d = sample_ddpm(
        model=ddpm,
        schedule=schedule,
        cond=received,
        labels=label_d,
        z_dim=config.z_dim,
        seed=sample_seed + 1,
        sampling_method=config.sampling_method,
        ddim_steps=config.ddim_steps,
        ddim_eta=config.ddim_eta,
    )

    proposed_e = generator(pred_e, label_e)
    proposed_d = generator(pred_d, label_d)

    return {
        "z_e": z_e,
        "z_d": z_d,
        "received": received,

        "clean_e": clean_e,
        "clean_d": clean_d,

        "no_sep_e": no_sep_e,
        "no_sep_d": no_sep_d,

        "proposed_e": proposed_e,
        "proposed_d": proposed_d,
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
    """
    metric_name: 'psnr' or 'ssim'
    """

    dataset_key = (
        "eurosat"
        if dataset_name == "EuroSAT"
        else "deepglobe"
    )

    field_map = {
        "Clean Latent": f"{dataset_key}_clean_{metric_name}",
        "No Separation": f"{dataset_key}_no_sep_{metric_name}",
        "Proposed DDPM": f"{dataset_key}_proposed_{metric_name}",
    }

    fig, ax = plt.subplots(
        figsize=(8, 5.2)
    )

    x = [
        float(row["snr_db"])
        for row in rows
    ]

    for scenario in SCENARIO_ORDER:
        field = field_map[scenario]

        y = [
            float(row[field])
            for row in rows
        ]

        ax.plot(
            x,
            y,
            color=SCENARIO_COLORS[scenario],
            linestyle=SCENARIO_STYLES[scenario],
            marker=SCENARIO_MARKERS[scenario],
            markersize=6,
            linewidth=2.2,
            label=scenario,
        )

    ax.set_xlabel(
        "SNR (dB)",
        fontsize=18,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=18,
    )

    ax.tick_params(
        axis="both",
        labelsize=16,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    # Keep identical y-axis scales across the two datasets
    # so the left/right subfigures are visually comparable.
    if metric_name == "psnr":
        ax.set_ylim(12.0, 30.5)
        ax.set_yticks([12, 15, 18, 21, 24, 27, 30])
    elif metric_name == "ssim":
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))

    ax.legend(
        fontsize=16,
        frameon=True,
        loc="best",
    )

    fig.tight_layout()

    base_name = (
        f"{dataset_name}_{metric_name.upper()}_vs_SNR"
    )

    png_path = os.path.join(
        save_root,
        base_name + ".png",
    )

    pdf_path = os.path.join(
        save_root,
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

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(
        config.save_root,
        exist_ok=True,
    )

    print(
        f"DDPM sampler: {config.sampling_method}, "
        f"steps={config.ddim_steps}, "
        f"eta={config.ddim_eta}"
    )

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

    if pair_count <= 0:
        raise RuntimeError(
            "No test pairs are available."
        )

    # Same image pairs for every SNR.
    subset_generator = torch.Generator().manual_seed(
        config.seed
    )

    e_indices = torch.randperm(
        len(eurosat),
        generator=subset_generator,
    )[:pair_count].tolist()

    d_indices = torch.randperm(
        len(deepglobe),
        generator=subset_generator,
    )[:pair_count].tolist()

    loader_e = DataLoader(
        Subset(eurosat, e_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
    )

    loader_d = DataLoader(
        Subset(deepglobe, d_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
    )

    encoder, generator, ddpm, schedule = load_models(
        config,
        device,
    )

    rows: List[dict] = []

    for snr_index, snr_db in enumerate(
        config.snr_values
    ):
        print("=" * 90)
        print(
            f"Evaluating AWGN two-user SISO-MAC, "
            f"SNR={snr_db} dB"
        )

        acc = MetricAccumulator()
        qualitative_saved = False

        for batch_index, (
            batch_e,
            batch_d,
        ) in enumerate(
            zip(loader_e, loader_d)
        ):
            image_e = get_image(batch_e).to(
                device,
                non_blocking=True,
            )

            image_d = get_image(batch_d).to(
                device,
                non_blocking=True,
            )

            current = min(
                image_e.size(0),
                image_d.size(0),
            )

            image_e = image_e[:current]
            image_d = image_d[:current]

            # Same AWGN realization for every scenario.
            channel_seed = (
                config.seed
                + snr_index * 100_000
                + batch_index
            )

            torch.manual_seed(channel_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(
                    channel_seed
                )

            noise_unit = torch.randn(
                current,
                config.z_dim,
                device=device,
            )

            sample_seed = (
                config.seed
                + snr_index * 1_000_000
                + batch_index * 10
            )

            result = evaluate_batch(
                encoder=encoder,
                generator=generator,
                ddpm=ddpm,
                schedule=schedule,
                image_e=image_e,
                image_d=image_d,
                snr_db=snr_db,
                noise_unit=noise_unit,
                sample_seed=sample_seed,
                config=config,
            )

            # ---------------------------------------------
            # EuroSAT metrics
            # ---------------------------------------------
            add_reconstruction_metrics(
                acc,
                "eurosat_clean",
                result["clean_e"],
                image_e,
            )

            add_reconstruction_metrics(
                acc,
                "eurosat_no_sep",
                result["no_sep_e"],
                image_e,
            )

            add_reconstruction_metrics(
                acc,
                "eurosat_proposed",
                result["proposed_e"],
                image_e,
            )

            # ---------------------------------------------
            # DeepGlobe metrics
            # ---------------------------------------------
            add_reconstruction_metrics(
                acc,
                "deepglobe_clean",
                result["clean_d"],
                image_d,
            )

            add_reconstruction_metrics(
                acc,
                "deepglobe_no_sep",
                result["no_sep_d"],
                image_d,
            )

            add_reconstruction_metrics(
                acc,
                "deepglobe_proposed",
                result["proposed_d"],
                image_d,
            )

            if (
                not qualitative_saved
                and float(snr_db)
                in [
                    float(value)
                    for value in config.qualitative_snrs
                ]
            ):
                filename = (
                    f"qualitative_SNR{snr_db:g}.png"
                )

                save_qualitative_grid(
                    os.path.join(
                        config.save_root,
                        filename,
                    ),
                    image_e,
                    image_d,
                    result,
                    config.qualitative_count,
                )

                qualitative_saved = True

            print(
                f"SNR={snr_db:>4} dB | "
                f"batch {batch_index + 1}/{len(loader_e)} complete"
            )

        row = {
            "snr_db": snr_db,
            "num_pairs": acc.count(),

            "eurosat_clean_psnr": acc.mean(
                "eurosat_clean_psnr"
            ),
            "eurosat_no_sep_psnr": acc.mean(
                "eurosat_no_sep_psnr"
            ),
            "eurosat_proposed_psnr": acc.mean(
                "eurosat_proposed_psnr"
            ),

            "eurosat_clean_ssim": acc.mean(
                "eurosat_clean_ssim"
            ),
            "eurosat_no_sep_ssim": acc.mean(
                "eurosat_no_sep_ssim"
            ),
            "eurosat_proposed_ssim": acc.mean(
                "eurosat_proposed_ssim"
            ),

            "deepglobe_clean_psnr": acc.mean(
                "deepglobe_clean_psnr"
            ),
            "deepglobe_no_sep_psnr": acc.mean(
                "deepglobe_no_sep_psnr"
            ),
            "deepglobe_proposed_psnr": acc.mean(
                "deepglobe_proposed_psnr"
            ),

            "deepglobe_clean_ssim": acc.mean(
                "deepglobe_clean_ssim"
            ),
            "deepglobe_no_sep_ssim": acc.mean(
                "deepglobe_no_sep_ssim"
            ),
            "deepglobe_proposed_ssim": acc.mean(
                "deepglobe_proposed_ssim"
            ),
        }

        rows.append(row)

        # Save progressively in case the run is interrupted.
        write_csv(
            os.path.join(
                config.save_root,
                "interference_ablation_metrics.csv",
            ),
            rows,
        )

        print(
            f"EuroSAT | Clean={row['eurosat_clean_psnr']:.3f} dB | "
            f"NoSep={row['eurosat_no_sep_psnr']:.3f} dB | "
            f"Proposed={row['eurosat_proposed_psnr']:.3f} dB"
        )

        print(
            f"DeepGlobe | Clean={row['deepglobe_clean_psnr']:.3f} dB | "
            f"NoSep={row['deepglobe_no_sep_psnr']:.3f} dB | "
            f"Proposed={row['deepglobe_proposed_psnr']:.3f} dB"
        )

    # -----------------------------------------------------
    # Final plots: datasets are kept separate.
    # -----------------------------------------------------
    plot_dataset_metric(
        rows=rows,
        dataset_name="EuroSAT",
        metric_name="psnr",
        ylabel="PSNR (dB)",
        save_root=config.save_root,
    )

    plot_dataset_metric(
        rows=rows,
        dataset_name="DeepGlobe",
        metric_name="psnr",
        ylabel="PSNR (dB)",
        save_root=config.save_root,
    )

    plot_dataset_metric(
        rows=rows,
        dataset_name="EuroSAT",
        metric_name="ssim",
        ylabel="SSIM",
        save_root=config.save_root,
    )

    plot_dataset_metric(
        rows=rows,
        dataset_name="DeepGlobe",
        metric_name="ssim",
        ylabel="SSIM",
        save_root=config.save_root,
    )

    print("=" * 90)
    print("Experiment completed.")
    print("Results:", config.save_root)
    print(
        "Comparison: Clean Latent vs No Separation vs Proposed DDPM"
    )
    print(
        "EuroSAT and DeepGlobe are reported separately."
    )


if __name__ == "__main__":
    main()
