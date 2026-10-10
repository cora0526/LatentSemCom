"""
Coarse-to-diffusion latent separator for a frozen conditional VAE-GAN.

Goal
----
EuroSAT image   -> frozen shared VAE encoder -> z_E
DeepGlobe image -> frozen shared VAE encoder -> z_D
z_rx = z_E + z_D + AWGN

Instead of asking DDPM to generate both 256-D latents from pure Gaussian noise,
this script uses two stages:

1. Deterministic coarse separator
       z_rx -> (z_E_init, z_D_init)

2. Low-noise conditional diffusion refinement
       noisy(z_init, t_start) -> (z_E_hat, z_D_hat)

The frozen conditional GAN then decodes the recovered latents.

Frozen checkpoint
-----------------
./save_model/vae_gan_from_scratch_best.pth

Important
---------
The encoder and generator classes below must exactly match the classes used to
produce the VAE-GAN checkpoint.
"""

from __future__ import annotations

import copy
import csv
import glob
import math
import os
import random
from itertools import cycle
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ============================================================
# Dataset definitions
#   0 = EuroSAT
#   1 = DeepGlobe
# ============================================================
class SatelliteDataset(Dataset):
    def __init__(self, root: str, transform=None, domain_label: int = 1):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform
        self.domain_label = int(domain_label)
        if not self.files:
            raise RuntimeError(f"No matching DeepGlobe images found in:\n{root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        image = Image.open(self.files[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.domain_label


class DomainLabelDataset(Dataset):
    def __init__(self, dataset: Dataset, domain_label: int):
        self.dataset = dataset
        self.domain_label = int(domain_label)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return image, self.domain_label


# ============================================================
# Frozen VAE-GAN classes
# These definitions must match the checkpoint source exactly.
# ============================================================
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar


class ConditionalGenerator(nn.Module):
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

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        y = self.label_embedding(labels)
        h = self.fc(torch.cat([z, y], dim=1))
        h = h.view(z.size(0), 512, 4, 4)
        return self.decoder(h)


# ============================================================
# Normalization helpers
# ============================================================
class LatentNormalizer:
    def __init__(self, stats: Dict[str, torch.Tensor]):
        required = {"mean_e", "std_e", "mean_d", "std_d", "mean_mix", "std_mix"}
        missing = required.difference(stats)
        if missing:
            raise KeyError(f"Missing latent statistics: {sorted(missing)}")
        self.stats = stats

    def e_norm(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.stats["mean_e"]) / self.stats["std_e"]

    def d_norm(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.stats["mean_d"]) / self.stats["std_d"]

    def mix_norm(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.stats["mean_mix"]) / self.stats["std_mix"]

    def e_denorm(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.stats["std_e"] + self.stats["mean_e"]

    def d_denorm(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.stats["std_d"] + self.stats["mean_d"]

    def pair_norm(self, z_e: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.e_norm(z_e), self.d_norm(z_d)], dim=1)

    def pair_denorm(self, pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_e_n, z_d_n = pair.chunk(2, dim=1)
        return self.e_denorm(z_e_n), self.d_denorm(z_d_n)


# ============================================================
# Coarse deterministic separator
# ============================================================
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CoarseLatentSeparator(nn.Module):
    """Predict both normalized source latents directly from normalized z_rx."""

    def __init__(
        self,
        z_dim: int = 256,
        hidden_dim: int = 768,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(hidden_dim, expansion=2, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.shared_norm = nn.LayerNorm(hidden_dim)
        self.head_e = nn.Linear(hidden_dim, z_dim)
        self.head_d = nn.Linear(hidden_dim, z_dim)

    def forward(self, cond_mix: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(cond_mix)
        h = self.blocks(h)
        h = self.shared_norm(h)
        return torch.cat([self.head_e(h), self.head_d(h)], dim=1)


# ============================================================
# Low-noise conditional diffusion refiner
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, hidden_dim * 2),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(condition).chunk(2, dim=1)
        h = self.norm(x)
        h = h * (1.0 + scale) + shift
        return x + self.net(h)


class ConditionalLatentRefiner(nn.Module):
    """
    Predict epsilon for a paired 512-D latent state.

    Inputs
    ------
    x_t:
        Noisy paired source latent [z_E_t, z_D_t].
    t:
        Diffusion timestep.
    cond_mix:
        Normalized received mixture.
    coarse_pair:
        Deterministic separator output. This anchors the refinement to the
        received sample and greatly reduces label-only generation.
    """

    def __init__(
        self,
        z_dim: int = 256,
        time_dim: int = 128,
        hidden_dim: int = 768,
        condition_dim: int = 512,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        pair_dim = 2 * z_dim
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.state_input = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.SiLU(),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(z_dim + pair_dim + time_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock(hidden_dim, condition_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pair_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        cond_mix: torch.Tensor,
        coarse_pair: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        condition = self.condition_encoder(
            torch.cat([cond_mix, coarse_pair, t_emb], dim=1)
        )
        h = self.state_input(x_t)
        for block in self.blocks:
            h = block(h, condition)
        return self.output(h)


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars)
        )
        self.register_buffer("sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars))
        self.register_buffer(
            "sqrt_recipm1_alpha_bars", torch.sqrt(1.0 / alpha_bars - 1.0)
        )
        self.register_buffer(
            "posterior_variance", torch.clamp(posterior_variance, min=1e-20)
        )

    @staticmethod
    def extract(values: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.size(0), *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_ab = self.extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_omb = self.extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return sqrt_ab * x0 + sqrt_omb * noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        recip = self.extract(self.sqrt_recip_alpha_bars, t, x_t.shape)
        recipm1 = self.extract(self.sqrt_recipm1_alpha_bars, t, x_t.shape)
        return recip * x_t - recipm1 * eps


# ============================================================
# Generic utilities
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(model: nn.Module, value: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = value


def append_csv(path: str, row: Dict[str, float]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def add_awgn_by_snr(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    if math.isinf(snr_db):
        return signal
    signal_power = signal.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    return signal + torch.randn_like(signal) * torch.sqrt(noise_power)


def choose_snr(epoch: int) -> float:
    """Progressive channel curriculum."""
    if epoch <= 5:
        return math.inf
    if epoch <= 15:
        return random.uniform(20.0, 30.0)
    if epoch <= 30:
        return random.uniform(10.0, 30.0)
    if epoch <= 50:
        return random.uniform(5.0, 30.0)
    return random.uniform(0.0, 30.0)


def latent_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "mse": F.mse_loss(pred, target).item(),
        "cosine": F.cosine_similarity(pred, target, dim=1).mean().item(),
    }


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_parameters = dict(ema_model.named_parameters())
    model_parameters = dict(model.named_parameters())
    for name, ema_parameter in ema_parameters.items():
        ema_parameter.mul_(decay).add_(model_parameters[name], alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name])


# ============================================================
# VAE-GAN loading and latent-statistics estimation
# ============================================================
def load_vae_gan_models(
    checkpoint_path: str,
    device: str,
    z_dim: int,
    label_dim: int,
    num_classes: int,
) -> Tuple[nn.Module, nn.Module]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Cannot find VAE-GAN checkpoint:\n{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "encoder" not in checkpoint or "generator" not in checkpoint:
        raise KeyError("Checkpoint must contain 'encoder' and 'generator'.")

    encoder = SharedVAEEncoder(z_dim=z_dim).to(device)
    generator = ConditionalGenerator(
        z_dim=z_dim,
        num_classes=num_classes,
        label_dim=label_dim,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    generator.load_state_dict(checkpoint["generator"], strict=True)

    encoder.eval()
    generator.eval()
    set_requires_grad(encoder, False)
    set_requires_grad(generator, False)

    print(f"Loaded VAE-GAN epoch = {checkpoint.get('epoch', 'unknown')}")
    print(f"Loaded VAE-GAN stage = {checkpoint.get('stage', 'unknown')}")
    return encoder, generator


@torch.no_grad()
def estimate_latent_statistics(
    encoder: nn.Module,
    loader_e: DataLoader,
    loader_d: DataLoader,
    device: str,
    max_batches: int = 200,
    eps: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    """Estimate per-dimension means/stds for both domains and their mixture."""
    encoder.eval()
    sum_e = sum_d = sum_m = None
    sq_e = sq_d = sq_m = None
    count = 0

    iterator_e: Iterable = cycle(loader_e)
    iterator_d: Iterable = cycle(loader_d)

    print("Estimating per-dimension latent statistics...")
    for _ in range(max_batches):
        image_e, _ = next(iterator_e)
        image_d, _ = next(iterator_d)
        image_e = image_e.to(device, non_blocking=True)
        image_d = image_d.to(device, non_blocking=True)

        z_e, _ = encoder(image_e)
        z_d, _ = encoder(image_d)
        z_m = z_e + z_d

        batch_sum_e = z_e.sum(dim=0)
        batch_sum_d = z_d.sum(dim=0)
        batch_sum_m = z_m.sum(dim=0)
        batch_sq_e = z_e.square().sum(dim=0)
        batch_sq_d = z_d.square().sum(dim=0)
        batch_sq_m = z_m.square().sum(dim=0)

        if sum_e is None:
            sum_e, sum_d, sum_m = batch_sum_e, batch_sum_d, batch_sum_m
            sq_e, sq_d, sq_m = batch_sq_e, batch_sq_d, batch_sq_m
        else:
            sum_e += batch_sum_e
            sum_d += batch_sum_d
            sum_m += batch_sum_m
            sq_e += batch_sq_e
            sq_d += batch_sq_d
            sq_m += batch_sq_m
        count += z_e.size(0)

    if count == 0:
        raise RuntimeError("No samples were available for latent statistics.")

    def finish(total: torch.Tensor, total_sq: torch.Tensor):
        mean = total / count
        variance = total_sq / count - mean.square()
        std = torch.sqrt(variance.clamp_min(eps))
        return mean.unsqueeze(0), std.unsqueeze(0)

    mean_e, std_e = finish(sum_e, sq_e)
    mean_d, std_d = finish(sum_d, sq_d)
    mean_mix, std_mix = finish(sum_m, sq_m)

    print(
        "Average latent std | "
        f"E={std_e.mean().item():.4f} | "
        f"D={std_d.mean().item():.4f} | "
        f"mix={std_mix.mean().item():.4f}"
    )
    return {
        "mean_e": mean_e,
        "std_e": std_e,
        "mean_d": mean_d,
        "std_d": std_d,
        "mean_mix": mean_mix,
        "std_mix": std_mix,
    }


# ============================================================
# Low-noise refinement sampler
# ============================================================
@torch.no_grad()
def refine_from_coarse_ddim(
    model: nn.Module,
    schedule: DiffusionSchedule,
    coarse_pair: torch.Tensor,
    cond_mix: torch.Tensor,
    start_t: int = 60,
    sample_steps: int = 30,
    initial_noise: torch.Tensor | None = None,
    eta: float = 0.0,
    clip_x0: float | None = 5.0,
) -> torch.Tensor:
    """
    SDEdit-style refinement:
      1. Add moderate diffusion noise to the coarse estimate at start_t.
      2. Reverse only from start_t to zero.

    eta=0 gives deterministic DDIM after the initial noise is fixed.
    """
    model.eval()
    batch = coarse_pair.size(0)
    device = coarse_pair.device

    start_t = int(max(1, min(start_t, schedule.timesteps - 1)))
    sample_steps = int(max(2, min(sample_steps, start_t + 1)))

    if initial_noise is None:
        initial_noise = torch.randn_like(coarse_pair)
    t_start = torch.full((batch,), start_t, device=device, dtype=torch.long)
    x = schedule.q_sample(coarse_pair, t_start, initial_noise)

    step_values = torch.linspace(start_t, 0, sample_steps, device=device)
    step_values = torch.round(step_values).long()
    step_values = torch.unique_consecutive(step_values)
    if step_values[-1].item() != 0:
        step_values = torch.cat([step_values, torch.zeros(1, device=device, dtype=torch.long)])

    for index, step in enumerate(step_values):
        current = int(step.item())
        next_step = (
            int(step_values[index + 1].item())
            if index + 1 < len(step_values)
            else -1
        )
        t = torch.full((batch,), current, device=device, dtype=torch.long)
        pred_eps = model(x, t, cond_mix, coarse_pair)
        pred_x0 = schedule.predict_x0_from_eps(x, t, pred_eps)
        if clip_x0 is not None:
            pred_x0 = pred_x0.clamp(-clip_x0, clip_x0)

        alpha_bar_t = schedule.extract(schedule.alpha_bars, t, x.shape)
        if next_step < 0:
            x = pred_x0
            continue

        t_next = torch.full((batch,), next_step, device=device, dtype=torch.long)
        alpha_bar_next = schedule.extract(schedule.alpha_bars, t_next, x.shape)

        sigma = eta * torch.sqrt(
            ((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t)).clamp_min(0.0)
            * (1.0 - alpha_bar_t / alpha_bar_next).clamp_min(0.0)
        )
        direction = torch.sqrt((1.0 - alpha_bar_next - sigma.square()).clamp_min(0.0))
        noise = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)
        x = torch.sqrt(alpha_bar_next) * pred_x0 + direction * pred_eps + sigma * noise

    return x


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate(
    separator: nn.Module,
    refiner: nn.Module,
    schedule: DiffusionSchedule,
    encoder: nn.Module,
    generator: nn.Module,
    normalizer: LatentNormalizer,
    fixed_e: torch.Tensor,
    fixed_d: torch.Tensor,
    fixed_initial_noise: torch.Tensor,
    epoch: int,
    save_root: str,
    validation_snr_db: float,
    refine_start_t: int,
    ddim_steps: int,
) -> Dict[str, float]:
    separator.eval()
    refiner.eval()
    encoder.eval()
    generator.eval()

    batch = fixed_e.size(0)
    label_e = torch.zeros(batch, dtype=torch.long, device=fixed_e.device)
    label_d = torch.ones(batch, dtype=torch.long, device=fixed_e.device)

    z_e, _ = encoder(fixed_e)
    z_d, _ = encoder(fixed_d)
    z_mix_clean = z_e + z_d
    z_rx = add_awgn_by_snr(z_mix_clean, validation_snr_db)
    cond_n = normalizer.mix_norm(z_rx)

    coarse_pair_n = separator(cond_n)
    refined_pair_n = refine_from_coarse_ddim(
        model=refiner,
        schedule=schedule,
        coarse_pair=coarse_pair_n,
        cond_mix=cond_n,
        start_t=refine_start_t,
        sample_steps=ddim_steps,
        initial_noise=fixed_initial_noise,
        eta=0.0,
    )
    coarse_e, coarse_d = normalizer.pair_denorm(coarse_pair_n)
    pred_e, pred_d = normalizer.pair_denorm(refined_pair_n)

    recon_coarse_e = generator(coarse_e, label_e)
    recon_coarse_d = generator(coarse_d, label_d)
    recon_e = generator(pred_e, label_e)
    recon_d = generator(pred_d, label_d)
    oracle_e = generator(z_e, label_e)
    oracle_d = generator(z_d, label_d)

    # Zero-condition test. Both the separator and refiner now lose sample-specific input.
    zero_cond = torch.zeros_like(cond_n)
    zero_coarse_pair = separator(zero_cond)
    zero_pair = refine_from_coarse_ddim(
        model=refiner,
        schedule=schedule,
        coarse_pair=zero_coarse_pair,
        cond_mix=zero_cond,
        start_t=refine_start_t,
        sample_steps=ddim_steps,
        initial_noise=fixed_initial_noise,
        eta=0.0,
    )
    zero_e, zero_d = normalizer.pair_denorm(zero_pair)
    zero_recon_e = generator(zero_e, label_e)
    zero_recon_d = generator(zero_d, label_d)

    # Partner swap: retain E and change only the D partner.
    permutation = torch.randperm(batch, device=z_d.device)
    swapped_mix = z_e + z_d[permutation]
    swapped_cond = normalizer.mix_norm(swapped_mix)
    swapped_coarse = separator(swapped_cond)
    swapped_pair = refine_from_coarse_ddim(
        model=refiner,
        schedule=schedule,
        coarse_pair=swapped_coarse,
        cond_mix=swapped_cond,
        start_t=refine_start_t,
        sample_steps=ddim_steps,
        initial_noise=fixed_initial_noise,
        eta=0.0,
    )
    swapped_e, _ = normalizer.pair_denorm(swapped_pair)
    swapped_recon_e = generator(swapped_e, label_e)

    coarse_metrics_e = latent_metrics(coarse_e, z_e)
    coarse_metrics_d = latent_metrics(coarse_d, z_d)
    refined_metrics_e = latent_metrics(pred_e, z_e)
    refined_metrics_d = latent_metrics(pred_d, z_d)

    metrics = {
        "coarse_latent_mse_e": coarse_metrics_e["mse"],
        "coarse_latent_mse_d": coarse_metrics_d["mse"],
        "refined_latent_mse_e": refined_metrics_e["mse"],
        "refined_latent_mse_d": refined_metrics_d["mse"],
        "coarse_latent_cos_e": coarse_metrics_e["cosine"],
        "coarse_latent_cos_d": coarse_metrics_d["cosine"],
        "refined_latent_cos_e": refined_metrics_e["cosine"],
        "refined_latent_cos_d": refined_metrics_d["cosine"],
        "coarse_image_l1_e_to_real": F.l1_loss(recon_coarse_e, fixed_e).item(),
        "coarse_image_l1_d_to_real": F.l1_loss(recon_coarse_d, fixed_d).item(),
        "refined_image_l1_e_to_real": F.l1_loss(recon_e, fixed_e).item(),
        "refined_image_l1_d_to_real": F.l1_loss(recon_d, fixed_d).item(),
        "coarse_image_l1_e_to_oracle": F.l1_loss(recon_coarse_e, oracle_e).item(),
        "coarse_image_l1_d_to_oracle": F.l1_loss(recon_coarse_d, oracle_d).item(),
        "refined_image_l1_e_to_oracle": F.l1_loss(recon_e, oracle_e).item(),
        "refined_image_l1_d_to_oracle": F.l1_loss(recon_d, oracle_d).item(),
        "coarse_mixture_mse_clean": F.mse_loss(coarse_e + coarse_d, z_mix_clean).item(),
        "refined_mixture_mse_clean": F.mse_loss(pred_e + pred_d, z_mix_clean).item(),
        "zero_cond_l1_e_to_real": F.l1_loss(zero_recon_e, fixed_e).item(),
        "zero_cond_l1_d_to_real": F.l1_loss(zero_recon_d, fixed_d).item(),
        "partner_swap_e_change": F.l1_loss(swapped_recon_e, recon_e).item(),
    }

    count = fixed_e.size(0)
    nrow = max(1, int(math.sqrt(count)))
    save_image((fixed_e + 1.0) / 2.0, os.path.join(save_root, f"val_real_E_ep{epoch}.png"), nrow=nrow)
    save_image((fixed_d + 1.0) / 2.0, os.path.join(save_root, f"val_real_D_ep{epoch}.png"), nrow=nrow)
    save_image((oracle_e + 1.0) / 2.0, os.path.join(save_root, f"val_oracle_E_ep{epoch}.png"), nrow=nrow)
    save_image((oracle_d + 1.0) / 2.0, os.path.join(save_root, f"val_oracle_D_ep{epoch}.png"), nrow=nrow)
    save_image((recon_coarse_e + 1.0) / 2.0, os.path.join(save_root, f"val_coarse_E_ep{epoch}.png"), nrow=nrow)
    save_image((recon_coarse_d + 1.0) / 2.0, os.path.join(save_root, f"val_coarse_D_ep{epoch}.png"), nrow=nrow)
    save_image((recon_e + 1.0) / 2.0, os.path.join(save_root, f"val_refined_E_ep{epoch}.png"), nrow=nrow)
    save_image((recon_d + 1.0) / 2.0, os.path.join(save_root, f"val_refined_D_ep{epoch}.png"), nrow=nrow)
    save_image((zero_recon_e + 1.0) / 2.0, os.path.join(save_root, f"val_zero_cond_E_ep{epoch}.png"), nrow=nrow)
    save_image((zero_recon_d + 1.0) / 2.0, os.path.join(save_root, f"val_zero_cond_D_ep{epoch}.png"), nrow=nrow)
    save_image((swapped_recon_e + 1.0) / 2.0, os.path.join(save_root, f"val_partner_swap_E_ep{epoch}.png"), nrow=nrow)
    return metrics


# ============================================================
# Main training
# ============================================================
def main() -> None:
    # ---------------------------
    # General settings
    # ---------------------------
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 256
    label_dim = 64
    num_classes = 2

    # Two-stage training.
    coarse_epochs = 25
    refiner_epochs = 75
    total_epochs = coarse_epochs + refiner_epochs
    steps_per_epoch = 1000
    batch_size = 32
    num_workers = 4

    coarse_lr = 2e-4
    refiner_lr = 1e-4
    grad_clip = 1.0
    ema_decay = 0.999

    # Low-noise diffusion. The model never needs to learn the full 400-step
    # pure-noise generation problem.
    diffusion_steps = 100
    max_train_t = 80
    refine_start_t = 60
    ddim_steps = 30

    # Coarse separator losses.
    lambda_coarse_latent = 1.0
    lambda_coarse_consistency = 0.20
    lambda_coarse_image = 0.05

    # Diffusion refiner losses.
    lambda_eps = 1.0
    lambda_refine_x0 = 0.50
    lambda_refine_consistency = 0.20
    lambda_refine_image = 0.05
    lambda_stay_near_coarse = 0.02

    validation_snr_db = 20.0
    validate_every = 5
    fixed_count = 16
    stats_batches = 200

    # ---------------------------
    # Paths
    # ---------------------------
    data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    data_root_d = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    vae_gan_checkpoint = "./save_model/vae_gan_from_scratch_best.pth"

    save_root = "./save_data/DDPM_coarse_to_diffusion_v5"
    model_root = "./save_model"
    latest_path = os.path.join(model_root, "DDPM_coarse_to_diffusion_v5_latest.pth")
    best_path = os.path.join(model_root, "DDPM_coarse_to_diffusion_v5_best.pth")
    history_path = os.path.join(save_root, "training_history.csv")
    validation_history_path = os.path.join(save_root, "validation_history.csv")

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)
    set_seed(seed)

    # ---------------------------
    # Data
    # ---------------------------
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    base_e = datasets.ImageFolder(root=data_root_e, transform=transform)
    base_e = Subset(base_e, range(min(50000, len(base_e))))
    dataset_e = DomainLabelDataset(base_e, domain_label=0)

    dataset_d = SatelliteDataset(data_root_d, transform=transform, domain_label=1)
    dataset_d = Subset(dataset_d, range(min(50000, len(dataset_d))))

    loader_e = DataLoader(
        dataset_e,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    loader_d = DataLoader(
        dataset_d,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    fixed_count = min(fixed_count, len(dataset_e), len(dataset_d))
    fixed_e = torch.stack([dataset_e[i][0] for i in range(fixed_count)]).to(device)
    fixed_d = torch.stack([dataset_d[i][0] for i in range(fixed_count)]).to(device)

    # ---------------------------
    # Frozen VAE-GAN and latent normalization
    # ---------------------------
    encoder, generator = load_vae_gan_models(
        checkpoint_path=vae_gan_checkpoint,
        device=device,
        z_dim=z_dim,
        label_dim=label_dim,
        num_classes=num_classes,
    )
    latent_stats = estimate_latent_statistics(
        encoder=encoder,
        loader_e=loader_e,
        loader_d=loader_d,
        device=device,
        max_batches=stats_batches,
    )
    normalizer = LatentNormalizer(latent_stats)

    # ---------------------------
    # Trainable models
    # ---------------------------
    separator = CoarseLatentSeparator(
        z_dim=z_dim,
        hidden_dim=768,
        num_blocks=6,
        dropout=0.0,
    ).to(device)
    refiner = ConditionalLatentRefiner(
        z_dim=z_dim,
        time_dim=128,
        hidden_dim=768,
        condition_dim=512,
        num_blocks=6,
        dropout=0.0,
    ).to(device)
    ema_refiner = copy.deepcopy(refiner).to(device)
    ema_refiner.eval()
    set_requires_grad(ema_refiner, False)

    schedule = DiffusionSchedule(timesteps=diffusion_steps).to(device)
    coarse_optimizer = torch.optim.AdamW(
        separator.parameters(), lr=coarse_lr, weight_decay=1e-5
    )
    refiner_optimizer = torch.optim.AdamW(
        refiner.parameters(), lr=refiner_lr, weight_decay=1e-5
    )

    fixed_initial_noise = torch.randn(fixed_count, z_dim * 2, device=device)
    label_e_fixed = torch.zeros(fixed_count, dtype=torch.long, device=device)
    label_d_fixed = torch.ones(fixed_count, dtype=torch.long, device=device)

    print(f"Device                 = {device}")
    print(f"EuroSAT images         = {len(dataset_e)}")
    print(f"DeepGlobe images       = {len(dataset_d)}")
    print(f"Coarse/refiner epochs  = {coarse_epochs}/{refiner_epochs}")
    print(f"Steps per epoch        = {steps_per_epoch}")
    print(f"Diffusion steps        = {diffusion_steps}")
    print(f"Max train t/start t    = {max_train_t}/{refine_start_t}")
    print(f"DDIM validation steps  = {ddim_steps}")

    iter_e = cycle(loader_e)
    iter_d = cycle(loader_d)
    global_step = 0
    best_score = float("inf")

    for epoch in range(1, total_epochs + 1):
        is_coarse_stage = epoch <= coarse_epochs
        stage_name = "coarse" if is_coarse_stage else "refiner"

        if is_coarse_stage:
            separator.train()
            refiner.eval()
            set_requires_grad(separator, True)
            set_requires_grad(refiner, False)
        else:
            # Freeze the coarse separator after it has learned a stable inverse.
            separator.eval()
            refiner.train()
            set_requires_grad(separator, False)
            set_requires_grad(refiner, True)

        totals = {
            "total": 0.0,
            "coarse_latent": 0.0,
            "coarse_cons": 0.0,
            "coarse_img": 0.0,
            "eps": 0.0,
            "x0": 0.0,
            "refine_cons": 0.0,
            "refine_img": 0.0,
            "stay": 0.0,
        }

        for step in range(steps_per_epoch):
            image_e, _ = next(iter_e)
            image_d, _ = next(iter_d)
            image_e = image_e.to(device, non_blocking=True)
            image_d = image_d.to(device, non_blocking=True)

            with torch.no_grad():
                z_e, _ = encoder(image_e)
                z_d, _ = encoder(image_d)
                z_mix_clean = z_e + z_d
                train_snr_db = choose_snr(epoch)
                z_rx = add_awgn_by_snr(z_mix_clean, train_snr_db)

                z_e_n = normalizer.e_norm(z_e)
                z_d_n = normalizer.d_norm(z_d)
                target_pair_n = torch.cat([z_e_n, z_d_n], dim=1)
                cond_n = normalizer.mix_norm(z_rx)

                batch_now = z_e.size(0)
                label_e = torch.zeros(batch_now, dtype=torch.long, device=device)
                label_d = torch.ones(batch_now, dtype=torch.long, device=device)
                oracle_img_e = generator(z_e, label_e)
                oracle_img_d = generator(z_d, label_d)

            if is_coarse_stage:
                coarse_pair_n = separator(cond_n)
                coarse_e, coarse_d = normalizer.pair_denorm(coarse_pair_n)

                loss_coarse_latent = F.mse_loss(coarse_pair_n, target_pair_n)
                loss_coarse_cons = F.mse_loss(
                    coarse_e + coarse_d,
                    z_mix_clean,
                )
                coarse_img_e = generator(coarse_e, label_e)
                coarse_img_d = generator(coarse_d, label_d)
                loss_coarse_img = 0.5 * (
                    F.l1_loss(coarse_img_e, oracle_img_e)
                    + F.l1_loss(coarse_img_d, oracle_img_d)
                )
                loss = (
                    lambda_coarse_latent * loss_coarse_latent
                    + lambda_coarse_consistency * loss_coarse_cons
                    + lambda_coarse_image * loss_coarse_img
                )

                coarse_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(separator.parameters(), grad_clip)
                coarse_optimizer.step()

                loss_eps = torch.zeros((), device=device)
                loss_x0 = torch.zeros((), device=device)
                loss_refine_cons = torch.zeros((), device=device)
                loss_refine_img = torch.zeros((), device=device)
                loss_stay = torch.zeros((), device=device)

            else:
                with torch.no_grad():
                    coarse_pair_n = separator(cond_n)

                # Train only at low-to-moderate timesteps, because inference
                # also starts from a noisy coarse estimate rather than x_T.
                batch_now = target_pair_n.size(0)
                t = torch.randint(
                    0,
                    max_train_t + 1,
                    (batch_now,),
                    device=device,
                )
                diffusion_noise = torch.randn_like(target_pair_n)
                x_t = schedule.q_sample(target_pair_n, t, diffusion_noise)
                pred_eps = refiner(x_t, t, cond_n, coarse_pair_n)
                pred_x0_n = schedule.predict_x0_from_eps(x_t, t, pred_eps)

                loss_eps = F.mse_loss(pred_eps, diffusion_noise)
                loss_x0 = F.mse_loss(pred_x0_n, target_pair_n)

                pred_e, pred_d = normalizer.pair_denorm(pred_x0_n)
                loss_refine_cons = F.mse_loss(
                    pred_e + pred_d,
                    z_mix_clean,
                )
                refined_img_e = generator(pred_e, label_e)
                refined_img_d = generator(pred_d, label_d)
                loss_refine_img = 0.5 * (
                    F.l1_loss(refined_img_e, oracle_img_e)
                    + F.l1_loss(refined_img_d, oracle_img_d)
                )

                # Small anchor term: refinement may correct the coarse result,
                # but should not ignore it and become an unconstrained generator.
                loss_stay = F.smooth_l1_loss(pred_x0_n, coarse_pair_n)
                loss = (
                    lambda_eps * loss_eps
                    + lambda_refine_x0 * loss_x0
                    + lambda_refine_consistency * loss_refine_cons
                    + lambda_refine_image * loss_refine_img
                    + lambda_stay_near_coarse * loss_stay
                )

                refiner_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), grad_clip)
                refiner_optimizer.step()
                update_ema(ema_refiner, refiner, ema_decay)

                loss_coarse_latent = torch.zeros((), device=device)
                loss_coarse_cons = torch.zeros((), device=device)
                loss_coarse_img = torch.zeros((), device=device)

            totals["total"] += loss.item()
            totals["coarse_latent"] += loss_coarse_latent.item()
            totals["coarse_cons"] += loss_coarse_cons.item()
            totals["coarse_img"] += loss_coarse_img.item()
            totals["eps"] += loss_eps.item()
            totals["x0"] += loss_x0.item()
            totals["refine_cons"] += loss_refine_cons.item()
            totals["refine_img"] += loss_refine_img.item()
            totals["stay"] += loss_stay.item()

            if global_step % 200 == 0:
                snr_text = "clean" if math.isinf(train_snr_db) else f"{train_snr_db:.1f} dB"
                if is_coarse_stage:
                    print(
                        f"Epoch {epoch}/{total_epochs} [{stage_name}] | "
                        f"Step {step}/{steps_per_epoch} | SNR={snr_text} | "
                        f"Total={loss.item():.4f} | "
                        f"Lat={loss_coarse_latent.item():.4f} | "
                        f"Cons={loss_coarse_cons.item():.4f} | "
                        f"Img={loss_coarse_img.item():.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch}/{total_epochs} [{stage_name}] | "
                        f"Step {step}/{steps_per_epoch} | SNR={snr_text} | "
                        f"Total={loss.item():.4f} | "
                        f"eps={loss_eps.item():.4f} | x0={loss_x0.item():.4f} | "
                        f"Cons={loss_refine_cons.item():.4f} | "
                        f"Img={loss_refine_img.item():.4f} | "
                        f"Stay={loss_stay.item():.4f}"
                    )
            global_step += 1

        averages = {key: value / steps_per_epoch for key, value in totals.items()}
        history_row = {
            "epoch": epoch,
            "stage": stage_name,
            **averages,
        }
        append_csv(history_path, history_row)

        print("=" * 120)
        if is_coarse_stage:
            print(
                f"Epoch {epoch}/{total_epochs} [{stage_name}] | "
                f"Total={averages['total']:.4f} | "
                f"Lat={averages['coarse_latent']:.4f} | "
                f"Cons={averages['coarse_cons']:.4f} | "
                f"Img={averages['coarse_img']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch}/{total_epochs} [{stage_name}] | "
                f"Total={averages['total']:.4f} | eps={averages['eps']:.4f} | "
                f"x0={averages['x0']:.4f} | Cons={averages['refine_cons']:.4f} | "
                f"Img={averages['refine_img']:.4f} | Stay={averages['stay']:.4f}"
            )
        print("=" * 120)

        checkpoint = {
            "epoch": epoch,
            "stage": stage_name,
            "separator": separator.state_dict(),
            "refiner": refiner.state_dict(),
            "ema_refiner": ema_refiner.state_dict(),
            "coarse_optimizer": coarse_optimizer.state_dict(),
            "refiner_optimizer": refiner_optimizer.state_dict(),
            "latent_stats": {key: value.detach().cpu() for key, value in latent_stats.items()},
            "config": {
                "z_dim": z_dim,
                "label_dim": label_dim,
                "num_classes": num_classes,
                "coarse_epochs": coarse_epochs,
                "refiner_epochs": refiner_epochs,
                "diffusion_steps": diffusion_steps,
                "max_train_t": max_train_t,
                "refine_start_t": refine_start_t,
                "ddim_steps": ddim_steps,
                "validation_snr_db": validation_snr_db,
                "vae_gan_checkpoint": vae_gan_checkpoint,
            },
        }
        torch.save(checkpoint, latest_path)

        should_validate = (
            epoch == 1
            or epoch == coarse_epochs
            or epoch % validate_every == 0
            or epoch == total_epochs
        )
        if should_validate:
            # During coarse pretraining the untrained refiner would make the
            # refined output meaningless. Use the coarse result directly by
            # setting start_t=1; after coarse training use normal refinement.
            validation_start_t = 1 if is_coarse_stage else refine_start_t
            validation_model = ema_refiner if not is_coarse_stage else refiner
            validation_metrics = validate(
                separator=separator,
                refiner=validation_model,
                schedule=schedule,
                encoder=encoder,
                generator=generator,
                normalizer=normalizer,
                fixed_e=fixed_e,
                fixed_d=fixed_d,
                fixed_initial_noise=fixed_initial_noise,
                epoch=epoch,
                save_root=save_root,
                validation_snr_db=validation_snr_db,
                refine_start_t=validation_start_t,
                ddim_steps=ddim_steps,
            )
            append_csv(
                validation_history_path,
                {"epoch": epoch, "stage": stage_name, **validation_metrics},
            )
            checkpoint["validation"] = validation_metrics
            torch.save(
                checkpoint,
                os.path.join(model_root, f"DDPM_coarse_to_diffusion_v5_ep{epoch}.pth"),
            )
            print("Validation:", validation_metrics)

            # Select the best model by actual reconstruction quality, while
            # still penalizing a poor mixture decomposition.
            if is_coarse_stage:
                score = 0.5 * (
                    validation_metrics["coarse_image_l1_e_to_oracle"]
                    + validation_metrics["coarse_image_l1_d_to_oracle"]
                ) + 0.05 * validation_metrics["coarse_mixture_mse_clean"]
            else:
                score = 0.5 * (
                    validation_metrics["refined_image_l1_e_to_oracle"]
                    + validation_metrics["refined_image_l1_d_to_oracle"]
                ) + 0.05 * validation_metrics["refined_mixture_mse_clean"]

            if score < best_score:
                best_score = score
                torch.save(checkpoint, best_path)
                print(f"Saved best checkpoint | score={best_score:.6f}")

    print("Training completed.")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Best checkpoint:   {best_path}")
    print(f"Validation images: {save_root}")


if __name__ == "__main__":
    main()
