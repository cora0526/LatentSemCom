"""
Compare two latent-separation systems under the same test images and SNRs.

Method A: Shared VAE Encoder -> DDPM separator -> Conditional VAE Decoder
Method B: VAE-GAN Encoder     -> DDPM separator -> Conditional GAN Generator

Required files in your project:
    ./model.py
        Generator_Satellite_Conditional

    ./save_model/shared_vae_satellite_latest.pth
    ./save_model/DDPM_vae_latent_separator_latest.pth

    ./save_model/joint_vae_gan_satellite_latest.pth
    ./save_model/DDPM_vae_gan_latent_separator_latest.pth

Outputs:
    ./save_data/compare_VAE_vs_VAEGAN_DDPM/
        metrics_long.csv
        comparison_summary.csv
        qualitative_SNR*.png
        PSNR_vs_SNR.png
        SSIM_vs_SNR.png
        Latent_Cosine_vs_SNR.png
        Latent_MSE_vs_SNR.png
        Consistency_MSE_vs_SNR.png

Notes:
- Both methods are tested on exactly the same image pairs.
- The transmitted latent is encoder output mu, not sampled z.
- Set max_test_pairs higher for final paper results. The default is intentionally
  modest because full 400-step DDPM sampling is expensive.
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

from model import Generator_Satellite_Conditional


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    z_dim: int = 256
    label_dim: int = 64
    num_classes: int = 2
    diffusion_steps: int = 400

    batch_size: int = 16
    num_workers: int = 0

    # Increase to 500/1000+ pairs for final paper tables.
    max_test_pairs: int = 32

    snr_values: Tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30)

    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    vae_checkpoint: str = "./save_model/shared_vae_satellite_latest.pth"
    vae_ddpm_checkpoint: str = "./save_model/DDPM_vae_latent_separator_latest.pth"

    vaegan_checkpoint: str = "./save_model/joint_vae_gan_satellite_latest.pth"
    vaegan_ddpm_checkpoint: str = "./save_model/DDPM_vae_gan_latent_separator_latest.pth"

    save_root: str = "./save_data/compare_VAE_vs_VAEGAN_DDPM"

    # Save qualitative grids at these SNRs.
    qualitative_snrs: Tuple[float, ...] = (0, 10, 20, 30)
    qualitative_count: int = 8


# ============================================================
# Dataset
# ============================================================
class SatelliteDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform
        if not self.files:
            raise RuntimeError(f"No matching DeepGlobe images found in:\n{root}")

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


# ============================================================
# Shared encoder used by both supplied training files
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

    def forward(self, x: torch.Tensor):
        h = self.features(x).flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 10.0)
        return mu, logvar


# ============================================================
# Conditional VAE decoder
# ============================================================
class ConditionalVAEDecoder(nn.Module):
    def __init__(self, z_dim: int = 256, num_classes: int = 2, label_dim: int = 64):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.fc = nn.Sequential(
            nn.Linear(z_dim + label_dim, 512 * 4 * 4),
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


# ============================================================
# DDPM model: same architecture used in both training scripts
# ============================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
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
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, z_t, timesteps, labels, cond):
        t_emb = self.time_embedding(timesteps)
        y_emb = self.label_embedding(labels)
        h = torch.cat([z_t, cond, t_emb, y_emb], dim=1)
        return self.output_layer(self.blocks(self.input_layer(h)))


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int = 400, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars))
        self.register_buffer(
            "sqrt_recipm1_alpha_bars", torch.sqrt(1.0 / alpha_bars - 1.0)
        )
        self.register_buffer(
            "posterior_variance", torch.clamp(posterior_variance, min=1e-20)
        )

    @staticmethod
    def extract(values: torch.Tensor, t: torch.Tensor, x_shape):
        out = values.gather(0, t)
        return out.reshape(t.size(0), *((1,) * (len(x_shape) - 1)))

    def predict_x0_from_eps(self, x_t, t, eps):
        recip = self.extract(self.sqrt_recip_alpha_bars, t, x_t.shape)
        recipm1 = self.extract(self.sqrt_recipm1_alpha_bars, t, x_t.shape)
        return recip * x_t - recipm1 * eps


# ============================================================
# Model bundle and loading
# ============================================================
@dataclass
class MethodBundle:
    name: str
    encoder: nn.Module
    decoder: nn.Module
    ddpm: nn.Module
    schedule: DiffusionSchedule


def freeze(model: nn.Module):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def require_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required checkpoint does not exist:\n{path}")


def load_ddpm(path: str, config: Config, device: str):
    require_file(path)
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)

    model = LatentConditionalDDPM(
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
    ).to(device)
    model.load_state_dict(state, strict=True)
    freeze(model)

    saved_steps = checkpoint.get("config", {}).get(
        "diffusion_steps", config.diffusion_steps
    )
    if int(saved_steps) != config.diffusion_steps:
        raise ValueError(
            f"Checkpoint {path} used diffusion_steps={saved_steps}, "
            f"but Config.diffusion_steps={config.diffusion_steps}."
        )

    return model


def load_vae_method(config: Config, device: str):
    require_file(config.vae_checkpoint)
    checkpoint = torch.load(config.vae_checkpoint, map_location=device)
    if "encoder" not in checkpoint or "decoder" not in checkpoint:
        raise KeyError("VAE checkpoint must contain 'encoder' and 'decoder'.")

    encoder = SharedVAEEncoder(config.z_dim).to(device)
    decoder = ConditionalVAEDecoder(
        config.z_dim, config.num_classes, config.label_dim
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)

    ddpm = load_ddpm(config.vae_ddpm_checkpoint, config, device)
    schedule = DiffusionSchedule(config.diffusion_steps).to(device)

    freeze(encoder)
    freeze(decoder)
    return MethodBundle("VAE-DDPM-VAE", encoder, decoder, ddpm, schedule)


def load_vaegan_method(config: Config, device: str):
    require_file(config.vaegan_checkpoint)
    checkpoint = torch.load(config.vaegan_checkpoint, map_location=device)
    if "encoder" not in checkpoint or "generator" not in checkpoint:
        raise KeyError("VAE-GAN checkpoint must contain 'encoder' and 'generator'.")

    encoder = SharedVAEEncoder(config.z_dim).to(device)
    decoder = Generator_Satellite_Conditional(
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
        out_ch=3,
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    decoder.load_state_dict(checkpoint["generator"], strict=True)

    ddpm = load_ddpm(config.vaegan_ddpm_checkpoint, config, device)
    schedule = DiffusionSchedule(config.diffusion_steps).to(device)

    freeze(encoder)
    freeze(decoder)
    return MethodBundle("VAE-DDPM-GAN", encoder, decoder, ddpm, schedule)


# ============================================================
# Channel and sampling
# ============================================================
def add_awgn(signal: torch.Tensor, snr_db: float, noise_unit: torch.Tensor):
    if math.isinf(float(snr_db)):
        return signal
    power = signal.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
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
):
    """Full ancestral sampling using the same 400-step process as training validation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = torch.randn(cond.size(0), z_dim, device=cond.device)
    for step in reversed(range(schedule.timesteps)):
        t = torch.full(
            (cond.size(0),), step, dtype=torch.long, device=cond.device
        )
        pred_eps = model(x, t, labels, cond)
        pred_x0 = schedule.predict_x0_from_eps(x, t, pred_eps)

        beta_t = schedule.extract(schedule.betas, t, x.shape)
        alpha_t = schedule.extract(schedule.alphas, t, x.shape)
        alpha_bar_t = schedule.extract(schedule.alpha_bars, t, x.shape)
        alpha_bar_prev = schedule.extract(schedule.alpha_bars_prev, t, x.shape)

        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef_xt = (
            (1.0 - alpha_bar_prev)
            * torch.sqrt(alpha_t)
            / (1.0 - alpha_bar_t)
        )
        mean = coef_x0 * pred_x0 + coef_xt * x

        if step > 0:
            variance = schedule.extract(schedule.posterior_variance, t, x.shape)
            x = mean + torch.sqrt(variance) * torch.randn_like(x)
        else:
            x = mean
    return x


# ============================================================
# Metrics
# ============================================================
def to_01(images: torch.Tensor):
    return ((images.detach().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def batch_psnr(pred: torch.Tensor, target: torch.Tensor):
    pred = to_01(pred)
    target = to_01(target)
    mse = (pred - target).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()


def batch_ssim(pred: torch.Tensor, target: torch.Tensor):
    pred_np = to_01(pred).permute(0, 2, 3, 1).cpu().numpy()
    target_np = to_01(target).permute(0, 2, 3, 1).cpu().numpy()
    values = []
    for prediction, reference in zip(pred_np, target_np):
        values.append(
            structural_similarity(
                reference,
                prediction,
                data_range=1.0,
                channel_axis=-1,
            )
        )
    return np.asarray(values, dtype=np.float64)


def latent_mse_per_sample(pred: torch.Tensor, target: torch.Tensor):
    return (pred - target).pow(2).mean(dim=1).detach().cpu().numpy()


def latent_cosine_per_sample(pred: torch.Tensor, target: torch.Tensor):
    return F.cosine_similarity(pred, target, dim=1).detach().cpu().numpy()


def consistency_per_sample(pred_e, pred_d, clean_mix):
    return (
        (pred_e + pred_d - clean_mix)
        .pow(2)
        .mean(dim=1)
        .detach()
        .cpu()
        .numpy()
    )


class MetricAccumulator:
    def __init__(self):
        self.values: Dict[str, List[float]] = {}

    def add(self, name: str, values):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        self.values.setdefault(name, []).extend(array.tolist())

    def mean(self, name: str):
        values = self.values.get(name, [])
        return float(np.mean(values)) if values else float("nan")

    def count(self):
        if not self.values:
            return 0
        return len(next(iter(self.values.values())))


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_batch(
    bundle: MethodBundle,
    image_e: torch.Tensor,
    image_d: torch.Tensor,
    snr_db: float,
    noise_unit: torch.Tensor,
    sample_seed: int,
):
    batch_size = image_e.size(0)
    label_e = torch.zeros(batch_size, dtype=torch.long, device=image_e.device)
    label_d = torch.ones(batch_size, dtype=torch.long, device=image_e.device)

    z_e, _ = bundle.encoder(image_e)
    z_d, _ = bundle.encoder(image_d)
    clean_mix = z_e + z_d
    received = add_awgn(clean_mix, snr_db, noise_unit)

    pred_e = sample_ddpm(
        bundle.ddpm,
        bundle.schedule,
        received,
        label_e,
        z_e.size(1),
        seed=sample_seed,
    )
    pred_d = sample_ddpm(
        bundle.ddpm,
        bundle.schedule,
        received,
        label_d,
        z_d.size(1),
        seed=sample_seed + 1,
    )

    oracle_e = bundle.decoder(z_e, label_e)
    oracle_d = bundle.decoder(z_d, label_d)
    separated_e = bundle.decoder(pred_e, label_e)
    separated_d = bundle.decoder(pred_d, label_d)

    return {
        "z_e": z_e,
        "z_d": z_d,
        "clean_mix": clean_mix,
        "received": received,
        "pred_e": pred_e,
        "pred_d": pred_d,
        "oracle_e": oracle_e,
        "oracle_d": oracle_d,
        "separated_e": separated_e,
        "separated_d": separated_d,
    }


def add_domain_metrics(
    accumulator: MetricAccumulator,
    prefix: str,
    separated: torch.Tensor,
    oracle: torch.Tensor,
    real: torch.Tensor,
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
):
    accumulator.add(f"{prefix}_psnr_real", batch_psnr(separated, real))
    accumulator.add(f"{prefix}_ssim_real", batch_ssim(separated, real))
    accumulator.add(f"{prefix}_psnr_oracle", batch_psnr(separated, oracle))
    accumulator.add(f"{prefix}_ssim_oracle", batch_ssim(separated, oracle))
    accumulator.add(f"{prefix}_oracle_psnr_real", batch_psnr(oracle, real))
    accumulator.add(f"{prefix}_oracle_ssim_real", batch_ssim(oracle, real))
    accumulator.add(f"{prefix}_latent_mse", latent_mse_per_sample(pred_z, target_z))
    accumulator.add(
        f"{prefix}_latent_cosine", latent_cosine_per_sample(pred_z, target_z)
    )


def save_qualitative_grid(
    save_path: str,
    real_e: torch.Tensor,
    real_d: torch.Tensor,
    result_vae: dict,
    result_vaegan: dict,
    count: int,
):
    count = min(count, real_e.size(0), real_d.size(0))

    # Every row block has the same sample order.
    rows = [
        real_e[:count],
        result_vae["oracle_e"][:count],
        result_vae["separated_e"][:count],
        result_vaegan["oracle_e"][:count],
        result_vaegan["separated_e"][:count],
        real_d[:count],
        result_vae["oracle_d"][:count],
        result_vae["separated_d"][:count],
        result_vaegan["oracle_d"][:count],
        result_vaegan["separated_d"][:count],
    ]
    grid = make_grid(torch.cat(rows, dim=0), nrow=count, padding=2, normalize=True, value_range=(-1, 1))
    save_image(grid, save_path)


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary_row(method: str, snr_db: float, acc: MetricAccumulator):
    e_psnr = acc.mean("E_psnr_real")
    d_psnr = acc.mean("D_psnr_real")
    e_ssim = acc.mean("E_ssim_real")
    d_ssim = acc.mean("D_ssim_real")
    e_mse = acc.mean("E_latent_mse")
    d_mse = acc.mean("D_latent_mse")
    e_cos = acc.mean("E_latent_cosine")
    d_cos = acc.mean("D_latent_cosine")

    return {
        "method": method,
        "snr_db": snr_db,
        "num_pairs": acc.count(),
        "E_psnr_real": e_psnr,
        "D_psnr_real": d_psnr,
        "avg_psnr_real": np.mean([e_psnr, d_psnr]),
        "E_ssim_real": e_ssim,
        "D_ssim_real": d_ssim,
        "avg_ssim_real": np.mean([e_ssim, d_ssim]),
        "E_psnr_oracle": acc.mean("E_psnr_oracle"),
        "D_psnr_oracle": acc.mean("D_psnr_oracle"),
        "E_ssim_oracle": acc.mean("E_ssim_oracle"),
        "D_ssim_oracle": acc.mean("D_ssim_oracle"),
        "E_oracle_psnr_real": acc.mean("E_oracle_psnr_real"),
        "D_oracle_psnr_real": acc.mean("D_oracle_psnr_real"),
        "E_oracle_ssim_real": acc.mean("E_oracle_ssim_real"),
        "D_oracle_ssim_real": acc.mean("D_oracle_ssim_real"),
        "E_latent_mse": e_mse,
        "D_latent_mse": d_mse,
        "avg_latent_mse": np.mean([e_mse, d_mse]),
        "E_latent_cosine": e_cos,
        "D_latent_cosine": d_cos,
        "avg_latent_cosine": np.mean([e_cos, d_cos]),
        "consistency_mse": acc.mean("consistency_mse"),
    }


def plot_metric(rows: List[dict], metric: str, ylabel: str, save_path: str):
    methods = sorted({row["method"] for row in rows})
    plt.figure(figsize=(7, 5))
    for method in methods:
        selected = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: float(row["snr_db"]),
        )
        x = [float(row["snr_db"]) for row in selected]
        y = [float(row[metric]) for row in selected]
        plt.plot(x, y, marker="o", label=method)
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


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

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    eurosat = datasets.ImageFolder(config.eurosat_root, transform=transform)
    deepglobe = SatelliteDataset(config.deepglobe_root, transform=transform)

    pair_count = min(config.max_test_pairs, len(eurosat), len(deepglobe))
    # Fixed deterministic subset for a reproducible comparison.
    generator = torch.Generator().manual_seed(config.seed)
    e_indices = torch.randperm(len(eurosat), generator=generator)[:pair_count].tolist()
    d_indices = torch.randperm(len(deepglobe), generator=generator)[:pair_count].tolist()

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

    print("Loading VAE-DDPM-VAE...")
    vae_method = load_vae_method(config, device)
    print("Loading VAE-DDPM-GAN...")
    vaegan_method = load_vaegan_method(config, device)
    methods = [vae_method, vaegan_method]

    rows: List[dict] = []

    for snr_index, snr_db in enumerate(config.snr_values):
        print("=" * 90)
        print(f"Evaluating SNR={snr_db} dB")

        accumulators = {method.name: MetricAccumulator() for method in methods}
        qualitative_saved = False

        for batch_index, (batch_e, batch_d) in enumerate(zip(loader_e, loader_d)):
            image_e = get_image(batch_e).to(device, non_blocking=True)
            image_d = get_image(batch_d).to(device, non_blocking=True)
            current = min(image_e.size(0), image_d.size(0))
            image_e = image_e[:current]
            image_d = image_d[:current]

            # Same standard-normal channel realization for both methods.
            channel_seed = config.seed + snr_index * 10000 + batch_index
            torch.manual_seed(channel_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(channel_seed)
            noise_unit = torch.randn(current, config.z_dim, device=device)

            method_results = {}
            for method_index, method in enumerate(methods):
                # Same initial/reverse sampling seed across methods for fairness.
                sample_seed = config.seed + snr_index * 100000 + batch_index * 10
                result = evaluate_batch(
                    method,
                    image_e,
                    image_d,
                    snr_db,
                    noise_unit,
                    sample_seed,
                )
                method_results[method.name] = result

                acc = accumulators[method.name]
                add_domain_metrics(
                    acc,
                    "E",
                    result["separated_e"],
                    result["oracle_e"],
                    image_e,
                    result["pred_e"],
                    result["z_e"],
                )
                add_domain_metrics(
                    acc,
                    "D",
                    result["separated_d"],
                    result["oracle_d"],
                    image_d,
                    result["pred_d"],
                    result["z_d"],
                )
                acc.add(
                    "consistency_mse",
                    consistency_per_sample(
                        result["pred_e"],
                        result["pred_d"],
                        result["clean_mix"],
                    ),
                )

            if (
                not qualitative_saved
                and float(snr_db) in [float(x) for x in config.qualitative_snrs]
            ):
                filename = f"qualitative_SNR{snr_db:g}.png"
                save_qualitative_grid(
                    os.path.join(config.save_root, filename),
                    image_e,
                    image_d,
                    method_results["VAE-DDPM-VAE"],
                    method_results["VAE-DDPM-GAN"],
                    config.qualitative_count,
                )
                qualitative_saved = True

            print(
                f"SNR={snr_db:>4} | batch {batch_index + 1}/{len(loader_e)} complete"
            )

        for method in methods:
            row = build_summary_row(method.name, snr_db, accumulators[method.name])
            rows.append(row)
            print(
                f"{method.name}: "
                f"PSNR={row['avg_psnr_real']:.3f}, "
                f"SSIM={row['avg_ssim_real']:.4f}, "
                f"LatCos={row['avg_latent_cosine']:.4f}, "
                f"LatMSE={row['avg_latent_mse']:.4f}"
            )

        # Save progressively in case a later SNR evaluation is interrupted.
        write_csv(os.path.join(config.save_root, "metrics_long.csv"), rows)

    # Create a wide comparison table: one row per SNR.
    wide_rows = []
    for snr_db in config.snr_values:
        vae_row = next(
            row
            for row in rows
            if row["method"] == "VAE-DDPM-VAE" and float(row["snr_db"]) == float(snr_db)
        )
        gan_row = next(
            row
            for row in rows
            if row["method"] == "VAE-DDPM-GAN" and float(row["snr_db"]) == float(snr_db)
        )
        wide_rows.append(
            {
                "snr_db": snr_db,
                "VAE_avg_PSNR": vae_row["avg_psnr_real"],
                "VAEGAN_avg_PSNR": gan_row["avg_psnr_real"],
                "PSNR_difference_VAE_minus_VAEGAN": vae_row["avg_psnr_real"]
                - gan_row["avg_psnr_real"],
                "VAE_avg_SSIM": vae_row["avg_ssim_real"],
                "VAEGAN_avg_SSIM": gan_row["avg_ssim_real"],
                "SSIM_difference_VAE_minus_VAEGAN": vae_row["avg_ssim_real"]
                - gan_row["avg_ssim_real"],
                "VAE_avg_latent_MSE": vae_row["avg_latent_mse"],
                "VAEGAN_avg_latent_MSE": gan_row["avg_latent_mse"],
                "VAE_avg_latent_cosine": vae_row["avg_latent_cosine"],
                "VAEGAN_avg_latent_cosine": gan_row["avg_latent_cosine"],
                "VAE_consistency_MSE": vae_row["consistency_mse"],
                "VAEGAN_consistency_MSE": gan_row["consistency_mse"],
            }
        )

    write_csv(os.path.join(config.save_root, "comparison_summary.csv"), wide_rows)

    plot_metric(
        rows,
        "avg_psnr_real",
        "Average PSNR vs. real image (dB)",
        os.path.join(config.save_root, "PSNR_vs_SNR.png"),
    )
    plot_metric(
        rows,
        "avg_ssim_real",
        "Average SSIM vs. real image",
        os.path.join(config.save_root, "SSIM_vs_SNR.png"),
    )
    plot_metric(
        rows,
        "avg_latent_cosine",
        "Average latent cosine similarity",
        os.path.join(config.save_root, "Latent_Cosine_vs_SNR.png"),
    )
    plot_metric(
        rows,
        "avg_latent_mse",
        "Average latent MSE",
        os.path.join(config.save_root, "Latent_MSE_vs_SNR.png"),
    )
    plot_metric(
        rows,
        "consistency_mse",
        "Mixture consistency MSE",
        os.path.join(config.save_root, "Consistency_MSE_vs_SNR.png"),
    )

    print("=" * 90)
    print("Comparison completed.")
    print(f"Results: {config.save_root}")
    print("For final paper results, increase Config.max_test_pairs.")


if __name__ == "__main__":
    main()
