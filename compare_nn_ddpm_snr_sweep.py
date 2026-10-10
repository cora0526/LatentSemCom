"""SNR 0..30 dB comparison: Clean / No separation / NN / DDPM.
Run: python compare_nn_ddpm_snr_sweep.py
Dependencies: torch torchvision numpy Pillow matplotlib.
Standalone: no imports from other project Python files.
Default: 500 pairs, SNR 0,5,...,30 dB, Rayleigh amplitude fading + AWGN.
Database overlap is ALLOWED by default; use --exclude-bank-images for disjoint
paths. This checks paths only, not renamed/copied or same-source crop images.
The bank should be made by build_latent_database.py with the SAME checkpoint.
NN searches the raw received mixture directly within each dataset label.
No channel equalization, pair-sum search or latent normalization is performed.
DDPM model uses the original 50-step deterministic DDIM inference by default.
PSNR is computed per image on [0,1], then averaged per dataset (no SSIM).
Noise variance uses each faded mixture's power; channel and unit-noise draws
are shared across SNRs, and all methods receive the same observation per SNR.
"""
import argparse
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hashlib
import json
from pathlib import Path
import glob
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    z_dim: int = 256
    label_dim: int = 64
    num_classes: int = 2
    diffusion_steps: int = 400
    sampling_method: str = "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    scratch_vaegan_checkpoint: str = "./save_model/vae_gan_from_scratch_best.pth"
    scratch_vaegan_ddpm_checkpoint: str = "./save_model/DDPM_vaegan_scratch_latent_separator_best.pth"
    save_root: str = "./save_data/ddpm_separation_demo_rayleigh_5dB"
    snr_db: float = 5.0
    num_test_pairs: int = 10


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



def checkpoint_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))



METHODS = ("clean", "no_sep", "nn", "ddpm")
NAMES = {"clean": "Clean reconstruction", "no_sep": "No separation",
         "nn": "Nearest-neighbor retrieval", "ddpm": "Proposed DDPM"}


class PairedImages(Dataset):
    def __init__(self, paths_e, paths_d, pairs):
        self.paths_e, self.paths_d, self.pairs = paths_e, paths_d, pairs
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ei, di = self.pairs[index]
        with Image.open(self.paths_e[ei]) as im:
            e = self.transform(im.convert("RGB"))
        with Image.open(self.paths_d[di]) as im:
            d = self.transform(im.convert("RGB"))
        return e, d


@torch.inference_mode()
def nearest_batch(query, latents, row_ids, chunk_size):
    # Direct squared differences avoid cancellation from norm/matmul formulas.
    # Process each query within a chunk to avoid allocating [B,chunk,D].
    best = torch.full((len(query),), float("inf"), device=query.device)
    chosen = torch.full((len(query),), -1, dtype=torch.long, device=query.device)
    for start in range(0, len(row_ids), chunk_size):
        ids = row_ids[start:start + chunk_size].to(query.device)
        candidates = latents[row_ids[start:start + chunk_size]].to(query.device)
        for i in range(len(query)):
            distances = (candidates - query[i]).square().sum(dim=1)
            value, index = distances.min(dim=0)
            better = value < best[i]
            best[i] = torch.where(better, value, best[i])
            chosen[i] = torch.where(better, ids[index], chosen[i])
    chosen = chosen.cpu()
    if (chosen < 0).any():
        raise RuntimeError("No finite nearest neighbor found")
    return latents[chosen].to(query.device), chosen


def psnr_per_image(reference, output):
    reference = (reference.clamp(-1, 1) + 1) / 2
    output = (output.clamp(-1, 1) + 1) / 2
    mse = (reference - output).square().flatten(1).mean(1)
    # Numerical floor gives an upper reporting bound of 120 dB.
    return -10 * torch.log10(mse.clamp_min(1e-12))


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot_results(rows, out, channel, overlap_allowed):
    styles = {"clean": ("#777777", "--", None),
              "no_sep": ("#d58518", "-", "s"),
              "nn": ("#2878b5", "-", "^"),
              "ddpm": ("#c6374a", "-", "o")}
    values = [r[f"{m}_psnr_db"] for r in rows for m in METHODS]
    low, high = min(values), max(values)
    margin = max((high-low)*0.08, 0.5)
    mode = "Bank overlap allowed" if overlap_allowed else "Bank paths excluded"
    for dataset in ("EuroSAT", "DeepGlobe"):
        selected = sorted([r for r in rows if r["dataset"] == dataset],
                          key=lambda r: r["snr_db"])
        fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
        for method in METHODS:
            color, linestyle, marker = styles[method]
            ax.plot([r["snr_db"] for r in selected],
                    [r[f"{method}_psnr_db"] for r in selected],
                    label=NAMES[method], color=color, linestyle=linestyle,
                    marker=marker, linewidth=2, markersize=6)
        ax.set_xlabel("SNR (dB)", fontsize=14)
        ax.set_ylabel("Mean PSNR (dB)", fontsize=14)
        ax.set_title(f"{dataset} | {channel} + AWGN\n{mode}", fontsize=14)
        ax.set_xticks([r["snr_db"] for r in selected])
        ax.set_ylim(low-margin, high+margin)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=10)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{dataset}_PSNR_vs_SNR.{ext}", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", default="./save_data/latent_database/latent_database.pt")
    parser.add_argument("--eurosat-root", default=Config.eurosat_root)
    parser.add_argument("--deepglobe-root", default=Config.deepglobe_root)
    parser.add_argument("--model-dir", default="./save_model")
    parser.add_argument("--output-dir", default="./save_data/nn_ddpm_snr_comparison")
    parser.add_argument("--num-pairs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--nn-chunk-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--channel", choices=["rayleigh", "awgn"], default="rayleigh")
    parser.add_argument("--exclude-bank-images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-method", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--diffusion-steps", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.num_pairs, args.batch_size, args.nn_chunk_size) <= 0 or args.num_workers < 0:
        parser.error("Counts and batch/chunk sizes must be positive; num-workers >= 0.")
    if args.diffusion_steps < 2 or not 2 <= args.ddim_steps <= args.diffusion_steps:
        parser.error("Require 2 <= ddim-steps <= diffusion-steps.")
    if not all(math.isfinite(x) for x in args.snrs):
        parser.error("SNR values must be finite")
    args.snrs = sorted(set(args.snrs))
    config = Config(seed=args.seed, z_dim=256, diffusion_steps=args.diffusion_steps,
                    sampling_method=args.sampling_method, ddim_steps=args.ddim_steps,
                    eurosat_root=args.eurosat_root, deepglobe_root=args.deepglobe_root,
                    scratch_vaegan_checkpoint=os.path.join(args.model_dir, "vae_gan_from_scratch_best.pth"),
                    scratch_vaegan_ddpm_checkpoint=os.path.join(args.model_dir, "DDPM_vaegan_scratch_latent_separator_best.pth"))
    for path in [args.bank, config.scratch_vaegan_checkpoint, config.scratch_vaegan_ddpm_checkpoint]:
        require_file(path)
    out = Path(args.output_dir)
    if not args.overwrite and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output is not empty: {out}. Change --output-dir or use --overwrite.")
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    latents, labels, paths = bank["latents"].float().contiguous(), bank["labels"], bank["image_paths"]
    if (latents.ndim != 2 or labels.ndim != 1 or len(latents) != len(labels)
            or len(paths) != len(labels) or not torch.isfinite(latents).all()
            or not ((labels == 0) | (labels == 1)).all()):
        raise ValueError("Invalid bank latents, labels or paths")
    if bank["metadata"].get("checkpoint_sha256") != checkpoint_hash(config.scratch_vaegan_checkpoint):
        raise ValueError("Bank was built with a different VAE-GAN checkpoint; rebuild it.")
    config.z_dim = latents.shape[1]
    bank_ids = [torch.where(labels == label)[0] for label in (0, 1)]
    if any(len(ids) == 0 for ids in bank_ids):
        raise ValueError("Both dataset labels must have candidates")
    bank_paths = {canonical(p) for p in paths}
    paths_e = [p for p, _ in datasets.ImageFolder(args.eurosat_root).samples]
    paths_d = SatelliteDataset(args.deepglobe_root).files
    eligible = [[i for i, p in enumerate(group)
                 if not args.exclude_bank_images or canonical(p) not in bank_paths]
                for group in (paths_e, paths_d)]
    if min(map(len, eligible)) < args.num_pairs:
        raise ValueError(f"Only {len(eligible[0])}/{len(eligible[1])} eligible images. "
                         "Reduce --num-pairs or supply separate test folders. "
                         "Omit --exclude-bank-images if overlap is intended.")
    rng = random.Random(args.seed)
    pairs = list(zip(rng.sample(eligible[0], args.num_pairs),
                     rng.sample(eligible[1], args.num_pairs)))
    from torch.utils.data import DataLoader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    loader = DataLoader(PairedImages(paths_e, paths_d, pairs), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=device == "cuda")
    encoder, generator, ddpm, schedule = load_models(config, device)
    out.mkdir(parents=True, exist_ok=True)
    pair_rows = [{"pair": i+1, "eurosat_path": str(Path(paths_e[ei]).resolve()),
                  "deepglobe_path": str(Path(paths_d[di]).resolve()),
                  "eurosat_in_bank": canonical(paths_e[ei]) in bank_paths,
                  "deepglobe_in_bank": canonical(paths_d[di]) in bank_paths}
                 for i, (ei, di) in enumerate(pairs)]
    write_csv(out / "test_pairs.csv", pair_rows)
    info = {"settings": vars(args), "device": device, "bank_metadata": bank["metadata"],
            "eurosat_overlap_count": sum(r["eurosat_in_bank"] for r in pair_rows),
            "deepglobe_overlap_count": sum(r["deepglobe_in_bank"] for r in pair_rows),
            "overlap_check": "paths only; not image content or encoder training history",
            "channel": "independent unit-mean-power Rayleigh amplitudes per user and image" if args.channel == "rayleigh" else "unit gains",
            "snr_definition": "per-image faded-mixture power / expected AWGN power",
            "psnr": "mean of per-image PSNR on clamped [0,1]; MSE floor 1e-12",
            "nn": "per-label Euclidean nearest neighbor of raw received mixture",
            "shared_across_snrs": "same pairs, fading, unit noise, DDPM initial seed"}
    (out / "run_config.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Device: {device}; pairs: {len(pairs)}; bank size: {len(latents)}")
    print(f"Test images also in bank: EuroSAT={info['eurosat_overlap_count']}, DeepGlobe={info['deepglobe_overlap_count']}")
    summaries, details = [], []
    with torch.inference_mode():
        for snr in args.snrs:
            sums = {dataset: {method: 0.0 for method in METHODS} for dataset in ("EuroSAT", "DeepGlobe")}
            offset = 0
            for batch_index, (image_e, image_d) in enumerate(loader):
                images = [image_e.to(device), image_d.to(device)]
                b = len(image_e)
                target_labels = [torch.full((b,), label, dtype=torch.long, device=device) for label in (0, 1)]
                zs = [encoder(im)[0] for im in images]
                # Reset channel RNG per batch: same fading and unit-noise for every SNR.
                torch.manual_seed(args.seed + 100_000 + batch_index)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed + 100_000 + batch_index)
                if args.channel == "rayleigh":
                    gains = [torch.sqrt(torch.randn(b, 2, device=device).square().sum(1, keepdim=True)/2) for _ in (0, 1)]
                else:
                    gains = [torch.ones(b, 1, device=device) for _ in (0, 1)]
                mixed = gains[0]*zs[0] + gains[1]*zs[1]
                received = add_awgn(mixed, snr, torch.randn_like(mixed))
                for k, dataset in enumerate(("EuroSAT", "DeepGlobe")):
                    nearest, selected = nearest_batch(received, latents, bank_ids[k], args.nn_chunk_size)
                    predicted = sample_ddpm(ddpm, schedule, received, target_labels[k], config.z_dim,
                                            args.seed + 1_000_000 + batch_index*2 + k,
                                            config.sampling_method, config.ddim_steps, config.ddim_eta)
                    if not torch.isfinite(predicted).all():
                        raise RuntimeError(f"Non-finite DDPM latent at SNR={snr}, batch={batch_index}")
                    candidates = {"clean": zs[k], "no_sep": received, "nn": nearest, "ddpm": predicted}
                    scores = {}
                    for method, z in candidates.items():
                        output = generator(z, target_labels[k])
                        if not torch.isfinite(output).all():
                            raise RuntimeError(f"Non-finite output: {dataset}/{method}")
                        values = psnr_per_image(images[k], output).cpu().tolist()
                        scores[method] = values
                        sums[dataset][method] += sum(values)
                    for j in range(b):
                        row = {"snr_db": snr, "dataset": dataset, "pair": offset+j+1,
                               "nn_bank_row": int(selected[j]), "nn_image_path": paths[int(selected[j])]}
                        row.update({f"{method}_psnr_db": scores[method][j] for method in METHODS})
                        details.append(row)
                offset += b
                print(f"SNR={snr:g} dB: {offset}/{len(pairs)} pairs", flush=True)
            for dataset in ("EuroSAT", "DeepGlobe"):
                row = {"snr_db": snr, "dataset": dataset, "num_pairs": len(pairs)}
                row.update({f"{method}_psnr_db": sums[dataset][method]/len(pairs) for method in METHODS})
                summaries.append(row)
                print(dataset + " | " + " | ".join(f"{m}={row[f'{m}_psnr_db']:.3f} dB" for m in METHODS))
            # Save completed SNR points so a later interruption does not lose them.
            write_csv(out / "psnr_summary.csv", summaries)
            write_csv(out / "psnr_per_image.csv", details)
    plot_results(summaries, out, "Rayleigh" if args.channel == "rayleigh" else "Unit gain", not args.exclude_bank_images)
    print("Done. Charts, CSVs and run settings:", out.resolve())


if __name__ == "__main__":
    main()
