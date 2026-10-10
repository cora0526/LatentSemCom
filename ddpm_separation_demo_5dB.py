"""Save ten paired inputs and DDPM reconstructions with Rayleigh fading at 5 dB.

Run from the project folder containing save_model, or supply --model-dir.
Requires torch, torchvision, Pillow, and numpy.
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eurosat-root", default=Config.eurosat_root)
    parser.add_argument("--deepglobe-root", default=Config.deepglobe_root)
    parser.add_argument("--model-dir", default="./save_model")
    parser.add_argument("--output-dir", default=Config.save_root)
    args = parser.parse_args()
    config = Config(
        eurosat_root=args.eurosat_root,
        deepglobe_root=args.deepglobe_root,
        scratch_vaegan_checkpoint=os.path.join(args.model_dir, "vae_gan_from_scratch_best.pth"),
        scratch_vaegan_ddpm_checkpoint=os.path.join(args.model_dir, "DDPM_vaegan_scratch_latent_separator_best.pth"),
        save_root=args.output_dir,
    )
    device = "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    eurosat = datasets.ImageFolder(config.eurosat_root, transform=transform)
    deepglobe = SatelliteDataset(config.deepglobe_root, transform=transform)
    if len(eurosat) < config.num_test_pairs or len(deepglobe) < config.num_test_pairs:
        raise ValueError("Each dataset must contain at least 10 images.")
    indices = torch.Generator().manual_seed(config.seed)
    e_indices = torch.randperm(len(eurosat), generator=indices)[:10].tolist()
    d_indices = torch.randperm(len(deepglobe), generator=indices)[:10].tolist()
    encoder, generator, ddpm, schedule = load_models(config, device)
    os.makedirs(config.save_root, exist_ok=True)

    with torch.inference_mode():
        for i, (ei, di) in enumerate(zip(e_indices, d_indices), start=1):
            image_e = get_image(eurosat[ei]).unsqueeze(0).to(device)
            image_d = get_image(deepglobe[di]).unsqueeze(0).to(device)
            label_e = torch.zeros(1, dtype=torch.long, device=device)
            label_d = torch.ones(1, dtype=torch.long, device=device)
            z_e, _ = encoder(image_e)
            z_d, _ = encoder(image_d)
            torch.manual_seed(config.seed + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.seed + i)
            # One independent, unit-average-power Rayleigh amplitude per user
            # and image pair. The real-valued latent uses amplitude fading.
            h_e = torch.sqrt(torch.randn(1, 2, device=device).square().sum(dim=1, keepdim=True) / 2)
            h_d = torch.sqrt(torch.randn(1, 2, device=device).square().sum(dim=1, keepdim=True) / 2)
            faded_mix = h_e * z_e + h_d * z_d
            noise_unit = torch.randn_like(faded_mix)
            received = add_awgn(faded_mix, config.snr_db, noise_unit)
            pred_e = sample_ddpm(ddpm, schedule, received, label_e, config.z_dim,
                                 config.seed + i * 10, config.sampling_method,
                                 config.ddim_steps, config.ddim_eta)
            pred_d = sample_ddpm(ddpm, schedule, received, label_d, config.z_dim,
                                 config.seed + i * 10 + 1, config.sampling_method,
                                 config.ddim_steps, config.ddim_eta)
            output_e = generator(pred_e, label_e)
            output_d = generator(pred_d, label_d)
            prefix = f"sample_{i:02d}"
            for name, tensor in (
                ("input_eurosat", image_e),
                ("input_deepglobe", image_d),
                ("output_eurosat", output_e),
                ("output_deepglobe", output_d),
            ):
                save_image((tensor[0].cpu().clamp(-1, 1) + 1) / 2,
                           os.path.join(config.save_root, f"{prefix}_{name}.png"))
            print(f"[{i:02d}/10] Saved four images")
    print(f"Done: {os.path.abspath(config.save_root)}")


if __name__ == "__main__":
    main()
