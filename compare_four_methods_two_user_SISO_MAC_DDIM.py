"""
Compare three latent-separation systems over a two-user SISO multiple-access channel
using the same image pairs, SNR values, channel realizations, noise realizations,
and DDPM sampling seeds.

System model:
    z_rx = h_E * z_E + h_D * z_D + n

Each transmitter has one antenna, and the receiver has one antenna. The two
sources use independent scalar flat-fading coefficients h_E and h_D.

Red curve:
    VAE-DDPM-VAE
    Shared VAE encoder -> VAE latent DDPM -> Conditional VAE decoder

Purple curve:
    VAE-DDPM-GAN
    Scratch VAE-GAN encoder -> Scratch latent DDPM -> Scratch conditional GAN generator

Blue curve:
    VAE-JointMLP-GAN
    Scratch VAE-GAN encoder -> Deterministic joint MLP separator -> Scratch conditional GAN generator

Required checkpoints:
    ./save_model/shared_vae_satellite_latest.pth
    ./save_model/DDPM_vae_latent_separator_latest.pth
    ./save_model/vae_gan_from_scratch_best.pth
    ./save_model/DDPM_vaegan_scratch_latent_separator_best.pth
    ./save_model/MLP_vaegan_scratch_latent_separator_best.pth

Outputs:
    ./save_data/compare_VAE_DDPM_VAE_vs_VAE_DDPM_GAN_vs_VAE_JointMLP_GAN/
        metrics_long.csv
        comparison_summary.csv
        qualitative_SNR*.png
        PSNR_vs_SNR.png
        SSIM_vs_SNR.png
        Latent_Cosine_vs_SNR.png
        Latent_MSE_vs_SNR.png
        Consistency_MSE_vs_SNR.png
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

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    z_dim: int = 256
    label_dim: int = 64
    num_classes: int = 2
    diffusion_steps: int = 400

    # Use the same deterministic sampler as the revised DDPM training/validation.
    sampling_method: str = "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0

    batch_size: int = 16
    num_workers: int = 0
    max_test_pairs: int = 32

    snr_values: Tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30)
    channel_types: Tuple[str, ...] = ("AWGN-SISO-MAC", "Rayleigh-SISO-MAC", "Rician-SISO-MAC")
    rician_k_db: float = 6.0

    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    # Red baseline: VAE encoder + VAE DDPM + VAE decoder
    vae_checkpoint: str = "./save_model/shared_vae_satellite_latest.pth"
    vae_ddpm_checkpoint: str = "./save_model/DDPM_vae_latent_separator_latest.pth"

    # Purple proposed method: scratch VAE-GAN encoder + scratch DDPM + GAN generator
    scratch_vaegan_checkpoint: str = "./save_model/vae_gan_from_scratch_best.pth"
    scratch_vaegan_ddpm_checkpoint: str = (
        "./save_model/DDPM_vaegan_scratch_latent_separator_best.pth"
    )

    # Blue baseline: same scratch VAE-GAN encoder/generator + deterministic MLP separator
    scratch_vaegan_mlp_checkpoint: str = (
        "./save_model/MLP_vaegan_scratch_latent_separator_best.pth"
    )

    joint_vaegan_checkpoint = "./save_model/joint_vae_gan_satellite_latest.pth"
    joint_vaegan_ddpm_checkpoint = "./save_model/DDPM_vae_gan_latent_separator_v2_latest.pth"

    # New folder, so previous six-method results are not overwritten.
    save_root: str = "./save_data/compare_three_methods_two_user_SISO_MAC_joint_explicit"

    qualitative_snrs: Tuple[float, ...] = (0, 10, 20, 30)
    qualitative_count: int = 8

class SatelliteDataset(Dataset):

    def __init__(self, root: str, transform=None):
        self.files = sorted(glob.glob(os.path.join(root, '*_sat_*.jpg')))
        self.transform = transform
        if not self.files:
            raise RuntimeError(f'No matching DeepGlobe images found in:\n{root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        image = Image.open(self.files[index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

def get_image(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch

class SharedVAEEncoder(nn.Module):

    def __init__(self, z_dim: int=256):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        feature_dim = 512 * 4 * 4
        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim)

    def forward(self, x: torch.Tensor):
        h = self.features(x).flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 10.0)
        return (mu, logvar)

class ConditionalVAEDecoder(nn.Module):

    def __init__(self, z_dim: int=256, num_classes: int=2, label_dim: int=64):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.fc = nn.Sequential(nn.Linear(z_dim + label_dim, 512 * 4 * 4), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())

    def forward(self, z: torch.Tensor, labels: torch.Tensor):
        label_feature = self.label_embedding(labels)
        h = self.fc(torch.cat([z, label_feature], dim=1))
        h = h.view(z.size(0), 512, 4, 4)
        return self.decoder(h)

class ScratchConditionalGenerator(nn.Module):

    def __init__(self, z_dim: int=256, num_classes: int=2, label_dim: int=64):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.fc = nn.Sequential(nn.Linear(z_dim + label_dim, 512 * 4 * 4), nn.BatchNorm1d(512 * 4 * 4), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh())

    def forward(self, z: torch.Tensor, labels: torch.Tensor):
        label_feature = self.label_embedding(labels)
        h = self.fc(torch.cat([z, label_feature], dim=1))
        h = h.view(z.size(0), 512, 4, 4)
        return self.decoder(h)

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor):
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * -scale)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding

class ResidualMLPBlock(nn.Module):

    def __init__(self, dim: int, dropout: float=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor):
        return x + self.net(x)

class LatentConditionalDDPM(nn.Module):

    def __init__(self, z_dim: int=256, num_classes: int=2, time_dim: int=128, label_dim: int=64, hidden_dim: int=768, num_blocks: int=4, dropout: float=0.0):
        super().__init__()
        self.time_embedding = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        input_dim = z_dim + z_dim + time_dim + label_dim
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_layer = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, z_dim))

    def forward(self, z_t, timesteps, labels, cond):
        t_emb = self.time_embedding(timesteps)
        y_emb = self.label_embedding(labels)
        h = torch.cat([z_t, cond, t_emb, y_emb], dim=1)
        return self.output_layer(self.blocks(self.input_layer(h)))


class JointLatentMLPSeparator(nn.Module):
    """Predict two source latents from one scalar-channel SISO-MAC mixture."""

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
                ResidualMLPBlock(hidden_dim, dropout=dropout)
                for _ in range(num_blocks)
            ]
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, z_dim * 2),
        )
        self.z_dim = z_dim

    def forward(self, z_rx: torch.Tensor):
        hidden = self.input_layer(z_rx)
        hidden = self.blocks(hidden)
        pair = self.output_layer(hidden)
        return pair[:, : self.z_dim], pair[:, self.z_dim :]


class DiffusionSchedule(nn.Module):

    def __init__(self, timesteps: int=400, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        self.timesteps = timesteps
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alpha_bars_prev', alpha_bars_prev)
        self.register_buffer('sqrt_recip_alpha_bars', torch.sqrt(1.0 / alpha_bars))
        self.register_buffer('sqrt_recipm1_alpha_bars', torch.sqrt(1.0 / alpha_bars - 1.0))
        self.register_buffer('posterior_variance', torch.clamp(posterior_variance, min=1e-20))

    @staticmethod
    def extract(values: torch.Tensor, t: torch.Tensor, x_shape):
        out = values.gather(0, t)
        return out.reshape(t.size(0), *(1,) * (len(x_shape) - 1))

    def predict_x0_from_eps(self, x_t, t, eps):
        recip = self.extract(self.sqrt_recip_alpha_bars, t, x_t.shape)
        recipm1 = self.extract(self.sqrt_recipm1_alpha_bars, t, x_t.shape)
        return recip * x_t - recipm1 * eps

@dataclass
class MethodBundle:
    name: str
    encoder: nn.Module
    decoder: nn.Module
    ddpm: nn.Module | None = None
    schedule: DiffusionSchedule | None = None
    adapter: nn.Module | None = None
    separator: nn.Module | None = None
    mode: str = 'ddpm'
    conditional_encoder: bool = False

def freeze(model: nn.Module):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

def require_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Required checkpoint does not exist:\n{path}')

def load_ddpm(path: str, config: Config, device: str):
    require_file(path)
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get('model', checkpoint)
    model = LatentConditionalDDPM(z_dim=config.z_dim, num_classes=config.num_classes, label_dim=config.label_dim).to(device)
    model.load_state_dict(state, strict=True)
    freeze(model)
    saved_steps = checkpoint.get('config', {}).get('diffusion_steps', config.diffusion_steps)
    if int(saved_steps) != config.diffusion_steps:
        raise ValueError(f'Checkpoint {path} used diffusion_steps={saved_steps}, but Config.diffusion_steps={config.diffusion_steps}.')
    return model

def load_vae_method(config: Config, device: str):
    require_file(config.vae_checkpoint)
    checkpoint = torch.load(config.vae_checkpoint, map_location=device)
    if 'encoder' not in checkpoint or 'decoder' not in checkpoint:
        raise KeyError("VAE checkpoint must contain 'encoder' and 'decoder'.")
    encoder = SharedVAEEncoder(config.z_dim).to(device)
    decoder = ConditionalVAEDecoder(config.z_dim, config.num_classes, config.label_dim).to(device)
    encoder.load_state_dict(checkpoint['encoder'], strict=True)
    decoder.load_state_dict(checkpoint['decoder'], strict=True)
    ddpm = load_ddpm(config.vae_ddpm_checkpoint, config, device)
    schedule = DiffusionSchedule(config.diffusion_steps).to(device)
    freeze(encoder)
    freeze(decoder)
    return MethodBundle('VAE-DDPM-VAE', encoder, decoder, ddpm, schedule)

def load_scratch_vaegan_method(config: Config, device: str):
    require_file(config.scratch_vaegan_checkpoint)
    checkpoint = torch.load(config.scratch_vaegan_checkpoint, map_location=device)
    if 'encoder' not in checkpoint or 'generator' not in checkpoint:
        raise KeyError("Scratch VAE-GAN checkpoint must contain 'encoder' and 'generator'.")
    encoder = SharedVAEEncoder(config.z_dim).to(device)
    decoder = ScratchConditionalGenerator(z_dim=config.z_dim, num_classes=config.num_classes, label_dim=config.label_dim).to(device)
    encoder.load_state_dict(checkpoint['encoder'], strict=True)
    decoder.load_state_dict(checkpoint['generator'], strict=True)
    ddpm = load_ddpm(config.scratch_vaegan_ddpm_checkpoint, config, device)
    schedule = DiffusionSchedule(config.diffusion_steps).to(device)
    freeze(encoder)
    freeze(decoder)
    print('Loaded scratch VAE-GAN epoch =', checkpoint.get('epoch', 'unknown'))
    return MethodBundle(name='VAE-DDPM-GAN', encoder=encoder, decoder=decoder, ddpm=ddpm, schedule=schedule)

def load_joint_vaegan_ddpm_method(config: Config, device: str):
    require_file(config.joint_vaegan_checkpoint)
    checkpoint = torch.load(config.joint_vaegan_checkpoint, map_location=device)
    if 'encoder' not in checkpoint or 'generator' not in checkpoint:
        raise KeyError("Scratch VAE-GAN checkpoint must contain 'encoder' and 'generator'.")
    encoder, generator = load_vae_gan_models(
        config.joint_vaegan_checkpoint,
        device,
        config.z_dim,
        config.label_dim,
        config.num_classes,)
    ddpm = load_ddpm(config.joint_vaegan_ddpm_checkpoint, config, device)
    schedule = DiffusionSchedule(config.diffusion_steps).to(device)
    freeze(encoder)
    freeze(generator)
    print('Loaded scratch VAE-GAN epoch =', checkpoint.get('epoch', 'unknown'))
    return MethodBundle(name="JointVAEGAN-DDPM",encoder=encoder,decoder=generator,ddpm=ddpm,schedule=schedule,)


def load_scratch_vaegan_mlp_method(config: Config, device: str):
    """
    Load the same scratch VAE-GAN encoder/generator as VAE-DDPM-GAN,
    but replace the diffusion separator with the trained deterministic MLP.
    """
    require_file(config.scratch_vaegan_checkpoint)
    require_file(config.scratch_vaegan_mlp_checkpoint)

    vaegan_checkpoint = torch.load(
        config.scratch_vaegan_checkpoint,
        map_location=device,
    )
    if "encoder" not in vaegan_checkpoint or "generator" not in vaegan_checkpoint:
        raise KeyError(
            "Scratch VAE-GAN checkpoint must contain 'encoder' and 'generator'."
        )

    encoder = SharedVAEEncoder(config.z_dim).to(device)
    decoder = ScratchConditionalGenerator(
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
    ).to(device)
    encoder.load_state_dict(vaegan_checkpoint["encoder"], strict=True)
    decoder.load_state_dict(vaegan_checkpoint["generator"], strict=True)

    mlp_checkpoint = torch.load(
        config.scratch_vaegan_mlp_checkpoint,
        map_location=device,
    )
    mlp_state = mlp_checkpoint.get("model", mlp_checkpoint)
    saved_config = mlp_checkpoint.get("config", {})

    separator = JointLatentMLPSeparator(
        z_dim=int(saved_config.get("z_dim", config.z_dim)),
        hidden_dim=int(saved_config.get("hidden_dim", 768)),
        num_blocks=int(saved_config.get("num_blocks", 6)),
        dropout=float(saved_config.get("dropout", 0.0)),
    ).to(device)
    separator.load_state_dict(mlp_state, strict=True)

    freeze(encoder)
    freeze(decoder)
    freeze(separator)

    print(
        "Loaded scratch VAE-GAN epoch =",
        vaegan_checkpoint.get("epoch", "unknown"),
    )
    print(
        "Loaded MLP separator epoch =",
        mlp_checkpoint.get("epoch", "unknown"),
    )

    return MethodBundle(
        name="VAE-JointMLP-GAN",
        encoder=encoder,
        decoder=decoder,
        separator=separator,
        mode="direct_separator",
    )


def add_awgn(signal: torch.Tensor, snr_db: float, noise_unit: torch.Tensor):
    if math.isinf(float(snr_db)):
        return signal
    power = signal.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_std = torch.sqrt(power / snr_linear)
    return signal + noise_unit * noise_std


def make_fading_coefficients(
    channel_type: str,
    batch_size: int,
    device: str,
    rician_k_db: float,
):
    """
    Create two independent scalar flat-fading coefficients for a two-user SISO-MAC.

    Each transmitter and the receiver have one antenna. Because the latent vectors
    are real-valued, this simulation uses the fading-envelope magnitude as a
    real scalar coefficient:
        z_rx = h_E * z_E + h_D * z_D + n

    Rayleigh:
        h_i = sqrt((g_I^2 + g_Q^2) / 2), E[h_i^2] = 1

    Rician:
        h_i = |sqrt(K/(K+1)) + sqrt(1/(K+1)) * g|,
        where g is circular Gaussian and K is configured in dB.
    """
    if channel_type == "AWGN-SISO-MAC":
        ones = torch.ones(batch_size, 1, device=device)
        return ones, ones.clone()

    def rayleigh_gain():
        real = torch.randn(batch_size, 1, device=device)
        imag = torch.randn(batch_size, 1, device=device)
        return torch.sqrt((real.pow(2) + imag.pow(2)) / 2.0).clamp_min(1e-6)

    if channel_type == "Rayleigh-SISO-MAC":
        return rayleigh_gain(), rayleigh_gain()

    if channel_type == "Rician-SISO-MAC":
        k_linear = 10.0 ** (float(rician_k_db) / 10.0)
        los = math.sqrt(k_linear / (k_linear + 1.0))
        scatter_scale = math.sqrt(1.0 / (2.0 * (k_linear + 1.0)))

        def rician_gain():
            real = los + scatter_scale * torch.randn(batch_size, 1, device=device)
            imag = scatter_scale * torch.randn(batch_size, 1, device=device)
            return torch.sqrt(real.pow(2) + imag.pow(2)).clamp_min(1e-6)

        return rician_gain(), rician_gain()

    raise ValueError(f"Unknown channel type: {channel_type}")


def transmit_latent_pair(
    z_e: torch.Tensor,
    z_d: torch.Tensor,
    snr_db: float,
    channel_type: str,
    noise_unit: torch.Tensor,
    h_e: torch.Tensor,
    h_d: torch.Tensor,
):
    """Transmit two single-antenna users to one single-antenna receiver."""
    faded_mix = h_e * z_e + h_d * z_d
    received = add_awgn(faded_mix, snr_db, noise_unit)
    return received, faded_mix

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
    """
    Conditional latent sampler.

    DDIM with eta=0 matches the revised training script's validation setup and
    is deterministic for a fixed seed. The original 400-step ancestral DDPM
    sampler remains available by setting sampling_method="ddpm".
    """
    model.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x = torch.randn(cond.size(0), z_dim, device=cond.device)
    method = sampling_method.lower()

    if method == "ddpm":
        for step in reversed(range(schedule.timesteps)):
            t = torch.full(
                (cond.size(0),), step, dtype=torch.long, device=cond.device
            )
            pred_eps = model(x, t, labels, cond)
            pred_x0 = schedule.predict_x0_from_eps(x, t, pred_eps)
            if clip_x0 is not None:
                pred_x0 = pred_x0.clamp(-clip_x0, clip_x0)

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

    if method != "ddim":
        raise ValueError(f"Unknown sampling_method: {sampling_method}")

    steps = max(1, min(int(ddim_steps), schedule.timesteps))
    time_sequence = torch.linspace(
        schedule.timesteps - 1,
        0,
        steps,
        device=cond.device,
    ).round().long()
    time_sequence = torch.unique_consecutive(time_sequence)

    for index, step_tensor in enumerate(time_sequence):
        step = int(step_tensor.item())
        t = torch.full(
            (cond.size(0),), step, dtype=torch.long, device=cond.device
        )
        pred_eps = model(x, t, labels, cond)
        pred_x0 = schedule.predict_x0_from_eps(x, t, pred_eps)
        if clip_x0 is not None:
            pred_x0 = pred_x0.clamp(-clip_x0, clip_x0)

        alpha_bar_t = schedule.extract(schedule.alpha_bars, t, x.shape)

        if index == len(time_sequence) - 1:
            x = pred_x0
            continue

        prev_step = int(time_sequence[index + 1].item())
        t_prev = torch.full(
            (cond.size(0),), prev_step, dtype=torch.long, device=cond.device
        )
        alpha_bar_prev = schedule.extract(schedule.alpha_bars, t_prev, x.shape)

        sigma = float(ddim_eta) * torch.sqrt(
            torch.clamp(
                (1.0 - alpha_bar_prev)
                / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_prev),
                min=0.0,
            )
        )
        direction = torch.sqrt(
            torch.clamp(1.0 - alpha_bar_prev - sigma.pow(2), min=0.0)
        ) * pred_eps

        if float(ddim_eta) > 0.0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction + sigma * noise

    return x

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
        values.append(structural_similarity(reference, prediction, data_range=1.0, channel_axis=-1))
    return np.asarray(values, dtype=np.float64)

def latent_mse_per_sample(pred: torch.Tensor, target: torch.Tensor):
    return (pred - target).pow(2).mean(dim=1).detach().cpu().numpy()

def latent_cosine_per_sample(pred: torch.Tensor, target: torch.Tensor):
    return F.cosine_similarity(pred, target, dim=1).detach().cpu().numpy()

def consistency_per_sample(pred_e, pred_d, clean_mix):
    return (pred_e + pred_d - clean_mix).pow(2).mean(dim=1).detach().cpu().numpy()

class MetricAccumulator:

    def __init__(self):
        self.values: Dict[str, List[float]] = {}

    def add(self, name: str, values):
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        self.values.setdefault(name, []).extend(array.tolist())

    def mean(self, name: str):
        values = self.values.get(name, [])
        return float(np.mean(values)) if values else float('nan')

    def count(self):
        if not self.values:
            return 0
        return len(next(iter(self.values.values())))

@torch.no_grad()
def evaluate_batch(
    bundle: MethodBundle,
    image_e: torch.Tensor,
    image_d: torch.Tensor,
    snr_db: float,
    channel_type: str,
    noise_unit: torch.Tensor,
    h_e: torch.Tensor,
    h_d: torch.Tensor,
    sample_seed: int,
    sampling_method: str,
    ddim_steps: int,
    ddim_eta: float,
):
    batch_size = image_e.size(0)
    label_e = torch.zeros(batch_size, dtype=torch.long, device=image_e.device)
    label_d = torch.ones(batch_size, dtype=torch.long, device=image_e.device)
    if bundle.conditional_encoder:
        z_e, _ = bundle.encoder(image_e, label_e)
        z_d, _ = bundle.encoder(image_d, label_d)
    else:
        z_e, _ = bundle.encoder(image_e)
        z_d, _ = bundle.encoder(image_d)
    # AWGN reference mixture before fading. The actual channel mixture below is
    # h_E*z_E + h_D*z_D, corresponding to two SISO users sharing one receiver.
    clean_mix = z_e + z_d
    received, channel_mix = transmit_latent_pair(
        z_e=z_e,
        z_d=z_d,
        snr_db=snr_db,
        channel_type=channel_type,
        noise_unit=noise_unit,
        h_e=h_e,
        h_d=h_d,
    )
    if bundle.mode == 'direct_separator':
        if bundle.separator is None:
            raise RuntimeError('Direct-separator method has no separator model.')
        pred_e, pred_d = bundle.separator(received)
    elif bundle.mode == 'ddpm':
        if bundle.ddpm is None or bundle.schedule is None:
            raise RuntimeError('DDPM method is missing DDPM or schedule.')
        pred_e = sample_ddpm(
            bundle.ddpm,
            bundle.schedule,
            received,
            label_e,
            z_e.size(1),
            seed=sample_seed,
            sampling_method=sampling_method,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )
        pred_d = sample_ddpm(
            bundle.ddpm,
            bundle.schedule,
            received,
            label_d,
            z_d.size(1),
            seed=sample_seed + 1,
            sampling_method=sampling_method,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )
    else:
        raise ValueError(f'Unknown method mode: {bundle.mode}')
    oracle_e = bundle.decoder(z_e, label_e)
    oracle_d = bundle.decoder(z_d, label_d)
    corrected_e = pred_e
    corrected_d = pred_d
    delta_e = torch.zeros_like(pred_e)
    delta_d = torch.zeros_like(pred_d)
    if bundle.adapter is not None:
        corrected_e, delta_e = bundle.adapter(pred_e, label_e)
        corrected_d, delta_d = bundle.adapter(pred_d, label_d)
    separated_e = bundle.decoder(corrected_e, label_e)
    separated_d = bundle.decoder(corrected_d, label_d)
    return {
        'z_e': z_e,
        'z_d': z_d,
        'clean_mix': clean_mix,
        'channel_mix': channel_mix,
        'received': received,
        'h_e': h_e,
        'h_d': h_d,
        'pred_e': corrected_e,
        'pred_d': corrected_d,
        'raw_pred_e': pred_e,
        'raw_pred_d': pred_d,
        'delta_e': delta_e,
        'delta_d': delta_d,
        'oracle_e': oracle_e,
        'oracle_d': oracle_d,
        'separated_e': separated_e,
        'separated_d': separated_d,
    }

def add_domain_metrics(accumulator: MetricAccumulator, prefix: str, separated: torch.Tensor, oracle: torch.Tensor, real: torch.Tensor, pred_z: torch.Tensor, target_z: torch.Tensor):
    accumulator.add(f'{prefix}_psnr_real', batch_psnr(separated, real))
    accumulator.add(f'{prefix}_ssim_real', batch_ssim(separated, real))
    accumulator.add(f'{prefix}_psnr_oracle', batch_psnr(separated, oracle))
    accumulator.add(f'{prefix}_ssim_oracle', batch_ssim(separated, oracle))
    accumulator.add(f'{prefix}_oracle_psnr_real', batch_psnr(oracle, real))
    accumulator.add(f'{prefix}_oracle_ssim_real', batch_ssim(oracle, real))
    accumulator.add(f'{prefix}_latent_mse', latent_mse_per_sample(pred_z, target_z))
    accumulator.add(f'{prefix}_latent_cosine', latent_cosine_per_sample(pred_z, target_z))

def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def build_summary_row(method: str, channel: str, snr_db: float, acc: MetricAccumulator):
    e_psnr = acc.mean('E_psnr_real')
    d_psnr = acc.mean('D_psnr_real')
    e_ssim = acc.mean('E_ssim_real')
    d_ssim = acc.mean('D_ssim_real')
    e_mse = acc.mean('E_latent_mse')
    d_mse = acc.mean('D_latent_mse')
    e_cos = acc.mean('E_latent_cosine')
    d_cos = acc.mean('D_latent_cosine')
    return {'method': method, 'channel': channel, 'snr_db': snr_db, 'num_pairs': acc.count(), 'E_psnr_real': e_psnr, 'D_psnr_real': d_psnr, 'avg_psnr_real': np.mean([e_psnr, d_psnr]), 'E_ssim_real': e_ssim, 'D_ssim_real': d_ssim, 'avg_ssim_real': np.mean([e_ssim, d_ssim]), 'E_psnr_oracle': acc.mean('E_psnr_oracle'), 'D_psnr_oracle': acc.mean('D_psnr_oracle'), 'E_ssim_oracle': acc.mean('E_ssim_oracle'), 'D_ssim_oracle': acc.mean('D_ssim_oracle'), 'E_oracle_psnr_real': acc.mean('E_oracle_psnr_real'), 'D_oracle_psnr_real': acc.mean('D_oracle_psnr_real'), 'E_oracle_ssim_real': acc.mean('E_oracle_ssim_real'), 'D_oracle_ssim_real': acc.mean('D_oracle_ssim_real'), 'E_latent_mse': e_mse, 'D_latent_mse': d_mse, 'avg_latent_mse': np.mean([e_mse, d_mse]), 'E_latent_cosine': e_cos, 'D_latent_cosine': d_cos, 'avg_latent_cosine': np.mean([e_cos, d_cos]), 'consistency_mse': acc.mean('consistency_mse')}

METHOD_COLORS = {"VAE-DDPM-VAE": "red", "VAE-DDPM-GAN": "purple", "VAE-JointMLP-GAN": "tab:blue", "JointVAEGAN-DDPM":"green",}

METHOD_ORDER = ("VAE-DDPM-VAE", "VAE-DDPM-GAN", "VAE-JointMLP-GAN", "JointVAEGAN-DDPM",)

def set_requires_grad(model: nn.Module, value: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = value

def load_vae_gan_models(joint_checkpoint_path: str, device: str, z_dim: int, label_dim: int, num_classes: int,):
    if not os.path.exists(joint_checkpoint_path):
        raise FileNotFoundError(f"Cannot find joint VAE-GAN checkpoint:\n{joint_checkpoint_path}")

    checkpoint = torch.load(joint_checkpoint_path, map_location=device)
    if "encoder" not in checkpoint or "generator" not in checkpoint:
        raise KeyError("The checkpoint must contain 'encoder' and 'generator' state dictionaries.")

    encoder = SharedVAEEncoder(z_dim=z_dim).to(device)
    generator = Generator_Satellite_Conditional(z_dim=z_dim, num_classes=num_classes, label_dim=label_dim, out_ch=3,).to(device)

    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    generator.load_state_dict(checkpoint["generator"], strict=True)

    encoder.eval()
    generator.eval()
    set_requires_grad(encoder, False)
    set_requires_grad(generator, False)

    print(f"Loaded VAE-GAN checkpoint epoch = {checkpoint.get('epoch', 'unknown')}")
    print(f"Loaded VAE-GAN checkpoint stage = {checkpoint.get('stage', 'unknown')}")
    return encoder, generator


def save_qualitative_grid(save_path: str, real_e: torch.Tensor, real_d: torch.Tensor, result_vae: dict, result_gan: dict, result_mlp: dict, result_joint: dict, count: int):
    """
    Row order for each domain:
    real, oracle, VAE-DDPM-VAE, VAE-DDPM-GAN, VAE-JointMLP-GAN.
    """
    count = min(count, real_e.size(0), real_d.size(0))
    rows = [
        real_e[:count],
        result_vae["oracle_e"][:count],
        result_vae["separated_e"][:count],
        result_gan["separated_e"][:count],
        result_mlp["separated_e"][:count],
        result_joint["separated_e"][:count],

        real_d[:count],
        result_vae["oracle_d"][:count],
        result_vae["separated_d"][:count],
        result_gan["separated_d"][:count],
        result_mlp["separated_d"][:count],
        result_joint["separated_d"][:count],]
    grid = make_grid(torch.cat(rows, dim=0), nrow=count, padding=2, normalize=True, value_range=(-1, 1),)
    save_image(grid, save_path)

def plot_metric(rows: List[dict], metric: str, ylabel: str, save_path: str):
    available_methods = {row["method"] for row in rows}
    missing_methods = [method for method in METHOD_ORDER if method not in available_methods]
    if missing_methods:
        raise RuntimeError("The following methods have no evaluation rows and cannot be plotted: " + ", ".join(missing_methods))

    channel_styles = {"AWGN-SISO-MAC": "-", "Rayleigh-SISO-MAC": "--", "Rician-SISO-MAC": ":",}

    plt.figure(figsize=(10, 6))
    for method in METHOD_ORDER:
        for channel in sorted({row["channel"] for row in rows}):
            selected = sorted([row for row in rows if row["method"] == method and row["channel"] == channel],
                key=lambda row: float(row["snr_db"]),)
            if not selected:
                continue
            x = [float(row["snr_db"]) for row in selected]
            y = [float(row[metric]) for row in selected]
            plt.plot(x, y, marker="o", linewidth=2, linestyle=channel_styles.get(channel, "-"), color=METHOD_COLORS[method], label=f"{method} | {channel}",)
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
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

    print(
        f"DDPM sampler: {config.sampling_method}, "
        f"steps={config.ddim_steps}, eta={config.ddim_eta}"
    )

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    eurosat = datasets.ImageFolder(config.eurosat_root, transform=transform)
    deepglobe = SatelliteDataset(config.deepglobe_root, transform=transform)

    pair_count = min(config.max_test_pairs, len(eurosat), len(deepglobe))
    if pair_count <= 0:
        raise RuntimeError("No test pairs are available.")

    subset_generator = torch.Generator().manual_seed(config.seed)
    e_indices = torch.randperm(len(eurosat), generator=subset_generator)[:pair_count].tolist()
    d_indices = torch.randperm(len(deepglobe), generator=subset_generator)[:pair_count].tolist()
    loader_e = DataLoader(
        Subset(eurosat, e_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),)
    loader_d = DataLoader(
        Subset(deepglobe, d_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),)

    print("Loading red method: VAE-DDPM-VAE...")
    vae_method = load_vae_method(config, device)

    print("Loading purple method: VAE-DDPM-GAN...")
    gan_method = load_scratch_vaegan_method(config, device)

    print("Loading blue method: VAE-JointMLP-GAN (one forward pass outputs both source latents)...")
    mlp_method = load_scratch_vaegan_mlp_method(config, device)

    print("Loading green...")
    joint_method = load_joint_vaegan_ddpm_method(config,device)

    methods = [vae_method, gan_method, mlp_method, joint_method,]
    rows: List[dict] = []

    for channel_index, channel_type in enumerate(config.channel_types):
        for snr_index, snr_db in enumerate(config.snr_values):
            print("=" * 90)
            print(f"Evaluating channel={channel_type}, SNR={snr_db} dB")

            accumulators = {method.name: MetricAccumulator() for method in methods}
            qualitative_saved = False

            for batch_index, (batch_e, batch_d) in enumerate(zip(loader_e, loader_d)):
                image_e = get_image(batch_e).to(device, non_blocking=True)
                image_d = get_image(batch_d).to(device, non_blocking=True)
                current = min(image_e.size(0), image_d.size(0))
                image_e = image_e[:current]
                image_d = image_d[:current]

                # All methods receive exactly the same fading and noise realization.
                channel_seed = (config.seed + channel_index * 1_000_000 + snr_index * 10_000 + batch_index)
                torch.manual_seed(channel_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(channel_seed)

                h_e, h_d = make_fading_coefficients(channel_type=channel_type, batch_size=current, device=device, rician_k_db=config.rician_k_db,)
                noise_unit = torch.randn(current, config.z_dim, device=device)

                method_results = {}
                for method in methods:
                    sample_seed = (config.seed + channel_index * 10_000_000 + snr_index * 100_000 + batch_index * 10)
                    result = evaluate_batch(
                        method,
                        image_e,
                        image_d,
                        snr_db,
                        channel_type,
                        noise_unit,
                        h_e,
                        h_d,
                        sample_seed,
                        config.sampling_method,
                        config.ddim_steps,
                        config.ddim_eta,
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
                    # Compare the re-mixed predictions with the actual channel mixture.
                    predicted_channel_mix = (result["h_e"] * result["pred_e"] + result["h_d"] * result["pred_d"])
                    acc.add("consistency_mse",latent_mse_per_sample(predicted_channel_mix, result["channel_mix"],),)

                if (
                    not qualitative_saved
                    and float(snr_db)
                    in [float(value) for value in config.qualitative_snrs]
                ):
                    safe_channel = channel_type.replace("-", "_")
                    filename = f"qualitative_{safe_channel}_SNR{snr_db:g}.png"
                    save_qualitative_grid(
                        os.path.join(config.save_root, filename),
                        image_e,
                        image_d,
                        method_results["VAE-DDPM-VAE"],
                        method_results["VAE-DDPM-GAN"],
                        method_results["VAE-JointMLP-GAN"],
                        method_results["JointVAEGAN-DDPM"],
                        config.qualitative_count,
                    )
                    qualitative_saved = True

                print(
                    f"{channel_type} | SNR={snr_db:>4} | "
                    f"batch {batch_index + 1}/{len(loader_e)} complete"
                )

            for method in methods:
                row = build_summary_row(method.name, channel_type, snr_db, accumulators[method.name],)
                rows.append(row)
                print(
                    f"{method.name} | {channel_type}: "
                    f"PSNR={row['avg_psnr_real']:.3f}, "
                    f"SSIM={row['avg_ssim_real']:.4f}, "
                    f"LatCos={row['avg_latent_cosine']:.4f}, "
                    f"LatMSE={row['avg_latent_mse']:.4f}"
                )

            write_csv(os.path.join(config.save_root, "metrics_long.csv"), rows,)

    wide_rows = []
    for channel_type in config.channel_types:
        for snr_db in config.snr_values:
            row_by_method = {
                row["method"]: row
                for row in rows
                if (
                    row["channel"] == channel_type
                    and float(row["snr_db"]) == float(snr_db)
                )
            }
            if any(method not in row_by_method for method in METHOD_ORDER):
                continue

            vae_row = row_by_method["VAE-DDPM-VAE"]
            gan_row = row_by_method["VAE-DDPM-GAN"]
            mlp_row = row_by_method["VAE-JointMLP-GAN"]
            joint_row = row_by_method["JointVAEGAN-DDPM"]

            wide_rows.append({
                "channel": channel_type,
                "snr_db": snr_db,
                "VAE_DDPM_VAE_avg_PSNR": vae_row["avg_psnr_real"],
                "VAE_DDPM_GAN_avg_PSNR": gan_row["avg_psnr_real"],
                "VAE_JointMLP_GAN_avg_PSNR": mlp_row["avg_psnr_real"],
                "JointVAEGANDDPM_avg_PSNR": joint_row["avg_psnr_real"],
                "VAE_DDPM_VAE_avg_SSIM": vae_row["avg_ssim_real"],
                "VAE_DDPM_GAN_avg_SSIM": gan_row["avg_ssim_real"],
                "VAE_JointMLP_GAN_avg_SSIM": mlp_row["avg_ssim_real"],
                "JointVAEGANDDPM_avg_SSIM": joint_row["avg_ssim_real"],
                "VAE_DDPM_VAE_avg_latent_MSE": vae_row["avg_latent_mse"],
                "VAE_DDPM_GAN_avg_latent_MSE": gan_row["avg_latent_mse"],
                "VAE_JointMLP_GAN_avg_latent_MSE": mlp_row["avg_latent_mse"],
                "JointVAEGANDDPM_avg_latent_MSE": joint_row["avg_latent_mse"],
                "VAE_DDPM_VAE_avg_latent_cosine": vae_row["avg_latent_cosine"],
                "VAE_DDPM_GAN_avg_latent_cosine": gan_row["avg_latent_cosine"],
                "VAE_JointMLP_GAN_avg_latent_cosine": mlp_row["avg_latent_cosine"],
                "JointVAEGANDDPM_avg_latent_cosine": joint_row["avg_latent_cosine"],
                "VAE_DDPM_VAE_consistency_MSE": vae_row["consistency_mse"],
                "VAE_DDPM_GAN_consistency_MSE": gan_row["consistency_mse"],
                "VAE_JointMLP_GAN_consistency_MSE": mlp_row["consistency_mse"],
                "JointVAEGANDDPM_consistency_MSE": joint_row["consistency_mse"],
            })
    write_csv(os.path.join(config.save_root, "comparison_summary.csv"),wide_rows,)
    plot_metric(rows,"avg_psnr_real","Average PSNR vs. real image (dB)",os.path.join(config.save_root, "PSNR_vs_SNR.png"))
    plot_metric(rows, "avg_ssim_real", "Average SSIM vs. real image", os.path.join(config.save_root, "SSIM_vs_SNR.png"))
    plot_metric(rows, "avg_latent_cosine", "Average latent cosine similarity", os.path.join(config.save_root, "Latent_Cosine_vs_SNR.png"))
    plot_metric(rows, "avg_latent_mse", "Average latent MSE", os.path.join(config.save_root, "Latent_MSE_vs_SNR.png"),)
    plot_metric(rows, "consistency_mse", "Mixture consistency MSE", os.path.join(config.save_root, "Consistency_MSE_vs_SNR.png"),)
    print("=" * 90)
    print("Comparison completed.")
    print(f"Results: {config.save_root}")
    print("Red:    VAE-DDPM-VAE")
    print("Purple: VAE-DDPM-GAN")
    print("Blue:   VAE-JointMLP-GAN (jointly predicts EuroSAT and DeepGlobe latents)")
    print("Line color = method; line style = channel.")
    print("AWGN-SISO-MAC: solid, Rayleigh-SISO-MAC: dashed, Rician-SISO-MAC: dotted.")
    print(
        f"DDPM sampler: {config.sampling_method}, "
        f"steps={config.ddim_steps}, eta={config.ddim_eta}"
    )
    print("For final paper results, increase Config.max_test_pairs.")


if __name__ == "__main__":
    main()
