"""
Fine-tune the conditional GAN generator on latents separated by a frozen DDPM.

Source checkpoints expected from the three training programs:
1) Conditional GAN pretraining:
   ./save_model/G_wgangp_satellite_conditional_C.pth
   ./save_model/D_wgangp_satellite_conditional_C.pth
2) Joint VAE-encoder + GAN training:
   ./save_model/joint_vae_gan_satellite_latest.pth
3) Latent DDPM separator training:
   ./save_model/DDPM_vae_gan_latent_separator_latest.pth

Recommended workflow:
- Phase A builds a reusable cache of DDPM-separated latents using frozen Encoder/DDPM.
- Phase B fine-tunes only Generator + Critic from that cache.
- Encoder and DDPM are never updated.

The cache uses uint8 images and float16 latents to reduce disk usage.
"""

import copy
import csv
import glob
import math
import os
import random
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from model import Generator_Satellite_Conditional, Critic_Satellite_Conditional


# ============================================================
# Dataset wrappers
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


class CachedSeparatedLatentDataset(Dataset):
    """Reads one cache file created by build_latent_cache()."""

    def __init__(self, cache_path: str):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cannot find latent cache:\n{cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        required = {"z_hat", "z_true", "images_uint8", "labels", "snr_db"}
        missing = required.difference(payload)
        if missing:
            raise KeyError(f"Cache is missing fields: {sorted(missing)}")

        self.z_hat = payload["z_hat"]
        self.z_true = payload["z_true"]
        self.images_uint8 = payload["images_uint8"]
        self.labels = payload["labels"]
        self.snr_db = payload["snr_db"]

        n = self.z_hat.size(0)
        if not all(x.size(0) == n for x in [self.z_true, self.images_uint8, self.labels, self.snr_db]):
            raise ValueError("Cached tensors have inconsistent first dimensions.")

    def __len__(self) -> int:
        return self.z_hat.size(0)

    def __getitem__(self, idx: int):
        # Restore image from uint8 [0,255] to normalized [-1,1].
        image = self.images_uint8[idx].float().div(127.5).sub(1.0)
        return (
            self.z_hat[idx].float(),
            self.z_true[idx].float(),
            image,
            self.labels[idx].long(),
            self.snr_db[idx].float(),
        )


# ============================================================
# Shared VAE Encoder -- must match the joint training program
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


# ============================================================
# DDPM model -- must match the separator training program
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.output_layer = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, z_dim))

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        y_emb = self.label_embedding(labels)
        h = torch.cat([z_t, cond, t_emb, y_emb], dim=1)
        return self.output_layer(self.blocks(self.input_layer(h)))


class DiffusionSchedule(nn.Module):
    def __init__(self, timesteps: int = 400, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars))
        self.register_buffer("sqrt_recipm1_alpha_bars", torch.sqrt(1.0 / alpha_bars - 1.0))

    @staticmethod
    def extract(values: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
        out = values.gather(0, t)
        return out.reshape(t.size(0), *((1,) * (len(x_shape) - 1)))

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        recip = self.extract(self.sqrt_recip_alpha_bars, t, x_t.shape)
        recipm1 = self.extract(self.sqrt_recipm1_alpha_bars, t, x_t.shape)
        return recip * x_t - recipm1 * eps


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(model: nn.Module, value: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = value


def append_csv(path: str, row: Dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def add_awgn_by_snr(signal: torch.Tensor, snr_db: torch.Tensor | float) -> torch.Tensor:
    """Supports one scalar SNR for the whole batch."""
    snr_value = float(snr_db)
    if math.isinf(snr_value):
        return signal
    signal_power = signal.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    noise_power = signal_power / (10.0 ** (snr_value / 10.0))
    return signal + torch.randn_like(signal) * torch.sqrt(noise_power)


def gradient_penalty_conditional(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> torch.Tensor:
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real + (1.0 - epsilon) * fake
    interpolated.requires_grad_(True)
    mixed_scores = critic(interpolated, labels)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def image_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred01 = ((pred + 1.0) / 2.0).clamp(0.0, 1.0)
    target01 = ((target + 1.0) / 2.0).clamp(0.0, 1.0)
    mse = F.mse_loss(pred01, target01, reduction="none").flatten(1).mean(1)
    return (-10.0 * torch.log10(mse.clamp_min(1e-12))).mean().item()


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    schedule: DiffusionSchedule,
    cond: torch.Tensor,
    labels: torch.Tensor,
    z_dim: int,
    sampling_steps: int = 50,
    eta: float = 0.0,
) -> torch.Tensor:
    """Fast DDIM sampling using the DDPM model trained with 400 steps."""
    model.eval()
    total_steps = schedule.timesteps
    sampling_steps = max(2, min(sampling_steps, total_steps))
    sequence = torch.linspace(total_steps - 1, 0, sampling_steps, device=cond.device).long()
    # Remove duplicate indices caused by integer rounding while preserving descending order.
    sequence = torch.unique_consecutive(sequence)

    x = torch.randn(cond.size(0), z_dim, device=cond.device)
    for index, t_scalar in enumerate(sequence):
        t = torch.full((cond.size(0),), int(t_scalar.item()), device=cond.device, dtype=torch.long)
        eps = model(x, t, labels, cond)
        x0 = schedule.predict_x0_from_eps(x, t, eps)
        alpha_bar_t = schedule.alpha_bars[t_scalar]

        if index == len(sequence) - 1:
            x = x0
            break

        next_t_scalar = sequence[index + 1]
        alpha_bar_next = schedule.alpha_bars[next_t_scalar]
        sigma = eta * torch.sqrt(
            ((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t))
            * (1.0 - alpha_bar_t / alpha_bar_next)
        ).clamp_min(0.0)
        direction = torch.sqrt((1.0 - alpha_bar_next - sigma.pow(2)).clamp_min(0.0)) * eps
        noise = sigma * torch.randn_like(x) if eta > 0.0 else 0.0
        x = torch.sqrt(alpha_bar_next) * x0 + direction + noise

    return x


def load_models(
    joint_checkpoint_path: str,
    ddpm_checkpoint_path: str,
    device: str,
    z_dim: int,
    label_dim: int,
    num_classes: int,
):
    joint = torch.load(joint_checkpoint_path, map_location=device)
    if not all(k in joint for k in ["encoder", "generator", "critic"]):
        raise KeyError("Joint checkpoint must contain encoder, generator, and critic.")

    encoder = SharedVAEEncoder(z_dim).to(device)
    generator = Generator_Satellite_Conditional(
        z_dim=z_dim, num_classes=num_classes, label_dim=label_dim, out_ch=3
    ).to(device)
    critic = Critic_Satellite_Conditional(
        in_ch=3, num_classes=num_classes, feature_dim=512
    ).to(device)
    encoder.load_state_dict(joint["encoder"], strict=True)
    generator.load_state_dict(joint["generator"], strict=True)
    critic.load_state_dict(joint["critic"], strict=True)

    ddpm_payload = torch.load(ddpm_checkpoint_path, map_location=device)
    ddpm_state = ddpm_payload.get("model", ddpm_payload)
    ddpm = LatentConditionalDDPM(
        z_dim=z_dim,
        num_classes=num_classes,
        time_dim=128,
        label_dim=64,
        hidden_dim=768,
        num_blocks=4,
        dropout=0.0,
    ).to(device)
    ddpm.load_state_dict(ddpm_state, strict=True)

    encoder.eval()
    ddpm.eval()
    set_requires_grad(encoder, False)
    set_requires_grad(ddpm, False)
    return encoder, generator, critic, ddpm


# ============================================================
# Phase A: create separated-latent cache
# ============================================================
@torch.no_grad()
def build_latent_cache(
    encoder: nn.Module,
    ddpm: nn.Module,
    schedule: DiffusionSchedule,
    loader_e: DataLoader,
    loader_d: DataLoader,
    cache_path: str,
    device: str,
    z_dim: int,
    max_pairs: int,
    sampling_steps: int,
    min_snr_db: float,
    max_snr_db: float,
) -> None:
    print("\n[Phase A] Building DDPM-separated latent cache...")
    encoder.eval()
    ddpm.eval()

    z_hat_all, z_true_all = [], []
    image_all, label_all, snr_all = [], [], []
    pair_count = 0

    iter_e = iter(loader_e)
    iter_d = iter(loader_d)
    while pair_count < max_pairs:
        try:
            image_e, _ = next(iter_e)
        except StopIteration:
            iter_e = iter(loader_e)
            image_e, _ = next(iter_e)
        try:
            image_d, _ = next(iter_d)
        except StopIteration:
            iter_d = iter(loader_d)
            image_d, _ = next(iter_d)

        remaining = max_pairs - pair_count
        current = min(image_e.size(0), image_d.size(0), remaining)
        image_e = image_e[:current].to(device, non_blocking=True)
        image_d = image_d[:current].to(device, non_blocking=True)
        label_e = torch.zeros(current, dtype=torch.long, device=device)
        label_d = torch.ones(current, dtype=torch.long, device=device)

        z_e, _ = encoder(image_e)
        z_d, _ = encoder(image_d)
        z_mix = z_e + z_d
        snr = random.uniform(min_snr_db, max_snr_db)
        z_rx = add_awgn_by_snr(z_mix, snr)

        z_hat_e = sample_ddim(ddpm, schedule, z_rx, label_e, z_dim, sampling_steps)
        z_hat_d = sample_ddim(ddpm, schedule, z_rx, label_d, z_dim, sampling_steps)

        z_hat_all.append(torch.cat([z_hat_e, z_hat_d], dim=0).half().cpu())
        z_true_all.append(torch.cat([z_e, z_d], dim=0).half().cpu())
        images = torch.cat([image_e, image_d], dim=0)
        images_uint8 = (((images + 1.0) / 2.0).clamp(0, 1) * 255.0).round().byte().cpu()
        image_all.append(images_uint8)
        label_all.append(torch.cat([label_e, label_d], dim=0).byte().cpu())
        snr_all.append(torch.full((2 * current,), snr, dtype=torch.float16))

        pair_count += current
        print(f"  cached pairs: {pair_count}/{max_pairs} | SNR={snr:.1f} dB", end="\r")

    payload = {
        "z_hat": torch.cat(z_hat_all, dim=0),
        "z_true": torch.cat(z_true_all, dim=0),
        "images_uint8": torch.cat(image_all, dim=0),
        "labels": torch.cat(label_all, dim=0),
        "snr_db": torch.cat(snr_all, dim=0),
        "config": {
            "max_pairs": max_pairs,
            "sampling_steps": sampling_steps,
            "min_snr_db": min_snr_db,
            "max_snr_db": max_snr_db,
        },
    }
    torch.save(payload, cache_path)
    print(f"\nSaved cache: {cache_path}")
    print(f"Cached samples: {payload['z_hat'].size(0)}")


# ============================================================
# Validation for before/after Generator
# ============================================================
@torch.no_grad()
def validate_generators(
    original_generator: nn.Module,
    tuned_generator: nn.Module,
    validation_batch,
    save_root: str,
    epoch: int,
) -> Dict[str, float]:
    z_hat, z_true, real, labels, _ = validation_batch
    device = next(tuned_generator.parameters()).device
    z_hat = z_hat.to(device)
    z_true = z_true.to(device)
    real = real.to(device)
    labels = labels.to(device)

    original_generator.eval()
    tuned_generator.eval()
    before_sep = original_generator(z_hat, labels)
    after_sep = tuned_generator(z_hat, labels)
    before_oracle = original_generator(z_true, labels)
    after_oracle = tuned_generator(z_true, labels)

    metrics = {
        "before_sep_l1": F.l1_loss(before_sep, real).item(),
        "after_sep_l1": F.l1_loss(after_sep, real).item(),
        "before_sep_psnr": image_psnr(before_sep, real),
        "after_sep_psnr": image_psnr(after_sep, real),
        "oracle_drift_l1": F.l1_loss(after_oracle, before_oracle).item(),
        "after_oracle_l1": F.l1_loss(after_oracle, real).item(),
    }

    count = min(16, real.size(0))
    # Rows: real / before separated / after separated / original oracle / tuned oracle
    comparison = torch.cat(
        [real[:count], before_sep[:count], after_sep[:count], before_oracle[:count], after_oracle[:count]],
        dim=0,
    )
    save_image(
        ((comparison + 1.0) / 2.0).clamp(0, 1),
        os.path.join(save_root, f"comparison_ep{epoch}.png"),
        nrow=count,
    )
    return metrics


# ============================================================
# Main
# ============================================================
def main() -> None:
    # ---------------------------
    # Configuration
    # ---------------------------
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 256
    label_dim = 64
    num_classes = 2

    # Cache creation. 5,000 pairs -> 10,000 training samples.
    rebuild_cache = False
    cache_pairs = 5000
    cache_batch_size = 64
    ddim_sampling_steps = 50
    cache_snr_min = 0.0
    cache_snr_max = 30.0

    # Generator fine-tuning.
    epochs = 20
    batch_size = 64
    num_workers = 4
    lr_generator = 1e-5
    lr_critic = 2e-5
    n_critic = 1
    grad_clip = 5.0

    # Loss weights. Reconstruction is the main objective.
    # lambda_sep_recon = 1.0
    # lambda_oracle_recon = 0.25
    # lambda_oracle_distill = 0.10
    # lambda_adv = 0.002
    # lambda_random_adv = 0.001
    lambda_gp = 10.0

    lambda_sep_recon = 1.0
    lambda_oracle_recon = 0.25
    lambda_oracle_distill = 0.25

    lambda_adv = 1e-5
    lambda_random_adv = 5e-6

    save_every = 1

    # Paths from the three training files.
    data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    data_root_d = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    joint_checkpoint_path = "./save_model/joint_vae_gan_satellite_latest.pth"
    ddpm_checkpoint_path = "./save_model/DDPM_vae_gan_latent_separator_latest.pth"

    save_root = "./save_data/Finetune_GAN_on_DDPM_latents"
    model_root = "./save_model"
    cache_path = os.path.join(save_root, "ddpm_separated_latent_cache.pt")
    history_path = os.path.join(save_root, "finetune_history.csv")
    latest_path = os.path.join(model_root, "joint_generator_satellite_ddpm_finetuned.pth")
    latest_joint_path = os.path.join(model_root, "vae_gan_ddpm_finetuned_latest.pth")

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)
    set_seed(seed)

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    base_e = datasets.ImageFolder(data_root_e, transform=transform)
    base_e = Subset(base_e, range(min(50000, len(base_e))))
    dataset_e = DomainLabelDataset(base_e, 0)
    dataset_d = SatelliteDataset(data_root_d, transform=transform, domain_label=1)
    dataset_d = Subset(dataset_d, range(min(50000, len(dataset_d))))

    loader_e = DataLoader(
        dataset_e,
        batch_size=cache_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    loader_d = DataLoader(
        dataset_d,
        batch_size=cache_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    encoder, generator, critic, ddpm = load_models(
        joint_checkpoint_path,
        ddpm_checkpoint_path,
        device,
        z_dim,
        label_dim,
        num_classes,
    )
    schedule = DiffusionSchedule(timesteps=400).to(device)

    # Frozen reference protects the old mapping and allows before/after validation.
    original_generator = copy.deepcopy(generator).eval()
    set_requires_grad(original_generator, False)
    set_requires_grad(encoder, False)
    set_requires_grad(ddpm, False)

    if rebuild_cache or not os.path.exists(cache_path):
        build_latent_cache(
            encoder,
            ddpm,
            schedule,
            loader_e,
            loader_d,
            cache_path,
            device,
            z_dim,
            cache_pairs,
            ddim_sampling_steps,
            cache_snr_min,
            cache_snr_max,
        )
    else:
        print(f"Using existing cache: {cache_path}")

    cached_dataset = CachedSeparatedLatentDataset(cache_path)
    if len(cached_dataset) < 32:
        raise RuntimeError("The cache is too small for fine-tuning.")

    # Fixed deterministic validation subset at the start of the cache.
    validation_count = min(64, len(cached_dataset))
    validation_batch = tuple(
        torch.stack([cached_dataset[i][field] for i in range(validation_count)])
        for field in range(5)
    )

    train_loader = DataLoader(
        cached_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    set_requires_grad(generator, True)
    set_requires_grad(critic, True)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr_generator, betas=(0.0, 0.9))
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=lr_critic, betas=(0.0, 0.9))

    print(f"Device               = {device}")
    print(f"Cached train samples = {len(cached_dataset)}")
    print(f"Fine-tune epochs     = {epochs}")
    print("Frozen: Encoder + DDPM")
    print("Trainable: Generator + Critic")

    baseline_metrics = validate_generators(
        original_generator, generator, validation_batch, save_root, epoch=0
    )
    print("Before fine-tuning:", baseline_metrics)

    global_step = 0
    for epoch in range(1, epochs + 1):
        generator.train()
        critic.train()
        sums = {
            "total_g": 0.0,
            "sep_recon": 0.0,
            "oracle_recon": 0.0,
            "oracle_distill": 0.0,
            "adv": 0.0,
            "random_adv": 0.0,
            "critic": 0.0,
            "gp": 0.0,
        }
        batches = 0

        for z_hat, z_true, real, labels, _ in train_loader:
            z_hat = z_hat.to(device, non_blocking=True)
            z_true = z_true.to(device, non_blocking=True)
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            current = real.size(0)

            # -----------------------
            # Critic update
            # -----------------------
            set_requires_grad(critic, True)
            set_requires_grad(generator, False)
            critic_accum = 0.0
            gp_accum = 0.0
            for _ in range(n_critic):
                with torch.no_grad():
                    fake_sep = generator(z_hat, labels)
                    random_z = torch.randn(current, z_dim, device=device)
                    fake_random = generator(random_z, labels)
                fake_for_critic = fake_sep if random.random() < 0.75 else fake_random
                d_real = critic(real, labels)
                d_fake = critic(fake_for_critic, labels)
                gp = gradient_penalty_conditional(critic, real, fake_for_critic, labels, device)
                loss_d = d_fake.mean() - d_real.mean() + lambda_gp * gp
                optimizer_d.zero_grad(set_to_none=True)
                loss_d.backward()
                optimizer_d.step()
                critic_accum += loss_d.item()
                gp_accum += gp.item()

            # -----------------------
            # Generator update
            # -----------------------
            set_requires_grad(critic, False)
            set_requires_grad(generator, True)
            fake_sep = generator(z_hat, labels)
            fake_oracle = generator(z_true, labels)
            with torch.no_grad():
                original_oracle = original_generator(z_true, labels)

            loss_sep = F.l1_loss(fake_sep, real)
            loss_oracle = F.l1_loss(fake_oracle, real)
            loss_distill = F.l1_loss(fake_oracle, original_oracle)
            loss_adv = -critic(fake_sep, labels).mean()

            random_z = torch.randn(current, z_dim, device=device)
            random_fake = generator(random_z, labels)
            loss_random_adv = -critic(random_fake, labels).mean()

            loss_g = (
                lambda_sep_recon * loss_sep
                + lambda_oracle_recon * loss_oracle
                + lambda_oracle_distill * loss_distill
                + lambda_adv * loss_adv
                + lambda_random_adv * loss_random_adv
            )
            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            optimizer_g.step()

            sums["total_g"] += loss_g.item()
            sums["sep_recon"] += loss_sep.item()
            sums["oracle_recon"] += loss_oracle.item()
            sums["oracle_distill"] += loss_distill.item()
            sums["adv"] += loss_adv.item()
            sums["random_adv"] += loss_random_adv.item()
            sums["critic"] += critic_accum / n_critic
            sums["gp"] += gp_accum / n_critic
            batches += 1

            if global_step % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | Step {global_step} | "
                    f"G={loss_g.item():.4f} | SepL1={loss_sep.item():.4f} | "
                    f"OracleL1={loss_oracle.item():.4f} | Distill={loss_distill.item():.4f} | "
                    f"Adv={loss_adv.item():.4f} | D={loss_d.item():.4f} | GP={gp.item():.4f}"
                )
            global_step += 1

        averages = {key: value / max(1, batches) for key, value in sums.items()}
        metrics = validate_generators(
            original_generator, generator, validation_batch, save_root, epoch
        )
        print("=" * 120)
        print(f"Epoch {epoch}/{epochs} | {averages}")
        print(f"Validation: {metrics}")
        print("=" * 120)

        row = {"epoch": epoch, **averages, **metrics}
        append_csv(history_path, row)

        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "source_joint_checkpoint": joint_checkpoint_path,
            "source_ddpm_checkpoint": ddpm_checkpoint_path,
            "config": {
                "lr_generator": lr_generator,
                "lr_critic": lr_critic,
                "lambda_sep_recon": lambda_sep_recon,
                "lambda_oracle_recon": lambda_oracle_recon,
                "lambda_oracle_distill": lambda_oracle_distill,
                "lambda_adv": lambda_adv,
                "lambda_random_adv": lambda_random_adv,
                "cache_pairs": cache_pairs,
                "ddim_sampling_steps": ddim_sampling_steps,
            },
        }
        torch.save(checkpoint, latest_joint_path)
        torch.save(generator.state_dict(), latest_path)

        if epoch % save_every == 0:
            torch.save(
                checkpoint,
                os.path.join(model_root, f"vae_gan_ddpm_finetuned_ep{epoch}.pth"),
            )

    print("Fine-tuning completed.")
    print(f"Fine-tuned Generator: {latest_path}")
    print(f"Joint fine-tune checkpoint: {latest_joint_path}")
    print(f"Validation images/history: {save_root}")


if __name__ == "__main__":
    main()
