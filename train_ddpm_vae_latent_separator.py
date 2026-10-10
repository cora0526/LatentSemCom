import os
import csv
import math
import glob
import random
from itertools import cycle
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

# These classes must match the definitions used by your VAE training file.


# ============================================================
# Dataset definitions (same domain convention as your training)
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.domain_label


class DomainLabelDataset(Dataset):
    def __init__(self, dataset: Dataset, domain_label: int):
        self.dataset = dataset
        self.domain_label = int(domain_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return image, self.domain_label


# ============================================================
# Shared VAE encoder: must remain identical to the source file
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
# Conditional VAE Decoder: identical to the uploaded VAE file
#   0 = EuroSAT
#   1 = DeepGlobe
# ============================================================
class ConditionalVAEDecoder(nn.Module):
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
        label_feature = self.label_embedding(labels)
        conditional_z = torch.cat([z, label_feature], dim=1)
        h = self.fc(conditional_z)
        h = h.view(z.size(0), 512, 4, 4)
        return self.decoder(h)


# ============================================================
# DDPM components
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
    """Predict epsilon for p(z_target | z_mix, domain_label)."""

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
            *[ResidualMLPBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
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
    ) -> torch.Tensor:
        t_emb = self.time_embedding(timesteps)
        y_emb = self.label_embedding(labels)
        h = torch.cat([z_t, cond, t_emb, y_emb], dim=1)
        h = self.input_layer(h)
        h = self.blocks(h)
        return self.output_layer(h)


class DiffusionSchedule(nn.Module):
    def __init__(
        self,
        timesteps: int = 400,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
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
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
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


def append_csv(path: str, row: dict) -> None:
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


def latent_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    mse = F.mse_loss(pred, target).item()
    cosine = F.cosine_similarity(pred, target, dim=1).mean().item()
    return {"mse": mse, "cosine": cosine}


@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    schedule: DiffusionSchedule,
    cond: torch.Tensor,
    labels: torch.Tensor,
    z_dim: int,
    clip_x0: float | None = None,
) -> torch.Tensor:
    model.eval()
    x = torch.randn(cond.size(0), z_dim, device=cond.device)

    for step in reversed(range(schedule.timesteps)):
        t = torch.full((cond.size(0),), step, device=cond.device, dtype=torch.long)
        pred_eps = model(x, t, labels, cond)
        pred_x0 = schedule.predict_x0_from_eps(x, t, pred_eps)
        if clip_x0 is not None:
            pred_x0 = pred_x0.clamp(-clip_x0, clip_x0)

        beta_t = schedule.extract(schedule.betas, t, x.shape)
        alpha_t = schedule.extract(schedule.alphas, t, x.shape)
        alpha_bar_t = schedule.extract(schedule.alpha_bars, t, x.shape)
        alpha_bar_prev = schedule.extract(schedule.alpha_bars_prev, t, x.shape)

        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef_xt = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
        mean = coef_x0 * pred_x0 + coef_xt * x

        if step > 0:
            variance = schedule.extract(schedule.posterior_variance, t, x.shape)
            x = mean + torch.sqrt(variance) * torch.randn_like(x)
        else:
            x = mean

    return x


@torch.no_grad()
def validate(
    model: nn.Module,
    schedule: DiffusionSchedule,
    encoder: nn.Module,
    decoder: nn.Module,
    fixed_e: torch.Tensor,
    fixed_d: torch.Tensor,
    epoch: int,
    save_root: str,
    z_dim: int,
    validation_snr_db: float,
) -> dict:
    model.eval()
    encoder.eval()
    decoder.eval()

    label_e = torch.zeros(fixed_e.size(0), dtype=torch.long, device=fixed_e.device)
    label_d = torch.ones(fixed_d.size(0), dtype=torch.long, device=fixed_d.device)

    z_e, _ = encoder(fixed_e)
    z_d, _ = encoder(fixed_d)
    z_mix_clean = z_e + z_d
    z_rx = add_awgn_by_snr(z_mix_clean, validation_snr_db)

    pred_e = sample_ddpm(model, schedule, z_rx, label_e, z_dim)
    pred_d = sample_ddpm(model, schedule, z_rx, label_d, z_dim)

    recon_e = decoder(pred_e, label_e)
    recon_d = decoder(pred_d, label_d)
    oracle_e = decoder(z_e, label_e)
    oracle_d = decoder(z_d, label_d)

    # Zero-condition test: reveals label-only generation.
    zero_cond = torch.zeros_like(z_rx)
    zero_e = sample_ddpm(model, schedule, zero_cond, label_e, z_dim)
    zero_d = sample_ddpm(model, schedule, zero_cond, label_d, z_dim)
    zero_recon_e = decoder(zero_e, label_e)
    zero_recon_d = decoder(zero_d, label_d)

    # Partner-swap test: keep EuroSAT target, shuffle DeepGlobe partner.
    permutation = torch.randperm(z_d.size(0), device=z_d.device)
    swapped_mix = z_e + z_d[permutation]
    swapped_e = sample_ddpm(model, schedule, swapped_mix, label_e, z_dim)
    swapped_recon_e = decoder(swapped_e, label_e)

    def image_l1(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.l1_loss(a, b).item()

    metrics_e = latent_metrics(pred_e, z_e)
    metrics_d = latent_metrics(pred_d, z_d)
    metrics = {
        "latent_mse_e": metrics_e["mse"],
        "latent_mse_d": metrics_d["mse"],
        "latent_cos_e": metrics_e["cosine"],
        "latent_cos_d": metrics_d["cosine"],
        "image_l1_e_to_real": image_l1(recon_e, fixed_e),
        "image_l1_d_to_real": image_l1(recon_d, fixed_d),
        "image_l1_e_to_oracle": image_l1(recon_e, oracle_e),
        "image_l1_d_to_oracle": image_l1(recon_d, oracle_d),
        "zero_cond_l1_e_to_real": image_l1(zero_recon_e, fixed_e),
        "zero_cond_l1_d_to_real": image_l1(zero_recon_d, fixed_d),
        "partner_swap_e_change": image_l1(swapped_recon_e, recon_e),
    }

    count = fixed_e.size(0)
    nrow = max(1, int(math.sqrt(count)))
    save_image((fixed_e + 1.0) / 2.0, os.path.join(save_root, f"val_real_E_ep{epoch}.png"), nrow=nrow)
    save_image((fixed_d + 1.0) / 2.0, os.path.join(save_root, f"val_real_D_ep{epoch}.png"), nrow=nrow)
    save_image((oracle_e + 1.0) / 2.0, os.path.join(save_root, f"val_oracle_E_ep{epoch}.png"), nrow=nrow)
    save_image((oracle_d + 1.0) / 2.0, os.path.join(save_root, f"val_oracle_D_ep{epoch}.png"), nrow=nrow)
    save_image((recon_e + 1.0) / 2.0, os.path.join(save_root, f"val_sep_E_ep{epoch}.png"), nrow=nrow)
    save_image((recon_d + 1.0) / 2.0, os.path.join(save_root, f"val_sep_D_ep{epoch}.png"), nrow=nrow)
    save_image((zero_recon_e + 1.0) / 2.0, os.path.join(save_root, f"val_zero_cond_E_ep{epoch}.png"), nrow=nrow)
    save_image((zero_recon_d + 1.0) / 2.0, os.path.join(save_root, f"val_zero_cond_D_ep{epoch}.png"), nrow=nrow)
    save_image((swapped_recon_e + 1.0) / 2.0, os.path.join(save_root, f"val_partner_swap_E_ep{epoch}.png"), nrow=nrow)

    return metrics


def load_pretrained_vae(
    checkpoint_path: str,
    device: str,
    z_dim: int,
    label_dim: int,
    num_classes: int,
):
    """Load the encoder and conditional VAE decoder trained by the uploaded file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Cannot find pretrained VAE checkpoint:\n{checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "encoder" not in checkpoint or "decoder" not in checkpoint:
        raise KeyError(
            "The VAE checkpoint must contain 'encoder' and 'decoder' state dictionaries."
        )

    checkpoint_z_dim = int(checkpoint.get("z_dim", z_dim))
    if checkpoint_z_dim != z_dim:
        raise ValueError(
            f"z_dim mismatch: checkpoint={checkpoint_z_dim}, current setting={z_dim}"
        )

    encoder = SharedVAEEncoder(z_dim=z_dim).to(device)
    decoder = ConditionalVAEDecoder(
        z_dim=z_dim,
        num_classes=num_classes,
        label_dim=label_dim,
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    decoder.load_state_dict(checkpoint["decoder"], strict=True)

    encoder.eval()
    decoder.eval()
    set_requires_grad(encoder, False)
    set_requires_grad(decoder, False)

    print(f"Loaded pretrained VAE checkpoint epoch = {checkpoint.get('epoch', 'unknown')}")
    return encoder, decoder


# ============================================================
# Main training
# ============================================================
def main():
    # ---------------------------
    # Core settings
    # ---------------------------
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 256
    label_dim = 64
    num_classes = 2

    epochs = 100
    steps_per_epoch = 1000
    batch_size = 32
    num_workers = 4
    learning_rate = 2e-4
    grad_clip = 1.0

    # Use one consistent diffusion horizon for both training and sampling.
    diffusion_steps = 400
    lambda_x0 = 0.2

    # Channel curriculum. Start with a clean mixture, then add AWGN.
    clean_epochs = 20
    min_train_snr_db = 10.0
    max_train_snr_db = 30.0
    validation_snr_db = 20.0

    validate_every = 5
    fixed_count = 16

    # ---------------------------
    # Paths
    # ---------------------------
    data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    data_root_d = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    # This is produced by your uploaded VAE training file.
    vae_checkpoint_path = "./save_model/shared_vae_satellite_latest.pth"

    save_root = "./save_data/DDPM_VAE_latent_separator"
    model_root = "./save_model"
    latest_model_path = os.path.join(model_root, "DDPM_vae_latent_separator_latest.pth")
    history_path = os.path.join(save_root, "training_history.csv")
    validation_history_path = os.path.join(save_root, "validation_history.csv")

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)
    set_seed(seed)

    # ---------------------------
    # Datasets and loaders
    # ---------------------------
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

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
    # Frozen pretrained VAE encoder and decoder
    # ---------------------------
    encoder, decoder = load_pretrained_vae(
        checkpoint_path=vae_checkpoint_path,
        device=device,
        z_dim=z_dim,
        label_dim=label_dim,
        num_classes=num_classes,
    )

    # ---------------------------
    # DDPM and optimizer
    # ---------------------------
    ddpm = LatentConditionalDDPM(
        z_dim=z_dim,
        num_classes=num_classes,
        time_dim=128,
        label_dim=64,
        hidden_dim=768,
        num_blocks=4,
        dropout=0.0,
    ).to(device)
    schedule = DiffusionSchedule(timesteps=diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"Device             = {device}")
    print(f"EuroSAT images     = {len(dataset_e)}")
    print(f"DeepGlobe images   = {len(dataset_d)}")
    print(f"Steps per epoch    = {steps_per_epoch}")
    print(f"Diffusion steps    = {diffusion_steps}")

    iter_e = cycle(loader_e)
    iter_d = cycle(loader_d)
    global_step = 0

    for epoch in range(1, epochs + 1):
        ddpm.train()
        epoch_total = 0.0
        epoch_eps_e = 0.0
        epoch_eps_d = 0.0
        epoch_x0_e = 0.0
        epoch_x0_d = 0.0
        epoch_latent_std_e = 0.0
        epoch_latent_std_d = 0.0

        for step in range(steps_per_epoch):
            image_e, _ = next(iter_e)
            image_d, _ = next(iter_d)
            image_e = image_e.to(device, non_blocking=True)
            image_d = image_d.to(device, non_blocking=True)

            with torch.no_grad():
                # Deterministic transmitted representation: use mu, not sampled z.
                z_e, _ = encoder(image_e)
                z_d, _ = encoder(image_d)

                z_mix = z_e + z_d
                if epoch <= clean_epochs:
                    train_snr_db = math.inf
                    z_rx = z_mix
                else:
                    train_snr_db = random.uniform(min_train_snr_db, max_train_snr_db)
                    z_rx = add_awgn_by_snr(z_mix, train_snr_db)

            batch_now = z_e.size(0)
            t_e = torch.randint(0, diffusion_steps, (batch_now,), device=device)
            t_d = torch.randint(0, diffusion_steps, (batch_now,), device=device)
            noise_e = torch.randn_like(z_e)
            noise_d = torch.randn_like(z_d)
            z_e_t = schedule.q_sample(z_e, t_e, noise_e)
            z_d_t = schedule.q_sample(z_d, t_d, noise_d)

            label_e = torch.zeros(batch_now, dtype=torch.long, device=device)
            label_d = torch.ones(batch_now, dtype=torch.long, device=device)

            pred_noise_e = ddpm(z_e_t, t_e, label_e, cond=z_rx)
            pred_noise_d = ddpm(z_d_t, t_d, label_d, cond=z_rx)

            loss_eps_e = F.mse_loss(pred_noise_e, noise_e)
            loss_eps_d = F.mse_loss(pred_noise_d, noise_d)

            pred_z0_e = schedule.predict_x0_from_eps(z_e_t, t_e, pred_noise_e)
            pred_z0_d = schedule.predict_x0_from_eps(z_d_t, t_d, pred_noise_d)
            loss_x0_e = F.mse_loss(pred_z0_e, z_e)
            loss_x0_d = F.mse_loss(pred_z0_d, z_d)

            loss_eps = 0.5 * (loss_eps_e + loss_eps_d)
            loss_x0 = 0.5 * (loss_x0_e + loss_x0_d)
            loss = loss_eps + lambda_x0 * loss_x0

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), grad_clip)
            optimizer.step()

            epoch_total += loss.item()
            epoch_eps_e += loss_eps_e.item()
            epoch_eps_d += loss_eps_d.item()
            epoch_x0_e += loss_x0_e.item()
            epoch_x0_d += loss_x0_d.item()
            epoch_latent_std_e += z_e.std().item()
            epoch_latent_std_d += z_d.std().item()

            if global_step % 200 == 0:
                snr_text = "clean" if math.isinf(train_snr_db) else f"{train_snr_db:.1f} dB"
                print(
                    f"Epoch {epoch}/{epochs} | Step {step}/{steps_per_epoch} | "
                    f"SNR={snr_text} | Total={loss.item():.4f} | "
                    f"eps_E={loss_eps_e.item():.4f} | eps_D={loss_eps_d.item():.4f} | "
                    f"x0_E={loss_x0_e.item():.4f} | x0_D={loss_x0_d.item():.4f}"
                )
            global_step += 1

        averages = {
            "epoch": epoch,
            "total_loss": epoch_total / steps_per_epoch,
            "eps_loss_e": epoch_eps_e / steps_per_epoch,
            "eps_loss_d": epoch_eps_d / steps_per_epoch,
            "x0_loss_e": epoch_x0_e / steps_per_epoch,
            "x0_loss_d": epoch_x0_d / steps_per_epoch,
            "latent_std_e": epoch_latent_std_e / steps_per_epoch,
            "latent_std_d": epoch_latent_std_d / steps_per_epoch,
        }
        append_csv(history_path, averages)

        print("=" * 110)
        print(
            f"Epoch {epoch}/{epochs} | Total={averages['total_loss']:.4f} | "
            f"eps_E={averages['eps_loss_e']:.4f} | eps_D={averages['eps_loss_d']:.4f} | "
            f"x0_E={averages['x0_loss_e']:.4f} | x0_D={averages['x0_loss_d']:.4f} | "
            f"std_E={averages['latent_std_e']:.4f} | std_D={averages['latent_std_d']:.4f}"
        )
        print("=" * 110)

        checkpoint = {
            "epoch": epoch,
            "model": ddpm.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "z_dim": z_dim,
                "label_dim": label_dim,
                "num_classes": num_classes,
                "diffusion_steps": diffusion_steps,
                "lambda_x0": lambda_x0,
                "vae_checkpoint_path": vae_checkpoint_path,
                "domain_labels": {"EuroSAT": 0, "DeepGlobe": 1},
            },
        }
        torch.save(checkpoint, latest_model_path)

        if epoch % validate_every == 0 or epoch == 1 or epoch == epochs:
            validation_metrics = validate(
                model=ddpm,
                schedule=schedule,
                encoder=encoder,
                decoder=decoder,
                fixed_e=fixed_e,
                fixed_d=fixed_d,
                epoch=epoch,
                save_root=save_root,
                z_dim=z_dim,
                validation_snr_db=validation_snr_db,
            )
            append_csv(validation_history_path, {"epoch": epoch, **validation_metrics})
            print("Validation:", validation_metrics)
            torch.save(
                checkpoint,
                os.path.join(model_root, f"DDPM_vae_latent_separator_ep{epoch}.pth"),
            )

    print("Training completed.")
    print(f"Latest DDPM checkpoint: {latest_model_path}")
    print(f"Validation images:      {save_root}")


if __name__ == "__main__":
    main()
