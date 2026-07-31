
"""
Stable conditional VAE-GAN latent-separation training.

This version is designed to prevent the encoder / GAN latent space from
collapsing during separation-aware training.

Key design
----------
Stage 1: Conditional encoder warm-up
    - Freeze pretrained Generator.
    - Train Encoder only with direct image reconstruction + weak KL.
    - Use deterministic mu for reconstruction.
    - Keep the best encoder according to direct validation reconstruction.

Stage 2: Separator training
    - Restore and freeze the best Stage-1 Encoder.
    - Keep Generator frozen.
    - Train only the latent Separator.
    - Separation losses cannot move the encoder latent distribution.

This avoids the failure mode where the encoder changes its latent distribution
to make separation loss easier but leaves the pretrained Generator manifold.

Domains
-------
0 = EuroSAT
1 = DeepGlobe

After training:
    image + label -> frozen Conditional Encoder -> mu
    z_mix = mu_E + mu_D (+ AWGN)
    z_mix -> trained Separator -> z_E_hat, z_D_hat
    frozen Conditional Generator reconstructs both outputs

Important
---------
A DDPM trained with the old encoder latent space must be retrained using the
new conditional encoder checkpoint.
"""

import os
import csv
import glob
import math
import random
import copy
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import Generator_Satellite_Conditional


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    # Stage 1: encoder inversion training
    encoder_epochs: int = 60

    # Stage 2: separator-only training
    separator_epochs: int = 80

    batch_size: int = 64
    num_workers: int = 4

    z_dim: int = 256
    generator_label_dim: int = 64
    encoder_label_dim: int = 32
    num_classes: int = 2

    separator_hidden_dim: int = 768
    separator_blocks: int = 4

    lr_encoder: float = 1e-4
    lr_separator: float = 1e-4

    # Encoder objective
    lambda_recon: float = 1.0
    beta_max: float = 1e-5
    kl_warmup_epochs: int = 20

    # Separator objective
    lambda_latent_sep: float = 0.20
    lambda_image_sep: float = 1.00
    lambda_mix: float = 0.10
    lambda_cosine: float = 0.05

    grad_clip: float = 5.0
    save_every: int = 5
    fixed_count: int = 32

    # Separator channel curriculum
    separator_clean_epochs: int = 15
    separator_medium_until: int = 40
    medium_snr_min: float = 20.0
    final_snr_min: float = 10.0
    max_snr: float = 30.0

    validation_snr_db: float = 20.0

    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    pretrained_generator_path: str = (
        "./save_model/G_wgangp_satellite_conditional_C.pth"
    )

    save_root: str = (
        "./save_data/Conditional_VAEGAN_SeparationStable"
    )
    model_root: str = "./save_model"

    encoder_best_path: str = (
        "./save_model/conditional_encoder_direct_best.pth"
    )
    separator_best_path: str = (
        "./save_model/latent_separator_stable_best.pth"
    )
    joint_latest_path: str = (
        "./save_model/conditional_vaegan_separator_stable_latest.pth"
    )


EURO_SAT_LABEL = 0
DEEP_GLOBE_LABEL = 1


# ============================================================
# Datasets
# ============================================================
class SatelliteDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform=None,
        domain_label: int = DEEP_GLOBE_LABEL,
    ):
        self.files = sorted(
            glob.glob(os.path.join(root, "*_sat_*.jpg"))
        )
        self.transform = transform
        self.domain_label = int(domain_label)

        if not self.files:
            raise RuntimeError(
                f"No matching DeepGlobe images found in:\n{root}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, self.domain_label


class DomainLabelDataset(Dataset):
    def __init__(self, dataset: Dataset, domain_label: int):
        self.dataset = dataset
        self.domain_label = int(domain_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return image, self.domain_label


# ============================================================
# Conditional encoder
# ============================================================
class ConditionalVAEEncoder(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        num_classes: int = 2,
        label_dim: int = 32,
    ):
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

        self.label_embedding = nn.Embedding(
            num_classes,
            label_dim,
        )

        feature_dim = 512 * 4 * 4
        joint_dim = feature_dim + label_dim

        self.fc_mu = nn.Linear(joint_dim, z_dim)
        self.fc_logvar = nn.Linear(joint_dim, z_dim)

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.features(image).flatten(start_dim=1)
        label_feature = self.label_embedding(labels)

        joint = torch.cat(
            [feature, label_feature],
            dim=1,
        )

        mu = self.fc_mu(joint)
        logvar = torch.clamp(
            self.fc_logvar(joint),
            min=-10.0,
            max=10.0,
        )

        return mu, logvar


# ============================================================
# Separator
# ============================================================
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class JointLatentSeparator(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        hidden_dim: int = 768,
        num_blocks: int = 4,
    ):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(hidden_dim)
                for _ in range(num_blocks)
            ]
        )

        self.head_e = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, z_dim),
        )

        self.head_d = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(
        self,
        z_received: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_layer(z_received)
        hidden = self.blocks(hidden)

        return self.head_e(hidden), self.head_d(hidden)


# ============================================================
# Utilities
# ============================================================
def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    kl = -0.5 * (
        1.0
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )
    return kl.mean()


def add_awgn_by_snr(
    signal: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    if math.isinf(snr_db):
        return signal

    signal_power = (
        signal.pow(2)
        .mean(dim=1, keepdim=True)
        .clamp_min(1e-12)
    )

    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear

    noise = (
        torch.randn_like(signal)
        * torch.sqrt(noise_power)
    )

    return signal + noise


def get_separator_snr(
    separator_epoch: int,
    config: Config,
) -> float:
    if separator_epoch <= config.separator_clean_epochs:
        return math.inf

    if separator_epoch <= config.separator_medium_until:
        return random.uniform(
            config.medium_snr_min,
            config.max_snr,
        )

    return random.uniform(
        config.final_snr_min,
        config.max_snr,
    )


def cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return (
        1.0
        - F.cosine_similarity(
            prediction,
            target,
            dim=1,
        )
    ).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_requires_grad(
    model: nn.Module,
    value: bool,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = value


def append_csv(path: str, row: dict) -> None:
    exists = os.path.exists(path)

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(row.keys()),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def save_joint_checkpoint(
    path: str,
    stage: str,
    epoch: int,
    encoder: nn.Module,
    separator: nn.Module,
    generator: nn.Module,
    config: Config,
    metrics: dict | None = None,
) -> None:
    checkpoint = {
        "stage": stage,
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "separator": separator.state_dict(),
        "generator": generator.state_dict(),
        "config": vars(config),
        "domain_labels": {
            "EuroSAT": EURO_SAT_LABEL,
            "DeepGlobe": DEEP_GLOBE_LABEL,
        },
    }

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, path)


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate_encoder(
    encoder: nn.Module,
    generator: nn.Module,
    fixed_e: torch.Tensor,
    fixed_d: torch.Tensor,
    epoch: int,
    save_root: str,
) -> dict:
    encoder.eval()
    generator.eval()

    batch_size = min(
        fixed_e.size(0),
        fixed_d.size(0),
    )

    image_e = fixed_e[:batch_size]
    image_d = fixed_d[:batch_size]

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

    mu_e, _ = encoder(image_e, label_e)
    mu_d, _ = encoder(image_d, label_d)

    direct_e = generator(mu_e, label_e)
    direct_d = generator(mu_d, label_d)

    metrics = {
        "direct_l1_e":
            F.l1_loss(direct_e, image_e).item(),
        "direct_l1_d":
            F.l1_loss(direct_d, image_d).item(),
        "mu_mean_e": mu_e.mean().item(),
        "mu_mean_d": mu_d.mean().item(),
        "mu_std_e": mu_e.std().item(),
        "mu_std_d": mu_d.std().item(),
    }

    nrow = max(1, int(math.sqrt(batch_size)))

    save_image(
        (direct_e + 1.0) / 2.0,
        os.path.join(
            save_root,
            f"encoder_direct_E_ep{epoch}.png",
        ),
        nrow=nrow,
    )

    save_image(
        (direct_d + 1.0) / 2.0,
        os.path.join(
            save_root,
            f"encoder_direct_D_ep{epoch}.png",
        ),
        nrow=nrow,
    )

    return metrics


@torch.no_grad()
def validate_separator(
    encoder: nn.Module,
    separator: nn.Module,
    generator: nn.Module,
    fixed_e: torch.Tensor,
    fixed_d: torch.Tensor,
    epoch: int,
    save_root: str,
    snr_db: float,
) -> dict:
    encoder.eval()
    separator.eval()
    generator.eval()

    batch_size = min(
        fixed_e.size(0),
        fixed_d.size(0),
    )

    image_e = fixed_e[:batch_size]
    image_d = fixed_d[:batch_size]

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

    mu_e, _ = encoder(image_e, label_e)
    mu_d, _ = encoder(image_d, label_d)

    direct_e = generator(mu_e, label_e)
    direct_d = generator(mu_d, label_d)

    z_mix = mu_e + mu_d
    z_received = add_awgn_by_snr(z_mix, snr_db)

    pred_e, pred_d = separator(z_received)

    separated_e = generator(pred_e, label_e)
    separated_d = generator(pred_d, label_d)

    metrics = {
        "direct_l1_e":
            F.l1_loss(direct_e, image_e).item(),
        "direct_l1_d":
            F.l1_loss(direct_d, image_d).item(),

        "latent_mse_e":
            F.mse_loss(pred_e, mu_e).item(),
        "latent_mse_d":
            F.mse_loss(pred_d, mu_d).item(),

        "latent_cos_e":
            F.cosine_similarity(
                pred_e,
                mu_e,
                dim=1,
            ).mean().item(),
        "latent_cos_d":
            F.cosine_similarity(
                pred_d,
                mu_d,
                dim=1,
            ).mean().item(),

        "separated_l1_e":
            F.l1_loss(separated_e, image_e).item(),
        "separated_l1_d":
            F.l1_loss(separated_d, image_d).item(),

        "separated_to_direct_l1_e":
            F.l1_loss(separated_e, direct_e).item(),
        "separated_to_direct_l1_d":
            F.l1_loss(separated_d, direct_d).item(),

        "mixture_mse":
            F.mse_loss(
                pred_e + pred_d,
                z_mix,
            ).item(),
    }

    nrow = max(1, int(math.sqrt(batch_size)))

    images = {
        "direct_E": direct_e,
        "direct_D": direct_d,
        "separated_E": separated_e,
        "separated_D": separated_d,
    }

    for name, image in images.items():
        save_image(
            (image + 1.0) / 2.0,
            os.path.join(
                save_root,
                f"separator_{name}_ep{epoch}.png",
            ),
            nrow=nrow,
        )

    return metrics


# ============================================================
# Main
# ============================================================
def main():
    config = Config()

    if (
        config.device == "cuda"
        and not torch.cuda.is_available()
    ):
        print("CUDA unavailable; switching to CPU.")
        device = "cpu"
    else:
        device = config.device

    set_seed(config.seed)

    os.makedirs(config.save_root, exist_ok=True)
    os.makedirs(config.model_root, exist_ok=True)

    encoder_history_path = os.path.join(
        config.save_root,
        "encoder_training_history.csv",
    )
    separator_history_path = os.path.join(
        config.save_root,
        "separator_training_history.csv",
    )
    validation_history_path = os.path.join(
        config.save_root,
        "validation_history.csv",
    )

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    base_e = datasets.ImageFolder(
        root=config.eurosat_root,
        transform=transform,
    )
    base_e = Subset(
        base_e,
        range(min(50000, len(base_e))),
    )
    dataset_e = DomainLabelDataset(
        base_e,
        EURO_SAT_LABEL,
    )

    base_d = SatelliteDataset(
        root=config.deepglobe_root,
        transform=transform,
        domain_label=DEEP_GLOBE_LABEL,
    )
    dataset_d = Subset(
        base_d,
        range(min(50000, len(base_d))),
    )

    loader_e = DataLoader(
        dataset_e,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
    )

    loader_d = DataLoader(
        dataset_d,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
    )

    steps_per_epoch = min(
        len(loader_e),
        len(loader_d),
    )

    print(f"Device              = {device}")
    print(f"EuroSAT images      = {len(dataset_e)}")
    print(f"DeepGlobe images    = {len(dataset_d)}")
    print(f"Steps per epoch     = {steps_per_epoch}")

    encoder = ConditionalVAEEncoder(
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.encoder_label_dim,
    ).to(device)

    separator = JointLatentSeparator(
        z_dim=config.z_dim,
        hidden_dim=config.separator_hidden_dim,
        num_blocks=config.separator_blocks,
    ).to(device)

    generator = Generator_Satellite_Conditional(
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.generator_label_dim,
        out_ch=3,
    ).to(device)

    if not os.path.isfile(
        config.pretrained_generator_path
    ):
        raise FileNotFoundError(
            "Cannot find pretrained generator:\n"
            f"{config.pretrained_generator_path}"
        )

    generator.load_state_dict(
        torch.load(
            config.pretrained_generator_path,
            map_location=device,
        ),
        strict=True,
    )

    generator.eval()
    set_requires_grad(generator, False)

    optimizer_encoder = torch.optim.Adam(
        encoder.parameters(),
        lr=config.lr_encoder,
        betas=(0.9, 0.999),
    )

    optimizer_separator = torch.optim.Adam(
        separator.parameters(),
        lr=config.lr_separator,
        betas=(0.9, 0.999),
    )

    fixed_count = min(
        config.fixed_count,
        len(dataset_e),
        len(dataset_d),
    )

    fixed_e = torch.stack([
        dataset_e[i][0]
        for i in range(fixed_count)
    ]).to(device)

    fixed_d = torch.stack([
        dataset_d[i][0]
        for i in range(fixed_count)
    ]).to(device)

    nrow = max(1, int(math.sqrt(fixed_count)))

    save_image(
        (fixed_e + 1.0) / 2.0,
        os.path.join(
            config.save_root,
            "real_EuroSAT.png",
        ),
        nrow=nrow,
    )

    save_image(
        (fixed_d + 1.0) / 2.0,
        os.path.join(
            config.save_root,
            "real_DeepGlobe.png",
        ),
        nrow=nrow,
    )

    # ========================================================
    # Stage 1: train conditional encoder only
    # ========================================================
    print("\n" + "=" * 100)
    print("STAGE 1: CONDITIONAL ENCODER WARM-UP")
    print("Generator frozen; separator not used.")
    print("=" * 100)

    best_encoder_score = float("inf")
    best_encoder_state = None

    for epoch in range(1, config.encoder_epochs + 1):
        encoder.train()
        separator.eval()
        generator.eval()

        set_requires_grad(encoder, True)
        set_requires_grad(separator, False)
        set_requires_grad(generator, False)

        beta = (
            config.beta_max
            * min(
                1.0,
                epoch / config.kl_warmup_epochs,
            )
        )

        sums = {
            "total": 0.0,
            "recon": 0.0,
            "kl": 0.0,
            "recon_e": 0.0,
            "recon_d": 0.0,
            "mu_std_e": 0.0,
            "mu_std_d": 0.0,
        }

        for step, (
            (image_e, _),
            (image_d, _),
        ) in enumerate(zip(loader_e, loader_d)):
            image_e = image_e.to(
                device,
                non_blocking=True,
            )
            image_d = image_d.to(
                device,
                non_blocking=True,
            )

            batch_now = min(
                image_e.size(0),
                image_d.size(0),
            )

            image_e = image_e[:batch_now]
            image_d = image_d[:batch_now]

            label_e = torch.zeros(
                batch_now,
                dtype=torch.long,
                device=device,
            )
            label_d = torch.ones(
                batch_now,
                dtype=torch.long,
                device=device,
            )

            mu_e, logvar_e = encoder(
                image_e,
                label_e,
            )
            mu_d, logvar_d = encoder(
                image_d,
                label_d,
            )

            # Deterministic mu is used because DDPM will later
            # transmit and separate mu, not a sampled latent.
            direct_e = generator(mu_e, label_e)
            direct_d = generator(mu_d, label_d)

            recon_e = F.l1_loss(direct_e, image_e)
            recon_d = F.l1_loss(direct_d, image_d)
            loss_recon = 0.5 * (recon_e + recon_d)

            loss_kl = 0.5 * (
                kl_divergence(mu_e, logvar_e)
                + kl_divergence(mu_d, logvar_d)
            )

            loss_total = (
                config.lambda_recon * loss_recon
                + beta * loss_kl
            )

            optimizer_encoder.zero_grad(
                set_to_none=True
            )
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(
                encoder.parameters(),
                max_norm=config.grad_clip,
            )

            optimizer_encoder.step()

            sums["total"] += loss_total.item()
            sums["recon"] += loss_recon.item()
            sums["kl"] += loss_kl.item()
            sums["recon_e"] += recon_e.item()
            sums["recon_d"] += recon_d.item()
            sums["mu_std_e"] += mu_e.std().item()
            sums["mu_std_d"] += mu_d.std().item()

            if step % 100 == 0:
                print(
                    f"Encoder Epoch {epoch}/"
                    f"{config.encoder_epochs} | "
                    f"Step={step}/{steps_per_epoch} | "
                    f"Total={loss_total.item():.4f} | "
                    f"ReconE={recon_e.item():.4f} | "
                    f"ReconD={recon_d.item():.4f} | "
                    f"KL={loss_kl.item():.4f} | "
                    f"beta={beta:.7f}"
                )

        averages = {
            key: value / steps_per_epoch
            for key, value in sums.items()
        }

        append_csv(
            encoder_history_path,
            {
                "epoch": epoch,
                "beta": beta,
                **averages,
            },
        )

        if (
            epoch == 1
            or epoch % config.save_every == 0
            or epoch == config.encoder_epochs
        ):
            metrics = validate_encoder(
                encoder=encoder,
                generator=generator,
                fixed_e=fixed_e,
                fixed_d=fixed_d,
                epoch=epoch,
                save_root=config.save_root,
            )

            encoder_score = 0.5 * (
                metrics["direct_l1_e"]
                + metrics["direct_l1_d"]
            )

            append_csv(
                validation_history_path,
                {
                    "stage": "encoder",
                    "epoch": epoch,
                    **metrics,
                },
            )

            print(
                "Encoder validation:",
                metrics,
            )

            if encoder_score < best_encoder_score:
                best_encoder_score = encoder_score
                best_encoder_state = copy.deepcopy(
                    encoder.state_dict()
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "encoder": best_encoder_state,
                        "generator":
                            generator.state_dict(),
                        "metrics": metrics,
                        "config": vars(config),
                    },
                    config.encoder_best_path,
                )

                print(
                    "Saved best encoder | "
                    f"avg direct L1="
                    f"{best_encoder_score:.6f}"
                )

    if best_encoder_state is None:
        raise RuntimeError(
            "No best encoder state was recorded."
        )

    # Restore the best direct-reconstruction encoder and freeze it.
    encoder.load_state_dict(best_encoder_state)
    encoder.eval()
    set_requires_grad(encoder, False)

    print("\nRestored best Stage-1 encoder.")
    print(
        f"Best average direct L1 = "
        f"{best_encoder_score:.6f}"
    )

    # ========================================================
    # Stage 2: train separator only
    # ========================================================
    print("\n" + "=" * 100)
    print("STAGE 2: SEPARATOR-ONLY TRAINING")
    print("Encoder and Generator are frozen.")
    print("=" * 100)

    separator.train()
    set_requires_grad(separator, True)

    best_separator_score = float("inf")

    for epoch in range(
        1,
        config.separator_epochs + 1,
    ):
        encoder.eval()
        generator.eval()
        separator.train()

        set_requires_grad(encoder, False)
        set_requires_grad(generator, False)
        set_requires_grad(separator, True)

        train_snr_db = get_separator_snr(
            epoch,
            config,
        )

        sums = {
            "total": 0.0,
            "latent_sep": 0.0,
            "image_sep": 0.0,
            "mix": 0.0,
            "cosine": 0.0,
        }

        for step, (
            (image_e, _),
            (image_d, _),
        ) in enumerate(zip(loader_e, loader_d)):
            image_e = image_e.to(
                device,
                non_blocking=True,
            )
            image_d = image_d.to(
                device,
                non_blocking=True,
            )

            batch_now = min(
                image_e.size(0),
                image_d.size(0),
            )

            image_e = image_e[:batch_now]
            image_d = image_d[:batch_now]

            label_e = torch.zeros(
                batch_now,
                dtype=torch.long,
                device=device,
            )
            label_d = torch.ones(
                batch_now,
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                mu_e, _ = encoder(
                    image_e,
                    label_e,
                )
                mu_d, _ = encoder(
                    image_d,
                    label_d,
                )

                # Explicit detach: separation gradients can never
                # alter the encoder latent distribution.
                mu_e = mu_e.detach()
                mu_d = mu_d.detach()

                z_mix = mu_e + mu_d
                z_received = add_awgn_by_snr(
                    z_mix,
                    train_snr_db,
                )

            pred_e, pred_d = separator(z_received)

            loss_latent_sep = 0.5 * (
                F.mse_loss(pred_e, mu_e)
                + F.mse_loss(pred_d, mu_d)
            )

            # Generator remains frozen, but gradients pass through
            # it to update the separator.
            separated_e = generator(pred_e, label_e)
            separated_d = generator(pred_d, label_d)

            loss_image_sep = 0.5 * (
                F.l1_loss(separated_e, image_e)
                + F.l1_loss(separated_d, image_d)
            )

            loss_mix = F.mse_loss(
                pred_e + pred_d,
                z_mix,
            )

            loss_cosine = 0.5 * (
                cosine_loss(pred_e, mu_e)
                + cosine_loss(pred_d, mu_d)
            )

            loss_total = (
                config.lambda_latent_sep
                * loss_latent_sep
                + config.lambda_image_sep
                * loss_image_sep
                + config.lambda_mix
                * loss_mix
                + config.lambda_cosine
                * loss_cosine
            )

            optimizer_separator.zero_grad(
                set_to_none=True
            )
            loss_total.backward()

            torch.nn.utils.clip_grad_norm_(
                separator.parameters(),
                max_norm=config.grad_clip,
            )

            optimizer_separator.step()

            sums["total"] += loss_total.item()
            sums["latent_sep"] += (
                loss_latent_sep.item()
            )
            sums["image_sep"] += (
                loss_image_sep.item()
            )
            sums["mix"] += loss_mix.item()
            sums["cosine"] += loss_cosine.item()

            if step % 100 == 0:
                snr_text = (
                    "clean"
                    if math.isinf(train_snr_db)
                    else f"{train_snr_db:.1f} dB"
                )

                print(
                    f"Separator Epoch {epoch}/"
                    f"{config.separator_epochs} | "
                    f"Step={step}/{steps_per_epoch} | "
                    f"SNR={snr_text} | "
                    f"Total={loss_total.item():.4f} | "
                    f"LatSep={loss_latent_sep.item():.4f} | "
                    f"ImgSep={loss_image_sep.item():.4f} | "
                    f"Mix={loss_mix.item():.4f} | "
                    f"Cos={loss_cosine.item():.4f}"
                )

        averages = {
            key: value / steps_per_epoch
            for key, value in sums.items()
        }

        append_csv(
            separator_history_path,
            {
                "epoch": epoch,
                "train_snr_db": (
                    "clean"
                    if math.isinf(train_snr_db)
                    else train_snr_db
                ),
                **averages,
            },
        )

        metrics = None

        if (
            epoch == 1
            or epoch % config.save_every == 0
            or epoch == config.separator_epochs
        ):
            metrics = validate_separator(
                encoder=encoder,
                separator=separator,
                generator=generator,
                fixed_e=fixed_e,
                fixed_d=fixed_d,
                epoch=epoch,
                save_root=config.save_root,
                snr_db=config.validation_snr_db,
            )

            append_csv(
                validation_history_path,
                {
                    "stage": "separator",
                    "epoch": epoch,
                    **metrics,
                },
            )

            print(
                "Separator validation:",
                metrics,
            )

            # Prioritize decoded image quality, not only latent MSE.
            separator_score = 0.5 * (
                metrics["separated_l1_e"]
                + metrics["separated_l1_d"]
            )

            if separator_score < best_separator_score:
                best_separator_score = separator_score

                torch.save(
                    {
                        "epoch": epoch,
                        "encoder":
                            encoder.state_dict(),
                        "separator":
                            separator.state_dict(),
                        "generator":
                            generator.state_dict(),
                        "metrics": metrics,
                        "config": vars(config),
                    },
                    config.separator_best_path,
                )

                print(
                    "Saved best separator | "
                    f"avg separated L1="
                    f"{best_separator_score:.6f}"
                )

        save_joint_checkpoint(
            path=config.joint_latest_path,
            stage="separator_only",
            epoch=epoch,
            encoder=encoder,
            separator=separator,
            generator=generator,
            config=config,
            metrics=metrics,
        )

    print("\nTraining completed.")
    print(
        "Best encoder:",
        config.encoder_best_path,
    )
    print(
        "Best separator:",
        config.separator_best_path,
    )
    print(
        "Latest joint checkpoint:",
        config.joint_latest_path,
    )
    print(
        "Encoder and Generator remained frozen during "
        "all separator training."
    )


if __name__ == "__main__":
    main()
