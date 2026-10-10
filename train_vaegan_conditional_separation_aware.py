
"""
Train a separation-aware conditional VAE-GAN for two satellite-image domains.

Goal
----
Learn latent representations that are:
1. reconstructable by the conditional GAN generator,
2. conditioned on the source/domain label,
3. easier to separate after linear superposition,
4. robust to AWGN added to the latent mixture.

Domains
-------
0 = EuroSAT
1 = DeepGlobe

Training pipeline
-----------------
EuroSAT image + label 0 -> Conditional Encoder -> z_E
DeepGlobe image + label 1 -> Conditional Encoder -> z_D

z_mix = z_E + z_D
z_rx  = z_mix + AWGN

z_rx -> Joint Latent Separator -> z_E_hat, z_D_hat

Generator(z_E, label 0)       -> direct reconstruction
Generator(z_D, label 1)       -> direct reconstruction
Generator(z_E_hat, label 0)   -> separated reconstruction
Generator(z_D_hat, label 1)   -> separated reconstruction

Stages
------
Stage 1:
    Freeze pretrained Generator/Critic.
    Train Conditional Encoder + Separator.

Stage 2:
    Fine-tune Encoder + Separator + Generator.
    Train Critic with WGAN-GP.

Important
---------
After this training, the DDPM must be retrained using this new conditional
encoder checkpoint. The old DDPM checkpoints are not compatible with the new
latent space.
"""

import os
import csv
import glob
import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import (
    Generator_Satellite_Conditional,
    Critic_Satellite_Conditional,
)


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    epochs: int = 120
    stage1_epochs: int = 60

    batch_size: int = 64
    num_workers: int = 4

    z_dim: int = 256
    label_dim: int = 64
    encoder_label_dim: int = 32
    separator_hidden_dim: int = 768
    separator_blocks: int = 4
    num_classes: int = 2

    lr_encoder_stage1: float = 1e-4
    lr_separator_stage1: float = 2e-4

    lr_encoder_stage2: float = 3e-5
    lr_separator_stage2: float = 5e-5
    lr_generator_stage2: float = 1e-5
    lr_critic_stage2: float = 5e-5

    beta_max: float = 1e-4
    kl_warmup_epochs: int = 20

    # Direct single-image reconstruction
    lambda_recon: float = 1.0

    # Separation-aware objectives
    lambda_latent_sep: float = 1.0
    lambda_image_sep: float = 0.50
    lambda_mix: float = 0.20
    lambda_residual: float = 0.02

    # GAN objectives, active in stage 2
    lambda_adv: float = 0.005
    lambda_sep_adv: float = 0.005
    lambda_random_gan: float = 0.002

    lambda_gp: float = 10.0
    n_critic: int = 3

    grad_clip: float = 5.0
    save_every: int = 5
    fixed_count: int = 32

    # Channel curriculum for separation branch
    clean_epochs: int = 15
    medium_snr_until: int = 40
    medium_snr_min: float = 20.0
    final_snr_min: float = 10.0
    max_snr: float = 30.0

    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    pretrained_generator_path: str = (
        "./save_model/G_wgangp_satellite_conditional_C.pth"
    )
    pretrained_critic_path: str = (
        "./save_model/D_wgangp_satellite_conditional_C.pth"
    )

    save_root: str = (
        "./save_data/Conditional_VAEGAN_SeparationAware"
    )
    model_root: str = "./save_model"

    latest_checkpoint: str = (
        "./save_model/"
        "conditional_vaegan_separation_aware_latest.pth"
    )
    best_checkpoint: str = (
        "./save_model/"
        "conditional_vaegan_separation_aware_best.pth"
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
        image = Image.open(
            self.files[index]
        ).convert("RGB")

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
# Conditional VAE Encoder
# ============================================================
class ConditionalVAEEncoder(nn.Module):
    """
    Encoder receives both image and domain label.

    CNN feature:
        [B, 512, 4, 4] -> [B, 8192]

    Label:
        [B] -> embedding [B, encoder_label_dim]

    Output:
        mu, logvar: [B, z_dim]
    """

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
# Joint latent separator
# ============================================================
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class JointLatentSeparator(nn.Module):
    """
    Input:
        received mixed latent [B, z_dim]

    Output:
        estimated EuroSAT latent [B, z_dim]
        estimated DeepGlobe latent [B, z_dim]

    A shared backbone with two output heads learns both sources jointly.
    """

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
        h = self.input_layer(z_received)
        h = self.blocks(h)

        z_e_hat = self.head_e(h)
        z_d_hat = self.head_d(h)

        return z_e_hat, z_d_hat


# ============================================================
# VAE / channel utilities
# ============================================================
def reparameterize(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return mu + std * epsilon


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

    return (
        signal
        + torch.randn_like(signal)
        * torch.sqrt(noise_power)
    )


def get_train_snr(
    epoch: int,
    config: Config,
) -> float:
    if epoch <= config.clean_epochs:
        return math.inf

    if epoch <= config.medium_snr_until:
        return random.uniform(
            config.medium_snr_min,
            config.max_snr,
        )

    return random.uniform(
        config.final_snr_min,
        config.max_snr,
    )


# ============================================================
# WGAN-GP
# ============================================================
def gradient_penalty_conditional(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> torch.Tensor:
    batch_size = real.size(0)

    epsilon = torch.rand(
        batch_size,
        1,
        1,
        1,
        device=device,
    )

    interpolated = (
        epsilon * real
        + (1.0 - epsilon) * fake
    )
    interpolated.requires_grad_(True)

    mixed_scores = critic(
        interpolated,
        labels,
    )

    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    return (
        (gradient_norm - 1.0).pow(2)
    ).mean()


# ============================================================
# General utilities
# ============================================================
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


def save_checkpoint(
    path: str,
    epoch: int,
    stage: str,
    encoder: nn.Module,
    separator: nn.Module,
    generator: nn.Module,
    critic: nn.Module,
    optimizers: dict,
    config: Config,
    validation: dict | None = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "stage": stage,
        "encoder": encoder.state_dict(),
        "separator": separator.state_dict(),
        "generator": generator.state_dict(),
        "critic": critic.state_dict(),
        "optimizers": {
            name: optimizer.state_dict()
            for name, optimizer in optimizers.items()
        },
        "config": vars(config),
        "domain_labels": {
            "EuroSAT": EURO_SAT_LABEL,
            "DeepGlobe": DEEP_GLOBE_LABEL,
        },
    }

    if validation is not None:
        checkpoint["validation"] = validation

    torch.save(checkpoint, path)


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate(
    encoder: nn.Module,
    separator: nn.Module,
    generator: nn.Module,
    fixed_e: torch.Tensor,
    fixed_d: torch.Tensor,
    epoch: int,
    save_root: str,
    validation_snr_db: float = 20.0,
) -> dict:
    encoder.eval()
    separator.eval()
    generator.eval()

    batch_size = min(
        fixed_e.size(0),
        fixed_d.size(0),
    )
    fixed_e = fixed_e[:batch_size]
    fixed_d = fixed_d[:batch_size]

    label_e = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=fixed_e.device,
    )
    label_d = torch.ones(
        batch_size,
        dtype=torch.long,
        device=fixed_e.device,
    )

    z_e, _ = encoder(fixed_e, label_e)
    z_d, _ = encoder(fixed_d, label_d)

    z_mix_clean = z_e + z_d
    z_rx = add_awgn_by_snr(
        z_mix_clean,
        validation_snr_db,
    )

    z_e_hat, z_d_hat = separator(z_rx)

    direct_e = generator(z_e, label_e)
    direct_d = generator(z_d, label_d)

    separated_e = generator(z_e_hat, label_e)
    separated_d = generator(z_d_hat, label_d)

    metrics = {
        "latent_mse_e":
            F.mse_loss(z_e_hat, z_e).item(),
        "latent_mse_d":
            F.mse_loss(z_d_hat, z_d).item(),

        "latent_cos_e":
            F.cosine_similarity(
                z_e_hat,
                z_e,
                dim=1,
            ).mean().item(),
        "latent_cos_d":
            F.cosine_similarity(
                z_d_hat,
                z_d,
                dim=1,
            ).mean().item(),

        "direct_image_l1_e":
            F.l1_loss(direct_e, fixed_e).item(),
        "direct_image_l1_d":
            F.l1_loss(direct_d, fixed_d).item(),

        "separated_image_l1_e":
            F.l1_loss(separated_e, fixed_e).item(),
        "separated_image_l1_d":
            F.l1_loss(separated_d, fixed_d).item(),

        "separated_to_oracle_l1_e":
            F.l1_loss(separated_e, direct_e).item(),
        "separated_to_oracle_l1_d":
            F.l1_loss(separated_d, direct_d).item(),

        "mixture_mse":
            F.mse_loss(
                z_e_hat + z_d_hat,
                z_mix_clean,
            ).item(),
    }

    nrow = max(
        1,
        int(math.sqrt(batch_size)),
    )

    images = {
        "real_E": fixed_e,
        "real_D": fixed_d,
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
                f"val_{name}_ep{epoch}.png",
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

    os.makedirs(
        config.save_root,
        exist_ok=True,
    )
    os.makedirs(
        config.model_root,
        exist_ok=True,
    )

    history_path = os.path.join(
        config.save_root,
        "training_history.csv",
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

    # Balanced paired loaders are essential for separation training.
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

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------
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
        label_dim=config.label_dim,
        out_ch=3,
    ).to(device)

    critic = Critic_Satellite_Conditional(
        in_ch=3,
        num_classes=config.num_classes,
        feature_dim=512,
    ).to(device)

    if not os.path.isfile(
        config.pretrained_generator_path
    ):
        raise FileNotFoundError(
            "Cannot find pretrained generator:\n"
            f"{config.pretrained_generator_path}"
        )

    if not os.path.isfile(
        config.pretrained_critic_path
    ):
        raise FileNotFoundError(
            "Cannot find pretrained critic:\n"
            f"{config.pretrained_critic_path}"
        )

    generator.load_state_dict(
        torch.load(
            config.pretrained_generator_path,
            map_location=device,
        ),
        strict=True,
    )

    critic.load_state_dict(
        torch.load(
            config.pretrained_critic_path,
            map_location=device,
        ),
        strict=True,
    )

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------
    optimizer_stage1 = torch.optim.Adam(
        [
            {
                "params": encoder.parameters(),
                "lr": config.lr_encoder_stage1,
            },
            {
                "params": separator.parameters(),
                "lr": config.lr_separator_stage1,
            },
        ],
        betas=(0.9, 0.999),
    )

    optimizer_stage2 = torch.optim.Adam(
        [
            {
                "params": encoder.parameters(),
                "lr": config.lr_encoder_stage2,
            },
            {
                "params": separator.parameters(),
                "lr": config.lr_separator_stage2,
            },
            {
                "params": generator.parameters(),
                "lr": config.lr_generator_stage2,
            },
        ],
        betas=(0.0, 0.9),
    )

    optimizer_critic = torch.optim.Adam(
        critic.parameters(),
        lr=config.lr_critic_stage2,
        betas=(0.0, 0.9),
    )

    optimizers = {
        "stage1": optimizer_stage1,
        "stage2": optimizer_stage2,
        "critic": optimizer_critic,
    }

    # --------------------------------------------------------
    # Fixed validation samples
    # --------------------------------------------------------
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

    save_image(
        (fixed_e + 1.0) / 2.0,
        os.path.join(
            config.save_root,
            "real_EuroSAT.png",
        ),
        nrow=max(1, int(math.sqrt(fixed_count))),
    )

    save_image(
        (fixed_d + 1.0) / 2.0,
        os.path.join(
            config.save_root,
            "real_DeepGlobe.png",
        ),
        nrow=max(1, int(math.sqrt(fixed_count))),
    )

    global_step = 0
    best_validation_score = float("inf")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    for epoch in range(1, config.epochs + 1):
        beta = (
            config.beta_max
            * min(
                1.0,
                epoch / config.kl_warmup_epochs,
            )
        )

        stage1 = epoch <= config.stage1_epochs

        if stage1:
            stage_name = "stage1_encoder_separator"

            encoder.train()
            separator.train()
            generator.eval()
            critic.eval()

            set_requires_grad(encoder, True)
            set_requires_grad(separator, True)
            set_requires_grad(generator, False)
            set_requires_grad(critic, False)
        else:
            stage_name = "stage2_joint_finetune"

            encoder.train()
            separator.train()
            generator.train()
            critic.train()

            set_requires_grad(encoder, True)
            set_requires_grad(separator, True)
            set_requires_grad(generator, True)
            set_requires_grad(critic, True)

        sums = {
            "total": 0.0,
            "recon": 0.0,
            "kl": 0.0,
            "latent_sep": 0.0,
            "image_sep": 0.0,
            "mix": 0.0,
            "residual": 0.0,
            "adv": 0.0,
            "sep_adv": 0.0,
            "random_adv": 0.0,
            "critic": 0.0,
            "gp": 0.0,
            "std_e": 0.0,
            "std_d": 0.0,
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

            train_snr_db = get_train_snr(
                epoch,
                config,
            )

            # =================================================
            # Critic update, stage 2 only
            # =================================================
            loss_critic = torch.zeros(
                (),
                device=device,
            )
            gp = torch.zeros(
                (),
                device=device,
            )

            if not stage1:
                set_requires_grad(critic, True)
                set_requires_grad(encoder, False)
                set_requires_grad(separator, False)
                set_requires_grad(generator, False)

                critic_total = 0.0
                gp_total = 0.0

                for _ in range(config.n_critic):
                    with torch.no_grad():
                        mu_e_det, logvar_e_det = encoder(
                            image_e,
                            label_e,
                        )
                        mu_d_det, logvar_d_det = encoder(
                            image_d,
                            label_d,
                        )

                        z_e_det = reparameterize(
                            mu_e_det,
                            logvar_e_det,
                        )
                        z_d_det = reparameterize(
                            mu_d_det,
                            logvar_d_det,
                        )

                        z_mix_det = (
                            mu_e_det
                            + mu_d_det
                        )
                        z_rx_det = add_awgn_by_snr(
                            z_mix_det,
                            train_snr_db,
                        )

                        sep_e_det, sep_d_det = separator(
                            z_rx_det
                        )

                        direct_fake_e = generator(
                            z_e_det,
                            label_e,
                        )
                        direct_fake_d = generator(
                            z_d_det,
                            label_d,
                        )

                        sep_fake_e = generator(
                            sep_e_det,
                            label_e,
                        )
                        sep_fake_d = generator(
                            sep_d_det,
                            label_d,
                        )

                    real_all = torch.cat(
                        [image_e, image_d],
                        dim=0,
                    )
                    label_all = torch.cat(
                        [label_e, label_d],
                        dim=0,
                    )

                    fake_all = torch.cat(
                        [
                            direct_fake_e,
                            direct_fake_d,
                            sep_fake_e,
                            sep_fake_d,
                        ],
                        dim=0,
                    )
                    fake_labels = torch.cat(
                        [
                            label_e,
                            label_d,
                            label_e,
                            label_d,
                        ],
                        dim=0,
                    )

                    critic_real = critic(
                        real_all,
                        label_all,
                    ).mean()

                    critic_fake = critic(
                        fake_all,
                        fake_labels,
                    ).mean()

                    # Use separated fakes for GP because separation
                    # quality is the main objective.
                    fake_gp = torch.cat(
                        [sep_fake_e, sep_fake_d],
                        dim=0,
                    )

                    gp = gradient_penalty_conditional(
                        critic=critic,
                        real=real_all,
                        fake=fake_gp,
                        labels=label_all,
                        device=device,
                    )

                    loss_critic_now = (
                        critic_fake
                        - critic_real
                        + config.lambda_gp * gp
                    )

                    optimizer_critic.zero_grad(
                        set_to_none=True
                    )
                    loss_critic_now.backward()
                    optimizer_critic.step()

                    critic_total += (
                        loss_critic_now.item()
                    )
                    gp_total += gp.item()

                loss_critic = torch.tensor(
                    critic_total / config.n_critic,
                    device=device,
                )
                gp = torch.tensor(
                    gp_total / config.n_critic,
                    device=device,
                )

            # =================================================
            # Encoder + Separator (+ Generator) update
            # =================================================
            set_requires_grad(critic, False)
            set_requires_grad(encoder, True)
            set_requires_grad(separator, True)
            set_requires_grad(generator, not stage1)

            mu_e, logvar_e = encoder(
                image_e,
                label_e,
            )
            mu_d, logvar_d = encoder(
                image_d,
                label_d,
            )

            sampled_e = reparameterize(
                mu_e,
                logvar_e,
            )
            sampled_d = reparameterize(
                mu_d,
                logvar_d,
            )

            direct_e = generator(
                sampled_e,
                label_e,
            )
            direct_d = generator(
                sampled_d,
                label_d,
            )

            # Direct reconstruction objective.
            loss_recon = 0.5 * (
                F.l1_loss(direct_e, image_e)
                + F.l1_loss(direct_d, image_d)
            )

            loss_kl = 0.5 * (
                kl_divergence(mu_e, logvar_e)
                + kl_divergence(mu_d, logvar_d)
            )

            # Separation is trained using deterministic mu because
            # this is the representation later transmitted to DDPM.
            z_mix_clean = mu_e + mu_d
            z_rx = add_awgn_by_snr(
                z_mix_clean,
                train_snr_db,
            )

            pred_e, pred_d = separator(z_rx)

            loss_latent_sep = 0.5 * (
                F.mse_loss(pred_e, mu_e)
                + F.mse_loss(pred_d, mu_d)
            )

            separated_e = generator(
                pred_e,
                label_e,
            )
            separated_d = generator(
                pred_d,
                label_d,
            )

            loss_image_sep = 0.5 * (
                F.l1_loss(separated_e, image_e)
                + F.l1_loss(separated_d, image_d)
            )

            loss_mix = F.mse_loss(
                pred_e + pred_d,
                z_mix_clean,
            )

            # Prevent the separator from drifting too far when the
            # direct latent estimate is already informative.
            loss_residual = 0.5 * (
                F.mse_loss(pred_e, mu_e.detach())
                + F.mse_loss(pred_d, mu_d.detach())
            )

            loss_adv = torch.zeros(
                (),
                device=device,
            )
            loss_sep_adv = torch.zeros(
                (),
                device=device,
            )
            loss_random_adv = torch.zeros(
                (),
                device=device,
            )

            if not stage1:
                loss_adv = -0.5 * (
                    critic(direct_e, label_e).mean()
                    + critic(direct_d, label_d).mean()
                )

                loss_sep_adv = -0.5 * (
                    critic(
                        separated_e,
                        label_e,
                    ).mean()
                    + critic(
                        separated_d,
                        label_d,
                    ).mean()
                )

                random_z_e = torch.randn(
                    batch_now,
                    config.z_dim,
                    device=device,
                )
                random_z_d = torch.randn(
                    batch_now,
                    config.z_dim,
                    device=device,
                )

                random_fake_e = generator(
                    random_z_e,
                    label_e,
                )
                random_fake_d = generator(
                    random_z_d,
                    label_d,
                )

                loss_random_adv = -0.5 * (
                    critic(
                        random_fake_e,
                        label_e,
                    ).mean()
                    + critic(
                        random_fake_d,
                        label_d,
                    ).mean()
                )

            loss_total = (
                config.lambda_recon * loss_recon
                + beta * loss_kl
                + config.lambda_latent_sep
                * loss_latent_sep
                + config.lambda_image_sep
                * loss_image_sep
                + config.lambda_mix
                * loss_mix
                + config.lambda_residual
                * loss_residual
                + config.lambda_adv
                * loss_adv
                + config.lambda_sep_adv
                * loss_sep_adv
                + config.lambda_random_gan
                * loss_random_adv
            )

            active_optimizer = (
                optimizer_stage1
                if stage1
                else optimizer_stage2
            )

            active_optimizer.zero_grad(
                set_to_none=True
            )
            loss_total.backward()

            trainable_parameters = (
                list(encoder.parameters())
                + list(separator.parameters())
            )

            if not stage1:
                trainable_parameters += list(
                    generator.parameters()
                )

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=config.grad_clip,
            )

            active_optimizer.step()

            sums["total"] += loss_total.item()
            sums["recon"] += loss_recon.item()
            sums["kl"] += loss_kl.item()
            sums["latent_sep"] += (
                loss_latent_sep.item()
            )
            sums["image_sep"] += (
                loss_image_sep.item()
            )
            sums["mix"] += loss_mix.item()
            sums["residual"] += (
                loss_residual.item()
            )
            sums["adv"] += loss_adv.item()
            sums["sep_adv"] += (
                loss_sep_adv.item()
            )
            sums["random_adv"] += (
                loss_random_adv.item()
            )
            sums["critic"] += loss_critic.item()
            sums["gp"] += gp.item()
            sums["std_e"] += mu_e.std().item()
            sums["std_d"] += mu_d.std().item()

            if global_step % 100 == 0:
                snr_text = (
                    "clean"
                    if math.isinf(train_snr_db)
                    else f"{train_snr_db:.1f} dB"
                )

                print(
                    f"Epoch {epoch}/{config.epochs} | "
                    f"Stage={stage_name} | "
                    f"Step={step}/{steps_per_epoch} | "
                    f"SNR={snr_text} | "
                    f"Total={loss_total.item():.4f} | "
                    f"Recon={loss_recon.item():.4f} | "
                    f"LatSep={loss_latent_sep.item():.4f} | "
                    f"ImgSep={loss_image_sep.item():.4f} | "
                    f"Mix={loss_mix.item():.4f} | "
                    f"KL={loss_kl.item():.4f}"
                )

            global_step += 1

        averages = {
            key: value / steps_per_epoch
            for key, value in sums.items()
        }

        row = {
            "epoch": epoch,
            "stage": stage_name,
            "beta": beta,
            **averages,
        }

        append_csv(
            history_path,
            row,
        )

        print("=" * 130)
        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"Stage={stage_name} | "
            f"Total={averages['total']:.4f} | "
            f"Recon={averages['recon']:.4f} | "
            f"LatSep={averages['latent_sep']:.4f} | "
            f"ImgSep={averages['image_sep']:.4f} | "
            f"Mix={averages['mix']:.4f} | "
            f"KL={averages['kl']:.4f} | "
            f"stdE={averages['std_e']:.4f} | "
            f"stdD={averages['std_d']:.4f}"
        )
        print("=" * 130)

        validation = None

        if (
            epoch % config.save_every == 0
            or epoch == 1
            or epoch == config.stage1_epochs
            or epoch == config.epochs
        ):
            validation = validate(
                encoder=encoder,
                separator=separator,
                generator=generator,
                fixed_e=fixed_e,
                fixed_d=fixed_d,
                epoch=epoch,
                save_root=config.save_root,
                validation_snr_db=20.0,
            )

            append_csv(
                validation_history_path,
                {
                    "epoch": epoch,
                    **validation,
                },
            )

            print(
                "Validation:",
                validation,
            )

            score = 0.5 * (
                validation[
                    "separated_image_l1_e"
                ]
                + validation[
                    "separated_image_l1_d"
                ]
            )

            if score < best_validation_score:
                best_validation_score = score

                save_checkpoint(
                    path=config.best_checkpoint,
                    epoch=epoch,
                    stage=stage_name,
                    encoder=encoder,
                    separator=separator,
                    generator=generator,
                    critic=critic,
                    optimizers=optimizers,
                    config=config,
                    validation=validation,
                )

                print(
                    "Saved best checkpoint | "
                    f"avg separated image L1="
                    f"{best_validation_score:.6f}"
                )

        save_checkpoint(
            path=config.latest_checkpoint,
            epoch=epoch,
            stage=stage_name,
            encoder=encoder,
            separator=separator,
            generator=generator,
            critic=critic,
            optimizers=optimizers,
            config=config,
            validation=validation,
        )

    torch.save(
        encoder.state_dict(),
        os.path.join(
            config.model_root,
            "conditional_separation_aware_encoder.pth",
        ),
    )

    torch.save(
        separator.state_dict(),
        os.path.join(
            config.model_root,
            "joint_latent_separator.pth",
        ),
    )

    torch.save(
        generator.state_dict(),
        os.path.join(
            config.model_root,
            "separation_aware_generator.pth",
        ),
    )

    print("Training completed.")
    print(
        "Latest checkpoint:",
        config.latest_checkpoint,
    )
    print(
        "Best checkpoint:",
        config.best_checkpoint,
    )
    print(
        "Next step: retrain DDPM using encoder(image, label)."
    )


if __name__ == "__main__":
    main()
