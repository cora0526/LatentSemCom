import csv
import glob
import math
import os
import random
from itertools import cycle
from typing import Tuple

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
# Frozen VAE encoder/decoder
# These definitions must match shared_vae_satellite_latest.pth
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
        h = self.fc(torch.cat([z, label_feature], dim=1))
        h = h.view(z.size(0), 512, 4, 4)
        return self.decoder(h)


# ============================================================
# Deterministic joint MLP separator
# Input : received latent mixture z_rx
# Output: predicted EuroSAT latent and DeepGlobe latent
# ============================================================
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


class JointLatentMLPSeparator(nn.Module):
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

    def forward(self, z_rx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_layer(z_rx)
        h = self.blocks(h)
        pair = self.output_layer(h)
        return pair[:, : self.z_dim], pair[:, self.z_dim :]


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
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = signal_power / snr_linear
    return signal + torch.randn_like(signal) * torch.sqrt(noise_power)


def latent_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "mse": F.mse_loss(pred, target).item(),
        "cosine": F.cosine_similarity(pred, target, dim=1).mean().item(),
    }


@torch.no_grad()
def validate_mlp(
    model: nn.Module,
    encoder: nn.Module,
    decoder: nn.Module,
    fixed_e: torch.Tensor,
    fixed_d: torch.Tensor,
    epoch: int,
    save_root: str,
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

    pred_e, pred_d = model(z_rx)

    recon_e = decoder(pred_e, label_e)
    recon_d = decoder(pred_d, label_d)
    oracle_e = decoder(z_e, label_e)
    oracle_d = decoder(z_d, label_d)

    metrics_e = latent_metrics(pred_e, z_e)
    metrics_d = latent_metrics(pred_d, z_d)

    metrics = {
        "latent_mse_e": metrics_e["mse"],
        "latent_mse_d": metrics_d["mse"],
        "latent_cos_e": metrics_e["cosine"],
        "latent_cos_d": metrics_d["cosine"],
        "image_l1_e_to_real": F.l1_loss(recon_e, fixed_e).item(),
        "image_l1_d_to_real": F.l1_loss(recon_d, fixed_d).item(),
        "image_l1_e_to_oracle": F.l1_loss(recon_e, oracle_e).item(),
        "image_l1_d_to_oracle": F.l1_loss(recon_d, oracle_d).item(),
        "consistency_mse_clean": F.mse_loss(pred_e + pred_d, z_mix_clean).item(),
        "consistency_mse_received": F.mse_loss(pred_e + pred_d, z_rx).item(),
    }

    count = fixed_e.size(0)
    nrow = max(1, int(math.sqrt(count)))
    save_image(
        (fixed_e + 1.0) / 2.0,
        os.path.join(save_root, f"val_real_E_ep{epoch}.png"),
        nrow=nrow,
    )
    save_image(
        (fixed_d + 1.0) / 2.0,
        os.path.join(save_root, f"val_real_D_ep{epoch}.png"),
        nrow=nrow,
    )
    save_image(
        (oracle_e + 1.0) / 2.0,
        os.path.join(save_root, f"val_oracle_E_ep{epoch}.png"),
        nrow=nrow,
    )
    save_image(
        (oracle_d + 1.0) / 2.0,
        os.path.join(save_root, f"val_oracle_D_ep{epoch}.png"),
        nrow=nrow,
    )
    save_image(
        (recon_e + 1.0) / 2.0,
        os.path.join(save_root, f"val_sep_E_ep{epoch}.png"),
        nrow=nrow,
    )
    save_image(
        (recon_d + 1.0) / 2.0,
        os.path.join(save_root, f"val_sep_D_ep{epoch}.png"),
        nrow=nrow,
    )

    return metrics


def load_pretrained_vae(
    checkpoint_path: str,
    device: str,
    z_dim: int,
    label_dim: int,
    num_classes: int,
):
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
# Main training: VAE -> Joint MLP -> VAE
# ============================================================
def main():
    # Core settings
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

    hidden_dim = 768
    num_blocks = 6
    dropout = 0.0

    # Loss weights
    lambda_consistency = 0.2
    lambda_image = 0.05

    # Channel curriculum
    clean_epochs = 20
    min_train_snr_db = 10.0
    max_train_snr_db = 30.0
    validation_snr_db = 20.0

    validate_every = 5
    fixed_count = 16

    # Paths
    data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    data_root_d = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    vae_checkpoint_path = "./save_model/shared_vae_satellite_latest.pth"

    save_root = "./save_data/MLP_VAE_latent_separator"
    model_root = "./save_model"
    latest_model_path = os.path.join(
        model_root, "MLP_vae_latent_separator_latest.pth"
    )
    best_model_path = os.path.join(
        model_root, "MLP_vae_latent_separator_best.pth"
    )
    history_path = os.path.join(save_root, "training_history.csv")
    validation_history_path = os.path.join(save_root, "validation_history.csv")

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

    encoder, decoder = load_pretrained_vae(
        checkpoint_path=vae_checkpoint_path,
        device=device,
        z_dim=z_dim,
        label_dim=label_dim,
        num_classes=num_classes,
    )

    mlp = JointLatentMLPSeparator(
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        mlp.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    print(f"Device             = {device}")
    print(f"EuroSAT images     = {len(dataset_e)}")
    print(f"DeepGlobe images   = {len(dataset_d)}")
    print(f"Steps per epoch    = {steps_per_epoch}")
    print("System             = VAE -> Joint MLP -> VAE")

    iter_e = cycle(loader_e)
    iter_d = cycle(loader_d)
    global_step = 0
    best_validation_mse = float("inf")

    for epoch in range(1, epochs + 1):
        mlp.train()
        epoch_total = 0.0
        epoch_latent = 0.0
        epoch_consistency = 0.0
        epoch_image = 0.0

        for step in range(steps_per_epoch):
            image_e, _ = next(iter_e)
            image_d, _ = next(iter_d)
            image_e = image_e.to(device, non_blocking=True)
            image_d = image_d.to(device, non_blocking=True)

            with torch.no_grad():
                # Deterministic transmitted representation: use encoder mu.
                z_e, _ = encoder(image_e)
                z_d, _ = encoder(image_d)
                z_mix_clean = z_e + z_d

                if epoch <= clean_epochs:
                    train_snr_db = math.inf
                    z_rx = z_mix_clean
                else:
                    train_snr_db = random.uniform(
                        min_train_snr_db,
                        max_train_snr_db,
                    )
                    z_rx = add_awgn_by_snr(z_mix_clean, train_snr_db)

                label_e = torch.zeros(
                    z_e.size(0), dtype=torch.long, device=device
                )
                label_d = torch.ones(
                    z_d.size(0), dtype=torch.long, device=device
                )

                # Oracle VAE reconstructions are the image-space targets.
                oracle_e = decoder(z_e, label_e)
                oracle_d = decoder(z_d, label_d)

            pred_e, pred_d = mlp(z_rx)

            loss_latent = 0.5 * (
                F.mse_loss(pred_e, z_e) + F.mse_loss(pred_d, z_d)
            )

            # Enforce that the two predicted clean sources re-sum to the clean mixture.
            loss_consistency = F.mse_loss(pred_e + pred_d, z_mix_clean)

            recon_e = decoder(pred_e, label_e)
            recon_d = decoder(pred_d, label_d)
            loss_image = 0.5 * (
                F.l1_loss(recon_e, oracle_e)
                + F.l1_loss(recon_d, oracle_d)
            )

            loss = (
                loss_latent
                + lambda_consistency * loss_consistency
                + lambda_image * loss_image
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), grad_clip)
            optimizer.step()

            epoch_total += loss.item()
            epoch_latent += loss_latent.item()
            epoch_consistency += loss_consistency.item()
            epoch_image += loss_image.item()

            if global_step % 200 == 0:
                snr_text = (
                    "clean"
                    if math.isinf(train_snr_db)
                    else f"{train_snr_db:.1f} dB"
                )
                print(
                    f"Epoch {epoch}/{epochs} | Step {step}/{steps_per_epoch} | "
                    f"SNR={snr_text} | Total={loss.item():.4f} | "
                    f"Lat={loss_latent.item():.4f} | "
                    f"Cons={loss_consistency.item():.4f} | "
                    f"Img={loss_image.item():.4f}"
                )
            global_step += 1

        averages = {
            "epoch": epoch,
            "total_loss": epoch_total / steps_per_epoch,
            "latent_loss": epoch_latent / steps_per_epoch,
            "consistency_loss": epoch_consistency / steps_per_epoch,
            "image_loss": epoch_image / steps_per_epoch,
        }
        append_csv(history_path, averages)

        print("=" * 100)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Total={averages['total_loss']:.4f} | "
            f"Lat={averages['latent_loss']:.4f} | "
            f"Cons={averages['consistency_loss']:.4f} | "
            f"Img={averages['image_loss']:.4f}"
        )
        print("=" * 100)

        checkpoint = {
            "epoch": epoch,
            "model": mlp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "z_dim": z_dim,
                "hidden_dim": hidden_dim,
                "num_blocks": num_blocks,
                "dropout": dropout,
                "lambda_consistency": lambda_consistency,
                "lambda_image": lambda_image,
                "vae_checkpoint_path": vae_checkpoint_path,
                "domain_labels": {"EuroSAT": 0, "DeepGlobe": 1},
            },
        }
        torch.save(checkpoint, latest_model_path)

        if epoch % validate_every == 0 or epoch == 1 or epoch == epochs:
            validation_metrics = validate_mlp(
                model=mlp,
                encoder=encoder,
                decoder=decoder,
                fixed_e=fixed_e,
                fixed_d=fixed_d,
                epoch=epoch,
                save_root=save_root,
                validation_snr_db=validation_snr_db,
            )
            append_csv(
                validation_history_path,
                {"epoch": epoch, **validation_metrics},
            )
            checkpoint["validation"] = validation_metrics

            epoch_path = os.path.join(
                model_root,
                f"MLP_vae_latent_separator_ep{epoch}.pth",
            )
            torch.save(checkpoint, epoch_path)
            print("Validation:", validation_metrics)

            avg_latent_mse = 0.5 * (
                validation_metrics["latent_mse_e"]
                + validation_metrics["latent_mse_d"]
            )
            if avg_latent_mse < best_validation_mse:
                best_validation_mse = avg_latent_mse
                torch.save(checkpoint, best_model_path)
                print(
                    f"Saved best MLP | avg latent MSE={best_validation_mse:.6f}"
                )

    print("Training completed.")
    print(f"Latest MLP checkpoint: {latest_model_path}")
    print(f"Best MLP checkpoint:   {best_model_path}")
    print(f"Validation images:     {save_root}")


if __name__ == "__main__":
    main()
