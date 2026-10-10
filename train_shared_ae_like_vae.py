import os
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from utils import SatelliteDataset


# ============================================================
# Basic utilities
# ============================================================

def get_img(batch):
    """
    ImageFolder returns (image, class_label).
    SatelliteDataset may return image directly or a tuple.
    """
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def append_csv(csv_path: str, row: dict) -> None:
    exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Shared deterministic AE Encoder
# ============================================================

class SharedAEEncoder(nn.Module):
    """
    Deterministic autoencoder encoder.

    Input:
        x: [B, 3, 64, 64]

    Output:
        z: [B, z_dim]

    This architecture is kept as close as possible to SharedVAEEncoder.
    The only structural difference is that the VAE's fc_mu and fc_logvar
    are replaced by one deterministic fc_z layer.
    """

    def __init__(self, z_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # [B, 3, 64, 64] -> [B, 64, 32, 32]
            nn.Conv2d(
                3,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # -> [B, 128, 16, 16]
            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # -> [B, 256, 8, 8]
            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # -> [B, 512, 4, 4]
            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        feature_dim = 512 * 4 * 4
        self.fc_z = nn.Linear(feature_dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = h.flatten(start_dim=1)
        z = self.fc_z(h)
        return z


# ============================================================
# Conditional AE Decoder
# ============================================================

class ConditionalAEDecoder(nn.Module):
    """
    Dataset labels:
        0 = EuroSAT
        1 = DeepGlobe

    Input:
        z:      [B, z_dim]
        labels: [B]

    Output:
        image: [B, 3, 64, 64], range approximately [-1, 1]

    This decoder is intentionally identical to ConditionalVAEDecoder,
    so the AE and VAE can be compared under the same decoder architecture.
    """

    def __init__(
        self,
        z_dim: int = 256,
        num_classes: int = 2,
        label_dim: int = 64,
    ):
        super().__init__()

        self.label_embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=label_dim,
        )

        self.fc = nn.Sequential(
            nn.Linear(z_dim + label_dim, 512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # [B, 512, 4, 4] -> [B, 256, 8, 8]
            nn.ConvTranspose2d(
                512,
                256,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # -> [B, 128, 16, 16]
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # -> [B, 64, 32, 32]
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # -> [B, 3, 64, 64]
            nn.ConvTranspose2d(
                64,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        label_feature = self.label_embedding(labels)

        conditional_z = torch.cat(
            [z, label_feature],
            dim=1,
        )

        h = self.fc(conditional_z)
        h = h.view(z.size(0), 512, 4, 4)

        return self.decoder(h)


# ============================================================
# Main training
# ============================================================

def main(
    epochs: int = 100,
    batch_size: int = 64,
    z_dim: int = 256,
    learning_rate: float = 1e-4,
    save_every: int = 5,
    device: str = "cuda",
):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable. Switching to CPU.")
        device = "cpu"

    save_root = "./save_data/SharedAE_Satellite_en_de"
    model_root = "./save_model"

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    csv_path = os.path.join(
        save_root,
        "training_history.csv",
    )

    # Remove an old history file to avoid mixing different runs.
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    eurosat_root = (
        r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    )
    deepglobe_root = (
        r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    )

    eurosat_dataset = datasets.ImageFolder(
        eurosat_root,
        transform=transform,
    )
    deepglobe_dataset = SatelliteDataset(
        deepglobe_root,
        transform=transform,
    )

    # Use at most 50,000 images from each dataset.
    eurosat_dataset = Subset(
        eurosat_dataset,
        range(min(50000, len(eurosat_dataset))),
    )
    deepglobe_dataset = Subset(
        deepglobe_dataset,
        range(min(50000, len(deepglobe_dataset))),
    )

    eurosat_loader = DataLoader(
        eurosat_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    deepglobe_loader = DataLoader(
        deepglobe_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    # This is identical to the VAE training logic:
    # every step contains one EuroSAT batch and one DeepGlobe batch.
    num_steps = min(
        len(eurosat_loader),
        len(deepglobe_loader),
    )

    if num_steps == 0:
        raise RuntimeError(
            "No training steps are available. "
            "Check dataset sizes and batch_size."
        )

    print(f"EuroSAT images   : {len(eurosat_dataset)}")
    print(f"DeepGlobe images : {len(deepglobe_dataset)}")
    print(f"Steps per epoch  : {num_steps}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    encoder = SharedAEEncoder(
        z_dim=z_dim,
    ).to(device)

    decoder = ConditionalAEDecoder(
        z_dim=z_dim,
        num_classes=2,
        label_dim=64,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(decoder.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
    )

    # --------------------------------------------------------
    # Fixed images for visualization
    # --------------------------------------------------------

    fixed_e = get_img(
        next(iter(eurosat_loader))
    )[:64].to(device)

    fixed_g = get_img(
        next(iter(deepglobe_loader))
    )[:64].to(device)

    fixed_label_e = torch.zeros(
        fixed_e.size(0),
        dtype=torch.long,
        device=device,
    )
    fixed_label_g = torch.ones(
        fixed_g.size(0),
        dtype=torch.long,
        device=device,
    )

    save_image(
        (fixed_e + 1.0) / 2.0,
        os.path.join(
            save_root,
            "real_EuroSAT.png",
        ),
        nrow=8,
    )
    save_image(
        (fixed_g + 1.0) / 2.0,
        os.path.join(
            save_root,
            "real_DeepGlobe.png",
        ),
        nrow=8,
    )

    global_step = 0
    best_reconstruction = float("inf")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        total_loss_sum = 0.0
        eurosat_recon_sum = 0.0
        deepglobe_recon_sum = 0.0

        z_mean_sum = 0.0
        z_std_sum = 0.0
        z_power_sum = 0.0

        for batch_e, batch_g in zip(
            eurosat_loader,
            deepglobe_loader,
        ):
            real_e = get_img(batch_e).to(
                device,
                non_blocking=True,
            )
            real_g = get_img(batch_g).to(
                device,
                non_blocking=True,
            )

            current_batch_size = min(
                real_e.size(0),
                real_g.size(0),
            )

            real_e = real_e[:current_batch_size]
            real_g = real_g[:current_batch_size]

            # ------------------------------------------------
            # Merge both datasets into one balanced batch
            # ------------------------------------------------

            real = torch.cat(
                [real_e, real_g],
                dim=0,
            )

            label_e = torch.zeros(
                current_batch_size,
                dtype=torch.long,
                device=device,
            )
            label_g = torch.ones(
                current_batch_size,
                dtype=torch.long,
                device=device,
            )
            labels = torch.cat(
                [label_e, label_g],
                dim=0,
            )

            # ------------------------------------------------
            # Deterministic AE forward
            # ------------------------------------------------

            z = encoder(real)
            recon = decoder(z, labels)

            # ------------------------------------------------
            # Reconstruction loss
            # ------------------------------------------------

            recon_e = recon[:current_batch_size]
            recon_g = recon[current_batch_size:]

            loss_recon_e = F.l1_loss(
                recon_e,
                real_e,
            )
            loss_recon_g = F.l1_loss(
                recon_g,
                real_g,
            )

            # Equal weighting for the two datasets.
            loss_recon = 0.5 * (
                loss_recon_e
                + loss_recon_g
            )

            # A deterministic AE has no KL loss.
            loss = loss_recon

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                + list(decoder.parameters()),
                max_norm=5.0,
            )

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            total_loss_sum += loss.item()
            eurosat_recon_sum += loss_recon_e.item()
            deepglobe_recon_sum += loss_recon_g.item()

            z_mean_sum += z.detach().mean().item()
            z_std_sum += z.detach().std().item()
            z_power_sum += z.detach().pow(2).mean().item()

            if global_step % 200 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Step {global_step} | "
                    f"Recon={loss_recon.item():.4f} | "
                    f"EuroSAT={loss_recon_e.item():.4f} | "
                    f"DeepGlobe={loss_recon_g.item():.4f} | "
                    f"z_mean={z.mean().item():.4f} | "
                    f"z_std={z.std().item():.4f} | "
                    f"z_power={z.pow(2).mean().item():.4f}"
                )

            global_step += 1

        # ----------------------------------------------------
        # Epoch statistics
        # ----------------------------------------------------

        avg_total = total_loss_sum / num_steps
        avg_eurosat = eurosat_recon_sum / num_steps
        avg_deepglobe = deepglobe_recon_sum / num_steps

        avg_z_mean = z_mean_sum / num_steps
        avg_z_std = z_std_sum / num_steps
        avg_z_power = z_power_sum / num_steps

        print("=" * 100)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Total={avg_total:.4f}"
        )
        print(
            f"EuroSAT L1={avg_eurosat:.4f} | "
            f"DeepGlobe L1={avg_deepglobe:.4f} | "
            f"z_mean={avg_z_mean:.4f} | "
            f"z_std={avg_z_std:.4f} | "
            f"z_power={avg_z_power:.4f}"
        )
        print("=" * 100)

        append_csv(
            csv_path,
            {
                "epoch": epoch + 1,
                "total_loss": avg_total,
                "reconstruction_loss": avg_total,
                "eurosat_reconstruction": avg_eurosat,
                "deepglobe_reconstruction": avg_deepglobe,
                "z_mean": avg_z_mean,
                "z_std": avg_z_std,
                "z_power": avg_z_power,
            },
        )

        # ----------------------------------------------------
        # Save latest and best checkpoints every epoch
        # ----------------------------------------------------

        checkpoint = {
            "epoch": epoch + 1,
            "z_dim": z_dim,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_loss": avg_total,
        }

        torch.save(
            checkpoint,
            os.path.join(
                model_root,
                "shared_ae_satellite_latest.pth",
            ),
        )

        if avg_total < best_reconstruction:
            best_reconstruction = avg_total

            torch.save(
                checkpoint,
                os.path.join(
                    model_root,
                    "shared_ae_satellite_best.pth",
                ),
            )

            print(
                f"Saved best AE checkpoint | "
                f"reconstruction={best_reconstruction:.6f}"
            )

        # ----------------------------------------------------
        # Save visualizations and epoch checkpoint
        # ----------------------------------------------------

        if (epoch + 1) % save_every == 0:
            encoder.eval()
            decoder.eval()

            with torch.no_grad():
                z_e = encoder(fixed_e)
                recon_e = decoder(
                    z_e,
                    fixed_label_e,
                )

                z_g = encoder(fixed_g)
                recon_g = decoder(
                    z_g,
                    fixed_label_g,
                )

            save_image(
                (recon_e + 1.0) / 2.0,
                os.path.join(
                    save_root,
                    f"recon_EuroSAT_ep{epoch + 1}.png",
                ),
                nrow=8,
            )

            save_image(
                (recon_g + 1.0) / 2.0,
                os.path.join(
                    save_root,
                    f"recon_DeepGlobe_ep{epoch + 1}.png",
                ),
                nrow=8,
            )

            torch.save(
                checkpoint,
                os.path.join(
                    model_root,
                    f"shared_ae_satellite_ep{epoch + 1}.pth",
                ),
            )

            torch.save(
                encoder.state_dict(),
                os.path.join(
                    model_root,
                    "shared_ae_encoder_satellite.pth",
                ),
            )

            torch.save(
                decoder.state_dict(),
                os.path.join(
                    model_root,
                    "conditional_ae_decoder_satellite.pth",
                ),
            )

            print(
                f"Saved reconstruction images and models "
                f"at epoch {epoch + 1}."
            )

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    torch.save(
        encoder.state_dict(),
        os.path.join(
            model_root,
            "shared_ae_encoder_satellite.pth",
        ),
    )

    torch.save(
        decoder.state_dict(),
        os.path.join(
            model_root,
            "conditional_ae_decoder_satellite.pth",
        ),
    )

    print("AE training completed.")
    print("For DDPM/channel experiments, use z = encoder(image).")


if __name__ == "__main__":
    main(
        epochs=100,
        batch_size=64,
        z_dim=256,
        learning_rate=1e-4,
        save_every=5,
        device="cuda",
    )
