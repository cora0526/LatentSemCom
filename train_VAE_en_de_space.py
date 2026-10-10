import os
import csv
from typing import Tuple

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


def reparameterize(mu: torch.Tensor,logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick:

        z = mu + std * epsilon
        epsilon ~ N(0, I)
    """
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)

    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    return mu + eps * std


def kl_divergence(mu: torch.Tensor,logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z|x) || N(0, I))

    Return the average KL loss per batch and latent dimension.
    """
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)

    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())

    return kl.mean()


def append_csv(csv_path: str, row: dict) -> None:
    exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Shared VAE Encoder
# ============================================================

class SharedVAEEncoder(nn.Module):
    """
    Input:
        x: [B, 3, 64, 64]

    Output:
        mu:     [B, z_dim]
        logvar: [B, z_dim]
    """

    def __init__(self, z_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # [B, 3, 64, 64] -> [B, 64, 32, 32]
            nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # -> [B, 128, 16, 16]
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
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

        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        h = h.flatten(start_dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


# ============================================================
# Conditional VAE Decoder
# ============================================================

class ConditionalVAEDecoder(nn.Module):
    """
    Dataset labels:
        0 = EuroSAT
        1 = DeepGlobe

    Input:
        z:      [B, z_dim]
        labels: [B]

    Output:
        image: [B, 3, 64, 64], range approximately [-1, 1]
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
    beta_max: float = 1e-3,
    kl_warmup_epochs: int = 20,
    learning_rate: float = 1e-4,
    save_every: int = 5,
    device: str = "cuda",
):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable. Switching to CPU.")
        device = "cpu"

    save_root = "./save_data/SharedVAE_Satellite_en_de"
    model_root = "./save_model"

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    csv_path = os.path.join(
        save_root,
        "training_history.csv",
    )

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

    eurosat_root = (r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT")
    deepglobe_root = (r"D:/Python/SemCom/saved_data/deepglobe_64/train")

    eurosat_dataset = datasets.ImageFolder(eurosat_root,transform=transform)
    deepglobe_dataset = SatelliteDataset(deepglobe_root,transform=transform)

    # 最多各使用 50,000 張
    eurosat_dataset = Subset( eurosat_dataset,range(min(50000, len(eurosat_dataset))))
    deepglobe_dataset = Subset(deepglobe_dataset,range(min(50000, len(deepglobe_dataset))))

    eurosat_loader = DataLoader(eurosat_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=(device == "cuda"),drop_last=True)
    deepglobe_loader = DataLoader(deepglobe_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=(device == "cuda"),drop_last=True)

    num_steps = min(len(eurosat_loader),len(deepglobe_loader))

    print(f"EuroSAT images   : {len(eurosat_dataset)}")
    print(f"DeepGlobe images : {len(deepglobe_dataset)}")
    print(f"Steps per epoch  : {num_steps}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    encoder = SharedVAEEncoder(z_dim=z_dim,).to(device)
    decoder = ConditionalVAEDecoder(z_dim=z_dim,num_classes=2,label_dim=64).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=learning_rate,betas=(0.9, 0.999))

    # --------------------------------------------------------
    # Fixed images for visualization
    # --------------------------------------------------------

    fixed_e = get_img(next(iter(eurosat_loader)))[:64].to(device)
    fixed_g = get_img(next(iter(deepglobe_loader)))[:64].to(device)

    fixed_label_e = torch.zeros(fixed_e.size(0),dtype=torch.long,device=device)
    fixed_label_g = torch.ones(fixed_g.size(0),dtype=torch.long,device=device)

    save_image((fixed_e + 1.0) / 2.0,os.path.join(save_root, "real_EuroSAT.png"),nrow=8)
    save_image((fixed_g + 1.0) / 2.0,os.path.join(save_root, "real_DeepGlobe.png"),nrow=8)

    global_step = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        # KL warm-up
        beta = beta_max * min(1.0,float(epoch + 1) / float(kl_warmup_epochs))

        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0

        eurosat_recon_sum = 0.0
        deepglobe_recon_sum = 0.0

        mu_mean_sum = 0.0
        mu_std_sum = 0.0

        for batch_e, batch_g in zip(eurosat_loader,deepglobe_loader):
            real_e = get_img(batch_e).to(device,non_blocking=True)
            real_g = get_img(batch_g).to(device,non_blocking=True)

            current_batch_size = min(real_e.size(0),real_g.size(0))

            real_e = real_e[:current_batch_size]
            real_g = real_g[:current_batch_size]

            # ------------------------------------------------
            # Merge both datasets into one balanced batch
            # ------------------------------------------------
            real = torch.cat([real_e, real_g],dim=0)
            label_e = torch.zeros(current_batch_size,dtype=torch.long,device=device)
            label_g = torch.ones(current_batch_size,dtype=torch.long,device=device)
            labels = torch.cat([label_e, label_g],dim=0)
            # ------------------------------------------------
            # VAE forward
            # ------------------------------------------------

            mu, logvar = encoder(real)

            # 訓練標準 VAE 時，使用 sampling z
            z = reparameterize(mu, logvar)

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

            # 兩個資料集等權重
            loss_recon = 0.5 * (
                loss_recon_e
                + loss_recon_g
            )

            # ------------------------------------------------
            # KL divergence
            # ------------------------------------------------

            loss_kl = kl_divergence(
                mu,
                logvar,
            )

            # ------------------------------------------------
            # Total VAE loss
            # ------------------------------------------------

            loss = loss_recon + beta * loss_kl

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
            recon_loss_sum += loss_recon.item()
            kl_loss_sum += loss_kl.item()

            eurosat_recon_sum += loss_recon_e.item()
            deepglobe_recon_sum += loss_recon_g.item()

            mu_mean_sum += mu.detach().mean().item()
            mu_std_sum += mu.detach().std().item()

            if global_step % 200 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Step {global_step} | "
                    f"beta={beta:.6f} | "
                    f"Recon={loss_recon.item():.4f} | "
                    f"KL={loss_kl.item():.4f} | "
                    f"Total={loss.item():.4f} | "
                    f"mu_mean={mu.mean().item():.4f} | "
                    f"mu_std={mu.std().item():.4f}"
                )

            global_step += 1

        # ----------------------------------------------------
        # Epoch statistics
        # ----------------------------------------------------

        avg_total = total_loss_sum / num_steps
        avg_recon = recon_loss_sum / num_steps
        avg_kl = kl_loss_sum / num_steps

        avg_eurosat = eurosat_recon_sum / num_steps
        avg_deepglobe = deepglobe_recon_sum / num_steps

        avg_mu_mean = mu_mean_sum / num_steps
        avg_mu_std = mu_std_sum / num_steps

        print("=" * 100)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"beta={beta:.6f} | "
            f"Total={avg_total:.4f} | "
            f"Recon={avg_recon:.4f} | "
            f"KL={avg_kl:.4f}"
        )
        print(
            f"EuroSAT L1={avg_eurosat:.4f} | "
            f"DeepGlobe L1={avg_deepglobe:.4f} | "
            f"mu_mean={avg_mu_mean:.4f} | "
            f"mu_std={avg_mu_std:.4f}"
        )
        print("=" * 100)

        append_csv(
            csv_path,
            {
                "epoch": epoch + 1,
                "beta": beta,
                "total_loss": avg_total,
                "reconstruction_loss": avg_recon,
                "kl_loss": avg_kl,
                "eurosat_reconstruction": avg_eurosat,
                "deepglobe_reconstruction": avg_deepglobe,
                "mu_mean": avg_mu_mean,
                "mu_std": avg_mu_std,
            },
        )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        if (epoch + 1) % save_every == 0:
            encoder.eval()
            decoder.eval()

            with torch.no_grad():
                # EuroSAT
                mu_e, logvar_e = encoder(fixed_e)
                z_e = reparameterize(mu_e, logvar_e)

                recon_e_sample = decoder(
                    z_e,
                    fixed_label_e,
                )

                # 後續 DDPM 會採用這個 deterministic mu
                recon_e_mu = decoder(
                    mu_e,
                    fixed_label_e,
                )

                # DeepGlobe
                mu_g, logvar_g = encoder(fixed_g)
                z_g = reparameterize(mu_g, logvar_g)

                recon_g_sample = decoder(
                    z_g,
                    fixed_label_g,
                )

                recon_g_mu = decoder(
                    mu_g,
                    fixed_label_g,
                )

            # Sampling reconstruction
            save_image(
                (recon_e_sample + 1.0) / 2.0,
                os.path.join(
                    save_root,
                    f"recon_EuroSAT_sample_ep{epoch + 1}.png",
                ),
                nrow=8,
            )

            save_image(
                (recon_g_sample + 1.0) / 2.0,
                os.path.join(
                    save_root,
                    f"recon_DeepGlobe_sample_ep{epoch + 1}.png",
                ),
                nrow=8,
            )

            # Deterministic mu reconstruction
            save_image(
                (recon_e_mu + 1.0) / 2.0,
                os.path.join(
                    save_root,
                    f"recon_EuroSAT_mu_ep{epoch + 1}.png",
                ),
                nrow=8,
            )

            save_image(
                (recon_g_mu + 1.0) / 2.0,
                os.path.join(
                    save_root,
                    f"recon_DeepGlobe_mu_ep{epoch + 1}.png",
                ),
                nrow=8,
            )

            checkpoint = {
                "epoch": epoch + 1,
                "z_dim": z_dim,
                "beta_max": beta_max,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(
                checkpoint,
                os.path.join(
                    model_root,
                    "shared_vae_satellite_latest.pth",
                ),
            )

            torch.save(
                encoder.state_dict(),
                os.path.join(
                    model_root,
                    "shared_vae_encoder_satellite.pth",
                ),
            )

            torch.save(
                decoder.state_dict(),
                os.path.join(
                    model_root,
                    "conditional_vae_decoder_satellite.pth",
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
            "shared_vae_encoder_satellite.pth",
        ),
    )

    torch.save(
        decoder.state_dict(),
        os.path.join(
            model_root,
            "conditional_vae_decoder_satellite.pth",
        ),
    )

    print("VAE training completed.")


if __name__ == "__main__":
    main(epochs=100,batch_size=64,z_dim=256,beta_max=1e-3,kl_warmup_epochs=20,learning_rate=1e-4,save_every=5,device="cuda")