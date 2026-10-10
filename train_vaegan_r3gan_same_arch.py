"""
Same-architecture VAE-GAN experiment using the R3GAN adversarial objective.

Architecture, datasets, reconstruction/feature/KL terms, and latent dimensions
are retained from the supplied WGAN-GP script. The adversarial component is
changed to paired relativistic logistic loss with zero-centered R1+R2 penalties.
"""


import csv
import glob
import os
import random
from dataclasses import dataclass, asdict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    save_root: str = "./save_data/VAE_GAN_R3GAN_same_arch"
    model_root: str = "./save_model"

    epochs: int = 150
    batch_size: int = 128
    num_workers: int = 4
    max_images_per_domain: int = 50000
    validation_ratio: float = 0.05

    z_dim: int = 256
    label_dim: int = 64
    num_classes: int = 2
    critic_feature_dim: int = 512

    lr_encoder: float = 1e-4
    lr_generator: float = 1e-4
    lr_critic: float = 5e-5
    n_critic: int = 1  # R3GAN uses one critic update per generator update
    grad_clip: float = 5.0

    beta_max: float = 1e-4
    kl_warmup_epochs: int = 30

    lambda_pixel: float = 1.0
    lambda_feature: float = 0.1
    lambda_recon_adv: float = 1e-3
    lambda_random_adv: float = 1e-3
    gamma_max: float = 0.2
    gamma_warmup_epochs: int = 30

    save_every: int = 5


class SatelliteDataset(Dataset):
    def __init__(self, root, transform=None, domain_label=1):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform
        self.domain_label = int(domain_label)
        if not self.files:
            raise RuntimeError(f"No DeepGlobe images found in:\n{root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = Image.open(self.files[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.domain_label


class DomainLabelDataset(Dataset):
    def __init__(self, dataset, domain_label):
        self.dataset = dataset
        self.domain_label = int(domain_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item[0] if isinstance(item, (tuple, list)) else item
        return image, self.domain_label


class SharedVAEEncoder(nn.Module):
    def __init__(self, z_dim=256):
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
        self.fc_mu = nn.Linear(512 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, z_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 10.0)
        return mu, logvar


class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim=256, num_classes=2, label_dim=64):
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

    def forward(self, z, labels):
        y = self.label_embedding(labels)
        h = self.fc(torch.cat([z, y], dim=1)).view(z.size(0), 512, 4, 4)
        return self.decoder(h)


class ConditionalCritic(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, feature_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.score_head = nn.Linear(feature_dim, 1)
        self.label_embedding = nn.Embedding(num_classes, feature_dim)

    def forward(self, x, labels, return_features=False):
        feature_map = self.features(x)
        feature = feature_map.sum(dim=(2, 3))
        score = self.score_head(feature).squeeze(1)
        score = score + (feature * self.label_embedding(labels)).sum(dim=1)
        if return_features:
            return score, feature
        return score


def reparameterize(mu, logvar):
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


def kl_divergence(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()


def set_requires_grad(model, value):
    for parameter in model.parameters():
        parameter.requires_grad = value


def zero_centered_gradient_penalty(samples, scores):
    """R1/R2 zero-centered gradient penalty used by R3GAN."""
    gradients = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=samples,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return gradients.square().flatten(1).sum(dim=1)


def r3gan_discriminator_loss(critic, real, fake, labels, gamma):
    """
    Regularized relativistic discriminator loss from R3GAN:

        softplus(-(D(real) - D(fake)))
        + gamma/2 * (R1(real) + R2(fake))

    Real and fake must have the same batch size and aligned labels.
    """
    real = real.detach().requires_grad_(True)
    fake = fake.detach().requires_grad_(True)

    real_scores = critic(real, labels)
    fake_scores = critic(fake, labels)
    relativistic_logits = real_scores - fake_scores
    adversarial = F.softplus(-relativistic_logits)

    r1 = zero_centered_gradient_penalty(real, real_scores)
    r2 = zero_centered_gradient_penalty(fake, fake_scores)
    loss = (adversarial + 0.5 * gamma * (r1 + r2)).mean()
    return loss, adversarial.mean(), r1.mean(), r2.mean()


def r3gan_generator_loss(fake_scores, real_scores):
    """Relativistic logistic generator loss: softplus(-(D(fake)-D(real)))."""
    return F.softplus(-(fake_scores - real_scores)).mean()


def image_psnr(pred, target):
    pred = ((pred + 1) / 2).clamp(0, 1)
    target = ((target + 1) / 2).clamp(0, 1)
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(1)
    return (-10 * torch.log10(mse.clamp_min(1e-12))).mean().item()


def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def validate(encoder, generator, loader, device, max_batches=20):
    encoder.eval()
    generator.eval()
    sums = {"l1_mu": 0.0, "l1_sample": 0.0, "psnr_mu": 0.0,
            "kl": 0.0, "mu_mean": 0.0, "mu_std": 0.0}
    count = 0

    for batch_index, (real, labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        real = real.to(device)
        labels = labels.to(device, dtype=torch.long)
        mu, logvar = encoder(real)
        sampled_z = reparameterize(mu, logvar)
        recon_mu = generator(mu, labels)
        recon_sample = generator(sampled_z, labels)

        sums["l1_mu"] += F.l1_loss(recon_mu, real).item()
        sums["l1_sample"] += F.l1_loss(recon_sample, real).item()
        sums["psnr_mu"] += image_psnr(recon_mu, real)
        sums["kl"] += kl_divergence(mu, logvar).item()
        sums["mu_mean"] += mu.mean().item()
        sums["mu_std"] += mu.std().item()
        count += 1

    return {f"val_{k}": v / max(1, count) for k, v in sums.items()}


@torch.no_grad()
def save_examples(encoder, generator, fixed_e, fixed_d, label_e, label_d,
                  random_z, save_root, epoch):
    encoder.eval()
    generator.eval()

    mu_e, logvar_e = encoder(fixed_e)
    mu_d, logvar_d = encoder(fixed_d)

    outputs = {
        f"EuroSAT_mu_ep{epoch}.png": generator(mu_e, label_e),
        f"DeepGlobe_mu_ep{epoch}.png": generator(mu_d, label_d),
        f"EuroSAT_sample_ep{epoch}.png":
            generator(reparameterize(mu_e, logvar_e), label_e),
        f"DeepGlobe_sample_ep{epoch}.png":
            generator(reparameterize(mu_d, logvar_d), label_d),
        f"random_EuroSAT_ep{epoch}.png": generator(random_z, label_e),
        f"random_DeepGlobe_ep{epoch}.png": generator(random_z, label_d),
    }
    for filename, images in outputs.items():
        save_image((images + 1) / 2, os.path.join(save_root, filename), nrow=8)


def save_checkpoint(path, epoch, encoder, generator, critic,
                    opt_e, opt_g, opt_d, config, validation):
    torch.save(
        {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_encoder": opt_e.state_dict(),
            "optimizer_generator": opt_g.state_dict(),
            "optimizer_critic": opt_d.state_dict(),
            "config": asdict(config),
            "validation": validation,
        },
        path,
    )


def main():
    config = Config()
    device = config.device if torch.cuda.is_available() else "cpu"

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(config.save_root, exist_ok=True)
    os.makedirs(config.model_root, exist_ok=True)
    history_path = os.path.join(config.save_root, "training_history.csv")
    if os.path.exists(history_path):
        os.remove(history_path)

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    euro_base = datasets.ImageFolder(config.eurosat_root, transform=transform)
    euro_base = Subset(
        euro_base,
        range(min(config.max_images_per_domain, len(euro_base))),
    )
    eurosat = DomainLabelDataset(euro_base, 0)

    deepglobe = SatelliteDataset(config.deepglobe_root, transform, 1)
    deepglobe = Subset(
        deepglobe,
        range(min(config.max_images_per_domain, len(deepglobe))),
    )

    full_dataset = ConcatDataset([eurosat, deepglobe])
    val_size = max(1, int(len(full_dataset) * config.validation_ratio))
    train_size = len(full_dataset) - val_size
    split_gen = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size], generator=split_gen
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(config.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(config.num_workers > 0),
    )

    fixed_count = min(64, len(eurosat), len(deepglobe))
    fixed_e = torch.stack([eurosat[i][0] for i in range(fixed_count)]).to(device)
    fixed_d = torch.stack([deepglobe[i][0] for i in range(fixed_count)]).to(device)
    label_e = torch.zeros(fixed_count, dtype=torch.long, device=device)
    label_d = torch.ones(fixed_count, dtype=torch.long, device=device)
    random_z = torch.randn(fixed_count, config.z_dim, device=device)

    save_image((fixed_e + 1) / 2, os.path.join(config.save_root, "real_EuroSAT.png"), nrow=8)
    save_image((fixed_d + 1) / 2, os.path.join(config.save_root, "real_DeepGlobe.png"), nrow=8)

    encoder = SharedVAEEncoder(config.z_dim).to(device)
    generator = ConditionalGenerator(
        config.z_dim, config.num_classes, config.label_dim
    ).to(device)
    critic = ConditionalCritic(
        config.num_classes, config.critic_feature_dim
    ).to(device)

    opt_e = torch.optim.Adam(encoder.parameters(), lr=config.lr_encoder, betas=(0.9, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr_generator, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(critic.parameters(), lr=config.lr_critic, betas=(0.0, 0.9))

    print(f"Device={device}")
    print(f"EuroSAT={len(eurosat)} | DeepGlobe={len(deepglobe)}")
    print("Encoder, Generator and Critic are initialized from scratch.")

    best_l1 = float("inf")
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        encoder.train()
        generator.train()
        critic.train()

        beta = config.beta_max * min(
            1.0, epoch / max(1, config.kl_warmup_epochs)
        )
        gamma = config.gamma_max * min(
            1.0, epoch / max(1, config.gamma_warmup_epochs)
        )
        sums = {
            "encoder_total": 0.0, "generator_total": 0.0,
            "critic_total": 0.0, "pixel": 0.0, "feature": 0.0,
            "kl": 0.0, "recon_adv": 0.0, "random_adv": 0.0,
            "r3_adv_d": 0.0, "r1": 0.0, "r2": 0.0,
            "mu_mean": 0.0, "mu_std": 0.0,
        }
        batches = 0

        for real, labels in train_loader:
            real = real.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            batch_now = real.size(0)

            # Critic: R3GAN regularized relativistic objective
            set_requires_grad(critic, True)
            set_requires_grad(encoder, False)
            set_requires_grad(generator, False)
            d_accum = 0.0
            d_adv_accum = 0.0
            r1_accum = 0.0
            r2_accum = 0.0

            for _ in range(config.n_critic):
                with torch.no_grad():
                    mu_d, logvar_d = encoder(real)
                    recon_d = generator(reparameterize(mu_d, logvar_d), labels)
                    random_fake_d = generator(
                        torch.randn(batch_now, config.z_dim, device=device), labels
                    )

                # Pair each fake sample with a real sample, as required by
                # the relativistic objective. Real samples are duplicated so
                # reconstructed and random-generated fakes are both trained.
                fake_all = torch.cat([recon_d, random_fake_d], dim=0)
                real_all = torch.cat([real, real], dim=0)
                pair_labels = torch.cat([labels, labels], dim=0)

                loss_d, adv_d, r1, r2 = r3gan_discriminator_loss(
                    critic, real_all, fake_all, pair_labels, gamma
                )

                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip)
                opt_d.step()

                d_accum += loss_d.item()
                d_adv_accum += adv_d.item()
                r1_accum += r1.item()
                r2_accum += r2.item()

            d_value = d_accum / config.n_critic
            d_adv_value = d_adv_accum / config.n_critic
            r1_value = r1_accum / config.n_critic
            r2_value = r2_accum / config.n_critic

            # Encoder
            set_requires_grad(critic, False)
            set_requires_grad(encoder, True)
            set_requires_grad(generator, False)

            mu_e, logvar_e = encoder(real)
            recon_e = generator(reparameterize(mu_e, logvar_e), labels)
            pixel_e = F.l1_loss(recon_e, real)

            with torch.no_grad():
                _, real_feat_e = critic(real, labels, return_features=True)
            _, recon_feat_e = critic(recon_e, labels, return_features=True)

            feature_e = F.mse_loss(recon_feat_e, real_feat_e)
            kl_e = kl_divergence(mu_e, logvar_e)
            loss_e = (
                config.lambda_pixel * pixel_e
                + config.lambda_feature * feature_e
                + beta * kl_e
            )

            opt_e.zero_grad(set_to_none=True)
            loss_e.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.grad_clip)
            opt_e.step()

            # Generator
            set_requires_grad(critic, False)
            set_requires_grad(encoder, False)
            set_requires_grad(generator, True)

            with torch.no_grad():
                mu_g, logvar_g = encoder(real)
                z_g = reparameterize(mu_g, logvar_g)
                _, real_feat_g = critic(real, labels, return_features=True)

            recon_g = generator(z_g, labels)
            score_g, recon_feat_g = critic(recon_g, labels, return_features=True)
            with torch.no_grad():
                real_score_g = critic(real, labels)

            pixel_g = F.l1_loss(recon_g, real)
            feature_g = F.mse_loss(recon_feat_g, real_feat_g)
            recon_adv = r3gan_generator_loss(score_g, real_score_g)

            random_fake_g = generator(
                torch.randn(batch_now, config.z_dim, device=device), labels
            )
            random_score_g = critic(random_fake_g, labels)
            random_adv = r3gan_generator_loss(random_score_g, real_score_g)

            loss_g = (
                config.lambda_pixel * pixel_g
                + config.lambda_feature * feature_g
                + config.lambda_recon_adv * recon_adv
                + config.lambda_random_adv * random_adv
            )

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip)
            opt_g.step()

            sums["encoder_total"] += loss_e.item()
            sums["generator_total"] += loss_g.item()
            sums["critic_total"] += d_value
            sums["pixel"] += pixel_g.item()
            sums["feature"] += feature_g.item()
            sums["kl"] += kl_e.item()
            sums["recon_adv"] += recon_adv.item()
            sums["random_adv"] += random_adv.item()
            sums["r3_adv_d"] += d_adv_value
            sums["r1"] += r1_value
            sums["r2"] += r2_value
            sums["mu_mean"] += mu_e.detach().mean().item()
            sums["mu_std"] += mu_e.detach().std().item()
            batches += 1

            if global_step % 200 == 0:
                print(
                    f"Epoch {epoch}/{config.epochs} | Step={global_step} | "
                    f"E={loss_e.item():.4f} | G={loss_g.item():.4f} | "
                    f"D={d_value:.4f} | Pixel={pixel_g.item():.4f} | "
                    f"Feature={feature_g.item():.4f} | KL={kl_e.item():.4f} | "
                    f"beta={beta:.6f} | gamma={gamma:.4f} | "
                    f"R1={r1_value:.4f} | R2={r2_value:.4f} | "
                    f"mu_std={mu_e.std().item():.4f}"
                )
            global_step += 1

        averages = {key: value / max(1, batches) for key, value in sums.items()}
        validation = validate(encoder, generator, val_loader, device)
        row = {"epoch": epoch, "beta": beta, "gamma": gamma, **averages, **validation}
        append_csv(history_path, row)

        print("=" * 110)
        print(f"Epoch {epoch}/{config.epochs} | {averages}")
        print("Validation:", validation)
        print("=" * 110)

        latest = os.path.join(config.model_root, "vae_gan_r3gan_same_arch_latest.pth")
        save_checkpoint(
            latest, epoch, encoder, generator, critic,
            opt_e, opt_g, opt_d, config, validation
        )

        if epoch % config.save_every == 0 or epoch == 1 or epoch == config.epochs:
            save_examples(
                encoder, generator, fixed_e, fixed_d, label_e, label_d,
                random_z, config.save_root, epoch
            )
            save_checkpoint(
                os.path.join(
                    config.model_root, f"vae_gan_r3gan_same_arch_ep{epoch}.pth"
                ),
                epoch, encoder, generator, critic,
                opt_e, opt_g, opt_d, config, validation
            )

        if validation["val_l1_mu"] < best_l1:
            best_l1 = validation["val_l1_mu"]
            save_checkpoint(
                os.path.join(config.model_root, "vae_gan_r3gan_same_arch_best.pth"),
                epoch, encoder, generator, critic,
                opt_e, opt_g, opt_d, config, validation
            )
            print(f"Saved best model | val_l1_mu={best_l1:.6f}")

    torch.save(
        encoder.state_dict(),
        os.path.join(config.model_root, "vae_gan_r3gan_same_arch_encoder.pth"),
    )
    torch.save(
        generator.state_dict(),
        os.path.join(config.model_root, "vae_gan_r3gan_same_arch_generator.pth"),
    )
    torch.save(
        critic.state_dict(),
        os.path.join(config.model_root, "vae_gan_r3gan_same_arch_critic.pth"),
    )

    print("Training completed.")
    print("For DDPM training, use encoder output mu, not sampled z.")


if __name__ == "__main__":
    main()
