import os
import csv
import glob
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

# 為了與你已訓練好的權重完全相容，
# Generator / Critic 的類別仍使用原本 model.py 中的定義。
from model import Generator_Satellite_Conditional, Critic_Satellite_Conditional


# ============================================================
# DeepGlobe dataset
# domain label:
#   1 = DeepGlobe
# ============================================================
class SatelliteDataset(Dataset):
    def __init__(self,root: str,transform=None,domain_label: int = 1):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform
        self.domain_label = int(domain_label)

        if len(self.files) == 0:raise RuntimeError(f"No matching DeepGlobe images found in:\n{root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, self.domain_label


# ============================================================
# Override original ImageFolder label with domain label
# EuroSAT:
#   original ImageFolder label = land-cover class
#   target domain label = 0
# ============================================================
class DomainLabelDataset(Dataset):
    def __init__(self, dataset: Dataset, domain_label: int):
        self.dataset = dataset
        self.domain_label = int(domain_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if isinstance(item, (tuple, list)):
            img = item[0]
        else:
            img = item

        return img, self.domain_label


# ============================================================
# Shared VAE Encoder
# EuroSAT and DeepGlobe use exactly the same encoder weights.
# ============================================================
class SharedVAEEncoder(nn.Module):
    """
    Input:
        image: [B, 3, 64, 64]

    Output:
        mu:     [B, z_dim]
        logvar: [B, z_dim]
    """

    def __init__(self, z_dim: int = 256):
        super().__init__()

        self.features = nn.Sequential(
            # 64 x 64 -> 32 x 32
            nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 x 32 -> 16 x 16
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 x 16 -> 8 x 8
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 x 8 -> 4 x 4
            nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        feature_dim = 512 * 4 * 4

        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim)

    def forward(self,x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        h = h.flatten(start_dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Prevent numerical overflow in exp(logvar).
        logvar = torch.clamp(logvar,min=-10.0,max=10.0)

        return mu, logvar


# ============================================================
# VAE utilities
# ============================================================
def reparameterize(mu: torch.Tensor,logvar: torch.Tensor) -> torch.Tensor:
    """
    z = mu + sigma * epsilon
    epsilon ~ N(0, I)
    """
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)

    return mu + std * epsilon


def kl_divergence(mu: torch.Tensor,logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z|x) || N(0, I))

    The value is averaged over both batch and latent dimensions.
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())

    return kl.mean()


# ============================================================
# Conditional WGAN-GP gradient penalty
# ============================================================
def gradient_penalty_conditional(critic: nn.Module,real: torch.Tensor,fake: torch.Tensor,labels: torch.Tensor,device: str) -> torch.Tensor:
    batch_size = real.size(0)

    epsilon = torch.rand(batch_size,1,1,1,device=device)

    interpolated = (epsilon * real + (1.0 - epsilon) * fake)
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated,labels)

    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size,-1)
    gradient_norm = gradients.norm(2,dim=1)

    return ((gradient_norm - 1.0) ** 2).mean()


# ============================================================
# File utilities
# ============================================================
def append_csv(csv_path: str,row: dict) -> None:
    file_exists = os.path.exists(csv_path)

    with open(csv_path,"a",newline="",encoding="utf-8") as file:
        writer = csv.DictWriter(file,fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def set_requires_grad(model: nn.Module,requires_grad: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = requires_grad


def save_checkpoint(
    path: str,
    epoch: int,
    stage_name: str,
    encoder: nn.Module,
    generator: nn.Module,
    critic: nn.Module,
    optimizer_encoder_stage1,
    optimizer_encoder_generator,
    optimizer_critic,
    config: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "stage": stage_name,
            "encoder": encoder.state_dict(),
            "generator": generator.state_dict(),
            "critic": critic.state_dict(),
            "optimizer_encoder_stage1":
                optimizer_encoder_stage1.state_dict(),
            "optimizer_encoder_generator":
                optimizer_encoder_generator.state_dict(),
            "optimizer_critic":
                optimizer_critic.state_dict(),
            "config": config,
        },
        path,
    )


# ============================================================
# Main
# ============================================================
def main():
    # --------------------------------------------------------
    # Basic settings
    # --------------------------------------------------------
    epochs = 100
    stage1_epochs = 40

    batch_size = 128
    z_dim = 256
    label_dim = 64
    num_classes = 2

    num_workers = 4
    save_every = 5
    seed = 42

    device = "cuda"

    # --------------------------------------------------------
    # Learning rates
    # --------------------------------------------------------
    lr_encoder_stage1 = 1e-4

    lr_encoder_stage2 = 5e-5
    lr_generator_stage2 = 1e-5
    lr_critic_stage2 = 5e-5

    # --------------------------------------------------------
    # Loss weights
    # --------------------------------------------------------
    # First run recommendation:
    # use 1e-4 first; increase later only if needed.
    beta_max = 1e-4
    kl_warmup_epochs = 20

    lambda_adv = 0.005
    lambda_random_gan = 0.005

    lambda_gp = 10.0
    n_critic = 3

    # --------------------------------------------------------
    # Domain labels
    # --------------------------------------------------------
    EURO_SAT_LABEL = 0
    DEEP_GLOBE_LABEL = 1

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    data_root_g = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    pretrained_generator_path = "./save_model/G_wgangp_satellite_conditional_C.pth"
    pretrained_critic_path = "./save_model/D_wgangp_satellite_conditional_C.pth"
    save_root = "./save_data/VAEEncoder_to_ConditionalGAN"

    model_root = "./save_model"

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    history_path = os.path.join(save_root,"training_history.csv")

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    # --------------------------------------------------------
    # EuroSAT dataset
    # --------------------------------------------------------
    base_eurosat = datasets.ImageFolder(root=data_root_e,transform=transform)
    base_eurosat = Subset(base_eurosat,range(min(50000, len(base_eurosat))))
    eurosat_dataset = DomainLabelDataset(dataset=base_eurosat,domain_label=EURO_SAT_LABEL)

    # --------------------------------------------------------
    # DeepGlobe dataset
    # --------------------------------------------------------
    deepglobe_dataset = SatelliteDataset(root=data_root_g,transform=transform,domain_label=DEEP_GLOBE_LABEL)
    deepglobe_dataset = Subset(deepglobe_dataset,range(min(50000, len(deepglobe_dataset))))

    # --------------------------------------------------------
    # Merge datasets
    # --------------------------------------------------------
    combined_dataset = ConcatDataset([eurosat_dataset,deepglobe_dataset])
    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    print(f"Device           = {device}")
    print(f"EuroSAT images   = {len(eurosat_dataset)}")
    print(f"DeepGlobe images = {len(deepglobe_dataset)}")
    print(f"Total images     = {len(combined_dataset)}")
    print(f"Steps per epoch  = {len(loader)}")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------
    encoder = SharedVAEEncoder(z_dim=z_dim).to(device)
    generator = Generator_Satellite_Conditional(z_dim=z_dim,num_classes=num_classes,label_dim=label_dim,out_ch=3).to(device)

    critic = Critic_Satellite_Conditional(in_ch=3,num_classes=num_classes,feature_dim=512).to(device)

    if not os.path.exists(pretrained_generator_path):
        raise FileNotFoundError(
            "Cannot find pretrained Generator:\n"
            f"{pretrained_generator_path}"
        )

    if not os.path.exists(pretrained_critic_path):
        raise FileNotFoundError(
            "Cannot find pretrained Critic:\n"
            f"{pretrained_critic_path}"
        )

    generator.load_state_dict(torch.load(pretrained_generator_path,map_location=device,))
    critic.load_state_dict(torch.load(pretrained_critic_path,map_location=device))

    print("Encoder device   =",next(encoder.parameters()).device)
    print("Generator device =",next(generator.parameters()).device)
    print("Critic device    =",next(critic.parameters()).device)

    # --------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------
    optimizer_encoder_stage1 = torch.optim.Adam(encoder.parameters(),lr=lr_encoder_stage1,betas=(0.9, 0.999))
    optimizer_encoder_generator = torch.optim.Adam(
        [
            {
                "params": encoder.parameters(),
                "lr": lr_encoder_stage2,
            },
            {
                "params": generator.parameters(),
                "lr": lr_generator_stage2,
            },
        ],
        betas=(0.0, 0.9),
    )

    optimizer_critic = torch.optim.Adam(critic.parameters(),lr=lr_critic_stage2,betas=(0.0, 0.9))

    # --------------------------------------------------------
    # Fixed samples
    # --------------------------------------------------------
    fixed_count_e = min(64,len(eurosat_dataset))
    fixed_count_g = min(64,len(deepglobe_dataset))

    fixed_e = torch.stack([eurosat_dataset[i][0]for i in range(fixed_count_e)]).to(device)
    fixed_g = torch.stack([deepglobe_dataset[i][0]for i in range(fixed_count_g)]).to(device)

    fixed_label_e = torch.full((fixed_e.size(0),),EURO_SAT_LABEL,dtype=torch.long,device=device)
    fixed_label_g = torch.full((fixed_g.size(0),),DEEP_GLOBE_LABEL,dtype=torch.long,device=device)
    fixed_random_z = torch.randn(64,z_dim,device=device)

    fixed_random_label_e = torch.full((64,),EURO_SAT_LABEL,dtype=torch.long,device=device)
    fixed_random_label_g = torch.full((64,),DEEP_GLOBE_LABEL,dtype=torch.long,device=device)
    save_image((fixed_e + 1.0) / 2.0, os.path.join(save_root,"real_EuroSAT.png",),nrow=8)
    save_image((fixed_g + 1.0) / 2.0,os.path.join(save_root,"real_DeepGlobe.png",),nrow=8)

    # --------------------------------------------------------
    # Config saved inside checkpoints
    # --------------------------------------------------------
    config = {
        "epochs": epochs,
        "stage1_epochs": stage1_epochs,
        "batch_size": batch_size,
        "z_dim": z_dim,
        "label_dim": label_dim,
        "num_classes": num_classes,
        "beta_max": beta_max,
        "kl_warmup_epochs": kl_warmup_epochs,
        "lambda_adv": lambda_adv,
        "lambda_random_gan": lambda_random_gan,
        "lambda_gp": lambda_gp,
        "n_critic": n_critic,
        "lr_encoder_stage1": lr_encoder_stage1,
        "lr_encoder_stage2": lr_encoder_stage2,
        "lr_generator_stage2": lr_generator_stage2,
        "lr_critic_stage2": lr_critic_stage2,
    }

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    global_step = 0

    for epoch in range(epochs):
        beta = beta_max * min(1.0,float(epoch + 1) / float(kl_warmup_epochs))

        # ====================================================
        # Stage selection
        # ====================================================
        if epoch < stage1_epochs:
            stage_name = "stage1_encoder_only"

            encoder.train()
            generator.eval()
            critic.eval()

            set_requires_grad(encoder,True)
            set_requires_grad(generator,False)
            set_requires_grad(critic,False)

        else:
            stage_name = "stage2_joint_vae_gan"
            encoder.train()
            generator.train()
            critic.train()

            set_requires_grad(encoder,True)
            set_requires_grad(generator,True)
            set_requires_grad(critic,True)

        # ====================================================
        # Epoch statistics
        # ====================================================
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_adv = 0.0
        epoch_random_adv = 0.0
        epoch_critic = 0.0
        epoch_gp = 0.0

        epoch_mu_mean = 0.0
        epoch_mu_std = 0.0

        euro_recon_sum = 0.0
        deep_recon_sum = 0.0
        euro_batches = 0
        deep_batches = 0

        num_batches = 0

        # ====================================================
        # Batch loop
        # ====================================================
        for real, labels in loader:
            real = real.to(device,non_blocking=True)
            labels = labels.to(device,dtype=torch.long,non_blocking=True)
            batch_now = real.size(0)

            # =================================================
            # Stage 1:
            # Frozen pretrained GAN, train shared VAE Encoder.
            # =================================================
            if epoch < stage1_epochs:
                mu, logvar = encoder(real)
                z_reconstruction = reparameterize(mu,logvar)
                reconstruction = generator(z_reconstruction,labels)
                loss_recon = F.l1_loss(reconstruction,real)
                loss_kl = kl_divergence(mu,logvar)
                loss_adv = torch.zeros((),device=device)
                loss_random_adv = torch.zeros((),device=device)
                loss_critic = torch.zeros((),device=device)
                gp = torch.zeros((),device=device)
                loss_total = loss_recon + beta * loss_kl
                optimizer_encoder_stage1.zero_grad(set_to_none=True)
                loss_total.backward()

                torch.nn.utils.clip_grad_norm_(encoder.parameters(),max_norm=5.0)

                optimizer_encoder_stage1.step()

            # =================================================
            # Stage 2:
            # Jointly train Encoder + Generator + Critic.
            # =================================================
            else:
                # ---------------------------------------------
                # A. Train Critic
                # ---------------------------------------------
                set_requires_grad(critic,True)
                set_requires_grad(encoder,False)
                set_requires_grad(generator,False)

                critic_loss_accum = 0.0
                gp_accum = 0.0

                for _ in range(n_critic):
                    with torch.no_grad():
                        mu_detached, logvar_detached = encoder(real)
                        z_recon_detached = reparameterize(mu_detached,logvar_detached)
                        fake_reconstruction = generator(z_recon_detached,labels)
                        random_z_detached = torch.randn(batch_now,z_dim,device=device)
                        fake_random = generator(random_z_detached,labels)

                    # Do not average images pixel-wise.
                    # Concatenate both fake groups so the Critic
                    # learns reconstruction fakes and random fakes.
                    fake_for_critic = torch.cat([fake_reconstruction,fake_random],dim=0)
                    labels_for_fake = torch.cat([labels,labels,],dim=0)
                    critic_real = critic(real,labels)
                    critic_fake = critic(fake_for_critic,labels_for_fake)

                    # GP requires matching real/fake batch sizes.
                    use_random_branch = (torch.rand((),device=device,).item() < 0.5)

                    if use_random_branch:
                        fake_for_gp = fake_random
                    else:
                        fake_for_gp = fake_reconstruction

                    gp = gradient_penalty_conditional(
                        critic=critic,
                        real=real,
                        fake=fake_for_gp,
                        labels=labels,
                        device=device,
                    )

                    loss_critic = (critic_fake.mean() - critic_real.mean() + lambda_gp * gp)
                    optimizer_critic.zero_grad(set_to_none=True)

                    loss_critic.backward()
                    optimizer_critic.step()

                    critic_loss_accum += (loss_critic.item())
                    gp_accum += gp.item()

                loss_critic = torch.tensor(critic_loss_accum / n_critic,device=device)

                gp = torch.tensor(gp_accum / n_critic,device=device)

                # ---------------------------------------------
                # B. Train Encoder + Generator
                # ---------------------------------------------
                set_requires_grad(critic,False)
                set_requires_grad(encoder,True)
                set_requires_grad(generator,True)

                mu, logvar = encoder(real)

                z_reconstruction = reparameterize(mu,logvar)
                reconstruction = generator(z_reconstruction,labels)
                loss_recon = F.l1_loss(reconstruction,real)

                loss_kl = kl_divergence(mu,logvar)

                # Make reconstruction images realistic.
                loss_adv = -critic(
                    reconstruction,
                    labels,
                ).mean()

                # Preserve original random-generation ability.
                random_z = torch.randn(batch_now,z_dim,device=device)
                random_fake = generator(random_z,labels)
                loss_random_adv = -critic(random_fake,labels).mean()
                loss_total = (loss_recon + beta * loss_kl + lambda_adv * loss_adv + lambda_random_gan * loss_random_adv)
                optimizer_encoder_generator.zero_grad(set_to_none=True)
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(generator.parameters()),max_norm=5.0)

                optimizer_encoder_generator.step()

            # =================================================
            # Per-domain reconstruction statistics
            # =================================================
            euro_mask = (
                labels == EURO_SAT_LABEL
            )
            deep_mask = (
                labels == DEEP_GLOBE_LABEL
            )

            if euro_mask.any():
                euro_loss = F.l1_loss(
                    reconstruction[euro_mask],
                    real[euro_mask],
                )

                euro_recon_sum += euro_loss.item()
                euro_batches += 1

            if deep_mask.any():
                deep_loss = F.l1_loss(
                    reconstruction[deep_mask],
                    real[deep_mask],
                )

                deep_recon_sum += deep_loss.item()
                deep_batches += 1

            # =================================================
            # Statistics
            # =================================================
            epoch_total += loss_total.item()
            epoch_recon += loss_recon.item()
            epoch_kl += loss_kl.item()
            epoch_adv += loss_adv.item()
            epoch_random_adv += loss_random_adv.item()
            epoch_critic += loss_critic.item()
            epoch_gp += gp.item()

            epoch_mu_mean += (mu.detach().mean().item())
            epoch_mu_std += (mu.detach().std().item())

            num_batches += 1

            if global_step % 200 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Stage={stage_name} | "
                    f"Step={global_step} | "
                    f"beta={beta:.6f} | "
                    f"Recon={loss_recon.item():.4f} | "
                    f"KL={loss_kl.item():.4f} | "
                    f"Adv={loss_adv.item():.4f} | "
                    f"RandomAdv={loss_random_adv.item():.4f} | "
                    f"D={loss_critic.item():.4f} | "
                    f"GP={gp.item():.4f} | "
                    f"Total={loss_total.item():.4f} | "
                    f"mu_mean={mu.mean().item():.4f} | "
                    f"mu_std={mu.std().item():.4f}"
                )

            global_step += 1

        # ====================================================
        # Epoch summary
        # ====================================================
        avg_total = epoch_total / num_batches
        avg_recon = epoch_recon / num_batches
        avg_kl = epoch_kl / num_batches
        avg_adv = epoch_adv / num_batches
        avg_random_adv = (epoch_random_adv / num_batches)
        avg_critic = epoch_critic / num_batches
        avg_gp = epoch_gp / num_batches

        avg_mu_mean = (epoch_mu_mean / num_batches)
        avg_mu_std = (epoch_mu_std / num_batches)

        avg_euro_recon = (euro_recon_sum / max(1, euro_batches))
        avg_deep_recon = (deep_recon_sum / max(1, deep_batches))

        print("=" * 120)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Stage={stage_name} | "
            f"beta={beta:.6f} | "
            f"Total={avg_total:.4f} | "
            f"Recon={avg_recon:.4f} | "
            f"KL={avg_kl:.4f} | "
            f"Adv={avg_adv:.4f} | "
            f"RandomAdv={avg_random_adv:.4f} | "
            f"D={avg_critic:.4f} | "
            f"GP={avg_gp:.4f}"
        )
        print(
            f"EuroSAT L1={avg_euro_recon:.4f} | "
            f"DeepGlobe L1={avg_deep_recon:.4f} | "
            f"mu_mean={avg_mu_mean:.4f} | "
            f"mu_std={avg_mu_std:.4f}"
        )
        print("=" * 120)

        append_csv(
            history_path,
            {
                "epoch": epoch + 1,
                "stage": stage_name,
                "beta": beta,
                "total_loss": avg_total,
                "reconstruction_loss": avg_recon,
                "kl_loss": avg_kl,
                "adversarial_loss": avg_adv,
                "random_adversarial_loss":
                    avg_random_adv,
                "critic_loss": avg_critic,
                "gradient_penalty": avg_gp,
                "eurosat_reconstruction":
                    avg_euro_recon,
                "deepglobe_reconstruction":
                    avg_deep_recon,
                "mu_mean": avg_mu_mean,
                "mu_std": avg_mu_std,
            },
        )

        # ====================================================
        # Save images and checkpoint
        # ====================================================
        if ((epoch + 1) % save_every == 0 or epoch + 1 == stage1_epochs or epoch + 1 == epochs):
            encoder.eval()
            generator.eval()

            with torch.no_grad():
                # ---------------------------------------------
                # EuroSAT latent and reconstructions
                # ---------------------------------------------
                mu_e, logvar_e = encoder(fixed_e)

                sample_z_e = reparameterize(mu_e,logvar_e)
                recon_e_mu = generator(mu_e,fixed_label_e)
                recon_e_sample = generator(sample_z_e,fixed_label_e)

                # ---------------------------------------------
                # DeepGlobe latent and reconstructions
                # ---------------------------------------------
                mu_g, logvar_g = encoder(fixed_g)

                sample_z_g = reparameterize(mu_g,logvar_g)
                recon_g_mu = generator(mu_g,fixed_label_g)
                recon_g_sample = generator(sample_z_g,fixed_label_g)

                # ---------------------------------------------
                # Random GAN generation
                # ---------------------------------------------
                random_e = generator(fixed_random_z,fixed_random_label_e)
                random_g = generator(fixed_random_z,fixed_random_label_g)

            save_image((recon_e_mu + 1.0) / 2.0,os.path.join(save_root,f"EuroSAT_mu_ep{epoch + 1}.png",),nrow=8)
            save_image((recon_e_sample + 1.0) / 2.0,os.path.join(save_root,f"EuroSAT_sample_ep{epoch + 1}.png"),nrow=8)
            save_image((recon_g_mu + 1.0) / 2.0,os.path.join(save_root,f"DeepGlobe_mu_ep{epoch + 1}.png",),nrow=8)
            save_image((recon_g_sample + 1.0) / 2.0,os.path.join(save_root,f"DeepGlobe_sample_ep{epoch + 1}.png"),nrow=8)
            save_image((random_e + 1.0) / 2.0,os.path.join(save_root,f"random_EuroSAT_ep{epoch + 1}.png"),nrow=8)
            save_image((random_g + 1.0) / 2.0,os.path.join(save_root,f"random_DeepGlobe_ep{epoch + 1}.png"),nrow=8)

            checkpoint_path = os.path.join(model_root,"joint_vae_gan_satellite_latest.pth")

            save_checkpoint(
                path=checkpoint_path,
                epoch=epoch + 1,
                stage_name=stage_name,
                encoder=encoder,
                generator=generator,
                critic=critic,
                optimizer_encoder_stage1=
                    optimizer_encoder_stage1,
                optimizer_encoder_generator=
                    optimizer_encoder_generator,
                optimizer_critic=
                    optimizer_critic,
                config=config,
            )

            print(f"Saved images and checkpoint at epoch {epoch + 1}.")

    # --------------------------------------------------------
    # Final individual model files
    # --------------------------------------------------------
    torch.save(encoder.state_dict(),os.path.join(model_root,"shared_vae_encoder_satellite.pth"))
    torch.save(generator.state_dict(),os.path.join(model_root,"joint_generator_satellite.pth"))
    torch.save(critic.state_dict(),os.path.join(model_root,"joint_critic_satellite.pth"))

    print("Training completed.")
    print(
        "DDPM should use encoder output mu, "
        "not sampled z.")


if __name__ == "__main__":
    main()
