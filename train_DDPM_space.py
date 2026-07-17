import os
import glob
import time
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import Satellite_Latent_DDPM, Generator_Satellite, Encoder_Satellite

# ============================================================
# Dataset
# ============================================================
class SatelliteDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform

        if len(self.files) == 0:
            raise RuntimeError(f"No '*_sat_*.jpg' images found in: {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # DeepGlobe source ID
        label = 0
        return img, label

# ============================================================
# DDPM schedule
# ============================================================
def linear_beta_schedule(T, device):
    return torch.linspace(1e-4,2e-2,T,device=device,dtype=torch.float32)

# ============================================================
# Dataset-level latent statistics
# 每個 latent 維度各自計算 mean/std
# ============================================================
@torch.no_grad()
def compute_latent_stats(encoder,loader,device,max_batches=200):
    encoder.eval()

    sum_z = None
    sum_z2 = None
    total_samples = 0

    for batch_idx, (x, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        z, _ = encoder(x)

        if sum_z is None:
            z_dim = z.size(1)

            sum_z = torch.zeros(1,z_dim,device=device,dtype=torch.float64)
            sum_z2 = torch.zeros(1,z_dim,device=device,dtype=torch.float64)

        z64 = z.double()

        sum_z += z64.sum(dim=0, keepdim=True)
        sum_z2 += (z64 ** 2).sum(dim=0, keepdim=True)
        total_samples += z.size(0)

    if total_samples == 0:
        raise RuntimeError("No samples were used to compute latent statistics.")

    mean = sum_z / total_samples

    variance = (sum_z2 / total_samples - mean ** 2).clamp(min=1e-6)

    std = torch.sqrt(variance)

    return mean.float(), std.float()

# ============================================================
# Weighted latent reconstruction loss
# 高 timestep 時 x0_pred 不穩，因此使用 alpha_bar 加權
# ============================================================
def weighted_latent_mse(pred, target, weight):
    """
    pred/target: [B, z_dim]
    weight:      [B, 1]
    """
    per_sample = ((pred - target) ** 2).mean(dim=1)

    return (per_sample * weight.squeeze(1)).mean()

# ============================================================
# Weighted image L1
# ============================================================
def weighted_image_l1(pred, target, weight):
    """
    pred/target: [B, C, H, W]
    weight:      [B, 1]
    """
    per_sample = (pred - target).abs().flatten(1).mean(dim=1)

    return (per_sample * weight.squeeze(1)).mean()

def main():
    # ========================================================
    # Settings
    # ========================================================
    device = "cuda"
    epochs = 100
    steps_per_epoch = 1000

    batch_size = 32
    z_dim = 256
    lr = 2e-4

    T = 1000

    # Channel/interference settings
    max_interference = 1.0
    channel_noise_std = 0.05

    # Loss weights
    lambda_eps = 1.0
    lambda_latent = 0.2
    lambda_image = 0.02

    stats_batches = 200

    save_root = "./save_data/DDPM_satellite_separator"
    model_root = "./save_model"

    G_path = "./save_model/G_end2end_satellite_C.pth"
    E_g_path = "./save_model/E_end2end_deepglobe_C.pth"
    E_e_path = "./save_model/E_end2end_EuroSAT_C.pth"

    final_model_path = ("./save_model/DDPM_satellite_separator_v2.pth")
    latest_checkpoint_path = os.path.join(save_root,"DDPM_satellite_separator_v2_latest.pth",)
    stats_path = ("./save_model/satellite_latent_stats_v2.pth")

    data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    data_root_g = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    # =========================
    # Dataset
    # =========================
    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    ds_e = datasets.ImageFolder(data_root_e, transform=tfm)
    ds_g = SatelliteDataset(data_root_g, transform=tfm)

    ds_e = Subset(ds_e, range(min(50000, len(ds_e))))
    ds_g = Subset(ds_g, range(min(50000, len(ds_g))))

    dl_e = DataLoader(ds_e,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    dl_g = DataLoader(ds_g,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    # =========================
    # Load G / Encoders
    # =========================
    G = Generator_Satellite(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load(G_path, map_location=device))
    G.eval()
    for p in G.parameters():
        p.requires_grad = False

    E_g = Encoder_Satellite(z_dim=z_dim).to(device)
    E_g.load_state_dict(torch.load(E_g_path, map_location=device))
    E_g.eval()
    for p in E_g.parameters():
        p.requires_grad = False

    E_e = Encoder_Satellite(z_dim=z_dim).to(device)
    E_e.load_state_dict(torch.load(E_e_path, map_location=device))
    E_e.eval()
    for p in E_e.parameters():
        p.requires_grad = False
    
    # ========================================================
    # Compute dataset-level latent statistics
    # ========================================================
    print("Computing DeepGlobe latent statistics...")

    mean_g, std_g = compute_latent_stats(encoder=E_g,loader=dl_g,device=device,max_batches=stats_batches,)
    print("Computing EuroSAT latent statistics...")

    mean_e, std_e = compute_latent_stats(encoder=E_e,loader=dl_e,device=device,max_batches=stats_batches,)

    torch.save({"mean_g": mean_g.cpu(),"std_g": std_g.cpu(),"mean_e": mean_e.cpu(),"std_e": std_e.cpu(),},stats_path,)

    print("DeepGlobe mean/std average:",f"{mean_g.mean().item():.6f},",f"{std_g.mean().item():.6f}",)
    print("EuroSAT mean/std average:",f"{mean_e.mean().item():.6f},",f"{std_e.mean().item():.6f}",)
    print(f"Saved latent statistics: {stats_path}")

    # 建立新的無限 iterator
    iter_e = cycle(dl_e)
    iter_g = cycle(dl_g)

    # =========================
    # DDPM
    # =========================
    DDPM = Satellite_Latent_DDPM(z_dim=z_dim).to(device)
    optimizer = torch.optim.AdamW(DDPM.parameters(),lr=lr,weight_decay=1e-4)

    betas = linear_beta_schedule(T, device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    sqrt_ab = torch.sqrt(alpha_bars)
    sqrt_1mab = torch.sqrt(1.0 - alpha_bars)

    # ========================================================
    # History
    # ========================================================
    epoch_total_history = []
    epoch_eps_history = []
    epoch_latent_history = []
    epoch_image_history = []

    start_time = time.time()

    # ========================================================
    # Training
    # ========================================================
    for epoch in range(epochs):
        DDPM.train()

        total_loss_sum = 0.0
        eps_loss_sum = 0.0
        latent_loss_sum = 0.0
        image_loss_sum = 0.0

        for step in range(steps_per_epoch):
            x_g, _ = next(iter_g)
            x_e, _ = next(iter_e)

            x_g = x_g.to(device,non_blocking=True)
            x_e = x_e.to(device,non_blocking=True)

            # ================================================
            # Frozen encoders
            # ================================================
            with torch.no_grad():
                z_g_raw, _ = E_g(x_g)
                z_e_raw, _ = E_e(x_e)

                z_g = (z_g_raw - mean_g) / std_g
                z_e = (z_e_raw - mean_e) / std_e

            B = z_g.size(0)

            # ================================================
            # 每個 sample 隨機干擾強度 0~1
            # ================================================
            interference_g = (max_interference * torch.rand(B,1,device=device))
            interference_e = (max_interference * torch.rand(B,1,device=device))

            channel_noise_g = (channel_noise_std * torch.randn_like(z_g))
            channel_noise_e = (channel_noise_std * torch.randn_like(z_e))

            # ================================================
            # Target-centric conditions
            #
            # DeepGlobe target:
            # z_g + interference * z_e
            #
            # EuroSAT target:
            # z_e + interference * z_g
            # ================================================
            cond_g = (z_g + interference_g * z_e + channel_noise_g) / torch.sqrt(1.0 + interference_g ** 2)
            cond_e = (z_e + interference_e * z_g + channel_noise_e) / torch.sqrt(1.0 + interference_e ** 2)

            # ================================================
            # DeepGlobe branch
            # ================================================
            y_g = torch.zeros(B,device=device,dtype=torch.long)
            noise_g = torch.randn_like(z_g)
            t_g = torch.randint(low=0,high=T,size=(B,),device=device,dtype=torch.long)
            sqrt_ab_g = sqrt_ab[t_g].view(B, 1)
            sqrt_1mab_g = sqrt_1mab[t_g].view(B, 1)

            zt_g = (sqrt_ab_g * z_g + sqrt_1mab_g * noise_g)

            eps_pred_g = DDPM(zt_g,t_g,y_g,cond=cond_g)

            z0_pred_g = (zt_g - sqrt_1mab_g * eps_pred_g) / sqrt_ab_g.clamp(min=1e-4)

            # 避免非常高 timestep 導致極端 latent
            z0_pred_g = z0_pred_g.clamp(min=-6.0,max=6.0)
            loss_eps_g = F.mse_loss(eps_pred_g,noise_g)

            # 高 t 時 auxiliary loss 自動變小
            aux_weight_g = alpha_bars[t_g].view(B, 1).detach()

            loss_latent_g = weighted_latent_mse(z0_pred_g,z_g,aux_weight_g)

            z0_pred_g_raw = (z0_pred_g * std_g + mean_g)

            # G frozen，但不能放在 no_grad 裡
            img_pred_g = G(z0_pred_g_raw.view(B,z_dim,1,1))

            with torch.no_grad():
                img_target_g = G(z_g_raw.view(B,z_dim,1,1))

            loss_img_g = weighted_image_l1(img_pred_g,img_target_g,aux_weight_g)

            # ================================================
            # EuroSAT branch
            # ================================================
            y_e = torch.ones(B,device=device,dtype=torch.long)
            noise_e = torch.randn_like(z_e)
            t_e = torch.randint(low=0,high=T,size=(B,),device=device,dtype=torch.long)
            sqrt_ab_e = sqrt_ab[t_e].view(B, 1)
            sqrt_1mab_e = sqrt_1mab[t_e].view(B, 1)
            zt_e = (sqrt_ab_e * z_e + sqrt_1mab_e * noise_e)
            eps_pred_e = DDPM(zt_e,t_e,y_e,cond=cond_e)
            z0_pred_e = (zt_e - sqrt_1mab_e * eps_pred_e) / sqrt_ab_e.clamp(min=1e-4)
            z0_pred_e = z0_pred_e.clamp(min=-6.0,max=6.0)
            loss_eps_e = F.mse_loss(eps_pred_e,noise_e)
            aux_weight_e = alpha_bars[t_e].view(B, 1).detach()
            loss_latent_e = weighted_latent_mse(z0_pred_e,z_e,aux_weight_e)
            z0_pred_e_raw = (z0_pred_e * std_e + mean_e)
            img_pred_e = G(z0_pred_e_raw.view(B,z_dim,1,1))

            with torch.no_grad():
                img_target_e = G(z_e_raw.view(B,z_dim,1,1))
            loss_img_e = weighted_image_l1(img_pred_e,img_target_e,aux_weight_e)

            # ================================================
            # Combined losses
            # ================================================
            loss_eps = 0.5 * (loss_eps_g + loss_eps_e)
            loss_latent = 0.5 * (loss_latent_g + loss_latent_e)
            loss_image = 0.5 * (loss_img_g + loss_img_e)
            loss = (lambda_eps * loss_eps + lambda_latent * loss_latent + lambda_image * loss_image)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DDPM.parameters(),max_norm=1.0)
            optimizer.step()

            total_loss_sum += loss.item()
            eps_loss_sum += loss_eps.item()
            latent_loss_sum += loss_latent.item()
            image_loss_sum += loss_image.item()

            if (step + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Step [{step+1}/{steps_per_epoch}] "
                    f"Loss={loss.item():.6f} "
                    f"Eps={loss_eps.item():.6f} "
                    f"Latent={loss_latent.item():.6f} "
                    f"Image={loss_image.item():.6f} "
                    f"IG={interference_g.mean().item():.3f} "
                    f"IE={interference_e.mean().item():.3f}"
                )

        # ====================================================
        # Epoch averages
        # ====================================================
        avg_total = (total_loss_sum / steps_per_epoch)
        avg_eps = (eps_loss_sum / steps_per_epoch)
        avg_latent = (latent_loss_sum / steps_per_epoch)
        avg_image = (image_loss_sum / steps_per_epoch)

        epoch_total_history.append(avg_total)
        epoch_eps_history.append(avg_eps)
        epoch_latent_history.append(avg_latent)
        epoch_image_history.append(avg_image)

        elapsed = (time.time() - start_time) / 60.0

        print(
            f"\nEpoch {epoch+1}/{epochs} | "
            f"Total={avg_total:.6f} | "
            f"Eps={avg_eps:.6f} | "
            f"Latent={avg_latent:.6f} | "
            f"Image={avg_image:.6f} | "
            f"Elapsed={elapsed:.1f} min\n"
        )

        # ====================================================
        # Save checkpoint
        # ====================================================
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": DDPM.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),

                "mean_g": mean_g.cpu(),
                "std_g": std_g.cpu(),
                "mean_e": mean_e.cpu(),
                "std_e": std_e.cpu(),

                "T": T,
                "z_dim": z_dim,

                "lambda_eps": lambda_eps,
                "lambda_latent": lambda_latent,
                "lambda_image": lambda_image,

                "epoch_total_history": epoch_total_history,
                "epoch_eps_history": epoch_eps_history,
                "epoch_latent_history": epoch_latent_history,
                "epoch_image_history": epoch_image_history,
            },latest_checkpoint_path,
        )

        # 每 10 epochs 額外保留一份
        if (epoch + 1) % 10 == 0:
            torch.save(DDPM.state_dict(),os.path.join(save_root,f"DDPM_satellite_v2_epoch_{epoch+1}.pth"))

        # ====================================================
        # Update epoch loss plot
        # ====================================================
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_total_history,label="Total Loss")
        plt.plot(epoch_eps_history,label="Noise Prediction Loss")
        plt.plot(epoch_latent_history,label="Weighted Latent Loss")
        plt.plot(epoch_image_history,label="Weighted Image Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Satellite Conditional Latent DDPM v2 Training")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        loss_fig_path = os.path.join(save_root,"loss_curve_v2.png")
        plt.savefig(loss_fig_path,dpi=300)
        plt.close()

    # ========================================================
    # Final model
    # ========================================================
    torch.save(DDPM.state_dict(),final_model_path)

    print(f"Saved final DDPM: {final_model_path}")
    print(f"Saved checkpoint: {latest_checkpoint_path}")
    print(f"Saved loss curve: {loss_fig_path}")


if __name__ == "__main__":
    main()