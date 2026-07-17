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
import csv
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
from model import Satellite_Latent_DDPM, Generator_Satellite, Encoder_Satellite

def calculate_psnr(pred, target, data_range=1.0):
    """
    pred, target: [B, C, H, W]，數值範圍 0~1
    """
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10(
        torch.tensor(data_range ** 2, device=pred.device) / (mse + 1e-10)
    )
    return psnr.mean().item()


def calculate_ssim(pred, target, data_range=1.0):
    """
    使用 skimage 計算 batch 平均 SSIM。
    pred, target: [B, C, H, W]，數值範圍 0~1
    """
    pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy()
    target_np = target.detach().cpu().permute(0, 2, 3, 1).numpy()

    values = []

    for pred_img, target_img in zip(pred_np, target_np):
        value = skimage_ssim(
            target_img,
            pred_img,
            data_range=data_range,
            channel_axis=-1
        )
        values.append(value)

    return float(np.mean(values))

class SatelliteDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = 0
        return img, label


def linear_beta_schedule(T, device):
    return torch.linspace(1e-4, 2e-2, T, device=device)

@torch.no_grad()
def sample_cond_latent(DDPM, cond, label, z_dim, T_sample, betas, alphas, alpha_bars):
    B = cond.size(0)
    zt = torch.randn(B, z_dim, device=cond.device)

    for i in reversed(range(T_sample)):
        t = torch.full((B,), i, device=cond.device, dtype=torch.long)

        eps_pred = DDPM(zt, t, label, cond)

        beta_t = betas[t].view(B, 1)
        alpha_t = alphas[t].view(B, 1)
        alpha_bar_t = alpha_bars[t].view(B, 1)

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            zt - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
        )

        if i > 0:
            noise = torch.randn_like(zt)
            zt = mean + torch.sqrt(beta_t) * noise
        else:
            zt = mean

    return zt


def main():
    # =========================
    # Settings
    # =========================
    device = "cuda" 

    batch_size = 32
    z_dim = 256

    T = 1000

    save_root = "./save_data/DDPM_satellite_separator"
    os.makedirs(save_root, exist_ok=True)
    os.makedirs("./save_model", exist_ok=True)

    G_path = "./save_model/G_end2end_satellite_C.pth"
    E_g_path = "./save_model/E_end2end_deepglobe_C.pth"
    E_e_path = "./save_model/E_end2end_EuroSAT_C.pth"
    DDPM_path = "./save_model/DDPM_satellite_separator_v2.pth"

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

    dl_e = DataLoader(
        ds_e,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    dl_g = DataLoader(
        ds_g,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    iter_e = cycle(dl_e)
    iter_g = cycle(dl_g)

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
 
    DDPM = Satellite_Latent_DDPM(z_dim=z_dim).to(device)
    DDPM.load_state_dict(torch.load(DDPM_path, map_location=device))
    DDPM.eval()
    for p in DDPM.parameters():
        p.requires_grad = False

    betas = linear_beta_schedule(T, device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)


        # =========================
    # Testing / Inference
    # =========================
    T_sample = 400

    alpha_list = [0.0,0.1,0.3,0.5,0.7,1.0]

    noise_std = 0.05

    # 測試多少個 batch
    # 10 batches × 32 images = 320 組測試資料
    num_test_batches = 10

    results = {"alpha": [],"latent_mse_g": [],"latent_mse_e": [],"psnr_g": [],"psnr_e": [],"ssim_g": [],"ssim_e": [],}

    print("=" * 80)
    print("Start DDPM separator evaluation")
    print("=" * 80)

    for alpha in alpha_list:

        latent_mse_g_sum = 0.0
        latent_mse_e_sum = 0.0

        psnr_g_sum = 0.0
        psnr_e_sum = 0.0

        ssim_g_sum = 0.0
        ssim_e_sum = 0.0

        for batch_idx in range(num_test_batches):

            x_g, _ = next(iter_g)
            x_e, _ = next(iter_e)

            x_g = x_g.to(device, non_blocking=True)
            x_e = x_e.to(device, non_blocking=True)

            with torch.no_grad():

                # ==========================================
                # Encoder：取得原始 latent
                # ==========================================
                z_g_raw, _ = E_g(x_g)
                z_e_raw, _ = E_e(x_e)

                # 每個 sample 的 mean/std
                z_g_mean = z_g_raw.mean(dim=1, keepdim=True)
                z_g_std = (z_g_raw.std(dim=1, keepdim=True) + 1e-6)

                z_e_mean = z_e_raw.mean(dim=1, keepdim=True)
                z_e_std = (z_e_raw.std(dim=1, keepdim=True) + 1e-6)

                # DDPM 使用 normalized latent
                z_g = (z_g_raw - z_g_mean) / z_g_std
                z_e = (z_e_raw - z_e_mean) / z_e_std

                B = z_g.size(0)

                y_g = torch.zeros(B,device=device,dtype=torch.long)

                y_e = torch.ones(B,device=device,dtype=torch.long)

                # ==========================================
                # Ground-truth reconstruction
                # 比較對象是 G(E(x))，而不是原始 x
                # 這樣只評估 separator，不混入 Encoder/GAN 誤差
                # ==========================================
                img_gt_g = G(z_g_raw.view(B, z_dim, 1, 1))

                img_gt_e = G(z_e_raw.view(B, z_dim, 1, 1))

                # ==========================================
                # 建立 latent interference
                # ==========================================
                noise_channel = (noise_std * torch.randn_like(z_g))

                z_rec_deq = (z_g + alpha * z_e + noise_channel)

                # 和訓練流程一致
                z_rec_deq = (z_rec_deq - z_rec_deq.mean(dim=1, keepdim=True)) / (z_rec_deq.std(dim=1, keepdim=True) + 1e-6)

                # ==========================================
                # DDPM separation
                # ==========================================
                z_hat_g = sample_cond_latent(DDPM=DDPM,cond=z_rec_deq,label=y_g,z_dim=z_dim,
                    T_sample=T_sample,betas=betas,alphas=alphas,alpha_bars=alpha_bars)

                z_hat_e = sample_cond_latent(DDPM=DDPM,cond=z_rec_deq,label=y_e,z_dim=z_dim,
                    T_sample=T_sample,betas=betas,alphas=alphas,alpha_bars=alpha_bars)

                # ==========================================
                # normalized latent -> raw latent
                # ==========================================
                z_hat_g_raw = (z_hat_g * z_g_std + z_g_mean)

                z_hat_e_raw = (z_hat_e * z_e_std + z_e_mean)

                # ==========================================
                # Generator reconstruction
                # ==========================================
                img_hat_g = G(z_hat_g_raw.view(B, z_dim, 1, 1))

                img_hat_e = G(z_hat_e_raw.view(B, z_dim, 1, 1))

                # Tanh output: -1~1 -> 0~1
                img_gt_g_01 = ((img_gt_g + 1.0) / 2.0).clamp(0.0, 1.0)

                img_gt_e_01 = ((img_gt_e + 1.0) / 2.0).clamp(0.0, 1.0)

                img_hat_g_01 = ((img_hat_g + 1.0) / 2.0).clamp(0.0, 1.0)

                img_hat_e_01 = ((img_hat_e + 1.0) / 2.0).clamp(0.0, 1.0)

                # ==========================================
                # Metrics
                # ==========================================
                latent_mse_g = F.mse_loss(z_hat_g,z_g).item()

                latent_mse_e = F.mse_loss(z_hat_e,z_e).item()

                psnr_g = calculate_psnr(img_hat_g_01,img_gt_g_01)

                psnr_e = calculate_psnr(img_hat_e_01,img_gt_e_01)

                ssim_g = calculate_ssim(img_hat_g_01,img_gt_g_01)

                ssim_e = calculate_ssim(img_hat_e_01,img_gt_e_01)

                latent_mse_g_sum += latent_mse_g
                latent_mse_e_sum += latent_mse_e

                psnr_g_sum += psnr_g
                psnr_e_sum += psnr_e

                ssim_g_sum += ssim_g
                ssim_e_sum += ssim_e

                # 第一個 batch 儲存代表圖片
                if batch_idx == 0:

                    alpha_name = str(alpha).replace(".", "p")

                    save_image(img_gt_g_01,os.path.join(save_root,f"gt_deepglobe_alpha_{alpha_name}.png"),nrow=8)

                    save_image(img_gt_e_01,os.path.join(save_root,f"gt_eurosat_alpha_{alpha_name}.png"),nrow=8)

                    save_image(
                        img_hat_g_01,
                        os.path.join(
                            save_root,
                            f"ddpm_deepglobe_alpha_{alpha_name}.png"
                        ),
                        nrow=8
                    )

                    save_image(
                        img_hat_e_01,
                        os.path.join(
                            save_root,
                            f"ddpm_eurosat_alpha_{alpha_name}.png"
                        ),
                        nrow=8
                    )

        # ==========================================
        # 每個 alpha 的平均結果
        # ==========================================
        latent_mse_g_avg = (
            latent_mse_g_sum / num_test_batches
        )

        latent_mse_e_avg = (
            latent_mse_e_sum / num_test_batches
        )

        psnr_g_avg = psnr_g_sum / num_test_batches
        psnr_e_avg = psnr_e_sum / num_test_batches

        ssim_g_avg = ssim_g_sum / num_test_batches
        ssim_e_avg = ssim_e_sum / num_test_batches

        results["alpha"].append(alpha)

        results["latent_mse_g"].append(
            latent_mse_g_avg
        )

        results["latent_mse_e"].append(
            latent_mse_e_avg
        )

        results["psnr_g"].append(psnr_g_avg)
        results["psnr_e"].append(psnr_e_avg)

        results["ssim_g"].append(ssim_g_avg)
        results["ssim_e"].append(ssim_e_avg)

        print(
            f"alpha={alpha:.1f} | "
            f"MSE-G={latent_mse_g_avg:.6f} | "
            f"MSE-E={latent_mse_e_avg:.6f} | "
            f"PSNR-G={psnr_g_avg:.3f} dB | "
            f"PSNR-E={psnr_e_avg:.3f} dB | "
            f"SSIM-G={ssim_g_avg:.4f} | "
            f"SSIM-E={ssim_e_avg:.4f}"
        )

    # =========================
    # Save CSV
    # =========================
    csv_path = os.path.join(save_root,"ddpm_separator_metrics.csv")

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "alpha",
            "latent_mse_deepglobe",
            "latent_mse_eurosat",
            "psnr_deepglobe",
            "psnr_eurosat",
            "ssim_deepglobe",
            "ssim_eurosat",
        ])

        for i in range(len(results["alpha"])):
            writer.writerow([
                results["alpha"][i],
                results["latent_mse_g"][i],
                results["latent_mse_e"][i],
                results["psnr_g"][i],
                results["psnr_e"][i],
                results["ssim_g"][i],
                results["ssim_e"][i],
            ])

    print(f"Saved CSV: {csv_path}")

    # =========================
    # Plot 1: Latent MSE vs alpha
    # =========================
    plt.figure(figsize=(8, 5))

    plt.plot(
        results["alpha"],
        results["latent_mse_g"],
        marker="o",
        label="DeepGlobe"
    )

    plt.plot(
        results["alpha"],
        results["latent_mse_e"],
        marker="s",
        label="EuroSAT"
    )

    plt.xlabel("Interference Strength α")
    plt.ylabel("Latent MSE")
    plt.title("Latent MSE vs Interference Strength")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    mse_plot_path = os.path.join(
        save_root,
        "latent_mse_vs_alpha.png"
    )

    plt.savefig(mse_plot_path, dpi=300)
    plt.close()

    # =========================
    # Plot 2: PSNR vs alpha
    # =========================
    plt.figure(figsize=(8, 5))

    plt.plot(
        results["alpha"],
        results["psnr_g"],
        marker="o",
        label="DeepGlobe"
    )

    plt.plot(
        results["alpha"],
        results["psnr_e"],
        marker="s",
        label="EuroSAT"
    )

    plt.xlabel("Interference Strength α")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Interference Strength")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    psnr_plot_path = os.path.join(
        save_root,
        "psnr_vs_alpha.png"
    )

    plt.savefig(psnr_plot_path, dpi=300)
    plt.close()

    # =========================
    # Plot 3: SSIM vs alpha
    # =========================
    plt.figure(figsize=(8, 5))

    plt.plot(
        results["alpha"],
        results["ssim_g"],
        marker="o",
        label="DeepGlobe"
    )

    plt.plot(
        results["alpha"],
        results["ssim_e"],
        marker="s",
        label="EuroSAT"
    )

    plt.xlabel("Interference Strength α")
    plt.ylabel("SSIM")
    plt.title("SSIM vs Interference Strength")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    ssim_plot_path = os.path.join(
        save_root,
        "ssim_vs_alpha.png"
    )

    plt.savefig(ssim_plot_path, dpi=300)
    plt.close()

    print("=" * 80)
    print(f"Saved MSE plot : {mse_plot_path}")
    print(f"Saved PSNR plot: {psnr_plot_path}")
    print(f"Saved SSIM plot: {ssim_plot_path}")
    print(f"Saved images and results to: {save_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()