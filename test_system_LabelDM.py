import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import math
from model import Ganerator_MNIST, VAE_Encoder_MNIST, LabelCondLatentDenoiser
from utils import linear_beta_schedule, denoise_latent_ddpm

def psnr_from_mse(mse: float, data_range: float = 2.0) -> float:
    # data_range=2 because [-1,1]
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range * data_range) / mse)

def whiten_latent(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # z: (B, z_dim)
    z = z - z.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, keepdim=True) + eps)
    return z

@torch.no_grad()
def eval_pipeline(
        z_dim = 64,
        device = "cuda",
        batch_size = 64,
        t_val=300,          # 給 DM 評估用
        lam=0.5,            # Fashion latent 加到 MNIST latent 的係數
        do_whiten=True,     # 是否白化 z_f
        out_dir="./save_data/test_labelDM",
):
    
    os.makedirs(out_dir, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
    ])

    ds = datasets.MNIST(root=r"D:\Python\SemCom\saved_data", train=True, download=True, transform=tfm)
    ds_f = datasets.FashionMNIST(root=r"D:\Python\SemCom\saved_data", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    x_m, _ = next(iter(dl))  #iter() 從dataloader取一個batch的data，next()去下一個batch 第一次呼叫此func就是取第一筆
    x_f, _ = next(iter(dl_f))
    x_m = x_m.to(device)    #convert to CUDA
    x_f = x_f.to(device)

    save_image((x_m + 1) / 2, f"./save_data/test_labelDM/input_MNIST.png", nrow=8)
    save_image((x_f + 1) / 2, f"./save_data/test_labelDM/input_fashion.png", nrow=8)

    G = Ganerator_MNIST(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load("./save_model/G_wgangp_C.pth"))
    G.eval()
    for p in G.parameters():
        p.requires_grad = False

    E = VAE_Encoder_MNIST(latent_dim=z_dim).to(device)
    E.load_state_dict(torch.load("./save_model/E_invert_to_G_C.pth"))
    E.eval()
    for p in E.parameters():
        p.requires_grad = False

    E_f = VAE_Encoder_MNIST(latent_dim=z_dim).to(device)
    E_f.load_state_dict(torch.load("./save_model/E_invert_to_G_F.pth"))
    E_f.eval()
    for p in E_f.parameters():
        p.requires_grad = False

    E_k = VAE_Encoder_MNIST(latent_dim=z_dim).to(device)
    E_k.load_state_dict(torch.load("./save_model/E_invert_to_G_K.pth"))
    E_k.eval()
    for p in E_k.parameters():
        p.requires_grad = False
    
    DM = LabelCondLatentDenoiser(num_classes=2, z_dim=z_dim, tdim=128, ydim=64).to(device)
    DM.load_state_dict(torch.load("./save_model/DM_latent_label.pth"))
    DM.eval()
    for p in DM.parameters():
        p.requires_grad = False

    # test
    z_m, _ = E(x_m)     # (B,z)
    z_f, _ = E_f(x_f)   # (B,z)

    z_f_use = whiten_latent(z_f) if do_whiten else z_f
    z_mix = z_m + lam * z_f_use 
    z_mix_k = z_m + lam * z_f_use 
    # -------------------------
    # GAN reconstruct
    # -------------------------
    x_g_m = G(z_m.view(batch_size, z_dim, 1, 1))
    x_g_f = G(z_f.view(batch_size, z_dim, 1, 1))
    x_g_mix = G(z_mix.view(batch_size, z_dim, 1, 1))

    # 存圖：baseline vs 加擾動
    save_image((x_g_m + 1) / 2, os.path.join(out_dir, "recon_G_from_zm.png"), nrow=8)
    save_image((x_g_f + 1) / 2, os.path.join(out_dir, "recon_G_from_zt.png"), nrow=8)
    save_image((x_g_mix + 1) / 2, os.path.join(out_dir, "recon_G_from_zm_plus_zf.png"), nrow=8)

    # DDPM schedule (must match training)
    T = 1000    
    betas = linear_beta_schedule(T, device=device)  #第T步時的噪聲
    alphas = 1.0 - betas    
    alpha_bars = torch.cumprod(alphas, dim=0)   #訊號保留的比例
    sqrt_ab = torch.sqrt(alpha_bars)    #forwarding noise
    sqrt_1mab = torch.sqrt(1 - alpha_bars)

    # fixed timestep tensor
    tt = torch.full((batch_size,), int(t_val), device=device, dtype=torch.long)

    # 用固定 seed，讓每次存圖一致（可選）
    torch.manual_seed(0)

    def dm_denoise_and_decode(z0, y, tag: str):
        # z0: (B, z_dim)
        eps = torch.randn_like(z0)  #產生隨機雜訊(高斯)
        zt = sqrt_ab[t_val].view(1) * z0 + sqrt_1mab[t_val].view(1) * eps

        # one-step denoise
        eps_pred = DM(zt, tt, y)
        z0_hat_1step = (zt - sqrt_1mab[t_val].view(1) * eps_pred) / sqrt_ab[t_val].view(1)

        # full reverse denoise (from t_val -> 0)
        # full reverse denoise (帶 label)
        z0_hat_full = denoise_latent_ddpm(
        DM, zt, betas, alphas, alpha_bars, y=y, t_start=int(t_val)
        )

        # GAN decode
        x_zt = G(zt.view(batch_size, z_dim, 1, 1))
        x_1step = G(z0_hat_1step.view(batch_size, z_dim, 1, 1))
        x_full = G(z0_hat_full.view(batch_size, z_dim, 1, 1))

        # save images（把 label 寫進檔名）
        y_id = int(y[0].item())  # 假設整個 batch 同一類
        save_image((x_zt + 1) / 2,  os.path.join(out_dir, f"{tag}_y{y_id}_G_from_zt_t{t_val}.png"), nrow=8)
        save_image((x_1step + 1) / 2, os.path.join(out_dir, f"{tag}_y{y_id}_G_denoise_1step_t{t_val}.png"), nrow=8)
        save_image((x_full + 1) / 2,  os.path.join(out_dir, f"{tag}_y{y_id}_G_denoise_full_t{t_val}.png"), nrow=8)

        # metrics
        latent_mse_noisy = F.mse_loss(zt, z0).item()
        latent_mse_1 = F.mse_loss(z0_hat_1step, z0).item()
        latent_mse_f = F.mse_loss(z0_hat_full, z0).item()

        latent_recovery_ratio = (latent_mse_noisy - latent_mse_f) / (latent_mse_noisy + 1e-12)

        x_ref = G(z0.view(batch_size, z_dim, 1, 1))
        img_mse_f = F.mse_loss(x_full, x_ref).item()
        img_psnr_f = psnr_from_mse(img_mse_f)

        print(
            f"[{tag} | y={y_id}] "
            f"latent_recovery={latent_recovery_ratio:.3f} | "
            f"latent_mse(full)={latent_mse_f:.4f} | "
            f"img_psnr(full)={img_psnr_f:.2f} dB"
        )

        y0 = torch.zeros(batch_size, device=device, dtype=torch.long)  # label 0
        y1 = torch.ones(batch_size, device=device, dtype=torch.long)   # label 1

        # # clean MNIST latent，用兩個 label 各 denoise 一次
        # dm_denoise_and_decode(z_m, y0, tag="clean")
        # dm_denoise_and_decode(z_m, y1, tag="clean")

        # mix latent，用兩個 label 各 denoise 一次
        # dm_denoise_and_decode(z_mix, y0, tag="mix")
        dm_denoise_and_decode(z_mix, y1, tag="mix")

        # # mix_k 同理
        # dm_denoise_and_decode(z_mix_k, y0, tag="mix_k")
        # dm_denoise_and_decode(z_mix_k, y1, tag="mix_k")

   

if __name__ == "__main__":
    # 你可以在這裡改參數
    eval_pipeline(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=64,
        z_dim=64,
        t_val=300,
        lam=0.5,
        do_whiten=True,
        out_dir="./save_data/test_labelDM"
    )