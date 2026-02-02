import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import math
from model import Ganerator_MNIST, VAE_Encoder_MNIST, LatentDenoiser
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
        out_dir="./save_data/test_eval",
):
    
    os.makedirs(out_dir, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
    ])

    ds = datasets.MNIST(root=r"D:\Python\SemCom\saved_data", train=True, download=True, transform=tfm)
    ds_f = datasets.FashionMNIST(root=r"D:\Python\SemCom\saved_data", train=True, download=True, transform=tfm)
    ds_k = datasets.KMNIST(root=r"D:\Python\SemCom\saved_data", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    dl_k = DataLoader(ds_k, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    x_m, _ = next(iter(dl))  #iter() 從dataloader取一個batch的data，next()去下一個batch 第一次呼叫此func就是取第一筆
    x_f, _ = next(iter(dl_f))
    x_k, _ = next(iter(dl_k))
    x_m = x_m.to(device)    #convert to CUDA
    x_f = x_f.to(device)
    x_k = x_k.to(device)

    save_image((x_m + 1) / 2, f"./save_data/test_eval/input_MNIST.png", nrow=8)
    save_image((x_f + 1) / 2, f"./save_data/test_eval/input_fashion.png", nrow=8)
    save_image((x_k + 1) / 2, f"./save_data/test_eval/input_KMNIST.png", nrow=8)

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
    
    DM = LatentDenoiser(z_dim=64, tdim=128).to(device)
    DM.load_state_dict(torch.load("./save_model/DM_latent_inter.pth"))
    DM.eval()
    for p in DM.parameters():
        p.requires_grad = False

    # test
    z_m, _ = E(x_m)     # (B,z)
    z_f, _ = E_f(x_f)   # (B,z)
    z_k, _ = E_k(x_k)

    z_f_use = whiten_latent(z_f) if do_whiten else z_f
    z_k_use = whiten_latent(z_k) if do_whiten else z_k
    z_mix = z_m + lam * z_f_use 
    z_mix_k = z_m + lam/2 * z_f_use + lam/2 * z_k_use
    # -------------------------
    # GAN reconstruct
    # -------------------------
    x_g_m = G(z_m.view(batch_size, z_dim, 1, 1))
    x_g_f = G(z_f.view(batch_size, z_dim, 1, 1))
    x_g_k = G(z_k.view(batch_size, z_dim, 1, 1))
    x_g_mix = G(z_mix.view(batch_size, z_dim, 1, 1))

    # 存圖：baseline vs 加擾動
    save_image((x_g_m + 1) / 2, f"./save_data/test_eval/recon_G_from_zm.png", nrow=8)
    save_image((x_g_f + 1) / 2, f"./save_data/test_eval/recon_G_from_zt.png", nrow=8)
    save_image((x_g_k + 1) / 2, f"./save_data/test_eval/recon_G_from_zk.png", nrow=8)
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

    def dm_denoise_and_decode(z0, tag: str):
        # z0: (B, z_dim)
        eps = torch.randn_like(z0)  #產生隨機雜訊(高斯)
        zt = sqrt_ab[t_val].view(1) * z0 + sqrt_1mab[t_val].view(1) * eps

        # one-step denoise
        eps_pred = DM(zt, tt)
        z0_hat_1step = (zt - sqrt_1mab[t_val].view(1) * eps_pred) / sqrt_ab[t_val].view(1)

        # full reverse denoise (from t_val -> 0)
        z0_hat_full = denoise_latent_ddpm(
            DM, zt, betas, alphas, alpha_bars, t_start=int(t_val)
        )

        # GAN decode
        x_zt = G(zt.view(batch_size, z_dim, 1, 1))
        x_1step = G(z0_hat_1step.view(batch_size, z_dim, 1, 1))
        x_full = G(z0_hat_full.view(batch_size, z_dim, 1, 1))

        # save images
        save_image((x_zt + 1) / 2,  os.path.join(out_dir, f"{tag}_G_from_zt_t{t_val}.png"), nrow=8)
        save_image((x_1step + 1) / 2, os.path.join(out_dir, f"{tag}_G_denoise_1step_t{t_val}.png"), nrow=8)
        save_image((x_full + 1) / 2,  os.path.join(out_dir, f"{tag}_G_denoise_full_t{t_val}.png"), nrow=8)

        # （可選）印一下指標，方便你確認 full 真的比 1step 好
        latent_mse_1 = F.mse_loss(z0_hat_1step, z0).item()
        latent_mse_f = F.mse_loss(z0_hat_full, z0).item()
        print(f"[{tag}] latent_mse_1step={latent_mse_1:.6f}  latent_mse_full={latent_mse_f:.6f}")

        # latent metrics
        latent_mse_noisy = F.mse_loss(zt, z0).item()
        latent_mse_1 = F.mse_loss(z0_hat_1step, z0).item()
        latent_mse_f = F.mse_loss(z0_hat_full, z0).item()

        latent_recovery_ratio = (
            latent_mse_noisy - latent_mse_f
        ) / (latent_mse_noisy + 1e-12)

        # image metrics
        x_ref = G(z0.view(batch_size, z_dim, 1, 1))

        img_mse_1 = F.mse_loss(x_1step, x_ref).item()
        img_mse_f = F.mse_loss(x_full,  x_ref).item()

        img_psnr_1 = psnr_from_mse(img_mse_1)
        img_psnr_f = psnr_from_mse(img_mse_f)

        print(
            f"[{tag}] "
            f"latent_recovery={latent_recovery_ratio:.3f} | "
            f"latent_mse(full)={latent_mse_f:.4f} | "
            f"img_psnr(full)={img_psnr_f:.2f} dB"
        )


    # 對 clean MNIST latent 做 denoise
    dm_denoise_and_decode(z_m, tag="clean")

    # 對 mix latent 做 denoise
    dm_denoise_and_decode(z_mix, tag="mix")

    dm_denoise_and_decode(z_mix_k, tag="mix_k")

   

if __name__ == "__main__":
    # 你可以在這裡改參數
    eval_pipeline(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=64,
        z_dim=64,
        t_val=300,
        lam=0.5,
        do_whiten=True,
        out_dir="./save_data/test_eval"
    )