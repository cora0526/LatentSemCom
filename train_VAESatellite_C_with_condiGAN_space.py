import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from model import Encoder_Satellite, Generator_Satellite_Conditional, Critic_Satellite_Conditional
from com_utils import mimo_2x2_channel_zf_train
from utils import SatelliteDataset
import csv
import math
from skimage.metrics import structural_similarity as structural_ssim
import os

def power_normalize(z, eps=1e-8):
    power = z.pow(2).mean(dim=1,keepdim=True)
    return z / torch.sqrt(power + eps)

def get_img(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def tensor_to_01(x):
    return ((x.detach().float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)

def batch_psnr(original, reconstructed, max_val=1.0):
    original_01 = tensor_to_01(original)
    reconstructed_01 = tensor_to_01(reconstructed)

    mse = F.mse_loss(
        reconstructed_01,
        original_01,
        reduction="none",
    ).flatten(1).mean(dim=1)

    psnr_values = torch.where(
        mse <= 1e-12,
        torch.full_like(mse, float("inf")),
        10.0 * torch.log10((max_val ** 2) / mse),
    )

    finite_values = psnr_values[torch.isfinite(psnr_values)]
    mean_psnr = (
        float("inf")
        if finite_values.numel() == 0
        else float(finite_values.mean().item())
    )

    return mean_psnr, psnr_values.numpy()


def batch_ssim(original, reconstructed):
    original_01 = tensor_to_01(original)
    reconstructed_01 = tensor_to_01(reconstructed)

    values = []

    for original_img, reconstructed_img in zip(
        original_01,
        reconstructed_01,
    ):
        if original_img.shape[0] == 1:
            original_np = original_img.squeeze(0).numpy()
            reconstructed_np = reconstructed_img.squeeze(0).numpy()

            value = structural_ssim(
                original_np,
                reconstructed_np,
                data_range=1.0,
            )
        else:
            original_np = original_img.permute(1, 2, 0).numpy()
            reconstructed_np = reconstructed_img.permute(1, 2, 0).numpy()

            value = structural_ssim(
                original_np,
                reconstructed_np,
                channel_axis=2,
                data_range=1.0,
            )

        values.append(float(value))

    return float(np.mean(values)), np.asarray(values)

def calculate_batch_metrics(original, reconstructed):
    mean_psnr, psnr_values = batch_psnr(
        original,
        reconstructed,
    )
    mean_ssim, ssim_values = batch_ssim(
        original,
        reconstructed,
    )

    return {
        "psnr_mean": mean_psnr,
        "psnr_min": float(np.min(psnr_values)),
        "psnr_max": float(np.max(psnr_values)),
        "ssim_mean": mean_ssim,
        "ssim_min": float(np.min(ssim_values)),
        "ssim_max": float(np.max(ssim_values)),
    }


def metric_text(value, digits=2):
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def append_metrics_csv(csv_path, row):
    exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def gradient_penalty_conditional(critic, real, fake, labels, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    mixed = epsilon * real + (1.0 - epsilon) * fake
    mixed.requires_grad_(True)

    mixed_scores = critic(mixed, labels)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=mixed,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

def main(epochs=100,batch_size=128,z_dim=256,save_every=5,device="cuda"):
    save_root = "./save_data/EncoderSatellite_CommonConditionGAN_TwoEncoder_with_fix_H"
    os.makedirs(save_root, exist_ok=True)

    clean_epochs = 20

    lambda_latent = 0.001
    lr_encoder = 1e-4

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])
    data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    data_root_g = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    ds_e = datasets.ImageFolder(data_root_e, transform=tfm)
    ds_g = SatelliteDataset(data_root_g, transform=tfm)

    ds_e = Subset(ds_e, range(min(50000, len(ds_e))))
    ds_g = Subset(ds_g, range(min(50000, len(ds_g))))

    dl_e = DataLoader(ds_e,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    dl_g = DataLoader(ds_g,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    
    num_steps = min(len(dl_e), len(dl_g))
    # ---- load pretrained G ----
    G = Generator_Satellite_Conditional(z_dim=z_dim,num_classes=2,label_dim=64,out_ch=3).to(device)
    G.load_state_dict(torch.load("./save_model/G_wgangp_satellite_conditional_C.pth", map_location=device))
    G.eval()

    for p in G.parameters():
        p.requires_grad = False

    # D = Critic_Satellite_Conditional(in_ch=3,num_classes=2,feature_dim=512).to(device)
    # D.load_state_dict(torch.load("./save_model/D_wgangp_satellite_conditional_C.pth", map_location=device))
    # D.eval()

    # for p in D.parameters():
    #     p.requires_grad = False

    # ---- encoder (use your class) ----
    E_e = Encoder_Satellite(z_dim=z_dim).to(device)  # EuroSAT encoder
    E_e.load_state_dict(torch.load("./save_model/E_invert_to_conditionG_C_EuroSAT_fixed_H.pth",map_location=device))
    
    E_g = Encoder_Satellite(z_dim=z_dim).to(device)  # DeepGlobe encoder
    E_g.load_state_dict(torch.load("./save_model/E_invert_to_conditionG_C_deepglobe_fixed_H.pth",map_location=device))

    opt = torch.optim.Adam(list(E_e.parameters()) + list(E_g.parameters()),lr=lr_encoder,betas=(0.0,0.9))

    # opt_D = torch.optim.Adam(D.parameters(),lr=lr_critic,betas=(0.0, 0.9))

    fixed_e = get_img(next(iter(dl_e)))[:64].to(device)
    fixed_g = get_img(next(iter(dl_g)))[:64].to(device)

    fixed_label_e = torch.zeros(fixed_e.size(0), dtype=torch.long, device=device)
    fixed_label_g = torch.ones(fixed_g.size(0), dtype=torch.long, device=device)

    save_image((fixed_e + 1) / 2, f"{save_root}/real_EuroSAT.png", nrow=8)
    save_image((fixed_g + 1) / 2, f"{save_root}/real_deepglobe.png", nrow=8)
    
    step = 0

    for ep in range(epochs):
        E_e.train()
        E_g.train()
        G.eval()
        # D.train()

        total_loss = 0.0
        total_e = 0.0
        total_g = 0.0

        for batch_e, batch_g in zip(dl_e, dl_g):
            real_e = get_img(batch_e).to(device, non_blocking=True)
            real_g = get_img(batch_g).to(device, non_blocking=True)

            batch_now = min(real_e.size(0), real_g.size(0))
            real_e = real_e[:batch_now]
            real_g = real_g[:batch_now]

            label_e = torch.zeros(batch_now, dtype=torch.long, device=device)
            label_g = torch.ones(batch_now, dtype=torch.long, device=device)

            mu_e_clean, _ = E_e(real_e)
            mu_g_clean, _ = E_g(real_g)

            mu_e_tx = power_normalize(mu_e_clean)
            mu_g_tx = power_normalize(mu_g_clean)

            if ep < clean_epochs:
                # mu_e_tx = mu_e_clean
                # mu_g_tx = mu_g_clean
                mu_e_rx = mu_e_tx
                mu_g_rx = mu_g_tx
                snr_db = None
                stage_name = "clean"
            else:
                if ep < 40:
                    snr_min, snr_max = 20.0, 30.0
                elif ep < 70:
                    snr_min, snr_max = 10.0, 30.0
                else:
                    snr_min, snr_max = 0.0, 30.0

                snr_db = torch.empty(1, device=device).uniform_(snr_min,snr_max).item()

                mu_e_rx, mu_g_rx = mimo_2x2_channel_zf_train(mu_e_tx,mu_g_tx,snr_db)
                stage_name = "channel"

            recon_e = G(mu_e_rx, label_e)
            recon_g = G(mu_g_rx, label_g)

            
            # ------------- Encoders + G step -------------
            loss_recon_e = F.l1_loss(recon_e, real_e)
            loss_recon_g = F.l1_loss(recon_g, real_g)

            if ep < clean_epochs:
                loss_latent_e = torch.zeros((), device=device)
                loss_latent_g = torch.zeros((), device=device)
            else:
                loss_latent_e = F.mse_loss(mu_e_rx, mu_e_tx)
                loss_latent_g = F.mse_loss(mu_g_rx, mu_g_tx)

            # loss_adv_e = -D(recon_e, label_e).mean()
            # loss_adv_g = -D(recon_g, label_g).mean()

            loss_e = (loss_recon_e + lambda_latent * loss_latent_e)
            loss_g = (loss_recon_g + lambda_latent * loss_latent_g)
            loss = loss_e + loss_g

            opt.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(E_e.parameters()) + list(E_g.parameters()) ,max_norm=5.0)
            opt.step()

            total_loss += loss.item()
            total_e += loss_e.item()
            total_g += loss_g.item()

            if step % 200 == 0:
                snr_text = "clean" if snr_db is None else f"{snr_db:.2f} dB"
                print(
                    f"ep {ep + 1}/{epochs} "
                    f"step {step} | "
                    f"stage={stage_name} | "
                    f"SNR={snr_text} | "
                    f"E={loss_e.item():.4f} | "
                    f"G={loss_g.item():.4f} | "
                    f"Total={loss.item():.4f}"
                )

            step += 1

        print(
            f"Epoch {ep + 1}/{epochs} | "
            f"EG={total_loss / num_steps:.4f} | "
            f"EuroSAT={total_e / num_steps:.4f} | "
            f"DeepGlobe={total_g / num_steps:.4f} | "
        )

        if (ep + 1) % save_every == 0:
            E_e.eval()
            E_g.eval()
            G.eval()

            with torch.no_grad():
                mu_e_clean, _ = E_e(fixed_e)
                mu_g_clean, _ = E_g(fixed_g)

                mu_e_tx = power_normalize(mu_e_clean)
                mu_g_tx = power_normalize(mu_g_clean)

                recon_e_clean = G(mu_e_tx, fixed_label_e)
                recon_g_clean = G(mu_g_tx, fixed_label_g)

                mu_e_rx, mu_g_rx = mimo_2x2_channel_zf_train(mu_e_tx,mu_g_tx,10.0)

                recon_e_snr10 = G(mu_e_rx, fixed_label_e)
                recon_g_snr10 = G(mu_g_rx, fixed_label_g)

                

                save_image((recon_e_clean + 1.0) / 2.0,f"{save_root}/recon_EuroSAT_clean_ep{ep + 1}.png",nrow=8)
                save_image((recon_g_clean + 1.0) / 2.0,f"{save_root}/recon_DeepGlobe_clean_ep{ep + 1}.png",nrow=8)
                save_image((recon_e_snr10 + 1.0) / 2.0,f"{save_root}/recon_EuroSAT_snr10_ep{ep + 1}.png",nrow=8)
                save_image((recon_g_snr10 + 1.0) / 2.0,f"{save_root}/recon_DeepGlobe_snr10_ep{ep + 1}.png",nrow=8)
            
            metrics_e_clean = calculate_batch_metrics(fixed_e,recon_e_clean)
            metrics_g_clean = calculate_batch_metrics(fixed_g,recon_g_clean)
            metrics_e_snr10 = calculate_batch_metrics(fixed_e,recon_e_snr10)
            metrics_g_snr10 = calculate_batch_metrics(fixed_g,recon_g_snr10)

            print("-" * 72)
            print("EuroSAT clean   | "
                f"PSNR={metric_text(metrics_e_clean['psnr_mean'])} dB | "
                f"SSIM={metrics_e_clean['ssim_mean']:.4f}")
            print("DeepGlobe clean | "
                f"PSNR={metric_text(metrics_g_clean['psnr_mean'])} dB | "
                f"SSIM={metrics_g_clean['ssim_mean']:.4f}")
            print("EuroSAT 10 dB   | "
                f"PSNR={metric_text(metrics_e_snr10['psnr_mean'])} dB | "
                f"SSIM={metrics_e_snr10['ssim_mean']:.4f}")
            print("DeepGlobe 10 dB | "
                f"PSNR={metric_text(metrics_g_snr10['psnr_mean'])} dB | "
                f"SSIM={metrics_g_snr10['ssim_mean']:.4f}")
            print("-" * 72)

            ep_num = ep + 1

            torch.save(E_e.state_dict(), "./save_model/E_invert_to_conditionG_C_EuroSAT_fixed_H.pth")
            torch.save(E_g.state_dict(), "./save_model/E_invert_to_conditionG_C_deepglobe_fixed_H.pth")
            torch.save(G.state_dict(),"./save_model/G_wgangp_satellite_conditional_C.pth",)
            # torch.save(D.state_dict(),"./save_model/D_wgangp_satellite_conditional_C.pth",)

    torch.save(E_e.state_dict(), "./save_model/E_invert_to_conditionG_C_EuroSAT_fixed_H.pth")
    torch.save(E_g.state_dict(), "./save_model/E_invert_to_conditionG_C_deepglobe_fixed_H.pth")
    torch.save(G.state_dict(),"./save_model/G_wgangp_satellite_conditional_finetuned.pth",)
    # torch.save(D.state_dict(),"./save_model/D_wgangp_satellite_conditional_finetuned.pth",)

    print("Saved encoders and fine-tuned conditional GAN.")


if __name__ == "__main__":
    main()