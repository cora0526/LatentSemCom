import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from model import Encoder_Satellite, Generator_Satellite_Conditional, LatentAdapter
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

def differentiable_ssim(x,y,window_size=11,c1=0.01 ** 2,c2=0.03 ** 2):
    x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    y = ((y + 1.0) / 2.0).clamp(0.0, 1.0)

    padding = window_size // 2

    mu_x = F.avg_pool2d(x,kernel_size=window_size,stride=1,padding=padding,)
    mu_y = F.avg_pool2d(y,kernel_size=window_size,stride=1,padding=padding,)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = (F.avg_pool2d(x * x,kernel_size=window_size,stride=1,padding=padding) - mu_x_sq)

    sigma_y_sq = (F.avg_pool2d(y * y,kernel_size=window_size,stride=1,padding=padding) - mu_y_sq)

    sigma_xy = (F.avg_pool2d( x * y,kernel_size=window_size,stride=1,padding=padding) - mu_xy)

    ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2) / ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2) + 1e-8))

    return ssim_map.mean()

def reconstruction_objective(reconstruction,target,l1_weight=0.8,ssim_weight=0.2):
    loss_l1 = F.l1_loss(reconstruction,target,)

    ssim_value = differentiable_ssim(reconstruction,target)

    loss_ssim = 1.0 - ssim_value

    total = (l1_weight * loss_l1 + ssim_weight * loss_ssim)

    return total, loss_l1, loss_ssim, ssim_value

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

def sample_curriculum_snr(epoch, device):
    if epoch < 3:
        return 30.0
    if epoch < 6:
        return 25.0
    if epoch < 10:
        return 20.0
    if epoch < 15:
        return 15.0
    return torch.empty(1, device=device).uniform_(10.0, 30.0).item()

def validate_and_save(encoder_e,encoder_g,adapter_e,adapter_g,generator,fixed_e,fixed_g,fixed_label_e,fixed_label_g,save_root,epoch_number,metrics_csv_path):
    encoder_e.eval()
    encoder_g.eval()
    adapter_e.eval()
    adapter_g.eval()
    generator.eval()

    with torch.no_grad():
        latent_e_raw, _ = encoder_e(fixed_e)
        latent_g_raw, _ = encoder_g(fixed_g)

        latent_e_base = power_normalize(latent_e_raw)
        latent_g_base = power_normalize(latent_g_raw)

        # 經過 Adapter
        latent_e_tx = power_normalize(adapter_e(latent_e_base))
        latent_g_tx = power_normalize(adapter_g(latent_g_base))

        # Clean reconstruction
        reconstruction_e_clean = generator(latent_e_tx,fixed_label_e)
        reconstruction_g_clean = generator(latent_g_tx,fixed_label_g)

        # 10 dB channel reconstruction
        latent_e_rx, latent_g_rx = (mimo_2x2_channel_zf_train(latent_e_tx,latent_g_tx,10.0))

        reconstruction_e_snr10 = generator(latent_e_rx,fixed_label_e)
        reconstruction_g_snr10 = generator(latent_g_rx,fixed_label_g)

    metrics = {
        "EuroSAT_clean": calculate_batch_metrics(fixed_e,reconstruction_e_clean,),
        "DeepGlobe_clean": calculate_batch_metrics(fixed_g,reconstruction_g_clean),
        "EuroSAT_snr10": calculate_batch_metrics(fixed_e,reconstruction_e_snr10),
        "DeepGlobe_snr10": calculate_batch_metrics(fixed_g,reconstruction_g_snr10)
    }

    print("\n" + "-" * 72)

    for key, value in metrics.items():
        print(f"{key:18s} | "f"PSNR={metric_text(value['psnr_mean'])} dB | "f"SSIM={value['ssim_mean']:.4f}")

    print("-" * 72)

    outputs = {
        "EuroSAT_clean": reconstruction_e_clean,
        "DeepGlobe_clean": reconstruction_g_clean,
        "EuroSAT_snr10": reconstruction_e_snr10,
        "DeepGlobe_snr10": reconstruction_g_snr10,
    }

    for key, image_batch in outputs.items():
        value = metrics[key]

        filename = (f"adapter_{key}_ep{epoch_number}_psnr{metric_text(value['psnr_mean'])}_ssim{value['ssim_mean']:.4f}.png")

        save_image(tensor_to_01(image_batch),os.path.join(save_root, filename),nrow=8)

        append_metrics_csv(
            metrics_csv_path,
            {
                "stage": "adapter",
                "epoch": epoch_number,
                "dataset_condition": key,
                **value,
            },
        )

    return metrics


def main(epochs=20, batch_size=128, z_dim=256, device="cuda"):
    save_root = "./save_data/EncoderSatellite_CommonConditionGAN_TwoEncoder_with_fix_H"
    os.makedirs(save_root, exist_ok=True)

    l1_weight = 0.95
    ssim_weight = 0.05
    clean_weight = 0.4
    channel_weight = 0.4
    consistency_weight = 0.2
    latent_identity_weight = 0.01
    lr_adapter = 1e-4

    # Use a stable channel range while tuning G.
    gan_finetune_snr_min = 10.0
    gan_finetune_snr_max = 30.0

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    generator_path = ("./save_model/G_conditional_last_layer_finetuned_v2.pth")
    encoder_e_path = ("./save_model/E_invert_to_conditionG_C_EuroSAT_fixed_H.pth")
    encoder_g_path = ("./save_model/E_invert_to_conditionG_C_deepglobe_fixed_H.pth")
    adapter_e_best_path = "./save_model/Adapter_EuroSAT_best.pth"
    adapter_g_best_path = "./save_model/Adapter_DeepGlobe_best.pth"
    metrics_csv_path = os.path.join(save_root, "adapter_validation_metrics.csv")

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

    loader_e = DataLoader(
        ds_e, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True,
    )
    loader_g = DataLoader(
        ds_g, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True,
    )
    num_steps = min(len(loader_e), len(loader_g))

    generator = Generator_Satellite_Conditional(
        z_dim=z_dim, num_classes=2, label_dim=64, out_ch=3
    ).to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))

    encoder_e = Encoder_Satellite(z_dim=z_dim).to(device)
    encoder_g = Encoder_Satellite(z_dim=z_dim).to(device)
    encoder_e.load_state_dict(torch.load(encoder_e_path, map_location=device))
    encoder_g.load_state_dict(torch.load(encoder_g_path, map_location=device))

    encoder_e.eval(); encoder_g.eval(); generator.eval()
    for model in (encoder_e, encoder_g, generator):
        for parameter in model.parameters():
            parameter.requires_grad = False

    adapter_e = LatentAdapter(z_dim=z_dim, hidden_dim=512, residual_scale=0.1).to(device)
    adapter_g = LatentAdapter(z_dim=z_dim, hidden_dim=512, residual_scale=0.1).to(device)

    optimizer = torch.optim.Adam(
        list(adapter_e.parameters()) + list(adapter_g.parameters()),
        lr=lr_adapter,
        betas=(0.9, 0.999),
    )

    fixed_e = get_img(next(iter(loader_e)))[:64].to(device)
    fixed_g = get_img(next(iter(loader_g)))[:64].to(device)
    fixed_label_e = torch.zeros(fixed_e.size(0), dtype=torch.long, device=device)
    fixed_label_g = torch.ones(fixed_g.size(0), dtype=torch.long, device=device)

    save_image(tensor_to_01(fixed_e), os.path.join(save_root, "real_EuroSAT.png"), nrow=8)
    save_image(tensor_to_01(fixed_g), os.path.join(save_root, "real_DeepGlobe.png"), nrow=8)

    validate_and_save(
        encoder_e, encoder_g, adapter_e, adapter_g, generator,
        fixed_e, fixed_g, fixed_label_e, fixed_label_g,
        save_root, 0, metrics_csv_path,
    )

    best_score = -float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        adapter_e.train(); adapter_g.train()
        encoder_e.eval(); encoder_g.eval(); generator.eval()
        snr_db = sample_curriculum_snr(epoch, device)

        total_loss = total_clean = total_channel = total_consistency = total_identity = 0.0

        for batch_e, batch_g in zip(loader_e, loader_g):
            real_e = get_img(batch_e).to(device, non_blocking=True)
            real_g = get_img(batch_g).to(device, non_blocking=True)
            current_batch = min(real_e.size(0), real_g.size(0))
            real_e = real_e[:current_batch]
            real_g = real_g[:current_batch]

            label_e = torch.zeros(current_batch, dtype=torch.long, device=device)
            label_g = torch.ones(current_batch, dtype=torch.long, device=device)

            with torch.no_grad():
                latent_e_raw, _ = encoder_e(real_e)
                latent_g_raw, _ = encoder_g(real_g)
                latent_e_base = power_normalize(latent_e_raw)
                latent_g_base = power_normalize(latent_g_raw)

            latent_e_tx = power_normalize(adapter_e(latent_e_base))
            latent_g_tx = power_normalize(adapter_g(latent_g_base))

            latent_e_rx, latent_g_rx = mimo_2x2_channel_zf_train(
                latent_e_tx, latent_g_tx, snr_db
            )

            recon_e_clean = generator(latent_e_tx, label_e)
            recon_g_clean = generator(latent_g_tx, label_g)
            recon_e_channel = generator(latent_e_rx, label_e)
            recon_g_channel = generator(latent_g_rx, label_g)

            clean_e_loss, _, _, _ = reconstruction_objective(
                recon_e_clean,
                real_e,
                l1_weight,
                ssim_weight,
            )

            clean_g_loss, _, _, _ = reconstruction_objective(
                recon_g_clean,
                real_g,
                l1_weight,
                ssim_weight,
            )

            channel_e_loss, _, _, _ = reconstruction_objective(
                recon_e_channel,
                real_e,
                l1_weight,
                ssim_weight,
            )

            channel_g_loss, _, _, _ = reconstruction_objective(
                recon_g_channel,
                real_g,
                l1_weight,
                ssim_weight,
            )

            loss_clean = clean_e_loss + clean_g_loss
            loss_channel = channel_e_loss + channel_g_loss
            loss_consistency = (
                F.l1_loss(recon_e_channel, recon_e_clean.detach())
                + F.l1_loss(recon_g_channel, recon_g_clean.detach())
            )
            loss_identity = (
                F.mse_loss(latent_e_tx, latent_e_base)
                + F.mse_loss(latent_g_tx, latent_g_base)
            )

            loss = (
                clean_weight * loss_clean
                + channel_weight * loss_channel
                + consistency_weight * loss_consistency
                + latent_identity_weight * loss_identity
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(adapter_e.parameters()) + list(adapter_g.parameters()),max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_clean += loss_clean.item()
            total_channel += loss_channel.item()
            total_consistency += loss_consistency.item()
            total_identity += loss_identity.item()

        print(
            f"Adapter epoch {epoch + 1}/{epochs} | SNR={snr_db:.2f} dB | "
            f"total={total_loss / num_steps:.4f} | "
            f"clean={total_clean / num_steps:.4f} | "
            f"channel={total_channel / num_steps:.4f} | "
            f"consistency={total_consistency / num_steps:.4f} | "
            f"identity={total_identity / num_steps:.4f}"
        )

        metrics = validate_and_save(
            encoder_e, encoder_g, adapter_e, adapter_g, generator,
            fixed_e, fixed_g, fixed_label_e, fixed_label_g,
            save_root, epoch + 1, metrics_csv_path,
        )

        mean_psnr = np.mean([metrics[key]["psnr_mean"] for key in metrics])
        mean_ssim = np.mean([metrics[key]["ssim_mean"] for key in metrics])
        min_psnr = min(metrics[key]["psnr_mean"] for key in metrics)
        min_ssim = min(metrics[key]["ssim_mean"] for key in metrics)
        current_score = mean_psnr + 8.0 * mean_ssim + 0.2 * min_psnr + 2.0 * min_ssim

        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch + 1
            torch.save(adapter_e.state_dict(), adapter_e_best_path)
            torch.save(adapter_g.state_dict(), adapter_g_best_path)
            print(f"Saved best adapters | epoch={best_epoch} | score={best_score:.4f}")

    print("\nAdapter training finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"EuroSAT adapter: {adapter_e_best_path}")
    print(f"DeepGlobe adapter: {adapter_g_best_path}")
    print(f"Metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()