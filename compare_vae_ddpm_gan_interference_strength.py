"""
VAE-DDPM-GAN interference-strength sensitivity test (no retraining required).

Fixed experiment:
    Scenario: Mixed-2T2R
    SNR: Config.fixed_snr_db (defined relative to the desired user's signal)
    Channels: AWGN, Rayleigh, and Rician

Receiver models:
    y_E = h_EE z_E + alpha * h_ED z_D + n_E
    y_D = h_DD z_D + alpha * h_DE z_E + n_D

The same image pairs, fading coefficients, unit-noise tensors, and DDIM seeds are
reused for every alpha value within each batch. This isolates the effect of
interference strength. Noise power is based only on the desired signal, so the
requested SNR remains fixed when alpha changes.

Outputs:
    interference_metrics_long.csv
    interference_summary.csv
    interference_summary.tex
    PSNR_vs_alpha_<channel>.png
    SSIM_vs_alpha_<channel>.png
    Latent_Cosine_vs_alpha_<channel>.png
    Latent_MSE_vs_alpha_<channel>.png
    qualitative_<channel>_alpha*.png
"""

from __future__ import annotations

import csv
import glob
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from model_shared_vaegan import load_encoder_generator
from model_seperator import DiffusionSchedule, load_vae_ddpm_gan, sample_ddpm


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"

    z_dim: int = 256
    label_dim: int = 64
    num_classes: int = 2
    diffusion_steps: int = 400

    sampling_method: str = "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0

    batch_size: int = 16
    num_workers: int = 0
    max_test_pairs: int = 32  # Increase to 500 or 1000 for final results.

    fixed_snr_db: float = 15.0
    alpha_values: Tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
    )
    channel_types: Tuple[str, ...] = ("AWGN", "Rayleigh", "Rician")
    rician_k_db: float = 6.0

    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    scratch_vaegan_checkpoint: str = (
        "./save_model/vae_gan_from_scratch_best.pth"
    )
    scratch_vaegan_ddpm_checkpoint: str = (
        "./save_model/DDPM_vaegan_scratch_latent_separator_best.pth"
    )

    save_root: str = "./save_data/VAE_DDPM_GAN_interference_strength"
    qualitative_alphas: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5)
    qualitative_count: int = 8


class SatelliteDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform
        if not self.files:
            raise RuntimeError(f"No matching DeepGlobe images found in:\n{root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.files[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


def get_image(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


@dataclass
class ModelBundle:
    encoder: nn.Module
    decoder: nn.Module
    ddpm: nn.Module
    schedule: DiffusionSchedule


def load_model(config: Config, device: str) -> ModelBundle:
    encoder, generator = load_encoder_generator(
        checkpoint=config.scratch_vaegan_checkpoint,
        device=device,
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
        strict=True,
        freeze=True,
    )
    ddpm, schedule = load_vae_ddpm_gan(
        checkpoint=config.scratch_vaegan_ddpm_checkpoint,
        device=device,
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
        diffusion_steps=config.diffusion_steps,
        strict=True,
        freeze=True,
    )
    return ModelBundle(encoder, generator, ddpm, schedule)


def make_fading_gain(
    channel_type: str,
    batch_size: int,
    device: torch.device,
    rician_k_db: float,
) -> torch.Tensor:
    if channel_type == "AWGN":
        return torch.ones(batch_size, 1, device=device)

    if channel_type == "Rayleigh":
        real = torch.randn(batch_size, 1, device=device)
        imag = torch.randn(batch_size, 1, device=device)
        return torch.sqrt((real.square() + imag.square()) / 2.0).clamp_min(1e-6)

    if channel_type == "Rician":
        k_linear = 10.0 ** (float(rician_k_db) / 10.0)
        los = math.sqrt(k_linear / (k_linear + 1.0))
        scatter = math.sqrt(1.0 / (2.0 * (k_linear + 1.0)))
        real = los + scatter * torch.randn(batch_size, 1, device=device)
        imag = scatter * torch.randn(batch_size, 1, device=device)
        return torch.sqrt(real.square() + imag.square()).clamp_min(1e-6)

    raise ValueError(f"Unknown channel type: {channel_type}")


def add_awgn_from_target_snr(
    received_without_noise: torch.Tensor,
    target_signal: torch.Tensor,
    snr_db: float,
    unit_noise: torch.Tensor,
) -> torch.Tensor:
    """Add noise whose power is defined relative to the desired signal only."""
    if math.isinf(float(snr_db)):
        return received_without_noise
    target_power = target_signal.square().mean(dim=1, keepdim=True).clamp_min(1e-12)
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_std = torch.sqrt(target_power / snr_linear)
    return received_without_noise + unit_noise * noise_std


def to_01(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred = to_01(pred)
    target = to_01(target)
    mse = (pred - target).square().flatten(1).mean(dim=1).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred_np = to_01(pred).permute(0, 2, 3, 1).cpu().numpy()
    target_np = to_01(target).permute(0, 2, 3, 1).cpu().numpy()
    values = [
        structural_similarity(ref, out, data_range=1.0, channel_axis=-1)
        for out, ref in zip(pred_np, target_np)
    ]
    return np.asarray(values, dtype=np.float64)


def latent_mse(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return (pred - target).square().mean(dim=1).detach().cpu().numpy()


def latent_cosine(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return F.cosine_similarity(pred, target, dim=1).detach().cpu().numpy()


class MetricAccumulator:
    def __init__(self):
        self.values: Dict[str, List[float]] = {}

    def add(self, name: str, values) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        self.values.setdefault(name, []).extend(array.tolist())

    def mean(self, name: str) -> float:
        values = self.values.get(name, [])
        return float(np.mean(values)) if values else float("nan")

    def count(self) -> int:
        return len(next(iter(self.values.values()))) if self.values else 0


def add_metrics(
    accumulator: MetricAccumulator,
    reconstruction: torch.Tensor,
    oracle: torch.Tensor,
    real: torch.Tensor,
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
) -> None:
    accumulator.add("psnr_real", batch_psnr(reconstruction, real))
    accumulator.add("ssim_real", batch_ssim(reconstruction, real))
    accumulator.add("psnr_oracle", batch_psnr(reconstruction, oracle))
    accumulator.add("ssim_oracle", batch_ssim(reconstruction, oracle))
    accumulator.add("oracle_psnr_real", batch_psnr(oracle, real))
    accumulator.add("oracle_ssim_real", batch_ssim(oracle, real))
    accumulator.add("latent_mse", latent_mse(pred_z, target_z))
    accumulator.add("latent_cosine", latent_cosine(pred_z, target_z))


@torch.no_grad()
def recover_latent(
    model: ModelBundle,
    received: torch.Tensor,
    label: torch.Tensor,
    seed: int,
    config: Config,
) -> torch.Tensor:
    return sample_ddpm(
        model.ddpm,
        model.schedule,
        received,
        label,
        seed=seed,
        sampling_method=config.sampling_method,
        ddim_steps=config.ddim_steps,
        ddim_eta=config.ddim_eta,
    )


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def alpha_to_isr_db(alpha: float) -> float:
    return float("-inf") if alpha == 0 else 20.0 * math.log10(alpha)


def build_summary_rows(
    accumulators: Dict[float, Dict[str, MetricAccumulator]],
    channel: str,
    config: Config,
) -> List[dict]:
    rows: List[dict] = []
    metrics = (
        "psnr_real",
        "ssim_real",
        "psnr_oracle",
        "ssim_oracle",
        "oracle_psnr_real",
        "oracle_ssim_real",
        "latent_mse",
        "latent_cosine",
    )

    for alpha in config.alpha_values:
        domain_rows = []
        for domain in ("EuroSAT", "DeepGlobe"):
            acc = accumulators[alpha][domain]
            row = {
                "channel": channel,
                "snr_db": config.fixed_snr_db,
                "alpha": alpha,
                "isr_db_amplitude": alpha_to_isr_db(alpha),
                "domain": domain,
                "num_samples": acc.count(),
            }
            row.update({metric: acc.mean(metric) for metric in metrics})
            rows.append(row)
            domain_rows.append(row)

        average = {
            "channel": channel,
            "snr_db": config.fixed_snr_db,
            "alpha": alpha,
            "isr_db_amplitude": alpha_to_isr_db(alpha),
            "domain": "Average",
            "num_samples": sum(row["num_samples"] for row in domain_rows),
        }
        for metric in metrics:
            average[metric] = float(np.mean([row[metric] for row in domain_rows]))
        rows.append(average)

    return rows


def save_latex_table(path: str, rows: List[dict], channel: str, config: Config) -> None:
    selected = [row for row in rows if row["channel"] == channel and row["domain"] == "Average"]
    if not selected:
        return

    best_psnr = max(row["psnr_real"] for row in selected)
    best_ssim = max(row["ssim_real"] for row in selected)
    best_cos = max(row["latent_cosine"] for row in selected)
    best_mse = min(row["latent_mse"] for row in selected)

    def bold(value: float, best: float, decimals: int) -> str:
        text = f"{value:.{decimals}f}"
        return rf"\textbf{{{text}}}" if np.isclose(value, best) else text

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Effect of interference strength on the VAE-DDPM-GAN system under {channel} at SNR $={config.fixed_snr_db:g}$ dB.}}",
        rf"\label{{tab:interference_{channel.lower()}}}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"$\alpha$ & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & Latent cosine $\uparrow$ & Latent MSE $\downarrow$ \\",
        r"\midrule",
    ]
    for row in selected:
        lines.append(
            f"{row['alpha']:.2f} & "
            f"{bold(row['psnr_real'], best_psnr, 3)} & "
            f"{bold(row['ssim_real'], best_ssim, 4)} & "
            f"{bold(row['latent_cosine'], best_cos, 4)} & "
            f"{bold(row['latent_mse'], best_mse, 4)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


DOMAIN_STYLES = {
    "EuroSAT": ("o", "-"),
    "DeepGlobe": ("s", "--"),
    "Average": ("^", "-"),
}


def plot_metric(
    rows: List[dict],
    channel: str,
    metric: str,
    ylabel: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(8.2, 5.4))
    for domain in ("EuroSAT", "DeepGlobe", "Average"):
        selected = sorted(
            [row for row in rows if row["channel"] == channel and row["domain"] == domain],
            key=lambda row: float(row["alpha"]),
        )
        marker, linestyle = DOMAIN_STYLES[domain]
        plt.plot(
            [row["alpha"] for row in selected],
            [row[metric] for row in selected],
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            label=domain,
        )
    plt.xlabel(r"Interference strength $\alpha$")
    plt.ylabel(ylabel)
    plt.title(f"VAE-DDPM-GAN Interference Sensitivity ({channel})")
    plt.xticks(sorted({float(row["alpha"]) for row in rows}))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()


def save_qualitative_grid(
    save_path: str,
    image_e: torch.Tensor,
    image_d: torch.Tensor,
    oracle_e: torch.Tensor,
    oracle_d: torch.Tensor,
    recon_e: torch.Tensor,
    recon_d: torch.Tensor,
    count: int,
) -> None:
    count = min(count, image_e.size(0), image_d.size(0))
    rows = [
        image_e[:count],
        oracle_e[:count],
        recon_e[:count],
        image_d[:count],
        oracle_d[:count],
        recon_d[:count],
    ]
    grid = make_grid(
        torch.cat(rows, dim=0),
        nrow=count,
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(grid, save_path)


def main() -> None:
    config = Config()
    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; switching to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(config.save_root, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    eurosat = datasets.ImageFolder(config.eurosat_root, transform=transform)
    deepglobe = SatelliteDataset(config.deepglobe_root, transform=transform)

    pair_count = min(config.max_test_pairs, len(eurosat), len(deepglobe))
    if pair_count <= 0:
        raise RuntimeError("No test pairs are available.")

    index_generator = torch.Generator().manual_seed(config.seed)
    e_indices = torch.randperm(len(eurosat), generator=index_generator)[:pair_count].tolist()
    d_indices = torch.randperm(len(deepglobe), generator=index_generator)[:pair_count].tolist()

    loader_e = DataLoader(
        Subset(eurosat, e_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    loader_d = DataLoader(
        Subset(deepglobe, d_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Loading VAE-DDPM-GAN...")
    model = load_model(config, str(device))
    all_rows: List[dict] = []

    for channel_index, channel in enumerate(config.channel_types):
        print("=" * 90)
        print(f"Channel={channel} | fixed SNR={config.fixed_snr_db:g} dB")
        accumulators = {
            alpha: {
                "EuroSAT": MetricAccumulator(),
                "DeepGlobe": MetricAccumulator(),
            }
            for alpha in config.alpha_values
        }
        qualitative_saved = {alpha: False for alpha in config.alpha_values}

        for batch_index, (batch_e, batch_d) in enumerate(zip(loader_e, loader_d)):
            image_e = get_image(batch_e).to(device, non_blocking=True)
            image_d = get_image(batch_d).to(device, non_blocking=True)
            current = min(image_e.size(0), image_d.size(0))
            image_e = image_e[:current]
            image_d = image_d[:current]

            # Same latent inputs, fading, and unit noise for all alpha values.
            condition_seed = config.seed + channel_index * 1_000_000 + batch_index
            torch.manual_seed(condition_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(condition_seed)

            with torch.no_grad():
                label_e = torch.zeros(current, dtype=torch.long, device=device)
                label_d = torch.ones(current, dtype=torch.long, device=device)
                z_e, _ = model.encoder(image_e)
                z_d, _ = model.encoder(image_d)
                oracle_e = model.decoder(z_e, label_e)
                oracle_d = model.decoder(z_d, label_d)

                h_ee = make_fading_gain(channel, current, device, config.rician_k_db)
                h_ed = make_fading_gain(channel, current, device, config.rician_k_db)
                h_dd = make_fading_gain(channel, current, device, config.rician_k_db)
                h_de = make_fading_gain(channel, current, device, config.rician_k_db)
                noise_e = torch.randn_like(z_e)
                noise_d = torch.randn_like(z_d)

                target_e = h_ee * z_e
                interference_e = h_ed * z_d
                target_d = h_dd * z_d
                interference_d = h_de * z_e

                for alpha_index, alpha in enumerate(config.alpha_values):
                    received_e = add_awgn_from_target_snr(
                        target_e + alpha * interference_e,
                        target_e,
                        config.fixed_snr_db,
                        noise_e,
                    )
                    received_d = add_awgn_from_target_snr(
                        target_d + alpha * interference_d,
                        target_d,
                        config.fixed_snr_db,
                        noise_d,
                    )

                    # Reuse the same reverse-process initial seeds across alpha values.
                    base_sample_seed = (
                        config.seed
                        + channel_index * 10_000_000
                        + batch_index * 10
                    )
                    pred_e = recover_latent(model, received_e, label_e, base_sample_seed, config)
                    pred_d = recover_latent(model, received_d, label_d, base_sample_seed + 1, config)
                    recon_e = model.decoder(pred_e, label_e)
                    recon_d = model.decoder(pred_d, label_d)

                    add_metrics(
                        accumulators[alpha]["EuroSAT"],
                        recon_e,
                        oracle_e,
                        image_e,
                        pred_e,
                        z_e,
                    )
                    add_metrics(
                        accumulators[alpha]["DeepGlobe"],
                        recon_d,
                        oracle_d,
                        image_d,
                        pred_d,
                        z_d,
                    )

                    if (
                        not qualitative_saved[alpha]
                        and any(np.isclose(alpha, value) for value in config.qualitative_alphas)
                    ):
                        alpha_name = str(alpha).replace(".", "p")
                        save_qualitative_grid(
                            os.path.join(
                                config.save_root,
                                f"qualitative_{channel}_alpha{alpha_name}.png",
                            ),
                            image_e,
                            image_d,
                            oracle_e,
                            oracle_d,
                            recon_e,
                            recon_d,
                            config.qualitative_count,
                        )
                        qualitative_saved[alpha] = True

            print(
                f"{channel} | batch {batch_index + 1}/{len(loader_e)} complete "
                f"for {len(config.alpha_values)} alpha values"
            )

        channel_rows = build_summary_rows(accumulators, channel, config)
        all_rows.extend(channel_rows)
        write_csv(
            os.path.join(config.save_root, "interference_metrics_long.csv"),
            all_rows,
        )

        for alpha in config.alpha_values:
            avg_row = next(
                row
                for row in channel_rows
                if row["domain"] == "Average" and np.isclose(row["alpha"], alpha)
            )
            print(
                f"alpha={alpha:>4.2f} | PSNR={avg_row['psnr_real']:.3f} | "
                f"SSIM={avg_row['ssim_real']:.4f} | "
                f"LatCos={avg_row['latent_cosine']:.4f} | "
                f"LatMSE={avg_row['latent_mse']:.4f}"
            )

        safe_channel = channel.replace("-", "_")
        plot_metric(
            all_rows,
            channel,
            "psnr_real",
            "PSNR vs. real image (dB)",
            os.path.join(config.save_root, f"PSNR_vs_alpha_{safe_channel}.png"),
        )
        plot_metric(
            all_rows,
            channel,
            "ssim_real",
            "SSIM vs. real image",
            os.path.join(config.save_root, f"SSIM_vs_alpha_{safe_channel}.png"),
        )
        plot_metric(
            all_rows,
            channel,
            "latent_cosine",
            "Latent cosine similarity",
            os.path.join(config.save_root, f"Latent_Cosine_vs_alpha_{safe_channel}.png"),
        )
        plot_metric(
            all_rows,
            channel,
            "latent_mse",
            "Latent MSE",
            os.path.join(config.save_root, f"Latent_MSE_vs_alpha_{safe_channel}.png"),
        )
        save_latex_table(
            os.path.join(config.save_root, f"interference_summary_{safe_channel}.tex"),
            all_rows,
            channel,
            config,
        )

    # Compact average-only CSV for paper tables.
    average_rows = [row for row in all_rows if row["domain"] == "Average"]
    write_csv(
        os.path.join(config.save_root, "interference_summary.csv"),
        average_rows,
    )

    print("=" * 90)
    print("Interference-strength test completed.")
    print(f"Results: {config.save_root}")
    print(
        "Important: SNR is defined relative to the desired signal only; "
        "therefore alpha changes interference without changing target SNR."
    )


if __name__ == "__main__":
    main()
