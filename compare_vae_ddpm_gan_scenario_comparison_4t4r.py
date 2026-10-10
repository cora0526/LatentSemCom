"""
Evaluate the VAE-DDPM-GAN system under four transmission scenarios:

1. EuroSAT-1T1R:
       y = h_E z_E + n

2. DeepGlobe-1T1R:
       y = h_D z_D + n

3. Mixed-2T2R:
       y_r = h_{r,E} z_E + h_{r,D} z_D + n_r
   Two independent receivers recover one EuroSAT and one DeepGlobe source.

4. Mixed-4T4R-Diagnostic:
       y_r = h_{r,E1} z_E1 + h_{r,E2} z_E2
           + h_{r,D1} z_D1 + h_{r,D2} z_D2 + n_r
   Four independent receivers are evaluated: E1, E2, D1, and D2.

Important:
The current DDPM only conditions on the two domain labels EuroSAT/DeepGlobe.
Therefore, the 4T4R experiment is a diagnostic stress test, not a claim of true
same-domain user separation. E1 and E2 use the same EuroSAT condition, while D1
and D2 use the same DeepGlobe condition. Poor identity-specific recovery or
duplicate outputs are expected and are themselves useful experimental evidence.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

    # DDIM is much faster than using all DDPM reverse steps.
    sampling_method: str = "ddim"
    ddim_steps: int = 50
    ddim_eta: float = 0.0

    batch_size: int = 16
    num_workers: int = 0
    max_test_pairs: int = 32

    snr_values: Tuple[float, ...] = (0, 5, 10, 15, 20, 25, 30)
    channel_types: Tuple[str, ...] = (
        "AWGN",
        "Rayleigh",
        "Rician",
    )
    scenarios: Tuple[str, ...] = (
        "EuroSAT-1T1R",
        "DeepGlobe-1T1R",
        "Mixed-2T2R",
        "Mixed-4T4R-Diagnostic",
    )
    rician_k_db: float = 6.0

    eurosat_root: str = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    deepglobe_root: str = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    scratch_vaegan_checkpoint: str = (
        "./save_model/vae_gan_from_scratch_best.pth"
    )
    scratch_vaegan_ddpm_checkpoint: str = (
        "./save_model/DDPM_vaegan_scratch_latent_separator_best.pth"
    )

    save_root: str = "./save_data/VAE_DDPM_GAN_scenario_comparison"

    qualitative_snrs: Tuple[float, ...] = (0, 10, 20, 30)
    qualitative_count: int = 8


class SatelliteDataset(Dataset):
    """DeepGlobe dataset using the existing *_sat_*.jpg filename convention."""

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
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


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

    return ModelBundle(
        encoder=encoder,
        decoder=generator,
        ddpm=ddpm,
        schedule=schedule,
    )


def add_awgn(
    signal: torch.Tensor,
    snr_db: float,
    noise_unit: torch.Tensor,
) -> torch.Tensor:
    if math.isinf(float(snr_db)):
        return signal

    signal_power = signal.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12)
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_std = torch.sqrt(signal_power / snr_linear)
    return signal + noise_unit * noise_std


def make_fading_gain(
    channel_type: str,
    batch_size: int,
    device: str,
    rician_k_db: float,
) -> torch.Tensor:
    """Return one real-valued flat-fading magnitude per sample."""

    if channel_type == "AWGN":
        return torch.ones(batch_size, 1, device=device)

    if channel_type == "Rayleigh":
        real = torch.randn(batch_size, 1, device=device)
        imag = torch.randn(batch_size, 1, device=device)
        return torch.sqrt((real.pow(2) + imag.pow(2)) / 2.0).clamp_min(1e-6)

    if channel_type == "Rician":
        k_linear = 10.0 ** (float(rician_k_db) / 10.0)
        los = math.sqrt(k_linear / (k_linear + 1.0))
        scatter_scale = math.sqrt(1.0 / (2.0 * (k_linear + 1.0)))

        real = los + scatter_scale * torch.randn(batch_size, 1, device=device)
        imag = scatter_scale * torch.randn(batch_size, 1, device=device)
        return torch.sqrt(real.pow(2) + imag.pow(2)).clamp_min(1e-6)

    raise ValueError(f"Unknown channel type: {channel_type}")


def make_single_user_observation(
    latent: torch.Tensor,
    snr_db: float,
    channel_type: str,
    rician_k_db: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = latent.size(0)
    device = str(latent.device)

    h = make_fading_gain(
        channel_type=channel_type,
        batch_size=batch_size,
        device=device,
        rician_k_db=rician_k_db,
    )
    clean_signal = h * latent
    noise_unit = torch.randn_like(clean_signal)
    received = add_awgn(clean_signal, snr_db, noise_unit)
    return received, clean_signal, h


def make_two_user_receiver_observation(
    z_e: torch.Tensor,
    z_d: torch.Tensor,
    snr_db: float,
    channel_type: str,
    rician_k_db: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create one receiver observation of the two-user MAC."""

    batch_size = z_e.size(0)
    device = str(z_e.device)

    h_e = make_fading_gain(
        channel_type=channel_type,
        batch_size=batch_size,
        device=device,
        rician_k_db=rician_k_db,
    )
    h_d = make_fading_gain(
        channel_type=channel_type,
        batch_size=batch_size,
        device=device,
        rician_k_db=rician_k_db,
    )

    clean_signal = h_e * z_e + h_d * z_d
    noise_unit = torch.randn_like(clean_signal)
    received = add_awgn(clean_signal, snr_db, noise_unit)
    return received, clean_signal, h_e, h_d


def make_four_user_receiver_observation(
    z_e1: torch.Tensor,
    z_e2: torch.Tensor,
    z_d1: torch.Tensor,
    z_d2: torch.Tensor,
    snr_db: float,
    channel_type: str,
    rician_k_db: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Create one independent receiver observation of the four-user MAC."""

    batch_size = z_e1.size(0)
    device = str(z_e1.device)

    h_e1 = make_fading_gain(channel_type, batch_size, device, rician_k_db)
    h_e2 = make_fading_gain(channel_type, batch_size, device, rician_k_db)
    h_d1 = make_fading_gain(channel_type, batch_size, device, rician_k_db)
    h_d2 = make_fading_gain(channel_type, batch_size, device, rician_k_db)

    clean_signal = (
        h_e1 * z_e1
        + h_e2 * z_e2
        + h_d1 * z_d1
        + h_d2 * z_d2
    )
    noise_unit = torch.randn_like(clean_signal)
    received = add_awgn(clean_signal, snr_db, noise_unit)

    return received, clean_signal, h_e1, h_e2, h_d1, h_d2


def to_01(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().float() + 1.0) / 2.0).clamp(0.0, 1.0)


def batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred = to_01(pred)
    target = to_01(target)
    mse = (pred - target).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).cpu().numpy()


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred_np = to_01(pred).permute(0, 2, 3, 1).cpu().numpy()
    target_np = to_01(target).permute(0, 2, 3, 1).cpu().numpy()

    values = []
    for prediction, reference in zip(pred_np, target_np):
        values.append(
            structural_similarity(
                reference,
                prediction,
                data_range=1.0,
                channel_axis=-1,
            )
        )
    return np.asarray(values, dtype=np.float64)


def latent_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
    return (pred - target).pow(2).mean(dim=1).detach().cpu().numpy()


def latent_cosine_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> np.ndarray:
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
        if not self.values:
            return 0
        return len(next(iter(self.values.values())))


def add_metrics(
    accumulator: MetricAccumulator,
    separated: torch.Tensor,
    oracle: torch.Tensor,
    real: torch.Tensor,
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
) -> None:
    accumulator.add("psnr_real", batch_psnr(separated, real))
    accumulator.add("ssim_real", batch_ssim(separated, real))
    accumulator.add("psnr_oracle", batch_psnr(separated, oracle))
    accumulator.add("ssim_oracle", batch_ssim(separated, oracle))
    accumulator.add("oracle_psnr_real", batch_psnr(oracle, real))
    accumulator.add("oracle_ssim_real", batch_ssim(oracle, real))
    accumulator.add("latent_mse", latent_mse_per_sample(pred_z, target_z))
    accumulator.add("latent_cosine", latent_cosine_per_sample(pred_z, target_z))


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


@torch.no_grad()
def evaluate_scenario_batch(
    model: ModelBundle,
    image_e: torch.Tensor,
    image_d: torch.Tensor,
    scenario: str,
    channel_type: str,
    snr_db: float,
    sample_seed: int,
    config: Config,
) -> Dict[str, Optional[dict]]:
    batch_size = image_e.size(0)
    label_e = torch.zeros(batch_size, dtype=torch.long, device=image_e.device)
    label_d = torch.ones(batch_size, dtype=torch.long, device=image_e.device)

    z_e, _ = model.encoder(image_e)
    z_d, _ = model.encoder(image_d)

    oracle_e = model.decoder(z_e, label_e)
    oracle_d = model.decoder(z_d, label_d)

    result: Dict[str, Optional[dict]] = {}

    if scenario == "EuroSAT-1T1R":
        received_e, clean_e, h_e = make_single_user_observation(
            latent=z_e,
            snr_db=snr_db,
            channel_type=channel_type,
            rician_k_db=config.rician_k_db,
        )
        pred_e = recover_latent(
            model=model,
            received=received_e,
            label=label_e,
            seed=sample_seed,
            config=config,
        )
        separated_e = model.decoder(pred_e, label_e)
        result["E"] = {
            "real": image_e,
            "oracle": oracle_e,
            "target_z": z_e,
            "pred_z": pred_e,
            "received": received_e,
            "clean_signal": clean_e,
            "h_target": h_e,
            "separated": separated_e,
        }

    elif scenario == "DeepGlobe-1T1R":
        received_d, clean_d, h_d = make_single_user_observation(
            latent=z_d,
            snr_db=snr_db,
            channel_type=channel_type,
            rician_k_db=config.rician_k_db,
        )
        pred_d = recover_latent(
            model=model,
            received=received_d,
            label=label_d,
            seed=sample_seed,
            config=config,
        )
        separated_d = model.decoder(pred_d, label_d)
        result["D"] = {
            "real": image_d,
            "oracle": oracle_d,
            "target_z": z_d,
            "pred_z": pred_d,
            "received": received_d,
            "clean_signal": clean_d,
            "h_target": h_d,
            "separated": separated_d,
        }

    elif scenario == "Mixed-2T2R":
        # Receiver E and receiver D have independent channel/noise realizations.
        received_e, clean_e, h_ee, h_ed = make_two_user_receiver_observation(
            z_e=z_e,
            z_d=z_d,
            snr_db=snr_db,
            channel_type=channel_type,
            rician_k_db=config.rician_k_db,
        )
        received_d, clean_d, h_de, h_dd = make_two_user_receiver_observation(
            z_e=z_e,
            z_d=z_d,
            snr_db=snr_db,
            channel_type=channel_type,
            rician_k_db=config.rician_k_db,
        )

        pred_e = recover_latent(
            model=model,
            received=received_e,
            label=label_e,
            seed=sample_seed,
            config=config,
        )
        pred_d = recover_latent(
            model=model,
            received=received_d,
            label=label_d,
            seed=sample_seed + 1,
            config=config,
        )

        separated_e = model.decoder(pred_e, label_e)
        separated_d = model.decoder(pred_d, label_d)

        result["E"] = {
            "real": image_e,
            "oracle": oracle_e,
            "target_z": z_e,
            "pred_z": pred_e,
            "received": received_e,
            "clean_signal": clean_e,
            "h_target": h_ee,
            "h_interference": h_ed,
            "separated": separated_e,
        }
        result["D"] = {
            "real": image_d,
            "oracle": oracle_d,
            "target_z": z_d,
            "pred_z": pred_d,
            "received": received_d,
            "clean_signal": clean_d,
            "h_target": h_dd,
            "h_interference": h_de,
            "separated": separated_d,
        }

    elif scenario == "Mixed-4T4R-Diagnostic":
        if batch_size < 2:
            raise RuntimeError(
                "Mixed-4T4R-Diagnostic requires batch_size >= 2 so the second "
                "EuroSAT and DeepGlobe users can use different images."
            )

        # Create a second user from each domain by cyclically shifting the batch.
        # This preserves the dataset distribution while ensuring E1 != E2 and D1 != D2
        # for ordinary batches.
        image_e2 = torch.roll(image_e, shifts=1, dims=0)
        image_d2 = torch.roll(image_d, shifts=1, dims=0)
        z_e2, _ = model.encoder(image_e2)
        z_d2, _ = model.encoder(image_d2)
        oracle_e2 = model.decoder(z_e2, label_e)
        oracle_d2 = model.decoder(z_d2, label_d)

        receiver_specs = (
            ("E1", label_e, z_e, image_e, oracle_e, sample_seed),
            ("E2", label_e, z_e2, image_e2, oracle_e2, sample_seed + 1),
            ("D1", label_d, z_d, image_d, oracle_d, sample_seed + 2),
            ("D2", label_d, z_d2, image_d2, oracle_d2, sample_seed + 3),
        )

        for receiver_name, receiver_label, target_z, real_image, oracle, seed in receiver_specs:
            received, clean_signal, h_e1, h_e2, h_d1, h_d2 = (
                make_four_user_receiver_observation(
                    z_e1=z_e,
                    z_e2=z_e2,
                    z_d1=z_d,
                    z_d2=z_d2,
                    snr_db=snr_db,
                    channel_type=channel_type,
                    rician_k_db=config.rician_k_db,
                )
            )

            pred_z = recover_latent(
                model=model,
                received=received,
                label=receiver_label,
                seed=seed,
                config=config,
            )
            separated = model.decoder(pred_z, receiver_label)

            result[receiver_name] = {
                "real": real_image,
                "oracle": oracle,
                "target_z": target_z,
                "pred_z": pred_z,
                "received": received,
                "clean_signal": clean_signal,
                "h_e1": h_e1,
                "h_e2": h_e2,
                "h_d1": h_d1,
                "h_d2": h_d2,
                "separated": separated,
            }

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return result


def build_summary_row(
    scenario: str,
    domain: str,
    channel: str,
    snr_db: float,
    accumulator: MetricAccumulator,
) -> dict:
    return {
        "scenario": scenario,
        "domain": domain,
        "channel": channel,
        "snr_db": snr_db,
        "num_samples": accumulator.count(),
        "psnr_real": accumulator.mean("psnr_real"),
        "ssim_real": accumulator.mean("ssim_real"),
        "psnr_oracle": accumulator.mean("psnr_oracle"),
        "ssim_oracle": accumulator.mean("ssim_oracle"),
        "oracle_psnr_real": accumulator.mean("oracle_psnr_real"),
        "oracle_ssim_real": accumulator.mean("oracle_ssim_real"),
        "latent_mse": accumulator.mean("latent_mse"),
        "latent_cosine": accumulator.mean("latent_cosine"),
    }


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


SCENARIO_COLORS = {
    "EuroSAT-1T1R": "#4C72B0",
    "DeepGlobe-1T1R": "#55A868",
    "Mixed-2T2R": "#C44E52",
    "Mixed-4T4R-Diagnostic": "#8172B2",
}

SCENARIO_MARKERS = {
    "EuroSAT-1T1R": "o",
    "DeepGlobe-1T1R": "s",
    "Mixed-2T2R": "^",
    "Mixed-4T4R-Diagnostic": "D",
}


def scenario_metric_value(rows: List[dict], scenario: str, metric: str) -> float:
    """Average active receiver domains for the selected scenario/SNR/channel."""
    selected = [float(row[metric]) for row in rows if row["scenario"] == scenario]
    return float(np.mean(selected)) if selected else float("nan")


def plot_metric_by_channel(
    rows: List[dict],
    metric: str,
    ylabel: str,
    save_root: str,
    config: Config,
) -> None:
    for channel in config.channel_types:
        plt.figure(figsize=(8.5, 5.5))

        channel_rows = [row for row in rows if row["channel"] == channel]
        for scenario in config.scenarios:
            x_values = []
            y_values = []

            for snr_db in config.snr_values:
                point_rows = [
                    row
                    for row in channel_rows
                    if row["scenario"] == scenario
                    and float(row["snr_db"]) == float(snr_db)
                ]
                if not point_rows:
                    continue

                x_values.append(float(snr_db))
                y_values.append(
                    scenario_metric_value(point_rows, scenario, metric)
                )

            if x_values:
                plt.plot(
                    x_values,
                    y_values,
                    marker=SCENARIO_MARKERS[scenario],
                    linewidth=2,
                    markersize=6,
                    color=SCENARIO_COLORS[scenario],
                    label=scenario,
                )

        plt.xlabel("SNR (dB)")
        plt.ylabel(ylabel)
        plt.title(f"VAE-DDPM-GAN Scenario Comparison ({channel})")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()

        safe_channel = channel.replace("-", "_")
        filename = f"{metric}_vs_SNR_{safe_channel}.png"
        plt.savefig(os.path.join(save_root, filename), dpi=250)
        plt.close()


def save_qualitative_grid(
    save_path: str,
    scenario: str,
    result: Dict[str, Optional[dict]],
    count: int,
) -> None:
    rows: List[torch.Tensor] = []

    # Each active receiver contributes three rows: real, oracle, reconstruction.
    for domain in ("E", "D", "E1", "E2", "D1", "D2"):
        domain_result = result.get(domain)
        if domain_result is None:
            continue

        available = min(
            count,
            domain_result["real"].size(0),
            domain_result["oracle"].size(0),
            domain_result["separated"].size(0),
        )
        rows.extend(
            [
                domain_result["real"][:available],
                domain_result["oracle"][:available],
                domain_result["separated"][:available],
            ]
        )

    if not rows:
        return

    # All active domains use the same batch size in this script.
    nrow = min(count, rows[0].size(0))
    grid = make_grid(
        torch.cat(rows, dim=0),
        nrow=nrow,
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(grid, save_path)


def build_wide_summary(rows: List[dict], config: Config) -> List[dict]:
    wide_rows: List[dict] = []

    for channel in config.channel_types:
        for snr_db in config.snr_values:
            output = {
                "channel": channel,
                "snr_db": snr_db,
            }

            for scenario in config.scenarios:
                selected = [
                    row
                    for row in rows
                    if row["channel"] == channel
                    and row["scenario"] == scenario
                    and float(row["snr_db"]) == float(snr_db)
                ]
                if not selected:
                    continue

                prefix = scenario.replace("-", "_")
                for metric in (
                    "psnr_real",
                    "ssim_real",
                    "latent_mse",
                    "latent_cosine",
                ):
                    output[f"{prefix}_avg_{metric}"] = float(
                        np.mean([float(row[metric]) for row in selected])
                    )

            wide_rows.append(output)

    return wide_rows


def main() -> None:
    config = Config()

    if config.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; switching to CPU.")
        device = "cpu"
    else:
        device = config.device

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    os.makedirs(config.save_root, exist_ok=True)

    print(
        f"DDPM sampler: {config.sampling_method}, "
        f"steps={config.ddim_steps}, eta={config.ddim_eta}"
    )

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    eurosat = datasets.ImageFolder(config.eurosat_root, transform=transform)
    deepglobe = SatelliteDataset(config.deepglobe_root, transform=transform)

    pair_count = min(config.max_test_pairs, len(eurosat), len(deepglobe))
    if pair_count <= 0:
        raise RuntimeError("No test pairs are available.")

    subset_generator = torch.Generator().manual_seed(config.seed)
    e_indices = torch.randperm(
        len(eurosat), generator=subset_generator
    )[:pair_count].tolist()
    d_indices = torch.randperm(
        len(deepglobe), generator=subset_generator
    )[:pair_count].tolist()

    loader_e = DataLoader(
        Subset(eurosat, e_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
    )
    loader_d = DataLoader(
        Subset(deepglobe, d_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device == "cuda"),
    )

    print("Loading VAE-DDPM-GAN...")
    model = load_model(config, device)

    rows: List[dict] = []

    for scenario_index, scenario in enumerate(config.scenarios):
        for channel_index, channel_type in enumerate(config.channel_types):
            for snr_index, snr_db in enumerate(config.snr_values):
                print("=" * 90)
                print(
                    f"Scenario={scenario} | channel={channel_type} | "
                    f"SNR={snr_db} dB"
                )

                accumulators = {
                    name: MetricAccumulator()
                    for name in ("E", "D", "E1", "E2", "D1", "D2")
                }
                qualitative_saved = False

                for batch_index, (batch_e, batch_d) in enumerate(
                    zip(loader_e, loader_d)
                ):
                    image_e = get_image(batch_e).to(device, non_blocking=True)
                    image_d = get_image(batch_d).to(device, non_blocking=True)
                    current = min(image_e.size(0), image_d.size(0))
                    image_e = image_e[:current]
                    image_d = image_d[:current]

                    # Reproducible, but different, realization for every condition.
                    condition_seed = (
                        config.seed
                        + scenario_index * 100_000_000
                        + channel_index * 1_000_000
                        + snr_index * 10_000
                        + batch_index
                    )
                    torch.manual_seed(condition_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(condition_seed)

                    sample_seed = (
                        config.seed
                        + scenario_index * 10_000_000
                        + channel_index * 100_000
                        + snr_index * 1_000
                        + batch_index * 10
                    )

                    result = evaluate_scenario_batch(
                        model=model,
                        image_e=image_e,
                        image_d=image_d,
                        scenario=scenario,
                        channel_type=channel_type,
                        snr_db=snr_db,
                        sample_seed=sample_seed,
                        config=config,
                    )

                    for domain in ("E", "D", "E1", "E2", "D1", "D2"):
                        domain_result = result.get(domain)
                        if domain_result is None:
                            continue

                        add_metrics(
                            accumulator=accumulators[domain],
                            separated=domain_result["separated"],
                            oracle=domain_result["oracle"],
                            real=domain_result["real"],
                            pred_z=domain_result["pred_z"],
                            target_z=domain_result["target_z"],
                        )

                    if (
                        not qualitative_saved
                        and float(snr_db)
                        in [float(value) for value in config.qualitative_snrs]
                    ):
                        safe_scenario = scenario.replace("-", "_")
                        safe_channel = channel_type.replace("-", "_")
                        filename = (
                            f"qualitative_{safe_scenario}_{safe_channel}_"
                            f"SNR{snr_db:g}.png"
                        )
                        save_qualitative_grid(
                            save_path=os.path.join(config.save_root, filename),
                            scenario=scenario,
                            result=result,
                            count=config.qualitative_count,
                        )
                        qualitative_saved = True

                    print(
                        f"{scenario} | {channel_type} | SNR={snr_db:>4} | "
                        f"batch {batch_index + 1}/{len(loader_e)} complete"
                    )

                for domain in ("E", "D", "E1", "E2", "D1", "D2"):
                    accumulator = accumulators[domain]
                    if accumulator.count() == 0:
                        continue

                    row = build_summary_row(
                        scenario=scenario,
                        domain=domain,
                        channel=channel_type,
                        snr_db=snr_db,
                        accumulator=accumulator,
                    )
                    rows.append(row)

                    print(
                        f"{scenario} | {domain} | {channel_type}: "
                        f"PSNR={row['psnr_real']:.3f}, "
                        f"SSIM={row['ssim_real']:.4f}, "
                        f"LatCos={row['latent_cosine']:.4f}, "
                        f"LatMSE={row['latent_mse']:.4f}"
                    )

                # Save partial progress after every condition.
                write_csv(
                    os.path.join(config.save_root, "metrics_long.csv"),
                    rows,
                )

    wide_rows = build_wide_summary(rows, config)
    write_csv(
        os.path.join(config.save_root, "comparison_summary.csv"),
        wide_rows,
    )

    plot_metric_by_channel(
        rows=rows,
        metric="psnr_real",
        ylabel="Average PSNR vs. real image (dB)",
        save_root=config.save_root,
        config=config,
    )
    plot_metric_by_channel(
        rows=rows,
        metric="ssim_real",
        ylabel="Average SSIM vs. real image",
        save_root=config.save_root,
        config=config,
    )
    plot_metric_by_channel(
        rows=rows,
        metric="latent_cosine",
        ylabel="Average latent cosine similarity",
        save_root=config.save_root,
        config=config,
    )
    plot_metric_by_channel(
        rows=rows,
        metric="latent_mse",
        ylabel="Average latent MSE",
        save_root=config.save_root,
        config=config,
    )

    print("=" * 90)
    print("Scenario comparison completed.")
    print(f"Results: {config.save_root}")
    print("Blue:   EuroSAT-1T1R")
    print("Green:  DeepGlobe-1T1R")
    print("Red:    Mixed-2T2R")
    print("Purple: Mixed-4T4R-Diagnostic")
    print(
        "Mixed-2T2R uses two independent receivers; each receives an "
        "independently faded/noisy EuroSAT + DeepGlobe mixture."
    )
    print(
        "Mixed-4T4R-Diagnostic is a stress test only: E1/E2 share label 0 and "
        "D1/D2 share label 1, so the current DDPM cannot explicitly identify "
        "same-domain users."
    )
    print(
        f"DDPM sampler: {config.sampling_method}, "
        f"steps={config.ddim_steps}, eta={config.ddim_eta}"
    )
    print("For final paper results, increase Config.max_test_pairs.")


if __name__ == "__main__":
    main()
