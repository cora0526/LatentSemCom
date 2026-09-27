"""
Preliminary observation of inter-user semantic interference.

Purpose
-------
This experiment is intended for the System Model chapter.

It compares semantic transmission:

1) No Interference
       encoder -> target latent + AWGN -> conditional generator

2) With Interference
       encoder -> target latent + interfering latent + AWGN
       -> conditional generator

IMPORTANT:
- No DDPM separator is used.
- No interference cancellation method is used.
- The purpose is only to demonstrate the degradation caused by
  inter-user semantic interference.
- EuroSAT and DeepGlobe are evaluated separately.
- This script uses a NEW output directory and NEW filenames, so it
  will NOT overwrite the previous DDPM ablation experiment.

Outputs
-------
./save_data/preliminary_interference_observation/

    preliminary_interference_observation.csv

    Preliminary_EuroSAT_PSNR_vs_SNR.png
    Preliminary_EuroSAT_PSNR_vs_SNR.pdf

    Preliminary_DeepGlobe_PSNR_vs_SNR.png
    Preliminary_DeepGlobe_PSNR_vs_SNR.pdf

    Preliminary_EuroSAT_SSIM_vs_SNR.png
    Preliminary_EuroSAT_SSIM_vs_SNR.pdf

    Preliminary_DeepGlobe_SSIM_vs_SNR.png
    Preliminary_DeepGlobe_SSIM_vs_SNR.pdf

    Preliminary_qualitative_SNR0.png
    Preliminary_qualitative_SNR10.png
    Preliminary_qualitative_SNR20.png
    Preliminary_qualitative_SNR30.png
"""

# =========================================================
# Imports
# =========================================================

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

from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


# =========================================================
# Configuration
# =========================================================

@dataclass
class Config:

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------
    seed: int = 42

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------
    device: str = "cuda"

    # -----------------------------------------------------
    # Semantic model
    # -----------------------------------------------------
    z_dim: int = 256
    label_dim: int = 64
    num_classes: int = 2

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
    batch_size: int = 16
    num_workers: int = 0

    # Number of paired EuroSAT / DeepGlobe test samples.
    # Increase if needed for final thesis results.
    max_test_pairs: int = 500

    # SNR sweep
    snr_values: Tuple[float, ...] = (
        0,
        5,
        10,
        15,
        20,
        25,
        30,
    )

    # -----------------------------------------------------
    # Dataset paths
    # -----------------------------------------------------
    eurosat_root: str = (
        r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    )

    deepglobe_root: str = (
        r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    )

    # -----------------------------------------------------
    # VAE-GAN checkpoint
    # -----------------------------------------------------
    scratch_vaegan_checkpoint: str = (
        "./save_model/vae_gan_from_scratch_best.pth"
    )

    # -----------------------------------------------------
    # NEW output folder
    #
    # Different from the previous DDPM experiment.
    # Therefore old files will NOT be overwritten.
    # -----------------------------------------------------
    save_root: str = (
        "./save_data/preliminary_interference_observation"
    )

    # -----------------------------------------------------
    # Qualitative reconstruction examples
    # -----------------------------------------------------
    qualitative_snrs: Tuple[float, ...] = (
        0,
        10,
        20,
        30,
    )

    qualitative_count: int = 8


# =========================================================
# Dataset
# =========================================================

class SatelliteDataset(Dataset):
    """
    DeepGlobe dataset loader.

    The original dataset uses filenames containing "_sat_".
    """

    def __init__(
        self,
        root: str,
        transform=None,
    ):
        self.files = sorted(
            glob.glob(
                os.path.join(
                    root,
                    "*_sat_*.jpg",
                )
            )
        )

        self.transform = transform

        if not self.files:
            raise RuntimeError(
                f"No matching DeepGlobe images found in:\n{root}"
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):

        image = Image.open(
            self.files[index]
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


def get_image(batch):
    """
    ImageFolder returns (image, label), while SatelliteDataset
    directly returns image.
    """

    if isinstance(batch, (tuple, list)):
        return batch[0]

    return batch


# =========================================================
# Semantic Encoder
# =========================================================

class SharedVAEEncoder(nn.Module):

    def __init__(
        self,
        z_dim: int = 256,
    ):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                4,
                2,
                1,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            nn.Conv2d(
                64,
                128,
                4,
                2,
                1,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            nn.Conv2d(
                128,
                256,
                4,
                2,
                1,
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            nn.Conv2d(
                256,
                512,
                4,
                2,
                1,
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),
        )

        feature_dim = 512 * 4 * 4

        self.fc_mu = nn.Linear(
            feature_dim,
            z_dim,
        )

        self.fc_logvar = nn.Linear(
            feature_dim,
            z_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        h = self.features(
            x
        ).flatten(
            start_dim=1
        )

        mu = self.fc_mu(h)

        logvar = torch.clamp(
            self.fc_logvar(h),
            -10.0,
            10.0,
        )

        return mu, logvar


# =========================================================
# Conditional Generator
# =========================================================

class ScratchConditionalGenerator(nn.Module):

    def __init__(
        self,
        z_dim: int = 256,
        num_classes: int = 2,
        label_dim: int = 64,
    ):
        super().__init__()

        self.label_embedding = nn.Embedding(
            num_classes,
            label_dim,
        )

        self.fc = nn.Sequential(

            nn.Linear(
                z_dim + label_dim,
                512 * 4 * 4,
            ),

            nn.BatchNorm1d(
                512 * 4 * 4
            ),

            nn.ReLU(
                inplace=True
            ),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                512,
                256,
                4,
                2,
                1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                256,
                128,
                4,
                2,
                1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64,
                3,
                4,
                2,
                1,
            ),

            nn.Tanh(),
        )

    def forward(
        self,
        z: torch.Tensor,
        labels: torch.Tensor,
    ):

        label_feature = self.label_embedding(
            labels
        )

        h = self.fc(
            torch.cat(
                [
                    z,
                    label_feature,
                ],
                dim=1,
            )
        )

        h = h.view(
            z.size(0),
            512,
            4,
            4,
        )

        return self.decoder(h)


# =========================================================
# Utility Functions
# =========================================================

def freeze(
    model: nn.Module,
):
    """
    Freeze a pretrained model for inference.
    """

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False


def require_file(
    path: str,
):
    """
    Check whether a required checkpoint exists.
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required checkpoint does not exist:\n{path}"
        )


# =========================================================
# Load Semantic Models
# =========================================================

def load_semantic_models(
    config: Config,
    device: str,
):
    """
    Only load the VAE encoder and conditional GAN generator.

    IMPORTANT:
    DDPM is intentionally NOT loaded because this experiment
    is only a preliminary observation of interference.
    """

    require_file(
        config.scratch_vaegan_checkpoint
    )

    checkpoint = torch.load(
        config.scratch_vaegan_checkpoint,
        map_location=device,
    )

    if (
        "encoder" not in checkpoint
        or "generator" not in checkpoint
    ):
        raise KeyError(
            "VAE-GAN checkpoint must contain "
            "'encoder' and 'generator'."
        )

    # -----------------------------------------------------
    # Encoder
    # -----------------------------------------------------

    encoder = SharedVAEEncoder(
        z_dim=config.z_dim,
    ).to(device)

    encoder.load_state_dict(
        checkpoint["encoder"],
        strict=True,
    )

    # -----------------------------------------------------
    # Generator
    # -----------------------------------------------------

    generator = ScratchConditionalGenerator(
        z_dim=config.z_dim,
        num_classes=config.num_classes,
        label_dim=config.label_dim,
    ).to(device)

    generator.load_state_dict(
        checkpoint["generator"],
        strict=True,
    )

    # -----------------------------------------------------
    # Freeze
    # -----------------------------------------------------

    freeze(encoder)
    freeze(generator)

    print(
        "Loaded VAE-GAN epoch =",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    return encoder, generator


# =========================================================
# AWGN Channel
# =========================================================

def add_awgn(
    signal: torch.Tensor,
    snr_db: float,
    noise_unit: torch.Tensor,
):
    """
    Add AWGN according to the instantaneous average power
    of the input latent signal.

    noise_unit is supplied externally so that the same
    normalized Gaussian noise realization can be reused.
    """

    if math.isinf(
        float(snr_db)
    ):
        return signal

    signal_power = (
        signal
        .pow(2)
        .mean(
            dim=1,
            keepdim=True,
        )
        .clamp_min(1e-12)
    )

    snr_linear = (
        10.0
        ** (
            float(snr_db)
            / 10.0
        )
    )

    noise_std = torch.sqrt(
        signal_power
        / snr_linear
    )

    return (
        signal
        + noise_unit * noise_std
    )


# =========================================================
# Metrics
# =========================================================

def to_01(
    images: torch.Tensor,
):
    """
    Convert [-1, 1] images to [0, 1].
    """

    return (
        (
            images.detach().float()
            + 1.0
        )
        / 2.0
    ).clamp(
        0.0,
        1.0,
    )


def batch_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
):

    pred = to_01(pred)
    target = to_01(target)

    mse = (
        (pred - target)
        .pow(2)
        .flatten(1)
        .mean(dim=1)
        .clamp_min(1e-12)
    )

    psnr = (
        10.0
        * torch.log10(
            1.0 / mse
        )
    )

    return (
        psnr
        .cpu()
        .numpy()
    )


def batch_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
):

    pred_np = (
        to_01(pred)
        .permute(
            0,
            2,
            3,
            1,
        )
        .cpu()
        .numpy()
    )

    target_np = (
        to_01(target)
        .permute(
            0,
            2,
            3,
            1,
        )
        .cpu()
        .numpy()
    )

    values = []

    for prediction, reference in zip(
        pred_np,
        target_np,
    ):

        value = structural_similarity(
            reference,
            prediction,
            data_range=1.0,
            channel_axis=-1,
        )

        values.append(value)

    return np.asarray(
        values,
        dtype=np.float64,
    )


# =========================================================
# Metric Accumulator
# =========================================================

class MetricAccumulator:

    def __init__(self):

        self.values: Dict[
            str,
            List[float],
        ] = {}

    def add(
        self,
        name: str,
        values,
    ):

        array = np.asarray(
            values,
            dtype=np.float64,
        ).reshape(-1)

        self.values.setdefault(
            name,
            [],
        ).extend(
            array.tolist()
        )

    def mean(
        self,
        name: str,
    ):

        values = self.values.get(
            name,
            [],
        )

        if not values:
            return float("nan")

        return float(
            np.mean(values)
        )

    def count(self):

        if not self.values:
            return 0

        return len(
            next(
                iter(
                    self.values.values()
                )
            )
        )


# =========================================================
# Preliminary Interference Experiment
# =========================================================

@torch.no_grad()
def evaluate_batch(
    encoder: nn.Module,
    generator: nn.Module,
    image_e: torch.Tensor,
    image_d: torch.Tensor,
    snr_db: float,
    noise_unit: torch.Tensor,
):
    """
    Preliminary observation.

    E = EuroSAT
        label = 0

    D = DeepGlobe
        label = 1

    -------------------------------------------------------
    Case 1: No Interference

        y_E = z_E + n
        y_D = z_D + n

    -------------------------------------------------------
    Case 2: With Interference

        y = z_E + z_D + n

    -------------------------------------------------------

    No separator, DDPM, ZF, MMSE, SIC, or other
    interference cancellation technique is used.
    """

    batch_size = image_e.size(0)

    # -----------------------------------------------------
    # Dataset conditions
    # -----------------------------------------------------

    label_e = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=image_e.device,
    )

    label_d = torch.ones(
        batch_size,
        dtype=torch.long,
        device=image_e.device,
    )

    # -----------------------------------------------------
    # Semantic Encoding
    #
    # Use the VAE latent mean as the transmitted semantic
    # representation, consistent with the previous script.
    # -----------------------------------------------------

    z_e, _ = encoder(
        image_e
    )

    z_d, _ = encoder(
        image_d
    )

    # =====================================================
    # CASE 1
    # No Interference
    # =====================================================

    # -----------------------------------------------------
    # EuroSAT transmission
    #
    # y_E = z_E + n
    # -----------------------------------------------------

    received_no_interference_e = add_awgn(
        signal=z_e,
        snr_db=snr_db,
        noise_unit=noise_unit,
    )

    # -----------------------------------------------------
    # DeepGlobe transmission
    #
    # y_D = z_D + n
    # -----------------------------------------------------

    received_no_interference_d = add_awgn(
        signal=z_d,
        snr_db=snr_db,
        noise_unit=noise_unit,
    )

    # -----------------------------------------------------
    # Semantic reconstruction
    # -----------------------------------------------------

    recon_no_interference_e = generator(
        received_no_interference_e,
        label_e,
    )

    recon_no_interference_d = generator(
        received_no_interference_d,
        label_d,
    )

    # =====================================================
    # CASE 2
    # With Interference
    # =====================================================

    # -----------------------------------------------------
    # Two-user semantic superposition
    #
    # y = z_E + z_D + n
    #
    # alpha = 1 in this preliminary experiment.
    # -----------------------------------------------------

    latent_mixture = (
        z_e
        + z_d
    )

    received_with_interference = add_awgn(
        signal=latent_mixture,
        snr_db=snr_db,
        noise_unit=noise_unit,
    )

    # -----------------------------------------------------
    # Direct reconstruction from the interfered latent.
    #
    # No separator is applied.
    # -----------------------------------------------------

    recon_with_interference_e = generator(
        received_with_interference,
        label_e,
    )

    recon_with_interference_d = generator(
        received_with_interference,
        label_d,
    )

    return {

        "z_e":
            z_e,

        "z_d":
            z_d,

        "received_no_interference_e":
            received_no_interference_e,

        "received_no_interference_d":
            received_no_interference_d,

        "received_with_interference":
            received_with_interference,

        "no_interference_e":
            recon_no_interference_e,

        "no_interference_d":
            recon_no_interference_d,

        "with_interference_e":
            recon_with_interference_e,

        "with_interference_d":
            recon_with_interference_d,
    }


# =========================================================
# Add Metrics
# =========================================================

def add_reconstruction_metrics(
    accumulator: MetricAccumulator,
    prefix: str,
    reconstruction: torch.Tensor,
    real: torch.Tensor,
):

    accumulator.add(
        f"{prefix}_psnr",
        batch_psnr(
            reconstruction,
            real,
        ),
    )

    accumulator.add(
        f"{prefix}_ssim",
        batch_ssim(
            reconstruction,
            real,
        ),
    )


# =========================================================
# CSV
# =========================================================

def write_csv(
    path: str,
    rows: List[dict],
):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(rows)


# =========================================================
# Qualitative Reconstruction Figure
# =========================================================

def save_qualitative_grid(
    save_path: str,
    real_e: torch.Tensor,
    real_d: torch.Tensor,
    result: dict,
    count: int,
):
    """
    Rows:

    EuroSAT:
        Original
        No Interference
        With Interference

    DeepGlobe:
        Original
        No Interference
        With Interference
    """

    count = min(
        count,
        real_e.size(0),
        real_d.size(0),
    )

    rows = [

        # -------------------------------------------------
        # EuroSAT
        # -------------------------------------------------

        real_e[:count],

        result[
            "no_interference_e"
        ][:count],

        result[
            "with_interference_e"
        ][:count],

        # -------------------------------------------------
        # DeepGlobe
        # -------------------------------------------------

        real_d[:count],

        result[
            "no_interference_d"
        ][:count],

        result[
            "with_interference_d"
        ][:count],
    ]

    grid = make_grid(
        torch.cat(
            rows,
            dim=0,
        ),
        nrow=count,
        padding=2,
        normalize=True,
        value_range=(-1, 1),
    )

    save_image(
        grid,
        save_path,
    )


# =========================================================
# Plot Style
# =========================================================

SCENARIO_COLORS = {

    "No Interference":
        "#6C63B8",

    "With Interference":
        "#D97757",
}


SCENARIO_STYLES = {

    "No Interference":
        "-",

    "With Interference":
        "--",
}


SCENARIO_MARKERS = {

    "No Interference":
        "o",

    "With Interference":
        "D",
}


SCENARIO_ORDER = (

    "No Interference",

    "With Interference",
)


# =========================================================
# Plotting
# =========================================================

def plot_dataset_metric(
    rows: List[dict],
    dataset_name: str,
    metric_name: str,
    ylabel: str,
    save_root: str,
):
    """
    Plot preliminary interference comparison.

    metric_name:
        "psnr"
        "ssim"
    """

    if dataset_name == "EuroSAT":
        dataset_key = "eurosat"
    else:
        dataset_key = "deepglobe"

    # -----------------------------------------------------
    # CSV field mapping
    # -----------------------------------------------------

    field_map = {

        "No Interference":
            (
                f"{dataset_key}"
                f"_no_interference"
                f"_{metric_name}"
            ),

        "With Interference":
            (
                f"{dataset_key}"
                f"_with_interference"
                f"_{metric_name}"
            ),
    }

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5.2)
    )

    # -----------------------------------------------------
    # X axis
    # -----------------------------------------------------

    x = [
        float(
            row["snr_db"]
        )
        for row in rows
    ]

    # -----------------------------------------------------
    # Curves
    # -----------------------------------------------------

    for scenario in SCENARIO_ORDER:

        field = field_map[
            scenario
        ]

        y = [
            float(
                row[field]
            )
            for row in rows
        ]

        ax.plot(
            x,
            y,

            color=SCENARIO_COLORS[
                scenario
            ],

            linestyle=SCENARIO_STYLES[
                scenario
            ],

            marker=SCENARIO_MARKERS[
                scenario
            ],

            markersize=6,

            linewidth=2.2,

            label=scenario,
        )

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    ax.set_xlabel(
        "SNR (dB)",
        fontsize=18,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=18,
    )

    # -----------------------------------------------------
    # Tick style
    # -----------------------------------------------------

    ax.tick_params(
        axis="both",
        labelsize=16,
    )

    # -----------------------------------------------------
    # Grid
    # -----------------------------------------------------

    ax.grid(
        True,
        alpha=0.3,
    )

    # -----------------------------------------------------
    # Same scales for EuroSAT and DeepGlobe
    # -----------------------------------------------------

    if metric_name == "psnr":

        ax.set_ylim(
            12.0,
            30.5,
        )

        ax.set_yticks(
            [
                12,
                15,
                18,
                21,
                24,
                27,
                30,
            ]
        )

    elif metric_name == "ssim":

        ax.set_ylim(
            0.0,
            1.0,
        )

        ax.set_yticks(
            np.arange(
                0.0,
                1.01,
                0.2,
            )
        )

    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------

    ax.legend(
        fontsize=16,
        frameon=True,
        loc="best",
    )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    fig.tight_layout()

    # -----------------------------------------------------
    # NEW filenames
    # -----------------------------------------------------

    base_name = (
        f"Preliminary_"
        f"{dataset_name}_"
        f"{metric_name.upper()}_vs_SNR"
    )

    png_path = os.path.join(
        save_root,
        base_name + ".png",
    )

    pdf_path = os.path.join(
        save_root,
        base_name + ".pdf",
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        png_path,
    )

    print(
        "Saved:",
        pdf_path,
    )


# =========================================================
# Main
# =========================================================

def main():

    config = Config()

    # =====================================================
    # Device
    # =====================================================

    if (
        config.device == "cuda"
        and not torch.cuda.is_available()
    ):

        print(
            "CUDA unavailable; switching to CPU."
        )

        device = "cpu"

    else:

        device = config.device

    print(
        "Using device:",
        device,
    )

    # =====================================================
    # Reproducibility
    # =====================================================

    random.seed(
        config.seed
    )

    np.random.seed(
        config.seed
    )

    torch.manual_seed(
        config.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            config.seed
        )

    # =====================================================
    # NEW output directory
    # =====================================================

    os.makedirs(
        config.save_root,
        exist_ok=True,
    )

    print("=" * 90)

    print(
        "Preliminary Interference Observation"
    )

    print(
        "No DDPM / separator / cancellation is used."
    )

    print(
        "Output directory:"
    )

    print(
        config.save_root
    )

    print("=" * 90)

    # =====================================================
    # Image Transform
    # =====================================================

    transform = transforms.Compose(
        [
            transforms.Resize(
                (64, 64)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5),
            ),
        ]
    )

    # =====================================================
    # EuroSAT
    # =====================================================

    eurosat = datasets.ImageFolder(
        config.eurosat_root,
        transform=transform,
    )

    # =====================================================
    # DeepGlobe
    # =====================================================

    deepglobe = SatelliteDataset(
        config.deepglobe_root,
        transform=transform,
    )

    # =====================================================
    # Number of paired test samples
    # =====================================================

    pair_count = min(
        config.max_test_pairs,
        len(eurosat),
        len(deepglobe),
    )

    if pair_count <= 0:

        raise RuntimeError(
            "No test pairs are available."
        )

    print(
        f"Number of paired test samples: {pair_count}"
    )

    # =====================================================
    # Fixed test subset
    #
    # The same EuroSAT / DeepGlobe pairs are used for
    # every SNR value.
    # =====================================================

    subset_generator = (
        torch.Generator()
        .manual_seed(
            config.seed
        )
    )

    e_indices = torch.randperm(
        len(eurosat),
        generator=subset_generator,
    )[:pair_count].tolist()

    d_indices = torch.randperm(
        len(deepglobe),
        generator=subset_generator,
    )[:pair_count].tolist()

    # =====================================================
    # DataLoaders
    # =====================================================

    loader_e = DataLoader(

        Subset(
            eurosat,
            e_indices,
        ),

        batch_size=config.batch_size,

        shuffle=False,

        num_workers=config.num_workers,

        pin_memory=(
            device == "cuda"
        ),
    )

    loader_d = DataLoader(

        Subset(
            deepglobe,
            d_indices,
        ),

        batch_size=config.batch_size,

        shuffle=False,

        num_workers=config.num_workers,

        pin_memory=(
            device == "cuda"
        ),
    )

    # =====================================================
    # Load semantic encoder + generator only
    # =====================================================

    encoder, generator = (
        load_semantic_models(
            config,
            device,
        )
    )

    # =====================================================
    # Results
    # =====================================================

    rows: List[dict] = []

    # =====================================================
    # SNR Sweep
    # =====================================================

    for snr_index, snr_db in enumerate(
        config.snr_values
    ):

        print("=" * 90)

        print(
            f"Evaluating SNR = {snr_db} dB"
        )

        print(
            "Case 1: No Interference"
        )

        print(
            "Case 2: With Interference"
        )

        acc = MetricAccumulator()

        qualitative_saved = False

        # =================================================
        # Batch evaluation
        # =================================================

        for batch_index, (
            batch_e,
            batch_d,
        ) in enumerate(
            zip(
                loader_e,
                loader_d,
            )
        ):

            # ---------------------------------------------
            # Images
            # ---------------------------------------------

            image_e = (
                get_image(
                    batch_e
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            image_d = (
                get_image(
                    batch_d
                )
                .to(
                    device,
                    non_blocking=True,
                )
            )

            current = min(
                image_e.size(0),
                image_d.size(0),
            )

            image_e = (
                image_e[:current]
            )

            image_d = (
                image_d[:current]
            )

            # ---------------------------------------------
            # Fixed AWGN realization
            #
            # The same normalized Gaussian noise is used
            # for no-interference and interference cases.
            # ---------------------------------------------

            channel_seed = (
                config.seed
                + snr_index * 100_000
                + batch_index
            )

            torch.manual_seed(
                channel_seed
            )

            if torch.cuda.is_available():

                torch.cuda.manual_seed_all(
                    channel_seed
                )

            noise_unit = torch.randn(
                current,
                config.z_dim,
                device=device,
            )

            # ---------------------------------------------
            # Evaluation
            # ---------------------------------------------

            result = evaluate_batch(

                encoder=encoder,

                generator=generator,

                image_e=image_e,

                image_d=image_d,

                snr_db=snr_db,

                noise_unit=noise_unit,
            )

            # =================================================
            # EuroSAT
            # =================================================

            add_reconstruction_metrics(

                acc,

                "eurosat_no_interference",

                result[
                    "no_interference_e"
                ],

                image_e,
            )

            add_reconstruction_metrics(

                acc,

                "eurosat_with_interference",

                result[
                    "with_interference_e"
                ],

                image_e,
            )

            # =================================================
            # DeepGlobe
            # =================================================

            add_reconstruction_metrics(

                acc,

                "deepglobe_no_interference",

                result[
                    "no_interference_d"
                ],

                image_d,
            )

            add_reconstruction_metrics(

                acc,

                "deepglobe_with_interference",

                result[
                    "with_interference_d"
                ],

                image_d,
            )

            # =================================================
            # Qualitative Example
            # =================================================

            if (
                not qualitative_saved
                and float(snr_db)
                in [
                    float(value)
                    for value
                    in config.qualitative_snrs
                ]
            ):

                filename = (
                    "Preliminary_qualitative_"
                    f"SNR{snr_db:g}.png"
                )

                save_qualitative_grid(

                    os.path.join(
                        config.save_root,
                        filename,
                    ),

                    image_e,

                    image_d,

                    result,

                    config.qualitative_count,
                )

                qualitative_saved = True

            # ---------------------------------------------
            # Progress
            # ---------------------------------------------

            print(
                f"SNR={snr_db:>4} dB | "
                f"batch "
                f"{batch_index + 1}/"
                f"{min(len(loader_e), len(loader_d))} "
                f"complete"
            )

        # =================================================
        # Average Results
        # =================================================

        row = {

            "snr_db":
                snr_db,

            "num_pairs":
                acc.count(),

            # ---------------------------------------------
            # EuroSAT PSNR
            # ---------------------------------------------

            "eurosat_no_interference_psnr":
                acc.mean(
                    "eurosat_no_interference_psnr"
                ),

            "eurosat_with_interference_psnr":
                acc.mean(
                    "eurosat_with_interference_psnr"
                ),

            # ---------------------------------------------
            # EuroSAT SSIM
            # ---------------------------------------------

            "eurosat_no_interference_ssim":
                acc.mean(
                    "eurosat_no_interference_ssim"
                ),

            "eurosat_with_interference_ssim":
                acc.mean(
                    "eurosat_with_interference_ssim"
                ),

            # ---------------------------------------------
            # DeepGlobe PSNR
            # ---------------------------------------------

            "deepglobe_no_interference_psnr":
                acc.mean(
                    "deepglobe_no_interference_psnr"
                ),

            "deepglobe_with_interference_psnr":
                acc.mean(
                    "deepglobe_with_interference_psnr"
                ),

            # ---------------------------------------------
            # DeepGlobe SSIM
            # ---------------------------------------------

            "deepglobe_no_interference_ssim":
                acc.mean(
                    "deepglobe_no_interference_ssim"
                ),

            "deepglobe_with_interference_ssim":
                acc.mean(
                    "deepglobe_with_interference_ssim"
                ),
        }

        rows.append(
            row
        )

        # =================================================
        # Save CSV progressively
        #
        # NEW filename -> old experiment is untouched.
        # =================================================

        csv_path = os.path.join(

            config.save_root,

            "preliminary_interference_observation.csv",
        )

        write_csv(
            csv_path,
            rows,
        )

        # =================================================
        # Print Results
        # =================================================

        print("-" * 90)

        print(
            "EuroSAT:"
        )

        print(
            f"  No Interference   = "
            f"{row['eurosat_no_interference_psnr']:.3f} dB"
        )

        print(
            f"  With Interference = "
            f"{row['eurosat_with_interference_psnr']:.3f} dB"
        )

        print()

        print(
            "DeepGlobe:"
        )

        print(
            f"  No Interference   = "
            f"{row['deepglobe_no_interference_psnr']:.3f} dB"
        )

        print(
            f"  With Interference = "
            f"{row['deepglobe_with_interference_psnr']:.3f} dB"
        )

    # =====================================================
    # Final PSNR Figures
    # =====================================================

    plot_dataset_metric(

        rows=rows,

        dataset_name="EuroSAT",

        metric_name="psnr",

        ylabel="PSNR (dB)",

        save_root=config.save_root,
    )

    plot_dataset_metric(

        rows=rows,

        dataset_name="DeepGlobe",

        metric_name="psnr",

        ylabel="PSNR (dB)",

        save_root=config.save_root,
    )

    # =====================================================
    # Final SSIM Figures
    # =====================================================

    plot_dataset_metric(

        rows=rows,

        dataset_name="EuroSAT",

        metric_name="ssim",

        ylabel="SSIM",

        save_root=config.save_root,
    )

    plot_dataset_metric(

        rows=rows,

        dataset_name="DeepGlobe",

        metric_name="ssim",

        ylabel="SSIM",

        save_root=config.save_root,
    )

    # =====================================================
    # Done
    # =====================================================

    print("=" * 90)

    print(
        "Preliminary interference observation completed."
    )

    print()

    print(
        "No DDPM separator was used."
    )

    print(
        "No interference cancellation was used."
    )

    print()

    print(
        "Comparison:"
    )

    print(
        "  No Interference"
    )

    print(
        "  With Interference"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        config.save_root
    )

    print()

    print(
        "Previous DDPM ablation results were NOT overwritten."
    )

    print("=" * 90)


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    main()