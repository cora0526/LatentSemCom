import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as structural_ssim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from model import (
    Encoder_Satellite,
    Generator_Satellite_Conditional,
    LatentAdapter,
)
from com_utils import mimo_2x2_channel_zf_train
from utils import SatelliteDataset


# ============================================================
# Settings
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT_EUROSAT = (
    r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
)
DATA_ROOT_DEEPGLOBE = (
    r"D:/Python/SemCom/saved_data/deepglobe_64/train"
)

SAVE_ROOT = (
    r"D:/Python/SysTest/propose_VAE_GAN_2by2_adapter"
)

GENERATOR_PATH = (
    "./save_model/G_conditional_last_layer_finetuned_v2.pth"
)

ENCODER_EUROSAT_PATH = (
    "./save_model/"
    "E_invert_to_conditionG_C_EuroSAT_fixed_H.pth"
)

ENCODER_DEEPGLOBE_PATH = (
    "./save_model/"
    "E_invert_to_conditionG_C_deepglobe_fixed_H.pth"
)

ADAPTER_EUROSAT_PATH = (
    "./save_model/Adapter_EuroSAT_best.pth"
)

ADAPTER_DEEPGLOBE_PATH = (
    "./save_model/Adapter_DeepGlobe_best.pth"
)

Z_DIM = 256
NUM_IMAGES = 150
BATCH_SIZE = 50

SNR_LIST = list(range(0, 31, 2))

SAVE_RECON_IMAGES = True
SAVE_GRID_IMAGES = True


# ============================================================
# Helper functions
# ============================================================
def power_normalize(z, eps=1e-8):
    power = z.pow(2).mean(
        dim=1,
        keepdim=True,
    )

    return z / torch.sqrt(
        power + eps
    )


def get_img(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]

    return batch


def tensor_to_01(x):
    """
    Convert [-1,1] tensor to [0,1].
    """
    return (
        (x.detach().float().cpu() + 1.0) / 2.0
    ).clamp(0.0, 1.0)


def batch_psnr(
    original,
    reconstructed,
    max_val=1.0,
):
    """
    Input tensors are expected in [-1,1].
    Returns per-image PSNR.
    """
    original_01 = tensor_to_01(original)
    reconstructed_01 = tensor_to_01(
        reconstructed
    )

    mse = F.mse_loss(
        reconstructed_01,
        original_01,
        reduction="none",
    ).flatten(1).mean(dim=1)

    psnr_values = torch.where(
        mse <= 1e-12,
        torch.full_like(
            mse,
            float("inf"),
        ),
        10.0 * torch.log10(
            (max_val ** 2) / mse
        ),
    )

    return psnr_values.numpy()


def batch_ssim(
    original,
    reconstructed,
):
    """
    Input tensors are expected in [-1,1].
    Returns per-image SSIM.
    """
    original_01 = tensor_to_01(original)
    reconstructed_01 = tensor_to_01(
        reconstructed
    )

    ssim_values = []

    for original_img, reconstructed_img in zip(
        original_01,
        reconstructed_01,
    ):
        original_np = (
            original_img
            .permute(1, 2, 0)
            .numpy()
        )

        reconstructed_np = (
            reconstructed_img
            .permute(1, 2, 0)
            .numpy()
        )

        value = structural_ssim(
            original_np,
            reconstructed_np,
            channel_axis=2,
            data_range=1.0,
        )

        ssim_values.append(
            float(value)
        )

    return np.asarray(
        ssim_values,
        dtype=np.float64,
    )


def summarize_metric(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite_values = values[
        np.isfinite(values)
    ]

    if len(finite_values) == 0:
        return {
            "mean": float("inf"),
            "max": float("inf"),
            "min": float("inf"),
            "std": float("nan"),
        }

    return {
        "mean": float(
            np.mean(finite_values)
        ),
        "max": float(
            np.max(finite_values)
        ),
        "min": float(
            np.min(finite_values)
        ),
        "std": float(
            np.std(finite_values)
        ),
    }


# ============================================================
# Model loading
# ============================================================
def load_models(device):
    generator = (
        Generator_Satellite_Conditional(
            z_dim=Z_DIM,
            num_classes=2,
            label_dim=64,
            out_ch=3,
        )
        .to(device)
    )

    encoder_e = Encoder_Satellite(
        z_dim=Z_DIM
    ).to(device)

    encoder_g = Encoder_Satellite(
        z_dim=Z_DIM
    ).to(device)

    adapter_e = LatentAdapter(
        z_dim=Z_DIM,
        hidden_dim=512,
        residual_scale=0.1,
    ).to(device)

    adapter_g = LatentAdapter(
        z_dim=Z_DIM,
        hidden_dim=512,
        residual_scale=0.1,
    ).to(device)

    generator.load_state_dict(
        torch.load(
            GENERATOR_PATH,
            map_location=device,
        )
    )

    encoder_e.load_state_dict(
        torch.load(
            ENCODER_EUROSAT_PATH,
            map_location=device,
        )
    )

    encoder_g.load_state_dict(
        torch.load(
            ENCODER_DEEPGLOBE_PATH,
            map_location=device,
        )
    )

    adapter_e.load_state_dict(
        torch.load(
            ADAPTER_EUROSAT_PATH,
            map_location=device,
        )
    )

    adapter_g.load_state_dict(
        torch.load(
            ADAPTER_DEEPGLOBE_PATH,
            map_location=device,
        )
    )

    models = [
        generator,
        encoder_e,
        encoder_g,
        adapter_e,
        adapter_g,
    ]

    for model in models:
        model.eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

    return (
        generator,
        encoder_e,
        encoder_g,
        adapter_e,
        adapter_g,
    )


# ============================================================
# Dataset loading
# ============================================================
def build_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])

    dataset_e = datasets.ImageFolder(
        DATA_ROOT_EUROSAT,
        transform=transform,
    )

    dataset_g = SatelliteDataset(
        DATA_ROOT_DEEPGLOBE,
        transform=transform,
    )

    dataset_e = Subset(
        dataset_e,
        range(
            min(
                NUM_IMAGES,
                len(dataset_e),
            )
        ),
    )

    dataset_g = Subset(
        dataset_g,
        range(
            min(
                NUM_IMAGES,
                len(dataset_g),
            )
        ),
    )

    loader_e = DataLoader(
        dataset_e,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    loader_g = DataLoader(
        dataset_g,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return loader_e, loader_g


# ============================================================
# Evaluation at one SNR
# ============================================================
def evaluate_one_snr(
    snr_db,
    loader_e,
    loader_g,
    generator,
    encoder_e,
    encoder_g,
    adapter_e,
    adapter_g,
    device,
):
    euro_psnr_values = []
    euro_ssim_values = []

    deep_psnr_values = []
    deep_ssim_values = []

    euro_original_all = []
    euro_recon_all = []

    deep_original_all = []
    deep_recon_all = []

    euro_index = 0
    deep_index = 0

    snr_root = os.path.join(
        SAVE_ROOT,
        f"test_{snr_db}",
    )

    if SAVE_RECON_IMAGES:
        os.makedirs(
            snr_root,
            exist_ok=True,
        )

    for batch_e, batch_g in zip(
        loader_e,
        loader_g,
    ):
        real_e = get_img(batch_e).to(
            device,
            non_blocking=True,
        )

        real_g = get_img(batch_g).to(
            device,
            non_blocking=True,
        )

        current_batch = min(
            real_e.size(0),
            real_g.size(0),
        )

        real_e = real_e[:current_batch]
        real_g = real_g[:current_batch]

        label_e = torch.zeros(
            current_batch,
            dtype=torch.long,
            device=device,
        )

        label_g = torch.ones(
            current_batch,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            latent_e_raw, _ = encoder_e(
                real_e
            )

            latent_g_raw, _ = encoder_g(
                real_g
            )

            latent_e_base = power_normalize(
                latent_e_raw
            )

            latent_g_base = power_normalize(
                latent_g_raw
            )

            latent_e_tx = power_normalize(
                adapter_e(
                    latent_e_base
                )
            )

            latent_g_tx = power_normalize(
                adapter_g(
                    latent_g_base
                )
            )

            latent_e_rx, latent_g_rx = (
                mimo_2x2_channel_zf_train(
                    latent_e_tx,
                    latent_g_tx,
                    float(snr_db),
                )
            )

            recon_e = generator(
                latent_e_rx,
                label_e,
            )

            recon_g = generator(
                latent_g_rx,
                label_g,
            )

        euro_psnr_values.extend(
            batch_psnr(
                real_e,
                recon_e,
            ).tolist()
        )

        euro_ssim_values.extend(
            batch_ssim(
                real_e,
                recon_e,
            ).tolist()
        )

        deep_psnr_values.extend(
            batch_psnr(
                real_g,
                recon_g,
            ).tolist()
        )

        deep_ssim_values.extend(
            batch_ssim(
                real_g,
                recon_g,
            ).tolist()
        )

        if SAVE_GRID_IMAGES:
            euro_original_all.append(
                tensor_to_01(real_e)
            )
            euro_recon_all.append(
                tensor_to_01(recon_e)
            )

            deep_original_all.append(
                tensor_to_01(real_g)
            )
            deep_recon_all.append(
                tensor_to_01(recon_g)
            )

        if SAVE_RECON_IMAGES:
            real_e_01 = tensor_to_01(
                real_e
            )
            recon_e_01 = tensor_to_01(
                recon_e
            )

            real_g_01 = tensor_to_01(
                real_g
            )
            recon_g_01 = tensor_to_01(
                recon_g
            )

            for batch_index in range(
                current_batch
            ):
                save_image(
                    real_g_01[batch_index],
                    os.path.join(
                        snr_root,
                        (
                            "transmit_M_"
                            f"{deep_index}.png"
                        ),
                    ),
                )

                save_image(
                    recon_g_01[batch_index],
                    os.path.join(
                        snr_root,
                        (
                            "invert_recon_M_ep"
                            f"{deep_index}.png"
                        ),
                    ),
                )

                save_image(
                    real_e_01[batch_index],
                    os.path.join(
                        snr_root,
                        (
                            "transmit_F_"
                            f"{euro_index}.png"
                        ),
                    ),
                )

                save_image(
                    recon_e_01[batch_index],
                    os.path.join(
                        snr_root,
                        (
                            "invert_recon_F_ep"
                            f"{euro_index}.png"
                        ),
                    ),
                )

                deep_index += 1
                euro_index += 1

    euro_psnr = summarize_metric(
        euro_psnr_values
    )
    euro_ssim = summarize_metric(
        euro_ssim_values
    )

    deep_psnr = summarize_metric(
        deep_psnr_values
    )
    deep_ssim = summarize_metric(
        deep_ssim_values
    )

    if SAVE_GRID_IMAGES:
        euro_original_batch = torch.cat(
            euro_original_all,
            dim=0,
        )
        euro_recon_batch = torch.cat(
            euro_recon_all,
            dim=0,
        )

        deep_original_batch = torch.cat(
            deep_original_all,
            dim=0,
        )
        deep_recon_batch = torch.cat(
            deep_recon_all,
            dim=0,
        )

        save_image(
            euro_original_batch[:64],
            os.path.join(
                snr_root,
                "grid_original_EuroSAT.png",
            ),
            nrow=8,
        )

        save_image(
            euro_recon_batch[:64],
            os.path.join(
                snr_root,
                "grid_recon_EuroSAT.png",
            ),
            nrow=8,
        )

        save_image(
            deep_original_batch[:64],
            os.path.join(
                snr_root,
                "grid_original_DeepGlobe.png",
            ),
            nrow=8,
        )

        save_image(
            deep_recon_batch[:64],
            os.path.join(
                snr_root,
                "grid_recon_DeepGlobe.png",
            ),
            nrow=8,
        )

    print("\n" + "=" * 64)
    print(f"SNR = {snr_db} dB")
    print("=" * 64)

    print(
        "EuroSAT PSNR   | "
        f"mean={euro_psnr['mean']:.3f} | "
        f"max={euro_psnr['max']:.3f} | "
        f"min={euro_psnr['min']:.3f}"
    )

    print(
        "EuroSAT SSIM   | "
        f"mean={euro_ssim['mean']:.4f} | "
        f"max={euro_ssim['max']:.4f} | "
        f"min={euro_ssim['min']:.4f}"
    )

    print(
        "DeepGlobe PSNR | "
        f"mean={deep_psnr['mean']:.3f} | "
        f"max={deep_psnr['max']:.3f} | "
        f"min={deep_psnr['min']:.3f}"
    )

    print(
        "DeepGlobe SSIM | "
        f"mean={deep_ssim['mean']:.4f} | "
        f"max={deep_ssim['max']:.4f} | "
        f"min={deep_ssim['min']:.4f}"
    )

    return {
        "SNR": snr_db,

        "deepglobe_PSNR_mean": deep_psnr["mean"],
        "deepglobe_PSNR_max": deep_psnr["max"],
        "deepglobe_PSNR_min": deep_psnr["min"],
        "deepglobe_PSNR_std": deep_psnr["std"],

        "deepglobe_SSIM_mean": deep_ssim["mean"],
        "deepglobe_SSIM_max": deep_ssim["max"],
        "deepglobe_SSIM_min": deep_ssim["min"],
        "deepglobe_SSIM_std": deep_ssim["std"],

        "EuroSAT_PSNR_mean": euro_psnr["mean"],
        "EuroSAT_PSNR_max": euro_psnr["max"],
        "EuroSAT_PSNR_min": euro_psnr["min"],
        "EuroSAT_PSNR_std": euro_psnr["std"],

        "EuroSAT_SSIM_mean": euro_ssim["mean"],
        "EuroSAT_SSIM_max": euro_ssim["max"],
        "EuroSAT_SSIM_min": euro_ssim["min"],
        "EuroSAT_SSIM_std": euro_ssim["std"],
    }


# ============================================================
# Plotting
# ============================================================
def save_plots(result_table):
    plt.figure(figsize=(6, 4))

    plt.plot(
        result_table["SNR"],
        result_table[
            "deepglobe_PSNR_mean"
        ],
        "o-",
        label="DeepGlobe",
    )

    plt.plot(
        result_table["SNR"],
        result_table[
            "EuroSAT_PSNR_mean"
        ],
        "s-",
        label="EuroSAT",
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Average PSNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_ROOT,
            "adapter_PSNR_vs_SNR.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()

    plt.figure(figsize=(6, 4))

    plt.plot(
        result_table["SNR"],
        result_table[
            "deepglobe_SSIM_mean"
        ],
        "o-",
        label="DeepGlobe",
    )

    plt.plot(
        result_table["SNR"],
        result_table[
            "EuroSAT_SSIM_mean"
        ],
        "s-",
        label="EuroSAT",
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Average SSIM")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            SAVE_ROOT,
            "adapter_SSIM_vs_SNR.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(
        SAVE_ROOT,
        exist_ok=True,
    )

    (
        generator,
        encoder_e,
        encoder_g,
        adapter_e,
        adapter_g,
    ) = load_models(DEVICE)

    loader_e, loader_g = build_loaders()

    all_results = []

    for snr_db in SNR_LIST:
        result = evaluate_one_snr(
            snr_db=snr_db,
            loader_e=loader_e,
            loader_g=loader_g,
            generator=generator,
            encoder_e=encoder_e,
            encoder_g=encoder_g,
            adapter_e=adapter_e,
            adapter_g=adapter_g,
            device=DEVICE,
        )

        all_results.append(
            result
        )

    result_table = pd.DataFrame(
        all_results
    )

    csv_path = os.path.join(
        SAVE_ROOT,
        "metric_results.csv",
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "# Experiment : "
            "MIMO latent communication "
            "with domain-specific latent adapters\n"
        )
        file.write(
            "# Dataset    : "
            "DeepGlobe + EuroSAT\n"
        )
        file.write(
            f"# Images     : "
            f"{NUM_IMAGES} per dataset per SNR\n"
        )
        file.write(
            "# Channel    : "
            "2x2 MIMO + noise + ZF\n"
        )
        file.write(
            "# Metric     : "
            "PSNR / SSIM mean, max, min, std\n"
        )
        file.write(
            "# Pipeline   : "
            "Encoder -> Adapter -> "
            "Power Normalize -> Channel -> "
            "Frozen Conditional Generator\n"
        )
        file.write("\n")

    result_table.to_csv(
        csv_path,
        mode="a",
        index=False,
    )

    save_plots(
        result_table
    )

    print("\nFinished.")
    print(
        f"Results saved to:\n{SAVE_ROOT}"
    )
    print(
        f"CSV saved to:\n{csv_path}"
    )


if __name__ == "__main__":
    main()
