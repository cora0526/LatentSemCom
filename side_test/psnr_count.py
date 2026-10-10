import os
import math
import numpy as np
from PIL import Image


# ============================================================
# Settings
# ============================================================

ROOT = "./recovered_original"

MNIST_ORIGINAL = os.path.join(
    ROOT, "MNIST", "rank1_recovered_original.png"
)
MNIST_RECON = os.path.join(
    ROOT, "MNIST", "cropped_reconstruction.png"
)

FASHION_ORIGINAL = os.path.join(
    ROOT, "FashionMNIST", "rank1_recovered_original.png"
)
FASHION_RECON = os.path.join(
    ROOT, "FashionMNIST", "cropped_reconstruction.png"
)


# ============================================================
# Split torchvision make_grid
# ============================================================

def split_grid(
    image_path,
    rows=5,
    cols=5,
    image_size=28,
    padding=2
):
    """
    Split torchvision save_image/make_grid 5x5 grid.

    torchvision default:
        padding = 2
        image_size = 28

    5x5 grid:
        2 + 5 * (28 + 2) = 152 pixels
    """

    img = Image.open(image_path).convert("L")

    img = np.asarray(img, dtype=np.float32) / 255.0

    print(f"\nLoading: {image_path}")
    print("Image shape:", img.shape)

    images = []

    for row in range(rows):
        for col in range(cols):

            y = padding + row * (image_size + padding)
            x = padding + col * (image_size + padding)

            crop = img[
                y:y + image_size,
                x:x + image_size
            ]

            if crop.shape != (image_size, image_size):
                raise ValueError(
                    f"Wrong crop size at row={row}, col={col}: "
                    f"{crop.shape}"
                )

            images.append(crop)

    return np.stack(images)


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(original, reconstructed):
    """
    Images are assumed to be in [0, 1].
    """

    mse = np.mean(
        (original - reconstructed) ** 2
    )

    if mse == 0:
        return float("inf")

    psnr = 10.0 * math.log10(1.0 / mse)

    return psnr


# ============================================================
# Evaluate one dataset
# ============================================================

def evaluate(
    dataset_name,
    original_path,
    recon_path
):

    print("\n")
    print("=" * 60)
    print(dataset_name)
    print("=" * 60)

    original = split_grid(original_path)
    recon = split_grid(recon_path)

    if original.shape != recon.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{original.shape} vs {recon.shape}"
        )

    psnr_values = []

    print("\nIndividual PSNR:")
    print("-" * 45)

    for i in range(len(original)):

        value = calculate_psnr(
            original[i],
            recon[i]
        )

        psnr_values.append(value)

        row = i // 5 + 1
        col = i % 5 + 1

        print(
            f"Image {i+1:02d} "
            f"(row {row}, col {col}): "
            f"{value:.2f} dB"
        )

    psnr_values = np.asarray(psnr_values)

    # 避免 inf 影響統計
    finite_values = psnr_values[
        np.isfinite(psnr_values)
    ]

    mean_psnr = np.mean(finite_values)
    std_psnr = np.std(finite_values)
    min_psnr = np.min(finite_values)
    max_psnr = np.max(finite_values)

    print("\n" + "-" * 45)
    print(f"{dataset_name} Results")
    print("-" * 45)

    print(f"Mean PSNR : {mean_psnr:.2f} dB")
    print(f"Std PSNR  : {std_psnr:.2f} dB")
    print(f"Min PSNR  : {min_psnr:.2f} dB")
    print(f"Max PSNR  : {max_psnr:.2f} dB")

    return psnr_values, mean_psnr, std_psnr


# ============================================================
# MNIST
# ============================================================

mnist_psnr, mnist_mean, mnist_std = evaluate(
    "MNIST",
    MNIST_ORIGINAL,
    MNIST_RECON
)


# ============================================================
# FashionMNIST
# ============================================================

fashion_psnr, fashion_mean, fashion_std = evaluate(
    "FashionMNIST",
    FASHION_ORIGINAL,
    FASHION_RECON
)


# ============================================================
# Final result
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"MNIST        : "
    f"{mnist_mean:.2f} ± {mnist_std:.2f} dB"
)

print(
    f"FashionMNIST : "
    f"{fashion_mean:.2f} ± {fashion_std:.2f} dB"
)