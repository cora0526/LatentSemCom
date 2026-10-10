import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from skimage.metrics import structural_similarity as ssim


# ============================================================
# 1. Settings
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 改成你自己的 reconstruction 圖片位置
MNIST_RECON_PATH = "./Reconstructed_MNIST_images.png"
FASHION_RECON_PATH = "./Reconstructed_FashionMNIST_images.png"

# Dataset 儲存位置
DATA_ROOT = "./data"

# 結果輸出位置
OUTPUT_ROOT = "./recovered_original"

# 每張 reconstruction 最後留下幾個候選
TOP_K = 10

# 第一階段 MSE 搜尋先留下多少張，再做 SSIM
PRESELECT_K = 200

# Data batch size
SEARCH_BATCH_SIZE = 5000

os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("Device:", DEVICE)


# ============================================================
# 2. Load MNIST / FashionMNIST
# ============================================================

print("\nLoading datasets...")

mnist = datasets.MNIST(
    root=DATA_ROOT,
    train=True,                  # 重要！你原本程式是 train=True
    download=True
)

fashion = datasets.FashionMNIST(
    root=DATA_ROOT,
    train=True,                  # 重要！你原本程式是 train=True
    download=True
)

print("MNIST:", len(mnist))
print("FashionMNIST:", len(fashion))


# ============================================================
# 3. Split torchvision 5x5 grid
# ============================================================

def split_torchvision_grid(image_path,
                           rows=5,
                           cols=5,
                           image_size=28,
                           padding=2):
    """
    專門切 torchvision make_grid 預設產生的圖片。

    torchvision make_grid:
        padding = 2
        image size = 28x28

    5x5 時：
        2 + 5*28 + 5*2 = 152
    """

    image = Image.open(image_path).convert("L")

    image_np = np.array(image).astype(np.float32) / 255.0

    h, w = image_np.shape

    expected_h = padding + rows * (image_size + padding)
    expected_w = padding + cols * (image_size + padding)

    print("\nGrid:", image_path)
    print("Actual size :", w, "x", h)
    print("Expected    :", expected_w, "x", expected_h)

    if w != expected_w or h != expected_h:
        print(
            "WARNING: image size does not exactly match "
            "torchvision make_grid default layout."
        )

    images = []

    for r in range(rows):
        for c in range(cols):

            y = padding + r * (image_size + padding)
            x = padding + c * (image_size + padding)

            crop = image_np[
                y:y + image_size,
                x:x + image_size
            ]

            if crop.shape != (28, 28):
                raise ValueError(
                    f"Wrong crop shape at row={r}, col={c}: "
                    f"{crop.shape}"
                )

            crop_tensor = torch.from_numpy(crop).float()

            images.append(crop_tensor)

    return images


# ============================================================
# 4. Dataset → [0,1] tensor
# ============================================================

def get_dataset_tensors(dataset):

    # MNIST / FashionMNIST 原始資料為 uint8 [0,255]
    images = dataset.data.float() / 255.0
    labels = dataset.targets.long()

    return images, labels


mnist_images, mnist_labels = get_dataset_tensors(mnist)
fashion_images, fashion_labels = get_dataset_tensors(fashion)


# ============================================================
# 5. Optional centering
# ============================================================

def center_of_mass_shift(img):
    """
    把影像稍微移到質量中心。
    
    不會拿這個結果當 original，
    只用在 similarity matching。
    """

    img = img.clone()

    h, w = img.shape

    mass = img.sum()

    if mass < 1e-8:
        return img

    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij"
    )

    cy = (yy * img).sum() / mass
    cx = (xx * img).sum() / mass

    target_y = (h - 1) / 2
    target_x = (w - 1) / 2

    shift_y = int(round((target_y - cy).item()))
    shift_x = int(round((target_x - cx).item()))

    img = torch.roll(
        img,
        shifts=(shift_y, shift_x),
        dims=(0, 1)
    )

    return img


# ============================================================
# 6. Basic image normalization
# ============================================================

def normalize_for_matching(img):
    """
    reconstruction 與 original 的亮度可能不完全相同，
    因此 matching 前做 min-max normalization。
    """

    img = img.clone()

    min_val = img.min()
    max_val = img.max()

    if max_val - min_val > 1e-8:
        img = (img - min_val) / (max_val - min_val)

    return img


# ============================================================
# 7. Search first-stage candidates using MSE
# ============================================================

def mse_presearch(
        target,
        dataset_images,
        preselect_k=200,
        batch_size=5000
):

    target = normalize_for_matching(target)
    target = target.to(DEVICE)

    all_scores = []

    total = dataset_images.shape[0]

    for start in range(0, total, batch_size):

        end = min(start + batch_size, total)

        batch = dataset_images[start:end].to(DEVICE)

        # [B,28,28]
        mse = torch.mean(
            (batch - target.unsqueeze(0)) ** 2,
            dim=(1, 2)
        )

        all_scores.append(mse.cpu())

    all_scores = torch.cat(all_scores)

    values, indices = torch.topk(
        all_scores,
        k=min(preselect_k, total),
        largest=False
    )

    return indices, values


# ============================================================
# 8. SSIM calculation
# ============================================================

def calculate_ssim(target, candidate):

    target_np = target.cpu().numpy()
    candidate_np = candidate.cpu().numpy()

    score = ssim(
        target_np,
        candidate_np,
        data_range=1.0
    )

    return score


# ============================================================
# 9. Final ranking
# ============================================================

def rank_candidates(
        target,
        dataset_images,
        dataset_labels,
        pre_indices,
        top_k=10
):
    """
    Final score:
        MSE lower = better
        SSIM higher = better

    score = normalized MSE + normalized (1-SSIM)
    """

    target = normalize_for_matching(target)

    results = []

    for idx in pre_indices.tolist():

        candidate = dataset_images[idx]

        # ------------------------------------
        # MSE
        # ------------------------------------
        mse_value = F.mse_loss(
            candidate,
            target
        ).item()

        # ------------------------------------
        # SSIM
        # ------------------------------------
        ssim_value = calculate_ssim(
            target,
            candidate
        )

        results.append({
            "index": idx,
            "label": int(dataset_labels[idx]),
            "mse": mse_value,
            "ssim": ssim_value
        })

    # --------------------------------------------------------
    # Normalize scores
    # --------------------------------------------------------

    mse_array = np.array(
        [r["mse"] for r in results]
    )

    ssim_array = np.array(
        [r["ssim"] for r in results]
    )

    mse_min = mse_array.min()
    mse_max = mse_array.max()

    if mse_max > mse_min:
        mse_norm = (
            (mse_array - mse_min) /
            (mse_max - mse_min)
        )
    else:
        mse_norm = np.zeros_like(mse_array)

    # SSIM 越大越好，所以轉成 distance
    ssim_distance = 1.0 - ssim_array

    ssim_min = ssim_distance.min()
    ssim_max = ssim_distance.max()

    if ssim_max > ssim_min:
        ssim_norm = (
            (ssim_distance - ssim_min) /
            (ssim_max - ssim_min)
        )
    else:
        ssim_norm = np.zeros_like(ssim_distance)

    # --------------------------------------------------------
    # Combined score
    #
    # MSE : 50%
    # SSIM: 50%
    # --------------------------------------------------------

    combined = (
        0.5 * mse_norm +
        0.5 * ssim_norm
    )

    for i, r in enumerate(results):
        r["score"] = float(combined[i])

    results.sort(
        key=lambda x: x["score"]
    )

    return results[:top_k]


# ============================================================
# 10. Save candidate comparison
# ============================================================

def save_candidate_grid(
        target,
        results,
        dataset_images,
        save_path,
        title=None
):
    """
    第一張 = reconstruction
    後面 = Top K original candidates
    """

    count = len(results) + 1

    fig, axes = plt.subplots(
        1,
        count,
        figsize=(2 * count, 2.5)
    )

    # Reconstruction
    axes[0].imshow(
        target.cpu().numpy(),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0].set_title("Recon")
    axes[0].axis("off")

    for rank, result in enumerate(results):

        idx = result["index"]

        candidate = dataset_images[idx]

        ax = axes[rank + 1]

        ax.imshow(
            candidate.cpu().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        ax.set_title(
            f"#{rank+1}\n"
            f"idx={idx}\n"
            f"L={result['label']}\n"
            f"SSIM={result['ssim']:.3f}"
        )

        ax.axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. Main searching function
# ============================================================

def recover_dataset(
        reconstruction_path,
        dataset_name,
        dataset_images,
        dataset_labels
):

    save_dir = os.path.join(
        OUTPUT_ROOT,
        dataset_name
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    candidate_dir = os.path.join(
        save_dir,
        "candidate_grids"
    )

    os.makedirs(
        candidate_dir,
        exist_ok=True
    )

    print("\n")
    print("=" * 70)
    print("Processing:", dataset_name)
    print("=" * 70)

    # --------------------------------------------------------
    # Split 5x5 reconstruction
    # --------------------------------------------------------

    targets = split_torchvision_grid(
        reconstruction_path
    )

    # Save cropped reconstruction for checking
    recon_tensor = torch.stack(targets).unsqueeze(1)

    save_image(
        recon_tensor,
        os.path.join(
            save_dir,
            "cropped_reconstruction.png"
        ),
        nrow=5,
        padding=2
    )

    all_results = []

    rank1_originals = []

    # --------------------------------------------------------
    # Search each reconstruction
    # --------------------------------------------------------

    for image_id, target in enumerate(targets):

        row = image_id // 5
        col = image_id % 5

        print(
            f"\n[{dataset_name}] "
            f"Image {image_id+1}/25 "
            f"(row={row+1}, col={col+1})"
        )

        # Stage 1
        pre_indices, pre_mse = mse_presearch(
            target,
            dataset_images,
            preselect_k=PRESELECT_K,
            batch_size=SEARCH_BATCH_SIZE
        )

        # Stage 2
        results = rank_candidates(
            target,
            dataset_images,
            dataset_labels,
            pre_indices,
            top_k=TOP_K
        )

        all_results.append(results)

        # Rank 1
        best_idx = results[0]["index"]

        rank1_originals.append(
            dataset_images[best_idx]
        )

        # Print
        print(
            "Rank | Dataset idx | Label | "
            "MSE       | SSIM      | Score"
        )

        for rank, result in enumerate(results):

            print(
                f"{rank+1:4d} | "
                f"{result['index']:11d} | "
                f"{result['label']:5d} | "
                f"{result['mse']:.6f} | "
                f"{result['ssim']:.6f} | "
                f"{result['score']:.6f}"
            )

        # Candidate grid
        candidate_path = os.path.join(
            candidate_dir,
            f"image_{image_id:02d}_"
            f"r{row+1}_c{col+1}.png"
        )

        save_candidate_grid(
            target,
            results,
            dataset_images,
            candidate_path,
            title=(
                f"{dataset_name} "
                f"row {row+1}, col {col+1}"
            )
        )

    # --------------------------------------------------------
    # Save Rank-1 recovered grid
    # --------------------------------------------------------

    rank1_tensor = torch.stack(
        rank1_originals
    ).unsqueeze(1)

    save_image(
        rank1_tensor,
        os.path.join(
            save_dir,
            "rank1_recovered_original.png"
        ),
        nrow=5,
        padding=2
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        save_dir,
        "candidate_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "image_id",
            "row",
            "column",
            "rank",
            "dataset_index",
            "label",
            "mse",
            "ssim",
            "combined_score"
        ])

        for image_id, results in enumerate(all_results):

            row = image_id // 5
            col = image_id % 5

            for rank, result in enumerate(results):

                writer.writerow([
                    image_id,
                    row + 1,
                    col + 1,
                    rank + 1,
                    result["index"],
                    result["label"],
                    result["mse"],
                    result["ssim"],
                    result["score"]
                ])

    print("\nSaved:")
    print(
        os.path.join(
            save_dir,
            "rank1_recovered_original.png"
        )
    )

    print(csv_path)

    return all_results


# ============================================================
# 12. Recover MNIST
# ============================================================

mnist_results = recover_dataset(
    reconstruction_path=MNIST_RECON_PATH,
    dataset_name="MNIST",
    dataset_images=mnist_images,
    dataset_labels=mnist_labels
)


# ============================================================
# 13. Recover FashionMNIST
# ============================================================

fashion_results = recover_dataset(
    reconstruction_path=FASHION_RECON_PATH,
    dataset_name="FashionMNIST",
    dataset_images=fashion_images,
    dataset_labels=fashion_labels
)


# ============================================================
# Finished
# ============================================================

print("\n")
print("=" * 70)
print("Finished!")
print("=" * 70)

print(
    f"""
Results are saved in:

{OUTPUT_ROOT}/MNIST/
    cropped_reconstruction.png
    rank1_recovered_original.png
    candidate_results.csv
    candidate_grids/

{OUTPUT_ROOT}/FashionMNIST/
    cropped_reconstruction.png
    rank1_recovered_original.png
    candidate_results.csv
    candidate_grids/
"""
)