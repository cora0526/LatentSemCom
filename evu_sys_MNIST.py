import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    return (20 * torch.log10(torch.tensor(max_val))
            - 10 * torch.log10(mse)).item()


def calc_set(root, prefix_tx, prefix_rx):
    psnr_list = []
    ssim_list = []

    for i in range(150):

        tx = read_image(f"{root}/{prefix_tx}{i}.png").float() / 255.0
        rx = read_image(f"{root}/{prefix_rx}{i}.png").float() / 255.0

        if tx.shape != rx.shape:
            rx = resize(rx, tx.shape[-2:])

        p = psnr(tx, rx)

        tx_np = tx.permute(1,2,0).numpy()
        rx_np = rx.permute(1,2,0).numpy()

        s = ssim(
            tx_np,
            rx_np,
            channel_axis=2,
            data_range=1.0
        )

        psnr_list.append(p)
        ssim_list.append(s)

    print("="*40)
    print(prefix_tx)
    print("="*40)

    print(f"PSNR")
    print(f"  Max : {np.max(psnr_list):.3f}")
    print(f"  Min : {np.min(psnr_list):.3f}")
    print(f"  Mean: {np.mean(psnr_list):.3f}")

    print()

    print(f"SSIM")
    print(f"  Max : {np.max(ssim_list):.4f}")
    print(f"  Min : {np.min(ssim_list):.4f}")
    print(f"  Mean: {np.mean(ssim_list):.4f}")

    return {
    "psnr_mean": np.mean(psnr_list),
    "psnr_max": np.max(psnr_list),
    "psnr_min": np.min(psnr_list),
    "ssim_mean": np.mean(ssim_list),
    "ssim_max": np.max(ssim_list),
    "ssim_min": np.min(ssim_list),
    }   

tests = []


MNIST_psnr = []
MNIST_ssim = []

FashionMNIST_psnr = []
FashionMNIST_ssim = []

# root = f"D:/Python/SysTest/propose_VAE_GAN_2by2"
root = f"D:/Python/SysTest/propose_septerate_GAN"


for j in range(0,31,2):
    data_root = root + f"/test_{j}"
    print(f"\n========== test_{j} ==========\n")
    # deepglobe
    result_M = calc_set(
        data_root,
        "transmit_M_",
        "invert_recon_M_ep"
    )

    result_F = calc_set(
        data_root,
        "transmit_F_",
        "invert_recon_F_ep"
    )
    tests.append(j)

    MNIST_psnr.append(result_M["psnr_mean"])
    MNIST_ssim.append(result_M["ssim_mean"])

    FashionMNIST_psnr.append(result_F["psnr_mean"])
    FashionMNIST_ssim.append(result_F["ssim_mean"])



plt.figure(figsize=(6,4))

plt.plot(tests, MNIST_psnr, 'o-', label='MNIST')
plt.plot(tests, FashionMNIST_psnr, 's-', label='FashionMNIST')

plt.xlabel("SNR (dB)")
plt.ylabel("Average PSNR (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))

plt.plot(tests, MNIST_ssim, 'o-', label='MNIST')
plt.plot(tests, FashionMNIST_ssim, 's-', label='FashionMNIST')

plt.xlabel("SNR (dB)")
plt.ylabel("Average SSIM")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

result_table = pd.DataFrame({
    "SNR": tests,
    "MNIST_PSNR_mean": MNIST_psnr,
    "MNIST_SSIM_mean": MNIST_ssim,
    "FashionMNIST_PSNR_mean": FashionMNIST_psnr,
    "FashionMNIST_SSIM_mean": FashionMNIST_ssim,
})

save_result_path = root + "/metric_results.csv"
# save_result_path = "D:/Python/SysTest/propose_VAE_GAN_2by2/metric_results.csv"

# ===== 寫入說明 =====
with open(save_result_path, "w", encoding="utf-8") as f:
    f.write("# Experiment : MIMO-OFDM Latent Communication no DDPM\n")
    f.write("# Dataset    : MNIST + FashionMNIST\n")
    f.write("# Images     : 150 per SNR\n")
    f.write("# Metric     : Mean PSNR / Mean SSIM\n")
    f.write("# Note       : propose sperated gan\n")
    f.write("\n")

# ===== 接著把 DataFrame 接在後面 =====
result_table.to_csv(
    save_result_path,
    mode="a",
    index=False
)