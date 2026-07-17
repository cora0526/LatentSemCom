import pandas as pd
import matplotlib.pyplot as plt

# =====================
# CSV
# =====================
file_pro = "D:/Python/SysTest/propose_VAE_GAN_2by2/metric_results.csv"
file_vae = "D:/Python/SysTest/baseline_vae_2by2/metric_results.csv"
file_jscc = "D:/Python/SysTest/baseline_JSCC/metric_results.csv"
file_sep_pro = 'D:/Python/SysTest/propose_septerate_GAN/metric_results.csv'
file_traditional = 'D:/Python/SysTest/baseline_JPEG2000_MNIST/metric_results.csv'

data_pro = pd.read_csv(file_pro, comment="#")
data_vae = pd.read_csv(file_vae, comment="#")
data_jscc = pd.read_csv(file_jscc, comment="#")
data_seppro = pd.read_csv(file_sep_pro, comment="#")
data_traditional = pd.read_csv(file_traditional, comment="#")

datasets = {
    "Proposed": data_pro,
    "VAE": data_vae,
    "DeepJSCC": data_jscc,
    "Sep_pro": data_seppro,
    "Traditional":  data_traditional
}

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

plots = [
    ("MNIST_PSNR_mean", "MNIST PSNR", "PSNR (dB)", axs[0,0]),
    ("MNIST_SSIM_mean", "MNIST SSIM", "SSIM", axs[0,1]),
    ("FashionMNIST_PSNR_mean", "FashionMNIST PSNR", "PSNR (dB)", axs[1,0]),
    ("FashionMNIST_SSIM_mean", "FashionMNIST SSIM", "SSIM", axs[1,1]),
]

markers = ['o', 's', '^', 'D', 'x', '*']

for col, title, ylabel, ax in plots:

    for (name, df), marker in zip(datasets.items(), markers):

        ax.plot(df["SNR"],
                df[col],
                marker=marker,
                linewidth=2,
                markersize=5,
                label=name)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

plt.tight_layout()

plt.savefig("Comparison.png", dpi=300)

plt.show()