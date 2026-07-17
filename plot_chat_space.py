import pandas as pd
import matplotlib.pyplot as plt

# =====================
# CSV
# =====================
file_pro = "D:/Python/SysTest/propose_VAE_GAN_2by2_space/metric_results.csv"
file_vae = "D:/Python/SysTest/baseline_vae_2by2_space/metric_results.csv"
file_jscc = "D:/Python/SysTest/baseline_JSCC_space/metric_results.csv"
file_sep_pro = 'D:/Python/SysTest/propose_seperate_GAN_2by2_space/metric_results.csv'
file_sep_pro_new = 'D:/Python/SysTest/propose_VAE_GAN_2by2_adapter/metric_results.csv'


data_pro = pd.read_csv(file_pro, comment="#")
data_vae = pd.read_csv(file_vae, comment="#")
data_jscc = pd.read_csv(file_jscc, comment="#")
data_seppro = pd.read_csv(file_sep_pro, comment="#")
data_seppro_new = pd.read_csv(file_sep_pro_new, comment="#")

datasets = {
    "Proposed": data_pro,
    "VAE": data_vae,
    "DeepJSCC": data_jscc,
    "Sep_pro": data_seppro,
    "Proposed_new": data_seppro_new
}

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

plots = [
    ("deepglobe_PSNR_mean", "deepglobe PSNR", "PSNR (dB)", axs[0,0]),
    ("deepglobe_SSIM_mean", "deepglobe SSIM", "SSIM", axs[0,1]),
    ("EuroSAT_PSNR_mean", "EuroSAT PSNR", "PSNR (dB)", axs[1,0]),
    ("EuroSAT_SSIM_mean", "EuroSAT SSIM", "SSIM", axs[1,1]),
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