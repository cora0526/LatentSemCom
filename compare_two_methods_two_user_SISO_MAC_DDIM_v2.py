import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# Config
# =========================================================

CSV_PATH = (
    r"D:/VAE_GAN_DM/save_data/"
    r"compare_two_methods_two_user_SISO_MAC_v2/"
    r"metrics_long.csv"
)

SAVE_ROOT = (
    r"D:/VAE_GAN_DM/save_data/"
    r"compare_two_methods_two_user_SISO_MAC_v3/"
    r"replot"
)

os.makedirs(SAVE_ROOT, exist_ok=True)


# =========================================================
# Plot settings
# =========================================================

METHOD_COLORS = {
    "VAE-DDPM-VAE": "#D97757",   # soft terracotta
    "VAE-DDPM-GAN": "#6C63B8",   # muted purple
}

METHOD_LABELS = {
    "VAE-DDPM-VAE": "VAE",
    "VAE-DDPM-GAN": "Proposed",
}

METHOD_ORDER = [
    "VAE-DDPM-VAE",
    "VAE-DDPM-GAN",
]

CHANNEL_STYLES = {
    "AWGN-SISO-MAC": "-",
    "Rayleigh-SISO-MAC": "--",
    "Rician-SISO-MAC": ":",
}

CHANNEL_MARKERS = {
    "AWGN-SISO-MAC": "o",       # circle
    "Rayleigh-SISO-MAC": "s",   # square
    "Rician-SISO-MAC": "D",     # diamond
}

CHANNEL_LABELS = {
    "AWGN-SISO-MAC": "AWGN",
    "Rayleigh-SISO-MAC": "Rayleigh",
    "Rician-SISO-MAC": "Rician",
}

CHANNEL_ORDER = [
    "AWGN-SISO-MAC",
    "Rayleigh-SISO-MAC",
    "Rician-SISO-MAC",
]


# =========================================================
# Plot function
# =========================================================

def plot_metric(
    df,
    metric,
    ylabel,
    save_name,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    # =====================================================
    # Actual curves
    # =====================================================

    for method in METHOD_ORDER:
        for channel in CHANNEL_ORDER:

            selected = df[
                (df["method"] == method)
                & (df["channel"] == channel)
            ].copy()

            if selected.empty:
                continue

            selected = selected.sort_values("snr_db")

            x = selected["snr_db"].astype(float)
            y = selected[metric].astype(float)

            label = (
                f"{METHOD_LABELS[method]} + "
                f"{CHANNEL_LABELS[channel]}"
            )

            ax.plot(
                x,
                y,
                color=METHOD_COLORS[method],
                linestyle=CHANNEL_STYLES[channel],
                marker=CHANNEL_MARKERS[channel],
                markersize=6,
                linewidth=2,
                markeredgewidth=0.8,
                label=label,
            )

    # =====================================================
    # Axis
    # =====================================================

    ax.set_xlabel(
        "SNR (dB)",
        fontsize=18,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=18,
    )

    ax.tick_params(
        axis="both",
        labelsize=16,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    # =====================================================
    # Legend
    # =====================================================

    ax.legend(
        ncol=2,
        loc="lower right",
        fontsize=16,
        frameon=True,
        handlelength=2.2,
        columnspacing=1.5,
        handletextpad=0.6,
    )

    # =====================================================
    # Save
    # =====================================================

    fig.tight_layout()

    save_path = os.path.join(
        SAVE_ROOT,
        save_name,
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        os.path.join(SAVE_ROOT, save_name + ".pdf"),
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {save_path}")


# =========================================================
# Main
# =========================================================

def main():

    print("Reading CSV:")
    print(CSV_PATH)

    df = pd.read_csv(CSV_PATH)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMethods:")
    print(df["method"].unique())

    print("\nChannels:")
    print(df["channel"].unique())

    print("\nSNR:")
    print(sorted(df["snr_db"].unique()))

    # -----------------------------------------------------
    # PSNR
    # -----------------------------------------------------

    plot_metric(
        df=df,
        metric="avg_psnr_real",
        ylabel="Average PSNR vs. real image (dB)",
        save_name="PSNR_vs_SNR.png",
    )

    # -----------------------------------------------------
    # SSIM
    # -----------------------------------------------------

    plot_metric(
        df=df,
        metric="avg_ssim_real",
        ylabel="Average SSIM vs. real image",
        save_name="SSIM_vs_SNR.png",
    )

    # -----------------------------------------------------
    # Latent Cosine
    # -----------------------------------------------------

    plot_metric(
        df=df,
        metric="avg_latent_cosine",
        ylabel="Average latent cosine similarity",
        save_name="Latent_Cosine_vs_SNR.png",
    )

    # -----------------------------------------------------
    # Latent MSE
    # -----------------------------------------------------

    plot_metric(
        df=df,
        metric="avg_latent_mse",
        ylabel="Average latent MSE",
        save_name="Latent_MSE_vs_SNR.png",
    )

    # -----------------------------------------------------
    # Consistency MSE
    # -----------------------------------------------------

    plot_metric(
        df=df,
        metric="consistency_mse",
        ylabel="Mixture consistency MSE",
        save_name="Consistency_MSE_vs_SNR.png",
    )

    print("\nAll plots completed.")


if __name__ == "__main__":
    main()