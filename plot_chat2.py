from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================
# Deep_WGANGP_VAE_DDPM\random_alpha_model_test\multi_interferer_sweep_snr10\multi_interferer_random_alpha_ddpm_metrics.csv
CSV_PATH = (
    r"D:\VAE_GAN_DM\save_data\Deep_WGANGP_VAE_DDPM\random_alpha_model_test\multi_interferer_sweep_snr10"
    r"\multi_interferer_random_alpha_ddpm_metrics.csv"
)

SAVE_DIR = (
    r"D:\VAE_GAN_DM\save_data\Deep_WGANGP_VAE_DDPM"
    r"\multi_interferer_plots"
)

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Figure style
# ============================================================

LINEWIDTH = 2.4
MARKERSIZE = 6.5

COLORS = {
    1: "#1178B4",   # 1 interferer
    2: "#119E82",   # 2 interferers
    3: "#E3A000",   # 3 interferers
    4: "#D55E00",   # 4 interferers
    5: "#CC79A7",
}

MARKERS = {
    1: "o",
    2: "s",
    3: "D",
    4: "^",
    5: "v",
}

LINESTYLES = {
    1: "-",
    2: "--",
    3: "-.",
    4: ":",
    5: "-",
}


def apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",

        "font.size": 14,

        "axes.labelsize": 18,
        "axes.titlesize": 18,

        "xtick.labelsize": 16,
        "ytick.labelsize": 16,

        "legend.fontsize": 16,

        "axes.linewidth": 1.0,

        "figure.dpi": 120,
        "savefig.dpi": 300,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax):

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.22,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.tick_params(
        direction="out",
        length=4,
        width=0.9,
    )


# ============================================================
# Save figure
# ============================================================

def save_figure(fig, save_dir, filename):

    png_path = os.path.join(
        save_dir,
        filename + ".png"
    )

    pdf_path = os.path.join(
        save_dir,
        filename + ".pdf"
    )

    fig.tight_layout()

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

    print("Saved:")
    print(png_path)
    print(pdf_path)


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip()

df["target"] = (
    df["target"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print("=" * 70)
print("CSV loaded")
print("=" * 70)

print("Columns:")
print(df.columns.tolist())

print()

print("Targets:")
print(df["target"].unique())

print()

print("Number of interferers:")
print(
    sorted(
        df["num_interferers"].unique()
    )
)

print()

print("Alpha values:")
print(
    sorted(
        df["alpha"].unique()
    )
)

print()

print("SNR:")
print(
    df["fixed_snr_db"].unique()
)

print("=" * 70)


# ============================================================
# Validate columns
# ============================================================

required_columns = {

    "target",

    "num_interferers",

    "alpha",

    "fixed_snr_db",

    "psnr_ddpm",

    "ssim_ddpm",

    "latent_mse_ddpm",

    "latent_cosine_ddpm",
}


missing = required_columns - set(df.columns)

if missing:

    raise RuntimeError(
        "Missing columns:\n"
        + "\n".join(
            sorted(missing)
        )
    )


# ============================================================
# Generic plotting function
# ============================================================

def plot_metric_vs_alpha(
    df,
    target,
    metric_column,
    ylabel,
    filename,
):

    data = df[
        df["target"] == target.lower()
    ].copy()

    if data.empty:

        raise RuntimeError(
            f"No data found for target = {target}"
        )


    fig, ax = plt.subplots(
        figsize=(8.0, 5.4)
    )


    # --------------------------------------------------------
    # One curve for each number of interferers
    # --------------------------------------------------------

    interferer_values = sorted(
        data["num_interferers"].unique()
    )


    for n_interferers in interferer_values:

        subset = data[
            data["num_interferers"]
            == n_interferers
        ].copy()

        subset = subset.sort_values(
            "alpha"
        )


        ax.plot(

            subset["alpha"],

            subset[metric_column],

            color=COLORS.get(
                int(n_interferers),
                None
            ),

            marker=MARKERS.get(
                int(n_interferers),
                "o"
            ),

            linestyle=LINESTYLES.get(
                int(n_interferers),
                "-"
            ),

            linewidth=LINEWIDTH,

            markersize=MARKERSIZE,

            label=(
                f"{int(n_interferers)} "
                + (
                    "Interferer"
                    if int(n_interferers) == 1
                    else "Interferers"
                )
            ),
        )


    # --------------------------------------------------------
    # Axis
    # --------------------------------------------------------

    ax.set_xlabel(
        r"Interference Amplitude $\alpha$"
    )

    ax.set_ylabel(
        ylabel
    )


    # Force all alpha values to appear
    alpha_values = sorted(
        data["alpha"].unique()
    )

    ax.set_xticks(
        alpha_values
    )


    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    style_axis(ax)


    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="0.8",
        loc="best",
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_figure(
        fig,
        SAVE_DIR,
        filename,
    )


# ============================================================
# Apply style
# ============================================================

apply_paper_style()


# ============================================================
# EuroSAT
# ============================================================

plot_metric_vs_alpha(

    df=df,

    target="eurosat",

    metric_column="psnr_ddpm",

    ylabel="PSNR (dB)",

    filename="EuroSAT_PSNR_vs_Alpha_MultiInterferer",
)


plot_metric_vs_alpha(

    df=df,

    target="eurosat",

    metric_column="ssim_ddpm",

    ylabel="SSIM",

    filename="EuroSAT_SSIM_vs_Alpha_MultiInterferer",
)


plot_metric_vs_alpha(

    df=df,

    target="eurosat",

    metric_column="latent_mse_ddpm",

    ylabel="Latent MSE",

    filename="EuroSAT_LatentMSE_vs_Alpha_MultiInterferer",
)


plot_metric_vs_alpha(

    df=df,

    target="eurosat",

    metric_column="latent_cosine_ddpm",

    ylabel="Latent Cosine Similarity",

    filename="EuroSAT_LatentCosine_vs_Alpha_MultiInterferer",
)


# ============================================================
# DeepGlobe
# ============================================================

plot_metric_vs_alpha(

    df=df,

    target="deepglobe",

    metric_column="psnr_ddpm",

    ylabel="PSNR (dB)",

    filename="DeepGlobe_PSNR_vs_Alpha_MultiInterferer",
)


plot_metric_vs_alpha(

    df=df,

    target="deepglobe",

    metric_column="ssim_ddpm",

    ylabel="SSIM",

    filename="DeepGlobe_SSIM_vs_Alpha_MultiInterferer",
)


plot_metric_vs_alpha(

    df=df,

    target="deepglobe",

    metric_column="latent_mse_ddpm",

    ylabel="Latent MSE",

    filename="DeepGlobe_LatentMSE_vs_Alpha_MultiInterferer",
)


plot_metric_vs_alpha(

    df=df,

    target="deepglobe",

    metric_column="latent_cosine_ddpm",

    ylabel="Latent Cosine Similarity",

    filename="DeepGlobe_LatentCosine_vs_Alpha_MultiInterferer",
)


# ============================================================
# Finished
# ============================================================

print()
print("=" * 70)
print("All figures completed.")
print("Saved to:")
print(SAVE_DIR)
print("=" * 70)