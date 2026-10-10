from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Thesis-friendly plotting style
# ============================================================

# Soft, readable, color-blind-friendly palette


COLORS = {
    # Main three curves
    "clean": "#7C8790",      # gray-blue
    "ddpm": "#6B67C9",       # soft violet-blue
    "no_sep": "#E9896A",     # soft coral-orange

    # Latent-space plots
    "mse": "#E9896A",
    "cosine": "#6B67C9",

    # Dataset comparison
    "eurosat": "#6B67C9",
    "deepglobe": "#E9896A",
}

LINEWIDTH = 2.2
MARKERSIZE = 5.5
GRID_ALPHA = 0.22


def apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# def style_axis(ax):
#     ax.grid(True, linestyle="--", linewidth=0.7, alpha=GRID_ALPHA)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.tick_params(direction="out", length=4, width=0.8)

def style_axis(ax):
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=GRID_ALPHA
    )

    # Keep all four borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

    ax.tick_params(
        direction="out",
        length=4,
        width=0.8
    )


def save_figure(fig, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(
        output_dir / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / f"{filename}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def validate_columns(df: pd.DataFrame):
    required = {
        "target",
        "alpha",
        "psnr_clean",
        "psnr_no_separation",
        "psnr_ddpm",
        "ssim_clean",
        "ssim_no_separation",
        "ssim_ddpm",
        "latent_mse_ddpm",
        "latent_cosine_ddpm",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "CSV is missing required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
            + "\n\nAvailable columns:\n"
            + ", ".join(df.columns)
        )


def prepare_domain(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    part = df[df["target"].str.lower() == domain.lower()].copy()

    if part.empty:
        raise RuntimeError(
            f"No rows found for target='{domain}'. "
            f"Available target values: {df['target'].unique().tolist()}"
        )

    part = part.sort_values("alpha")
    return part


# ============================================================
# Single-domain PSNR
# ============================================================

def plot_psnr(df: pd.DataFrame, domain: str, output_dir: Path):
    d = prepare_domain(df, domain)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        d["alpha"],
        d["psnr_clean"],
        color=COLORS["clean"],
        linestyle="--",
        marker="s",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        label="Clean Latent",
    )

    ax.plot(
        d["alpha"],
        d["psnr_no_separation"],
        color=COLORS["no_sep"],
        linestyle=":",
        marker="D",
        markersize=MARKERSIZE - 0.5,
        linewidth=LINEWIDTH,
        label="No Separation",
    )

    ax.plot(
        d["alpha"],
        d["psnr_ddpm"],
        color=COLORS["ddpm"],
        linestyle="-",
        marker="o",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH + 0.2,
        label="Proposed DDPM",
    )

    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("PSNR (dB)")
    # ax.set_title(f"{domain.capitalize()} — PSNR vs. Interference Strength")

    style_axis(ax)
    ax.legend(frameon=False)

    save_figure(
        fig,
        output_dir,
        f"{domain.lower()}_PSNR_vs_alpha_pretty",
    )


# ============================================================
# Single-domain SSIM
# ============================================================

def plot_ssim(df: pd.DataFrame, domain: str, output_dir: Path):
    d = prepare_domain(df, domain)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        d["alpha"],
        d["ssim_clean"],
        color=COLORS["clean"],
        linestyle="--",
        marker="s",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        label="Clean Latent",
    )

    ax.plot(
        d["alpha"],
        d["ssim_no_separation"],
        color=COLORS["no_sep"],
        linestyle=":",
        marker="D",
        markersize=MARKERSIZE - 0.5,
        linewidth=LINEWIDTH,
        label="No Separation",
    )

    ax.plot(
        d["alpha"],
        d["ssim_ddpm"],
        color=COLORS["ddpm"],
        linestyle="-",
        marker="o",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH + 0.2,
        label="Proposed DDPM",
    )

    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("SSIM")
    # ax.set_title(f"{domain.capitalize()} — SSIM vs. Interference Strength")

    style_axis(ax)
    ax.legend(frameon=False)

    save_figure(
        fig,
        output_dir,
        f"{domain.lower()}_SSIM_vs_alpha_pretty",
    )


# ============================================================
# Latent MSE
# ============================================================

def plot_latent_mse(df: pd.DataFrame, domain: str, output_dir: Path):
    d = prepare_domain(df, domain)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        d["alpha"],
        d["latent_mse_ddpm"],
        color=COLORS["mse"],
        linestyle="-",
        marker="o",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH + 0.2,
    )

    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("Latent MSE")
    # ax.set_title(f"{domain.capitalize()} — Latent MSE vs. Interference Strength")

    style_axis(ax)

    save_figure(
        fig,
        output_dir,
        f"{domain.lower()}_latent_MSE_vs_alpha_pretty",
    )


# ============================================================
# Latent cosine similarity
# ============================================================

def plot_latent_cosine(df: pd.DataFrame, domain: str, output_dir: Path):
    d = prepare_domain(df, domain)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(
        d["alpha"],
        d["latent_cosine_ddpm"],
        color=COLORS["cosine"],
        linestyle="-",
        marker="o",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH + 0.2,
    )

    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("Latent Cosine Similarity")
    # ax.set_title(
    #     f"{domain.capitalize()} — Latent Cosine Similarity vs. Interference Strength"
    # )

    style_axis(ax)

    save_figure(
        fig,
        output_dir,
        f"{domain.lower()}_latent_cosine_vs_alpha_pretty",
    )


# ============================================================
# Thesis-style 2-panel PSNR figure
# ============================================================

def plot_psnr_two_panel(df: pd.DataFrame, output_dir: Path):
    euro = prepare_domain(df, "eurosat")
    deep = prepare_domain(df, "deepglobe")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.2),
        sharey=False,
    )

    for ax, d, title in [
        (axes[0], euro, "(a) EuroSAT"),
        (axes[1], deep, "(b) DeepGlobe"),
    ]:
        ax.plot(
            d["alpha"],
            d["psnr_clean"],
            color=COLORS["clean"],
            linestyle="--",
            marker="s",
            markersize=4.8,
            linewidth=2.0,
            label="Clean Latent",
        )

        ax.plot(
            d["alpha"],
            d["psnr_no_separation"],
            color=COLORS["no_sep"],
            linestyle=":",
            marker="D",
            markersize=4.4,
            linewidth=2.0,
            label="No Separation",
        )

        ax.plot(
            d["alpha"],
            d["psnr_ddpm"],
            color=COLORS["ddpm"],
            linestyle="-",
            marker="o",
            markersize=4.8,
            linewidth=2.3,
            label="Proposed DDPM",
        )

        ax.set_xlabel(r"Interference amplitude $\alpha$")
        # ax.set_title(title)
        style_axis(ax)

    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend(frameon=False, loc="best")

    save_figure(
        fig,
        output_dir,
        "PSNR_vs_alpha_two_panel_pretty",
    )


# ============================================================
# Thesis-style 2-panel SSIM figure
# ============================================================

def plot_ssim_two_panel(df: pd.DataFrame, output_dir: Path):
    euro = prepare_domain(df, "eurosat")
    deep = prepare_domain(df, "deepglobe")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.2),
        sharey=False,
    )

    for ax, d, title in [
        (axes[0], euro, "(a) EuroSAT"),
        (axes[1], deep, "(b) DeepGlobe"),
    ]:
        ax.plot(
            d["alpha"],
            d["ssim_clean"],
            color=COLORS["clean"],
            linestyle="--",
            marker="s",
            markersize=4.8,
            linewidth=2.0,
            label="Clean Latent",
        )

        ax.plot(
            d["alpha"],
            d["ssim_no_separation"],
            color=COLORS["no_sep"],
            linestyle=":",
            marker="D",
            markersize=4.4,
            linewidth=2.0,
            label="No Separation",
        )

        ax.plot(
            d["alpha"],
            d["ssim_ddpm"],
            color=COLORS["ddpm"],
            linestyle="-",
            marker="o",
            markersize=4.8,
            linewidth=2.3,
            label="Proposed DDPM",
        )

        ax.set_xlabel(r"Interference amplitude $\alpha$")
        # ax.set_title(title)
        style_axis(ax)

    axes[0].set_ylabel("SSIM")
    axes[0].legend(frameon=False, loc="best")

    save_figure(
        fig,
        output_dir,
        "SSIM_vs_alpha_two_panel_pretty",
    )


# ============================================================
# Direct comparison: DDPM vs No Separation gain
# ============================================================

def plot_ddpm_gain(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for domain, color, marker in [
        ("eurosat", COLORS["eurosat"], "o"),
        ("deepglobe", COLORS["deepglobe"], "s"),
    ]:
        d = prepare_domain(df, domain)
        gain = d["psnr_ddpm"] - d["psnr_no_separation"]

        ax.plot(
            d["alpha"],
            gain,
            color=color,
            marker=marker,
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH + 0.2,
            label=domain.capitalize(),
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"Interference amplitude $\alpha$")
    ax.set_ylabel("PSNR Gain over No Separation (dB)")
    # ax.set_title("DDPM Separation Gain vs. Interference Strength")

    style_axis(ax)
    ax.legend(frameon=False)

    save_figure(
        fig,
        output_dir,
        "DDPM_PSNR_gain_vs_alpha_pretty",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        type=str,
        default=(
            "./save_data/Deep_WGANGP_VAE_DDPM/"
            "random_alpha_model_test/alpha_sweep_snr10/"
            "random_alpha_ddpm_alpha_sweep_metrics.csv"
        ),
        help="Path to random_alpha_ddpm_alpha_sweep_metrics.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "./save_data/Deep_WGANGP_VAE_DDPM/"
            "random_alpha_model_test/alpha_sweep_snr10/"
            "pretty_plots"
        ),
        help="Folder for generated PNG/PDF figures",
    )

    args = parser.parse_args()

    apply_paper_style()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{csv_path.resolve()}"
        )

    df = pd.read_csv(csv_path)
    validate_columns(df)

    # Make target strings consistent.
    df["target"] = df["target"].astype(str).str.strip().str.lower()

    print("=" * 72)
    print("Random-alpha DDPM CSV plotting")
    print("=" * 72)
    print(f"CSV         : {csv_path.resolve()}")
    print(f"Output      : {output_dir.resolve()}")
    print(f"Rows        : {len(df)}")
    print(f"Targets     : {df['target'].unique().tolist()}")
    print(
        "Alpha range : "
        f"{df['alpha'].min():.2f} -> {df['alpha'].max():.2f}"
    )

    # EuroSAT
    plot_psnr(df, "eurosat", output_dir)
    plot_ssim(df, "eurosat", output_dir)
    plot_latent_mse(df, "eurosat", output_dir)
    plot_latent_cosine(df, "eurosat", output_dir)

    # DeepGlobe
    plot_psnr(df, "deepglobe", output_dir)
    plot_ssim(df, "deepglobe", output_dir)
    plot_latent_mse(df, "deepglobe", output_dir)
    plot_latent_cosine(df, "deepglobe", output_dir)

    # Thesis-ready combined figures
    plot_psnr_two_panel(df, output_dir)
    plot_ssim_two_panel(df, output_dir)

    # Additional useful figure
    plot_ddpm_gain(df, output_dir)

    print("\nCompleted.")
    print(f"Figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
