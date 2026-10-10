"""
Plot evaluation results saved by evaluate_deepglobe_four_separators.py.

Expected input files
--------------------
summary_metrics.csv
per_image_metrics.csv

The script automatically reads the columns in the provided CSV format and
generates:

1. Mean ± standard deviation bar charts from summary_metrics.csv
2. Per-image boxplots from per_image_metrics.csv
3. Latent-quality versus image-quality scatter plots
4. A normalized multi-metric overview plot
5. A compact ranking CSV

No seaborn is required.

Install dependencies:
    pip install pandas numpy matplotlib

Usage:
    python plot_saved_metrics.py

Place this script in the same folder as the two CSV files, or modify Config.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    summary_csv: str = "./summary_metrics.csv"
    per_image_csv: str = "./per_image_metrics.csv"
    output_dir: str = "./metric_plots"

    dpi: int = 220
    figure_width: float = 10.0
    figure_height: float = 6.0

    # Save vector PDF copies in addition to PNG.
    save_pdf: bool = True

    # Add numeric values above summary bars.
    annotate_bars: bool = True

    # Number of bins for per-image histograms.
    histogram_bins: int = 15


# ============================================================
# Display names and metric definitions
# ============================================================

METHOD_ORDER = (
    "VAE-DDPM-VAE",
    "VAE-MLP-VAE",
    "VAE-DDPM-GAN",
    "VAE-MLP-GAN",
)

# Prefixes used by per_image_metrics.csv.
METHOD_PREFIXES = {
    "VAE-DDPM-VAE": "VAE_DDPM_VAE",
    "VAE-MLP-VAE": "VAE_MLP_VAE",
    "VAE-DDPM-GAN": "VAE_DDPM_GAN",
    "VAE-MLP-GAN": "VAE_MLP_GAN",
}

# Summary CSV metric:
# (mean column, std column, y-axis label, higher is better)
SUMMARY_METRICS: Dict[str, Tuple[str, str, str, bool]] = {
    "PSNR_real": (
        "psnr_real_mean",
        "psnr_real_std",
        "PSNR vs real image (dB)",
        True,
    ),
    "SSIM_real": (
        "ssim_real_mean",
        "ssim_real_std",
        "SSIM vs real image",
        True,
    ),
    "LPIPS_real": (
        "lpips_real_mean",
        "lpips_real_std",
        "LPIPS vs real image",
        False,
    ),
    "Image_MAE": (
        "image_mae_real_mean",
        "image_mae_real_std",
        "Image MAE",
        False,
    ),
    "Image_MSE": (
        "image_mse_real_mean",
        "image_mse_real_std",
        "Image MSE",
        False,
    ),
    "Latent_MSE": (
        "latent_mse_mean",
        "latent_mse_std",
        "Latent MSE",
        False,
    ),
    "Latent_NMSE": (
        "latent_nmse_mean",
        "latent_nmse_std",
        "Latent NMSE",
        False,
    ),
    "Latent_MAE": (
        "latent_mae_mean",
        "latent_mae_std",
        "Latent MAE",
        False,
    ),
    "Latent_Cosine": (
        "latent_cosine_mean",
        "latent_cosine_std",
        "Latent cosine similarity",
        True,
    ),
    "PSNR_oracle": (
        "psnr_oracle_mean",
        "psnr_oracle_std",
        "PSNR vs oracle reconstruction (dB)",
        True,
    ),
    "SSIM_oracle": (
        "ssim_oracle_mean",
        "ssim_oracle_std",
        "SSIM vs oracle reconstruction",
        True,
    ),
    "Oracle_PSNR_real": (
        "oracle_psnr_real_mean",
        "oracle_psnr_real_std",
        "Oracle PSNR vs real image (dB)",
        True,
    ),
    "Oracle_SSIM_real": (
        "oracle_ssim_real_mean",
        "oracle_ssim_real_std",
        "Oracle SSIM vs real image",
        True,
    ),
}

# Per-image suffix:
# (display name, y-axis label)
PER_IMAGE_METRICS: Dict[str, Tuple[str, str]] = {
    "psnr_real": (
        "PSNR_real",
        "PSNR vs real image (dB)",
    ),
    "ssim_real": (
        "SSIM_real",
        "SSIM vs real image",
    ),
    "lpips_real": (
        "LPIPS_real",
        "LPIPS vs real image",
    ),
    "image_mae_real": (
        "Image_MAE",
        "Image MAE",
    ),
    "image_mse_real": (
        "Image_MSE",
        "Image MSE",
    ),
    "latent_mse": (
        "Latent_MSE",
        "Latent MSE",
    ),
    "latent_nmse": (
        "Latent_NMSE",
        "Latent NMSE",
    ),
    "latent_mae": (
        "Latent_MAE",
        "Latent MAE",
    ),
    "latent_cosine": (
        "Latent_Cosine",
        "Latent cosine similarity",
    ),
    "psnr_oracle": (
        "PSNR_oracle",
        "PSNR vs oracle reconstruction (dB)",
    ),
    "ssim_oracle": (
        "SSIM_oracle",
        "SSIM vs oracle reconstruction",
    ),
}


# ============================================================
# General utilities
# ============================================================

def safe_filename(text: str) -> str:
    return (
        text.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def ordered_methods(summary: pd.DataFrame) -> List[str]:
    available = set(summary["method"].astype(str))

    ordered = [
        method
        for method in METHOD_ORDER
        if method in available
    ]

    extras = sorted(available.difference(ordered))
    return ordered + extras


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    filename: str,
    config: Config,
) -> None:
    png_path = output_dir / f"{filename}.png"

    figure.savefig(
        png_path,
        dpi=config.dpi,
        bbox_inches="tight",
    )

    if config.save_pdf:
        pdf_path = output_dir / f"{filename}.pdf"
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )

    plt.close(figure)


def validate_inputs(
    summary_path: Path,
    per_image_path: Path,
) -> None:
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"summary_metrics.csv does not exist:\n{summary_path}"
        )

    if not per_image_path.is_file():
        raise FileNotFoundError(
            f"per_image_metrics.csv does not exist:\n{per_image_path}"
        )


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()


# ============================================================
# Summary bar charts
# ============================================================

def plot_summary_bar(
    summary: pd.DataFrame,
    metric_name: str,
    mean_column: str,
    std_column: str,
    ylabel: str,
    output_dir: Path,
    config: Config,
) -> None:
    if mean_column not in summary.columns:
        print(
            f"Skipping {metric_name}: "
            f"missing column {mean_column}"
        )
        return

    methods = ordered_methods(summary)
    selected = (
        summary.set_index("method")
        .reindex(methods)
    )

    means = pd.to_numeric(
        selected[mean_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    if std_column in selected.columns:
        stds = pd.to_numeric(
            selected[std_column],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=float)
    else:
        stds = np.zeros_like(means)

    valid = np.isfinite(means)
    methods = [
        method
        for method, keep in zip(methods, valid)
        if keep
    ]
    means = means[valid]
    stds = stds[valid]

    if len(methods) == 0:
        return

    figure, axis = plt.subplots(
        figsize=(
            config.figure_width,
            config.figure_height,
        )
    )

    positions = np.arange(len(methods))

    bars = axis.bar(
        positions,
        means,
        yerr=stds,
        capsize=6,
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        methods,
        rotation=20,
        ha="right",
    )
    axis.set_ylabel(ylabel)
    axis.set_title(
        f"{ylabel}: mean ± standard deviation"
    )
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    if config.annotate_bars:
        finite_values = means[np.isfinite(means)]

        if len(finite_values) > 0:
            value_range = (
                finite_values.max()
                - finite_values.min()
            )
            offset = (
                value_range * 0.025
                if value_range > 0
                else max(
                    abs(finite_values.max()) * 0.025,
                    0.01,
                )
            )

            for bar, mean, std in zip(
                bars,
                means,
                stds,
            ):
                axis.text(
                    bar.get_x()
                    + bar.get_width() / 2.0,
                    mean + std + offset,
                    f"{mean:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_dir=output_dir,
        filename=f"bar_{metric_name}",
        config=config,
    )


# ============================================================
# Per-image distribution charts
# ============================================================

def collect_per_image_values(
    per_image: pd.DataFrame,
    suffix: str,
) -> Tuple[List[str], List[np.ndarray]]:
    labels: List[str] = []
    values: List[np.ndarray] = []

    for method in METHOD_ORDER:
        prefix = METHOD_PREFIXES[method]
        column = f"{prefix}_{suffix}"

        if column not in per_image.columns:
            continue

        data = numeric_series(
            per_image,
            column,
        ).to_numpy(dtype=float)

        if data.size == 0:
            continue

        labels.append(method)
        values.append(data)

    return labels, values


def plot_boxplot(
    per_image: pd.DataFrame,
    suffix: str,
    display_name: str,
    ylabel: str,
    output_dir: Path,
    config: Config,
) -> None:
    labels, values = collect_per_image_values(
        per_image,
        suffix,
    )

    if not values:
        print(
            f"Skipping boxplot {display_name}: "
            "no matching columns"
        )
        return

    figure, axis = plt.subplots(
        figsize=(
            config.figure_width,
            config.figure_height,
        )
    )

    axis.boxplot(
        values,
        labels=labels,
        showmeans=True,
    )

    axis.set_ylabel(ylabel)
    axis.set_title(
        f"Per-image distribution: {ylabel}"
    )
    axis.tick_params(
        axis="x",
        labelrotation=20,
    )
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_dir=output_dir,
        filename=f"boxplot_{display_name}",
        config=config,
    )


def plot_histogram(
    per_image: pd.DataFrame,
    suffix: str,
    display_name: str,
    xlabel: str,
    output_dir: Path,
    config: Config,
) -> None:
    labels, values = collect_per_image_values(
        per_image,
        suffix,
    )

    if not values:
        return

    figure, axis = plt.subplots(
        figsize=(
            config.figure_width,
            config.figure_height,
        )
    )

    for label, data in zip(labels, values):
        axis.hist(
            data,
            bins=config.histogram_bins,
            alpha=0.45,
            label=label,
        )

    axis.set_xlabel(xlabel)
    axis.set_ylabel("Image count")
    axis.set_title(
        f"Per-image histogram: {xlabel}"
    )
    axis.grid(
        axis="y",
        alpha=0.3,
    )
    axis.legend()

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_dir=output_dir,
        filename=f"histogram_{display_name}",
        config=config,
    )


# ============================================================
# Scatter plots
# ============================================================

def plot_latent_vs_image_scatter(
    per_image: pd.DataFrame,
    x_suffix: str,
    y_suffix: str,
    x_label: str,
    y_label: str,
    filename: str,
    output_dir: Path,
    config: Config,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            config.figure_width,
            config.figure_height,
        )
    )

    plotted = False

    for method in METHOD_ORDER:
        prefix = METHOD_PREFIXES[method]
        x_column = f"{prefix}_{x_suffix}"
        y_column = f"{prefix}_{y_suffix}"

        if (
            x_column not in per_image.columns
            or y_column not in per_image.columns
        ):
            continue

        paired = per_image[
            [x_column, y_column]
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()

        if paired.empty:
            continue

        axis.scatter(
            paired[x_column],
            paired[y_column],
            alpha=0.65,
            label=method,
        )

        # Add one linear trend line for each method.
        if len(paired) >= 2:
            x_values = paired[x_column].to_numpy(
                dtype=float
            )
            y_values = paired[y_column].to_numpy(
                dtype=float
            )

            slope, intercept = np.polyfit(
                x_values,
                y_values,
                1,
            )

            x_line = np.linspace(
                x_values.min(),
                x_values.max(),
                100,
            )
            y_line = slope * x_line + intercept

            axis.plot(
                x_line,
                y_line,
                linewidth=1.5,
            )

        plotted = True

    if not plotted:
        plt.close(figure)
        return

    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(
        f"{y_label} versus {x_label}"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_dir=output_dir,
        filename=filename,
        config=config,
    )


# ============================================================
# Normalized overview
# ============================================================

def min_max_normalize(
    values: np.ndarray,
    higher_is_better: bool,
) -> np.ndarray:
    values = values.astype(float)

    finite = np.isfinite(values)

    if not finite.any():
        return np.full_like(values, np.nan)

    minimum = np.nanmin(values)
    maximum = np.nanmax(values)

    if math.isclose(minimum, maximum):
        normalized = np.ones_like(values)
    else:
        normalized = (
            values - minimum
        ) / (maximum - minimum)

    if not higher_is_better:
        normalized = 1.0 - normalized

    return normalized


def plot_normalized_overview(
    summary: pd.DataFrame,
    output_dir: Path,
    config: Config,
) -> pd.DataFrame:
    """
    Convert selected metrics to a 0–1 score where higher is always better.

    This is only a visual overview. It should not replace the original metrics
    in a scientific report because min-max normalization depends on the methods
    included in the comparison.
    """
    selected_metrics = {
        "PSNR": (
            "psnr_real_mean",
            True,
        ),
        "SSIM": (
            "ssim_real_mean",
            True,
        ),
        "LPIPS": (
            "lpips_real_mean",
            False,
        ),
        "Latent cosine": (
            "latent_cosine_mean",
            True,
        ),
        "Latent NMSE": (
            "latent_nmse_mean",
            False,
        ),
    }

    methods = ordered_methods(summary)

    frame = (
        summary.set_index("method")
        .reindex(methods)
    )

    score_data: Dict[str, np.ndarray] = {}

    for label, (
        column,
        higher_is_better,
    ) in selected_metrics.items():
        if column not in frame.columns:
            continue

        raw = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        score_data[label] = min_max_normalize(
            raw,
            higher_is_better,
        )

    if not score_data:
        return pd.DataFrame()

    score_frame = pd.DataFrame(
        score_data,
        index=methods,
    )

    score_frame["Overall mean score"] = (
        score_frame.mean(axis=1)
    )

    figure, axis = plt.subplots(
        figsize=(
            config.figure_width,
            config.figure_height,
        )
    )

    positions = np.arange(len(methods))
    width = 0.8 / len(score_data)

    for metric_index, (
        metric_name,
        values,
    ) in enumerate(score_data.items()):
        axis.bar(
            positions
            + (
                metric_index
                - (len(score_data) - 1) / 2
            )
            * width,
            values,
            width=width,
            label=metric_name,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        methods,
        rotation=20,
        ha="right",
    )
    axis.set_ylim(0.0, 1.08)
    axis.set_ylabel(
        "Normalized score (higher is better)"
    )
    axis.set_title(
        "Normalized multi-metric overview"
    )
    axis.grid(
        axis="y",
        alpha=0.3,
    )
    axis.legend(
        ncol=2,
    )

    figure.tight_layout()

    save_figure(
        figure=figure,
        output_dir=output_dir,
        filename="normalized_metric_overview",
        config=config,
    )

    return score_frame


# ============================================================
# Ranking table
# ============================================================

def create_ranking_csv(
    summary: pd.DataFrame,
    normalized_scores: pd.DataFrame,
    output_path: Path,
) -> None:
    methods = ordered_methods(summary)

    frame = (
        summary.set_index("method")
        .reindex(methods)
    )

    columns = [
        "psnr_real_mean",
        "ssim_real_mean",
        "lpips_real_mean",
        "latent_mse_mean",
        "latent_nmse_mean",
        "latent_cosine_mean",
    ]

    available_columns = [
        column
        for column in columns
        if column in frame.columns
    ]

    output = frame[available_columns].copy()

    if not normalized_scores.empty:
        output["normalized_overall_score"] = (
            normalized_scores[
                "Overall mean score"
            ]
        )

        output = output.sort_values(
            "normalized_overall_score",
            ascending=False,
        )

    output.insert(
        0,
        "rank",
        np.arange(1, len(output) + 1),
    )

    output.to_csv(
        output_path,
        encoding="utf-8-sig",
    )


# ============================================================
# Console summary
# ============================================================

def print_summary(
    summary: pd.DataFrame,
) -> None:
    columns = [
        "method",
        "psnr_real_mean",
        "ssim_real_mean",
        "lpips_real_mean",
        "latent_mse_mean",
        "latent_cosine_mean",
    ]

    available = [
        column
        for column in columns
        if column in summary.columns
    ]

    print("\nSummary metrics")
    print("-" * 100)
    print(
        summary[available]
        .to_string(index=False)
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    config = Config()

    summary_path = Path(config.summary_csv)
    per_image_path = Path(config.per_image_csv)
    output_dir = Path(config.output_dir)

    validate_inputs(
        summary_path,
        per_image_path,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.read_csv(summary_path)
    per_image = pd.read_csv(per_image_path)

    if "method" not in summary.columns:
        raise KeyError(
            "summary_metrics.csv must contain a 'method' column."
        )

    print(f"Summary rows: {len(summary)}")
    print(f"Per-image rows: {len(per_image)}")
    print_summary(summary)

    # --------------------------------------------------------
    # 1. Mean ± std bar chart for every summary metric
    # --------------------------------------------------------
    for metric_name, (
        mean_column,
        std_column,
        ylabel,
        _higher_is_better,
    ) in SUMMARY_METRICS.items():
        plot_summary_bar(
            summary=summary,
            metric_name=metric_name,
            mean_column=mean_column,
            std_column=std_column,
            ylabel=ylabel,
            output_dir=output_dir,
            config=config,
        )

    # --------------------------------------------------------
    # 2. Per-image boxplot and histogram
    # --------------------------------------------------------
    for suffix, (
        display_name,
        ylabel,
    ) in PER_IMAGE_METRICS.items():
        plot_boxplot(
            per_image=per_image,
            suffix=suffix,
            display_name=display_name,
            ylabel=ylabel,
            output_dir=output_dir,
            config=config,
        )

        plot_histogram(
            per_image=per_image,
            suffix=suffix,
            display_name=display_name,
            xlabel=ylabel,
            output_dir=output_dir,
            config=config,
        )

    # --------------------------------------------------------
    # 3. Relationship scatter plots
    # --------------------------------------------------------
    plot_latent_vs_image_scatter(
        per_image=per_image,
        x_suffix="latent_mse",
        y_suffix="psnr_real",
        x_label="Latent MSE",
        y_label="PSNR vs real image (dB)",
        filename="scatter_latent_MSE_vs_PSNR",
        output_dir=output_dir,
        config=config,
    )

    plot_latent_vs_image_scatter(
        per_image=per_image,
        x_suffix="latent_cosine",
        y_suffix="ssim_real",
        x_label="Latent cosine similarity",
        y_label="SSIM vs real image",
        filename="scatter_latent_cosine_vs_SSIM",
        output_dir=output_dir,
        config=config,
    )

    plot_latent_vs_image_scatter(
        per_image=per_image,
        x_suffix="latent_nmse",
        y_suffix="lpips_real",
        x_label="Latent NMSE",
        y_label="LPIPS vs real image",
        filename="scatter_latent_NMSE_vs_LPIPS",
        output_dir=output_dir,
        config=config,
    )

    # --------------------------------------------------------
    # 4. Normalized overview and ranking
    # --------------------------------------------------------
    normalized_scores = plot_normalized_overview(
        summary=summary,
        output_dir=output_dir,
        config=config,
    )

    create_ranking_csv(
        summary=summary,
        normalized_scores=normalized_scores,
        output_path=output_dir / "method_ranking.csv",
    )

    if not normalized_scores.empty:
        normalized_scores.to_csv(
            output_dir
            / "normalized_metric_scores.csv",
            encoding="utf-8-sig",
        )

    print("\nPlot generation completed.")
    print(f"Results saved to:\n{output_dir.resolve()}")
    print("\nMain plot groups:")
    print("  bar_*.png / bar_*.pdf")
    print("  boxplot_*.png / boxplot_*.pdf")
    print("  histogram_*.png / histogram_*.pdf")
    print("  scatter_*.png / scatter_*.pdf")
    print("  normalized_metric_overview.png / .pdf")
    print("  method_ranking.csv")
    print("  normalized_metric_scores.csv")


if __name__ == "__main__":
    main()
