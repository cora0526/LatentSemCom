import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

CSV_PATH = (
    r"D:\VAE_GAN_DM\save_data\Deep_WGANGP_VAE_DDPM"
    r"\ddpm_label_cond_siso_random_alpha_training_history.csv"
)

SAVE_DIR = (
    r"D:\VAE_GAN_DM\save_data\Deep_WGANGP_VAE_DDPM"
    r"\convergence_plot"
)

os.makedirs(SAVE_DIR, exist_ok=True)

PNG_PATH = os.path.join(
    SAVE_DIR,
    "DDPM_Training_Convergence.png"
)

PDF_PATH = os.path.join(
    SAVE_DIR,
    "DDPM_Training_Convergence.pdf"
)


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip()

print("Columns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head())


# ============================================================
# Read data
# ============================================================

epoch = df["epoch"]

total_loss = df["total_loss"]
eps_eurosat = df["eps_eurosat"]
eps_deepglobe = df["eps_deepglobe"]


# ============================================================
# Optional smoothing
# ============================================================
# 你的 CSV 是 epoch-average loss，
# 所以不需要很重的 smoothing。
# 這裡使用 3-epoch moving average。

SMOOTH = True
WINDOW = 3

if SMOOTH:
    total_plot = total_loss.rolling(
        window=WINDOW,
        min_periods=1,
        center=True
    ).mean()

    eurosat_plot = eps_eurosat.rolling(
        window=WINDOW,
        min_periods=1,
        center=True
    ).mean()

    deepglobe_plot = eps_deepglobe.rolling(
        window=WINDOW,
        min_periods=1,
        center=True
    ).mean()

else:
    total_plot = total_loss
    eurosat_plot = eps_eurosat
    deepglobe_plot = eps_deepglobe


# ============================================================
# Colors
# ============================================================

COLOR_TOTAL     = "#4C72B0"   # 深藍
COLOR_EUROSAT   = "#C44E52"   # 磚紅
COLOR_DEEPGLOBE = "#55A868"


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.0, 5.4)
)


# ------------------------------------------------------------
# Total loss
# ------------------------------------------------------------

ax.plot(
    epoch,
    total_plot,
    color=COLOR_TOTAL,
    linewidth=2.3,
    linestyle="-",
    label="Total Loss"
)


# ------------------------------------------------------------
# EuroSAT epsilon loss
# ------------------------------------------------------------

ax.plot(
    epoch,
    eurosat_plot,
    color=COLOR_EUROSAT,
    linewidth=2.0,
    linestyle="--",
    label=r"EuroSAT $\epsilon$-Loss"
)


# ------------------------------------------------------------
# DeepGlobe epsilon loss
# ------------------------------------------------------------

ax.plot(
    epoch,
    deepglobe_plot,
    color=COLOR_DEEPGLOBE,
    linewidth=2.0,
    linestyle="-.",
    label=r"DeepGlobe $\epsilon$-Loss"
)


# ============================================================
# Axis labels
# ============================================================

ax.set_xlabel(
    "Epoch",
    fontsize=18
)

ax.set_ylabel(
    "Training Loss",
    fontsize=18
)


# ============================================================
# Tick settings
# ============================================================

ax.tick_params(
    axis="both",
    which="major",
    labelsize=16
)


# ============================================================
# Grid
# ============================================================

ax.grid(
    True,
    linestyle="-",
    linewidth=0.7,
    alpha=0.22
)


# ============================================================
# Border
# ============================================================

for spine in ax.spines.values():
    spine.set_linewidth(1.0)


# ============================================================
# Legend
# ============================================================

ax.legend(
    loc="upper right",
    fontsize=16,
    frameon=True,
    fancybox=False,
    edgecolor="0.8"
)


# ============================================================
# IMPORTANT:
# 不加 plt.title()
#
# 論文 Figure title 交給 LaTeX caption
# ============================================================


# ============================================================
# Layout
# ============================================================

plt.tight_layout()


# ============================================================
# Save
# ============================================================

plt.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    PDF_PATH,
    bbox_inches="tight"
)


# ============================================================
# Show
# ============================================================

plt.show()


print("\nSaved:")
print(PNG_PATH)
print(PDF_PATH)

# ============================================================
# x0 Reconstruction Loss
# ============================================================

x0_eurosat = df["x0_eurosat"]
x0_deepglobe = df["x0_deepglobe"]

x0_eurosat_plot = x0_eurosat.rolling(
    window=3,
    min_periods=1,
    center=True
).mean()

x0_deepglobe_plot = x0_deepglobe.rolling(
    window=3,
    min_periods=1,
    center=True
).mean()


fig, ax = plt.subplots(
    figsize=(8.0, 5.4)
)

ax.plot(
    epoch,
    x0_eurosat_plot,
    color="#DD8452",
    linewidth=2.2,
    linestyle="--",
    label=r"EuroSAT $x_0$-Loss"
)

ax.plot(
    epoch,
    x0_deepglobe_plot,
    color="#6666BB",
    linewidth=2.2,
    linestyle="-",
    label=r"DeepGlobe $x_0$-Loss"
)

ax.set_xlabel(
    "Epoch",
    fontsize=18
)

ax.set_ylabel(
    r"$x_0$ Reconstruction Loss",
    fontsize=18
)

ax.tick_params(
    axis="both",
    labelsize=16
)

ax.grid(
    True,
    linewidth=0.7,
    alpha=0.22
)

ax.legend(
    fontsize=16,
    frameon=True,
    fancybox=False,
    edgecolor="0.8"
)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "DDPM_x0_Loss.pdf"
    ),
    bbox_inches="tight"
)

plt.savefig(
    os.path.join(
        SAVE_DIR,
        "DDPM_x0_Loss.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()