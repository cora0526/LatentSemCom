import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from utils import WaveformDataset
from torch.utils.data import DataLoader
from model import DDPMWaveDenoiser1D
import os
device = "cuda"

# ===== dataset =====
train_m = WaveformDataset(
    r"D:/VAE_GAN_DM/latent_bits_M/bits_8/latent_tran_encode"
)

train_f = WaveformDataset(
    r"D:/VAE_GAN_DM/latent_bits_F/bits_8/latent_tran_encode"
)

train_dl_m = DataLoader(
    train_m,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

train_dl_f = DataLoader(
    train_f,
    batch_size=4,
    shuffle=True,
    num_workers=0
)
# ===== DDPM schedule =====
T = 1000

betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

model = DDPMWaveDenoiser1D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print("num files =", len(train_f.files))
print("start training")


loss_history = []

for epoch in range(200):
    print("epoch start", epoch)
    model.train()

    total_loss = 0

    for batch_idx, ((x_m, target), (x_f, interference)) in enumerate(zip(train_dl_m, train_dl_f)):

        interference = interference.to(device) 
        target = target.to(device)

        B = target.shape[0]

        # normalize interference power
        interference = interference / (
            torch.sqrt(torch.mean(interference ** 2, dim=(1, 2), keepdim=True)) + 1e-8
        )

        # normalize target power
        target = target / (
            torch.sqrt(torch.mean(target ** 2, dim=(1, 2), keepdim=True)) + 1e-8
        )

        # random timestep
        t = torch.randint(0, T, (B,), device=device)

        sqrt_ab = torch.sqrt(alpha_bars[t])[:, None, None]
        sqrt_1mab = torch.sqrt(1.0 - alpha_bars[t])[:, None, None]

        # DDPM forward process
        x_t = sqrt_ab * target + sqrt_1mab * interference

        # model predicts interference/noise
        pred_noise = model(x_t, t)

        pred_clean_scaled = x_t - sqrt_1mab * pred_noise
        target_scaled = sqrt_ab * target

        loss = criterion(pred_clean_scaled, target_scaled)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (batch_idx + 1)
    loss_history.append(avg_loss)

    print(epoch, avg_loss)

save_dir = "./save_model"

torch.save(
    model.state_dict(),
    "./save_model/wave_denoiser.pth"
)

print("model saved")


# ===== plot loss curve =====
plt.figure(figsize=(8, 5))
plt.plot(loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()

# 也可以存圖
plt.savefig("./save_data/loss_curve.png", dpi=300)

