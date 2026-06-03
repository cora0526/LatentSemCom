from torch.utils.data import DataLoader
from utils import IQDataset 
from model import IQDenoiser
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda"

dataset = IQDataset(r"D:/VAE_GAN_DM/latent_bits_F/bits_8/tran_encode_RF_DDPM")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

model = IQDenoiser().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

criterion = nn.MSELoss()

for epoch in range(50):

    total_loss = 0

    for noisy, clean in loader:

        noisy = noisy.to(device)
        clean = clean.to(device)

        pred = model(noisy)

        loss = criterion(pred, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(epoch, total_loss)


torch.save(
    model.state_dict(),
    r"D:/VAE_GAN_DM/save_model/RF_DDPM/iq_denoiser.pth"
)