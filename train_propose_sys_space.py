import torch
import glob
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image
import time
import os
import time
from model import Generator_Satellite, Encoder_Satellite

class SatelliteDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = 1  

        return img, label

def mimo_2x2_channel_zf_trainable(z_m, z_f, snr_db):
    device = z_m.device

    H = torch.tensor([
        [1.0, 0.15],
        [0.10, 0.90]
    ], dtype=torch.float32, device=device)

    H = H / torch.norm(H, p="fro") * torch.sqrt(torch.tensor(2.0, device=device))

    x_pair = torch.stack([z_m, z_f], dim=-1)   # [B, z_dim, 2]

    y_pair = torch.matmul(x_pair, H.T)

    signal_power = torch.mean(y_pair ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = torch.sqrt(noise_power) * torch.randn_like(y_pair)

    y_pair = y_pair + noise

    H_inv = torch.inverse(H)

    x_hat_pair = torch.matmul(y_pair, H_inv.T)

    z_m_hat = x_hat_pair[:, :, 0]
    z_f_hat = x_hat_pair[:, :, 1]

    return z_m_hat, z_f_hat

start_time = time.time()

batch_size = 64
epochs = 100
bits = 8
z_dim = 256
lr_E = 2e-4
lr_G = 1e-5
snr_min = 0
snr_max = 30 
device = "cuda"
G_path = "./save_model/G_wgangp_satellite.pth"
G_f_path = "./save_model/G_wgangp_satellite_EuroSAT.pth"
E_path = "./save_model/E_invert_to_G_satellite.pth"
E_f_path = "./save_model/E_invert_to_G_satellite_EuroSAT.pth"

save_root = "./save_data/EncoderSatellite_sepGAN_TwoEncoder"
os.makedirs(save_root, exist_ok=True)

data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
data_root_g = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

ds_e = datasets.ImageFolder(data_root_e, transform=tfm)
ds_g = SatelliteDataset(data_root_g, transform=tfm)

ds_e = Subset(ds_e, range(min(50000, len(ds_e))))
ds_g = Subset(ds_g, range(min(50000, len(ds_g))))

dl_e = DataLoader(
    ds_e,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)

dl_g = DataLoader(
    ds_g,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True
)

G = Generator_Satellite(z_dim=z_dim).to(device)
G.load_state_dict(torch.load(G_path, map_location=device))
G.eval()

G_f = Generator_Satellite(z_dim=z_dim).to(device)
G_f.load_state_dict(torch.load(G_path, map_location=device))
G_f.eval()

E = Encoder_Satellite(z_dim=z_dim).to(device)
E.load_state_dict(torch.load(E_path, map_location=device))
E.eval()

E_f = Encoder_Satellite(z_dim=z_dim).to(device)
E_f.load_state_dict(torch.load(E_f_path, map_location=device))
E_f.eval()

G = G.to(device)
G_f = G_f.to(device)
E = E.to(device)
E_f = E_f.to(device)

G.train()
G_f.train()
E.train()
E_f.train()

for p in G.parameters():
    p.requires_grad = True

for p in G_f.parameters():
    p.requires_grad = True

for p in E.parameters():
    p.requires_grad = True

for p in E_f.parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(
    [
        {"params": E.parameters(), "lr": lr_E},
        {"params": E_f.parameters(), "lr": lr_E},
        {"params": G.parameters(), "lr": lr_G},
        {"params": G_f.parameters(), "lr": lr_G}
    ],
    betas=(0.5, 0.999)
)

for epoch in range(epochs):

    total_loss = 0
    total_e_loss = 0
    total_g_loss = 0

    for i, (batch_e, batch_g) in enumerate(zip(dl_e, dl_g)):

        if isinstance(batch_e, (list, tuple)):
            x_e = batch_e[0]
        else:
            x_e = batch_e

        if isinstance(batch_g, (list, tuple)):
            x_g = batch_g[0]
        else:
            x_g = batch_g

        x_e = x_e.to(device)
        x_g = x_g.to(device)

        snr_db = torch.empty(1).uniform_(snr_min, snr_max).item()

        z_e, _ = E(x_e)
        z_g, _ = E_f(x_g)

        z_e_hat, z_g_hat = mimo_2x2_channel_zf_trainable(
            z_e,
            z_g,
            snr_db
        )

        z_e_hat = z_e_hat.view(batch_size, z_dim, 1, 1)
        z_g_hat = z_g_hat.view(batch_size, z_dim, 1, 1)

        recon_e = G(z_e_hat)
        recon_g = G_f(z_g_hat)

        loss_e = F.l1_loss(recon_e, x_e)
        loss_g = F.l1_loss(recon_g, x_g)

        loss = 0.5 * loss_e + 0.5 * loss_g

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_e_loss += loss_e.item()
        total_g_loss += loss_g.item()

        if i % 50 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{i}/{min(len(dl_e), len(dl_g))}] "
                f"EuroSAT: {loss_e.item():.4f} "
                f"DeepGlobe: {loss_g.item():.4f} "
                f"Total: {loss.item():.4f}"
            )

    avg_loss = total_loss / min(len(dl_e), len(dl_g))
    avg_e_loss = total_e_loss / min(len(dl_e), len(dl_g))
    avg_g_loss = total_g_loss / min(len(dl_e), len(dl_g))

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.6f} "
        f"EuroSAT: {avg_e_loss:.6f} "
        f"DeepGlobe: {avg_g_loss:.6f}"
    )
# =====================
# final save
# =====================
torch.save(E.state_dict(), "./save_model/E_end2end_deepglobe_sep.pth")
torch.save(E_f.state_dict(), "./save_model/E_end2end_EuroSAT_sep.pth")
torch.save(G.state_dict(), "./save_model/G_end2end_deepglobe_sep.pth")
torch.save(G_f.state_dict(), "./save_model/G_end2end_EuroSAT_sep.pth")

end_time = time.time()
print(f"執行時間: {end_time - start_time:.4f} 秒")