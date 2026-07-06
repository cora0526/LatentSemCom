import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import time
import os
import time
from model_baseline import VAE_RGB64
from com_utils import mimo_2x2_channel_zf
from PIL import Image
import glob

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

        return img

start_time = time.time()

# test = 30
batch_size = 64
z_dim = 256
device = "cuda"

tfm = transforms.Compose([
        transforms.ToTensor()
    ])
data_root_d = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
ds_d = SatelliteDataset(data_root_d, transform=tfm)
ds_d = Subset(ds_d, range(min(50000, len(ds_d))))
data_root_s = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
ds_s = datasets.ImageFolder(data_root_s, transform=tfm)
ds_s = Subset(ds_s, range(min(50000, len(ds_s))))
dl_d = DataLoader(ds_d,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
dl_s = DataLoader(ds_s,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)


vae_d = VAE_RGB64(latent_dim=z_dim).to(device)
vae_d.load_state_dict(torch.load("./save_model/baseline_vae_deepglobe.pth", map_location=device))
vae_d.eval()

vae_s = VAE_RGB64(latent_dim=z_dim).to(device)
vae_s.load_state_dict(torch.load("./save_model/baseline_vae_EuroSAT.pth", map_location=device))
vae_s.eval()

# for j in range(0,29,2):
j=30
save_root = f"D:/Python/SysTest/baseline_vae_2by2_space/test_{j}"
os.makedirs(save_root, exist_ok=True)
for i, (batch_d, batch_s) in enumerate(zip(dl_d, dl_s)):
    
    print("ID: ",i)

    x_m = batch_d

    if isinstance(batch_s, (list, tuple)):
        x_f = batch_s[0]
    else:
        x_f = batch_s

    # 匯入影像
    x_m = x_m.to(device)
    x_f = x_f.to(device)
    x_m = x_m[:25]
    x_f = x_f[:25]
    save_image(x_m, f"{save_root}/transmit_M_{i}.png", nrow=5)
    save_image(x_f, f"{save_root}/transmit_F_{i}.png", nrow=5)
    
    # encode
    with torch.no_grad():
        z_m, logvar_m = vae_d.encode(x_m)
        z_f, logvar_f = vae_s.encode(x_f)
    latent_shape_m = list(z_m.shape)
    latent_shape_f = list(z_f.shape)

    # channel
    z_m_hat, z_f_hat = mimo_2x2_channel_zf(z_m, z_f, j)

    # decode
    with torch.no_grad():
        recon_m = vae_d.decode(z_m_hat)
        recon_f = vae_s.decode(z_f_hat)

    save_image(recon_m , f"{save_root}/invert_recon_M_ep{i}.png", nrow=5)
    save_image(recon_f , f"{save_root}/invert_recon_F_ep{i}.png", nrow=5)
    
    if i == 149:
        break
    else:
        continue

end_time = time.time()
print(f"times:{i}")
print(f"執行時間: {end_time - start_time:.4f} 秒")
