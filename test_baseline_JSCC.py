import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.utils import save_image, make_grid
from model_baseline import JSCC_MIMO_RGB64
from PIL import Image
import os
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

def awgn_channel(z, snr_db):
    signal_power = torch.mean(z ** 2)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = torch.sqrt(noise_power) * torch.randn_like(z)

    z_noisy = z + noise

    return z_noisy

device = "cuda"

batch_size = 25

tfm = transforms.Compose([
        transforms.ToTensor()
    ])
data_root_a = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
data_root_b = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
ds_a = SatelliteDataset(data_root_a, transform=tfm)
ds_b = datasets.ImageFolder(data_root_b, transform=tfm)
dl_a = DataLoader(ds_a,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
dl_b = DataLoader(ds_b,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)

JSCC_m = JSCC_MIMO_RGB64().to(device)
JSCC_m.load_state_dict(torch.load("./save_model/jscc_mimo_deepglobe.pth",map_location=device))
JSCC_m.eval()

JSCC_f = JSCC_MIMO_RGB64().to(device)
JSCC_f.load_state_dict(torch.load("./save_model/jscc_mimo_EuroSAT.pth",map_location=device))
JSCC_f.eval()

tests = []

mnist_psnr = []
mnist_ssim = []

fashion_psnr = []
fashion_ssim = []

for j in range(0, 31, 2):

    save_root = f"D:/Python/SysTest/baseline_JSCC_space/test_{j}"
    os.makedirs(save_root, exist_ok=True)

    for i, (batch_a, batch_b) in enumerate(zip(dl_a, dl_b)):

        print("ID:", i)

        x_m = batch_a

        if isinstance(batch_b, (list, tuple)):
            x_f = batch_b[0]
        else:
            x_f = batch_b

        x_m = x_m.to(device)
        x_f = x_f.to(device)

        save_image(x_m, f"{save_root}/transmit_M_{i}.png", nrow=5)
        save_image(x_f, f"{save_root}/transmit_F_{i}.png", nrow=5)

        with torch.no_grad():
            recon_m = JSCC_m(x_m, snr_db=j)
            recon_f = JSCC_f(x_f, snr_db=j)

        save_image(recon_m, f"{save_root}/invert_recon_M_ep{i}.png", nrow=5)
        save_image(recon_f, f"{save_root}/invert_recon_F_ep{i}.png", nrow=5)

        if i == 149:
            break
