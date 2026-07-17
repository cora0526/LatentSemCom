import torch
import glob
from PIL import Image
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from scipy.io import savemat
import time
import os
from model import Generator_Satellite, Encoder_Satellite
import time
from com_utils import mimo_2x2_channel_zf

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

start_time = time.time()

batch_size = 64
bits = 8
z_dim = 256
device = "cuda"
G_path = "./save_model/G_end2end_satellite_C.pth"
# G_f_path = "./save_model/G_end2end_EuroSAT_sep.pth"
E_path = "./save_model/E_end2end_deepglobe_C.pth"
E_f_path = "./save_model/E_end2end_EuroSAT_C.pth"

# save_root = "./save_data/EncoderSatellite_SepGAN_TwoEncoder"
# os.makedirs(save_root, exist_ok=True)

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

# G_f = Generator_Satellite(z_dim=z_dim).to(device)
# G_f.load_state_dict(torch.load(G_f_path, map_location=device))
# G_f.eval()

E = Encoder_Satellite(z_dim=z_dim).to(device)
E.load_state_dict(torch.load(E_path, map_location=device))
E.eval()

E_f = Encoder_Satellite(z_dim=z_dim).to(device)
E_f.load_state_dict(torch.load(E_f_path, map_location=device))
E_f.eval()

for j in range(0,31,2):
    # j=30
    test = j
    save_root = f"D:/Python/SysTest/propose_VAE_GAN_2by2_space/test_{test}"
    os.makedirs(save_root, exist_ok=True)
    for i, (batch_e, batch_g) in enumerate(zip(dl_e, dl_g)):
        data_id = i
        print("ID: ",data_id)

        # 匯入影像
        x_m = batch_e[0].to(device)
        x_f = batch_g[0].to(device)
        x_m = x_m[:25]
        x_f = x_f[:25]
        save_image((x_m+1)/2, f"{save_root}/transmit_M_{i}.png", nrow=5)
        save_image((x_f+1)/2, f"{save_root}/transmit_F_{i}.png", nrow=5)
        
        # encode
        with torch.no_grad():
            z_m, _ = E(x_m)
            z_f, _ = E_f(x_f)
        latent_shape_m = list(z_m.shape)
        latent_shape_f = list(z_f.shape)

        # channal
        z_m_hat, z_f_hat = mimo_2x2_channel_zf(z_m, z_f, j)

        b = z_m_hat.size(0)

        # decode
        with torch.no_grad():
            z_m_hat = z_m_hat.view(b, z_dim, 1, 1) 
            recon_m = G(z_m_hat)
            z_f_hat = z_f_hat.view(b, z_dim, 1, 1) 
            recon_f = G(z_f_hat)

        save_image(torch.clamp((recon_m+1)/2,0,1) , f"{save_root}/invert_recon_M_ep{data_id}.png", nrow=5)
        save_image(torch.clamp((recon_f+1)/2,0,1) , f"{save_root}/invert_recon_F_ep{data_id}.png", nrow=5)
        

        if i == 150:
            break
        else:
            continue

end_time = time.time()
print(f"times:{i}")
print(f"執行時間: {end_time - start_time:.4f} 秒")
