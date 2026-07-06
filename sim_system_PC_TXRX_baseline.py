import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from scipy.io import savemat
import time
import os
from com_utils import array_to_bits,quantization
import time
import numpy as np
from model_baseline import VAE_MNIST
from com_utils import mimo_2x2_channel_zf

start_time = time.time()

# test = 30
batch_size = 64
bits = 8
z_dim = 64
device = "cuda"

# lloyd max codebook
lloyd_max_codebook = torch.load(f"./save_data/lloyd_max_codebook/lloyd_codebook_{bits}.pth")
q = lloyd_max_codebook["q"]
b = lloyd_max_codebook["b"]    

tfm = transforms.Compose([
        transforms.ToTensor()
    ])
ds = datasets.MNIST(root=r"C:/Users/user/Documents/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
ds_f = datasets.FashionMNIST(root=r"C:/Users/user/Documents/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

vae_m = VAE_MNIST(latent_dim=z_dim).to(device)
vae_m.load_state_dict(torch.load("./save_model/baseline_vae_MNIST.pth", map_location=device))
vae_m.eval()

vae_f = VAE_MNIST(latent_dim=z_dim).to(device)
vae_f.load_state_dict(torch.load("./save_model/baseline_vae_FashionMNIST.pth", map_location=device))
vae_f.eval()

for j in range(0,29,2):
    # j=29
    save_root = f"D:/Python/SysTest/baseline_vae_2by2/test_{j}"
    os.makedirs(save_root, exist_ok=True)
    for i, ((x_m, _), (x_f, _)) in enumerate(zip(dl, dl_f)):
        data_id = i
        print("ID: ",data_id)

        # 匯入影像
        x_m = x_m.to(device)
        x_f = x_f.to(device)
        x_m = x_m[:25]
        x_f = x_f[:25]
        save_image(x_m, f"{save_root}/transmit_M_{i}.png", nrow=5)
        save_image(x_f, f"{save_root}/transmit_F_{i}.png", nrow=5)
        
        # encode
        with torch.no_grad():
            z_m, logvar_m = vae_m.encode(x_m)
            z_f, logvar_f = vae_f.encode(x_f)
        latent_shape_m = list(z_m.shape)
        latent_shape_f = list(z_f.shape)

        # channel
        z_m_hat, z_f_hat = mimo_2x2_channel_zf(z_m, z_f, j)

        # decode
        with torch.no_grad():
            recon_m = vae_m.decode(z_m_hat)
            recon_f = vae_f.decode(z_f_hat)

        save_image(recon_m , f"{save_root}/invert_recon_M_ep{data_id}.png", nrow=5)
        save_image(recon_f , f"{save_root}/invert_recon_F_ep{data_id}.png", nrow=5)
        
        if i == 149:
            break
        else:
            continue

    end_time = time.time()
    print(f"times:{i}")
    print(f"執行時間: {end_time - start_time:.4f} 秒")
