import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from scipy.io import savemat
import time
import os
from com_utils import array_to_bits,quantization
from mlset_util import import_all_model
import time
import numpy as np
from com_utils import mimo_2x2_channel_zf

start_time = time.time()

batch_size = 64
bits = 8
z_dim = 64
device = "cuda"
# G_path = "./save_model/G_wgangp_MNIST.pth"
G_f_path = "./save_model/G_end2end_FashionMNIST_sep.pth"
# E_path = "./save_model/E_invert_to_G_only_MNIST.pth"
# E_f_path = "./save_model/E_invert_to_G_only_FashionMNIST.pth"
G_path = "./save_model/G_end2end_MNIST_sep.pth"
E_path = "./save_model/E_end2end_MNIST_sep.pth"
E_f_path = "./save_model/E_end2end_FashionMNIST_sep.pth"

tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
    ])
ds = datasets.MNIST(root=r"C:/Users/user/Documents/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
ds_f = datasets.FashionMNIST(root=r"C:/Users/user/Documents/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

G,G_f,E,E_f = import_all_model(G_path=G_path, G_f_path=G_f_path,E_path=E_path,E_f_path=E_f_path)

for j in range(0,31,2):
    test = j
    save_root = f"D:/Python/SysTest/propose_septerate_GAN/test_{test}"
    os.makedirs(save_root, exist_ok=True)
    for i, ((x_m, _), (x_f, _)) in enumerate(zip(dl, dl_f)):
        data_id = i
        print("ID: ",data_id)

        # 匯入影像
        x_m = x_m.to(device)
        x_f = x_f.to(device)
        x_m = x_m[:25]
        x_f = x_f[:25]
        grid_m = make_grid(x_m, nrow=5, normalize=True)
        grid_f = make_grid(x_f, nrow=5, normalize=True)
        save_image(grid_m, f"{save_root}/transmit_M_{i}.png", nrow=5)
        save_image(grid_f, f"{save_root}/transmit_F_{i}.png", nrow=5)
        
        # encode
        with torch.no_grad():
            z_m, logvar = E(x_m)
            z_f, logvar = E_f(x_f)
        latent_shape_m = list(z_m.shape)
        latent_shape_f = list(z_f.shape)

        # channal
        z_m_hat, z_f_hat = mimo_2x2_channel_zf(z_m, z_f, j)

        # decode
        with torch.no_grad():
            z_m_hat = z_m_hat.view(25, z_dim, 1, 1) 
            recon_m = G(z_m_hat)
            z_f_hat = z_f_hat.view(25, z_dim, 1, 1) 
            recon_f = G_f(z_f_hat)

        save_image((recon_m+1)/2 , f"{save_root}/invert_recon_M_ep{data_id}.png", nrow=5)
        save_image((recon_f+1)/2 , f"{save_root}/invert_recon_F_ep{data_id}.png", nrow=5)
        

        if i == 150:
            break
        else:
            continue

end_time = time.time()
print(f"times:{i}")
print(f"執行時間: {end_time - start_time:.4f} 秒")
