import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from scipy.io import loadmat
import time
import os
from com_utils import dequantization,bits_to_array
import time
import numpy as np
from model_baseline import VAE_MNIST

start_time = time.time()

# test = 24
batch_size = 64
bits = 8
z_dim = 64
device = "cuda"
t_val=300         # 給 DM 評估用
lam=0.5          # Fashion latent 加到 MNIST latent 的係數
do_whiten=True     # 是否白化 z_f
out_dir="./save_data/test_sys"
G_path = "./save_model/G_wgangp_C.pth"
E_path = "./save_model/E_invert_to_G_C_only.pth"
E_f_path = "./save_model/E_invert_to_G_F.pth"
DM_path = "./save_model/DM_latent_label_with_Channel_1.pth"
# save_root = f"D:/Python/SysTest/test_{test}"
# os.makedirs(save_root, exist_ok=True)

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

for j in range(0,31,2):
# j=0
    save_root = f"D:/Python/SysTest/baseline_vae_2by2/test_{j}"
    os.makedirs(save_root, exist_ok=True)
    for i, ((x_m, _), (x_f, _)) in enumerate(zip(dl, dl_f)):
        data_id = i
        print("ID: ",data_id)

        rx_file = os.path.join(save_root, f"rx_bits_{data_id}.mat")

        if not os.path.exists(rx_file):
            print(f"{rx_file} not found")
            break

        rx = loadmat(rx_file)

        latent_shape = (25, z_dim)

        bits_array_m = rx["receive_data_0"].astype(np.uint8).flatten()
        x_hat_m = bits_to_array(bits_array_m,bits)
        x_hat_m = dequantization(x_hat_m,bits=bits)
        # x_hat_m = dequantization(x_hat_m,bits=bits,type='lloyd',q=q)
        z_m_hat = x_hat_m.reshape(latent_shape)
        z_m_hat = torch.tensor(z_m_hat, device=device, dtype=torch.float32)
        with torch.no_grad():
            # z_m_hat = z_m_hat.view(25, z_dim, 1, 1) 
            recon_m = vae_m.decode(z_m_hat)
            save_image(recon_m , f"{save_root}/invert_recon_M_ep{data_id}.png", nrow=5)

        recon_grid_m = make_grid(recon_m, nrow=5, normalize=True)


        bits_array_f = rx["receive_data_1"].astype(np.uint8).flatten()
        x_hat_f = bits_to_array(bits_array_f,bits)
        x_hat_f = dequantization(x_hat_f,bits=bits)
        # x_hat_f = dequantization(x_hat_f,bits=bits,type='lloyd',q=q)
        z_f_hat = x_hat_f.reshape(latent_shape)
        z_f_hat = torch.tensor(z_f_hat, device=device, dtype=torch.float32)
        with torch.no_grad():
            # z_f_hat = z_f_hat.view(25, z_dim, 1, 1) 
            recon_f = vae_f.decode(z_f_hat)
            save_image(recon_f , f"{save_root}/invert_recon_F_ep{data_id}.png", nrow=5)

            

        recon_grid_f = make_grid(recon_f, nrow=5, normalize=True)

        if i == 150:
            break
        else:
            continue
        

end_time = time.time()
print(f"times:{i}")
print(f"執行時間: {end_time - start_time:.4f} 秒")
