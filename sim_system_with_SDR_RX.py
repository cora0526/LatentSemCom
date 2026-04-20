import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os
import math
import socket
from utils import linear_beta_schedule, denoise_latent_ddpm, whiten_latent
from com_utils import array_to_bits,quantization,dequantization,bits_to_array, send_to_matlab, send_msg, recv_exact, recv_msg
from mlset_util import import_all_model

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
DM_path = "./save_model/DM_latent_label_with_Channel_1.pth.pth"

# TCP/IP setting
HOST = '127.0.0.1'
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

# tfm = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
#     ])

# ds = datasets.MNIST(root=r"C:/Users/admin/Documents/Python/saved_data", train=True, download=True, transform=tfm)
# ds_f = datasets.FashionMNIST(root=r"C:/Users/admin/Documents/Python/saved_data", train=True, download=True, transform=tfm)
# dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
# dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)



# G,E,E_f,DM = import_all_model()
i=0
latent_shape = [25, 64]
while(1):
    receive = recv_msg(client)
    id = receive["ID"]
    Label = receive["Label"]
    print(id)
    if Label == 0:
        print("Label 0 received")
        send_msg(client, {"ACK": 1})
        # bits_array_m = receive["bits"]
        # x_hat_m = bits_to_array(bits_array_m, bit_width=bits)
        # z_hat_m = dequantization (x_hat_m,bits)
        # z_hat_m = z_hat_m.reshape(latent_shape)
        # z_hat_m = torch.tensor(z_hat_m, device=device, dtype=torch.float32)

        # B, z_dim = z_hat_m.shape
        # xhat_m = G(z_hat_m.view(B, z_dim, 1, 1))
        # xhat_m_show = (xhat_m + 1) / 2
        # grid_rec_m = make_grid(xhat_m_show[:25], nrow=5)

        # save_image(grid_rec_m, f"./save_data/mnist_{id}.png")
        # plt.figure(figsize=(5,5))
        # plt.title("Recover as MNIST ID=",id)
        # plt.imshow(grid_rec_m.permute(1,2,0).cpu().numpy(), cmap='gray')
        # plt.axis('off')
        # plt.show(block=False)
        # plt.pause(1.5)
        # plt.close()
        i += 1
    elif Label == 1:
        print("Label 1 received")
        send_msg(client, {"ACK": 1})
        # bits_array_f = receive["bits"]
        # x_hat_f = bits_to_array(bits_array_f, bit_width=bits)
        # z_hat_f = dequantization (x_hat_f,bits)
        # z_hat_f = z_hat_f.reshape(latent_shape)
        # z_hat_f = torch.tensor(z_hat_f, device=device, dtype=torch.float32)

        # B, z_dim = z_hat_f.shape
        # xhat_f = G(z_hat_f.view(B, z_dim, 1, 1))
        # xhat_f_show = (xhat_f + 1) / 2
        # grid_rec_f = make_grid(xhat_f_show[:25], nrow=5)

        # save_image(grid_rec_f, f"./save_data/FashionMNIST_{id}.png")
        # plt.figure(figsize=(10,5))
        # plt.title("Recover as FashionMNIST ID=",id)
        # plt.imshow(grid_rec_f.permute(1,2,0).cpu().numpy(), cmap='gray')
        # plt.axis('off')
        # plt.show(block=False)
        
        # plt.pause(1.5)
        # plt.close()
        i +=1
        
    if i == 50:
        break