import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import socket
from com_utils import array_to_bits,quantization,dequantization,bits_to_array, send_to_matlab, send_msg, recv_exact, recv_msg
from mlset_util import import_all_model
batch_size = 64
bits = 8
z_dim = 64
device = "cpu"
t_val=300         # 給 DM 評估用
lam=0.5          # Fashion latent 加到 MNIST latent 的係數
do_whiten=True     # 是否白化 z_f
out_dir="./save_data/test_sys"
G_path = "./save_model/G_wgangp_C.pth"
E_path = "./save_model/E_invert_to_G_M_quan.pth"
E_f_path = "./save_model/E_invert_to_G_F_qaun.pth"
G,E,E_f = import_all_model()

# TCP/IP setting
HOST = '127.0.0.1'
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
    ])

ds = datasets.MNIST(root=r"C:/Users/admin/Documents/Python/saved_data", train=True, download=True, transform=tfm)
ds_f = datasets.FashionMNIST(root=r"C:/Users/admin/Documents/Python/saved_data", train=True, download=True, transform=tfm)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

(x_m, _), (x_f, _) = next(zip(dl, dl_f))

x_m = x_m.to(device)
x_f = x_f.to(device)

x_m = x_m[:25]
x_f = x_f[:25]

with torch.no_grad():
    z_m, logvar = E(x_m)
    z_f, logvar = E_f(x_f)
latent_shape_m = list(z_m.shape)
latent_shape_f = list(z_f.shape)


i=0
latent_shape = [25, 64]
while(1):
    receive = recv_msg(client)
    id = receive["ID"]
    Label = receive["Label"]
    print(id)
    if Label == 0:
        print("Label 0 received")
        send_msg(client, {"type": "OK"})
        bits_array_m = receive["bits"]
        print("bits:")
        print(bits_array_m[:100])
        x_hat_m = bits_to_array(bits_array_m,bits)
        print("bits to array:")
        print(x_hat_m[:100])
        x_hat_m = dequantization(x_hat_m,bits)
        z_m_hat = x_hat_m.reshape(latent_shape_m)
        z_m_hat = torch.tensor(z_m_hat, device=device, dtype=torch.float32)
        with torch.no_grad():
            z_m_hat = z_m_hat.view(x_m.size(0), z_dim, 1, 1) 
            recon_m = G(z_m_hat)
            save_image((recon_m+1)/2 , f"./save_data/test_sys_rx/invert_recon_M_ep{id}.png", nrow=5)
    
        recon_grid_m = make_grid(recon_m, nrow=5, normalize=True)
        plt.figure(figsize=(5,5))
        plt.imshow(recon_grid_m.permute(1, 2, 0).cpu())
        plt.title(f"Batch {i}")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(1.5)   # 顯示1.5秒
        plt.close()
        
        
        
    elif Label == 1:
        print("Label 1 received")
        if i == 10:
            send_msg(client, {"type": "END"})
            break
        else:
            send_msg(client, {"type": "OK"})
        bits_array_f = receive["bits"]
        print("bits:")
        print(bits_array_f[:100])
        x_hat_f = bits_to_array(bits_array_f,bits)
        print("bits to array:")
        print(x_hat_f[:100])
        x_hat_f = dequantization(x_hat_f,bits)
        z_f_hat = x_hat_f.reshape(latent_shape_f)
        z_f_hat = torch.tensor(z_f_hat, device=device, dtype=torch.float32)
        with torch.no_grad():
            z_f_hat = z_f_hat.view(x_f.size(0), z_dim, 1, 1) 
            recon_f = G(z_f_hat)
            save_image((recon_f+1)/2 , f"./save_data/test_sys_rx/invert_recon_F_ep{id}.png", nrow=5)

            i +=1

        recon_grid_f = make_grid(recon_f, nrow=5, normalize=True)
        plt.figure(figsize=(5,5))
        plt.imshow(recon_grid_f.permute(1, 2, 0).cpu())
        plt.title(f"Batch {i}")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(1.5)   # 顯示1.5秒
        plt.close()

        
    