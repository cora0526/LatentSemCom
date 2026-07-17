import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import socket
import time
import os
import numpy as np
from utils import linear_beta_schedule, denoise_latent_ddpm, whiten_latent
from com_utils import array_to_bits,quantization,dequantization,bits_to_array, send_to_matlab, send_msg, recv_exact, recv_msg
from mlset_util import import_all_model

test = 1
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
save_root = f"D:/Python/SysTest/test_{test}"
os.makedirs(save_root, exist_ok=True)

# lloyd max codebook
lloyd_max_codebook = torch.load(f"./save_data/lloyd_max_codebook/lloyd_codebook_{bits}.pth")
q = lloyd_max_codebook["q"]
b = lloyd_max_codebook["b"]    

# TCP/IP setting
HOST = '127.0.0.1'
PORT = 5000

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((HOST, PORT))

tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
    ])
ds = datasets.MNIST(root=r"C:/Users/user/Documents/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
ds_f = datasets.FashionMNIST(root=r"C:/Users/user/Documents/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

# x_m, _ = next(iter(dl))  #iter() 從dataloader取一個batch的data，next()去下一個batch 第一次呼叫此func就是取第一筆
# x_f, _ = next(iter(dl_f))
# x_m = x_m.to(device)    #convert to CUDA
# x_f = x_f.to(device)

# save_image((x_m + 1) / 2, f"./save_data/test_channel/input_MNIST.png", nrow=8)
# save_image((x_f + 1) / 2, f"./save_data/test_channel/input_fashion.png", nrow=8)


G,E,E_f = import_all_model()
a = 0
for i, ((x_m, _), (x_f, _)) in enumerate(zip(dl, dl_f)):
    if i == 0 and a == 0:
        a == 1
    elif i == 1 and a == 1:
        i == 0
        a == 2

    data_id = i
    print("ID: ",data_id)

    # 匯入影像
    x_m = x_m.to(device)
    x_f = x_f.to(device)
    x_m = x_m[:25]
    x_f = x_f[:25]
    grid_m = make_grid(x_m, nrow=5, normalize=True)
    grid_f = make_grid(x_f, nrow=5, normalize=True)
    save_image(grid_m, f"D:/Python/SysTest/test_{test}/transmit_M_{i}.png", nrow=5)
    save_image(grid_f, f"D:/Python/SysTest/test_{test}/transmit_F_{i}.png", nrow=5)
    # 顯示
    plt.figure(figsize=(5,5))
    plt.imshow(grid_m.permute(1, 2, 0).cpu())
    plt.title(f"Batch {i}")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(1.5)   # 顯示1.5秒
    plt.close()
    plt.figure(figsize=(5,5))
    plt.imshow(grid_f.permute(1, 2, 0).cpu())
    plt.title(f"Batch {i}")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(1.5)   # 顯示1.5秒
    plt.close()

    # encode
    with torch.no_grad():
        z_m, logvar = E(x_m)
        z_f, logvar = E_f(x_f)
    latent_shape_m = list(z_m.shape)
    latent_shape_f = list(z_f.shape)

    # quantization
    z_m_q,_,_ = quantization (z_m,bits)
    z_f_q,_,_ = quantization (z_f,bits)
    # z_m_q, _, _= quantization(z_m,type='lloyd',q=q,b=b)
    # z_f_q, _, _= quantization(z_f,type='lloyd',q=q,b=b)
    # print("===== z_q =====")
    # print("type:", type(z_m_q))
    # print("dtype:", z_m_q.dtype)
    # print("shape:", z_m_q.shape)
    # print("numel:", z_m_q.numel())   # 總元素數
    # print()
    # 轉 bit
    bits_array_m = array_to_bits(z_m_q, bit_width=bits)
    bits_array_f = array_to_bits(z_f_q, bit_width=bits)
    # print(bits_array_m)
    print("Send m to matlab:")
    print("qantization:")
    print(z_m_q[:100])
    print("bits:")
    print(bits_array_m[:100])

    data_id_bits = [int(b) for b in format(data_id, '010b')]     # 10 bits
    # 例如 data_id=3 → [0,0,0,0,0,0,0,0,1,1]
    size_val_m = bits_array_m.size
    size_bits_m = [int(b) for b in format(size_val_m, '016b')]   # 16 bits
    label_bits_m = [0, 0, 0, 0,0]
    bits_array_m = data_id_bits + label_bits_m + size_bits_m + bits_array_m.tolist()
    # bits_array_m = bits_array_m.tolist()
    # send msg to matlab
    msg_m = {
        "ID": data_id,
        "type": "bits",
        "label": 0,
        "bits": bits_array_m
    }
    while True:
        send_msg(client, msg_m)
        # print("Sent:", msg_m)

        reply = recv_msg(client)
        # print("Reply from MATLAB:", reply)

        if (reply["ID"]==data_id):
            print(data_id)
            time.sleep(0.5)
            break

    print("Send f to matlab:")
    print("qantization:")
    print(z_f_q[:100])
    print("bits:")
    print(bits_array_f[:100])
    size_val_f = bits_array_f.size
    size_bits_f = [int(b) for b in format(size_val_f, '016b')]
    label_bits_f = [1, 1, 1, 1, 1]
    bits_array_f = data_id_bits + label_bits_f + size_bits_f + bits_array_f.tolist()
    # bits_array_f = bits_array_f.tolist()
    msg_f = {
        "ID": data_id,
        "type": "bits",
        "label": 1,
        "bits": bits_array_f
    }

    while True:
        send_msg(client, msg_f)
        # print("Sent:", msg_f)

        reply = recv_msg(client)
        print("Reply from MATLAB:", reply)

        if (reply["ID"]==data_id):
            # print(data_id)
            time.sleep(0.5)
            break
    print("======================================")
    while True:
        receive = recv_msg(client)
        print("Reply from SDR:", reply)
        if (receive["status"]==201):
            data_id+=1
            break

    
    if i == 9:
        msg_end = {
        "ID": 9999,
        "type": "end",
        "label": 999,
        "bits": 99999
        }
        send_msg(client, msg_end)
        break