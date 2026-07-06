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

start_time = time.time()

# test = 30
batch_size = 64
bits = 8
z_dim = 64
device = "cuda"
G_path = "./save_model/G_wgangp_C.pth"
E_path = "./save_model/E_invert_to_G_C_only.pth"
E_f_path = "./save_model/E_invert_to_G_F.pth"
DM_path = "./save_model/DM_latent_label_with_Channel_1.pth"
# save_root = f"D:/Python/SysTest/baseline_vae_2by2/test_{test}"
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

# for j in range(0,31,2):
j=30
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

    data_id_bits = [int(b) for b in format(data_id, '010b')]     # 10 bits
    # 例如 data_id=3 → [0,0,0,0,0,0,0,0,1,1]
    size_val_m = bits_array_m.size
    size_bits_m = [int(b) for b in format(size_val_m, '016b')]   # 16 bits
    label_bits_m = [0, 0, 0, 0,0]
    bits_array_m = data_id_bits + label_bits_m + size_bits_m + bits_array_m.tolist()
    # bits_array_m = bits_array_m.tolist()
    size_val_f = bits_array_f.size
    size_bits_f = [int(b) for b in format(size_val_f, '016b')]
    label_bits_f = [1, 1, 1, 1, 1]
    bits_array_f = data_id_bits + label_bits_f + size_bits_f + bits_array_f.tolist()
    # bits_array_f = bits_array_f.tolist()

    # 這裡要存起來 msg_f和msg_m================================================================== 
    mat_save_path = f"{save_root}/tx_bits_{data_id}.mat"

    savemat(mat_save_path, {
        "ID": data_id,
        "bits_m": np.array(bits_array_m, dtype=np.uint8),
        "bits_f": np.array(bits_array_f, dtype=np.uint8),
        "label_m": 0,
        "label_f": 1,
        "latent_shape_m": np.array(latent_shape_m),
        "latent_shape_f": np.array(latent_shape_f),
        "bit_width": bits
    })
    # 這裡要讀檔    

    if i == 150:
        break
    else:
        continue

end_time = time.time()
print(f"times:{i}")
print(f"執行時間: {end_time - start_time:.4f} 秒")
