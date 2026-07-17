import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import time
import os
from mlset_util import import_all_model
import time
from com_utils import mimo_2x2_channel_zf
from sim_baseline_JPEG_LDPC import jpeg_ldpc_mimo_baseline

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
    transforms.ToTensor()
])

ds = datasets.MNIST(root=r"C:/Users/user/Documents/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
ds_f = datasets.FashionMNIST(root=r"C:/Users/user/Documents/Python/SemCom/saved_data", train=True, download=True, transform=tfm)

dl = DataLoader(ds, batch_size=25, shuffle=True, num_workers=0, drop_last=True)
dl_f = DataLoader(ds_f, batch_size=25, shuffle=True, num_workers=0, drop_last=True)

for j in range(0,31,2):
    # j = 30
    test = j
    save_root = f"D:/Python/SysTest/baseline_JPEG_MNIST/test_{test}"
    os.makedirs(save_root, exist_ok=True)
    for i, ((x_m, _), (x_f, _)) in enumerate(zip(dl, dl_f)):
        data_id = i

        print("ID:", i)

        save_image(x_m, f"{save_root}/transmit_M_{i}.png", nrow=5)
        save_image(x_f, f"{save_root}/transmit_F_{i}.png", nrow=5)
        
        recon_m_list = []
        recon_f_list = []

        for k in range(x_m.size(0)):
            recon_m, recon_f = jpeg_ldpc_mimo_baseline(
                x_m[k],
                x_f[k],
                snr_db=j,
                jpeg_quality=50
            )

            recon_m_list.append(recon_m)
            recon_f_list.append(recon_f)

        save_image((recon_m+1)/2 , f"{save_root}/invert_recon_M_ep{data_id}.png", nrow=5)
        save_image((recon_f+1)/2 , f"{save_root}/invert_recon_F_ep{data_id}.png", nrow=5)
        

        if i == 150:
            break
        else:
            continue

end_time = time.time()
print(f"times:{i}")
print(f"執行時間: {end_time - start_time:.4f} 秒")
