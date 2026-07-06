from model import Ganerator_MNIST, VAE_Encoder_MNIST, LabelCondLatentDenoiserDeep
from train_VAE_encoder import train_encoder_invert_gan
from utils import linear_beta_schedule, denoise_latent_ddpm, eps_from_Ef, ddpm_loss_label_cond, sample_latent_label_cond
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from com_utils import array_to_bits, bits_to_array,dequantization,quantization,send_msg,recv_msg
from torchvision.utils import save_image
from itertools import cycle
import numpy as np
import socket
import matplotlib.pyplot as plt

epochs = 200
batch_size = 16
z_dim = 64
y_dim = 64
device = "cuda"
bits = 8

# fix settings
# transformer
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# dataset and dataloader
ds = datasets.MNIST(root=r"D:\Python\SemCom\saved_data", train=True, download=True, transform=tfm)
ds_f = datasets.FashionMNIST(root=r"D:\Python\SemCom\saved_data", train=True, download=True, transform=tfm)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
f_iter = cycle(dl_f) 
m_iter = cycle(dl)

# 固定測試集資料
fixed_x, _ = next(iter(dl))
fixed_x = fixed_x[:64].to(device)
save_image(fixed_x, f"./save_data/test_labelDMwithChannel_1/dm_recon_test_M.png", nrow=8)
fixed_x_F, _ = next(iter(dl_f))
fixed_x_F = fixed_x_F[:64].to(device)
save_image(fixed_x_F, f"./save_data/test_labelDMwithChannel_1/dm_recon_test_F.png", nrow=8)

# 導入encoder和GAN模型 都是freeze的
G = Ganerator_MNIST(z_dim=z_dim).to(device)
G.load_state_dict(torch.load("./save_model/G_wgangp_C.pth"))
G.eval()
for p in G.parameters():
    p.requires_grad = False

E = VAE_Encoder_MNIST(latent_dim=z_dim).to(device)
E.load_state_dict(torch.load("./save_model/E_invert_to_G_C_only.pth"))
E.eval()
for p in E.parameters():
    p.requires_grad = False

E_f = VAE_Encoder_MNIST(latent_dim=z_dim).to(device)
E_f.load_state_dict(torch.load("./save_model/E_invert_to_G_F.pth"))
E_f.eval()
for p in E_f.parameters():
    p.requires_grad = False

# TCP/IP setting
HOST = '127.0.0.1'
PORT = 5000
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    client.connect((HOST, PORT))
    print(f"Connected to {HOST}:{PORT}")
except Exception as e:
    print(f"Connection failed: {e}")

# DDPM settings
T = 1000
betas = linear_beta_schedule(T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

sqrt_ab = torch.sqrt(alpha_bars)
sqrt_1mab = torch.sqrt(1 - alpha_bars)

#輸入DM model 
DM =  LabelCondLatentDenoiserDeep(num_classes=2, z_dim=z_dim, tdim=128, ydim=y_dim,hidden_dim=512,num_blocks=6).to(device)
opt = torch.optim.Adam(DM.parameters(), lr=2e-4)

T_train = int(0.4 * T)   # 先訓練 t <= 400
b_max = 0.6              # 干擾先不要太強
log_every = 50

loss_history = []
latent_loss_history = []
img_loss_history = []
step_history = []
loss_m_history = []
loss_f_history = []
a = 0
for epoch in range(200):
    total_loss = 0.0
    num_batches = 0
    # for step in range(min(len(dl), len(dl_f))):
    for step in range(1000):
    # encode 影像
        x_m, _ = next(m_iter)
        x_f, _ = next(f_iter)

        x_m = x_m.to(device)
        x_f = x_f.to(device)

        with torch.no_grad():
            z_m, _ = E(x_m)      # MNIST latent, shape = [B, 64]
            z_f, _ = E_f(x_f)
        # z_m = z_m.squeeze(0)
        # z_f = z_f.squeeze(0)
        # ------------------------
        # Quantization
        # ------------------------
        z_q_m,_,_ = quantization (z_m,bits)
        z_q_f,_,_ = quantization (z_f,bits)
        org_shape = z_q_m.shape
        
        bits_array_m = array_to_bits(z_q_m, bit_width=bits)
        bits_array_f = array_to_bits(z_q_f, bit_width=bits)

        # ------------------------
        # Send MNIST
        # ------------------------
        msg_m = {
            "label": 0,
            "bits": bits_array_m.tolist()
        }
        while True:
            send_msg(client, msg_m)
            # print("send MNIST")
            reply = recv_msg(client)
            if (reply == "ACK"):
                break
        # ------------------------
        # Send Fashion
        # ------------------------
        msg_f = {
            "label": 1,
            "bits": bits_array_f.tolist()
        }
        while True:
            send_msg(client, msg_f)
            # print("send MNISTFashion")
            reply = recv_msg(client)
            if (reply == "ACK"):
                break

        # ------------------------
        # Receive SDR result
        # ------------------------
        while True:
            reply = recv_msg(client)
            if reply["RECEIVEDATA"] == 1:
                rx_bits = np.array(reply["replybits"])
                # print(rx_bits.shape)
                break

        receive_array = bits_to_array(rx_bits, bit_width=bits)
        z_rec_deq = dequantization (receive_array,bits)
        z_rec_deq = z_rec_deq.reshape(org_shape)
        z_rec_deq = z_rec_deq.float().to(device)
        B = z_rec_deq.size(0)

        # ------------------------
        # Random target
        # ------------------------
        # target_label = torch.randint(0,2,(1,),device=device).item()
        # target_label = 0
        # if target_label == 0:
        #     y = torch.zeros(B, device=device, dtype=torch.long)
        #     z_tgt = z_m
        # else:
        #     y = torch.ones(B, device=device, dtype=torch.long)
        #     z_tgt = z_f

        # ------------------------
        # DM
        # ------------------------
        t = torch.randint(0, T_train, (B,), device=device, dtype=torch.long)

        y_m = torch.zeros(B, device=device, dtype=torch.long)
        z_hat_m = DM(z_rec_deq, t, y_m)
        loss_latent_m = F.mse_loss(z_hat_m, z_m)

        y_f = torch.ones(B, device=device, dtype=torch.long)
        z_hat_f = DM(z_rec_deq, t, y_f)
        loss_latent_f = F.mse_loss(z_hat_f, z_f)
        
        # z_tgt_hat = (zt - sqrt_1mab[t].view(B, 1) * eps_pred) / sqrt_ab[t].view(B, 1)
        xhat_m = G(z_hat_m.view(B, z_dim, 1, 1))
        xhat_f = G(z_hat_f.view(B, z_dim, 1, 1))
        with torch.no_grad():
            xref_m = G(z_m.view(B, z_dim, 1, 1))
            xref_f = G(z_f.view(B, z_dim, 1, 1))

        loss_img_m = F.mse_loss(xhat_m, xref_m)
        loss_img_f = F.mse_loss(xhat_f, xref_f)
        loss = (
            loss_latent_m
            + loss_latent_f
            + 0.08 * (loss_img_m + loss_img_f)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        num_batches += 1

        steps_per_epoch = 1000
        global_step = epoch * steps_per_epoch + step + 1

        loss_history.append(loss.item())
        # latent_loss_history.append(loss_latent.item())
        # img_loss_history.append(loss_img.item())
        step_history.append(global_step)
        loss_m_history.append(loss_latent_m.item())
        loss_f_history.append(loss_latent_f.item())

        if (step + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Step [{step+1}/{min(len(dl),len(dl_f))}] "
                f"Loss={loss.item():.6f} "
                # f"Latent={loss_latent.item():.6f} "
                # f"Img={loss_img.item():.6f}"
            )
        
        if (step + 1) % 1000 == 0:
            z_rec_deq = torch.tensor(z_rec_deq,dtype=torch.float32,device=device)

            with torch.no_grad():
                img_hat_m = G(z_hat_m.view(B,z_dim,1,1))
                img_hat_f = G(z_hat_f.view(B,z_dim,1,1))

                img_gt_m = G(z_m.view(B,z_dim,1,1))
                img_gt_f = G(z_f.view(B,z_dim,1,1))
                img_rx = G(z_rec_deq.view(B,z_dim,1,1))
                # img_gt = G(z_m.view(B,z_dim,1,1))

            save_image(img_rx[:64], f"./save_data/test_labelDMwithChannel_3/rx_mixed_{a}.png", nrow=8)
            save_image(img_hat_m[:64], f"./save_data/test_labelDMwithChannel_3/denoise_mnist_{a}.png", nrow=8)
            save_image(img_gt_m[:64], f"./save_data/test_labelDMwithChannel_3/gt_mnist_{a}.png", nrow=8)

            save_image(img_hat_f[:64], f"./save_data/test_labelDMwithChannel_3/denoise_fashion_{a}.png", nrow=8)
            save_image(img_gt_f[:64], f"./save_data/test_labelDMwithChannel_3/gt_fashion_{a}.png", nrow=8) 
            a = a+1
        
    plt.figure()
    plt.plot(step_history, loss_history, label="Total loss")
    plt.plot(step_history, loss_m_history, label="MNIST latent")
    plt.plot(step_history, loss_f_history, label="Fashion latent")
    # plt.plot(step_history, latent_loss_history, label="Latent loss")
    # plt.plot(step_history, img_loss_history, label="Image loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("DM Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./save_data/test_labelDMwithChannel_3/loss_curve{a}.png")
    print(
        f"Epoch {epoch+1} Avg Loss = "
        f"{total_loss/num_batches:.6f}"
    )

torch.save(DM.state_dict(), "./save_model/DM_latent_label_with_Channel_3.pth")
print("Saved DM: DM_latent_label_with_Channel_3.pth")

plt.figure()
plt.plot(step_history, loss_history, label="Total loss")
plt.plot(step_history, loss_m_history, label="MNIST latent")
plt.plot(step_history, loss_f_history, label="Fashion latent")
# plt.plot(step_history, latent_loss_history, label="Latent loss")
# plt.plot(step_history, img_loss_history, label="Image loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("DM Training Loss")
plt.legend()
plt.grid(True)
plt.savefig("./save_data/test_labelDMwithChannel_1/loss_curve.png")
# plt.show()