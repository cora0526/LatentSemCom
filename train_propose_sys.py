import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import time
import os
from mlset_util import import_all_model
import time

def mimo_2x2_channel_zf_trainable(z_m, z_f, snr_db):
    device = z_m.device

    H = torch.tensor([
        [1.0, 0.15],
        [0.10, 0.90]
    ], dtype=torch.float32, device=device)

    H = H / torch.norm(H, p="fro") * torch.sqrt(torch.tensor(2.0, device=device))

    x_pair = torch.stack([z_m, z_f], dim=-1)   # [B, z_dim, 2]

    y_pair = torch.matmul(x_pair, H.T)

    signal_power = torch.mean(y_pair ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = torch.sqrt(noise_power) * torch.randn_like(y_pair)

    y_pair = y_pair + noise

    H_inv = torch.inverse(H)

    x_hat_pair = torch.matmul(y_pair, H_inv.T)

    z_m_hat = x_hat_pair[:, :, 0]
    z_f_hat = x_hat_pair[:, :, 1]

    return z_m_hat, z_f_hat

start_time = time.time()

batch_size = 64
epochs = 100
bits = 8
z_dim = 64
lr_E = 2e-4
lr_G = 1e-5
snr_min = 0
snr_max = 30 
device = "cuda"
G_path = "./save_model/G_wgangp_MNIST.pth"
G_f_path = "./save_model/G_wgangp_FashionMNIST.pth"
E_path = "./save_model/E_invert_to_G_only_MNIST.pth"
E_f_path = "./save_model/E_invert_to_G_only_FashionMNIST.pth"

save_root = "./save_data/end2end_GAN_MIMO"
os.makedirs(save_root, exist_ok=True)

# lloyd max codebook
lloyd_max_codebook = torch.load(f"./save_data/lloyd_max_codebook/lloyd_codebook_{bits}.pth")
q = lloyd_max_codebook["q"]
b = lloyd_max_codebook["b"]    

tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # -> [-1,1]
    ])
ds = datasets.MNIST(root=r"C:/Users/user/Documents/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
ds_f = datasets.FashionMNIST(root=r"C:/Users/user/Documents/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

G,G_f,E,E_f = import_all_model(G_path=G_path, G_f_path=G_f_path,E_path=E_path,E_f_path=E_f_path)

G = G.to(device)
G_f = G_f.to(device)
E = E.to(device)
E_f = E_f.to(device)

G.train()
G_f.train()
E.train()
E_f.train()

for p in G.parameters():
    p.requires_grad = True

for p in G_f.parameters():
    p.requires_grad = True

for p in E.parameters():
    p.requires_grad = True

for p in E_f.parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(
    [
        {"params": E.parameters(), "lr": lr_E},
        {"params": E_f.parameters(), "lr": lr_E},
        {"params": G.parameters(), "lr": lr_G},
        {"params": G_f.parameters(), "lr": lr_G}
    ],
    betas=(0.5, 0.999)
)

for epoch in range(epochs):

    total_loss = 0
    total_m_loss = 0
    total_f_loss = 0

    for i, ((x_m, _), (x_f, _)) in enumerate(zip(dl, dl_f)):

        x_m = x_m.to(device)
        x_f = x_f.to(device)

        # random SNR
        snr_db = torch.empty(1).uniform_(snr_min, snr_max).item()

        # =====================
        # encode
        # =====================
        z_m, _ = E(x_m)
        z_f, _ = E_f(x_f)

        # =====================
        # 2x2 MIMO channel + ZF
        # =====================
        z_m_hat, z_f_hat = mimo_2x2_channel_zf_trainable(z_m,z_f,snr_db)

        # =====================
        # decode by GAN generator
        # =====================
        z_m_hat = z_m_hat.view(batch_size, z_dim, 1, 1)
        z_f_hat = z_f_hat.view(batch_size, z_dim, 1, 1)

        recon_m = G(z_m_hat)
        recon_f = G_f(z_f_hat)

        # =====================
        # loss
        # =====================
        loss_m = F.l1_loss(recon_m, x_m)
        loss_f = F.l1_loss(recon_f, x_f)

        loss = 0.5 * loss_m + 0.5 * loss_f

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_m_loss += loss_m.item()
        total_f_loss += loss_f.item()

    avg_loss = total_loss / len(dl)
    avg_m_loss = total_m_loss / len(dl)
    avg_f_loss = total_f_loss / len(dl)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {avg_loss:.6f} "
        f"MNIST: {avg_m_loss:.6f} "
        f"Fashion: {avg_f_loss:.6f}"
    )

    # =====================
    # save image
    # =====================
    if (epoch + 1) % 10 == 0:

        G.eval()
        G_f.eval()
        E.eval()
        E_f.eval()

        with torch.no_grad():

            x_m, _ = next(iter(dl))
            x_f, _ = next(iter(dl_f))

            x_m = x_m.to(device)
            x_f = x_f.to(device)

            z_m, _ = E(x_m)
            z_f, _ = E_f(x_f)

            z_m_hat, z_f_hat = mimo_2x2_channel_zf_trainable(
                z_m,
                z_f,
                snr_db=30
            )

            z_m_hat = z_m_hat.view(batch_size, z_dim, 1, 1)
            z_f_hat = z_f_hat.view(batch_size, z_dim, 1, 1)

            recon_m = G(z_m_hat)
            recon_f = G_f(z_f_hat)

            save_image(
                (x_m[:25] + 1) / 2,
                f"{save_root}/transmit_M_epoch_{epoch+1}.png",
                nrow=5
            )

            save_image(
                (recon_m[:25] + 1) / 2,
                f"{save_root}/recon_M_epoch_{epoch+1}.png",
                nrow=5
            )

            save_image(
                (x_f[:25] + 1) / 2,
                f"{save_root}/transmit_F_epoch_{epoch+1}.png",
                nrow=5
            )

            save_image(
                (recon_f[:25] + 1) / 2,
                f"{save_root}/recon_F_epoch_{epoch+1}.png",
                nrow=5
            )

        torch.save(E.state_dict(), "./save_model/E_end2end_MNIST_sep.pth")
        torch.save(E_f.state_dict(), "./save_model/E_end2end_FashionMNIST_sep.pth")
        torch.save(G.state_dict(), "./save_model/G_end2end_MNIST_sep.pth")
        torch.save(G_f.state_dict(), "./save_model/G_end2end_FashionMNIST_sep.pth")

        G.train()
        G_f.train()
        E.train()
        E_f.train()

# =====================
# final save
# =====================
torch.save(E.state_dict(), "./save_model/E_end2end_MNIST_sep.pth")
torch.save(E_f.state_dict(), "./save_model/E_end2end_FashionMNIST_sep.pth")
torch.save(G.state_dict(), "./save_model/G_end2end_MNIST_sep.pth")
torch.save(G_f.state_dict(), "./save_model/G_end2end_FashionMNIST_sep.pth")

end_time = time.time()
print(f"執行時間: {end_time - start_time:.4f} 秒")