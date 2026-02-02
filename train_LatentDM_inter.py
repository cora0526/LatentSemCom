from model import Ganerator_MNIST, VAE_Encoder_MNIST, LatentDenoiser
from train_VAE_encoder import train_encoder_invert_gan
from utils import linear_beta_schedule, denoise_latent_ddpm, eps_from_Ef
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from torchvision.utils import save_image
from itertools import cycle

def train_latentDM(
    epochs = 200,
    batch_size = 128,
    z_dim = 64,
    device = "cuda",
    save_every = 5
):
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

    # 固定測試集資料
    fixed_x, _ = next(iter(dl))
    fixed_x = fixed_x[:64].to(device)
    save_image(fixed_x, f"./save_data/dm_recon_test.png", nrow=8)

    # 導入encoder和GAN模型 都是freeze的
    G = Ganerator_MNIST(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load("./save_model/G_wgangp.pth"))
    G.eval()
    for p in G.parameters():
        p.requires_grad = False

    E = VAE_Encoder_MNIST(latent_dim=z_dim).to(device)
    E.load_state_dict(torch.load("./save_model/E_invert_to_G_C.pth"))
    E.eval()
    for p in E.parameters():
        p.requires_grad = False

    E_f = VAE_Encoder_MNIST(latent_dim=z_dim).to(device)
    E_f.load_state_dict(torch.load("./save_model/E_invert_to_G_F.pth"))
    E_f.eval()
    for p in E_f.parameters():
        p.requires_grad = False
    
    # DDPM settings
    T = 1000
    betas = linear_beta_schedule(T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    sqrt_ab = torch.sqrt(alpha_bars)
    sqrt_1mab = torch.sqrt(1 - alpha_bars)
    
    #輸入DM model 
    DM = LatentDenoiser(z_dim=64, tdim=128).to(device)
    opt = torch.optim.Adam(DM.parameters(), lr=2e-4)
    
    # 開始訓練
    for epoch in range(epochs):
        num_batches = 0
        total_loss = 0.0
        for x, _ in dl:
            x = x.to(device)
            b = x.size(0)

            # -------------------------
            # encode to latent
            # -------------------------
            with torch.no_grad():
                mu, _ = E(x)
                z0 = mu                         # (B,64)

            # -------------------------
            # sample timestep
            # -------------------------
            # 隨機抽一個步數
            t = torch.randint(0, T, (b,), device=device)
            
            # 產生一個和latent相同大小的noise
            eps_f = eps_from_Ef(f_iter, E_f, device)
            eps = eps_f[:b] 
            # eps = torch.randn_like(z0)
            lam = 0.5  # 0~1，越大越偏向 E_f 噪聲
            eps = (1 - lam) * torch.randn_like(z0) + lam * eps_f[:b]

            zt = (
                sqrt_ab[t].view(b,1) * z0 + sqrt_1mab[t].view(b,1) * eps
            )

            # -------------------------
            # predict noise
            # -------------------------
            eps_pred = DM(zt, t)

            loss = F.mse_loss(eps_pred, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        if (epoch + 1) % save_every == 0:
            DM.eval()
            with torch.no_grad():
                # 1) encode
                mu, _ = E(fixed_x)
                z0 = mu  # (64, z_dim)

                # 2) 先挑一個噪聲步數來測（例如 t=T-1 最難）
                t_val = int(0.3 * T)   # 300
                tt = torch.full((z0.size(0),), t_val, device=device, dtype=torch.long)

                torch.manual_seed(0)        # 固定每次產生的noise
                eps = torch.randn_like(z0)

                zT = sqrt_ab[t_val] * z0 + sqrt_1mab[t_val] * eps

                # 3) denoise (reverse diffusion)
                z0_hat = denoise_latent_ddpm(DM, zT, betas, alphas, alpha_bars)
                
                eps_pred = DM(zT, tt)
                z0_hat_0 = (zT - sqrt_1mab[t_val] * eps_pred) / sqrt_ab[t_val]
                # 4) generate
                x_hat = G(z0_hat.view(z0_hat.size(0), z_dim, 1, 1))
                x_hat_0 = G(z0_hat_0.view(z0_hat_0.size(0), z_dim, 1, 1))
                x_ = G(z0.view(z0.size(0), z_dim, 1, 1))
                # 5) save
                latent_mse_direct = F.mse_loss(z0_hat_0, z0).item()
                latent_mse_full = F.mse_loss(z0_hat, z0).item()
                print(f"[val] ep={epoch+1} t={t_val} mse_direct={latent_mse_direct:.6f} mse_full={latent_mse_full:.6f}")
                save_image((x_hat + 1) / 2, f"./save_data/dm_recon_ep{epoch+1}.png", nrow=8)
                save_image((x_hat_0 + 1) / 2, f"./save_data/dm_recon_0{epoch+1}_t{t_val}.png", nrow=8)
                save_image((x_ + 1) / 2, f"./save_data/g_recon_ep{epoch+1}.png", nrow=8)

            DM.train()

    torch.save(DM.state_dict(), "./save_model/DM_latent_inter.pth")
    print("Saved DM: DM_latent_inter.pth")
    
if __name__ == "__main__":
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="candy273321-national-central-university",
    # Set the wandb project where this run will be logged.
    project="LatentDM_MNIST",
    # Track hyperparameters and run metadata.
    config={
        "architecture": "DM-Latent",
        "dataset": "MNIST",
        "epochs": 1000,
        "batch_size": 128,
        "z_dim": 64,
        "device": "cuda"
    },
)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_latentDM(device=device)
    run.finish()