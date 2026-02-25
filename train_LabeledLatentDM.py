from model import Ganerator_MNIST, VAE_Encoder_MNIST, LabelCondLatentDenoiser
from train_VAE_encoder import train_encoder_invert_gan
from utils import linear_beta_schedule, denoise_latent_ddpm, eps_from_Ef, ddpm_loss_label_cond, sample_latent_label_cond
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
    m_iter = cycle(dl)

    # 固定測試集資料
    fixed_x, _ = next(iter(dl))
    fixed_x = fixed_x[:64].to(device)
    save_image(fixed_x, f"./save_data/dm_recon_test_M.png", nrow=8)
    fixed_x_F, _ = next(iter(dl_f))
    fixed_x_F = fixed_x_F[:64].to(device)
    save_image(fixed_x_F, f"./save_data/dm_recon_test_F.png", nrow=8)

    # 導入encoder和GAN模型 都是freeze的
    G = Ganerator_MNIST(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load("./save_model/G_wgangp_C.pth"))
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
    DM =  LabelCondLatentDenoiser(num_classes=2, z_dim=z_dim, tdim=128, ydim=64).to(device)
    opt = torch.optim.Adam(DM.parameters(), lr=2e-4)
    
    log_every = 50
    # 開始訓練
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for step in range(max(len(dl), len(dl_f))):
            # -------------------------
            # 隨機選 domain
            # -------------------------
            use_fashion = torch.rand(1).item() < 0.5

            if use_fashion:
                x_f, _ = next(f_iter)
                x = x_f.to(device)
                Enc = E_f
                y = torch.ones(x.size(0), device=device, dtype=torch.long)
                
                # interference 來自 MNIST
                eps_f = eps_from_Ef(m_iter, E, device)

            else:
                x_m, _ = next(m_iter)
                x = x_m.to(device)
                Enc = E
                y = torch.zeros(x.size(0), device=device, dtype=torch.long)
                
                # interference 來自 Fashion
                eps_f = eps_from_Ef(f_iter, E_f, device)

            b = x.size(0)

            # -------------------------
            # encode
            # -------------------------
            with torch.no_grad():
                mu, _ = Enc(x)
                z0 = mu

            # -------------------------
            # sample timestep
            # -------------------------
            t = torch.randint(0, T, (b,), device=device)

            # -------------------------
            # noise (interference + Gaussian)
            # -------------------------
            lam = 0.5
            eps = (1 - lam) * torch.randn_like(z0) + lam * eps_f[:b]
            zt = sqrt_ab[t].view(b, 1) * z0 + sqrt_1mab[t].view(b, 1) * eps

            # -------------------------
            # predict noise
            # -------------------------
            eps_pred = DM(zt, t, y)
            loss = F.mse_loss(eps_pred, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

            if (step + 1) % log_every == 0:
                running_avg = total_loss / num_batches
                print(f"Epoch {epoch+1}/{epochs} | step {step+1} | Avg Loss: {running_avg:.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        print(f"[Epoch End] {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")
        if (epoch + 1) % save_every == 0:
            DM.eval()
            with torch.no_grad():
                # -------------------------
                # 0) encode fixed test sets
                # -------------------------
                mu_m, _ = E(fixed_x)         # (B, z_dim)  MNIST latent
                mu_f, _ = E_f(fixed_x_F)     # (B, z_dim)  Fashion latent
                z0_m = mu_m
                z0_f = mu_f

                B = z0_m.size(0)

                # -------------------------
                # 1) build "received" mixed latent + noise
                #    z_mix = a*z_m + b*z_f
                # -------------------------
                a = 0.5
                b = 0.5
                z0_mix = a * z0_m + b * z0_f

                # choose a diffusion noise level for validation
                t_val = int(0.3 * T)   # e.g., 300
                t = torch.full((B,), t_val, device=device, dtype=torch.long)

                # Gaussian noise (channel noise)
                torch.manual_seed(0)
                eps = torch.randn_like(z0_mix)

                # forward noising (timestep t_val)
                zT_mix = sqrt_ab[t_val] * z0_mix + sqrt_1mab[t_val] * eps

                # -------------------------
                # 2) run DM twice with different labels
                #    y=0 -> recover MNIST-like latent
                #    y=1 -> recover Fashion-like latent
                # -------------------------
                y_m = torch.zeros(B, device=device, dtype=torch.long)  # MNIST label
                y_f = torch.ones(B, device=device, dtype=torch.long)   # Fashion label

                # predict eps for each label
                eps_pred_m = DM(zT_mix, t, y_m)
                eps_pred_f = DM(zT_mix, t, y_f)

                # one-step posterior mean estimate of z0 (like your z0_hat_0)
                z0_hat_m = (zT_mix - sqrt_1mab[t_val] * eps_pred_m) / sqrt_ab[t_val]
                z0_hat_f = (zT_mix - sqrt_1mab[t_val] * eps_pred_f) / sqrt_ab[t_val]

                # -------------------------
                # 3) metrics (how close to original z0_m / z0_f)
                # -------------------------
                mse_m = F.mse_loss(z0_hat_m, z0_m).item()
                mse_f = F.mse_loss(z0_hat_f, z0_f).item()
                print(f"[mix-val] ep={epoch+1} t={t_val} mse_to_M={mse_m:.6f} mse_to_F={mse_f:.6f}")

                # -------------------------
                # 4) decode with Generator
                # -------------------------
                xhat_m = G(z0_hat_m.view(B, z_dim, 1, 1))
                xhat_f = G(z0_hat_f.view(B, z_dim, 1, 1))

                # references: decode clean latents too (optional but recommended)
                xref_m = G(z0_m.view(B, z_dim, 1, 1))
                xref_f = G(z0_f.view(B, z_dim, 1, 1))

                # -------------------------
                # 5) save images
                # -------------------------
                save_image((xref_m + 1) / 2, f"./save_data/test_labelDM/ref_M_ep{epoch+1}.png", nrow=8)
                save_image((xref_f + 1) / 2, f"./save_data/test_labelDM/ref_F_ep{epoch+1}.png", nrow=8)

                save_image((xhat_m + 1) / 2, f"./save_data/test_labelDM/mix_recon_as_M_ep{epoch+1}_t{t_val}.png", nrow=8)
                save_image((xhat_f + 1) / 2, f"./save_data/test_labelDM/mix_recon_as_F_ep{epoch+1}_t{t_val}.png", nrow=8)

                # save model
                torch.save(DM.state_dict(), "./save_model/DM_latent_label.pth")
                print("Saved DM: DM_latent_label.pth")

            DM.train()

    torch.save(DM.state_dict(), "./save_model/DM_latent_label.pth")
    print("Saved DM: DM_latent_label.pth")
    
if __name__ == "__main__":
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="candy273321-national-central-university",
    # Set the wandb project where this run will be logged.
    project="LabelLatentDM_MNIST",
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