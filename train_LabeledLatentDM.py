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
    save_image(fixed_x, f"./save_data/test_labelDM/dm_recon_test_M.png", nrow=8)
    fixed_x_F, _ = next(iter(dl_f))
    fixed_x_F = fixed_x_F[:64].to(device)
    save_image(fixed_x_F, f"./save_data/test_labelDM/dm_recon_test_F.png", nrow=8)

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
    
    T_train = int(0.4 * T)   # 先訓練 t <= 400
    b_max = 0.6              # 干擾先不要太強
    log_every = 50

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for step in range(max(len(dl), len(dl_f))):
            x_m, _ = next(m_iter)
            x_f, _ = next(f_iter)
            x_m = x_m.to(device)
            x_f = x_f.to(device)
            B = min(x_m.size(0), x_f.size(0))
            x_m = x_m[:B]
            x_f = x_f[:B]

            with torch.no_grad():
                z_m, _ = E(x_m)
                z_f, _ = E_f(x_f)

            # -------------------------
            # 1) random choose target label
            # -------------------------
            use_fashion = (torch.rand((), device=device) < 0.5)
            if use_fashion:
                y = torch.ones(B, device=device, dtype=torch.long)   # target label=1
                z_tgt = z_f
                z_int = z_m
            else:
                y = torch.zeros(B, device=device, dtype=torch.long)  # target label=0
                z_tgt = z_m
                z_int = z_f

            # -------------------------
            # 2) sample timestep (curriculum)
            # -------------------------
            t = torch.randint(0, T_train, (B,), device=device, dtype=torch.long)

            # -------------------------
            # 3) build total noise = Gaussian + interference-as-noise
            #    重要：把 z_int 當成 noise component
            # -------------------------
            b = torch.empty(B, 1, device=device).uniform_(0.0, b_max)  # interference strength per sample

            eps_g = torch.randn_like(z_tgt)  # AWGN component

            # normalize interference so scale is stable
            eps_i = z_int
            eps_i = eps_i / (eps_i.std(dim=0, keepdim=True) + 1e-6)

            # mix them as total noise
            eps_total = (1.0 - b) * eps_g + b * eps_i  # (B, z_dim)

            # -------------------------
            # 4) forward diffusion with x0 = z_tgt
            # -------------------------
            zt = sqrt_ab[t].view(B, 1) * z_tgt + sqrt_1mab[t].view(B, 1) * eps_total

            # -------------------------
            # 5) DM predicts total noise conditioned on label
            # -------------------------
            eps_pred = DM(zt, t, y)

            # reconstruct target latent
            z_tgt_hat = (zt - sqrt_1mab[t].view(B, 1) * eps_pred) / sqrt_ab[t].view(B, 1)

            # -------------------------
            # 6) loss: only target recovery
            # -------------------------
            loss = F.mse_loss(z_tgt_hat, z_tgt)

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
                # encode fixed sets
                z0_m, _ = E(fixed_x)         # (64, z_dim)
                z0_f, _ = E_f(fixed_x_F)     # (64, z_dim)
                B = z0_m.size(0)

                # validation settings (跟 training 對齊，但固定住方便比較)
                t_val = 300
                t = torch.full((B,), t_val, device=device, dtype=torch.long)

                b_val = 0.4  # 干擾強度（可調 0.2/0.4/0.6 看曲線）
                b = torch.full((B, 1), b_val, device=device)

                torch.manual_seed(0)
                eps_g = torch.randn_like(z0_m)

                # ---------- Case A: target=MNIST (y=0), interference=Fashion ----------
                y0 = torch.zeros(B, device=device, dtype=torch.long)
                z_tgt = z0_m
                z_int = z0_f
                eps_i = z_int / (z_int.std(dim=0, keepdim=True) + 1e-6)
                eps_total = (1.0 - b) * eps_g + b * eps_i

                zt = sqrt_ab[t_val] * z_tgt + sqrt_1mab[t_val] * eps_total

                eps_pred = DM(zt, t, y0)
                z_hat = (zt - sqrt_1mab[t_val] * eps_pred) / sqrt_ab[t_val]

                mse0 = F.mse_loss(z_hat, z_tgt).item()
                print(f"[val-target=MNIST] ep={epoch+1} t={t_val} b={b_val} mse={mse0:.6f}")

                xhat0 = G(z_hat.view(B, z_dim, 1, 1))
                xref0 = G(z_tgt.view(B, z_dim, 1, 1))

                save_image((xref0 + 1) / 2, f"./save_data/test_labelDM/ref_M_ep{epoch+1}.png", nrow=8)
                save_image((xhat0 + 1) / 2, f"./save_data/test_labelDM/denoise_as_M_ep{epoch+1}_t{t_val}_b{b_val}.png", nrow=8)

                # ---------- Case B: target=Fashion (y=1), interference=MNIST ----------
                y1 = torch.ones(B, device=device, dtype=torch.long)
                z_tgt = z0_f
                z_int = z0_m
                eps_i = z_int / (z_int.std(dim=0, keepdim=True) + 1e-6)
                eps_total = (1.0 - b) * eps_g + b * eps_i

                zt = sqrt_ab[t_val] * z_tgt + sqrt_1mab[t_val] * eps_total

                eps_pred = DM(zt, t, y1)
                z_hat = (zt - sqrt_1mab[t_val] * eps_pred) / sqrt_ab[t_val]

                mse1 = F.mse_loss(z_hat, z_tgt).item()
                print(f"[val-target=Fashion] ep={epoch+1} t={t_val} b={b_val} mse={mse1:.6f}")

                xhat1 = G(z_hat.view(B, z_dim, 1, 1))
                xref1 = G(z_tgt.view(B, z_dim, 1, 1))

                # -------------------------
                # 5) save images
                # -------------------------
                # if epoch==4:
                #     save_image((xref_m + 1) / 2, f"./save_data/test_labelDM/ref_M_ep{epoch+1}.png", nrow=8)
                #     save_image((xref_f + 1) / 2, f"./save_data/test_labelDM/ref_F_ep{epoch+1}.png", nrow=8)

                save_image((xref1 + 1) / 2, f"./save_data/test_labelDM/ref_F_ep{epoch+1}.png", nrow=8)
                save_image((xhat1 + 1) / 2, f"./save_data/test_labelDM/denoise_as_F_ep{epoch+1}_t{t_val}_b{b_val}.png", nrow=8)
                # save_image((xhat_m + 1) / 2, f"./save_data/test_labelDM/mix_recon_as_M_ep{epoch+1}_t{t_val}.png", nrow=8)
                # save_image((xhat_f + 1) / 2, f"./save_data/test_labelDM/mix_recon_as_F_ep{epoch+1}_t{t_val}.png", nrow=8)

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