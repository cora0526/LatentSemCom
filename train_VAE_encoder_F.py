import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import Ganerator_MNIST, VAE_Encoder_MNIST
import wandb

def train_encoder_invert_gan(
    epochs=20,
    batch_size=128,
    z_dim=64,
    lr=2e-4,
    device="cuda",
    save_every=5,
):
        # same preprocessing as GAN training: [-1,1]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    ds = datasets.FashionMNIST(root=r"D:\Python\SemCom\saved_data", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # ---- load pretrained G ----
    G = Ganerator_MNIST(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load("./save_model/G_wgangp_C.pth", map_location=device))
    G.eval()
    for p in G.parameters():
        p.requires_grad = False

    # ---- encoder (use your class) ----
    E = VAE_Encoder_MNIST(latent_dim=z_dim).to(device)
    optE = torch.optim.Adam(E.parameters(), lr=lr)

    fixed_x, _ = next(iter(dl))
    fixed_x = fixed_x[:64].to(device)
    fixed_x_cpu = fixed_x.detach().cpu()
    save_image((fixed_x_cpu + 1) / 2, "./save_data/fixed_x.png",nrow=8)

    step = 0
    for ep in range(epochs):
        E.train()
        total = 0.0

        for real, _ in dl:
            real = real.to(device)
            b = real.size(0)

            mu, logvar = E(real)          # (B, z_dim)
            z = mu.view(b, z_dim, 1, 1)   # (B, z_dim, 1, 1)

            recon = G(z)                  # (B,1,28,28)

            # Reconstruction loss (real & recon are in [-1,1])
            loss_recon = F.l1_loss(recon, real)

            # Optional latent regularization: keep z near N(0,1) scale
            loss_z = (mu**2).mean()

            loss = loss_recon + 1e-3 * loss_z

            optE.zero_grad(set_to_none=True)
            loss.backward()
            optE.step()

            total += loss.item()
            if step % 200 == 0:
                print(f"[E-invert] ep {ep}/{epochs} step {step} | recon {loss_recon.item():.4f} | zreg {loss_z.item():.4f} | total {loss.item():.4f}")
            step += 1

        print(f"Epoch {ep} avg loss: {total/len(dl):.4f}")

        if (ep + 1) % save_every == 0:
            E.eval()
            with torch.no_grad():
                mu, _ = E(fixed_x)
                z = mu.view(mu.size(0), z_dim, 1, 1)
                recon = G(z)
                save_image((recon + 1) / 2, f"./save_data/invert_recon_ep{ep+1}.png", nrow=8)

    torch.save(E.state_dict(), "./save_model/E_invert_to_G_F.pth")
    print("Saved encoder: E_invert_to_G_F.pth")

if __name__ == "__main__":
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="candy273321-national-central-university",
    # Set the wandb project where this run will be logged.
    project="VAE-Encoder-invert-GAN",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 2e-4,
        "architecture": "VAE-Encoder",
        "dataset": "FashionMNIST",
        "epochs": 20,
        "batch_size": 128,
        "z_dim": 64,
        "device": "cuda"
    },
)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_encoder_invert_gan(device=device)
    run.finish()