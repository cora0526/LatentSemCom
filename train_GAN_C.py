import torch
from torch.utils.data import DataLoader,ConcatDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model import Ganerator_MNIST, Discriminator_MNIST
from utils import gradient_penalty
import wandb

def train_wgangp_mnist(
    epochs=200,
    batch_size=128,
    z_dim=64,
    lr=2e-4,
    n_critic=5,
    lambda_gp=10.0,
    device="cuda"
):
    
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST: (x-0.5)/0.5 -> [-1,1]
    ])
    ds_f = datasets.FashionMNIST(root="D:/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
    ds_m = datasets.MNIST(root="D:/Python/SemCom/saved_data", train=True, download=True, transform=tfm)
    ds = ConcatDataset([ds_f, ds_m])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    G = Ganerator_MNIST(z_dim=z_dim).to(device)
    D = Discriminator_MNIST().to(device)

    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    fixed_z = torch.randn(64, z_dim, 1, 1, device=device)
    step = 0
    for ep in range(epochs):
        for real, _ in dl:
            real = real.to(device)  # (B,1,28,28)
            b = real.size(0)

            # -------------------------
            # 1) Train Critic n_critic times
            # -------------------------
            for _ in range(n_critic):
                z = torch.randn(b, z_dim, 1, 1, device=device)
                fake = G(z).detach()

                d_real = D(real).mean()
                d_fake = D(fake).mean()

                gp = gradient_penalty(D, real, fake, device)
                loss_D = (d_fake - d_real) + lambda_gp * gp

                optD.zero_grad(set_to_none=True)
                loss_D.backward()
                optD.step()

                # refresh real batch if you want strictness; but MNIST ok to reuse

            # -------------------------
            # 2) Train Generator once
            # -------------------------
            z = torch.randn(b, z_dim, 1, 1, device=device)
            fake = G(z)
            loss_G = -D(fake).mean()

            optG.zero_grad(set_to_none=True)
            loss_G.backward()
            optG.step()

            if step % 200 == 0:
                print(f"ep {ep}/{epochs} step {step} | lossD {loss_D.item():.3f} | lossG {loss_G.item():.3f} | Dreal {d_real.item():.3f} | Dfake {d_fake.item():.3f}")

            step += 1

        if (ep+1) % 10 == 0:
            with torch.no_grad():
                samples = G(fixed_z)
                # back to [0,1] for saving: (x+1)/2
                save_image((samples + 1) / 2, f"./save_data/GAN_C/samples_ep{ep}.png", nrow=8)

            

    torch.save(G.state_dict(), "./save_model/G_wgangp_C.pth")
    torch.save(D.state_dict(), "./save_model/D_wgangp_C.pth")
    print("Saved: G_wgangp_C.pth, D_wgangp_C.pth")

if __name__ == "__main__":
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="candy273321-national-central-university",
    # Set the wandb project where this run will be logged.
    project="GAN MNIST+FashionMNIST",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 2e-4,
        "architecture": "GAN-Wasserstein-GP",
        "dataset": "MNIST+FashionMNIST",
        "epochs": 500,
        "batch_size": 128,
        "z_dim": 64,
        "n_critic": 5,
        "lambda_gp": 10.0,
        "device": "cuda"
    },
)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_wgangp_mnist(device=device)
    run.finish()