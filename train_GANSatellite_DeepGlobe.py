import torch
import glob
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from model import Generator_Satellite, Critic_Satellite
from utils import gradient_penalty


class SatelliteDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img


def main():
    epochs = 150
    batch_size = 64
    z_dim = 256
    lr = 1e-4
    n_critic = 5
    lambda_gp = 10.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    save_root = "./save_data/GANSatellite_64"
    os.makedirs(save_root, exist_ok=True)
    os.makedirs("./save_model", exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    ds = SatelliteDataset(data_root, transform=tfm)
    print("original num images =", len(ds))

    # 先只用 50000 張，不然你的資料可能太多
    ds = Subset(ds, range(min(50000, len(ds))))
    print("used num images =", len(ds))

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    print("steps per epoch =", len(dl))
    print("device =", device)

    G = Generator_Satellite(z_dim=z_dim, out_ch=3).to(device)
    D = Critic_Satellite(in_ch=3).to(device)

    print("G device =", next(G.parameters()).device)
    print("D device =", next(D.parameters()).device)

    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    fixed_z = torch.randn(64, z_dim, 1, 1, device=device)

    real_sample = next(iter(dl))
    save_image((real_sample[:64] + 1) / 2, f"{save_root}/real_batch.png", nrow=8)

    step = 0

    for ep in range(epochs):
        for real in dl:
            real = real.to(device, non_blocking=True)
            b = real.size(0)

            for _ in range(n_critic):
                z = torch.randn(b, z_dim, 1, 1, device=device)
                fake = G(z).detach()

                d_real = D(real).mean()
                d_fake = D(fake).mean()

                gp = gradient_penalty(D, real, fake, device)
                loss_D = d_fake - d_real + lambda_gp * gp

                optD.zero_grad(set_to_none=True)
                loss_D.backward()
                optD.step()

            z = torch.randn(b, z_dim, 1, 1, device=device)
            fake = G(z)

            loss_G = -D(fake).mean()

            optG.zero_grad(set_to_none=True)
            loss_G.backward()
            optG.step()

            if step % 200 == 0:
                print(
                    f"ep {ep+1}/{epochs} step {step} | "
                    f"lossD {loss_D.item():.3f} | "
                    f"lossG {loss_G.item():.3f} | "
                    f"Dreal {d_real.item():.3f} | "
                    f"Dfake {d_fake.item():.3f}"
                )

            step += 1

        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                samples = G(fixed_z)
                samples = (samples + 1) / 2
                save_image(samples, f"{save_root}/samples_ep{ep+1}.png", nrow=8)

    torch.save(G.state_dict(), "./save_model/G_wgangp_satellite.pth")
    torch.save(D.state_dict(), "./save_model/D_wgangp_satellite.pth")
    print("Saved: G_wgangp_satellite.pth, D_wgangp_satellite.pth")


if __name__ == "__main__":
    main()