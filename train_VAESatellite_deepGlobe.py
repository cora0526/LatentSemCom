import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image
from model import Encoder_Satellite, Generator_Satellite
from utils import SatelliteDataset
import os

def main(
    epochs=100,
    batch_size=128,
    z_dim=256,
    lr=2e-4,
    save_every=5,
    device="cuda"
):
        # same preprocessing as GAN training: [-1,1]
    data_root = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    save_root = "./save_data/GANSatellite_64_encoder"
    os.makedirs(save_root, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    ds = SatelliteDataset(data_root, transform=tfm)
    ds = Subset(ds, range(min(50000, len(ds))))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    # ---- load pretrained G ----
    G = Generator_Satellite(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load("./save_model/G_wgangp_satellite.pth", map_location=device))
    G.eval()

    for p in G.parameters():
        p.requires_grad = False

    # ---- encoder (use your class) ----
    E = Encoder_Satellite(z_dim=z_dim).to(device)
    optE = torch.optim.Adam(E.parameters(), lr=lr)

    fixed_x = next(iter(dl))
    fixed_x = fixed_x[:64].to(device)
    
    save_image((fixed_x.detach().cpu() + 1) / 2, f"{save_root}/real_batch.png",nrow=8)
    
    step = 0
    for ep in range(epochs):
        E.train()
        total = 0.0

        for real in dl:
            real = real.to(device)
            b = real.size(0)

            mu, logvar = E(real)
            z = mu.view(b, z_dim, 1, 1)

            recon = G(z)

            loss_recon = F.l1_loss(recon, real)
            loss_z = (mu ** 2).mean()

            loss = loss_recon + 1e-3 * loss_z

            optE.zero_grad(set_to_none=True)
            loss.backward()
            optE.step()

            total += loss.item()
            if step % 200 == 0:
                print(
                    f"[E-invert] ep {ep+1}/{epochs} step {step} | "
                    f"recon {loss_recon.item():.4f} | "
                    f"zreg {loss_z.item():.4f} | "
                    f"total {loss.item():.4f}"
                )
            step += 1

        print(f"Epoch {ep+1}/{epochs} avg loss: {total / len(dl):.4f}")

        if (ep + 1) % save_every == 0:
            E.eval()
            with torch.no_grad():
                mu, _ = E(fixed_x)
                z = mu.view(mu.size(0), z_dim, 1, 1)
                recon = G(z)
                save_image(
                    (recon + 1) / 2,
                    f"{save_root}/invert_recon_ep{ep+1}.png",
                    nrow=8
                )
                l1_diff = F.l1_loss(recon, fixed_x).item()
                mse_diff = F.mse_loss(recon, fixed_x).item()
                psnr = 10 * torch.log10(torch.tensor(4.0, device=device) / F.mse_loss(recon, fixed_x)).item()

                print(
                    f"[Fixed eval] ep {ep+1} | "
                    f"L1: {l1_diff:.4f} | "
                    f"MSE: {mse_diff:.6f} | "
                    f"PSNR: {psnr:.2f} dB"
                )


    torch.save(E.state_dict(), f"./save_model/E_invert_to_G_satellite.pth")
    print("Saved encoder: E_invert_to_G_satellite.pth")

if __name__ == "__main__":
    main()