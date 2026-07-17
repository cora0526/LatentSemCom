import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from model import Encoder_Satellite, Generator_Satellite
from com_utils import mimo_2x2_channel_zf_train
from utils import SatelliteDataset
from random import randint
import os

def power_normalize(z, eps=1e-8):
    power = z.pow(2).mean(
        dim=1,
        keepdim=True
    )

    return z / torch.sqrt(
        power + eps
    )

def get_img(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def main(epochs=100,batch_size=128,z_dim=256,lr=1e-4,save_every=5,device="cuda"):
    save_root = "./save_data/EncoderSatellite_CommonGAN_TwoEncoder_with_fix_H"
    os.makedirs(save_root, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    data_root_g = r"D:/Python/SemCom/saved_data/deepglobe_64/train"

    ds_e = datasets.ImageFolder(data_root_e, transform=tfm)
    ds_g = SatelliteDataset(data_root_g, transform=tfm)

    ds_e = Subset(ds_e, range(min(50000, len(ds_e))))
    ds_g = Subset(ds_g, range(min(50000, len(ds_g))))

    dl_e = DataLoader(
        ds_e,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    dl_g = DataLoader(
        ds_g,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # ---- load pretrained G ----
    G = Generator_Satellite(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load("./save_model/G_wgangp_satellite_c.pth", map_location=device))
    G.eval()

    for p in G.parameters():
        p.requires_grad = False

    # ---- encoder (use your class) ----
    E_e = Encoder_Satellite(z_dim=z_dim).to(device)  # EuroSAT encoder
    E_g = Encoder_Satellite(z_dim=z_dim).to(device)  # DeepGlobe encoder

    opt = torch.optim.Adam(
        list(E_e.parameters()) + list(E_g.parameters()),
        lr=lr
    )

    fixed_e = get_img(next(iter(dl_e)))[:64].to(device)
    fixed_g = get_img(next(iter(dl_g)))[:64].to(device)

    save_image((fixed_e + 1) / 2, f"{save_root}/real_EuroSAT.png", nrow=8)
    save_image((fixed_g + 1) / 2, f"{save_root}/real_deepglobe.png", nrow=8)
    
    step = 0
    for ep in range(epochs):
        E_e.train()
        E_g.train()

        total = 0.0
        total_e = 0.0
        total_g = 0.0

        for batch_e, batch_g in zip(dl_e, dl_g):
            real_e = get_img(batch_e).to(device, non_blocking=True)
            real_g = get_img(batch_g).to(device, non_blocking=True)

            b_e = real_e.size(0)
            b_g = real_g.size(0)

            mu_e_clean, _ = E_e(real_e)
            mu_g_clean, _ = E_g(real_g)

            snr_db = torch.empty(1,device=device).uniform_(0.0, 30.0).item()

            mu_e_tx = power_normalize(mu_e_clean)
            mu_g_tx = power_normalize(mu_g_clean)

            mu_e, mu_g = mimo_2x2_channel_zf_train(mu_e_tx, mu_g_tx, snr_db)

            z_e = mu_e.view(b_e, z_dim, 1, 1)
            z_g = mu_g.view(b_g, z_dim, 1, 1)

            recon_e = G(z_e)
            recon_g = G(z_g)

            loss_recon_e = F.l1_loss(recon_e, real_e)
            loss_recon_g = F.l1_loss(recon_g, real_g)

            loss_latent_e = F.mse_loss(mu_e,mu_e_tx)
            loss_latent_g = F.mse_loss(mu_g,mu_g_tx)


            lambda_latent = 0.1

            loss_e = (loss_recon_e + lambda_latent * loss_latent_e )

            loss_g = (loss_recon_g + lambda_latent * loss_latent_g )

            loss = loss_e + loss_g

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
            total_e += loss_e.item()
            total_g += loss_g.item()

            if step % 200 == 0:
                print(
                    f"ep {ep+1}/{epochs} step {step} | "
                    f"EuroSAT {loss_e.item():.4f} | "
                    f"DeepGlobe {loss_g.item():.4f} | "
                    f"Total {loss.item():.4f}"
                )

            step += 1

        print(
            f"Epoch {ep+1}/{epochs} | "
            f"avg total {total / len(dl_e):.4f} | "
            f"EuroSAT {total_e / len(dl_e):.4f} | "
            f"DeepGlobe {total_g / len(dl_e):.4f}"
        )

        if (ep + 1) % save_every == 0:
            E_e.eval()
            E_g.eval()

            with torch.no_grad():
                mu_e_clean, _ = E_e(fixed_e)
                mu_g_clean, _ = E_g(fixed_g)

                mu_e_tx = power_normalize(mu_e_clean)
                mu_g_tx = power_normalize(mu_g_clean)

                test_snr_db = 10.0

                mu_e_rx, mu_g_rx = mimo_2x2_channel_zf_train(
                    mu_e_tx,
                    mu_g_tx,
                    test_snr_db
                )

                recon_e = G(
                    mu_e_rx.view(mu_e_rx.size(0), z_dim, 1, 1)
                )

                recon_g = G(
                    mu_g_rx.view(mu_g_rx.size(0), z_dim, 1, 1)
                )

                save_image(
                    (recon_e + 1) / 2,
                    f"{save_root}/recon_EuroSAT_ep{ep+1}.png",
                    nrow=8
                )

                save_image(
                    (recon_g + 1) / 2,
                    f"{save_root}/recon_deepglobe_ep{ep+1}.png",
                    nrow=8
                )

            torch.save(
                E_e.state_dict(),
                "./save_model/E_invert_to_G_C_EuroSAT_fixed_H.pth"
            )
            torch.save(
                E_g.state_dict(),
                "./save_model/E_invert_to_G_C_deepglobe_fixed_H.pth"
            )

    torch.save(E_e.state_dict(), "./save_model/E_invert_to_G_C_EuroSAT_fixed_H.pth")
    torch.save(E_g.state_dict(), "./save_model/E_invert_to_G_C_deepglobe_fixed_H.pth")

    print("Saved two encoders.")


if __name__ == "__main__":
    main()