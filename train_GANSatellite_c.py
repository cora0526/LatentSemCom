import torch
import glob
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms, datasets
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

        label = 1  

        return img, label

def main():
    epochs = 200
    batch_size = 128
    z_dim = 256
    lr = 5e-5
    n_critic = 5
    lambda_gp = 10.0
    device = "cuda" 

    data_root_e = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
    data_root_g = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
    save_root = "./save_data/GANSatellite_c"
    os.makedirs(save_root, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    ds_e = datasets.ImageFolder(root = data_root_e,transform=tfm)
    ds_e = Subset(ds_e, range(min(50000, len(ds_e))))
    # dl_e = DataLoader(ds_e,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,drop_last=True)
    ds_g = SatelliteDataset(data_root_g, transform=tfm)
    ds_g = Subset(ds_g, range(min(50000, len(ds_g))))
    # dl_g = DataLoader(ds_g,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,drop_last=True)
    ds = ConcatDataset([ds_e, ds_g])
    dl = DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True,drop_last=True)
    
    G = Generator_Satellite(z_dim=z_dim, out_ch=3).to(device)
    D = Critic_Satellite(in_ch=3).to(device)

    print("G device =", next(G.parameters()).device)
    print("D device =", next(D.parameters()).device)

    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    fixed_z = torch.randn(64, z_dim, 1, 1, device=device)

    real_sample,_ = next(iter(dl))
    real_sample = real_sample.to(device)
    save_image((real_sample[:64] + 1) / 2, f"{save_root}/real_batch.png", nrow=8)

    step = 0

    for ep in range(epochs):
        for batch in dl:
            if isinstance(batch, (list, tuple)):
                real = batch[0]
            else:
                real = batch

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

    torch.save(G.state_dict(), "./save_model/G_wgangp_satellite_c.pth")
    torch.save(D.state_dict(), "./save_model/D_wgangp_satellite_c.pth")
    print("Saved: G_wgangp_satellite_EuroSAT.pth, D_wgangp_satellite_c.pth")


if __name__ == "__main__":
    main()