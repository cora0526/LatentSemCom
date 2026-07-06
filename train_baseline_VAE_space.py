from model_baseline import VAE_RGB64, vae_loss_RGB64
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from PIL import Image
import glob
import os
import torch

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

# =====================
# Train
# =====================
def train_vae(dataset_name):
    device = "cuda" 
    latent_dim = 256
    batch_size = 128
    epochs = 100
    lr = 2e-4
    beta = 0.001

    save_root = f"D:/Python/SysTest/train_baseline/VAE_{dataset_name}"
    os.makedirs(save_root, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset_name == "deepglobe":
        data_root = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
        save_root = "./save_data/VAESatellite_64"
        os.makedirs(save_root, exist_ok=True)
        ds = SatelliteDataset(data_root, transform=tfm)
    elif dataset_name == "EuroSAT":
        data_root = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
        save_root = "./save_data/VAESatellite_EuroSAT_64"
        os.makedirs(save_root, exist_ok=True)
        ds = datasets.ImageFolder(data_root, transform=tfm)
    else:
        raise ValueError("dataset_name must be 'deepglobe' or 'EuroSAT'")

    ds = Subset(ds, range(min(50000, len(ds))))
    train_loader = DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)

    model = VAE_RGB64(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0

        for x in train_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]
            
            x = x.to(device)

            x_hat, mu, logvar = model(x)

            loss, recon_loss, kl_loss = vae_loss_RGB64(x_hat, x, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        avg_loss = total_loss / len(ds)
        avg_recon = total_recon / len(ds)
        avg_kl = total_kl / len(ds)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {avg_loss:.4f} "
            f"Recon: {avg_recon:.4f} "
            f"KL: {avg_kl:.4f}"
        )

        # save reconstruction
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                x = next(iter(train_loader))

                if isinstance(x, (list, tuple)):
                    x = x[0]

                x = x.to(device)

                x_hat, _, _ = model(x)

                comparison = torch.cat([x[:8], x_hat[:8]], dim=0)

                save_image(
                    comparison,
                    f"{save_root}/recon_epoch_{epoch+1}.png",
                    nrow=8
                )

            torch.save(
                model.state_dict(),
                f"./save_model/baseline_vae_{dataset_name}.pth"
            )

    torch.save(model.state_dict(), f"./save_model/baseline_vae_{dataset_name}.pth")


if __name__ == "__main__":
    train_vae(dataset_name="deepglobe")
    # train_vae(dataset_name="EuroSAT")