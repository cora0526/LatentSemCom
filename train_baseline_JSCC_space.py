import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model_baseline import JSCC_MIMO_RGB64
from PIL import Image
import glob

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
    
def train_jscc(dataset_name="MNIST", snr_min=0, snr_max=30):
    device = "cuda" 

    batch_size = 128
    epochs = 50
    lr = 1e-3

    save_root = f"D:/Python/SysTest/train_baseline/JSCC_MIMO_{dataset_name}"
    os.makedirs(save_root, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset_name == "deepglobe":
        data_root = r"D:/Python/SemCom/saved_data/deepglobe_64/train"
        save_root = "./save_data/JSCCSatellite_64"
        os.makedirs(save_root, exist_ok=True)
        ds = SatelliteDataset(data_root, transform=tfm)
    elif dataset_name == "EuroSAT":
        data_root = r"D:/Python/SemCom/saved_data/EuroSAT/EuroSAT"
        save_root = "./save_data/JSCCSatellite_EuroSAT_64"
        os.makedirs(save_root, exist_ok=True)
        ds = datasets.ImageFolder(data_root, transform=tfm)
    else:
        raise ValueError("dataset_name must be 'deepglobe' or 'EuroSAT'")

    ds = Subset(ds, range(min(50000, len(ds))))
    train_loader = DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)

    model = JSCC_MIMO_RGB64(random_snr=True,snr_min=snr_min,snr_max=snr_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        total_loss = 0

        for x in train_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]

            x = x.to(device)

            x_hat = model(x)

            loss = F.mse_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(ds)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Dataset: {dataset_name} "
            f"Train SNR: random {snr_min}-{snr_max} dB "
            f"Loss: {avg_loss:.6f}"
        )

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                x = next(iter(train_loader))
                if isinstance(x, (list, tuple)):
                    x = x[0]
                x = x.to(device)

                x_hat = model(x, snr_db=30)

                comparison = torch.cat([x[:25], x_hat[:25]], dim=0)

                save_image(
                    comparison,
                    f"{save_root}/recon_epoch_{epoch+1}.png",
                    nrow=5
                )

            torch.save(
                model.state_dict(),
                f"./save_model/jscc_mimo_{dataset_name}.pth"
            )

    torch.save(
        model.state_dict(),
        f"./save_model/jscc_mimo_{dataset_name}.pth"
    )


if __name__ == "__main__":
    train_jscc(dataset_name="deepglobe", snr_min=0, snr_max=30)
    train_jscc(dataset_name="EuroSAT", snr_min=0, snr_max=30)

