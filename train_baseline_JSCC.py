import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model_baseline import JSCC_MIMO

def train_jscc(dataset_name="MNIST", snr_min=0, snr_max=30):
    device = "cuda" 

    batch_size = 128
    epochs = 50
    lr = 1e-3

    save_root = f"D:/Python/SysTest/train_baseline/JSCC_MIMO_{dataset_name}"
    os.makedirs(save_root, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            root=r"D:/Python/SemCom/saved_data",
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root=r"D:/Python/SemCom/saved_data",
            train=True,
            download=True,
            transform=transform
        )
    else:
        raise ValueError("dataset_name must be MNIST or FashionMNIST")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    model = JSCC_MIMO(random_snr=True,snr_min=snr_min,snr_max=snr_max).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        total_loss = 0

        for x, _ in train_loader:
            x = x.to(device)

            x_hat = model(x)

            if dataset_name == "MNIST":
                loss = F.binary_cross_entropy(x_hat, x)
            else:
                loss = F.mse_loss(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_dataset)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Dataset: {dataset_name} "
            f"Train SNR: random {snr_min}-{snr_max} dB "
            f"Loss: {avg_loss:.6f}"
        )

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                x, _ = next(iter(train_loader))
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
    train_jscc(dataset_name="MNIST", snr_min=0, snr_max=30)
    train_jscc(dataset_name="FashionMNIST", snr_min=0, snr_max=30)

