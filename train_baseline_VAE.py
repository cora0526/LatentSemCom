from model_baseline import VAE_MNIST, vae_loss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import torch
# =====================
# Train
# =====================
def train_vae(dataset_name="MNIST"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 64
    batch_size = 128
    epochs = 50
    lr = 1e-3

    save_root = f"D:/Python/SysTest/train_baseline/VAE_{dataset_name}"
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
        raise ValueError("dataset_name must be 'MNIST' or 'FashionMNIST'")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    model = VAE_MNIST(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0

        for x, _ in train_loader:
            x = x.to(device)

            x_hat, mu, logvar = model(x)

            loss, recon_loss, kl_loss = vae_loss(x_hat, x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        avg_loss = total_loss / len(train_dataset)
        avg_recon = total_recon / len(train_dataset)
        avg_kl = total_kl / len(train_dataset)

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
                x, _ = next(iter(train_loader))
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
    train_vae(dataset_name="MNIST")
    # train_vae(dataset_name="FashionMNIST")