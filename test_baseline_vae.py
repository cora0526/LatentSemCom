import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from model_baseline import VAE_MNIST

device = "cuda"
z_dim = 64
batch_size = 25

# 與訓練保持一致
tfm = transforms.Compose([
    transforms.ToTensor()
])

# Dataset
ds = datasets.MNIST(
    root=r"D:/Python/SemCom/saved_data",
    train=False,
    download=True,
    transform=tfm
)

ds_f = datasets.FashionMNIST(
    root=r"D:/Python/SemCom/saved_data",
    train=False,
    download=True,
    transform=tfm
)

dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
dl_f = DataLoader(ds_f, batch_size=batch_size, shuffle=False)

# Load model
vae_m = VAE_MNIST(latent_dim=z_dim).to(device)
vae_m.load_state_dict(torch.load("./save_model/baseline_vae_MNIST.pth", map_location=device))
vae_m.eval()

vae_f = VAE_MNIST(latent_dim=z_dim).to(device)
vae_f.load_state_dict(torch.load("./save_model/baseline_vae_FashionMNIST.pth", map_location=device))
vae_f.eval()

# ===== MNIST =====
x_m, _ = next(iter(dl))
x_m = x_m.to(device)

with torch.no_grad():
    mu, logvar = vae_m.encode(x_m)
    recon_m = vae_m.decode(mu)

save_image(
    make_grid(x_m, nrow=5),
    "MNIST_original_5x5.png"
)

save_image(
    make_grid(recon_m, nrow=5),
    "MNIST_recon_5x5.png"
)

comparison = torch.cat([x_m, recon_m], dim=0)

save_image(
    comparison,
    "MNIST_compare.png",
    nrow=5
)

# ===== FashionMNIST =====
x_f, _ = next(iter(dl_f))
x_f = x_f.to(device)

with torch.no_grad():
    mu, logvar = vae_f.encode(x_f)
    recon_f = vae_f.decode(mu)

save_image(
    make_grid(x_f, nrow=5),
    "Fashion_original_5x5.png"
)

save_image(
    make_grid(recon_f, nrow=5),
    "Fashion_recon_5x5.png"
)

comparison = torch.cat([x_f, recon_f], dim=0)

save_image(
    comparison,
    "Fashion_compare.png",
    nrow=5
)