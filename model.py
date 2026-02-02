import torch, math
import torch.nn as nn
import torch.nn.functional as F


# MNIST WGAN-GP
class Ganerator_MNIST(nn.Module):
    def __init__(self, z_dim=64, out_ch=1):
        super(Ganerator_MNIST, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
    
class Discriminator_MNIST(nn.Module):
    def __init__(self, in_ch=1):
        super(Discriminator_MNIST, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.LazyLinear(1)
        )

    def forward(self, x):
        return self.model(x).view(-1)


class VAE_Encoder_MNIST(nn.Module):
    def __init__(self, latent_dim=64):
        super(VAE_Encoder_MNIST, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # 28 → 14
            nn.GroupNorm(8, 32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 14 → 7
            nn.GroupNorm(8, 64),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.model(x)
        h = self.flatten(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
# DDPM for MNIST


def timeembedding(t, dim):
    # t: (B,) int64
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb  # (B, dim)

class LatentDenoiser(nn.Module):
    def __init__(self, tdim=128, z_dim=64):
        super().__init__()
        self.tdim = tdim

        self.net = nn.Sequential(
            nn.Linear(z_dim + tdim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, z_dim),
        )

    def forward(self, zt, t):
        te = timeembedding(t, self.tdim)
        h = torch.cat([zt, te], dim=1)
        return self.net(h)