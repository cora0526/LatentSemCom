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
    
# conditional DDPM for MNIST
class LabelCondLatentDenoiser(nn.Module):
    def __init__(self, num_classes: int, tdim: int = 128, z_dim: int = 64, ydim: int = 64):
        super().__init__()
        self.tdim = tdim
        self.z_dim = z_dim
        self.num_classes = num_classes

        # label embedding: y -> (B, ydim)
        self.y_emb = nn.Embedding(num_classes, ydim)

        in_dim = z_dim + tdim + ydim

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, z_dim),
        )

    def forward(self, zt: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        zt: (B, z_dim) received / noisy / mixed latent
        t:  (B,) long, denoising level or dummy timestep
        y:  (B,) long class label
        return z_hat: (B, z_dim) predicted clean target latent
        """
        te = timeembedding(t, self.tdim)     # (B, tdim)
        ye = self.y_emb(y)                  # (B, ydim)
        h = torch.cat([zt, te, ye], dim=1)  # (B, z_dim+tdim+ydim)
        return self.net(h)
    


class DDPMWaveDenoiser1D(nn.Module):
    def __init__(self, ch=64, time_dim=128):
        super().__init__()

        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, ch),
            nn.ReLU(),
            nn.Linear(ch, ch)
        )

        self.conv1 = nn.Conv1d(2, ch, 9, padding=4)
        self.conv2 = nn.Conv1d(ch, ch, 9, padding=4)
        self.conv3 = nn.Conv1d(ch, ch, 9, padding=4)
        self.conv4 = nn.Conv1d(ch, ch, 9, padding=4)
        self.out = nn.Conv1d(ch, 2, 9, padding=4)

        self.act = nn.ReLU()
        self.time_dim = time_dim

    def forward(self, x, t):

        temb = timeembedding(t, self.time_dim)
        temb = self.time_proj(temb)
        temb = temb.unsqueeze(-1)   # [B, ch, 1]

        h = self.conv1(x)
        h = self.act(h + temb)

        h = self.act(self.conv2(h) + temb)
        h = self.act(self.conv3(h) + temb)
        h = self.act(self.conv4(h) + temb)

        out = self.out(h)

        return out

class IQDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(2, 32, 9, padding=4),
            nn.ReLU(),

            nn.Conv1d(32, 64, 9, padding=4),
            nn.ReLU(),

            nn.Conv1d(64, 32, 9, padding=4),
            nn.ReLU(),

            nn.Conv1d(32, 2, 9, padding=4)
        )

    def forward(self, x):
        return self.net(x)    

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class LabelCondLatentDenoiserDeep(nn.Module):
    def __init__(
        self,
        num_classes: int,
        z_dim: int = 64,
        tdim: int = 128,
        ydim: int = 64,
        hidden_dim: int = 512,
        num_blocks: int = 6,
    ):
        super().__init__()

        self.tdim = tdim
        self.z_dim = z_dim
        self.num_classes = num_classes

        self.y_emb = nn.Embedding(num_classes, ydim)

        in_dim = z_dim + tdim + ydim

        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, zt, t, y):
        te = timeembedding(t, self.tdim).to(zt.device)
        ye = self.y_emb(y)

        h = torch.cat([zt, te, ye], dim=1)

        h = self.input(h)
        h = self.blocks(h)

        delta = self.output(h)

        # residual denoising: 學修正量
        z_hat = zt + delta

        return z_hat
    
class Generator_Satellite(nn.Module):
    def __init__(self, z_dim=128, out_ch=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0),  # 1 -> 4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),    # 4 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),     # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, out_ch, 4, 2, 1),  # 32 -> 64
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
    
class Critic_Satellite(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),       # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),         # 32 -> 16
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),        # 16 -> 8
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),        # 8 -> 4
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0)           # 4 -> 1
        )

    def forward(self, x):
        return self.model(x).view(-1)
    
class Encoder_Satellite(nn.Module):
    def __init__(self, in_ch=3, z_dim=128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),      # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),        # 32 -> 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),       # 16 -> 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),       # 8 -> 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(512*4*4, z_dim)
        self.fc_logvar = nn.Linear(512*4*4, z_dim)

    def forward(self, x):
        h = self.backbone(x)
        h = self.flatten(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar