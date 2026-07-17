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
    
# =========================
# DDPM Latent Separator
# MNIST / Satellite
# =========================

def timeembedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


class DDPMResBlock(nn.Module):
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


class CondLatentDDPM(nn.Module):
    """
    Conditional latent DDPM.

    Input:
        zt   : noisy target latent
        t    : diffusion timestep
        y    : label / source id
        cond : received mixed latent, e.g. z_rx or [z_rx, H, SNR]

    Output:
        eps_pred : predicted Gaussian noise
    """

    def __init__(self,z_dim,cond_dim,num_classes=2,tdim=128,ydim=64,hidden_dim=512,num_blocks=6,):
        super().__init__()

        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.tdim = tdim

        self.y_emb = nn.Embedding(num_classes, ydim)

        in_dim = z_dim + cond_dim + tdim + ydim

        self.input = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[DDPMResBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, zt, t, y, cond):
        te = timeembedding(t, self.tdim).to(zt.device)
        ye = self.y_emb(y)

        h = torch.cat([zt, cond, te, ye], dim=1)

        h = self.input(h)
        h = self.blocks(h)

        eps_pred = self.output(h)
        return eps_pred
    
class MNIST_Latent_DDPM(CondLatentDDPM):
    def __init__(self):
        super().__init__(
            z_dim=64,
            cond_dim=64,
            num_classes=2,
            tdim=128,
            ydim=64,
            hidden_dim=512,
            num_blocks=6,
        )

class Satellite_Latent_DDPM(CondLatentDDPM):
    def __init__(self, z_dim=256):
        super().__init__(
            z_dim=z_dim,
            cond_dim=z_dim,
            num_classes=2,
            tdim=128,
            ydim=64,
            hidden_dim=768,
            num_blocks=8,
        )

# ============================================================
# Conditional Generator
# ============================================================
class Generator_Satellite_Conditional(nn.Module):
    def __init__(
        self,
        z_dim=256,
        num_classes=2,
        label_dim=64,
        out_ch=3
    ):
        super().__init__()

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.label_dim = label_dim

        # label: [B] -> [B, label_dim]
        self.label_embedding = nn.Embedding(num_classes,label_dim)

        # z 和 label embedding 串接
        input_dim = z_dim + label_dim

        self.model = nn.Sequential(
            # [B, input_dim, 1, 1] -> [B, 512, 4, 4]
            nn.ConvTranspose2d(input_dim,512,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 4 -> 8
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 8 -> 16
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 16 -> 32
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32 -> 64
            nn.ConvTranspose2d(64,out_ch,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        """
        z:
            [B, z_dim]
            或 [B, z_dim, 1, 1]

        labels:
            [B], dtype=torch.long
        """

        if z.ndim == 4:
            z = z.flatten(1)

        if z.ndim != 2:
            raise ValueError(f"z should be [B,z_dim] or [B,z_dim,1,1], "f"but got {z.shape}")

        labels = labels.long()

        label_emb = self.label_embedding(labels)

        # [B, z_dim + label_dim]
        z_cond = torch.cat([z, label_emb],dim=1)

        # [B, z_dim + label_dim, 1, 1]
        z_cond = z_cond.unsqueeze(-1).unsqueeze(-1)

        return self.model(z_cond)


# ============================================================
# Conditional Projection Critic
# ============================================================
class Critic_Satellite_Conditional(nn.Module):
    def __init__(
        self,
        in_ch=3,
        num_classes=2,
        feature_dim=512
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.features = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(in_ch,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 16
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(128,affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 -> 8
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(
                256,
                affine=True
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 -> 4
            nn.Conv2d(256,feature_dim,kernel_size=4,stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(feature_dim,affine=True),nn.LeakyReLU(0.2, inplace=True))

        # unconditional critic score
        self.output_layer = nn.Linear(feature_dim,1)

        # projection label embedding
        self.label_embedding = nn.Embedding(num_classes,feature_dim)

    def forward(self, x, labels):
        """
        x:
            [B,3,64,64]

        labels:
            [B], dtype=torch.long

        return:
            [B]
        """

        labels = labels.long()

        # [B, feature_dim, 4, 4]
        feature_map = self.features(x)

        # global sum pooling
        # [B, feature_dim]
        feature = feature_map.sum(dim=(2, 3))

        # unconditional score
        unconditional_score = (self.output_layer(feature).squeeze(1))
        

        # label projection
        label_emb = self.label_embedding(labels)

        projection_score = torch.sum(feature * label_emb,dim=1)
        

        return (unconditional_score + projection_score)
    
class LatentAdapter(nn.Module):
    """Residual MLP adapter. Starts close to identity."""

    def __init__(self, z_dim=256, hidden_dim=512, residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return z + self.residual_scale * self.net(z)