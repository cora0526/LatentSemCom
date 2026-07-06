import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================
# VAE Model MNIST FashionMNIST
# =============================
class VAE_MNIST(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # [B,32,14,14]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # [B,64,7,7]
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 64 * 7 * 7)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # [B,32,14,14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # [B,1,28,28]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 64, 7, 7)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


# =====================
# Loss
# =====================
def vae_loss(x_hat, x, mu, logvar):
    recon_loss = F.mse_loss(x_hat, x, reduction="sum")

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + kl_loss, recon_loss, kl_loss


# ==============================
# JSCC Model MNIST FashionMNIST
# ==============================
class JSCC_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),

            nn.Conv2d(64, 2, 3, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


class RealMIMOChannel(nn.Module):
    def __init__(self, snr_db=20, random_snr=False, snr_min=0, snr_max=30):
        super().__init__()

        H = torch.tensor([
            [1.0, 0.15],
            [0.10, 9.0]
        ], dtype=torch.float32)

        H = H / torch.norm(H, p="fro") * np.sqrt(2)

        self.register_buffer("H_mat", H)

        self.snr_db = snr_db
        self.random_snr = random_snr
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self, x, snr_db=None):
        B, C, H, W = x.shape

        assert C % 2 == 0, "Channel dimension C must be even for 2x2 MIMO."

        if snr_db is None:
            if self.random_snr and self.training:
                snr_db = torch.empty(1, device=x.device).uniform_(
                    self.snr_min,
                    self.snr_max
                ).item()
            else:
                snr_db = self.snr_db

        x_perm = x.permute(0, 2, 3, 1)
        x_pair = x_perm.reshape(-1, 2)

        y_pair = x_pair @ self.H_mat.T

        signal_power = y_pair.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))

        noise = torch.randn_like(y_pair) * torch.sqrt(noise_power)

        y_pair = y_pair + noise

        y_perm = y_pair.reshape(B, H, W, C)
        y = y_perm.permute(0, 3, 1, 2)

        return y


class JSCC_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(2, 64, 4, 2, 1),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),

            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class JSCC_MIMO(nn.Module):
    def __init__(self, snr_db=20, random_snr=False, snr_min=0, snr_max=30):
        super().__init__()

        self.encoder = JSCC_Encoder()
        self.channel = RealMIMOChannel(
            snr_db=snr_db,
            random_snr=random_snr,
            snr_min=snr_min,
            snr_max=snr_max
        )
        self.decoder = JSCC_Decoder()

    def forward(self, x, snr_db=None):
        z = self.encoder(x)
        y = self.channel(z, snr_db=snr_db)
        x_hat = self.decoder(y)
        return x_hat
    
# ========================
# VAE Model for 64x64 RGB
# ========================
class VAE_RGB64(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),    # [B,32,32,32]
            nn.ReLU(),

            nn.Conv2d(32, 64, 4, 2, 1),   # [B,64,16,16]
            nn.ReLU(),

            nn.Conv2d(64, 128, 4, 2, 1),  # [B,128,8,8]
            nn.ReLU(),

            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 128 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # [B,64,16,16]
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # [B,32,32,32]
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # [B,3,64,64]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 128, 8, 8)
        x_hat = self.decoder(h)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss_RGB64(x_hat, x, mu, logvar, beta=0.001):
    recon_loss = F.mse_loss(x_hat, x, reduction="sum")

    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    loss = recon_loss + beta * kl_loss

    return loss, recon_loss, kl_loss

# ========================
# JSCC Model for 64x64 RGB
# ========================

class JSCC_Encoder_RGB64(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),    # [B,32,32,32]
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),   # [B,64,16,16]
            nn.ReLU(),

            nn.Conv2d(64, 2, 3, 1, 1)     # [B,2,16,16]
        )

    def forward(self, x):
        return self.net(x)


class RealMIMOChannel_RGB64(nn.Module):
    def __init__(self, snr_db=20, random_snr=False, snr_min=0, snr_max=30):
        super().__init__()

        H = torch.tensor([
            [1.0, 0.15],
            [0.10, 9.0]
        ], dtype=torch.float32)

        H = H / torch.norm(H, p="fro") * np.sqrt(2)

        self.register_buffer("H_mat", H)

        self.snr_db = snr_db
        self.random_snr = random_snr
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self, x, snr_db=None):
        B, C, H, W = x.shape

        assert C % 2 == 0, "Channel dimension C must be even for 2x2 MIMO."

        if snr_db is None:
            if self.random_snr and self.training:
                snr_db = torch.empty(1, device=x.device).uniform_(
                    self.snr_min,
                    self.snr_max
                ).item()
            else:
                snr_db = self.snr_db

        x_perm = x.permute(0, 2, 3, 1)
        x_pair = x_perm.reshape(-1, 2)

        y_pair = x_pair @ self.H_mat.T

        signal_power = y_pair.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))

        noise = torch.randn_like(y_pair) * torch.sqrt(noise_power)

        y_pair = y_pair + noise

        y_perm = y_pair.reshape(B, H, W, C)
        y = y_perm.permute(0, 3, 1, 2)

        return y


class JSCC_Decoder_RGB64(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(2, 64, 4, 2, 1),   # [B,64,32,32]
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # [B,32,64,64]
            nn.ReLU(),

            nn.Conv2d(32, 3, 3, 1, 1),            # [B,3,64,64]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class JSCC_MIMO_RGB64(nn.Module):
    def __init__(self, snr_db=20, random_snr=False, snr_min=0, snr_max=30):
        super().__init__()

        self.encoder = JSCC_Encoder_RGB64()

        self.channel = RealMIMOChannel(
            snr_db=snr_db,
            random_snr=random_snr,
            snr_min=snr_min,
            snr_max=snr_max
        )

        self.decoder = JSCC_Decoder_RGB64()

    def forward(self, x, snr_db=None):
        z = self.encoder(x)
        y = self.channel(z, snr_db=snr_db)
        x_hat = self.decoder(y)
        return x_hat