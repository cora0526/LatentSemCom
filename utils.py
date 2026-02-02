import torch
import torch.nn.functional as F

def gradient_penalty(discriminator, real_data, fake_data, device='cpu'):
    batch_size = real_data.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    x_hat = eps * real_data + (1 - eps) * fake_data
    x_hat.requires_grad_(True)

    d_hat = discriminator(x_hat)
    gradients = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_patency = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_patency

def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    # x, x_hat: (B,1,28,28)
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")  # 或 "mean"
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2, device="cuda"):
    return torch.linspace(beta_start, beta_end, T, device=device)

@torch.no_grad()
def denoise_latent_ddpm(DM, zT, betas, alphas, alpha_bars,t_start=None):
    """
    zT: (B, z_dim)  noisy latent at time T-1
    return z0_hat: (B, z_dim)
    """
    device = zT.device
    T = betas.size(0)
    if t_start is None:
        t_start = T - 1

    sqrt_ab = torch.sqrt(alpha_bars)              # sqrt(alpha_bar_t)
    sqrt_1mab = torch.sqrt(1 - alpha_bars)        # sqrt(1-alpha_bar_t)

    z = zT
    for t in reversed(range(t_start + 1)):
        tt = torch.full((z.size(0),), t, device=device, dtype=torch.long)
        eps = DM(z, tt)

        # predict x0 (here x0 is latent z0)
        z0_hat = (z - sqrt_1mab[t] * eps) / (sqrt_ab[t] + 1e-12)

        if t == 0:
            z = z0_hat
        else:
            # deterministic update to t-1 (DDIM eta=0)
            z = sqrt_ab[t - 1] * z0_hat + sqrt_1mab[t - 1] * eps

    return z

def eps_from_Ef(f_iter, E_f, device):
    x_f, _ = next(f_iter)
    x_f = x_f.to(device)
    with torch.no_grad():
        mu_f, _ = E_f(x_f)          # (batch_size, z_dim)
        eps = mu_f

    # 建議：白化成近似 N(0,1)，讓訓練更穩
    eps = eps - eps.mean(dim=0, keepdim=True)
    eps = eps / (eps.std(dim=0, keepdim=True) + 1e-6)
    return eps