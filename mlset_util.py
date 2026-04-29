import torch
from pytorch_msssim import ssim
from model import Ganerator_MNIST, VAE_Encoder_MNIST, LabelCondLatentDenoiser

def import_all_model (
        z_dim=64, 
        device="cuda", 
        G_path = "./save_model/G_wgangp_C.pth",
        E_path = "./save_model/E_invert_to_G_C_only.pth",
        E_f_path = "./save_model/E_invert_to_G_F.pth"):
    G = Ganerator_MNIST(z_dim=z_dim).to(device)
    G.load_state_dict(torch.load(G_path))
    G.eval()
    for p in G.parameters():
        p.requires_grad = False

    E = VAE_Encoder_MNIST(latent_dim=z_dim).to(device)
    E.load_state_dict(torch.load(E_path))
    E.eval()
    for p in E.parameters():
        p.requires_grad = False

    E_f = VAE_Encoder_MNIST(latent_dim=z_dim).to(device)
    E_f.load_state_dict(torch.load(E_f_path))
    E_f.eval()
    for p in E_f.parameters():
        p.requires_grad = False

    # DM = LabelCondLatentDenoiser(num_classes=2, z_dim=z_dim, tdim=128, ydim=64).to(device)
    # DM.load_state_dict(torch.load(DM_path))
    # DM.eval()
    # for p in DM.parameters():
    #     p.requires_grad = False

    return G,E,E_f

def mse(img1, img2):
    return torch.mean((img1 - img2) ** 2)

def psnr(img1, img2):
    mse_val = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse_val))

def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=1.0)

if __name__ == "__main__":
    
    G,E,E_f,DM = import_all_model()