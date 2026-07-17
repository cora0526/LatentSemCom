import torch
import glob
from PIL import Image
import os
import glob

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset,ConcatDataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from model import Generator_Satellite_Conditional, Critic_Satellite_Conditional


# ============================================================
# DeepGlobe dataset
# domain label:
#   1 = DeepGlobe
# ============================================================
class SatelliteDataset(Dataset):
    def __init__(self,root,transform=None,domain_label=1):
        self.files = sorted(glob.glob(os.path.join(root, "*_sat_*.jpg")))

        self.transform = transform
        self.domain_label = int(domain_label)

        if len(self.files) == 0:
            raise RuntimeError(f"No matching DeepGlobe images found in:\n{root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(
            self.files[idx]
        ).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, self.domain_label


# ============================================================
# Override original dataset label with domain label
#
# EuroSAT:
#   original ImageFolder label = land-cover class
#   target domain label = 0
# ============================================================
class DomainLabelDataset(Dataset):
    def __init__(self,dataset,domain_label,):
        self.dataset = dataset
        self.domain_label = int(domain_label)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if isinstance(item, (tuple, list)):
            img = item[0]
        else:
            img = item

        return img, self.domain_label


# ============================================================
# Conditional WGAN-GP gradient penalty
# ============================================================
def gradient_penalty_conditional(critic, real, fake, labels, device,):
    batch_size = real.size(0)

    epsilon = torch.rand(batch_size,1,1,1,device=device,)

    interpolated = ( epsilon * real + (1.0 - epsilon) * fake)

    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated,labels)

    gradients = torch.autograd.grad(outputs=mixed_scores,inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.reshape(batch_size,-1,)

    gradient_norm = gradients.norm(2,dim=1)

    gp = ((gradient_norm - 1.0) ** 2).mean()

    return gp


# ============================================================
# Main
# ============================================================
def main():
    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------
    epochs = 200
    batch_size = 128

    z_dim = 256
    label_dim = 64
    num_classes = 2

    lr = 5e-5
    n_critic = 5
    lambda_gp = 10.0

    device = "cuda"

    num_workers = 4

    # --------------------------------------------------------
    # Domain definition
    # --------------------------------------------------------
    EURO_SAT_LABEL = 0
    DEEP_GLOBE_LABEL = 1

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    data_root_e = (
        r"D:/Python/SemCom/saved_data/"
        r"EuroSAT/EuroSAT"
    )

    data_root_g = (
        r"D:/Python/SemCom/saved_data/"
        r"deepglobe_64/train"
    )

    save_root = ("./save_data/"
        "GANSatellite_conditional")

    model_root = "./save_model"

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(model_root, exist_ok=True)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    seed = 42

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------
    tfm = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    # ========================================================
    # EuroSAT dataset
    # Original ImageFolder labels are discarded.
    # Every EuroSAT image receives domain label 0.
    # ========================================================
    base_ds_e = datasets.ImageFolder(root=data_root_e,transform=tfm,)

    base_ds_e = Subset(base_ds_e,range(min(50000,len(base_ds_e))))

    ds_e = DomainLabelDataset(base_ds_e,domain_label=EURO_SAT_LABEL)

    # ========================================================
    # DeepGlobe dataset
    # Every DeepGlobe image receives domain label 1.
    # ========================================================
    base_ds_g = SatelliteDataset(root=data_root_g,transform=tfm,domain_label=DEEP_GLOBE_LABEL)

    base_ds_g = Subset(base_ds_g,range(min(50000,len(base_ds_g))))

    ds_g = base_ds_g

    # ========================================================
    # Merge two domains
    # ========================================================
    ds = ConcatDataset([ds_e,ds_g])

    dl = DataLoader(ds,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True,persistent_workers=(num_workers > 0),drop_last=True)

    print(f"EuroSAT images  = {len(ds_e)}")

    print(f"DeepGlobe images = {len(ds_g)}")

    print(f"Total images     = {len(ds)}")

    # ========================================================
    # Models
    # ========================================================
    G = Generator_Satellite_Conditional(z_dim=z_dim,num_classes=num_classes,label_dim=label_dim,out_ch=3).to(device)

    D = Critic_Satellite_Conditional(in_ch=3,num_classes=num_classes,feature_dim=512).to(device)

    print("G device =",next(G.parameters()).device)

    print("D device =",next(D.parameters()).device)

    # ========================================================
    # Optimizers
    # ========================================================
    optG = torch.optim.Adam(G.parameters(),lr=lr,betas=(0.0, 0.9))

    optD = torch.optim.Adam(D.parameters(),lr=lr,betas=(0.0, 0.9))

    # ========================================================
    # Fixed latent and labels
    # ========================================================
    fixed_z = torch.randn(64,z_dim,device=device)

    fixed_label_e = torch.full((64,),EURO_SAT_LABEL,dtype=torch.long,device=device)

    fixed_label_g = torch.full((64,),DEEP_GLOBE_LABEL,dtype=torch.long,device=device)

    # ========================================================
    # Save one real mixed batch
    # ========================================================
    real_sample, real_labels = next(iter(dl))

    real_sample = real_sample.to(device)

    save_image((real_sample[:64] + 1.0) / 2.0,os.path.join(save_root,"real_mixed_batch.png"),nrow=8)

    # ========================================================
    # Save separate real-domain samples
    # ========================================================
    real_e_sample = torch.stack([ds_e[i][0]for i in range(min(64, len(ds_e)))])

    real_g_sample = torch.stack([ds_g[i][0]for i in range(min(64, len(ds_g)))])

    save_image((real_e_sample + 1.0) / 2.0, os.path.join(save_root,"real_EuroSAT.png"),nrow=8)

    save_image((real_g_sample + 1.0) / 2.0, os.path.join(save_root,"real_DeepGlobe.png"),nrow=8)

    # ========================================================
    # Training
    # ========================================================
    step = 0

    for ep in range(epochs):
        G.train()
        D.train()

        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        num_batches = 0

        for real, labels in dl:
            real = real.to(device,non_blocking=True)

            labels = labels.to(device,dtype=torch.long,non_blocking=True)

            batch_now = real.size(0)

            # =================================================
            # Train Critic
            # =================================================
            for _ in range(n_critic):
                z = torch.randn(batch_now,z_dim,device=device)

                with torch.no_grad():
                    fake = G(z,labels)

                d_real = D(real,labels)

                d_fake = D(fake,labels)

                gp = gradient_penalty_conditional(critic=D,real=real,fake=fake,labels=labels,device=device)

                loss_D = (d_fake.mean() - d_real.mean() + lambda_gp * gp)

                optD.zero_grad(set_to_none=True)

                loss_D.backward()

                optD.step()

            # =================================================
            # Train Generator
            # =================================================
            z = torch.randn(batch_now,z_dim,device=device)

            fake = G(z,labels)

            loss_G = -D(fake,labels).mean()

            optG.zero_grad(set_to_none=True)

            loss_G.backward()

            optG.step()

            epoch_loss_d += loss_D.item()
            epoch_loss_g += loss_G.item()
            num_batches += 1

            if step % 200 == 0:
                euro_count = int((labels == EURO_SAT_LABEL).sum().item())

                deep_count = int((labels == DEEP_GLOBE_LABEL).sum().item())

                print(
                    f"ep {ep + 1}/{epochs} "
                    f"step {step} | "
                    f"lossD {loss_D.item():.3f} | "
                    f"lossG {loss_G.item():.3f} | "
                    f"Dreal {d_real.mean().item():.3f} | "
                    f"Dfake {d_fake.mean().item():.3f} | "
                    f"GP {gp.item():.3f} | "
                    f"E={euro_count}, G={deep_count}"
                )

            step += 1

        print(
            f"Epoch {ep + 1}/{epochs} | "
            f"avg lossD "
            f"{epoch_loss_d / num_batches:.4f} | "
            f"avg lossG "
            f"{epoch_loss_g / num_batches:.4f}"
        )

        # ====================================================
        # Save generated samples
        # ====================================================
        if (ep + 1) % 10 == 0:
            G.eval()

            with torch.no_grad():
                samples_e = G(fixed_z,fixed_label_e)

                samples_g = G(fixed_z,fixed_label_g)

                samples_e = (samples_e + 1.0) / 2.0

                samples_g = (samples_g + 1.0) / 2.0

                save_image(samples_e,os.path.join(save_root,("samples_EuroSAT_"f"ep{ep + 1}.png")),nrow=8)

                save_image(samples_g,os.path.join(save_root,("samples_DeepGlobe_"f"ep{ep + 1}.png"),),nrow=8)

            # Save checkpoint every 10 epochs
            torch.save(
                {
                    "epoch": ep + 1,
                    "generator": G.state_dict(),
                    "critic": D.state_dict(),
                    "optimizer_g": optG.state_dict(),
                    "optimizer_d": optD.state_dict(),
                    "z_dim": z_dim,
                    "label_dim": label_dim,
                    "num_classes": num_classes,
                },
                os.path.join(
                    model_root,
                    (
                        "conditional_wgangp_"
                        f"satellite_ep{ep + 1}.pth"
                    ),
                ),
            )

    # ========================================================
    # Save final models
    # ========================================================
    torch.save(G.state_dict(),os.path.join(model_root,"G_wgangp_satellite_conditional_C.pth",))

    torch.save(D.state_dict(),os.path.join(model_root,"D_wgangp_satellite_conditional_C.pth",))

    print("Saved conditional Generator and Critic.")


if __name__ == "__main__":
    main()