"""
Reusable VAE-GAN model definitions for EuroSAT and DeepGlobe.

This module matches checkpoints produced by the training script that saves:

    ./save_model/vae_gan_from_scratch_best.pth
    ./save_model/vae_gan_from_scratch_latest.pth

Dataset labels:
    0 = EuroSAT
    1 = DeepGlobe

Expected image shape:
    [B, 3, 64, 64]

Expected image range:
    [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn


Tensor = torch.Tensor
CheckpointLike = Union[str, Path, Mapping[str, Any]]


# ============================================================
# Basic utilities
# ============================================================

def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Reparameterization trick:

        z = mu + exp(0.5 * logvar) * epsilon
        epsilon ~ N(0, I)
    """
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return mu + std * epsilon


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    KL(q(z|x) || N(0, I)).

    Returns the mean KL divergence over the batch and latent dimensions.
    """
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


def set_requires_grad(model: nn.Module, value: bool) -> None:
    """Enable or disable gradients for all parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = value


def freeze_model(model: nn.Module) -> nn.Module:
    """Switch to evaluation mode and disable gradients."""
    model.eval()
    set_requires_grad(model, False)
    return model


# ============================================================
# VAE encoder
# ============================================================

class SharedVAEEncoder(nn.Module):
    """
    Shared VAE encoder for 64x64 RGB satellite images.

    Input:
        x: [B, 3, 64, 64]

    Output:
        mu:     [B, z_dim]
        logvar: [B, z_dim]
    """

    def __init__(self, z_dim: int = 256) -> None:
        super().__init__()

        self.z_dim = int(z_dim)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        feature_dim = 512 * 4 * 4
        self.fc_mu = nn.Linear(feature_dim, self.z_dim)
        self.fc_logvar = nn.Linear(feature_dim, self.z_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.features(x).flatten(start_dim=1)
        mu = self.fc_mu(h)
        logvar = torch.clamp(
            self.fc_logvar(h),
            min=-10.0,
            max=10.0,
        )
        return mu, logvar


# ============================================================
# Conditional generator
# ============================================================

class ConditionalGenerator(nn.Module):
    """
    Conditional generator used as the VAE-GAN decoder.

    Labels:
        0 = EuroSAT
        1 = DeepGlobe

    Input:
        z:      [B, z_dim]
        labels: [B]

    Output:
        image: [B, 3, 64, 64], approximately in [-1, 1]
    """

    def __init__(
        self,
        z_dim: int = 256,
        num_classes: int = 2,
        label_dim: int = 64,
    ) -> None:
        super().__init__()

        self.z_dim = int(z_dim)
        self.num_classes = int(num_classes)
        self.label_dim = int(label_dim)

        self.label_embedding = nn.Embedding(
            num_embeddings=self.num_classes,
            embedding_dim=self.label_dim,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + self.label_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: Tensor, labels: Tensor) -> Tensor:
        if labels.dtype != torch.long:
            labels = labels.long()

        label_feature = self.label_embedding(labels)
        h = self.fc(torch.cat([z, label_feature], dim=1))
        h = h.view(z.size(0), 512, 4, 4)
        return self.decoder(h)


# ============================================================
# Conditional critic
# ============================================================

class ConditionalCritic(nn.Module):
    """
    Projection-style conditional critic.

    Input:
        x:      [B, 3, 64, 64]
        labels: [B]

    Output:
        score: [B]

    When return_features=True:
        returns (score, feature)
    """

    def __init__(
        self,
        num_classes: int = 2,
        feature_dim: int = 512,
    ) -> None:
        super().__init__()

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, self.feature_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.score_head = nn.Linear(self.feature_dim, 1)
        self.label_embedding = nn.Embedding(
            self.num_classes,
            self.feature_dim,
        )

    def forward(
        self,
        x: Tensor,
        labels: Tensor,
        return_features: bool = False,
    ):
        if labels.dtype != torch.long:
            labels = labels.long()

        feature_map = self.features(x)
        feature = feature_map.sum(dim=(2, 3))

        score = self.score_head(feature).squeeze(1)
        projection = (
            feature * self.label_embedding(labels)
        ).sum(dim=1)
        score = score + projection

        if return_features:
            return score, feature

        return score


# ============================================================
# Full VAE-GAN wrapper
# ============================================================

@dataclass
class VAEGANOutput:
    reconstruction: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor


class ConditionalVAEGAN(nn.Module):
    """
    Full VAE-GAN wrapper.

    mode="sample":
        Uses reparameterized latent z.

    mode="mu":
        Uses deterministic encoder mean mu.
        This is the mode used by the DDPM/MLP separator pipeline.
    """

    def __init__(
        self,
        z_dim: int = 256,
        num_classes: int = 2,
        label_dim: int = 64,
        critic_feature_dim: int = 512,
        include_critic: bool = True,
    ) -> None:
        super().__init__()

        self.z_dim = int(z_dim)
        self.num_classes = int(num_classes)
        self.label_dim = int(label_dim)
        self.critic_feature_dim = int(critic_feature_dim)

        self.encoder = SharedVAEEncoder(z_dim=self.z_dim)

        self.generator = ConditionalGenerator(
            z_dim=self.z_dim,
            num_classes=self.num_classes,
            label_dim=self.label_dim,
        )

        self.critic: Optional[ConditionalCritic]
        if include_critic:
            self.critic = ConditionalCritic(
                num_classes=self.num_classes,
                feature_dim=self.critic_feature_dim,
            )
        else:
            self.critic = None

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.encoder(x)

    def generate(self, z: Tensor, labels: Tensor) -> Tensor:
        return self.generator(z, labels)

    def discriminate(
        self,
        x: Tensor,
        labels: Tensor,
        return_features: bool = False,
    ):
        if self.critic is None:
            raise RuntimeError(
                "This ConditionalVAEGAN was created without a critic."
            )

        return self.critic(
            x,
            labels,
            return_features=return_features,
        )

    def forward(
        self,
        x: Tensor,
        labels: Tensor,
        mode: str = "sample",
    ) -> VAEGANOutput:
        mu, logvar = self.encode(x)

        if mode == "sample":
            z = reparameterize(mu, logvar)
        elif mode == "mu":
            z = mu
        else:
            raise ValueError(
                f"Unknown mode={mode!r}. Expected 'sample' or 'mu'."
            )

        reconstruction = self.generate(z, labels)

        return VAEGANOutput(
            reconstruction=reconstruction,
            mu=mu,
            logvar=logvar,
            z=z,
        )


# ============================================================
# Checkpoint loading
# ============================================================

def _load_checkpoint_data(
    checkpoint: CheckpointLike,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Mapping[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        checkpoint_path = Path(checkpoint)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist:\n{checkpoint_path}"
            )

        checkpoint_data = torch.load(
            checkpoint_path,
            map_location=map_location,
        )
    else:
        checkpoint_data = checkpoint

    if not isinstance(checkpoint_data, Mapping):
        raise TypeError(
            "Checkpoint must be a path or a mapping."
        )

    return checkpoint_data


def _read_checkpoint_config(
    checkpoint_data: Mapping[str, Any],
) -> Mapping[str, Any]:
    config = checkpoint_data.get("config", {})
    return config if isinstance(config, Mapping) else {}


def load_vaegan(
    checkpoint: CheckpointLike,
    device: Union[str, torch.device] = "cpu",
    z_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
    label_dim: Optional[int] = None,
    critic_feature_dim: Optional[int] = None,
    include_critic: bool = True,
    strict: bool = True,
    freeze: bool = True,
) -> ConditionalVAEGAN:
    """
    Load the complete VAE-GAN checkpoint.

    Expected checkpoint keys:
        encoder
        generator
        critic

    Architecture values are automatically read from checkpoint["config"]
    when present.
    """
    checkpoint_data = _load_checkpoint_data(
        checkpoint,
        map_location=device,
    )

    config = _read_checkpoint_config(checkpoint_data)

    resolved_z_dim = int(
        z_dim if z_dim is not None
        else config.get("z_dim", 256)
    )
    resolved_num_classes = int(
        num_classes if num_classes is not None
        else config.get("num_classes", 2)
    )
    resolved_label_dim = int(
        label_dim if label_dim is not None
        else config.get("label_dim", 64)
    )
    resolved_critic_feature_dim = int(
        critic_feature_dim if critic_feature_dim is not None
        else config.get("critic_feature_dim", 512)
    )

    if "encoder" not in checkpoint_data:
        raise KeyError("Checkpoint does not contain 'encoder'.")

    if "generator" not in checkpoint_data:
        raise KeyError("Checkpoint does not contain 'generator'.")

    model = ConditionalVAEGAN(
        z_dim=resolved_z_dim,
        num_classes=resolved_num_classes,
        label_dim=resolved_label_dim,
        critic_feature_dim=resolved_critic_feature_dim,
        include_critic=include_critic,
    ).to(device)

    model.encoder.load_state_dict(
        checkpoint_data["encoder"],
        strict=strict,
    )
    model.generator.load_state_dict(
        checkpoint_data["generator"],
        strict=strict,
    )

    if include_critic:
        if "critic" not in checkpoint_data:
            raise KeyError(
                "include_critic=True, but checkpoint does not contain 'critic'."
            )

        assert model.critic is not None
        model.critic.load_state_dict(
            checkpoint_data["critic"],
            strict=strict,
        )

    if freeze:
        freeze_model(model)
    else:
        model.train()

    return model


def load_encoder_generator(
    checkpoint: CheckpointLike,
    device: Union[str, torch.device] = "cpu",
    z_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
    label_dim: Optional[int] = None,
    strict: bool = True,
    freeze: bool = True,
) -> Tuple[SharedVAEEncoder, ConditionalGenerator]:
    """
    Load only the encoder and generator.

    Recommended for DDPM/MLP separator training and evaluation.
    """
    model = load_vaegan(
        checkpoint=checkpoint,
        device=device,
        z_dim=z_dim,
        num_classes=num_classes,
        label_dim=label_dim,
        include_critic=False,
        strict=strict,
        freeze=freeze,
    )

    return model.encoder, model.generator


def load_encoder_generator_critic(
    checkpoint: CheckpointLike,
    device: Union[str, torch.device] = "cpu",
    z_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
    label_dim: Optional[int] = None,
    critic_feature_dim: Optional[int] = None,
    strict: bool = True,
    freeze: bool = True,
) -> Tuple[
    SharedVAEEncoder,
    ConditionalGenerator,
    ConditionalCritic,
]:
    """
    Load encoder, generator, and critic as separate objects.
    """
    model = load_vaegan(
        checkpoint=checkpoint,
        device=device,
        z_dim=z_dim,
        num_classes=num_classes,
        label_dim=label_dim,
        critic_feature_dim=critic_feature_dim,
        include_critic=True,
        strict=strict,
        freeze=freeze,
    )

    assert model.critic is not None

    return model.encoder, model.generator, model.critic


__all__ = [
    "SharedVAEEncoder",
    "ConditionalGenerator",
    "ConditionalCritic",
    "ConditionalVAEGAN",
    "VAEGANOutput",
    "reparameterize",
    "kl_divergence",
    "set_requires_grad",
    "freeze_model",
    "load_vaegan",
    "load_encoder_generator",
    "load_encoder_generator_critic",
]
