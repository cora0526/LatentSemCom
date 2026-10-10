"""
Reusable Shared Conditional VAE models for EuroSAT and DeepGlobe.

Dataset labels:
    0 = EuroSAT
    1 = DeepGlobe

Input image shape: [B, 3, 64, 64]
Latent shape:      [B, z_dim]
Image range:       approximately [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

Tensor = torch.Tensor
CheckpointLike = Union[str, Path, Mapping[str, Any]]


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """z = mu + std * epsilon, epsilon ~ N(0, I)."""
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """Mean KL(q(z|x) || N(0, I))."""
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean()


class SharedVAEEncoder(nn.Module):
    """Shared encoder for 64x64 RGB satellite images."""

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
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar


class ConditionalVAEDecoder(nn.Module):
    """Conditional decoder: label 0=EuroSAT, label 1=DeepGlobe."""

    def __init__(self, z_dim: int = 256, num_classes: int = 2, label_dim: int = 64) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.num_classes = int(num_classes)
        self.label_dim = int(label_dim)
        self.label_embedding = nn.Embedding(self.num_classes, self.label_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim + self.label_dim, 512 * 4 * 4),
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
        labels = labels.long()
        label_feature = self.label_embedding(labels)
        h = self.fc(torch.cat([z, label_feature], dim=1))
        h = h.view(z.size(0), 512, 4, 4)
        return self.decoder(h)


@dataclass
class VAEOutput:
    reconstruction: Tensor
    mu: Tensor
    logvar: Tensor
    z: Tensor


class SharedConditionalVAE(nn.Module):
    """Complete VAE wrapper supporting stochastic z or deterministic mu."""

    def __init__(self, z_dim: int = 256, num_classes: int = 2, label_dim: int = 64) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.num_classes = int(num_classes)
        self.label_dim = int(label_dim)
        self.encoder = SharedVAEEncoder(self.z_dim)
        self.decoder = ConditionalVAEDecoder(self.z_dim, self.num_classes, self.label_dim)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.encoder(x)

    def decode(self, z: Tensor, labels: Tensor) -> Tensor:
        return self.decoder(z, labels)

    def forward(self, x: Tensor, labels: Tensor, mode: str = "sample") -> VAEOutput:
        mu, logvar = self.encode(x)
        if mode == "sample":
            z = reparameterize(mu, logvar)
        elif mode == "mu":
            z = mu
        else:
            raise ValueError(f"Unknown mode={mode!r}; use 'sample' or 'mu'.")
        reconstruction = self.decode(z, labels)
        return VAEOutput(reconstruction, mu, logvar, z)


def set_requires_grad(model: nn.Module, value: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = value


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    set_requires_grad(model, False)
    return model


def _load_checkpoint_data(checkpoint: CheckpointLike, map_location=None) -> Mapping[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist:\n{path}")
        data = torch.load(path, map_location=map_location)
    else:
        data = checkpoint
    if not isinstance(data, Mapping):
        raise TypeError("Checkpoint must be a path or mapping.")
    return data


def load_shared_vae(
    checkpoint: CheckpointLike,
    device: Union[str, torch.device] = "cpu",
    z_dim: int = 256,
    num_classes: int = 2,
    label_dim: int = 64,
    strict: bool = True,
    freeze: bool = True,
) -> SharedConditionalVAE:
    """Load checkpoint containing 'encoder' and 'decoder'."""
    data = _load_checkpoint_data(checkpoint, map_location=device)
    if "encoder" not in data or "decoder" not in data:
        raise KeyError("Checkpoint must contain 'encoder' and 'decoder'.")
    checkpoint_z_dim = int(data.get("z_dim", z_dim))
    if checkpoint_z_dim != int(z_dim):
        raise ValueError(f"z_dim mismatch: checkpoint={checkpoint_z_dim}, requested={z_dim}")
    model = SharedConditionalVAE(z_dim, num_classes, label_dim).to(device)
    model.encoder.load_state_dict(data["encoder"], strict=strict)
    model.decoder.load_state_dict(data["decoder"], strict=strict)
    if freeze:
        freeze_model(model)
    return model


def load_encoder_decoder(
    checkpoint: CheckpointLike,
    device: Union[str, torch.device] = "cpu",
    z_dim: int = 256,
    num_classes: int = 2,
    label_dim: int = 64,
    strict: bool = True,
    freeze: bool = True,
) -> Tuple[SharedVAEEncoder, ConditionalVAEDecoder]:
    model = load_shared_vae(
        checkpoint=checkpoint,
        device=device,
        z_dim=z_dim,
        num_classes=num_classes,
        label_dim=label_dim,
        strict=strict,
        freeze=freeze,
    )
    return model.encoder, model.decoder


__all__ = [
    "SharedVAEEncoder",
    "ConditionalVAEDecoder",
    "SharedConditionalVAE",
    "VAEOutput",
    "reparameterize",
    "kl_divergence",
    "set_requires_grad",
    "freeze_model",
    "load_shared_vae",
    "load_encoder_decoder",
]
