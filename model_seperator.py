"""
Reusable latent-separator models for the four satellite systems.

File name:
    model_seperator.py

Included separators:
    1. VAE-DDPM-VAE      -> VAEDDPMVAESeparator
    2. VAE-MLP-VAE       -> VAEMLPVAESeparator
    3. VAE-DDPM-GAN      -> VAEDDPMGANSeparator
    4. VAE-MLP-GAN       -> VAEMLPGANSeparator

The two DDPM systems use the same network architecture but different checkpoints.
The two MLP systems also use the same architecture but different checkpoints.

Default checkpoints:
    ./save_model/DDPM_vae_latent_separator_latest.pth
    ./save_model/MLP_vae_latent_separator_best.pth
    ./save_model/DDPM_vaegan_scratch_latent_separator_best.pth
    ./save_model/MLP_vaegan_scratch_latent_separator_best.pth
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
CheckpointLike = Union[str, Path, Mapping[str, Any]]


# ============================================================
# Shared utilities
# ============================================================

def set_requires_grad(model: nn.Module, value: bool) -> None:
    """Enable or disable gradients for all parameters."""
    for parameter in model.parameters():
        parameter.requires_grad = value


def freeze_model(model: nn.Module) -> nn.Module:
    """Switch a model to eval mode and disable gradients."""
    model.eval()
    set_requires_grad(model, False)
    return model


def _load_checkpoint(
    checkpoint: CheckpointLike,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Mapping[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        checkpoint_path = Path(checkpoint)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist:\n{checkpoint_path}"
            )

        data = torch.load(
            checkpoint_path,
            map_location=map_location,
        )
    else:
        data = checkpoint

    if not isinstance(data, Mapping):
        raise TypeError(
            "Checkpoint must be a file path or a mapping."
        )

    return data


def _extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor]:
    """
    Supported checkpoint forms:

        {"model": state_dict, ...}
        {"state_dict": state_dict, ...}
        state_dict
    """
    if "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, Mapping):
        raise TypeError("Could not find a valid model state dictionary.")

    return state


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    config = checkpoint.get("config", {})
    return config if isinstance(config, Mapping) else {}


# ============================================================
# Shared MLP blocks
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding used by the latent DDPM."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)

        frequencies = torch.exp(
            torch.arange(
                half,
                device=timesteps.device,
                dtype=torch.float32,
            )
            * -scale
        )

        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat(
            [angles.sin(), angles.cos()],
            dim=1,
        )

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualMLPBlock(nn.Module):
    """Residual fully connected block shared by DDPM and MLP separators."""

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


# ============================================================
# DDPM separator architecture
# ============================================================

class LatentConditionalDDPM(nn.Module):
    """
    Conditional latent DDPM denoiser.

    Input:
        z_t:       [B, z_dim]
        timesteps: [B]
        labels:    [B]
        cond:      [B, z_dim]

    Output:
        predicted epsilon: [B, z_dim]

    Labels:
        0 = EuroSAT
        1 = DeepGlobe
    """

    def __init__(
        self,
        z_dim: int = 256,
        num_classes: int = 2,
        time_dim: int = 128,
        label_dim: int = 64,
        hidden_dim: int = 768,
        num_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.z_dim = int(z_dim)
        self.num_classes = int(num_classes)
        self.time_dim = int(time_dim)
        self.label_dim = int(label_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.label_embedding = nn.Embedding(
            self.num_classes,
            self.label_dim,
        )

        input_dim = (
            self.z_dim
            + self.z_dim
            + self.time_dim
            + self.label_dim
        )

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    self.hidden_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.z_dim),
        )

    def forward(
        self,
        z_t: Tensor,
        timesteps: Tensor,
        labels: Tensor,
        cond: Tensor,
    ) -> Tensor:
        if labels.dtype != torch.long:
            labels = labels.long()

        t_embedding = self.time_embedding(timesteps)
        label_embedding = self.label_embedding(labels)

        hidden = torch.cat(
            [z_t, cond, t_embedding, label_embedding],
            dim=1,
        )

        hidden = self.input_layer(hidden)
        hidden = self.blocks(hidden)
        return self.output_layer(hidden)


class VAEDDPMVAESeparator(LatentConditionalDDPM):
    """
    DDPM separator used by:

        Shared VAE encoder
            -> DDPM separator
            -> Conditional VAE decoder
    """


class VAEDDPMGANSeparator(LatentConditionalDDPM):
    """
    DDPM separator used by:

        Scratch VAE-GAN encoder
            -> DDPM separator
            -> Conditional GAN generator
    """


# ============================================================
# Joint MLP separator architecture
# ============================================================

class JointLatentMLPSeparator(nn.Module):
    """
    Predict both source latent vectors from one received latent mixture.

    Input:
        z_rx: [B, z_dim]

    Output:
        pred_e: [B, z_dim]
        pred_d: [B, z_dim]
    """

    def __init__(
        self,
        z_dim: int = 256,
        hidden_dim: int = 768,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.z_dim = int(z_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.dropout = float(dropout)

        self.input_layer = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.SiLU(),
        )

        self.blocks = nn.Sequential(
            *[
                ResidualMLPBlock(
                    self.hidden_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.z_dim * 2),
        )

    def forward(self, z_rx: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = self.input_layer(z_rx)
        hidden = self.blocks(hidden)
        pair = self.output_layer(hidden)

        pred_e = pair[:, : self.z_dim]
        pred_d = pair[:, self.z_dim :]
        return pred_e, pred_d


class VAEMLPVAESeparator(JointLatentMLPSeparator):
    """
    MLP separator used by:

        Shared VAE encoder
            -> Joint MLP separator
            -> Conditional VAE decoder
    """


class VAEMLPGANSeparator(JointLatentMLPSeparator):
    """
    MLP separator used by:

        Scratch VAE-GAN encoder
            -> Joint MLP separator
            -> Conditional GAN generator
    """


# ============================================================
# Diffusion schedule
# ============================================================

class DiffusionSchedule(nn.Module):
    """Linear-beta diffusion schedule used by the latent DDPM."""

    def __init__(
        self,
        timesteps: int = 400,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()

        self.timesteps = int(timesteps)

        betas = torch.linspace(
            beta_start,
            beta_end,
            self.timesteps,
            dtype=torch.float32,
        )

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(
            alpha_bars[:-1],
            (1, 0),
            value=1.0,
        )

        posterior_variance = (
            betas
            * (1.0 - alpha_bars_prev)
            / (1.0 - alpha_bars)
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)

        self.register_buffer(
            "sqrt_recip_alpha_bars",
            torch.sqrt(1.0 / alpha_bars),
        )

        self.register_buffer(
            "sqrt_recipm1_alpha_bars",
            torch.sqrt(1.0 / alpha_bars - 1.0),
        )

        self.register_buffer(
            "posterior_variance",
            torch.clamp(
                posterior_variance,
                min=1e-20,
            ),
        )

    @staticmethod
    def extract(
        values: Tensor,
        timesteps: Tensor,
        x_shape,
    ) -> Tensor:
        out = values.gather(0, timesteps)
        return out.reshape(
            timesteps.size(0),
            *(1,) * (len(x_shape) - 1),
        )

    def q_sample(
        self,
        x0: Tensor,
        timesteps: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward diffusion q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = torch.sqrt(
            self.extract(
                self.alpha_bars,
                timesteps,
                x0.shape,
            )
        )

        sqrt_one_minus_alpha_bar = torch.sqrt(
            1.0
            - self.extract(
                self.alpha_bars,
                timesteps,
                x0.shape,
            )
        )

        x_t = (
            sqrt_alpha_bar * x0
            + sqrt_one_minus_alpha_bar * noise
        )

        return x_t, noise

    def predict_x0_from_eps(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        eps: Tensor,
    ) -> Tensor:
        reciprocal = self.extract(
            self.sqrt_recip_alpha_bars,
            timesteps,
            x_t.shape,
        )

        reciprocal_minus_one = self.extract(
            self.sqrt_recipm1_alpha_bars,
            timesteps,
            x_t.shape,
        )

        return (
            reciprocal * x_t
            - reciprocal_minus_one * eps
        )


# ============================================================
# DDPM/DDIM sampling
# ============================================================

@torch.no_grad()
def sample_ddpm(
    model: LatentConditionalDDPM,
    schedule: DiffusionSchedule,
    cond: Tensor,
    labels: Tensor,
    seed: int = 42,
    sampling_method: str = "ddim",
    ddim_steps: int = 50,
    ddim_eta: float = 0.0,
    clip_x0: Optional[float] = None,
) -> Tensor:
    """
    Sample one source latent from a received mixture.

    Recommended:
        sampling_method="ddim"
        ddim_steps=50
        ddim_eta=0.0

    For EuroSAT:
        labels = zeros

    For DeepGlobe:
        labels = ones
    """
    model.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    batch_size = cond.size(0)
    z_dim = cond.size(1)

    x = torch.randn(
        batch_size,
        z_dim,
        device=cond.device,
        dtype=cond.dtype,
    )

    method = sampling_method.lower()

    if method == "ddpm":
        for step in reversed(range(schedule.timesteps)):
            timesteps = torch.full(
                (batch_size,),
                step,
                dtype=torch.long,
                device=cond.device,
            )

            predicted_eps = model(
                x,
                timesteps,
                labels,
                cond,
            )

            predicted_x0 = schedule.predict_x0_from_eps(
                x,
                timesteps,
                predicted_eps,
            )

            if clip_x0 is not None:
                predicted_x0 = predicted_x0.clamp(
                    -clip_x0,
                    clip_x0,
                )

            beta_t = schedule.extract(
                schedule.betas,
                timesteps,
                x.shape,
            )
            alpha_t = schedule.extract(
                schedule.alphas,
                timesteps,
                x.shape,
            )
            alpha_bar_t = schedule.extract(
                schedule.alpha_bars,
                timesteps,
                x.shape,
            )
            alpha_bar_prev = schedule.extract(
                schedule.alpha_bars_prev,
                timesteps,
                x.shape,
            )

            coefficient_x0 = (
                beta_t
                * torch.sqrt(alpha_bar_prev)
                / (1.0 - alpha_bar_t)
            )

            coefficient_xt = (
                (1.0 - alpha_bar_prev)
                * torch.sqrt(alpha_t)
                / (1.0 - alpha_bar_t)
            )

            mean = (
                coefficient_x0 * predicted_x0
                + coefficient_xt * x
            )

            if step > 0:
                variance = schedule.extract(
                    schedule.posterior_variance,
                    timesteps,
                    x.shape,
                )
                x = (
                    mean
                    + torch.sqrt(variance)
                    * torch.randn_like(x)
                )
            else:
                x = mean

        return x

    if method != "ddim":
        raise ValueError(
            f"Unknown sampling_method={sampling_method!r}. "
            "Expected 'ddpm' or 'ddim'."
        )

    steps = max(
        1,
        min(int(ddim_steps), schedule.timesteps),
    )

    time_sequence = torch.linspace(
        schedule.timesteps - 1,
        0,
        steps,
        device=cond.device,
    ).round().long()

    time_sequence = torch.unique_consecutive(time_sequence)

    for index, step_tensor in enumerate(time_sequence):
        step = int(step_tensor.item())

        timesteps = torch.full(
            (batch_size,),
            step,
            dtype=torch.long,
            device=cond.device,
        )

        predicted_eps = model(
            x,
            timesteps,
            labels,
            cond,
        )

        predicted_x0 = schedule.predict_x0_from_eps(
            x,
            timesteps,
            predicted_eps,
        )

        if clip_x0 is not None:
            predicted_x0 = predicted_x0.clamp(
                -clip_x0,
                clip_x0,
            )

        alpha_bar_t = schedule.extract(
            schedule.alpha_bars,
            timesteps,
            x.shape,
        )

        if index == len(time_sequence) - 1:
            x = predicted_x0
            continue

        previous_step = int(
            time_sequence[index + 1].item()
        )

        previous_timesteps = torch.full(
            (batch_size,),
            previous_step,
            dtype=torch.long,
            device=cond.device,
        )

        alpha_bar_previous = schedule.extract(
            schedule.alpha_bars,
            previous_timesteps,
            x.shape,
        )

        sigma = float(ddim_eta) * torch.sqrt(
            torch.clamp(
                (1.0 - alpha_bar_previous)
                / (1.0 - alpha_bar_t)
                * (
                    1.0
                    - alpha_bar_t
                    / alpha_bar_previous
                ),
                min=0.0,
            )
        )

        direction = torch.sqrt(
            torch.clamp(
                1.0
                - alpha_bar_previous
                - sigma.pow(2),
                min=0.0,
            )
        ) * predicted_eps

        if float(ddim_eta) > 0.0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (
            torch.sqrt(alpha_bar_previous)
            * predicted_x0
            + direction
            + sigma * noise
        )

    return x


@torch.no_grad()
def separate_with_ddpm(
    model: LatentConditionalDDPM,
    schedule: DiffusionSchedule,
    received_latent: Tensor,
    seed: int = 42,
    sampling_method: str = "ddim",
    ddim_steps: int = 50,
    ddim_eta: float = 0.0,
    clip_x0: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Separate EuroSAT and DeepGlobe latents with one conditional DDPM.

    Returns:
        pred_euro_sat
        pred_deep_globe
    """
    batch_size = received_latent.size(0)
    device = received_latent.device

    label_e = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=device,
    )

    label_d = torch.ones(
        batch_size,
        dtype=torch.long,
        device=device,
    )

    pred_e = sample_ddpm(
        model=model,
        schedule=schedule,
        cond=received_latent,
        labels=label_e,
        seed=seed,
        sampling_method=sampling_method,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        clip_x0=clip_x0,
    )

    pred_d = sample_ddpm(
        model=model,
        schedule=schedule,
        cond=received_latent,
        labels=label_d,
        seed=seed + 1,
        sampling_method=sampling_method,
        ddim_steps=ddim_steps,
        ddim_eta=ddim_eta,
        clip_x0=clip_x0,
    )

    return pred_e, pred_d


# ============================================================
# Generic checkpoint loaders
# ============================================================

def load_ddpm_separator(
    checkpoint: CheckpointLike,
    device: Union[str, torch.device] = "cpu",
    separator_class=LatentConditionalDDPM,
    z_dim: Optional[int] = None,
    num_classes: Optional[int] = None,
    time_dim: Optional[int] = None,
    label_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    num_blocks: Optional[int] = None,
    dropout: Optional[float] = None,
    diffusion_steps: Optional[int] = None,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    strict: bool = True,
    freeze: bool = True,
) -> Tuple[LatentConditionalDDPM, DiffusionSchedule]:
    """
    Load a latent DDPM and its matching diffusion schedule.
    """
    checkpoint_data = _load_checkpoint(
        checkpoint,
        map_location=device,
    )
    config = _checkpoint_config(checkpoint_data)

    resolved_z_dim = int(
        z_dim
        if z_dim is not None
        else config.get("z_dim", 256)
    )
    resolved_num_classes = int(
        num_classes
        if num_classes is not None
        else config.get("num_classes", 2)
    )
    resolved_time_dim = int(
        time_dim
        if time_dim is not None
        else config.get("time_dim", 128)
    )
    resolved_label_dim = int(
        label_dim
        if label_dim is not None
        else config.get("label_dim", 64)
    )
    resolved_hidden_dim = int(
        hidden_dim
        if hidden_dim is not None
        else config.get("hidden_dim", 768)
    )
    resolved_num_blocks = int(
        num_blocks
        if num_blocks is not None
        else config.get("num_blocks", 4)
    )
    resolved_dropout = float(
        dropout
        if dropout is not None
        else config.get("dropout", 0.0)
    )
    resolved_diffusion_steps = int(
        diffusion_steps
        if diffusion_steps is not None
        else config.get("diffusion_steps", 400)
    )

    model = separator_class(
        z_dim=resolved_z_dim,
        num_classes=resolved_num_classes,
        time_dim=resolved_time_dim,
        label_dim=resolved_label_dim,
        hidden_dim=resolved_hidden_dim,
        num_blocks=resolved_num_blocks,
        dropout=resolved_dropout,
    ).to(device)

    state = _extract_state_dict(checkpoint_data)
    model.load_state_dict(state, strict=strict)

    schedule = DiffusionSchedule(
        timesteps=resolved_diffusion_steps,
        beta_start=beta_start,
        beta_end=beta_end,
    ).to(device)

    if freeze:
        freeze_model(model)
        schedule.eval()
    else:
        model.train()

    return model, schedule


def load_mlp_separator(
    checkpoint: CheckpointLike,
    device: Union[str, torch.device] = "cpu",
    separator_class=JointLatentMLPSeparator,
    z_dim: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    num_blocks: Optional[int] = None,
    dropout: Optional[float] = None,
    strict: bool = True,
    freeze: bool = True,
) -> JointLatentMLPSeparator:
    """Load a deterministic joint MLP separator."""
    checkpoint_data = _load_checkpoint(
        checkpoint,
        map_location=device,
    )
    config = _checkpoint_config(checkpoint_data)

    resolved_z_dim = int(
        z_dim
        if z_dim is not None
        else config.get("z_dim", 256)
    )
    resolved_hidden_dim = int(
        hidden_dim
        if hidden_dim is not None
        else config.get("hidden_dim", 768)
    )
    resolved_num_blocks = int(
        num_blocks
        if num_blocks is not None
        else config.get("num_blocks", 6)
    )
    resolved_dropout = float(
        dropout
        if dropout is not None
        else config.get("dropout", 0.0)
    )

    model = separator_class(
        z_dim=resolved_z_dim,
        hidden_dim=resolved_hidden_dim,
        num_blocks=resolved_num_blocks,
        dropout=resolved_dropout,
    ).to(device)

    state = _extract_state_dict(checkpoint_data)
    model.load_state_dict(state, strict=strict)

    if freeze:
        freeze_model(model)
    else:
        model.train()

    return model


# ============================================================
# Dedicated loaders for the four systems
# ============================================================

def load_vae_ddpm_vae(
    checkpoint: CheckpointLike = (
        "./save_model/DDPM_vae_latent_separator_latest.pth"
    ),
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> Tuple[VAEDDPMVAESeparator, DiffusionSchedule]:
    """Load the DDPM used by VAE-DDPM-VAE."""
    model, schedule = load_ddpm_separator(
        checkpoint=checkpoint,
        device=device,
        separator_class=VAEDDPMVAESeparator,
        **kwargs,
    )
    return model, schedule


def load_vae_mlp_vae(
    checkpoint: CheckpointLike = (
        "./save_model/MLP_vae_latent_separator_best.pth"
    ),
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> VAEMLPVAESeparator:
    """Load the MLP used by VAE-MLP-VAE."""
    return load_mlp_separator(
        checkpoint=checkpoint,
        device=device,
        separator_class=VAEMLPVAESeparator,
        **kwargs,
    )


def load_vae_ddpm_gan(
    checkpoint: CheckpointLike = (
        "./save_model/"
        "DDPM_vaegan_scratch_latent_separator_best.pth"
    ),
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> Tuple[VAEDDPMGANSeparator, DiffusionSchedule]:
    """Load the DDPM used by VAE-DDPM-GAN."""
    model, schedule = load_ddpm_separator(
        checkpoint=checkpoint,
        device=device,
        separator_class=VAEDDPMGANSeparator,
        **kwargs,
    )
    return model, schedule


def load_vae_mlp_gan(
    checkpoint: CheckpointLike = (
        "./save_model/"
        "MLP_vaegan_scratch_latent_separator_best.pth"
    ),
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> VAEMLPGANSeparator:
    """Load the MLP used by VAE-MLP-GAN."""
    return load_mlp_separator(
        checkpoint=checkpoint,
        device=device,
        separator_class=VAEMLPGANSeparator,
        **kwargs,
    )


__all__ = [
    # Shared components
    "SinusoidalTimeEmbedding",
    "ResidualMLPBlock",
    "DiffusionSchedule",

    # Generic architectures
    "LatentConditionalDDPM",
    "JointLatentMLPSeparator",

    # Named architectures
    "VAEDDPMVAESeparator",
    "VAEMLPVAESeparator",
    "VAEDDPMGANSeparator",
    "VAEMLPGANSeparator",

    # Sampling
    "sample_ddpm",
    "separate_with_ddpm",

    # Generic loaders
    "load_ddpm_separator",
    "load_mlp_separator",

    # Dedicated loaders
    "load_vae_ddpm_vae",
    "load_vae_mlp_vae",
    "load_vae_ddpm_gan",
    "load_vae_mlp_gan",

    # Utilities
    "set_requires_grad",
    "freeze_model",
]

