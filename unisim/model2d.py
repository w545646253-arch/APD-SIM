"""Strictly two-dimensional conditional score model for formal APD-SIM R2.

This module intentionally does not import the legacy 3-D model.  Every spatial
operator is Conv2d/ConvTranspose2d and every image tensor is ``(B,C,H,W)``.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 4:
            raise ValueError("time embedding dimension must be at least four")
        self.dim = int(dim)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError(f"timestep must be (B,), got {tuple(timestep.shape)}")
        half = self.dim // 2
        scale = math.log(10000.0) / max(1, half - 1)
        frequency = torch.exp(
            torch.arange(half, device=timestep.device, dtype=torch.float32) * -scale
        )
        angle = timestep.float().unsqueeze(1) * frequency.unsqueeze(0)
        embedding = torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)
        return F.pad(embedding, (0, self.dim - embedding.shape[1]))


class ResidualBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_dim: int,
        dropout: float,
        groups: int,
    ):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("GroupNorm groups must divide every residual channel count")
        self.out_channels = int(out_channels)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, image: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(image)))
        hidden = hidden + self.time_projection(time_embedding).view(
            -1, self.out_channels, 1, 1
        )
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.residual(image)


class Downsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv(image)


class Upsample2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv(image)


class APDConditionedUNet2D(nn.Module):
    """Fixed-slot, mask-conditioned epsilon predictor using only 2-D tensors."""

    architecture_contract = "APD_DMD_R2_STRICT_2D_CONV_V1"

    def __init__(
        self,
        in_channels: int = 31,
        base_channels: int = 48,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        time_dim: int = 128,
        groups: int = 8,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = tuple(int(value) for value in channel_mults)
        self.num_res_blocks = int(num_res_blocks)
        if self.in_channels != 31:
            raise ValueError("Formal APD-DMD R2 requires 1 + 15 slots + 15 mask channels")

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding2D(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.initial = nn.Conv2d(self.in_channels, base_channels, kernel_size=3, padding=1)

        skip_channels: List[int] = [base_channels]
        down_layers: List[nn.Module] = []
        current = base_channels
        for level, multiplier in enumerate(self.channel_mults):
            target = base_channels * multiplier
            for _ in range(self.num_res_blocks):
                down_layers.append(
                    ResidualBlock2D(
                        current,
                        target,
                        time_dim=time_dim,
                        dropout=dropout,
                        groups=groups,
                    )
                )
                current = target
                skip_channels.append(current)
            if level != len(self.channel_mults) - 1:
                down_layers.append(Downsample2D(current))
                skip_channels.append(current)
        self.down = nn.ModuleList(down_layers)

        self.middle1 = ResidualBlock2D(
            current, current, time_dim=time_dim, dropout=dropout, groups=groups
        )
        self.middle2 = ResidualBlock2D(
            current, current, time_dim=time_dim, dropout=dropout, groups=groups
        )

        up_layers: List[nn.Module] = []
        for level, multiplier in reversed(list(enumerate(self.channel_mults))):
            target = base_channels * multiplier
            for _ in range(self.num_res_blocks + 1):
                skip = skip_channels.pop()
                up_layers.append(
                    ResidualBlock2D(
                        current + skip,
                        target,
                        time_dim=time_dim,
                        dropout=dropout,
                        groups=groups,
                    )
                )
                current = target
            if level != 0:
                up_layers.append(Upsample2D(current))
        if skip_channels:
            raise AssertionError("Internal U-Net skip accounting error")
        self.up = nn.ModuleList(up_layers)
        self.final_norm = nn.GroupNorm(groups, current)
        self.final = nn.Conv2d(current, 1, kernel_size=1)

    def forward(self, conditioned: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if conditioned.ndim != 4:
            raise ValueError(f"Formal model requires (B,C,H,W), got {tuple(conditioned.shape)}")
        if conditioned.shape[1] != self.in_channels:
            raise ValueError(
                f"Condition has {conditioned.shape[1]} channels, expected {self.in_channels}"
            )
        if timestep.shape != (conditioned.shape[0],):
            raise ValueError(
                f"timestep shape {tuple(timestep.shape)} does not match batch {conditioned.shape[0]}"
            )
        embedded = self.time_embedding(timestep)
        hidden = self.initial(conditioned)
        skips: List[torch.Tensor] = [hidden]
        for layer in self.down:
            if isinstance(layer, ResidualBlock2D):
                hidden = layer(hidden, embedded)
            else:
                hidden = layer(hidden)
            skips.append(hidden)
        hidden = self.middle2(self.middle1(hidden, embedded), embedded)
        for layer in self.up:
            if isinstance(layer, ResidualBlock2D):
                skip = skips.pop()
                if hidden.shape[-2:] != skip.shape[-2:]:
                    raise ValueError("Input spatial dimensions must be divisible by the U-Net scale")
                hidden = layer(torch.cat((hidden, skip), dim=1), embedded)
            else:
                hidden = layer(hidden)
        return self.final(F.silu(self.final_norm(hidden)))


def assert_strictly_2d_model(model: nn.Module) -> None:
    forbidden = (nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose3d)
    offenders = [module.__class__.__qualname__ for module in model.modules() if isinstance(module, forbidden)]
    if offenders:
        raise RuntimeError("Formal 2-D model contains forbidden spatial operators: " + ", ".join(offenders))


__all__ = ["APDConditionedUNet2D", "assert_strictly_2d_model"]
