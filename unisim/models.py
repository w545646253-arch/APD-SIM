
from __future__ import annotations

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) int or float"""
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SeparableConv3d(nn.Module):
    """Approximate 3D conv by (1,3,3) then (3,1,1)."""
    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.conv_xy = nn.Conv3d(in_ch, out_ch, kernel_size=(1,3,3), padding=(0,1,1), bias=bias)
        self.conv_z  = nn.Conv3d(out_ch, out_ch, kernel_size=(3,1,1), padding=(1,0,0), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_xy(x)
        x = self.conv_z(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0, groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_ch)
        self.conv1 = SeparableConv3d(in_ch, out_ch)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = SeparableConv3d(out_ch, out_ch)
        self.res_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # add time embedding
        temb = self.time_mlp(t_emb).view(-1, self.out_ch, 1, 1, 1)
        h = h + temb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose3d(ch, ch, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class UNet3DConditioned(nn.Module):
    """
    Unified 2D/3D score model.

    Input: concat([x_t, cond_up]) with shape (B, 1+K, Z, H, W)
    Output: predicted noise eps with shape (B, 1, Z, H, W)

    Notes:
    - works for Z=1 (2D) and Z>1 (3D)
    - uses separable 3D convs to reduce compute
    """
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        time_dim: int = 128,
        groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.init_conv = SeparableConv3d(in_channels, base_channels)

        # Down path
        chs: List[int] = [base_channels]
        downs = []
        in_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                downs.append(ResBlock(in_ch, out_ch, time_dim=time_dim, dropout=dropout, groups=groups))
                in_ch = out_ch
                chs.append(in_ch)
            if i != len(channel_mults) - 1:
                downs.append(Downsample(in_ch))
                chs.append(in_ch)
        self.downs = nn.ModuleList(downs)

        # Middle
        self.mid1 = ResBlock(in_ch, in_ch, time_dim=time_dim, dropout=dropout, groups=groups)
        self.mid2 = ResBlock(in_ch, in_ch, time_dim=time_dim, dropout=dropout, groups=groups)

        # Up path
        ups = []
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):  # +1 to consume skip after downsample
                skip_ch = chs.pop()
                ups.append(ResBlock(in_ch + skip_ch, out_ch, time_dim=time_dim, dropout=dropout, groups=groups))
                in_ch = out_ch
            if i != 0:
                ups.append(Upsample(in_ch))
        self.ups = nn.ModuleList(ups)

        self.final_norm = nn.GroupNorm(num_groups=groups, num_channels=in_ch)
        self.final_conv = nn.Conv3d(in_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, Z, H, W)
        t: (B,) timesteps
        """
        t_emb = self.time_emb(t)

        x = self.init_conv(x)
        skips: List[torch.Tensor] = [x]

        for layer in self.downs:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
                skips.append(x)
            else:
                x = layer(x)
                skips.append(x)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        for layer in self.ups:
            if isinstance(layer, ResBlock):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t_emb)
            else:
                x = layer(x)

        x = self.final_norm(x)
        x = F.silu(x)
        return self.final_conv(x)
