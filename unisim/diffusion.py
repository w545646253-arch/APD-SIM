
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal."""
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-6, 0.999)


@dataclass
class DiffusionConfig:
    T: int = 1000
    beta_schedule: str = "cosine"  # "cosine" or "linear"
    ddim_eta: float = 0.0


class DiffusionScheduler:
    def __init__(self, cfg: DiffusionConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        if cfg.beta_schedule == "cosine":
            betas = cosine_beta_schedule(cfg.T)
        elif cfg.beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, cfg.T)
        else:
            raise ValueError(f"Unknown beta_schedule: {cfg.beta_schedule}")

        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion: sample x_t ~ q(x_t|x0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        # gather per-batch coefficients
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(self, xt: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return (xt - sqrt_1mab * eps) / (sqrt_ab + 1e-8)

    def ddim_timesteps(self, num_steps: int) -> List[int]:
        """Evenly spaced timesteps from T-1 to 0."""
        if num_steps >= self.cfg.T:
            return list(range(self.cfg.T - 1, -1, -1))
        step = self.cfg.T // num_steps
        ts = list(range(self.cfg.T - 1, -1, -step))
        if ts[-1] != 0:
            ts.append(0)
        return ts

    def ddim_step(self, xt: torch.Tensor, t: int, t_prev: int, eps: torch.Tensor, x0_hat: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
        """One DDIM update."""
        ab_t = self.alphas_cumprod[t]
        ab_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

        sqrt_ab_prev = torch.sqrt(ab_prev)
        sqrt_one_minus_ab_prev = torch.sqrt(1.0 - ab_prev)

        if eta == 0.0:
            # deterministic
            return sqrt_ab_prev * x0_hat + sqrt_one_minus_ab_prev * eps

        # stochastic DDIM
        sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t)) * torch.sqrt(1 - ab_t / ab_prev)
        noise = torch.randn_like(xt)
        dir_term = torch.sqrt(1 - ab_prev - sigma ** 2) * eps
        return sqrt_ab_prev * x0_hat + dir_term + sigma * noise


class EMA:
    """Exponential moving average for model parameters (helps sampling)."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)
