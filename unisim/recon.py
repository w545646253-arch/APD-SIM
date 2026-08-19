from __future__ import annotations

"""unisim/recon.py (v4: physics-stepwise + robust DC)

This module implements a physics-guided conditional DDIM sampler for 2D-SIM/3D-SIM.

Key ideas:
  - Diffusion prior (UNet score model) + differentiable SIM forward model.
  - Step-wise data-consistency (DC) corrector using a Poisson-Gaussian likelihood.
  - Continuation (optics progressive degradation): early steps use slightly degraded optics,
    later steps tighten to nominal.
  - Optional self-calibration of illumination parameters.

Changes vs v3:
  - Fix missing `math` import.
  - Add variance floor (model-mismatch) for DC likelihood to prevent over-confident
    gradients on normalized data.
  - Optional TV regularization and gradient clipping inside DC to suppress pattern leakage.
  - Add `dc_noisy_power` so DC weight can decay near the final steps (prevents last-step speckle).

The API is backward compatible: new arguments have safe defaults.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import math
import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .diffusion import DiffusionScheduler
from .sim_forward import (
    SIMConfig,
    build_psf,
    generate_patterns,
    apply_psf,
    downsample_xy,
    theta_nominal,
    nll_poisson_gaussian_cam,
)
from .utils import read_tiff, save_tiff


# -------------------------
# I/O helpers
# -------------------------

def _ensure_kzhw(y: np.ndarray) -> np.ndarray:
    """Standardize raw SIM stack into (K,Z,H,W) numpy."""
    y = np.asarray(y)
    if y.ndim == 2:
        return y[None, None, ...]
    if y.ndim == 3:
        return y[:, None, ...]
    if y.ndim == 4:
        return y
    raise ValueError(f"Unsupported raw shape: {y.shape}")


def load_raw_stack(path: Union[str, Path]) -> np.ndarray:
    """Load raw SIM stack from tif/tiff or a folder of tif frames."""
    path = Path(path)
    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in (".tif", ".tiff")])
        if not files:
            raise RuntimeError(f"No tif/tiff found in: {path}")
        frames = [np.asarray(read_tiff(f)) for f in files]
        frames = [fr if fr.ndim == 3 else fr[None, ...] for fr in frames]  # (Z,H,W)
        y = np.stack(frames, axis=0)  # (K,Z,H,W)
        return _ensure_kzhw(y)
    y = np.asarray(read_tiff(path))
    return _ensure_kzhw(y)


def normalize_raw_stack(
    y: np.ndarray,
    p_low: float = 0.5,
    p_high: float = 99.5,
    min_range: float = 1e-2,
) -> np.ndarray:
    """Global percentile normalization to [0,1] with denominator clamping."""
    y = y.astype(np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(y, p_low))
    hi = float(np.percentile(y, p_high))
    denom = max(hi - lo, float(min_range))
    y = (y - lo) / (denom + 1e-8)
    return np.clip(y, 0.0, 1.0)


# -------------------------
# Conditioning utilities
# -------------------------

def pad_to_kmax(y: torch.Tensor, kmax: int) -> torch.Tensor:
    """(B,K,Z,H,W) -> (B,kmax,Z,H,W) with zero padding."""
    if y.ndim != 5:
        raise ValueError(f"pad_to_kmax expects 5D (B,K,Z,H,W), got {y.shape}")
    B, K, Z, H, W = y.shape
    if K > kmax:
        raise ValueError(f"K={K} > kmax={kmax}")
    if K == kmax:
        return y
    out = torch.zeros((B, kmax, Z, H, W), device=y.device, dtype=y.dtype)
    out[:, :K] = y
    return out


def make_kmask(
    B: int,
    K: int,
    Z: int,
    H: int,
    W: int,
    kmax: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Binary mask (B,kmax,Z,H,W): 1 for valid frames, 0 for padded/missing."""
    m = torch.zeros((B, kmax, Z, H, W), device=device, dtype=dtype)
    m[:, :K] = 1.0
    return m


def _resize_xy_5d(y: torch.Tensor, target_h: int, target_w: int, mode_up: str = "bilinear") -> torch.Tensor:
    """Resize only (H,W) for a 5D tensor (B,C,Z,H,W)."""
    if y.ndim != 5:
        raise ValueError(f"_resize_xy_5d expects 5D (B,C,Z,H,W), got {y.shape}")
    B, C, Z, H, W = y.shape
    if (H, W) == (target_h, target_w):
        return y
    y2 = y.permute(0, 2, 1, 3, 4).contiguous().view(B * Z, C, H, W)
    if target_h >= H or target_w >= W:
        y2r = F.interpolate(y2, size=(target_h, target_w), mode=mode_up, align_corners=False)
    else:
        y2r = F.interpolate(y2, size=(target_h, target_w), mode="area")
    y_out = y2r.view(B, Z, C, target_h, target_w).permute(0, 2, 1, 3, 4).contiguous()
    return y_out


def match_cond_to_x(cond: torch.Tensor, x: torch.Tensor, mode_up: str = "bilinear") -> torch.Tensor:
    """Make (B,C,Z,H,W) cond match x's (Z,H,W)."""
    if cond.ndim != 5 or x.ndim != 5:
        raise ValueError(f"match_cond_to_x expects 5D tensors, got cond={cond.shape}, x={x.shape}")
    _, _, Zx, Hx, Wx = x.shape
    if cond.shape[2] != Zx:
        if cond.shape[2] == 1 and Zx > 1:
            cond = cond.expand(cond.shape[0], cond.shape[1], Zx, cond.shape[3], cond.shape[4]).contiguous()
        else:
            cond = cond[:, :, :Zx]
    return _resize_xy_5d(cond, Hx, Wx, mode_up=mode_up)


# -------------------------
# Robust likelihood & regularizers
# -------------------------

def robust_nll_poisson_gaussian_cam(
    y: torch.Tensor,
    mu: torch.Tensor,
    photon_scale: Union[float, torch.Tensor],
    read_noise_e: Union[float, torch.Tensor],
    *,
    var_floor: float = 0.0,
    eps: float = 1e-8,
    reduce: str = "mean",
) -> torch.Tensor:
    """Same noise model as sim_forward.nll_poisson_gaussian_cam, but with variance floor.

    On normalized data in [0,1], photon_scale can be large (e.g. 5k-10k), making
    Var[y] = mu/photon_scale tiny and the likelihood over-confident.

    A small var_floor (e.g. 0.0004 = 0.02^2) models mismatch + unmodelled noise.
    """
    if float(var_floor) <= 0.0:
        return nll_poisson_gaussian_cam(y, mu, photon_scale=photon_scale, read_noise_e=read_noise_e, eps=eps, reduce=reduce)

    if not torch.is_tensor(photon_scale):
        ps = torch.tensor([float(photon_scale)], device=y.device, dtype=y.dtype)
    else:
        ps = photon_scale.to(device=y.device, dtype=y.dtype)

    if not torch.is_tensor(read_noise_e):
        rn = torch.tensor([float(read_noise_e)], device=y.device, dtype=y.dtype)
    else:
        rn = read_noise_e.to(device=y.device, dtype=y.dtype)

    ps = ps.clamp_min(1e-12)
    mu_pos = mu.clamp_min(0.0)
    var = mu_pos / ps + (rn / ps) ** 2 + eps
    var = torch.clamp(var, min=float(var_floor))
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)

    if reduce == "none":
        return nll
    if reduce == "sum":
        return nll.sum()
    if reduce == "mean":
        return nll.mean()
    raise ValueError(f"Unknown reduce: {reduce}")


def tv3d_l1(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic L1 TV for (B,1,Z,H,W)."""
    dz = (x[:, :, 1:] - x[:, :, :-1]).abs().mean() if x.shape[2] > 1 else x.new_tensor(0.0)
    dy = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() if x.shape[3] > 1 else x.new_tensor(0.0)
    dx = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().mean() if x.shape[4] > 1 else x.new_tensor(0.0)
    return dz + dy + dx


def clip_grad_norm_(g: torch.Tensor, max_norm: float, eps: float = 1e-12) -> torch.Tensor:
    """Return clipped gradient (does not modify in-place)."""
    if max_norm <= 0:
        return g
    n = torch.linalg.vector_norm(g)
    if torch.isfinite(n) and (n > max_norm):
        g = g * (max_norm / (n + eps))
    return g


# -------------------------
# Physics forward (cached PSF)
# -------------------------

@dataclass
class PSFCache:
    """Tiny cache for PSF by sigma_scale (float key)."""

    psf_by_sigma: Dict[float, torch.Tensor]

    def __init__(self):
        self.psf_by_sigma = {}

    def get(self, cfg: SIMConfig, device: torch.device, sigma_scale: float) -> torch.Tensor:
        key = float(np.round(sigma_scale, 6))
        if key in self.psf_by_sigma:
            return self.psf_by_sigma[key]
        psf, _ = build_psf(cfg, device=device, sigma_scale=sigma_scale)
        psf = psf.detach()
        self.psf_by_sigma[key] = psf
        return psf


def forward_mu_clean(
    x0: torch.Tensor,
    cfg: SIMConfig,
    mode: str,
    theta: Dict[str, torch.Tensor],
    psf_cache: Optional[PSFCache] = None,
) -> torch.Tensor:
    """Noiseless forward operator mu = A_theta(x0) in camera units.

    Differentiable, matches sim_forward.forward_clean, but supports PSF caching.
    Returns: (B,K,Z,Hc,Wc)
    """
    device = x0.device
    mode_l = mode.lower()
    if mode_l not in ("2d", "3d"):
        raise ValueError(f"mode must be '2d' or '3d', got: {mode}")

    # internal upsample
    if cfg.upsample != 1:
        B, C, Z, H, W = x0.shape
        x2 = x0.permute(0, 2, 1, 3, 4).reshape(B * Z, C, H, W)
        x2u = F.interpolate(x2, scale_factor=float(cfg.upsample), mode="bilinear", align_corners=False)
        Hu, Wu = x2u.shape[-2], x2u.shape[-1]
        x0u = x2u.reshape(B, Z, C, Hu, Wu).permute(0, 2, 1, 3, 4).contiguous()
    else:
        x0u = x0

    x_obj = x0u.sum(dim=1, keepdim=True) if x0u.shape[1] != 1 else x0u

    # PSF (cached)
    sigma_scale = float(theta.get("psf_sigma_scale", torch.tensor(cfg.psf_sigma_scale, device=device)).detach().reshape(-1)[0].item())
    if psf_cache is None:
        psf, _ = build_psf(cfg, device=device, sigma_scale=sigma_scale)
        psf = psf.detach()
    else:
        psf = psf_cache.get(cfg, device=device, sigma_scale=sigma_scale)

    patterns = generate_patterns(x_shape=x_obj.shape, cfg=cfg, mode=mode_l, theta=theta)  # (K,1,Z,Hu,Wu)
    B = x_obj.shape[0]
    K = patterns.shape[0]

    x_illum = x_obj.unsqueeze(1) * patterns.unsqueeze(0)  # (B,K,1,Z,Hu,Wu)
    x_illum = x_illum.reshape(B * K, 1, x_obj.shape[2], x_obj.shape[3], x_obj.shape[4])
    y_blur = apply_psf(x_illum, psf)  # (B*K,1,Z,Hu,Wu)
    y_blur = y_blur.reshape(B, K, 1, x_obj.shape[2], x_obj.shape[3], x_obj.shape[4])

    bg = theta.get("background", torch.tensor(cfg.background, device=device, dtype=y_blur.dtype))
    bg = bg.reshape(1).to(device=device, dtype=y_blur.dtype)
    y_blur = y_blur + bg.view(1, 1, 1, 1, 1, 1)

    y_blur = y_blur.squeeze(2)  # (B,K,Z,Hu,Wu)
    mu = downsample_xy(y_blur, factor=cfg.upsample)  # (B,K,Z,Hc,Wc)
    return mu


# -------------------------
# Step-wise schedules
# -------------------------

def _alpha_clean_frac(scheduler: DiffusionScheduler, t: int) -> float:
    """Return alpha_bar(t) in [0,1], where 0=very noisy, 1=almost clean."""
    a = scheduler.alphas_cumprod[t]
    return float(a.detach().cpu().item()) if torch.is_tensor(a) else float(a)


def dc_weight(
    clean_frac: float,
    *,
    w_max: float = 1.0,
    clean_power: float = 1.0,
    noisy_power: float = 0.0,
) -> float:
    """Annealed DC weight.

    clean_frac = alpha_bar(t) in [0,1]
    noisy_frac = 1 - clean_frac

    If noisy_power>0, the weight decays again near the final steps (noisy_frac->0),
    which prevents the last DDIM step from injecting speckle that will not be denoised.
    """
    c = float(max(0.0, min(1.0, clean_frac)))
    n = float(max(0.0, min(1.0, 1.0 - c)))
    return float(w_max * (c ** float(clean_power)) * (n ** float(noisy_power)))


def default_optics_degrade(noisy_frac: float, extra_blur: float = 0.35, mod_drop: float = 0.25) -> Tuple[float, float]:
    """Return (psf_sigma_mul, mod_depth_mul) given noisy_frac in [0,1]."""
    noisy = float(max(0.0, min(1.0, noisy_frac)))
    psf_mul = 1.0 + float(extra_blur) * noisy
    mod_mul = 1.0 - float(mod_drop) * noisy
    return float(psf_mul), float(mod_mul)


def _autocast_ctx(device: torch.device, enabled: bool):
    """Unified autocast context manager across torch versions."""
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


# -------------------------
# Main reconstruction API
# -------------------------

@torch.no_grad()
def reconstruct_ddim_physics_stepwise(
    raw_np: np.ndarray,
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    cfg: SIMConfig,
    mode: str,
    out_path: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    # diffusion
    num_steps: int = 50,
    kmax: int = 15,
    clip_x0: bool = True,
    use_amp: bool = False,
    # conditioning
    use_kmask: bool = True,
    normalize_input: bool = True,
    # physics guidance
    dc_iters_max: int = 1,
    dc_lr: float = 5e-2,
    dc_weight_max: float = 1.0,
    dc_power: float = 1.0,
    dc_noisy_power: float = 0.0,
    dc_start_step: int = 0,
    dc_var_floor: float = 0.0,
    dc_tv_weight: float = 0.0,
    dc_grad_clip: float = 0.0,
    # optics progressive degradation
    optics_extra_blur: float = 0.35,
    optics_mod_drop: float = 0.25,
    # self-calibration (optional)
    self_calibrate: bool = False,
    theta_lr: float = 2e-3,
    theta_power: float = 1.0,
    theta_noisy_power: float = 0.0,
    theta_update_every: int = 1,
    # debug
    return_debug: bool = False,
    # revised DMD protocol (required by new production entrypoints)
    protocol_id: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, object]]]:
    """Physics-stepwise conditional DDIM reconstruction.

    Args:
      raw_np: (K,Z,H,W)
      use_kmask: append KMAX binary mask channels to the condition input.
      dc_var_floor: variance floor in normalized [0,1] domain (e.g. 0.0004).
      dc_tv_weight: TV penalty weight inside DC (helps suppress pattern leakage).
      dc_noisy_power: multiply DC weight by (1-alpha_bar)^dc_noisy_power (reduces last-step artifacts).

    Returns:
      recon_np: (Z,H,W)
      optionally (recon_np, debug_dict)
    """
    protocol = None
    cfg_runtime = cfg
    if protocol_id is not None:
        # Local import avoids a module cycle: protocol_runtime reuses the
        # low-level cached forward implementation in this module.
        from .protocol_runtime import (
            embed_raw_to_slots,
            reorder_generated_to_raw,
            require_protocol,
            sim_config_for_protocol,
        )

        protocol = require_protocol(protocol_id)
        cfg_runtime = sim_config_for_protocol(cfg, protocol)

    device = torch.device(device)
    model = model.to(device).eval()

    raw_np = np.asarray(raw_np)
    raw_np = _ensure_kzhw(raw_np)
    if normalize_input:
        raw_np = normalize_raw_stack(raw_np)
    else:
        raw_np = raw_np.astype(np.float32)
        raw_np = np.nan_to_num(raw_np, nan=0.0, posinf=0.0, neginf=0.0)
        raw_np = np.clip(raw_np, 0.0, None)

    y = torch.from_numpy(raw_np).float().to(device)
    if y.ndim != 4:
        raise ValueError(f"raw_np must be (K,Z,H,W), got {raw_np.shape}")
    y = y.unsqueeze(0)
    B, K_in, Z, H, W = y.shape

    if protocol is not None:
        if K_in != protocol.frame_count:
            raise ValueError(
                f"{protocol.protocol_id} requires exactly {protocol.frame_count} raw frames; got {K_in}"
            )
        y_pad, protocol_mask = embed_raw_to_slots(y, protocol, kmax=kmax)
        kmask = protocol_mask if use_kmask else None
    else:
        # Explicit backward-compatibility branch for legacy callers.  Revised
        # DMD entrypoints always provide protocol_id and never enter it.
        y_pad = pad_to_kmax(y, kmax=kmax)
        kmask = make_kmask(B, K_in, Z, H, W, kmax=kmax, device=device, dtype=y_pad.dtype) if use_kmask else None

    x = torch.randn((B, 1, Z, H, W), device=device)

    theta_base = theta_nominal(cfg_runtime, mode=mode, device=device)
    theta_params = _init_theta_trainable(
        cfg_runtime,
        mode=mode,
        K_true=(protocol.frame_count if protocol is not None else _infer_K(cfg_runtime, mode)),
        device=device,
    ) if self_calibrate else None

    psf_cache = PSFCache()

    timesteps = scheduler.ddim_timesteps(num_steps=num_steps)
    eta = scheduler.cfg.ddim_eta

    last_theta_step: Optional[Dict[str, torch.Tensor]] = None
    last_nll_dc: Optional[float] = None

    for i, t in enumerate(tqdm(timesteps, desc=f"DDIM-Physics({mode})", ncols=90)):
        t_int = int(t)
        t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)

        y_cond = match_cond_to_x(y_pad, x)
        if use_kmask:
            m_cond = match_cond_to_x(kmask, x, mode_up="nearest")
            x_in = torch.cat([x, y_cond, m_cond], dim=1)
        else:
            x_in = torch.cat([x, y_cond], dim=1)

        with _autocast_ctx(device, enabled=bool(use_amp)):
            eps = model(x_in, t_tensor)
        eps = eps.to(dtype=x.dtype)

        x0_hat = scheduler.predict_x0_from_eps(x, t_tensor, eps)
        if clip_x0:
            x0_hat = x0_hat.clamp(0.0, 1.0)

        clean_frac = _alpha_clean_frac(scheduler, t_int)
        noisy_frac = 1.0 - clean_frac

        w_dc = dc_weight(clean_frac, w_max=dc_weight_max, clean_power=dc_power, noisy_power=dc_noisy_power)
        denom = max(float(dc_weight_max), 1e-12)
        dc_iters = int(np.round(float(dc_iters_max) * (w_dc / denom)))
        dc_iters = int(np.clip(dc_iters, 0, int(dc_iters_max)))
        if i < int(dc_start_step):
            dc_iters = 0
            w_dc = 0.0

        psf_mul, mod_mul = default_optics_degrade(noisy_frac, extra_blur=optics_extra_blur, mod_drop=optics_mod_drop)

        if (dc_iters > 0) and (w_dc > 0.0):
            with torch.enable_grad():
                x0_dc = x0_hat.detach().clone().requires_grad_(True)

                for _ in range(dc_iters):
                    theta_step = _compose_theta_for_step(theta_base, theta_params, psf_mul=psf_mul, mod_mul=mod_mul)
                    mu = forward_mu_clean(x0_dc, cfg_runtime, mode=mode, theta=theta_step, psf_cache=psf_cache)
                    if protocol is not None:
                        mu = reorder_generated_to_raw(mu, protocol)
                    else:
                        mu = mu[:, :K_in]

                    ps = theta_step.get("photon_scale", cfg_runtime.photon_scale)
                    rn = theta_step.get("read_noise_e", cfg_runtime.read_noise_e)
                    loss = robust_nll_poisson_gaussian_cam(
                        y[:, :K_in],
                        mu,
                        photon_scale=ps,
                        read_noise_e=rn,
                        var_floor=float(dc_var_floor),
                        reduce="mean",
                    )
                    if float(dc_tv_weight) > 0.0:
                        loss = loss + float(dc_tv_weight) * tv3d_l1(x0_dc)

                    grad = torch.autograd.grad(loss, x0_dc, retain_graph=False, create_graph=False)[0]
                    grad = clip_grad_norm_(grad, float(dc_grad_clip))

                    x0_dc = (x0_dc - float(dc_lr) * float(w_dc) * grad).detach().requires_grad_(True)
                    if clip_x0:
                        x0_dc = x0_dc.clamp(0.0, 1.0).detach().requires_grad_(True)

                x0_hat = x0_dc.detach()

        if self_calibrate and (theta_params is not None) and (theta_update_every > 0) and ((i % int(theta_update_every)) == 0):
            w_th = dc_weight(clean_frac, w_max=1.0, clean_power=theta_power, noisy_power=theta_noisy_power)
            if w_th > 0.0:
                with torch.enable_grad():
                    x0_fix = x0_hat.detach()
                    theta_step = _compose_theta_for_step(theta_base, theta_params, psf_mul=psf_mul, mod_mul=mod_mul)
                    mu = forward_mu_clean(x0_fix, cfg_runtime, mode=mode, theta=theta_step, psf_cache=psf_cache)
                    if protocol is not None:
                        mu = reorder_generated_to_raw(mu, protocol)
                    else:
                        mu = mu[:, :K_in]

                    ps = theta_step.get("photon_scale", cfg_runtime.photon_scale)
                    rn = theta_step.get("read_noise_e", cfg_runtime.read_noise_e)
                    loss_th = robust_nll_poisson_gaussian_cam(
                        y[:, :K_in],
                        mu,
                        photon_scale=ps,
                        read_noise_e=rn,
                        var_floor=float(dc_var_floor),
                        reduce="mean",
                    )

                    th_list = [p for p in theta_params.values()]
                    grads = torch.autograd.grad(loss_th, th_list, retain_graph=False, create_graph=False, allow_unused=True)
                    _apply_theta_grads(theta_params, grads, lr=float(theta_lr) * float(w_th))

        t_prev = int(timesteps[i + 1]) if i < (len(timesteps) - 1) else -1
        x = scheduler.ddim_step(x, t=t_int, t_prev=t_prev, eps=eps, x0_hat=x0_hat, eta=eta)

    last_theta_step = _compose_theta_for_step(theta_base, theta_params, psf_mul=1.0, mod_mul=1.0)

    if bool(return_debug):
        with torch.no_grad():
            mu = forward_mu_clean(x0_hat, cfg_runtime, mode=mode, theta=last_theta_step, psf_cache=psf_cache)
            if protocol is not None:
                mu = reorder_generated_to_raw(mu, protocol)
            else:
                mu = mu[:, :K_in]
            ps = last_theta_step.get("photon_scale", cfg_runtime.photon_scale)
            rn = last_theta_step.get("read_noise_e", cfg_runtime.read_noise_e)
            nll = robust_nll_poisson_gaussian_cam(
                y[:, :K_in],
                mu,
                photon_scale=ps,
                read_noise_e=rn,
                var_floor=float(dc_var_floor),
                reduce="mean",
            )
            last_nll_dc = float(nll.detach().cpu().item())

    recon = x0_hat.detach().cpu().numpy()[0, 0]
    save_tiff(out_path, recon.astype(np.float32))

    if not bool(return_debug):
        return recon

    debug: Dict[str, object] = {
        "K_in": int(K_in),
        "Z": int(Z),
        "H": int(H),
        "W": int(W),
        "dc_var_floor": float(dc_var_floor),
        "dc_tv_weight": float(dc_tv_weight),
        "dc_noisy_power": float(dc_noisy_power),
        "nll_dc": float(last_nll_dc) if last_nll_dc is not None else None,
        "protocol_id": protocol.protocol_id if protocol is not None else "LEGACY_UNBOUND",
        "protocol_hash": protocol.protocol_hash if protocol is not None else None,
        "raw_to_slot_mapping": list(protocol.raw_to_slot_mapping) if protocol is not None else list(range(K_in)),
        "theta": {},
    }

    th_out: Dict[str, object] = {}
    for k, v in (last_theta_step or {}).items():
        if torch.is_tensor(v):
            vv = v.detach().cpu().float().reshape(-1).numpy()
            if vv.size == 1:
                th_out[k] = float(vv[0])
            else:
                th_out[k] = vv.tolist()
        else:
            th_out[k] = v
    debug["theta"] = th_out
    return recon, debug


def reconstruct_ddim_protocol_stepwise(
    raw_np: np.ndarray,
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    cfg: SIMConfig,
    protocol_id: str,
    mode: str,
    out_path: Union[str, Path],
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, object]]]:
    """Revised production reconstruction entrypoint with mandatory protocol binding.

    Keeping this as a named wrapper prevents new callers from accidentally
    falling back to the legacy first-K semantics retained above solely for old
    scripts and reproducibility audits.
    """
    if not protocol_id:
        raise ValueError("protocol_id is required for revised APD-DMD reconstruction")
    return reconstruct_ddim_physics_stepwise(
        raw_np=raw_np,
        model=model,
        scheduler=scheduler,
        cfg=cfg,
        mode=mode,
        out_path=out_path,
        protocol_id=protocol_id,
        **kwargs,
    )


def reconstruct_conditional_ddim(
    raw_np: np.ndarray,
    model: torch.nn.Module,
    scheduler: DiffusionScheduler,
    cfg: SIMConfig,
    mode: str,
    out_path: Union[str, Path],
    *,
    calib_iters_per_step: int = 1,
    calib_lr: float = 2e-3,
    dc_iters: int = 1,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, object]]]:
    """Legacy API adapter retained for the pre-existing demo scripts.

    This adapter is deliberately not used by revised DMD production paths; it
    restores the repository's historical import surface while the mandatory-
    protocol wrapper above supplies the new fail-closed interface.
    """
    kwargs.setdefault("theta_update_every", max(1, int(calib_iters_per_step)))
    kwargs.setdefault("theta_lr", float(calib_lr))
    kwargs.setdefault("dc_iters_max", int(dc_iters))
    return reconstruct_ddim_physics_stepwise(
        raw_np=raw_np,
        model=model,
        scheduler=scheduler,
        cfg=cfg,
        mode=mode,
        out_path=out_path,
        **kwargs,
    )


# -------------------------
# Theta self-calibration helpers
# -------------------------

def _infer_K(cfg: SIMConfig, mode: str) -> int:
    mode_l = mode.lower()
    if mode_l == "2d":
        return len(cfg.angle_list) * len(cfg.phase_list_2d)
    if mode_l == "3d":
        return len(cfg.angle_list) * len(cfg.phase_list_3d)
    raise ValueError(f"Unknown mode: {mode}")


def _init_theta_trainable(cfg: SIMConfig, mode: str, K_true: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """Trainable theta for gradient-based self-calibration."""
    mode_l = mode.lower()
    n_angles = len(cfg.angle_list)

    th: Dict[str, torch.Tensor] = {
        "k_ratio_xy": torch.tensor([cfg.k_ratio_xy], device=device, dtype=torch.float32, requires_grad=True),
        "mod_depth": torch.tensor([cfg.modulation_depth], device=device, dtype=torch.float32, requires_grad=True),
        "background": torch.tensor([cfg.background], device=device, dtype=torch.float32, requires_grad=True),
        "phase_offsets": torch.zeros((K_true,), device=device, dtype=torch.float32, requires_grad=True),
        "angle_offsets": torch.zeros((n_angles,), device=device, dtype=torch.float32, requires_grad=True),
    }
    if mode_l == "3d":
        th["kz_ratio"] = torch.tensor([cfg.kz_ratio], device=device, dtype=torch.float32, requires_grad=True)
    else:
        th["kz_ratio"] = torch.tensor([0.0], device=device, dtype=torch.float32, requires_grad=False)
    return th


def _compose_theta_for_step(
    theta_base: Dict[str, torch.Tensor],
    theta_params: Optional[Dict[str, torch.Tensor]],
    psf_mul: float,
    mod_mul: float,
) -> Dict[str, torch.Tensor]:
    """Merge nominal theta with optional trainable params and apply step-wise degradation."""
    th = {k: v for k, v in theta_base.items()}
    if theta_params is not None:
        for k, v in theta_params.items():
            th[k] = v

    base_sigma = th.get("psf_sigma_scale", torch.tensor([1.0], device=list(th.values())[0].device))
    th["psf_sigma_scale"] = base_sigma.reshape(1) * float(psf_mul)

    base_mod = th.get("mod_depth", torch.tensor([1.0], device=list(th.values())[0].device))
    th["mod_depth"] = (base_mod.reshape(-1) * float(mod_mul)).clamp(0.05, 1.0)
    return th


def _apply_theta_grads(theta_params: Dict[str, torch.Tensor], grads: Tuple[Optional[torch.Tensor], ...], lr: float) -> None:
    """Simple SGD update with clamping (no optimizer object required)."""
    with torch.no_grad():
        for (k, p), g in zip(theta_params.items(), grads):
            if g is None:
                continue
            p -= float(lr) * g

            if k == "k_ratio_xy":
                p.clamp_(0.05, 1.5)
            elif k == "kz_ratio":
                p.clamp_(0.0, 1.5)
            elif k == "mod_depth":
                p.clamp_(0.05, 1.0)
            elif k == "background":
                p.clamp_(0.0, 1.0)
            elif k == "phase_offsets":
                p.clamp_(-math.pi, math.pi)
            elif k == "angle_offsets":
                p.clamp_(-10.0, 10.0)
