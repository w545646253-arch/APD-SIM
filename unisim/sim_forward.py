# SPDX-License-Identifier: MIT
# Copyright (c) 2024
#
# unisim/sim_forward.py
#
# v3 (physics-stepwise)
# ---------------------
# Key upgrades for high-level SIM diffusion work:
# 1) Differentiable illumination pattern generator (supports self-calibration gradients).
# 2) Correct Poisson-Gaussian NLL in camera units (consistent with forward_sim).
# 3) Fixed theta sampling: mismatch_scale/snr_scale are now *interpolation weights* (0=nominal, 1=full random).
#
# This file is designed to be a drop-in replacement of your previous sim_forward.py.

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, fields
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def read_tiff(path: str) -> np.ndarray:
    """Minimal TIFF reader with graceful fallbacks."""
    try:
        import tifffile  # type: ignore

        return tifffile.imread(path)
    except Exception:
        try:
            import imageio.v3 as iio  # type: ignore

            return iio.imread(path)
        except Exception as e:
            raise ImportError(
                "Failed to read TIFF. Please install 'tifffile' (recommended) or 'imageio'."
            ) from e


def normalize_percentile(
    x: Union[np.ndarray, torch.Tensor],
    lo: float = 1.0,
    hi: float = 99.0,
    eps: float = 1e-8,
) -> Union[np.ndarray, torch.Tensor]:
    """Percentile normalization to [0,1]."""
    if isinstance(x, np.ndarray):
        x_f = x.astype(np.float32, copy=False)
        lo_v = np.percentile(x_f, lo)
        hi_v = np.percentile(x_f, hi)
        y = (x_f - lo_v) / (hi_v - lo_v + eps)
        return np.clip(y, 0.0, 1.0)

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    x_f = x.float()
    flat = x_f.reshape(-1)
    try:
        lo_v = torch.quantile(flat, lo / 100.0)
        hi_v = torch.quantile(flat, hi / 100.0)
    except Exception:
        flat_np = flat.detach().cpu().numpy()
        lo_v = torch.tensor(np.percentile(flat_np, lo), device=x.device, dtype=x_f.dtype)
        hi_v = torch.tensor(np.percentile(flat_np, hi), device=x.device, dtype=x_f.dtype)

    y = (x_f - lo_v) / (hi_v - lo_v + eps)
    return y.clamp(0.0, 1.0)


TensorLike = Union[float, int, torch.Tensor]
RNG = Optional[Union[torch.Generator, np.random.Generator]]


def _device_of(x: Optional[torch.Tensor], fallback: str = "cpu") -> torch.device:
    if x is None:
        return torch.device(fallback)
    return x.device


def _as_tensor(x: TensorLike, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def _to_float(x: Union[float, int, torch.Tensor]) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().reshape(-1)[0].item())
    return float(x)

ABERRATION_KEYS: Tuple[str, ...] = (
    "aberr_defocus",
    "aberr_astig_x",
    "aberr_astig_y",
    "aberr_coma_x",
    "aberr_coma_y",
    "aberr_spherical",
)


def _theta_aberration_dict(
    theta: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Extract scalar aberration coefficients (waves RMS) from theta."""
    out: Dict[str, torch.Tensor] = {}
    if theta is None:
        return out
    for k in ABERRATION_KEYS:
        if k in theta:
            out[k] = _as_tensor(theta[k], device=device, dtype=dtype).reshape(1)
    return out


def _has_nonzero_aberration(aberration: Optional[Dict[str, torch.Tensor]], tol: float = 1e-12) -> bool:
    if not aberration:
        return False
    for v in aberration.values():
        if torch.is_tensor(v):
            if bool((v.detach().abs() > tol).any().item()):
                return True
        else:
            if abs(float(v)) > tol:
                return True
    return False


def _zernike_osa_basis(rho: torch.Tensor, phi: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Low-order OSA/ANSI-normalized Zernike basis on the unit disk."""
    z: Dict[str, torch.Tensor] = {}
    z["aberr_defocus"] = math.sqrt(3.0) * (2.0 * rho**2 - 1.0)
    z["aberr_astig_x"] = math.sqrt(6.0) * rho**2 * torch.cos(2.0 * phi)
    z["aberr_astig_y"] = math.sqrt(6.0) * rho**2 * torch.sin(2.0 * phi)
    z["aberr_coma_x"] = math.sqrt(8.0) * (3.0 * rho**3 - 2.0 * rho) * torch.cos(phi)
    z["aberr_coma_y"] = math.sqrt(8.0) * (3.0 * rho**3 - 2.0 * rho) * torch.sin(phi)
    z["aberr_spherical"] = math.sqrt(5.0) * (6.0 * rho**4 - 6.0 * rho**2 + 1.0)
    return z


def _make_aberrated_scalar_psf(
    cfg: "SIMConfig",
    device: torch.device,
    aberration: Dict[str, torch.Tensor],
    sigma_scale: Union[float, torch.Tensor] = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a scalar pupil-based 3D PSF with lateral aberration and Gaussian axial envelope.

    Aberration coefficients are expressed in waves RMS and applied through a 2D pupil phase.
    For 2D-SIM (Z=1) this directly gives an aberrated lateral PSF. For Z>1, the lateral PSF
    is combined with a Gaussian axial envelope to preserve backward compatibility with the
    existing 3D convolution API.
    """
    dtype = torch.float32
    sigma_scale_t = _as_tensor(sigma_scale, device=device, dtype=dtype).reshape(1)

    Nxy = int(cfg.psf_size_xy)
    Nz = int(cfg.psf_size_z)
    # Use a padded pupil grid for a cleaner crop of the diffraction pattern.
    Np = int(max(2 * Nxy, 128))

    coords = torch.linspace(-1.0, 1.0, Np, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rho = torch.sqrt(xx**2 + yy**2)
    phi = torch.atan2(yy, xx)
    pupil_mask = (rho <= 1.0).to(dtype=dtype)

    basis = _zernike_osa_basis(rho, phi)
    phase_waves = torch.zeros_like(rho)
    for k in ABERRATION_KEYS:
        if k in aberration:
            coeff = _as_tensor(aberration[k], device=device, dtype=dtype).reshape(1)
            phase_waves = phase_waves + coeff * basis[k]
    phase_waves = phase_waves * pupil_mask

    pupil = pupil_mask.to(torch.complex64) * torch.exp(1j * (2.0 * math.pi * phase_waves).to(torch.complex64))
    field = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(pupil)))
    psf_xy_full = (field.abs() ** 2).float()
    psf_xy_full = psf_xy_full / (psf_xy_full.sum() + 1e-12)

    # Center crop to cfg.psf_size_xy.
    cy = Np // 2
    cx = Np // 2
    hy = Nxy // 2
    hx = Nxy // 2
    psf_xy = psf_xy_full[cy - hy: cy - hy + Nxy, cx - hx: cx - hx + Nxy]
    psf_xy = psf_xy / (psf_xy.sum() + 1e-12)

    # Axial envelope (kept Gaussian for backward compatibility with the existing 3D conv API).
    wavelength_um = float(cfg.wavelength_nm) * 1e-3
    sigma_xy0 = 0.21 * wavelength_um / max(cfg.na, 1e-6)
    sigma_z = 2.0 * sigma_xy0 * sigma_scale_t
    z = (torch.arange(Nz, device=device, dtype=dtype) - Nz // 2) * float(cfg.z_step_um)
    w_z = torch.exp(-(z**2) / (2.0 * sigma_z**2))
    w_z = w_z / (w_z.sum() + 1e-12)

    psf = w_z[:, None, None] * psf_xy[None, :, :]
    psf = psf / (psf.sum() + 1e-12)

    otf = torch.fft.fftn(torch.fft.ifftshift(psf, dim=(-3, -2, -1)), dim=(-3, -2, -1))
    otf = torch.fft.fftshift(otf, dim=(-3, -2, -1))
    return psf.unsqueeze(0).unsqueeze(0), otf.unsqueeze(0).unsqueeze(0)


def rand_uniform(
    rng: RNG,
    low: float,
    high: float,
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Uniform random on [low, high]. Supports torch or numpy rng."""
    if high < low:
        raise ValueError(f"rand_uniform expects high>=low, got low={low}, high={high}")
    if isinstance(rng, np.random.Generator):
        u = rng.random(shape, dtype=np.float32)
        u = torch.from_numpy(u).to(device=device, dtype=dtype)
    else:
        u = torch.rand(shape, device=device, generator=rng, dtype=dtype)
    return low + (high - low) * u


def rand_log_uniform(
    rng: RNG,
    low: float,
    high: float,
    shape: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Log-uniform random on [low, high]."""
    low = max(float(low), eps)
    high = max(float(high), low + eps)
    log_low = math.log(low)
    log_high = math.log(high)
    u = rand_uniform(rng, log_low, log_high, shape=shape, device=device, dtype=dtype)
    return torch.exp(u)


@dataclass(init=False)
class SIMConfig:
    # ---- runtime / data shape helpers (kept for backward-compat; not all are used by this module) ----
    device: str = "cuda"
    grid_size: int = 256
    z_slices: int = 1
    # --------------------------------------------------------------------------

    cam_pixel_um: float = 6.5
    magnification: float = 60.0
    wavelength_nm: float = 488.0
    na: float = 1.4
    refractive_index: float = 1.518
    z_step_um: float = 0.125

    # internal oversampling in forward model
    upsample: int = 2

    # pattern geometry
    angle_list: Tuple[float, ...] = (0.0, 60.0, 120.0)  # degrees
    phase_list_2d: Tuple[float, ...] = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    phase_list_3d: Tuple[float, ...] = (
        0.0,
        2.0 * math.pi / 5.0,
        4.0 * math.pi / 5.0,
        6.0 * math.pi / 5.0,
        8.0 * math.pi / 5.0,
    )

    k_ratio_xy: float = 0.25
    kz_ratio: float = 0.0
    modulation_depth: float = 0.8

    # PSF
    psf_type: str = "gaussian"  # "gaussian" or "tiff"
    psf_path: Optional[str] = None
    psf_size_xy: int = 129
    psf_size_z: int = 33
    psf_sigma_scale: float = 1.0

    # randomization ranges
    rand_k_ratio_xy: Tuple[float, float] = (0.2, 0.35)
    rand_kz_ratio: Optional[Tuple[float, float]] = None
    rand_mod_depth: Tuple[float, float] = (0.2, 1.0)
    rand_modulation_depth: Optional[Tuple[float, float]] = None  # alias

    rand_phase_jitter: float = 0.0  # radians
    rand_angle_jitter: float = 0.0  # degrees
    rand_background: Tuple[float, float] = (0.0, 0.0)
    rand_psf_sigma_scale: Tuple[float, float] = (1.0, 1.0)

    # noise model
    photon_scale: float = 1.0
    read_noise_e: float = 0.0
    rand_photon_scale: Tuple[float, float] = (1.0, 1.0)
    rand_read_noise_e: Tuple[float, float] = (0.0, 0.0)

    background: float = 0.0
    use_theta_grad: bool = False

    def __init__(self, *args, **kwargs):
        """Compatibility-oriented __init__ that ignores unknown kwargs."""
        flds = [f for f in fields(self.__class__)]
        if args:
            if len(args) > len(flds):
                raise TypeError(f"SIMConfig expected at most {len(flds)} positional args, got {len(args)}")
            for f, v in zip(flds, args):
                setattr(self, f.name, v)

        remaining = dict(kwargs)
        for f in flds:
            if hasattr(self, f.name):
                continue
            if f.name in remaining:
                setattr(self, f.name, remaining.pop(f.name))
            else:
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                else:
                    setattr(self, f.name, f.default_factory())

        self.extra_kwargs = remaining
        for k, v in remaining.items():
            setattr(self, k, v)

        self.__post_init__()

    def __post_init__(self):
        # alias support
        if getattr(self, "rand_modulation_depth", None) is not None:
            v = self.rand_modulation_depth
            if isinstance(v, (list, tuple)) and len(v) == 2:
                self.rand_mod_depth = (float(v[0]), float(v[1]))
            else:
                self.rand_mod_depth = (float(v), float(v))

        if self.rand_kz_ratio is not None:
            if not (isinstance(self.rand_kz_ratio, (list, tuple)) and len(self.rand_kz_ratio) == 2):
                raise ValueError("rand_kz_ratio must be a tuple/list of length 2 or None.")
            self.rand_kz_ratio = (float(self.rand_kz_ratio[0]), float(self.rand_kz_ratio[1]))

        if not isinstance(self.upsample, int) or self.upsample < 1:
            raise ValueError(f"upsample must be a positive int, got {self.upsample}")


def make_psf_otf(
    cfg: SIMConfig,
    device: torch.device,
    sigma_scale: Union[float, torch.Tensor] = 1.0,
    aberration: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct PSF/OTF.

    Default path: Gaussian PSF (backward compatible).
    Aberrated path: scalar pupil-based lateral PSF with low-order Zernike phase aberrations.
    """
    if _has_nonzero_aberration(aberration):
        return _make_aberrated_scalar_psf(cfg, device=device, aberration=aberration or {}, sigma_scale=sigma_scale)

    sigma_scale_t = _as_tensor(sigma_scale, device=device, dtype=torch.float32).reshape(1)

    # Spatial sampling in object space
    px_um = cfg.cam_pixel_um / cfg.magnification / float(cfg.upsample)

    z = (torch.arange(cfg.psf_size_z, device=device) - cfg.psf_size_z // 2) * cfg.z_step_um
    y = (torch.arange(cfg.psf_size_xy, device=device) - cfg.psf_size_xy // 2) * px_um
    x = (torch.arange(cfg.psf_size_xy, device=device) - cfg.psf_size_xy // 2) * px_um
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    # Heuristic sigma values (keeps module self-contained)
    sigma_xy0 = 0.21 * cfg.wavelength_nm * 1e-3 / max(cfg.na, 1e-6)  # um
    sigma_xy = sigma_xy0 * sigma_scale_t
    sigma_z = 2.0 * sigma_xy

    psf = torch.exp(-(xx**2 + yy**2) / (2 * sigma_xy**2) - (zz**2) / (2 * sigma_z**2))
    psf = psf / (psf.sum() + 1e-12)

    otf = torch.fft.fftn(torch.fft.ifftshift(psf, dim=(-3, -2, -1)), dim=(-3, -2, -1))
    otf = torch.fft.fftshift(otf, dim=(-3, -2, -1))

    return psf.unsqueeze(0).unsqueeze(0), otf.unsqueeze(0).unsqueeze(0)


def build_psf(
    cfg: SIMConfig,
    device: torch.device,
    sigma_scale: Union[float, torch.Tensor] = 1.0,
    theta: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build PSF/OTF from cfg and optional theta aberration coefficients."""
    if cfg.psf_type.lower() == "tiff":
        if cfg.psf_path is None:
            raise ValueError("psf_type='tiff' requires psf_path to be set.")
        psf_np = read_tiff(cfg.psf_path).astype(np.float32, copy=False)
        psf = torch.from_numpy(psf_np).to(device=device)
        psf = normalize_percentile(psf, 1, 99)
        psf = psf / (psf.sum() + 1e-12)

        # ensure [1,1,Z,H,W]
        if psf.ndim == 2:
            psf = psf.unsqueeze(0)
        if psf.ndim == 3:
            psf = psf.unsqueeze(0)
        if psf.ndim == 3:
            psf = psf.unsqueeze(0).unsqueeze(0)
        if psf.ndim != 5:
            raise ValueError(f"Unexpected PSF shape from TIFF: {tuple(psf.shape)}")
        psf = psf[:1, :1, ...]

        otf = torch.fft.fftn(torch.fft.ifftshift(psf.squeeze(0).squeeze(0), dim=(-3, -2, -1)), dim=(-3, -2, -1))
        otf = torch.fft.fftshift(otf, dim=(-3, -2, -1))
        otf = otf.unsqueeze(0).unsqueeze(0)
        return psf, otf

    aberration = _theta_aberration_dict(theta, device=device, dtype=torch.float32)
    return make_psf_otf(cfg, device=device, sigma_scale=sigma_scale, aberration=aberration)


def theta_nominal(cfg: SIMConfig, mode: str, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Nominal theta dictionary (no randomness)."""
    mode_l = mode.lower()
    if device is None:
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else torch.device("cpu")

    k_ratio_xy = torch.tensor([cfg.k_ratio_xy], device=device, dtype=torch.float32)
    kz_ratio = torch.tensor([cfg.kz_ratio if mode_l == "3d" else 0.0], device=device, dtype=torch.float32)
    mod_depth = torch.tensor([cfg.modulation_depth], device=device, dtype=torch.float32)
    background = torch.tensor([cfg.background], device=device, dtype=torch.float32)
    psf_sigma_scale = torch.tensor([cfg.psf_sigma_scale], device=device, dtype=torch.float32)
    photon_scale = torch.tensor([cfg.photon_scale], device=device, dtype=torch.float32)
    read_noise_e = torch.tensor([cfg.read_noise_e], device=device, dtype=torch.float32)

    return {
        "k_ratio_xy": k_ratio_xy,
        "kz_ratio": kz_ratio,
        "mod_depth": mod_depth,
        "phase_offsets": torch.zeros(1, device=device, dtype=torch.float32),
        "angle_offsets": torch.zeros(1, device=device, dtype=torch.float32),
        "background": background,
        "psf_sigma_scale": psf_sigma_scale,
        "photon_scale": photon_scale,
        "read_noise_e": read_noise_e,
        "aberr_defocus": torch.zeros(1, device=device, dtype=torch.float32),
        "aberr_astig_x": torch.zeros(1, device=device, dtype=torch.float32),
        "aberr_astig_y": torch.zeros(1, device=device, dtype=torch.float32),
        "aberr_coma_x": torch.zeros(1, device=device, dtype=torch.float32),
        "aberr_coma_y": torch.zeros(1, device=device, dtype=torch.float32),
        "aberr_spherical": torch.zeros(1, device=device, dtype=torch.float32),
    }


def _blend(nom: torch.Tensor, rnd: torch.Tensor, w: float) -> torch.Tensor:
    """Linear blend: w=0 -> nominal, w=1 -> rnd."""
    w_t = float(max(0.0, min(1.0, w)))
    return nom + (rnd - nom) * w_t


def sample_theta(
    cfg: SIMConfig,
    mode: str,
    device: Optional[torch.device] = None,
    mismatch_scale: float = 1.0,
    snr_scale: float = 1.0,
    rng: RNG = None,
    # optional overrides
    photon_scale: Optional[float] = None,
    log_uniform_photon: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Sample theta parameters.

    IMPORTANT (fixed):
      - mismatch_scale and snr_scale are *interpolation weights* in [0,1]:
          0 -> nominal/matched
          1 -> fully randomized (within cfg.rand_* ranges)
      - This is the correct interpretation for staged training schedules.
    """
    if device is None:
        device = torch.device(cfg.device) if isinstance(cfg.device, str) else torch.device("cpu")

    mode_l = mode.lower()
    if mode_l not in ("2d", "3d"):
        raise ValueError(f"mode must be '2d' or '3d', got: {mode}")

    # clamp scales
    mismatch_w = float(np.clip(mismatch_scale, 0.0, 1.0))
    snr_w = float(np.clip(snr_scale, 0.0, 1.0))

    nom = theta_nominal(cfg, mode=mode_l, device=device)

    # ---- geometry & illumination ----
    k_ratio_xy_r = rand_uniform(rng, cfg.rand_k_ratio_xy[0], cfg.rand_k_ratio_xy[1], shape=(1,), device=device)

    if mode_l == "3d":
        if cfg.rand_kz_ratio is None:
            kz_ratio_r = torch.tensor([cfg.kz_ratio], device=device, dtype=torch.float32)
        else:
            kz_ratio_r = rand_uniform(rng, cfg.rand_kz_ratio[0], cfg.rand_kz_ratio[1], shape=(1,), device=device)
    else:
        kz_ratio_r = torch.tensor([0.0], device=device, dtype=torch.float32)

    mod_depth_r = rand_uniform(rng, cfg.rand_mod_depth[0], cfg.rand_mod_depth[1], shape=(1,), device=device)

    # offsets (nominal is 0)
    phase_off_r = rand_uniform(rng, -cfg.rand_phase_jitter, cfg.rand_phase_jitter, shape=(1,), device=device)
    angle_off_r = rand_uniform(rng, -cfg.rand_angle_jitter, cfg.rand_angle_jitter, shape=(1,), device=device)

    bg_r = rand_uniform(rng, cfg.rand_background[0], cfg.rand_background[1], shape=(1,), device=device)
    psf_sigma_r = rand_uniform(rng, cfg.rand_psf_sigma_scale[0], cfg.rand_psf_sigma_scale[1], shape=(1,), device=device)

    # blend by mismatch
    k_ratio_xy = _blend(nom["k_ratio_xy"], k_ratio_xy_r, mismatch_w)
    kz_ratio = _blend(nom["kz_ratio"], kz_ratio_r, mismatch_w)
    mod_depth = _blend(nom["mod_depth"], mod_depth_r, mismatch_w)
    phase_offsets = _blend(nom["phase_offsets"], phase_off_r, mismatch_w)
    angle_offsets = _blend(nom["angle_offsets"], angle_off_r, mismatch_w)
    background = _blend(nom["background"], bg_r, mismatch_w)
    psf_sigma_scale = _blend(nom["psf_sigma_scale"], psf_sigma_r, mismatch_w)

    # ---- noise (SNR degradation) ----
    # In forward_sim, larger photon_scale -> higher SNR; larger read_noise_e -> lower SNR.
    ps_lo, ps_hi = float(min(cfg.rand_photon_scale)), float(max(cfg.rand_photon_scale))
    rn_lo, rn_hi = float(min(cfg.rand_read_noise_e)), float(max(cfg.rand_read_noise_e))

    photon_nom = torch.tensor([float(cfg.photon_scale if photon_scale is None else photon_scale)], device=device, dtype=torch.float32)
    photon_nom = photon_nom.clamp(min=ps_lo, max=ps_hi)

    # "hard" photon scale is sampled in [ps_lo, photon_nom] (lower photons => harder)
    hard_ps_hi = float(max(ps_lo, float(photon_nom.item())))
    if hard_ps_hi <= ps_lo + 1e-12:
        photon_hard = torch.tensor([ps_lo], device=device, dtype=torch.float32)
    else:
        photon_hard = rand_log_uniform(rng, ps_lo, hard_ps_hi, shape=(1,), device=device) if log_uniform_photon else rand_uniform(rng, ps_lo, hard_ps_hi, shape=(1,), device=device)

    # "hard" read noise is sampled in [read_nom, rn_hi]
    read_nom = torch.tensor([float(cfg.read_noise_e)], device=device, dtype=torch.float32).clamp(min=rn_lo, max=rn_hi)
    hard_rn_lo = float(min(rn_hi, float(read_nom.item())))
    if rn_hi <= hard_rn_lo + 1e-12:
        read_hard = torch.tensor([rn_hi], device=device, dtype=torch.float32)
    else:
        read_hard = rand_uniform(rng, hard_rn_lo, rn_hi, shape=(1,), device=device)

    # blend in a monotonic way
    # photon: log-blend to respect multiplicative nature
    photon_scale_t = torch.exp(torch.log(photon_nom + 1e-12) * (1 - snr_w) + torch.log(photon_hard + 1e-12) * snr_w)
    # read noise: linear blend
    read_noise_t = read_nom * (1 - snr_w) + read_hard * snr_w

    theta = {
        "k_ratio_xy": k_ratio_xy,
        "kz_ratio": kz_ratio,
        "mod_depth": mod_depth,
        "phase_offsets": phase_offsets,
        "angle_offsets": angle_offsets,
        "background": background,
        "psf_sigma_scale": psf_sigma_scale,
        "photon_scale": photon_scale_t,
        "read_noise_e": read_noise_t,
    }

    if cfg.use_theta_grad:
        for k, v in theta.items():
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                theta[k] = v.clone()

    return theta


def _expand_param(
    v: torch.Tensor,
    K: int,
    n_angles: int,
    n_phases: int,
    kind: str,
) -> torch.Tensor:
    """Broadcast scalar / per-angle / per-phase / per-frame parameter to (K,)."""
    v = v.reshape(-1)
    if v.numel() == 1:
        return v.expand(K)
    if v.numel() == K:
        return v
    if kind == "angle" and v.numel() == n_angles:
        return v.repeat_interleave(n_phases)
    if kind == "phase" and v.numel() == n_phases:
        return v.repeat(n_angles)
    raise ValueError(f"Cannot broadcast {kind} parameter with numel={v.numel()} to K={K} (n_angles={n_angles}, n_phases={n_phases}).")


def generate_patterns(
    x_shape: Tuple[int, int, int, int, int],
    cfg: SIMConfig,
    mode: str,
    theta: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Differentiable illumination pattern generator.

    x_shape: (B,C,Z,H,W) of object-space tensor (after internal upsample in forward_clean).
    Returns: (K,1,Z,H,W)

    Notes:
      - theta["angle_offsets"] is in *degrees* (consistent with cfg.rand_angle_jitter).
      - theta["phase_offsets"] is in *radians*.
      - Supports scalar offsets or per-angle/per-phase/per-frame vectors.
    """
    mode_l = mode.lower()
    if mode_l not in ("2d", "3d"):
        raise ValueError(f"mode must be '2d' or '3d', got: {mode}")

    _B, _C, Z, H, W = x_shape
    device = _device_of(theta.get("k_ratio_xy", None), fallback=cfg.device)
    dtype = torch.float32

    # grids in object space (um)
    px_um = cfg.cam_pixel_um / cfg.magnification / float(cfg.upsample)
    yy = (torch.arange(H, device=device, dtype=dtype) - H // 2) * float(px_um)
    xx = (torch.arange(W, device=device, dtype=dtype) - W // 2) * float(px_um)
    y_grid, x_grid = torch.meshgrid(yy, xx, indexing="ij")  # (H,W)

    angles0_deg = torch.tensor(list(cfg.angle_list), device=device, dtype=dtype)  # (nA,)
    phases0 = torch.tensor(list(cfg.phase_list_2d if mode_l == "2d" else cfg.phase_list_3d), device=device, dtype=dtype)  # (nP,)
    nA = int(angles0_deg.numel())
    nP = int(phases0.numel())
    K = nA * nP

    # frame base angles/phases
    angles_deg = angles0_deg.repeat_interleave(nP)  # (K,)
    phases = phases0.repeat(nA)  # (K,)

    # offsets (broadcast to per-frame)
    angle_off = _as_tensor(theta.get("angle_offsets", 0.0), device=device, dtype=dtype)
    phase_off = _as_tensor(theta.get("phase_offsets", 0.0), device=device, dtype=dtype)
    angle_off_f = _expand_param(angle_off, K=K, n_angles=nA, n_phases=nP, kind="angle")
    phase_off_f = _expand_param(phase_off, K=K, n_angles=nA, n_phases=nP, kind="phase")

    angles_rad = (angles_deg + angle_off_f) * (math.pi / 180.0)  # (K,)
    phi = phases + phase_off_f  # (K,)

    # k ratios / modulation depth (broadcast to per-frame)
    k_ratio_xy = _expand_param(_as_tensor(theta.get("k_ratio_xy", cfg.k_ratio_xy), device=device, dtype=dtype), K=K, n_angles=nA, n_phases=nP, kind="angle")
    kz_ratio = _expand_param(_as_tensor(theta.get("kz_ratio", 0.0), device=device, dtype=dtype), K=K, n_angles=nA, n_phases=nP, kind="angle")
    mod_depth = _expand_param(_as_tensor(theta.get("mod_depth", cfg.modulation_depth), device=device, dtype=dtype), K=K, n_angles=nA, n_phases=nP, kind="angle")

    # kmax (cycles/um)
    wavelength_um = float(cfg.wavelength_nm) * 1e-3
    kmax = torch.tensor(float(cfg.na) / max(wavelength_um, 1e-9), device=device, dtype=dtype)
    kxy = k_ratio_xy * kmax
    kz = kz_ratio * kmax

    # per-frame kx, ky
    kx = kxy * torch.cos(angles_rad)
    ky = kxy * torch.sin(angles_rad)

    # wave in cycles
    wave_xy = kx[:, None, None] * x_grid[None, :, :] + ky[:, None, None] * y_grid[None, :, :]  # (K,H,W)

    tau = 2.0 * math.pi
    if mode_l == "2d":
        patt_hw = 1.0 + mod_depth[:, None, None] * torch.cos(tau * wave_xy + phi[:, None, None])  # (K,H,W)
        patt = patt_hw[:, None, None, :, :].expand(K, 1, Z, H, W)
        return patt

    # 3D: add axial modulation term
    zz = (torch.arange(Z, device=device, dtype=dtype) - Z // 2) * float(cfg.z_step_um)  # (Z,)
    wave_z = kz[:, None] * zz[None, :]  # (K,Z)
    total = wave_xy[:, None, :, :] + wave_z[:, :, None, None]  # (K,Z,H,W)
    patt_zyx = 1.0 + mod_depth[:, None, None, None] * torch.cos(tau * total + phi[:, None, None, None])
    return patt_zyx[:, None, :, :, :]


def apply_psf(x: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
    """Apply PSF via 3D convolution."""
    kz, ky, kx = psf.shape[-3], psf.shape[-2], psf.shape[-1]
    pad_z = kz // 2
    pad_y = ky // 2
    pad_x = kx // 2
    x_pad = F.pad(x, (pad_x, pad_x, pad_y, pad_y, pad_z, pad_z))
    return F.conv3d(x_pad, psf)


def downsample_xy(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample XY by integer factor using area interpolation."""
    if factor == 1:
        return x
    if factor < 1:
        raise ValueError(f"downsample factor must be >=1, got {factor}")
    B, C, Z, H, W = x.shape
    x2 = x.permute(0, 2, 1, 3, 4).reshape(B * Z, C, H, W)
    y2 = F.interpolate(x2, size=(H // factor, W // factor), mode="area")
    y = y2.reshape(B, Z, C, H // factor, W // factor).permute(0, 2, 1, 3, 4).contiguous()
    return y


def nll_poisson_gaussian(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Legacy Poisson-Gaussian NLL (sigma-parameterized)."""
    var = sigma**2 + mu.clamp_min(0.0) + eps
    return 0.5 * (torch.log(var) + (y - mu) ** 2 / var)


def nll_poisson_gaussian_cam(
    y: torch.Tensor,
    mu: torch.Tensor,
    photon_scale: Union[float, torch.Tensor],
    read_noise_e: Union[float, torch.Tensor],
    eps: float = 1e-8,
    reduce: str = "mean",
) -> torch.Tensor:
    """
    Correct Gaussian-approx NLL for the noise model implemented in forward_sim.

    forward_sim:
      mu_ph = mu * photon_scale
      y_ph ~ Poisson(mu_ph) + N(0, read_noise_e^2)
      y = y_ph / photon_scale

    => Var[y] ≈ mu/photon_scale + (read_noise_e/photon_scale)^2

    Args:
      reduce: "mean" | "sum" | "none"
    """
    ps = _as_tensor(photon_scale, device=y.device, dtype=y.dtype)
    rn = _as_tensor(read_noise_e, device=y.device, dtype=y.dtype)

    ps = ps.clamp_min(1e-12)
    mu_pos = mu.clamp_min(0.0)
    var = mu_pos / ps + (rn / ps) ** 2 + eps
    nll = 0.5 * (torch.log(var) + (y - mu) ** 2 / var)

    if reduce == "none":
        return nll
    if reduce == "sum":
        return nll.sum()
    if reduce == "mean":
        return nll.mean()
    raise ValueError(f"Unknown reduce: {reduce}")


def forward_clean(
    x0: torch.Tensor,
    cfg: SIMConfig,
    mode: str,
    theta: Optional[Dict[str, torch.Tensor]] = None,
    randomize: bool = False,
    rng: RNG = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Forward SIM measurement without noise."""
    device = x0.device
    mode_l = mode.lower()
    if mode_l not in ("2d", "3d"):
        raise ValueError(f"mode must be '2d' or '3d', got: {mode}")

    if theta is None:
        theta_used = sample_theta(cfg, mode=mode_l, device=device, mismatch_scale=1.0 if randomize else 0.0, snr_scale=1.0 if randomize else 0.0, rng=rng)
    else:
        theta_used = theta

    # PSF depends on sigma scale
    sigma_scale = _to_float(theta_used.get("psf_sigma_scale", cfg.psf_sigma_scale))
    psf, _otf = build_psf(cfg, device=device, sigma_scale=sigma_scale, theta=theta_used)

    # internal upsample of x0 in XY
    if cfg.upsample != 1:
        B, C, Z, H, W = x0.shape
        x2 = x0.permute(0, 2, 1, 3, 4).reshape(B * Z, C, H, W)
        x2u = F.interpolate(x2, scale_factor=float(cfg.upsample), mode="bilinear", align_corners=False)
        Hu, Wu = x2u.shape[-2], x2u.shape[-1]
        x0u = x2u.reshape(B, Z, C, Hu, Wu).permute(0, 2, 1, 3, 4).contiguous()
    else:
        x0u = x0

    # single-channel object
    x_obj = x0u.sum(dim=1, keepdim=True) if x0u.shape[1] != 1 else x0u

    patterns = generate_patterns(x_shape=x_obj.shape, cfg=cfg, mode=mode_l, theta=theta_used)  # (K,1,Z,Hu,Wu)

    B = x_obj.shape[0]
    K = patterns.shape[0]

    x_illum = x_obj.unsqueeze(1) * patterns.unsqueeze(0)  # (B,K,1,Z,Hu,Wu)
    x_illum_reshape = x_illum.reshape(B * K, 1, x_obj.shape[2], x_obj.shape[3], x_obj.shape[4])
    y_blur = apply_psf(x_illum_reshape, psf)
    y_blur = y_blur.reshape(B, K, 1, x_obj.shape[2], x_obj.shape[3], x_obj.shape[4])

    # background (photon domain before downsample)
    bg = _as_tensor(theta_used.get("background", cfg.background), device=device, dtype=y_blur.dtype).reshape(1)
    y_blur = y_blur + bg.view(1, 1, 1, 1, 1, 1)

    y_blur = y_blur.squeeze(2)  # (B,K,Z,Hu,Wu)
    y_cam = downsample_xy(y_blur, factor=cfg.upsample)  # (B,K,Z,Hc,Wc)
    return y_cam, theta_used


def forward_sim(
    x0: torch.Tensor,
    cfg: SIMConfig,
    mode: str,
    theta: Optional[Dict[str, torch.Tensor]] = None,
    randomize: bool = False,
    rng: RNG = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Full SIM forward with Poisson + Gaussian read noise."""
    device = x0.device
    mu, theta_used = forward_clean(x0, cfg, mode=mode, theta=theta, randomize=randomize, rng=rng)

    photon_scale = _as_tensor(theta_used.get("photon_scale", cfg.photon_scale), device=device, dtype=mu.dtype).reshape(1)
    read_noise_e = _as_tensor(theta_used.get("read_noise_e", cfg.read_noise_e), device=device, dtype=mu.dtype).reshape(1)
    photon_scale = photon_scale.clamp_min(1e-12)

    mu_ph = mu * photon_scale.view(1, 1, 1, 1, 1)
    y_ph = torch.poisson(mu_ph.clamp_min(0.0))
    if float(read_noise_e.item()) > 0.0:
        y_ph = y_ph + torch.randn_like(y_ph) * read_noise_e.view(1, 1, 1, 1, 1)

    # back to camera units
    y = y_ph / photon_scale.view(1, 1, 1, 1, 1)
    return y, theta_used
