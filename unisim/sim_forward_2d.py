"""Pure-4D differentiable SIM forward model bound to immutable DMD protocols.

The formal R2 training path uses only tensors shaped ``(B,C,H,W)``.  It does
not call the legacy 5-D forward implementation and contains no 3-D convolution.
Protocol-dependent information is limited to the registry's orientations,
phase labels, raw-frame order, and raw-to-slot mapping; every non-geometric
distribution is shared by K3, K6, and K9 through :class:`SIM2DConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .protocols import KMAX, ProtocolSpec
from .protocol_runtime import require_protocol


RNG = Optional[Union[np.random.Generator, torch.Generator]]

ABERRATION_KEYS = (
    "aberr_defocus",
    "aberr_astig_x",
    "aberr_astig_y",
    "aberr_coma_x",
    "aberr_coma_y",
    "aberr_spherical",
)


@dataclass(frozen=True)
class SIM2DConfig:
    cam_pixel_um: float = 6.5
    magnification: float = 60.0
    wavelength_nm: float = 488.0
    na: float = 1.4
    upsample: int = 2
    k_ratio_xy: float = 0.85
    modulation_depth: float = 0.9
    background: float = 0.02
    photon_scale: float = 8000.0
    read_noise_e: float = 1.6
    psf_sigma_scale: float = 1.0
    psf_size_xy: int = 21
    rand_k_ratio_xy: Tuple[float, float] = (0.75, 0.92)
    rand_mod_depth: Tuple[float, float] = (0.55, 1.0)
    rand_phase_jitter: float = 0.25
    rand_angle_jitter: float = 0.04
    rand_psf_sigma_scale: Tuple[float, float] = (0.85, 1.35)
    rand_background: Tuple[float, float] = (0.0, 0.06)
    rand_photon_scale: Tuple[float, float] = (1500.0, 20000.0)
    rand_read_noise_e: Tuple[float, float] = (1.2, 2.4)

    def __post_init__(self) -> None:
        if self.upsample < 1:
            raise ValueError("upsample must be positive")
        if self.psf_size_xy < 3 or not self.psf_size_xy % 2:
            raise ValueError("psf_size_xy must be an odd integer >= 3")
        for name in (
            "rand_k_ratio_xy",
            "rand_mod_depth",
            "rand_psf_sigma_scale",
            "rand_background",
            "rand_photon_scale",
            "rand_read_noise_e",
        ):
            bounds = getattr(self, name)
            if len(bounds) != 2 or float(bounds[0]) > float(bounds[1]):
                raise ValueError(f"Invalid range {name}={bounds}")


class SIM2DContractError(RuntimeError):
    pass


def _uniform(
    rng: RNG,
    low: float,
    high: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(rng, np.random.Generator):
        value = float(rng.uniform(low, high))
        return torch.tensor([value], device=device, dtype=torch.float32)
    value = torch.rand((1,), device=device, generator=rng)
    return value * (float(high) - float(low)) + float(low)


def _log_uniform(
    rng: RNG,
    low: float,
    high: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if low <= 0:
        raise ValueError("log-uniform lower bound must be positive")
    return torch.exp(_uniform(rng, math.log(low), math.log(high), device=device))


def nominal_theta_2d(cfg: SIM2DConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    # Keep the historical nominal-theta schema unchanged.  Aberration keys are
    # optional extensions supplied only by the Reviewer-1 robustness runner;
    # their absence is the exact zero-aberration path used for training and
    # the frozen nominal measurement bank.
    return {
        "k_ratio_xy": torch.tensor([cfg.k_ratio_xy], device=device),
        "mod_depth": torch.tensor([cfg.modulation_depth], device=device),
        "phase_offsets": torch.zeros((1,), device=device),
        "angle_offsets": torch.zeros((1,), device=device),
        "background": torch.tensor([cfg.background], device=device),
        "psf_sigma_scale": torch.tensor([cfg.psf_sigma_scale], device=device),
        "photon_scale": torch.tensor([cfg.photon_scale], device=device),
        "read_noise_e": torch.tensor([cfg.read_noise_e], device=device),
    }


def sample_theta_2d(
    cfg: SIM2DConfig,
    *,
    device: torch.device,
    mismatch_scale: float = 1.0,
    snr_scale: float = 1.0,
    rng: RNG = None,
) -> Dict[str, torch.Tensor]:
    mismatch = float(np.clip(mismatch_scale, 0.0, 1.0))
    snr = float(np.clip(snr_scale, 0.0, 1.0))
    nominal = nominal_theta_2d(cfg, device)

    def blend(key: str, randomized: torch.Tensor, weight: float) -> torch.Tensor:
        return nominal[key] + (randomized - nominal[key]) * weight

    # The declared distribution is the entire log-uniform interval.  At
    # snr_scale=1 the sample is therefore allowed on either side of nominal
    # (1500..20000 for the formal configuration), while intermediate severity
    # remains a geometric/log-space interpolation from nominal.
    photon_random = _log_uniform(
        rng,
        float(cfg.rand_photon_scale[0]),
        float(cfg.rand_photon_scale[1]),
        device=device,
    )
    photon = torch.exp(
        torch.log(nominal["photon_scale"]) * (1.0 - snr)
        + torch.log(photon_random) * snr
    )
    return {
        "k_ratio_xy": blend(
            "k_ratio_xy", _uniform(rng, *cfg.rand_k_ratio_xy, device=device), mismatch
        ),
        "mod_depth": blend(
            "mod_depth", _uniform(rng, *cfg.rand_mod_depth, device=device), mismatch
        ),
        "phase_offsets": blend(
            "phase_offsets",
            _uniform(rng, -cfg.rand_phase_jitter, cfg.rand_phase_jitter, device=device),
            mismatch,
        ),
        "angle_offsets": blend(
            "angle_offsets",
            _uniform(rng, -cfg.rand_angle_jitter, cfg.rand_angle_jitter, device=device),
            mismatch,
        ),
        "background": blend(
            "background", _uniform(rng, *cfg.rand_background, device=device), mismatch
        ),
        "psf_sigma_scale": blend(
            "psf_sigma_scale",
            _uniform(rng, *cfg.rand_psf_sigma_scale, device=device),
            mismatch,
        ),
        "photon_scale": photon,
        "read_noise_e": blend(
            "read_noise_e", _uniform(rng, *cfg.rand_read_noise_e, device=device), snr
        ),
    }


def _expand_frame_parameter(
    value: torch.Tensor, *, frame_count: int, orientation_count: int, phase_count: int, kind: str
) -> torch.Tensor:
    flat = value.reshape(-1)
    if flat.numel() == 1:
        return flat.expand(frame_count)
    if flat.numel() == frame_count:
        return flat
    if kind == "orientation" and flat.numel() == orientation_count:
        return flat.repeat_interleave(phase_count)
    if kind == "phase" and flat.numel() == phase_count:
        return flat.repeat(orientation_count)
    raise SIM2DContractError(f"Cannot broadcast {kind} parameter of length {flat.numel()}")


def _generator_to_raw_indices(spec: ProtocolSpec) -> Tuple[int, ...]:
    indices = []
    for binding in sorted(spec.raw_frame_bindings, key=lambda value: value.raw_frame_index):
        orientation = spec.orientation_ids.index(binding.physical_orientation_id)
        phase = spec.phase_ids.index(binding.physical_phase_id)
        indices.append(orientation * spec.phases_per_orientation + phase)
    if sorted(indices) != list(range(spec.frame_count)):
        raise SIM2DContractError("Protocol raw frame binding is not bijective")
    return tuple(indices)


def protocol_carrier_unit_vectors_2d(
    protocol: Union[str, ProtocolSpec], *, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Return controller-registered carrier directions after invariant checks."""
    spec = require_protocol(protocol)
    selected_device = device or torch.device("cpu")
    carrier = torch.tensor(spec.carrier_vectors, device=selected_device, dtype=torch.float32)
    carrier_norm = carrier.square().sum(dim=1).sqrt()
    if bool((carrier_norm <= 0).any().item()):
        raise SIM2DContractError(f"{spec.protocol_id} contains a zero carrier vector")
    unit_carrier = carrier / carrier_norm[:, None]
    registered_angles = torch.tensor(
        spec.orientation_angles, device=selected_device, dtype=torch.float32
    )
    carrier_angles = torch.rad2deg(torch.atan2(unit_carrier[:, 1], unit_carrier[:, 0])) % 180.0
    delta = torch.abs((carrier_angles - registered_angles + 90.0) % 180.0 - 90.0)
    if bool((delta > 0.2).any().item()):
        raise SIM2DContractError(
            f"{spec.protocol_id} carrier-vector direction disagrees with registered orientation"
        )
    return unit_carrier


def generate_patterns_2d(
    shape: Tuple[int, int, int, int],
    cfg: SIM2DConfig,
    protocol: Union[str, ProtocolSpec],
    theta: Dict[str, torch.Tensor],
) -> torch.Tensor:
    if len(shape) != 4:
        raise SIM2DContractError(f"Expected (B,C,H,W), got {shape}")
    spec = require_protocol(protocol)
    _batch, _channels, height, width = shape
    device = theta["k_ratio_xy"].device
    dtype = torch.float32
    pixel_um = cfg.cam_pixel_um / cfg.magnification / float(cfg.upsample)
    yy = (torch.arange(height, device=device, dtype=dtype) - height // 2) * pixel_um
    xx = (torch.arange(width, device=device, dtype=dtype) - width // 2) * pixel_um
    y_grid, x_grid = torch.meshgrid(yy, xx, indexing="ij")

    phases = torch.tensor(spec.nominal_phase_values, device=device, dtype=dtype)
    generator_phases = phases.repeat(spec.orientation_count)
    angle_offset = _expand_frame_parameter(
        theta["angle_offsets"],
        frame_count=spec.frame_count,
        orientation_count=spec.orientation_count,
        phase_count=spec.phases_per_orientation,
        kind="orientation",
    )
    phase_offset = _expand_frame_parameter(
        theta["phase_offsets"],
        frame_count=spec.frame_count,
        orientation_count=spec.orientation_count,
        phase_count=spec.phases_per_orientation,
        kind="phase",
    )
    ratio = _expand_frame_parameter(
        theta["k_ratio_xy"],
        frame_count=spec.frame_count,
        orientation_count=spec.orientation_count,
        phase_count=spec.phases_per_orientation,
        kind="orientation",
    )
    modulation = _expand_frame_parameter(
        theta["mod_depth"],
        frame_count=spec.frame_count,
        orientation_count=spec.orientation_count,
        phase_count=spec.phases_per_orientation,
        kind="orientation",
    )
    # Controller carrier magnitudes are DMD-space evidence and are not
    # calibrated to specimen cycles/um.  Consume their immutable direction,
    # verify it against the registered angle, then use only the APD specimen
    # k_ratio for magnitude.  Angle jitter rotates that registered basis.
    unit_carrier = protocol_carrier_unit_vectors_2d(spec, device=device).to(dtype=dtype)
    generator_unit = unit_carrier.repeat_interleave(spec.phases_per_orientation, dim=0)
    angle_radians = angle_offset * (math.pi / 180.0)
    cosine = torch.cos(angle_radians)
    sine = torch.sin(angle_radians)
    unit_x = generator_unit[:, 0] * cosine - generator_unit[:, 1] * sine
    unit_y = generator_unit[:, 0] * sine + generator_unit[:, 1] * cosine
    k_max = cfg.na / max(cfg.wavelength_nm * 1e-3, 1e-9)
    kx = ratio * k_max * unit_x
    ky = ratio * k_max * unit_y
    wave = kx[:, None, None] * x_grid + ky[:, None, None] * y_grid
    generated = 1.0 + modulation[:, None, None] * torch.cos(
        2.0 * math.pi * wave + (generator_phases + phase_offset)[:, None, None]
    )
    order = torch.tensor(_generator_to_raw_indices(spec), device=device, dtype=torch.long)
    return generated.index_select(0, order).unsqueeze(1)


def _zernike_osa_basis_2d(rho: torch.Tensor, phi: torch.Tensor) -> Dict[str, torch.Tensor]:
    """OSA/ANSI-normalized low-order modes used by the legacy robustness study."""
    return {
        "aberr_defocus": math.sqrt(3.0) * (2.0 * rho.square() - 1.0),
        "aberr_astig_x": math.sqrt(6.0) * rho.square() * torch.cos(2.0 * phi),
        "aberr_astig_y": math.sqrt(6.0) * rho.square() * torch.sin(2.0 * phi),
        "aberr_coma_x": math.sqrt(8.0) * (3.0 * rho.pow(3) - 2.0 * rho) * torch.cos(phi),
        "aberr_coma_y": math.sqrt(8.0) * (3.0 * rho.pow(3) - 2.0 * rho) * torch.sin(phi),
        "aberr_spherical": math.sqrt(5.0) * (
            6.0 * rho.pow(4) - 6.0 * rho.square() + 1.0
        ),
    }


def _has_nonzero_aberration_2d(theta: Dict[str, torch.Tensor]) -> bool:
    return any(
        key in theta and bool((theta[key].detach().abs() > 1e-12).any().item())
        for key in ABERRATION_KEYS
    )


def aberrated_psf_2d(
    cfg: SIM2DConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sigma_scale: torch.Tensor,
    theta: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Scalar pupil PSF with aberration coefficients expressed in waves RMS.

    The zero-aberration path deliberately remains :func:`gaussian_psf_2d` so
    historical formal-training measurements are byte-for-byte unaffected.
    ``sigma_scale`` is applied as a diffraction-pattern dilation through a
    differentiable resampling step, matching the role of the Gaussian scale.
    """
    size = int(cfg.psf_size_xy)
    padded = max(2 * size, 128)
    coords = torch.linspace(-1.0, 1.0, padded, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    rho = torch.sqrt(xx.square() + yy.square())
    phi = torch.atan2(yy, xx)
    pupil_mask = (rho <= 1.0).to(torch.float32)
    basis = _zernike_osa_basis_2d(rho, phi)
    phase_waves = torch.zeros_like(rho)
    for key in ABERRATION_KEYS:
        coefficient = theta.get(key)
        if coefficient is not None:
            phase_waves = phase_waves + coefficient.reshape(1).to(phase_waves) * basis[key]
    pupil = pupil_mask.to(torch.complex64) * torch.exp(
        (1j * 2.0 * math.pi * phase_waves * pupil_mask).to(torch.complex64)
    )
    field = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(pupil)))
    intensity = field.abs().square().to(torch.float32)
    center = padded // 2
    half = size // 2
    kernel = intensity[center - half : center - half + size, center - half : center - half + size]
    scale = float(sigma_scale.detach().reshape(-1)[0].cpu().item())
    if not math.isclose(scale, 1.0, rel_tol=0.0, abs_tol=1e-12):
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, size, device=device),
            torch.linspace(-1.0, 1.0, size, device=device),
            indexing="ij",
        )
        grid = torch.stack((grid_x / scale, grid_y / scale), dim=-1).unsqueeze(0)
        kernel = F.grid_sample(
            kernel[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )[0, 0]
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.to(dtype=dtype).view(1, 1, size, size)


def gaussian_psf_2d(
    cfg: SIM2DConfig, *, device: torch.device, dtype: torch.dtype, sigma_scale: torch.Tensor
) -> torch.Tensor:
    pixel_um = cfg.cam_pixel_um / cfg.magnification / float(cfg.upsample)
    coordinate = (
        torch.arange(cfg.psf_size_xy, device=device, dtype=dtype) - cfg.psf_size_xy // 2
    ) * pixel_um
    yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
    sigma = (0.21 * cfg.wavelength_nm * 1e-3 / max(cfg.na, 1e-6)) * sigma_scale.reshape(1)
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma.square()))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, cfg.psf_size_xy, cfg.psf_size_xy)


def forward_protocol_clean_2d(
    x0: torch.Tensor,
    cfg: SIM2DConfig,
    protocol: Union[str, ProtocolSpec],
    *,
    theta: Optional[Dict[str, torch.Tensor]] = None,
    randomize: bool = False,
    rng: RNG = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if x0.ndim != 4 or x0.shape[1] != 1:
        raise SIM2DContractError(f"Object must be (B,1,H,W), got {tuple(x0.shape)}")
    spec = require_protocol(protocol)
    theta_used = theta or (
        sample_theta_2d(cfg, device=x0.device, rng=rng)
        if randomize
        else nominal_theta_2d(cfg, x0.device)
    )
    if cfg.upsample != 1:
        object_up = F.interpolate(
            x0, scale_factor=float(cfg.upsample), mode="bilinear", align_corners=False
        )
    else:
        object_up = x0
    patterns = generate_patterns_2d(tuple(object_up.shape), cfg, spec, theta_used)
    batch, _one, height, width = object_up.shape
    # Multiply a 3-D batch of scalar objects by a 3-D stack of patterns.  The
    # resulting acquisition is directly (B,K,H,W); no five-dimensional
    # intermediate or singleton depth dimension exists in the formal path.
    illuminated = object_up[:, 0].unsqueeze(1) * patterns[:, 0].unsqueeze(0)
    flattened = illuminated.reshape(batch * spec.frame_count, 1, height, width)
    sigma_scale = theta_used["psf_sigma_scale"].to(dtype=x0.dtype)
    if _has_nonzero_aberration_2d(theta_used):
        psf = aberrated_psf_2d(
            cfg,
            device=x0.device,
            dtype=x0.dtype,
            sigma_scale=sigma_scale,
            theta=theta_used,
        )
    else:
        psf = gaussian_psf_2d(
            cfg,
            device=x0.device,
            dtype=x0.dtype,
            sigma_scale=sigma_scale,
        )
    blurred = F.conv2d(flattened, psf, padding=cfg.psf_size_xy // 2)
    raw = blurred.reshape(batch, spec.frame_count, height, width)
    raw = raw + theta_used["background"].to(dtype=raw.dtype).view(1, 1, 1, 1)
    if cfg.upsample != 1:
        raw = F.interpolate(raw, size=x0.shape[-2:], mode="area")
    return raw, theta_used


def forward_protocol_sim_2d(
    x0: torch.Tensor,
    cfg: SIM2DConfig,
    protocol: Union[str, ProtocolSpec],
    *,
    theta: Optional[Dict[str, torch.Tensor]] = None,
    randomize: bool = False,
    rng: RNG = None,
    noise_generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    mean, theta_used = forward_protocol_clean_2d(
        x0, cfg, protocol, theta=theta, randomize=randomize, rng=rng
    )
    photon = theta_used["photon_scale"].to(device=mean.device, dtype=mean.dtype).clamp_min(1e-12)
    read = theta_used["read_noise_e"].to(device=mean.device, dtype=mean.dtype)
    photons = torch.poisson(mean.clamp_min(0.0) * photon.view(1, 1, 1, 1), generator=noise_generator)
    if bool((read > 0).any().item()):
        photons = photons + torch.randn(
            photons.shape,
            device=photons.device,
            dtype=photons.dtype,
            generator=noise_generator,
        ) * read.view(1, 1, 1, 1)
    return photons / photon.view(1, 1, 1, 1), theta_used


def embed_raw_to_slots_2d(
    raw_frames: torch.Tensor, protocol: Union[str, ProtocolSpec]
) -> Tuple[torch.Tensor, torch.Tensor]:
    spec = require_protocol(protocol)
    if raw_frames.ndim != 4 or raw_frames.shape[1] != spec.frame_count:
        raise SIM2DContractError(
            f"{spec.protocol_id} requires (B,{spec.frame_count},H,W), got {tuple(raw_frames.shape)}"
        )
    batch, _frames, height, width = raw_frames.shape
    slotted = raw_frames.new_zeros((batch, KMAX, height, width))
    mask = raw_frames.new_zeros((batch, KMAX, height, width))
    for raw_index, slot_index in enumerate(spec.raw_to_slot_mapping):
        slotted[:, slot_index] = raw_frames[:, raw_index]
        mask[:, slot_index] = 1.0
    observed = tuple(int(v) for v in mask[0, :, 0, 0].tolist())
    if observed != tuple(spec.validity_mask):
        raise SIM2DContractError("Constructed slot mask disagrees with protocol registry")
    return slotted, mask


def extract_slots_to_raw_2d(
    slotted: torch.Tensor, protocol: Union[str, ProtocolSpec]
) -> torch.Tensor:
    spec = require_protocol(protocol)
    if slotted.ndim != 4 or slotted.shape[1] != KMAX:
        raise SIM2DContractError(f"Expected (B,{KMAX},H,W), got {tuple(slotted.shape)}")
    indices = torch.tensor(spec.raw_to_slot_mapping, device=slotted.device, dtype=torch.long)
    return slotted.index_select(1, indices)


def masked_poisson_gaussian_likelihood_2d(
    observed_slotted: torch.Tensor,
    predicted_slotted: torch.Tensor,
    protocol: Union[str, ProtocolSpec],
    *,
    photon_scale: Union[float, torch.Tensor],
    read_noise_e: Union[float, torch.Tensor],
    reduce: str = "mean",
) -> torch.Tensor:
    observed = extract_slots_to_raw_2d(observed_slotted, protocol)
    predicted = extract_slots_to_raw_2d(predicted_slotted, protocol)
    photon = torch.as_tensor(photon_scale, device=observed.device, dtype=observed.dtype).clamp_min(1e-12)
    read = torch.as_tensor(read_noise_e, device=observed.device, dtype=observed.dtype)
    variance = predicted.clamp_min(0.0) / photon + (read / photon).square() + 1e-8
    nll = 0.5 * (torch.log(variance) + (observed - predicted).square() / variance)
    if reduce == "none":
        return nll
    if reduce == "sum":
        return nll.sum()
    if reduce == "mean":
        return nll.mean()
    raise ValueError(f"Unknown reduction {reduce!r}")


__all__ = [
    "ABERRATION_KEYS",
    "SIM2DConfig",
    "SIM2DContractError",
    "aberrated_psf_2d",
    "embed_raw_to_slots_2d",
    "extract_slots_to_raw_2d",
    "forward_protocol_clean_2d",
    "forward_protocol_sim_2d",
    "generate_patterns_2d",
    "masked_poisson_gaussian_likelihood_2d",
    "nominal_theta_2d",
    "protocol_carrier_unit_vectors_2d",
    "sample_theta_2d",
]
