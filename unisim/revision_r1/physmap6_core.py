"""Single shared refinement core for strict DMD six-frame comparisons.

This module has no dataset, ground-truth, diffusion-model, or checkpoint
dependency.  ``masked_refine`` is therefore a per-sample physics-only masked
likelihood optimizer.  PhysMap-6 and APD-SIM-6 Stage 2 must call this exact
function with the same :class:`RefinementConfig`; their sole permitted
difference is ``initial_image``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping

import torch

from unisim.sim_forward_2d import (
    SIM2DConfig,
    embed_raw_to_slots_2d,
    forward_protocol_clean_2d,
    masked_poisson_gaussian_likelihood_2d,
)
from unisim.protocol_runtime import require_protocol


@dataclass(frozen=True)
class RefinementConfig:
    optimizer: str = "torch.optim.Adam"
    learning_rate: float = 0.005
    updates: int = 40
    lambda_prior: float = 0.0
    clip_min: float = 0.0
    clip_max: float = 1.0
    dtype: str = "float32"
    stopping_rule: str = "exactly_40_updates_no_early_stopping"
    objective: str = "masked_poisson_gaussian_camera_nll"
    nrmse_denominator_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.optimizer != "torch.optim.Adam":
            raise ValueError("The frozen APD Stage 2 optimizer must be Adam")
        if not math.isclose(self.learning_rate, 0.005, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("The frozen APD Stage 2 learning rate must be exactly 0.005")
        if self.updates != 40:
            raise ValueError("The frozen APD Stage 2 requires exactly 40 updates")
        if self.lambda_prior != 0.0:
            raise ValueError("Reviewer #1 strict main analysis requires lambda_prior=0")
        if self.dtype != "float32" or self.stopping_rule != "exactly_40_updates_no_early_stopping":
            raise ValueError("Strict numerical precision/stopping contract drift")
        if self.clip_min != 0.0 or self.clip_max != 1.0:
            raise ValueError("The frozen APD Stage 2 clipping interval must be exactly [0,1]")

    def receipt(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["method_description"] = "physics-only masked-likelihood optimization"
        payload["config_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload


@dataclass
class RefineResult:
    final_reconstruction: torch.Tensor
    objective_history: list[float]
    observed_nrmse_history: list[float]
    gradient_finite: bool
    output_finite: bool
    runtime_seconds: float
    peak_gpu_memory_bytes: int
    configuration_receipt: dict[str, Any]


@dataclass(frozen=True)
class ObservedFit:
    objective: float
    observed_nrmse: float
    finite: bool


def _theta_receipt(theta: Mapping[str, Any]) -> dict[str, list[float] | Any]:
    receipt: dict[str, list[float] | Any] = {}
    for key in sorted(theta):
        value = theta[key]
        receipt[key] = (
            [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
            if torch.is_tensor(value)
            else value
        )
    return receipt


def _diagnostics(
    estimate: torch.Tensor,
    observed_slotted: torch.Tensor,
    sim_config: SIM2DConfig,
    protocol_id: str,
    theta: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_raw, _ = forward_protocol_clean_2d(
        estimate, sim_config, protocol_id, theta=dict(theta), randomize=False
    )
    predicted_slotted, _ = embed_raw_to_slots_2d(predicted_raw, protocol_id)
    objective = masked_poisson_gaussian_likelihood_2d(
        observed_slotted,
        predicted_slotted,
        protocol_id,
        photon_scale=theta["photon_scale"],
        read_noise_e=theta["read_noise_e"],
        reduce="mean",
    )
    spec = require_protocol(protocol_id)
    slot_indices = torch.tensor(spec.raw_to_slot_mapping, device=observed_slotted.device)
    observed_raw = observed_slotted.index_select(1, slot_indices)
    numerator = torch.mean((observed_raw - predicted_raw).square()).sqrt()
    denominator = torch.mean(observed_raw.square()).sqrt().clamp_min(1e-12)
    return objective, numerator / denominator


@torch.no_grad()
def evaluate_observed_fit(
    reconstruction: torch.Tensor,
    observed_frames: torch.Tensor,
    acquisition_geometry: Mapping[str, Any],
    forward_operator: Mapping[str, Any],
) -> ObservedFit:
    """Evaluate the same masked objective/residual for any reconstruction.

    This supplies the requested native observed-frame diagnostics for WF and
    DiffWS without introducing a second optimizer or exposing ground truth to
    the physics-only path.
    """
    protocol_id = str(acquisition_geometry.get("protocol_id", ""))
    if protocol_id != "DMD_6F_2O3P" or observed_frames.ndim != 4 or observed_frames.shape[1] != 6:
        raise ValueError("Observed-fit diagnostics require registered DMD_6F_2O3P raw6")
    sim_config = forward_operator.get("sim_config")
    theta = forward_operator.get("theta")
    if not isinstance(sim_config, SIM2DConfig) or not isinstance(theta, Mapping):
        raise TypeError("forward_operator must provide SIM2DConfig and theta mapping")
    observed_slotted, _mask = embed_raw_to_slots_2d(observed_frames, protocol_id)
    objective, nrmse = _diagnostics(
        reconstruction, observed_slotted, sim_config, protocol_id, theta
    )
    finite = bool(torch.isfinite(objective).item() and torch.isfinite(nrmse).item())
    if not finite:
        raise FloatingPointError("R1C3_NONFINITE_RESULT: observed-fit diagnostics")
    return ObservedFit(float(objective.cpu()), float(nrmse.cpu()), True)


def masked_refine(
    initial_image: torch.Tensor,
    observed_frames: torch.Tensor,
    validity_mask: torch.Tensor,
    acquisition_geometry: Mapping[str, Any],
    forward_operator: Mapping[str, Any],
    refinement_config: RefinementConfig,
    *,
    diagnostic_updates: int | None = None,
    diagnostic_observer: Callable[[Mapping[str, Any]], None] | None = None,
) -> RefineResult:
    """Run the shared strict physics-only registered-protocol refinement.

    The function intentionally has no ``gt`` argument.  Only the registered raw
    observed frames are embedded into the registered 15-slot tensor; invalid
    slots are excluded by the protocol-aware likelihood implementation.
    """
    update_count = refinement_config.updates
    diagnostic_only = diagnostic_updates is not None
    if diagnostic_only:
        update_count = int(diagnostic_updates)
        if update_count not in {80, 160, 320}:
            raise ValueError("Diagnostic extension must end at 80, 160, or 320 updates")
        if update_count <= refinement_config.updates or diagnostic_observer is None:
            raise ValueError("Diagnostic extension requires an observer and must exceed 40 updates")
    elif diagnostic_observer is not None:
        raise ValueError("An observer is accepted only for an explicitly diagnostic extension")

    if initial_image.ndim != 4 or initial_image.shape[1] != 1:
        raise ValueError(f"initial_image must be (B,1,H,W), got {tuple(initial_image.shape)}")
    if observed_frames.ndim != 4:
        raise ValueError("observed_frames must be a 4-D BCHW tensor")
    if initial_image.shape[0] != observed_frames.shape[0] or initial_image.shape[-2:] != observed_frames.shape[-2:]:
        raise ValueError("Initial image and observed frame support differ")
    if initial_image.dtype != torch.float32 or observed_frames.dtype != torch.float32:
        raise ValueError("Strict refinement requires float32")
    protocol_id = str(acquisition_geometry.get("protocol_id", ""))
    spec = require_protocol(protocol_id)
    if observed_frames.shape[1] != spec.frame_count:
        if protocol_id == "DMD_6F_2O3P":
            raise ValueError("Exactly six observed raw frames are required")
        raise ValueError(
            f"Observed frame count {observed_frames.shape[1]} does not match "
            f"registered protocol {protocol_id} ({spec.frame_count})"
        )
    expected_mask = tuple(int(value) for value in spec.validity_mask)
    mask_vector = tuple(int(value) for value in validity_mask.detach().reshape(-1).cpu().tolist())
    if mask_vector != expected_mask:
        raise ValueError(f"Validity mask drift: {mask_vector}")
    sim_config = forward_operator.get("sim_config")
    theta = forward_operator.get("theta")
    if not isinstance(sim_config, SIM2DConfig) or not isinstance(theta, Mapping):
        raise TypeError("forward_operator must provide SIM2DConfig and theta mapping")

    observed_slotted, constructed_mask = embed_raw_to_slots_2d(observed_frames, protocol_id)
    constructed_vector = tuple(
        int(value) for value in constructed_mask[0, :, 0, 0].detach().cpu().tolist()
    )
    if constructed_vector != expected_mask:
        raise RuntimeError("R1C3_INPUT_IDENTITY_MISMATCH: constructed mask changed")
    if observed_slotted.data_ptr() == observed_frames.data_ptr():
        raise RuntimeError("Raw and slotted tensors unexpectedly alias")

    device = initial_image.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    estimate = initial_image.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([estimate], lr=refinement_config.learning_rate)
    objective_history: list[float] = []
    nrmse_history: list[float] = []
    gradient_finite = True
    raw_indices = torch.tensor(
        spec.raw_to_slot_mapping, device=observed_slotted.device, dtype=torch.long
    )
    observed_raw = observed_slotted.index_select(1, raw_indices)

    last_clip_diagnostics = {
        "clip_applied": False,
        "preclip_below_fraction": 0.0,
        "preclip_above_fraction": 0.0,
    }
    for update in range(update_count):
        optimizer.zero_grad(set_to_none=True)
        predicted_raw, _ = forward_protocol_clean_2d(
            estimate, sim_config, protocol_id, theta=dict(theta), randomize=False
        )
        predicted_slotted, _ = embed_raw_to_slots_2d(predicted_raw, protocol_id)
        objective = masked_poisson_gaussian_likelihood_2d(
            observed_slotted,
            predicted_slotted,
            protocol_id,
            photon_scale=theta["photon_scale"],
            read_noise_e=theta["read_noise_e"],
            reduce="mean",
        )
        if not bool(torch.isfinite(objective).item()):
            raise FloatingPointError("R1C3_NONFINITE_RESULT: objective")
        numerator = torch.mean((observed_raw - predicted_raw).square()).sqrt()
        denominator = torch.mean(observed_raw.square()).sqrt().clamp_min(
            refinement_config.nrmse_denominator_epsilon
        )
        nrmse = numerator / denominator
        if not bool(torch.isfinite(nrmse).item()):
            raise FloatingPointError("R1C3_NONFINITE_RESULT: observed NRMSE")
        # At iteration i, estimate is exactly the state after i completed
        # updates (i=0 is the initializer).  This reuses the differentiable
        # forward that drives Adam rather than executing a duplicate forward.
        objective_history.append(float(objective.detach().cpu()))
        nrmse_history.append(float(nrmse.detach().cpu()))
        objective.backward()
        if estimate.grad is None or not bool(torch.isfinite(estimate.grad).all().item()):
            gradient_finite = False
            raise FloatingPointError("R1C3_NONFINITE_RESULT: gradient")
        if diagnostic_observer is not None:
            gradient_norm = float(
                torch.linalg.vector_norm(estimate.grad.detach().to(dtype=torch.float64)).cpu()
            )
            diagnostic_observer(
                {
                    "update": update,
                    "estimate": estimate.detach(),
                    "objective": float(objective.detach().cpu()),
                    "observed_nrmse": float(nrmse.detach().cpu()),
                    "gradient_norm": gradient_norm,
                    **last_clip_diagnostics,
                }
            )
        # Keep the frozen formal operation order exact: Adam.step is followed
        # immediately by in-place clipping.  Diagnostic bookkeeping happens
        # only before the step and must not enter this critical interval.
        optimizer.step()
        with torch.no_grad():
            estimate.clamp_(refinement_config.clip_min, refinement_config.clip_max)

    # One final forward records the state after the 40th update, yielding the
    # requested initial+40 history without changing the optimizer trajectory.
    if diagnostic_observer is None:
        with torch.no_grad():
            final_objective, final_nrmse = _diagnostics(
                estimate, observed_slotted, sim_config, protocol_id, theta
            )
            objective_history.append(float(final_objective.detach().cpu()))
            nrmse_history.append(float(final_nrmse.detach().cpu()))
    else:
        # A diagnostic-only backward at the terminal state records its gradient
        # norm but deliberately performs no optimizer update.  This cannot alter
        # any state at the formal 40-update boundary.
        optimizer.zero_grad(set_to_none=True)
        final_objective, final_nrmse = _diagnostics(
            estimate, observed_slotted, sim_config, protocol_id, theta
        )
        if not bool(torch.isfinite(final_objective).item() and torch.isfinite(final_nrmse).item()):
            raise FloatingPointError("R1C3_NONFINITE_RESULT: diagnostic terminal state")
        final_objective.backward()
        if estimate.grad is None or not bool(torch.isfinite(estimate.grad).all().item()):
            raise FloatingPointError("R1C3_NONFINITE_RESULT: diagnostic terminal gradient")
        gradient_norm = float(
            torch.linalg.vector_norm(estimate.grad.detach().to(dtype=torch.float64)).cpu()
        )
        objective_history.append(float(final_objective.detach().cpu()))
        nrmse_history.append(float(final_nrmse.detach().cpu()))
        diagnostic_observer(
            {
                "update": update_count,
                "estimate": estimate.detach(),
                "objective": float(final_objective.detach().cpu()),
                "observed_nrmse": float(final_nrmse.detach().cpu()),
                "gradient_norm": gradient_norm,
                **last_clip_diagnostics,
            }
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    output = estimate.detach()
    output_finite = bool(torch.isfinite(output).all().item())
    if not output_finite or not all(math.isfinite(value) for value in objective_history + nrmse_history):
        raise FloatingPointError("R1C3_NONFINITE_RESULT: output or history")
    receipt = refinement_config.receipt()
    receipt.update(
        {
            "protocol_id": protocol_id,
            "protocol_hash": acquisition_geometry.get("protocol_hash"),
            "raw_frame_order": acquisition_geometry.get("raw_frame_order"),
            "validity_mask": list(expected_mask),
            "observed_frame_count": spec.frame_count,
            "invalid_slots_excluded": True,
            "forward_theta": _theta_receipt(theta),
            "history_includes_initial_and_each_post_update_state": True,
            "diagnostic_only_extension": diagnostic_only,
            "executed_updates": update_count,
            "formal_result_update": refinement_config.updates,
            "diagnostic_extension_does_not_replace_formal_result": diagnostic_only,
        }
    )
    return RefineResult(
        final_reconstruction=output,
        objective_history=objective_history,
        observed_nrmse_history=nrmse_history,
        gradient_finite=gradient_finite,
        output_finite=output_finite,
        runtime_seconds=runtime,
        peak_gpu_memory_bytes=int(peak),
        configuration_receipt=receipt,
    )


__all__ = [
    "ObservedFit",
    "RefineResult",
    "RefinementConfig",
    "evaluate_observed_fit",
    "masked_refine",
]
