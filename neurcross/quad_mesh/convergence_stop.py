"""Deterministic, fail-open convergence stopping for quad-mesh training.

The automatic gate deliberately makes one narrow claim: uniform-final-weight
loss has reached a three-snapshot plateau on a deterministic fixed probe. It
does not claim that MIQ or QEx topology has converged. The predicted 4-RoSy
field remains the required terminal product, while its trajectory diagnostics
never participate in the stopping decision.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import hashlib
import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


LOSS_NAMES = (
    "loss",
    "sdf_term",
    "inter_term",
    "eikonal_term",
    "normals_loss",
    "morse_term",
    "theta_hessian_term",
    "theta_neighbors_term",
)
UNWEIGHTED_LOSS_NAMES = LOSS_NAMES[1:]
FINAL_FIELD_FILENAME = "final_post_update_eval_cross_field.txt"
FINAL_REPORT_FILENAME = "final_post_update_eval.json"
LOSS_RELATIVE_EPSILON = float(np.finfo(np.float64).eps)


class ProbeStateIntegrityError(RuntimeError):
    """A probe could not prove that it restored training-process state."""


def validate_final_artifact_destination(logdir: Path) -> None:
    """Fail before training when the no-replace terminal bundle already exists."""

    destination = Path(logdir) / "convergence_stop"
    if os.path.lexists(str(destination)):
        raise FileExistsError(
            "refusing to start because the convergence-stop artifact directory "
            "already exists: {}".format(destination)
        )


@dataclass(frozen=True)
class AutoStopConfig:
    min_steps: int = 1500
    check_interval: int = 250
    loss_relative_tolerance: float = 0.10

    def validate(self) -> None:
        if self.min_steps < 1:
            raise ValueError("--auto_min_steps must be positive")
        if self.check_interval < 1:
            raise ValueError("--auto_check_interval must be positive")
        if self.min_steps - 2 * self.check_interval < 1:
            raise ValueError(
                "--auto_min_steps must leave room for two positive, exact "
                "check-interval snapshots"
            )
        value = float(self.loss_relative_tolerance)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "--auto_loss_relative_tolerance must be finite and non-negative"
            )


def validate_training_contract(args: Any, config: AutoStopConfig) -> Tuple[str, int, int]:
    """Validate constraints before the script creates logs or starts workers."""

    config.validate()
    if int(args.batch_size) != 1:
        raise ValueError("automatic/fixed convergence stopping requires --batch_size 1")
    if int(args.num_epochs) != 1:
        raise ValueError("automatic/fixed convergence stopping requires --num_epochs 1")
    horizon = int(args.n_samples) * int(args.num_epochs)
    if horizon < 1:
        raise ValueError("n_samples * num_epochs must be positive")
    if int(args.auto_probe_seed) < 0:
        raise ValueError("--auto_probe_seed must be a non-negative integer")
    fixed_steps = getattr(args, "fixed_steps", None)
    if fixed_steps is None:
        return "auto", horizon, horizon
    fixed_steps = int(fixed_steps)
    if fixed_steps < 1:
        raise ValueError("--fixed_steps must be a positive integer")
    if fixed_steps > horizon:
        raise ValueError(
            "--fixed_steps cannot exceed the Morse schedule/hard horizon "
            "n_samples * num_epochs"
        )
    return "fixed", horizon, fixed_steps


def _normalise_field(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    value = np.asarray(field, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 6 or value.shape[0] == 0:
        raise ValueError("cross field must have shape N x 6")
    if not np.isfinite(value).all():
        raise ValueError("cross field contains non-finite values")
    alpha = value[:, :3]
    beta = value[:, 3:]
    alpha_norm = np.linalg.norm(alpha, axis=1)
    beta_norm = np.linalg.norm(beta, axis=1)
    if np.any(alpha_norm <= 1e-12) or np.any(beta_norm <= 1e-12):
        raise ValueError("cross field contains a zero direction")
    return alpha / alpha_norm[:, None], beta / beta_norm[:, None]


def four_rosy_pair_metrics(
    previous_field: np.ndarray,
    current_field: np.ndarray,
    face_areas: np.ndarray,
) -> Dict[str, float]:
    """Compare fields modulo sign and quarter turns, in degrees."""

    previous_alpha, previous_beta = _normalise_field(previous_field)
    current_alpha, _ = _normalise_field(current_field)
    if previous_alpha.shape != current_alpha.shape:
        raise ValueError("cross-field snapshots have different shapes")
    areas = np.asarray(face_areas, dtype=np.float64).reshape(-1)
    if areas.shape[0] != current_alpha.shape[0]:
        raise ValueError("face-area count does not match cross-field rows")
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValueError("face areas must be finite and strictly positive")
    total_area = float(areas.sum())
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("total face area must be finite and positive")

    alpha_score = np.abs(np.einsum("ij,ij->i", current_alpha, previous_alpha))
    beta_score = np.abs(np.einsum("ij,ij->i", current_alpha, previous_beta))
    angles = np.degrees(
        np.arccos(np.clip(np.maximum(alpha_score, beta_score), 0.0, 1.0))
    )
    return {
        "face_area_weighted_rms_degrees": float(
            math.sqrt(float(np.dot(areas, np.square(angles))) / total_area)
        ),
        "p99_degrees": float(np.percentile(angles, 99.0)),
        "max_degrees": float(np.max(angles)),
        "fraction_over_half_degree": float(np.mean(angles > 0.5)),
    }


def loss_relative_span(scores: Sequence[float]) -> float:
    """Return the calibrated symmetric span for exactly three loss scores."""

    values = np.asarray(tuple(scores), dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("loss plateau requires exactly three scores")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("loss plateau scores must be finite and non-negative")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = float(
        (maximum - minimum) / max(minimum, LOSS_RELATIVE_EPSILON)
    )
    if not math.isfinite(span):
        raise FloatingPointError("loss relative span is non-finite")
    return span


class ConvergenceStopper:
    """Evaluate a three-snapshot fixed-probe loss plateau.

    Cross-field comparisons are best-effort diagnostics only. They never enter
    ``passes`` and a diagnostic failure never disables the loss gate.
    """

    def __init__(
        self,
        config: AutoStopConfig,
        face_areas: Optional[np.ndarray] = None,
        auto_enabled: bool = True,
    ) -> None:
        config.validate()
        self.config = config
        self.face_areas = None
        if face_areas is not None:
            areas = np.asarray(face_areas, dtype=np.float64).reshape(-1)
            if areas.size == 0 or not np.isfinite(areas).all() or np.any(areas <= 0.0):
                raise ValueError("face areas must be finite and strictly positive")
            self.face_areas = np.ascontiguousarray(areas)
        self.auto_enabled = bool(auto_enabled)
        self.auto_disabled_reason = None if auto_enabled else "fixed_steps_override"
        self._history = deque(maxlen=3)
        self.decisions: List[Dict[str, Any]] = []
        self.probe_errors: List[Dict[str, Any]] = []
        self.direction_diagnostic_errors: List[Dict[str, Any]] = []

    @property
    def first_probe_step(self) -> int:
        return self.config.min_steps - 2 * self.config.check_interval

    def should_probe(self, completed_steps: int) -> bool:
        step = int(completed_steps)
        return (
            self.auto_enabled
            and step >= self.first_probe_step
            and (step - self.first_probe_step) % self.config.check_interval == 0
        )

    def disable_after_probe_error(
        self, completed_steps: int, stage: str, error: BaseException
    ) -> Dict[str, Any]:
        record = {
            "step": int(completed_steps),
            "stage": str(stage),
            "exception_type": type(error).__name__,
            "message": str(error),
        }
        self.probe_errors.append(record)
        self.auto_enabled = False
        if self.auto_disabled_reason is None:
            self.auto_disabled_reason = "probe_error"
        return record

    def record_direction_diagnostic_error(
        self, completed_steps: int, stage: str, error: BaseException
    ) -> Dict[str, Any]:
        record = {
            "step": int(completed_steps),
            "stage": str(stage),
            "exception_type": type(error).__name__,
            "message": str(error),
            "affects_loss_gate": False,
        }
        self.direction_diagnostic_errors.append(record)
        return record

    def observe(
        self,
        completed_steps: int,
        cross_field: Optional[np.ndarray],
        final_weight_score: float,
    ) -> Dict[str, Any]:
        step = int(completed_steps)
        if not self.auto_enabled:
            raise RuntimeError("automatic convergence gate is disabled")
        if not self.should_probe(step):
            raise ValueError("observation step is not on the configured probe grid")
        score = float(final_weight_score)
        if not math.isfinite(score) or score < 0.0:
            raise ValueError("uniform final-weight score must be finite and non-negative")
        if self._history and step <= self._history[-1]["step"]:
            raise ValueError("probe steps must be strictly increasing")

        field = None
        direction_error = None
        if cross_field is not None:
            try:
                candidate = np.ascontiguousarray(
                    np.asarray(cross_field, dtype=np.float32)
                )
                _normalise_field(candidate)
                if self.face_areas is None:
                    raise ValueError("face areas are unavailable")
                if candidate.shape[0] != self.face_areas.shape[0]:
                    raise ValueError("cross-field row count does not match face areas")
                field = candidate.copy()
            except Exception as error:
                direction_error = self.record_direction_diagnostic_error(
                    step, "cross_field_snapshot", error
                )
        elif self.face_areas is not None:
            direction_error = self.record_direction_diagnostic_error(
                step,
                "cross_field_snapshot",
                RuntimeError("fixed probe did not provide a direction field"),
            )

        self._history.append({"step": step, "field": field, "score": score})
        expected_steps = [
            step - 2 * self.config.check_interval,
            step - self.config.check_interval,
            step,
        ]
        actual_steps = [item["step"] for item in self._history]
        eligible = step >= self.config.min_steps and actual_steps == expected_steps
        decision: Dict[str, Any] = {
            "step": step,
            "eligible": eligible,
            "window_steps": actual_steps,
            "uniform_final_weight_score": score,
            "passes": False,
            "stop_reason": None,
            "direction_field_diagnostics": {
                "participates_in_stop": False,
                "available": False,
                "pair_metrics": [],
                "error": direction_error,
            },
        }
        if not eligible:
            self.decisions.append(decision)
            return decision

        window = list(self._history)
        window_scores = [float(item["score"]) for item in window]
        minimum_score = min(window_scores)
        maximum_score = max(window_scores)
        relative_span = loss_relative_span(window_scores)
        passes = relative_span <= self.config.loss_relative_tolerance

        pair_rows = []
        diagnostic_error = direction_error
        if self.face_areas is not None and field is not None:
            try:
                for snapshot in window[:2]:
                    if snapshot["field"] is None:
                        raise RuntimeError(
                            "a preceding direction-field snapshot is unavailable"
                        )
                    metrics = four_rosy_pair_metrics(
                        snapshot["field"], field, self.face_areas
                    )
                    metrics["comparison_step"] = snapshot["step"]
                    pair_rows.append(metrics)
            except Exception as error:
                pair_rows = []
                diagnostic_error = self.record_direction_diagnostic_error(
                    step, "cross_field_pair_metrics", error
                )
        decision.update(
            {
                "loss_plateau_gate": {
                    "score_name": "uniform_final_weight_score",
                    "window_scores": window_scores,
                    "minimum": minimum_score,
                    "maximum": maximum_score,
                    "relative_span": relative_span,
                    "relative_span_formula": "(max_score-min_score)/max(min_score,float64_eps)",
                    "epsilon": LOSS_RELATIVE_EPSILON,
                    "maximum_allowed": self.config.loss_relative_tolerance,
                    "passes": passes,
                },
                "direction_field_diagnostics": {
                    "participates_in_stop": False,
                    "available": len(pair_rows) == 2,
                    "pair_metrics": pair_rows,
                    "error": diagnostic_error,
                },
                "passes": passes,
                "stop_reason": "loss_converged" if passes else None,
            }
        )
        self.decisions.append(decision)
        return decision

    def report(self) -> Dict[str, Any]:
        return {
            "claim_scope": (
                "three-snapshot plateau of deterministic fixed-probe "
                "uniform-final-weight loss only"
            ),
            "config": asdict(self.config),
            "window_snapshots_including_current": 3,
            "window_span_updates": 2 * self.config.check_interval,
            "loss_gate_formula": "(max_score-min_score)/max(min_score,float64_eps)",
            "loss_gate_epsilon": LOSS_RELATIVE_EPSILON,
            "first_probe_step": self.first_probe_step,
            "auto_enabled_at_end": self.auto_enabled,
            "auto_disabled_reason": self.auto_disabled_reason,
            "probe_errors": copy.deepcopy(self.probe_errors),
            "direction_field_diagnostics": {
                "participates_in_stop": False,
                "errors": copy.deepcopy(self.direction_diagnostic_errors),
            },
            "decisions": copy.deepcopy(self.decisions),
        }


def build_fixed_probe(train_set: Any, n_points: int, probe_seed: int) -> Dict[str, np.ndarray]:
    """Build a probe without touching Python, NumPy-global, or Torch RNGs."""

    generator = np.random.default_rng(int(probe_seed))
    manifold = np.ascontiguousarray(train_set.points, dtype=np.float32)
    normals = np.ascontiguousarray(train_set.mnfld_n, dtype=np.float32)
    local_u = np.ascontiguousarray(train_set.vector_u, dtype=np.float32)
    local_v = np.ascontiguousarray(train_set.vector_v, dtype=np.float32)
    nonmanifold = generator.uniform(
        -float(train_set.grid_range),
        float(train_set.grid_range),
        size=(int(n_points), 3),
    ).astype(np.float32)
    near = (
        manifold
        + np.asarray(train_set.sigmas)
        * generator.standard_normal(manifold.shape)
    ).astype(np.float32)
    arrays = {
        "manifold_points": manifold,
        "manifold_normals": normals,
        "nonmanifold_points": np.ascontiguousarray(nonmanifold),
        "near_points": np.ascontiguousarray(near),
        "local_coordinates_u": local_u,
        "local_coordinates_v": local_v,
    }
    row_count = manifold.shape[0]
    for name in (
        "manifold_normals",
        "near_points",
        "local_coordinates_u",
        "local_coordinates_v",
    ):
        if arrays[name].shape != (row_count, 3):
            raise ValueError("fixed probe has inconsistent manifold array shapes")
    return arrays


def final_morse_weight(
    current_iteration: int,
    n_iterations: int,
    params: Sequence[float],
    decay: str,
    initial_weight: float,
) -> float:
    """Pure equivalent of MorseLoss_quad_mesh.update_morse_weight."""

    if n_iterations < 1:
        raise ValueError("n_iterations must be positive")
    if len(params) < 2 or len(params[1:-1]) % 2 != 0:
        raise ValueError("invalid --decay_params sequence")
    points = list(
        zip(
            [params[0], *params[1:-1][1::2], params[-1]],
            [0, *params[1:-1][::2], 1],
        )
    )
    fraction = current_iteration / n_iterations
    end_weight, end_fraction = min(
        (item for item in points if item[1] >= fraction), key=lambda item: item[1]
    )
    start_weight, start_fraction = max(
        (item for item in points if item[1] <= fraction), key=lambda item: item[1]
    )
    if decay == "none":
        return float(initial_weight)
    if current_iteration < start_fraction * n_iterations:
        return float(start_weight)
    if decay == "step":
        return float(end_weight)
    if current_iteration >= end_fraction * n_iterations:
        return float(end_weight)
    if decay == "linear":
        # Preserve the deployed operation order, including its final-bit float
        # rounding; schedule tests compare this directly with the formal method.
        return float(
            start_weight
            + (end_weight - start_weight)
            * (current_iteration / n_iterations - start_fraction)
            / (end_fraction - start_fraction)
        )
    if decay == "quintic":
        return float(
            start_weight
            + (end_weight - start_weight)
            * (
                1.0
                - (
                    1.0
                    - (current_iteration / n_iterations - start_fraction)
                    / (end_fraction - start_fraction)
                )
                ** 5
            )
        )
    raise ValueError("unsupported Morse decay: {}".format(decay))


def _require_torch() -> Any:
    import torch

    return torch


def _copy_numpy_rng_state(state: Tuple[Any, ...]) -> Tuple[Any, ...]:
    return (state[0], state[1].copy(), state[2], state[3], state[4])


def _numpy_rng_equal(left: Tuple[Any, ...], right: Tuple[Any, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _snapshot_criterion_weights(criterion: Any, torch: Any) -> Optional[Dict[str, Any]]:
    if criterion is None:
        return None
    container = criterion.weights
    if not isinstance(container, list):
        raise TypeError("criterion.weights must be a list for transactional probing")
    entries = list(container)
    values = [
        value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for value in entries
    ]
    return {"container": container, "entries": entries, "values": values}


def _criterion_weights_changed(snapshot: Optional[Dict[str, Any]], criterion: Any, torch: Any) -> bool:
    if snapshot is None:
        return False
    current = criterion.weights
    if current is not snapshot["container"] or len(current) != len(snapshot["values"]):
        return True
    for value, expected in zip(current, snapshot["values"]):
        if torch.is_tensor(expected):
            if not torch.is_tensor(value) or not bool(torch.equal(value, expected)):
                return True
        elif torch.is_tensor(value) or value != expected:
            return True
    return False


def _restore_criterion_weights(snapshot: Optional[Dict[str, Any]], criterion: Any, torch: Any) -> None:
    if snapshot is None:
        return
    restored_entries = []
    for original, expected in zip(snapshot["entries"], snapshot["values"]):
        if torch.is_tensor(expected):
            if not torch.is_tensor(original):
                raise RuntimeError("cannot restore a changed criterion weight type")
            with torch.no_grad():
                original.copy_(expected)
            restored_entries.append(original)
        else:
            restored_entries.append(original)
    original_container = snapshot["container"]
    original_container[:] = restored_entries
    criterion.weights = original_container


def run_eval_preserving_training_state(
    net: Any, callback: Callable[[], Any], criterion: Any = None
) -> Tuple[Any, Dict[str, Any]]:
    """Run eval callback and transactionally restore training-process state."""

    torch = _require_torch()
    python_state = random.getstate()
    numpy_state = _copy_numpy_rng_state(np.random.get_state())
    torch_cpu_state = torch.get_rng_state().clone()
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    module_flags = [(module, bool(module.training)) for module in net.modules()]
    criterion_weights = _snapshot_criterion_weights(criterion, torch)
    result = None
    callback_error = None
    weights_changed = False
    restore_failures: List[Tuple[str, BaseException]] = []
    try:
        net.eval()
        result = callback()
    except BaseException as error:  # restoration must also cover exceptional probes
        callback_error = error
    finally:
        try:
            weights_changed = _criterion_weights_changed(
                criterion_weights, criterion, torch
            )
            if weights_changed:
                _restore_criterion_weights(criterion_weights, criterion, torch)
        except BaseException as error:
            restore_failures.append(("criterion.weights", error))
        finally:
            # Weight restoration must never prevent restoration of RNGs or
            # intentionally mixed module training flags.
            for index, (module, enabled) in enumerate(module_flags):
                try:
                    module.training = enabled
                except BaseException as error:
                    restore_failures.append(("module_flags[{}]".format(index), error))
            for stage, restore in (
                ("python_rng", lambda: random.setstate(python_state)),
                ("numpy_global_rng", lambda: np.random.set_state(numpy_state)),
                ("torch_cpu_rng", lambda: torch.set_rng_state(torch_cpu_state)),
            ):
                try:
                    restore()
                except BaseException as error:
                    restore_failures.append((stage, error))
            if cuda_states:
                try:
                    torch.cuda.set_rng_state_all(cuda_states)
                except BaseException as error:
                    restore_failures.append(("torch_cuda_rng", error))

    if restore_failures:
        stages = ", ".join(stage for stage, _ in restore_failures)
        raise ProbeStateIntegrityError(
            "fixed probe restoration raised for {}; refusing to continue".format(stages)
        ) from restore_failures[0][1]

    try:
        restored_python = random.getstate() == python_state
        restored_numpy = _numpy_rng_equal(np.random.get_state(), numpy_state)
        restored_torch_cpu = bool(torch.equal(torch.get_rng_state(), torch_cpu_state))
        restored_cuda = True
        if cuda_states:
            after_cuda = torch.cuda.get_rng_state_all()
            restored_cuda = len(after_cuda) == len(cuda_states) and all(
                bool(torch.equal(left, right))
                for left, right in zip(after_cuda, cuda_states)
            )
        restored_flags = all(
            module.training == enabled for module, enabled in module_flags
        )
        restored_weights = not _criterion_weights_changed(
            criterion_weights, criterion, torch
        )
    except BaseException as verification_error:
        raise ProbeStateIntegrityError(
            "fixed probe could not verify restored training process state"
        ) from verification_error
    if not all(
        (
            restored_python,
            restored_numpy,
            restored_torch_cpu,
            restored_cuda,
            restored_flags,
            restored_weights,
        )
    ):
        raise ProbeStateIntegrityError(
            "fixed probe failed to restore training process state; refusing to continue"
        )
    if weights_changed:
        mutation_error = RuntimeError(
            "fixed probe mutated criterion.weights; original weights were restored"
        )
        if callback_error is not None:
            raise mutation_error from callback_error
        raise mutation_error
    if callback_error is not None:
        raise callback_error
    return result, {
        "python_rng_restored": True,
        "numpy_global_rng_restored": True,
        "torch_cpu_rng_restored": True,
        "torch_cuda_rng_restored": True,
        "module_training_flags_restored": True,
        "criterion_weights_unchanged": True,
        "module_count": len(module_flags),
    }


def _tensor_from_probe(
    torch: Any,
    probe: Mapping[str, np.ndarray],
    name: str,
    device: Any,
    requires_grad: bool = False,
) -> Any:
    value = torch.from_numpy(probe[name]).unsqueeze(0).to(device)
    if requires_grad:
        value.requires_grad_()
    return value


def _cross_field_from_theta(torch: Any, theta: Any, local_u: Any, local_v: Any) -> np.ndarray:
    theta = theta.squeeze(0)
    basis_u = local_u.squeeze(0)
    basis_v = local_v.squeeze(0)
    alpha = basis_u * torch.cos(theta) + basis_v * torch.sin(theta)
    alpha = alpha / (alpha.norm(dim=-1, keepdim=True) + 1e-12)
    beta = -basis_u * torch.sin(theta) + basis_v * torch.cos(theta)
    beta = beta / (beta.norm(dim=-1, keepdim=True) + 1e-12)
    field = torch.cat((alpha, beta), dim=-1)
    if field.ndim != 2 or field.shape[1] != 6:
        raise RuntimeError("fixed-probe cross field is not N x 6")
    if not bool(torch.isfinite(field).all().item()):
        raise FloatingPointError("fixed-probe cross field contains non-finite values")
    return np.ascontiguousarray(field.detach().cpu().numpy(), dtype=np.float32)


def _weighted_score(terms: Mapping[str, float], weights: Sequence[float]) -> float:
    return float(
        weights[0] * terms["sdf_term"]
        + weights[1] * terms["inter_term"]
        + weights[2] * terms["theta_hessian_term"]
        + weights[3] * terms["eikonal_term"]
        + weights[4] * terms["theta_neighbors_term"]
        + weights[5] * terms["morse_term"]
    )


def run_fixed_probe(
    net: Any,
    criterion: Any,
    probe: Mapping[str, np.ndarray],
    args: Any,
    device: Any,
    final_score_weights: Sequence[float],
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """Evaluate fixed loss plus optional field diagnostic without perturbation."""

    torch = _require_torch()

    def callback() -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        manifold = _tensor_from_probe(
            torch, probe, "manifold_points", device, requires_grad=True
        )
        normals = _tensor_from_probe(torch, probe, "manifold_normals", device)
        nonmanifold = _tensor_from_probe(
            torch, probe, "nonmanifold_points", device, requires_grad=True
        )
        near = _tensor_from_probe(
            torch, probe, "near_points", device, requires_grad=True
        )
        local_u = _tensor_from_probe(torch, probe, "local_coordinates_u", device)
        local_v = _tensor_from_probe(torch, probe, "local_coordinates_v", device)
        features = torch.cat((manifold, normals, local_u, local_v), dim=-1)
        output_pred, theta = net(
            nonmanifold,
            manifold,
            near_points=near if args.morse_near else None,
            angle_features=features,
        )
        if theta is None:
            raise RuntimeError("fixed probe did not produce an angle field")
        loss_dict = criterion(
            output_pred,
            manifold,
            nonmanifold,
            normals,
            near_points=near if args.morse_near else None,
            batch_idx=0,
            logdir=None,
            filename=None,
            save_best=False,
            mnfld_pts_theta_output_pred=theta,
            local_coord_u=local_u,
            local_coord_v=local_v,
        )
        missing = [name for name in LOSS_NAMES if name not in loss_dict]
        if missing:
            raise RuntimeError("fixed probe loss is missing keys: {}".format(missing))
        unweighted = {
            name: float(loss_dict[name].detach().cpu().item())
            for name in UNWEIGHTED_LOSS_NAMES
        }
        current_weights = [
            float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
            for value in criterion.weights
        ]
        scheduled_score = _weighted_score(unweighted, current_weights)
        final_score = _weighted_score(unweighted, final_score_weights)
        reported = float(loss_dict["loss"].detach().cpu().item())
        numeric = [*unweighted.values(), scheduled_score, final_score, reported]
        if not all(math.isfinite(value) for value in numeric):
            raise FloatingPointError("fixed probe produced a non-finite score")
        if not math.isclose(reported, scheduled_score, rel_tol=2e-5, abs_tol=2e-6):
            raise RuntimeError("fixed-probe score disagrees with deployed loss mapping")
        field = None
        direction_diagnostic = {"available": False, "participates_in_stop": False}
        try:
            field = _cross_field_from_theta(torch, theta, local_u, local_v)
            direction_diagnostic["available"] = True
        except Exception as direction_error:
            direction_diagnostic["error"] = {
                "exception_type": type(direction_error).__name__,
                "message": str(direction_error),
            }
        return (
            {
                "mode": "eval",
                "unweighted_losses": unweighted,
                "scheduled_score_weights": current_weights,
                "scheduled_weight_score": scheduled_score,
                "uniform_final_score_weights": [float(item) for item in final_score_weights],
                "uniform_final_weight_score": final_score,
                "direction_field_diagnostic": direction_diagnostic,
            },
            field,
        )

    result, restoration = run_eval_preserving_training_state(
        net, callback, criterion=criterion
    )
    metrics, field = result
    metrics["state_restoration"] = restoration
    return metrics, field


def run_fixed_field_only(
    net: Any,
    probe: Mapping[str, np.ndarray],
    args: Any,
    device: Any,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Fallback terminal export when the loss part of a probe has failed."""

    torch = _require_torch()

    def callback() -> np.ndarray:
        manifold = _tensor_from_probe(torch, probe, "manifold_points", device)
        normals = _tensor_from_probe(torch, probe, "manifold_normals", device)
        nonmanifold = _tensor_from_probe(torch, probe, "nonmanifold_points", device)
        near = _tensor_from_probe(torch, probe, "near_points", device)
        local_u = _tensor_from_probe(torch, probe, "local_coordinates_u", device)
        local_v = _tensor_from_probe(torch, probe, "local_coordinates_v", device)
        features = torch.cat((manifold, normals, local_u, local_v), dim=-1)
        with torch.no_grad():
            _, theta = net(
                nonmanifold,
                manifold,
                near_points=near if args.morse_near else None,
                angle_features=features,
            )
        if theta is None:
            raise RuntimeError("terminal field probe did not produce an angle field")
        return _cross_field_from_theta(torch, theta, local_u, local_v)

    field, restoration = run_eval_preserving_training_state(net, callback)
    return {
        "mode": "eval",
        "loss_probe_available": False,
        "uniform_final_weight_score": None,
        "state_restoration": restoration,
    }, field


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_no_replace(staging: Path, destination: Path) -> None:
    """Atomically publish one complete directory without replacing a peer."""

    if os.name == "nt":
        # Windows os.rename refuses an existing destination.
        os.rename(str(staging), str(destination))
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError(
                "atomic no-replace directory publication requires renameat2"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(str(staging)),
            -100,
            os.fsencode(str(destination)),
            1,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(
                    error_number,
                    "refusing to replace convergence artifact directory",
                    str(destination),
                )
            raise OSError(error_number, os.strerror(error_number), str(destination))
        return
    raise RuntimeError(
        "atomic no-replace directory publication is unsupported on this platform"
    )


def write_final_artifacts(
    logdir: Path,
    cross_field: np.ndarray,
    report: Mapping[str, Any],
    *,
    lowercase_research_sha256: bool = False,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Atomically publish the mandatory terminal field and its report."""

    field = np.ascontiguousarray(np.asarray(cross_field, dtype=np.float32))
    _normalise_field(field)
    logdir = Path(logdir)
    output_dir = logdir / "convergence_stop"
    if os.path.lexists(str(output_dir)):
        raise FileExistsError(
            "refusing to replace an existing convergence-stop artifact directory"
        )
    staging = Path(tempfile.mkdtemp(prefix=".convergence_stop-stage-", dir=str(logdir)))
    field_staging = staging / FINAL_FIELD_FILENAME
    report_staging = staging / FINAL_REPORT_FILENAME
    try:
        np.savetxt(str(field_staging), field)
        with field_staging.open("r+b") as target:
            target.flush()
            os.fsync(target.fileno())

        enriched = copy.deepcopy(dict(report))
        enriched["artifact_commit"] = {
            "status": "complete",
            "publication": "atomic_directory_rename_no_replace",
            "report_published_with_field": True,
        }
        enriched["terminal_cross_field"] = {
            "available": True,
            "required_output": True,
            "relative_path": "convergence_stop/{}".format(FINAL_FIELD_FILENAME),
            "rows": int(field.shape[0]),
            "columns": int(field.shape[1]),
            "dtype_in_memory": str(field.dtype),
            "bytes": field_staging.stat().st_size,
            "sha256": (
                _sha256_file(field_staging).lower()
                if lowercase_research_sha256 else _sha256_file(field_staging)),
            "state_mode": "post-optimizer post-schedule eval",
        }
        payload = (
            json.dumps(enriched, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        with report_staging.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        _fsync_directory(staging)
        _publish_directory_no_replace(staging, output_dir)
        _fsync_directory(logdir)
    finally:
        if staging.exists():
            shutil.rmtree(str(staging))
    field_path = output_dir / FINAL_FIELD_FILENAME
    report_path = output_dir / FINAL_REPORT_FILENAME
    return field_path, report_path, enriched
