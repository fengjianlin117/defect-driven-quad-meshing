"""Opt-in geometry angle initialization and research artifact helpers."""
from __future__ import annotations

import atexit
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

GEOMETRY_INIT_SCHEMA = "trimesh2quad.geometry_init.v1"
RESEARCH_CHECKPOINT_SCHEMA = "trimesh2quad.neurcross.research_checkpoint.v1"
RESEARCH_TRACE_SCHEMA = "trimesh2quad.neurcross.research_trace.v1"
FIELD_FILENAME = "post_update_eval_cross_field.txt"
REPORT_FILENAME = "post_update_eval.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNIT_ATOL = 2e-4
_TANGENT_ATOL = 2e-4


class RuntimeStateIntegrityError(RuntimeError):
    """Geometry prefit could not prove complete process-state restoration."""


@dataclass(frozen=True)
class GeometryInitializationTarget:
    source_path: Path
    npz_path: Path
    mesh_path: Path
    mesh_sha256: str
    initialization_sha256: str
    face_count: int
    vertex_count: int
    field: np.ndarray
    confidence: np.ndarray
    constrained: np.ndarray
    z4: np.ndarray
    constraint_target_z4: np.ndarray

    @property
    def tangents(self) -> np.ndarray:
        return self.field[:, :3]

    def report(self) -> Dict[str, Any]:
        return {
            "schema_version": GEOMETRY_INIT_SCHEMA,
            "source_path": str(self.source_path),
            "npz_path": str(self.npz_path),
            "mesh_path": str(self.mesh_path),
            "mesh_sha256": self.mesh_sha256,
            "initialization_sha256": self.initialization_sha256,
            "face_count": self.face_count,
            "vertex_count": self.vertex_count,
            "confidence": {
                "minimum": float(self.confidence.min()),
                "maximum": float(self.confidence.max()),
                "mean": float(self.confidence.mean()),
            },
            "constrained_faces": int(np.count_nonzero(self.constrained)),
        }


@dataclass(frozen=True)
class ResearchPublicationPaths:
    """Lexically translate scientific references without touching publication I/O."""

    actual_shape_logdir: Path
    publication_shape_logdir: Optional[Path]

    @property
    def enabled(self) -> bool:
        return self.publication_shape_logdir is not None

    def publication_reference(self, actual_path: Path) -> Path:
        """Map one actual in-tree artifact reference to its publication location."""
        reference = Path(actual_path)
        if not self.enabled:
            return reference
        if not reference.is_absolute():
            raise ValueError("research artifact reference must be absolute")
        reference = _absolute_lexical_path(reference)
        try:
            relative = reference.relative_to(self.actual_shape_logdir)
        except ValueError as error:
            raise ValueError(
                "research artifact reference is outside the actual shape logdir: {}".format(
                    reference)) from error
        return self.publication_shape_logdir / relative

    def validate_trace_path(self, trace_path: Optional[Path]) -> None:
        """Publication-mode custom traces must move with the actual shape directory."""
        if not self.enabled or trace_path is None:
            return
        trace = Path(trace_path)
        if not trace.is_absolute():
            raise ValueError("research trace path must be absolute in publication mode")
        trace = _absolute_lexical_path(trace)
        try:
            trace.relative_to(self.actual_shape_logdir)
        except ValueError as error:
            raise ValueError(
                "research trace path must be inside the actual shape logdir in publication mode"
            ) from error


def build_research_publication_paths(
        actual_shape_logdir: Path,
        configured_publication_logdir: Optional[str]) -> ResearchPublicationPaths:
    """Build an opt-in lexical mapper without stat, resolve, mkdir, or writes."""
    actual = Path(actual_shape_logdir)
    if configured_publication_logdir is None:
        return ResearchPublicationPaths(actual, None)
    if (not isinstance(configured_publication_logdir, str)
            or not configured_publication_logdir.strip()):
        raise ValueError("--research_publication_logdir must be a non-empty absolute path")
    publication = Path(configured_publication_logdir)
    if not actual.is_absolute():
        raise ValueError("actual shape logdir must be absolute in publication mode")
    if not publication.is_absolute():
        raise ValueError("--research_publication_logdir must be an absolute path")
    actual = _absolute_lexical_path(actual)
    publication = _absolute_lexical_path(publication)
    if (actual == publication or actual in publication.parents
            or publication in actual.parents):
        raise ValueError("actual and publication shape logdirs must not overlap")
    return ResearchPublicationPaths(actual, publication)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().lower()


def _scalar(data: Mapping[str, np.ndarray], name: str) -> Any:
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError("{} must be an NPZ scalar".format(name))
    return value.item()


def _integer_scalar(data: Mapping[str, np.ndarray], name: str) -> int:
    value = np.asarray(data[name])
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("{} must be an integer NPZ scalar".format(name))
    parsed = int(value.item())
    if parsed < 1:
        raise ValueError("{} must be positive".format(name))
    return parsed


def _scalar_text(data: Mapping[str, np.ndarray], name: str) -> str:
    value = _scalar(data, name)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError("{} must be a string scalar".format(name))
    return value


def _load_companion(path: Path) -> Tuple[Path, Optional[Dict[str, Any]]]:
    if path.suffix.lower() != ".json":
        if path.suffix.lower() != ".npz":
            raise ValueError("--geometry_init_file must be geometry_init.npz or geometry_init.json")
        return path, None
    try:
        companion = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("geometry initialization JSON is invalid") from error
    if not isinstance(companion, dict) or companion.get("schema_version") != GEOMETRY_INIT_SCHEMA:
        raise ValueError("unsupported geometry initialization JSON schema")
    mesh = companion.get("mesh")
    field = companion.get("field")
    if not isinstance(mesh, dict) or not isinstance(field, dict):
        raise ValueError("geometry initialization JSON requires mesh and field objects")
    if field.get("coordinates") != "world" or field.get("face_order") != "obj":
        raise ValueError("geometry field must use world coordinates and OBJ face order")
    reference = field.get("npz")
    if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
        raise ValueError("geometry initialization JSON field.npz must be relative")
    npz_path = (path.parent / reference).resolve()
    parent = path.parent.resolve()
    if npz_path != parent and parent not in npz_path.parents:
        raise ValueError("geometry initialization JSON field.npz escapes its directory")
    return npz_path, companion


def _mesh_contract(mesh_path: Path) -> Tuple[int, int, np.ndarray]:
    import trimesh

    loaded = trimesh.load_mesh(str(mesh_path), process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("geometry initialization requires one triangle mesh")
    faces = np.asarray(loaded.faces)
    vertices = np.asarray(loaded.vertices)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError("geometry initialization requires a non-empty triangle mesh")
    normals = np.asarray(loaded.face_normals, dtype=np.float64)
    if normals.shape != (faces.shape[0], 3) or not np.isfinite(normals).all():
        raise ValueError("mesh face normals are invalid")
    norms = np.linalg.norm(normals, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("mesh contains a face with zero normal")
    return int(faces.shape[0]), int(vertices.shape[0]), normals / norms[:, None]


def load_geometry_initialization(initialization_path: Path, mesh_path: Path) -> GeometryInitializationTarget:
    """Load and strictly bind a world-space field to the exact input mesh."""
    source_path = Path(initialization_path).expanduser().resolve()
    input_mesh_path = Path(mesh_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError("geometry initialization file does not exist: {}".format(source_path))
    if not input_mesh_path.is_file():
        raise FileNotFoundError("input mesh does not exist: {}".format(input_mesh_path))
    npz_path, companion = _load_companion(source_path)
    if not npz_path.is_file():
        raise FileNotFoundError("geometry initialization NPZ does not exist: {}".format(npz_path))

    required = {"schema_version", "mesh_sha256", "face_count", "vertex_count",
                "face_order", "field", "confidence", "constrained", "z4",
                "constraint_target_z4"}
    with np.load(str(npz_path), allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError("geometry initialization NPZ is missing keys: {}".format(missing))
        schema_version = _scalar_text(archive, "schema_version")
        mesh_sha256 = _scalar_text(archive, "mesh_sha256")
        face_count = _integer_scalar(archive, "face_count")
        vertex_count = _integer_scalar(archive, "vertex_count")
        face_order = np.asarray(archive["face_order"])
        field = np.asarray(archive["field"])
        confidence = np.asarray(archive["confidence"])
        constrained = np.asarray(archive["constrained"])
        z4 = np.asarray(archive["z4"])
        constraint_target_z4 = np.asarray(archive["constraint_target_z4"])

    if schema_version != GEOMETRY_INIT_SCHEMA:
        raise ValueError("unsupported geometry initialization NPZ schema")
    if not _SHA256_RE.fullmatch(mesh_sha256):
        raise ValueError("mesh_sha256 must be 64 lowercase hexadecimal characters")
    if mesh_sha256 != sha256_file(input_mesh_path):
        raise ValueError("geometry initialization mesh SHA-256 does not match --data_path")
    actual_faces, actual_vertices, normals = _mesh_contract(input_mesh_path)
    if face_count != actual_faces or vertex_count != actual_vertices or face_count < 1:
        raise ValueError("geometry initialization mesh counts do not match --data_path")
    if face_order.dtype != np.int64 or face_order.shape != (face_count,):
        raise ValueError("face_order must be int64 with shape (F,)")
    if not np.array_equal(face_order, np.arange(face_count, dtype=np.int64)):
        raise ValueError("face_order must be the zero-based identity OBJ face order")
    if field.dtype != np.float64 or field.shape != (face_count, 6):
        raise ValueError("field must be float64 with shape (F,6)")
    if confidence.dtype != np.float64 or confidence.shape != (face_count,):
        raise ValueError("confidence must be float64 with shape (F,)")
    if constrained.dtype != np.bool_ or constrained.shape != (face_count,):
        raise ValueError("constrained must be bool with shape (F,)")
    if z4.dtype != np.complex128 or z4.shape != (face_count,):
        raise ValueError("z4 must be complex128 with shape (F,)")
    if (constraint_target_z4.dtype != np.complex128
            or constraint_target_z4.shape != (face_count,)):
        raise ValueError("constraint_target_z4 must be complex128 with shape (F,)")
    if not (np.isfinite(field).all() and np.isfinite(confidence).all()
            and np.isfinite(z4.real).all() and np.isfinite(z4.imag).all()
            and np.isfinite(constraint_target_z4.real).all()
            and np.isfinite(constraint_target_z4.imag).all()):
        raise ValueError("geometry initialization arrays contain non-finite values")
    for name, values in (("z4", z4), ("constraint_target_z4", constraint_target_z4)):
        if not np.allclose(np.abs(values), 1.0, rtol=0.0, atol=1.0e-9):
            raise ValueError("{} must contain unit complex values".format(name))
    if np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ValueError("geometry initialization confidence must lie in [0,1]")

    alpha, beta = field[:, :3], field[:, 3:]
    if not (np.allclose(np.linalg.norm(alpha, axis=1), 1.0, rtol=0.0, atol=_UNIT_ATOL)
            and np.allclose(np.linalg.norm(beta, axis=1), 1.0, rtol=0.0, atol=_UNIT_ATOL)):
        raise ValueError("geometry initialization field directions must be unit length")
    tangent_error = max(np.abs(np.einsum("ij,ij->i", alpha, normals)).max(),
                        np.abs(np.einsum("ij,ij->i", beta, normals)).max())
    if float(tangent_error) > _TANGENT_ATOL:
        raise ValueError("geometry initialization field is not tangent to input faces")
    if float(np.linalg.norm(np.cross(normals, alpha) - beta, axis=1).max()) > _TANGENT_ATOL:
        raise ValueError("geometry initialization beta must equal face_normal cross alpha")

    if companion is not None:
        mesh, description = companion["mesh"], companion["field"]
        if (mesh.get("sha256") != mesh_sha256
                or int(mesh.get("face_count", -1)) != face_count
                or int(mesh.get("vertex_count", -1)) != vertex_count):
            raise ValueError("geometry initialization JSON and NPZ mesh metadata disagree")
        if description.get("shape") != [face_count, 6] or description.get("dtype") != "float64":
            raise ValueError("geometry initialization JSON field metadata is invalid")

    return GeometryInitializationTarget(
        source_path, npz_path, input_mesh_path, mesh_sha256,
        sha256_file(npz_path), face_count, vertex_count,
        np.ascontiguousarray(field), np.ascontiguousarray(confidence),
        np.ascontiguousarray(constrained), np.ascontiguousarray(z4),
        np.ascontiguousarray(constraint_target_z4))


def validate_research_contract(args: Any, training_mode: str,
                               n_iterations: int, requested_steps: int) -> Tuple[int, ...]:
    """Reject opt-in contract errors before logs, workers, or GPU use."""
    angle_init = getattr(args, "angle_init", "random")
    geometry_path = getattr(args, "geometry_init_file", None)
    if angle_init not in ("random", "geometry"):
        raise ValueError("--angle_init must be random or geometry")
    prefit_lr = float(getattr(args, "geometry_prefit_lr", 1e-3))
    prefit_steps = int(getattr(args, "geometry_prefit_max_steps", 500))
    target_deg = float(getattr(args, "geometry_prefit_target_deg", 5.0))
    if not math.isfinite(prefit_lr) or prefit_lr <= 0.0:
        raise ValueError("--geometry_prefit_lr must be finite and positive")
    if prefit_steps < 1:
        raise ValueError("--geometry_prefit_max_steps must be positive")
    if not math.isfinite(target_deg) or target_deg < 0.0 or target_deg > 45.0:
        raise ValueError("--geometry_prefit_target_deg must be finite and in [0,45]")
    if angle_init == "geometry":
        if not geometry_path:
            raise ValueError("--angle_init geometry requires --geometry_init_file")
        if getattr(args, "load_path", None) is not None:
            raise ValueError("--load_path is mutually exclusive with --angle_init geometry")
    elif geometry_path is not None:
        raise ValueError("--geometry_init_file requires --angle_init geometry")

    checkpoint_steps = tuple(int(value) for value in
                             getattr(args, "research_checkpoint_steps", ()))
    if any(value < 1 for value in checkpoint_steps):
        raise ValueError("--research_checkpoint_steps values must be positive")
    if len(set(checkpoint_steps)) != len(checkpoint_steps):
        raise ValueError("--research_checkpoint_steps must not contain duplicates")
    checkpoint_steps = tuple(sorted(checkpoint_steps))
    if training_mode == "fixed":
        latest_reachable = int(requested_steps)
    elif training_mode == "auto":
        latest_reachable = int(args.auto_min_steps)
    else:
        raise ValueError("unknown training mode: {}".format(training_mode))
    if any(value > latest_reachable for value in checkpoint_steps):
        raise ValueError("research checkpoint could be skipped by termination")
    if any(value > int(n_iterations) for value in checkpoint_steps):
        raise ValueError("research checkpoint step exceeds training horizon")

    trace_enabled = bool(getattr(args, "research_trace", False))
    trace_path = getattr(args, "research_trace_path", None)
    if trace_path is not None and (not isinstance(trace_path, str) or not trace_path.strip()):
        raise ValueError("--research_trace_path must be a non-empty path")
    if trace_path is not None and not trace_enabled:
        raise ValueError("--research_trace_path requires --research_trace")
    return checkpoint_steps


def _absolute_lexical_path(path: Path) -> Path:
    """Make a path absolute without following its final symlink."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def resolve_research_trace_path(args: Any, logdir: Path) -> Optional[Path]:
    if not bool(getattr(args, "research_trace", False)):
        return None
    configured = getattr(args, "research_trace_path", None)
    if configured is None:
        return _absolute_lexical_path(Path(logdir) / "research_trace.jsonl")
    return _absolute_lexical_path(configured)


def research_checkpoint_directory(logdir: Path, step: int) -> Path:
    return Path(logdir) / "research_checkpoints" / "step_{:08d}".format(int(step))


def validate_research_output_destinations(logdir: Path, checkpoint_steps: Sequence[int],
                                          trace_path: Optional[Path]) -> None:
    """Perform no-replace destination checks without creating anything."""
    logdir = _absolute_lexical_path(logdir)
    checkpoint_root = logdir / "research_checkpoints"
    if checkpoint_steps and os.path.lexists(str(checkpoint_root)):
        if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
            raise NotADirectoryError(
                "research checkpoint parent must be a real directory: {}".format(
                    checkpoint_root))
    for step in checkpoint_steps:
        destination = research_checkpoint_directory(logdir, step)
        if os.path.lexists(str(destination)):
            raise FileExistsError("research checkpoint already exists: {}".format(destination))
    if trace_path is None:
        return
    trace_path = _absolute_lexical_path(trace_path)
    if os.path.lexists(str(trace_path)):
        raise FileExistsError("research trace already exists: {}".format(trace_path))
    if trace_path == logdir:
        raise ValueError("research trace path cannot be the shape log directory")
    if trace_path == logdir / "out.log":
        raise ValueError("research trace path collides with the training log")
    reserved_directories = (
        logdir / "convergence_stop",
        checkpoint_root,
    )
    if any(trace_path == reserved or reserved in trace_path.parents
           for reserved in reserved_directories):
        raise ValueError("research trace path collides with a reserved artifact directory")
    trace_parent = trace_path.parent
    if os.path.lexists(str(trace_parent)) and not trace_parent.is_dir():
        raise NotADirectoryError(
            "research trace parent is not a directory: {}".format(trace_parent))
    try:
        trace_path.relative_to(logdir)
        inside_logdir = True
    except ValueError:
        inside_logdir = False
    if not inside_logdir and not trace_parent.is_dir():
        raise FileNotFoundError("custom research trace parent does not exist: {}".format(
            trace_parent))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("structured research output cannot contain NaN or infinity")
    return value

_TERMINAL_FIELD_STATES = {
    "automatic": "automatic_terminal",
    "fixed_steps": "fixed_steps_terminal",
    "hard_horizon": "hard_horizon_terminal",
}


def build_research_field_provenance(
        *, seed: int, mesh_sha256: str, grid_res: int, angle_init: str,
        field_state: str, termination: Optional[str] = None,
        geometry_initialization: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Build strict provenance for a checkpoint or terminal direction field.

    A terminal state is named automatic_terminal only when the convergence
    gate actually stopped the run. Fixed-step and exhausted-horizon fields are
    deliberately labelled differently so downstream experiments cannot mistake
    them for automatic terminal observations.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("research provenance seed must be an integer")
    if not isinstance(mesh_sha256, str) or not _SHA256_RE.fullmatch(mesh_sha256):
        raise ValueError("research provenance requires a lowercase SHA256 mesh hash")
    if isinstance(grid_res, bool) or not isinstance(grid_res, (int, np.integer)):
        raise TypeError("research provenance grid_res must be an integer")
    if int(grid_res) < 1:
        raise ValueError("research provenance grid_res must be positive")
    if angle_init not in ("random", "geometry"):
        raise ValueError("research provenance angle_init must be random or geometry")

    field_state = str(field_state)
    if termination is None:
        if re.fullmatch(r"step_[1-9][0-9]*", field_state) is None:
            raise ValueError(
                "checkpoint field_state must be step_<positive optimizer update>")
    else:
        expected_state = _TERMINAL_FIELD_STATES.get(str(termination))
        if expected_state is None:
            raise ValueError("unknown terminal termination state: {}".format(termination))
        if field_state != expected_state:
            raise ValueError(
                "terminal field_state {!r} does not match termination {!r}".format(
                    field_state, termination))

    provenance = {
        "seed": int(seed),
        "mesh_sha256": mesh_sha256,
        "grid_res": int(grid_res),
        "angle_init": angle_init,
        "field_state": field_state,
    }
    if termination is not None:
        provenance["termination"] = str(termination)
        provenance["automatic_terminal"] = termination == "automatic"

    if angle_init == "geometry":
        if not isinstance(geometry_initialization, Mapping):
            raise ValueError(
                "geometry angle initialization requires initialization evidence")
        provenance["geometry_initialization"] = copy.deepcopy(
            dict(geometry_initialization))
    elif geometry_initialization is not None:
        raise ValueError(
            "random angle initialization cannot carry geometry initialization evidence")
    return provenance


def attach_terminal_research_provenance(
        report: Dict[str, Any], *, enabled: bool, seed: int,
        mesh_sha256: str, grid_res: int, angle_init: str, termination: str,
        geometry_initialization: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Conditionally extend a terminal report without changing default output."""
    if not enabled:
        return report
    if "research_provenance" in report:
        raise ValueError("terminal report already contains research_provenance")
    field_state = _TERMINAL_FIELD_STATES.get(str(termination))
    if field_state is None:
        raise ValueError("unknown terminal termination state: {}".format(termination))
    report["research_provenance"] = build_research_field_provenance(
        seed=seed, mesh_sha256=mesh_sha256, grid_res=grid_res,
        angle_init=angle_init, field_state=field_state,
        termination=termination, geometry_initialization=geometry_initialization)
    return report


class StructuredResearchTrace:
    """Flush each opt-in JSONL event to an exclusively-created path."""
    def __init__(self, path: Optional[Path],
                 metadata: Optional[Mapping[str, Any]] = None) -> None:
        self.path = None if path is None else Path(path)
        self.enabled = self.path is not None
        self._sequence = 0
        self._stream = None
        self._atexit_registered = False
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8")
        atexit.register(self.close)
        self._atexit_registered = True
        self.record("trace_start", schema_version=RESEARCH_TRACE_SCHEMA,
                    metadata={} if metadata is None else metadata)

    def record(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        row = {"sequence": self._sequence, "event": str(event)}
        row.update(payload)
        self._stream.write(json.dumps(_json_ready(row), sort_keys=True,
                                      allow_nan=False) + "\n")
        self._stream.flush()
        self._sequence += 1

    def close(self) -> None:
        if self._stream is not None and not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


def _normalise_field(field: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(np.asarray(field, dtype=np.float32))
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 6:
        raise ValueError("research cross field must have shape F x 6")
    if not np.isfinite(value).all():
        raise ValueError("research cross field contains non-finite values")
    if (np.any(np.linalg.norm(value[:, :3], axis=1) <= 1e-12)
            or np.any(np.linalg.norm(value[:, 3:], axis=1) <= 1e-12)):
        raise ValueError("research cross field contains a zero direction")
    return value


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_no_replace(staging: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(str(staging), str(destination))
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-replace research publication requires renameat2")
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                              ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(str(staging)), -100,
                           os.fsencode(str(destination)), 1)
        if result != 0:
            number = ctypes.get_errno()
            if number == errno.EEXIST:
                raise FileExistsError(number, "refusing to replace research checkpoint",
                                      str(destination))
            raise OSError(number, os.strerror(number), str(destination))
        return
    raise RuntimeError("atomic no-replace research publication is unsupported")


def write_research_checkpoint(logdir: Path, step: int, cross_field: np.ndarray,
                              report: Mapping[str, Any]) -> Tuple[Path, Path, Dict[str, Any]]:
    """Atomically publish one post-update field/report directory."""
    step = int(step)
    if step < 1:
        raise ValueError("research checkpoint step must be positive")
    field = _normalise_field(cross_field)
    output_dir = research_checkpoint_directory(logdir, step)
    if os.path.lexists(str(output_dir)):
        raise FileExistsError("refusing to replace research checkpoint: {}".format(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".step-{:08d}-stage-".format(step),
                                    dir=str(output_dir.parent)))
    field_staging = staging / FIELD_FILENAME
    report_staging = staging / REPORT_FILENAME
    enriched = copy.deepcopy(dict(report))
    try:
        np.savetxt(str(field_staging), field)
        with field_staging.open("r+b") as target:
            target.flush()
            os.fsync(target.fileno())
        enriched.update({
            "schema_version": RESEARCH_CHECKPOINT_SCHEMA,
            "step": step,
            "artifact_commit": {"status": "complete",
                                "publication": "atomic_directory_rename_no_replace"},
            "cross_field": {
                "relative_path": "research_checkpoints/step_{:08d}/{}".format(
                    step, FIELD_FILENAME),
                "rows": int(field.shape[0]), "columns": 6,
                "dtype_in_memory": str(field.dtype),
                "sha256": sha256_file(field_staging),
                "state_mode": "post-optimizer post-schedule eval",
            },
        })
        payload = (json.dumps(_json_ready(enriched), indent=2, sort_keys=True,
                              allow_nan=False) + "\n").encode("utf-8")
        with report_staging.open("xb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        _fsync_directory(staging)
        _publish_directory_no_replace(staging, output_dir)
        _fsync_directory(output_dir.parent)
    finally:
        if staging.exists():
            shutil.rmtree(str(staging))
    return output_dir / FIELD_FILENAME, output_dir / REPORT_FILENAME, enriched


def _copy_numpy_rng_state(state: Tuple[Any, ...]) -> Tuple[Any, ...]:
    return state[0], state[1].copy(), state[2], state[3], state[4]


def _numpy_rng_equal(left: Tuple[Any, ...], right: Tuple[Any, ...]) -> bool:
    return (left[0] == right[0] and np.array_equal(left[1], right[1])
            and left[2:] == right[2:])


def _snapshot_runtime_state(net: Any, torch: Any) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _copy_numpy_rng_state(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": ([state.clone() for state in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else []),
        "module_flags": [(module, bool(module.training)) for module in net.modules()],
    }


def _restore_runtime_state(torch: Any, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    failures = []
    for index, (module, enabled) in enumerate(snapshot["module_flags"]):
        try:
            module.training = enabled
        except BaseException as error:
            failures.append(("module_flags[{}]".format(index), error))
    for stage, restore in (
        ("python_rng", lambda: random.setstate(snapshot["python"])),
        ("numpy_rng", lambda: np.random.set_state(snapshot["numpy"])),
        ("torch_cpu_rng", lambda: torch.set_rng_state(snapshot["torch_cpu"])),
    ):
        try:
            restore()
        except BaseException as error:
            failures.append((stage, error))
    if snapshot["torch_cuda"]:
        try:
            torch.cuda.set_rng_state_all(snapshot["torch_cuda"])
        except BaseException as error:
            failures.append(("torch_cuda_rng", error))
    if failures:
        raise RuntimeStateIntegrityError(
            "geometry prefit restoration raised for {}".format(
                ", ".join(stage for stage, _ in failures))) from failures[0][1]

    checks = {
        "python_rng_restored": random.getstate() == snapshot["python"],
        "numpy_global_rng_restored": _numpy_rng_equal(np.random.get_state(), snapshot["numpy"]),
        "torch_cpu_rng_restored": bool(torch.equal(torch.get_rng_state(), snapshot["torch_cpu"])),
        "torch_cuda_rng_restored": True,
        "module_training_flags_restored": all(
            module.training == enabled for module, enabled in snapshot["module_flags"]),
    }
    if snapshot["torch_cuda"]:
        current = torch.cuda.get_rng_state_all()
        checks["torch_cuda_rng_restored"] = (
            len(current) == len(snapshot["torch_cuda"])
            and all(bool(torch.equal(left, right))
                    for left, right in zip(current, snapshot["torch_cuda"])))
    if not all(checks.values()):
        raise RuntimeStateIntegrityError("geometry prefit failed to restore runtime state")
    checks["module_count"] = len(snapshot["module_flags"])
    return checks


def _angle_features_and_target(target: GeometryInitializationTarget, train_set: Any,
                               device: Any, torch: Any) -> Tuple[Any, Any]:
    arrays = [
        np.asarray(train_set.points, dtype=np.float32),
        np.asarray(train_set.mnfld_n, dtype=np.float32),
        np.asarray(train_set.vector_u, dtype=np.float32),
        np.asarray(train_set.vector_v, dtype=np.float32),
    ]
    expected = (target.face_count, 3)
    if any(value.shape != expected or not np.isfinite(value).all() for value in arrays):
        raise ValueError("dataset arrays do not match geometry initialization face order")
    features_np = np.ascontiguousarray(np.concatenate(arrays, axis=1), dtype=np.float32)
    tangent = np.asarray(target.tangents, dtype=np.float32)
    local_u, local_v = arrays[2], arrays[3]
    u_component = np.einsum("ij,ij->i", tangent, local_u)
    v_component = np.einsum("ij,ij->i", tangent, local_v)
    projected_norm = np.sqrt(np.square(u_component) + np.square(v_component))
    if np.any(projected_norm <= 1e-6) or not np.isfinite(projected_norm).all():
        raise ValueError("geometry tangent cannot be represented in dataset local basis")
    target_theta = np.arctan2(v_component, u_component).astype(np.float32)
    features = torch.from_numpy(features_np).unsqueeze(0).to(device)
    angles = torch.from_numpy(target_theta).reshape(1, target.face_count, 1).to(device)
    return features, angles


def _four_rosy_metrics(torch: Any, predicted: Any, target: Any) -> Tuple[Any, Any]:
    if predicted.shape != target.shape:
        raise RuntimeError("decoder_angle output shape does not match geometry target")
    delta4 = 4.0 * (predicted - target)
    loss = (1.0 - torch.cos(delta4)).mean()
    cross_error = torch.acos(torch.clamp(torch.cos(delta4), -1.0, 1.0)) / 4.0
    mean_degrees = torch.rad2deg(cross_error).mean()
    if not (bool(torch.isfinite(loss).item())
            and bool(torch.isfinite(mean_degrees).item())):
        raise FloatingPointError("geometry prefit produced a non-finite metric")
    return loss, mean_degrees


def prefit_geometry_angle_decoder(net: Any, target: GeometryInitializationTarget,
                                  train_set: Any, device: Any, learning_rate: float,
                                  max_steps: int, target_degrees: float) -> Dict[str, Any]:
    """Prefit only decoder_angle; restore RNGs and module flags exactly."""
    import torch

    learning_rate, max_steps = float(learning_rate), int(max_steps)
    target_degrees = float(target_degrees)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("geometry prefit learning rate must be finite and positive")
    if max_steps < 1:
        raise ValueError("geometry prefit max_steps must be positive")
    if not math.isfinite(target_degrees) or not 0.0 <= target_degrees <= 45.0:
        raise ValueError("geometry prefit target_degrees must lie in [0,45]")
    if not hasattr(net, "decoder_angle") or not hasattr(net, "decoder"):
        raise TypeError("network must expose decoder and decoder_angle")

    started_at = time.perf_counter()
    protected = [(name, parameter, parameter.detach().clone())
                 for name, parameter in net.named_parameters()
                 if not name.startswith("decoder_angle.")]
    angle_parameters = list(net.decoder_angle.parameters())
    if not angle_parameters:
        raise ValueError("decoder_angle has no trainable parameters")
    runtime_snapshot = _snapshot_runtime_state(net, torch)
    restoration = None
    report = None
    try:
        features, target_theta = _angle_features_and_target(
            target, train_set, device, torch)
        net.eval()
        prefit_optimizer = torch.optim.Adam(angle_parameters, lr=learning_rate,
                                            weight_decay=0.0)

        def evaluate() -> Tuple[Any, Any]:
            predicted = net.decoder_angle(features) * 2.0 * torch.pi
            return _four_rosy_metrics(torch, predicted, target_theta)

        with torch.no_grad():
            initial_loss, initial_error = evaluate()
        history = [{"step": 0, "loss": float(initial_loss.cpu().item()),
                    "mean_cross_error_degrees": float(initial_error.cpu().item())}]
        completed = 0
        current_error = initial_error
        while completed < max_steps and float(current_error.cpu().item()) > target_degrees:
            prefit_optimizer.zero_grad(set_to_none=True)
            current_loss, _ = evaluate()
            current_loss.backward()
            prefit_optimizer.step()
            completed += 1
            with torch.no_grad():
                current_loss, current_error = evaluate()
            history.append({"step": completed,
                            "loss": float(current_loss.cpu().item()),
                            "mean_cross_error_degrees": float(current_error.cpu().item())})

        changed = [name for name, parameter, before in protected
                   if not bool(torch.equal(parameter.detach(), before))]
        if changed:
            raise RuntimeError("geometry prefit changed non-angle parameters: {}".format(changed))
        report = {
            "schema_version": "trimesh2quad.neurcross.geometry_prefit.v1",
            "target": target.report(),
            "objective": "mean(1-cos(4*(theta_pred-theta_target)))",
            "target_coordinates": "dataset local (u,v), converted from world tangent",
            "learning_rate": learning_rate,
            "maximum_steps": max_steps,
            "target_mean_cross_error_degrees": target_degrees,
            "completed_updates": completed,
            "reached_target": history[-1]["mean_cross_error_degrees"] <= target_degrees,
            "initial": history[0], "final": history[-1], "history": history,
            "optimizer_scope": "decoder_angle.parameters only",
            "optimizer_discarded_before_main_training": True,
            "protected_parameter_tensors": len(protected),
            "siren_and_non_angle_parameters_unchanged": True,
        }
    finally:
        restoration = _restore_runtime_state(torch, runtime_snapshot)
    report["state_restoration"] = restoration
    report["elapsed_seconds"] = time.perf_counter() - started_at
    return report
