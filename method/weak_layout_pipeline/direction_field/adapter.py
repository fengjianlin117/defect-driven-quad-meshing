"""Face-order-preserving NeurCross direction-field adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from weak_layout_pipeline.pipeline.mesh import Mesh, triangle_geometry


@dataclass(frozen=True)
class DirectionField:
    pd1: np.ndarray
    pd2: np.ndarray
    normals: np.ndarray
    audit: dict[str, object]


def _read_rows(path: str | Path, face_count: int) -> np.ndarray:
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape == (face_count, 3):
        return rows.astype(float)
    if rows.shape == (face_count, 4):
        ids = rows[:, 0].astype(np.int64)
        if not np.array_equal(ids, np.arange(face_count)):
            raise ValueError(f"field row ids do not preserve OBJ face order: {path}")
        return rows[:, 1:4].astype(float)
    raise ValueError(f"expected F x 3 or indexed F x 4 field, got {rows.shape}: {path}")


def _from_six(path: str | Path, face_count: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape == (face_count, 6):
        return rows[:, :3].astype(float), rows[:, 3:].astype(float)
    if rows.shape == (face_count, 7):
        ids = rows[:, 0].astype(np.int64)
        if not np.array_equal(ids, np.arange(face_count)):
            raise ValueError(f"F x 6 field row ids do not preserve face order: {path}")
        return rows[:, 1:4].astype(float), rows[:, 4:7].astype(float)
    raise ValueError(f"expected F x 6 or indexed F x 7 field, got {rows.shape}: {path}")


def load_direction_field(
    mesh: Mesh,
    pd1_path: str | Path,
    pd2_path: str | Path | None = None,
    *,
    tangent_tolerance: float = 1e-5,
) -> DirectionField:
    normals, _, _ = triangle_geometry(mesh)
    if pd2_path is None:
        raw1, raw2 = _from_six(pd1_path, mesh.face_count)
    else:
        raw1, raw2 = _read_rows(pd1_path, mesh.face_count), _read_rows(pd2_path, mesh.face_count)
    if not np.all(np.isfinite(raw1)) or not np.all(np.isfinite(raw2)):
        raise ValueError("direction field contains NaN or infinity")
    raw_norm1 = np.linalg.norm(raw1, axis=1)
    raw_norm2 = np.linalg.norm(raw2, axis=1)
    if np.any(raw_norm1 <= 1e-12) or np.any(raw_norm2 <= 1e-12):
        raise ValueError("direction field contains zero vectors")
    normal_component1 = np.abs(np.einsum("ij,ij->i", raw1, normals)) / raw_norm1
    normal_component2 = np.abs(np.einsum("ij,ij->i", raw2, normals)) / raw_norm2
    tangent1 = raw1 - np.einsum("ij,ij->i", raw1, normals)[:, None] * normals
    norms = np.linalg.norm(tangent1, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("PD1 projects to zero in at least one face")
    pd1 = tangent1 / norms[:, None]
    canonical_pd2 = np.cross(normals, pd1)
    tangent2 = raw2 - np.einsum("ij,ij->i", raw2, normals)[:, None] * normals
    orientation = np.sign(np.einsum("ij,ij->i", tangent2, canonical_pd2))
    orientation[orientation == 0] = 1
    pd2 = canonical_pd2 * orientation[:, None]
    orthogonality = np.abs(np.einsum("ij,ij->i", pd1, pd2))
    audit = {
        "schema_version": "weak-layout.direction-field-audit.v1",
        "face_count": mesh.face_count,
        "face_order_exact": True,
        "all_finite": True,
        "all_nonzero": True,
        "raw_pd1_max_normal_component": float(normal_component1.max()),
        "raw_pd2_max_normal_component": float(normal_component2.max()),
        "tangent_tolerance": tangent_tolerance,
        "raw_tangent_check_passed": bool(max(normal_component1.max(), normal_component2.max()) <= tangent_tolerance),
        "adapted_pd1_norm_max_error": float(np.max(np.abs(np.linalg.norm(pd1, axis=1) - 1))),
        "adapted_pd2_norm_max_error": float(np.max(np.abs(np.linalg.norm(pd2, axis=1) - 1))),
        "adapted_orthogonality_max_error": float(orthogonality.max()),
        "fourfold_valid": True,
        "pd2_orientation_flips_to_match_input": int(np.count_nonzero(orientation < 0)),
    }
    return DirectionField(pd1, pd2, normals, audit)


def write_indexed(path: str | Path, values: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(values):
            stream.write(f"{index} {row[0]:.15g} {row[1]:.15g} {row[2]:.15g}\n")
