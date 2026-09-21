"""Optional narrow-band complex fourfold field reconciliation."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from weak_layout_pipeline.direction_field.adapter import DirectionField
from weak_layout_pipeline.pipeline.mesh import Mesh, canonical_edge, edge_topology, face_neighbors


def _basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.tile(np.array([1.0, 0.0, 0.0]), (len(normals), 1))
    use_y = np.abs(normals[:, 0]) > 0.85
    axis[use_y] = [0.0, 1.0, 0.0]
    first = axis - np.einsum("ij,ij->i", axis, normals)[:, None] * normals
    first /= np.linalg.norm(first, axis=1)[:, None]
    return first, np.cross(normals, first)


def _angles(vectors: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    return np.arctan2(np.einsum("ij,ij->i", vectors, e2), np.einsum("ij,ij->i", vectors, e1))


def local_reconcile(
    mesh: Mesh,
    field: DirectionField,
    graph: dict[str, object],
    *,
    band_rings: int = 2,
    observation_weight: float = 1.0,
    smooth_weight: float = 0.35,
    graph_weight: float = 1.5,
) -> tuple[DirectionField, dict[str, object]]:
    incidents, adjacency = edge_topology(mesh)
    neighbors = face_neighbors(mesh)
    guide = np.zeros(mesh.face_count, dtype=complex)
    guide_weight = np.zeros(mesh.face_count)
    seeds: set[int] = set()
    for arc in graph.get("arcs", []):
        if arc.get("source") != "geometry" and arc.get("source") != "field":
            continue
        confidence = float(arc.get("confidence", 1.0))
        for edge in arc.get("mesh_edges", arc.get("edges", [])):
            u, v = int(edge[0]), int(edge[1])
            tangent3 = mesh.vertices[v] - mesh.vertices[u]
            for face, _ in incidents.get(canonical_edge(u, v), []):
                seeds.add(face)
                tangent = tangent3 - np.dot(tangent3, field.normals[face]) * field.normals[face]
                tangent /= max(float(np.linalg.norm(tangent)), 1e-30)
                e1, e2 = _basis(field.normals[face:face + 1])
                theta = _angles(tangent[None, :], e1, e2)[0]
                guide[face] += confidence * np.exp(4j * theta)
                guide_weight[face] += confidence
    band = set(seeds)
    frontier = set(seeds)
    for _ in range(band_rings):
        frontier = {g for face in frontier for g in neighbors[face]} - band
        band |= frontier
    if not band:
        return field, {"mode": "local-reconcile", "adjusted_face_count": 0, "reason": "empty structural band"}
    e1, e2 = _basis(field.normals)
    theta0 = _angles(field.pd1, e1, e2)
    z0 = np.exp(4j * theta0)
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    rhs = observation_weight * z0.copy()
    diagonal = np.full(mesh.face_count, observation_weight)
    for left, right, _ in adjacency:
        if left not in band or right not in band:
            continue
        # Complex transport uses the common 3-D representative projected into
        # each deterministic local tangent basis.
        transported = field.pd1[right] - np.dot(field.pd1[right], field.normals[left]) * field.normals[left]
        if np.linalg.norm(transported) <= 1e-12:
            phase = 1.0 + 0j
        else:
            transported /= np.linalg.norm(transported)
            target_angle = np.arctan2(np.dot(transported, e2[left]), np.dot(transported, e1[left]))
            phase = np.exp(4j * (theta0[left] - target_angle))
        diagonal[left] += smooth_weight
        diagonal[right] += smooth_weight
        rows += [left, right]
        cols += [right, left]
        data += [-smooth_weight * phase, -smooth_weight * np.conjugate(phase)]
    for face in seeds:
        if guide_weight[face] > 0:
            target = guide[face] / max(abs(guide[face]), 1e-30)
            weight = graph_weight * guide_weight[face]
            diagonal[face] += weight
            rhs[face] += weight * target
    rows.extend(range(mesh.face_count)); cols.extend(range(mesh.face_count)); data.extend(diagonal.tolist())
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(mesh.face_count, mesh.face_count))
    solved = spsolve(matrix, rhs)
    solved /= np.maximum(np.abs(solved), 1e-30)
    theta = np.angle(solved) / 4.0
    pd1 = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
    pd2 = np.cross(field.normals, pd1)
    change = np.degrees(np.abs(np.angle(solved / z0))) / 4.0
    outside = np.ones(mesh.face_count, dtype=bool)
    outside[list(band)] = False
    pd1[outside] = field.pd1[outside]
    pd2[outside] = field.pd2[outside]
    audit = dict(field.audit)
    audit["local_reconcile"] = True
    report = {
        "schema_version": "weak-layout.local-reconcile.v1",
        "mode": "local-reconcile",
        "adjusted_face_count": len(band),
        "seed_face_count": len(seeds),
        "band_rings": band_rings,
        "angle_change_mean_degrees": float(change[list(band)].mean()),
        "angle_change_p95_degrees": float(np.percentile(change[list(band)], 95)),
        "angle_change_max_degrees": float(change[list(band)].max()),
        "singularity_change": "measured by backend mismatch comparison",
    }
    return DirectionField(pd1, pd2, field.normals, audit), report
