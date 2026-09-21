"""MIQ, topology, surface, quality, structure, size, and field metrics."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from weak_layout_pipeline.pipeline.sampling import area_samples, surface_distances

from weak_layout_pipeline.direction_field.adapter import DirectionField
from weak_layout_pipeline.direction_field.compatibility import fourfold_error
from weak_layout_pipeline.pipeline.mesh import (
    Mesh,
    SurfaceProjector,
    bbox_diagonal,
    canonical_edge,
    edge_topology,
    load_obj,
    triangle_geometry,
    unique_edges,
)


@dataclass(frozen=True)
class GateProfile:
    """Named acceptance thresholds for research and release evaluation."""

    name: str
    surface_p95_bbox: float
    surface_max_bbox: float
    minimum_scaled_jacobian: float
    minimum_scaled_jacobian_p05: float
    maximum_planarity_p95: float
    maximum_field_mean_degrees: float
    minimum_structure_coverage: float
    gate_structure: bool
    require_closed_topology: bool


GATE_PROFILES = {
    "strict": GateProfile(
        name="strict",
        surface_p95_bbox=1e-3,
        surface_max_bbox=1e-2,
        minimum_scaled_jacobian=0.0,
        minimum_scaled_jacobian_p05=0.25,
        maximum_planarity_p95=0.15,
        maximum_field_mean_degrees=15.0,
        minimum_structure_coverage=0.90,
        gate_structure=True,
        require_closed_topology=True,
    ),
    "moderate": GateProfile(
        name="moderate",
        surface_p95_bbox=2e-2,
        surface_max_bbox=5e-2,
        minimum_scaled_jacobian=0.0,
        minimum_scaled_jacobian_p05=0.20,
        maximum_planarity_p95=0.35,
        maximum_field_mean_degrees=25.0,
        minimum_structure_coverage=0.75,
        gate_structure=False,
        require_closed_topology=True,
    ),
}


def _read_matrix(path: Path, header: bool = False) -> np.ndarray:
    if not path.is_file() or path.stat().st_size == 0:
        return np.empty((0, 0))
    return np.loadtxt(path, skiprows=1 if header else 0, ndmin=2)


def miq_metrics(
    directory: str | Path, expected_faces: int | None = None
) -> dict[str, object]:
    root = Path(directory)
    uv = _read_matrix(root / "miq_uv.txt", header=True)
    raw_fuv = _read_matrix(root / "miq_fuv.txt", header=True)
    if (
        uv.ndim != 2
        or uv.shape[1] != 2
        or not len(uv)
        or raw_fuv.ndim != 2
        or raw_fuv.shape[1] != 3
        or not len(raw_fuv)
        or not np.isfinite(uv).all()
        or not np.isfinite(raw_fuv).all()
        or not np.equal(raw_fuv, np.floor(raw_fuv)).all()
        or (raw_fuv < 0).any()
        or (raw_fuv >= len(uv)).any()
    ):
        raise ValueError("missing, empty, nonfinite, or invalid MIQ UV/FUV data")
    fuv = raw_fuv.astype(int)
    for filename, array in (("miq_uv.txt", uv), ("miq_fuv.txt", fuv)):
        declared = tuple(
            map(int, (root / filename).read_text().splitlines()[0].split())
        )
        if declared != array.shape:
            raise ValueError("MIQ matrix header does not match its data")
    if expected_faces is not None and len(fuv) != expected_faces:
        raise ValueError("MIQ FUV face count differs from the source mesh")
    signed = np.empty(0)
    if len(fuv) and len(uv):
        tri = uv[fuv]
        signed = 0.5 * (
            (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
            - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0])
        )
    scale = max(float(np.median(np.abs(signed))) if len(signed) else 0.0, 1e-30)
    nonzero = signed[np.abs(signed) > 1e-12 * scale]
    orientation = np.sign(np.median(nonzero)) if len(nonzero) else 1.0
    singular = _read_matrix(root / "miq_singularities_detailed.txt")
    seams = _read_matrix(root / "miq_seams.txt")
    return {
        "data_valid": True,
        "uv_vertex_count": len(uv),
        "fuv_face_count": len(fuv),
        "zero_uv_faces": int(np.count_nonzero(signed == 0)),
        "near_zero_uv_faces": int(np.count_nonzero(np.abs(signed) <= 1e-10 * scale)),
        "orientation_flips": int(
            np.count_nonzero(signed * orientation < -1e-12 * scale)
        ),
        "singularities": (
            int(np.count_nonzero(singular[:, 1])) if singular.shape[1] >= 2 else 0
        ),
        "seam_entries": int(np.count_nonzero(seams)) if len(seams) else 0,
    }


def topology_metrics(mesh: Mesh) -> dict[str, object]:
    incidents, adjacency = edge_topology(mesh)
    face_graph: list[list[int]] = [[] for _ in range(mesh.face_count)]
    for left, right, _ in adjacency:
        face_graph[left].append(right)
        face_graph[right].append(left)
    seen: set[int] = set()
    components = 0
    for start in range(mesh.face_count):
        if start in seen:
            continue
        components += 1
        queue = [start]
        seen.add(start)
        while queue:
            for nxt in face_graph[queue.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    duplicate = mesh.face_count - len(
        {tuple(sorted(map(int, face))) for face in mesh.faces}
    )
    degenerate = sum(len(set(map(int, face))) != len(face) for face in mesh.faces)
    nonmanifold_edges = sum(len(rows) > 2 for rows in incidents.values())
    boundary_edges = sum(len(rows) == 1 for rows in incidents.values())
    vertex_faces: dict[int, list[int]] = defaultdict(list)
    for face_id, face in enumerate(mesh.faces):
        for vertex in face:
            vertex_faces[int(vertex)].append(face_id)
    nonmanifold_vertices = 0
    for vertex, face_ids in vertex_faces.items():
        local = set(face_ids)
        if not local:
            continue
        reached = {face_ids[0]}
        queue = [face_ids[0]]
        while queue:
            for nxt in face_graph[queue.pop()]:
                if nxt in local and nxt not in reached:
                    reached.add(nxt)
                    queue.append(nxt)
        if reached != local:
            nonmanifold_vertices += 1
    return {
        "pure_quad": mesh.faces.shape[1] == 4,
        "closed": boundary_edges == 0,
        "connected": components == 1,
        "connected_components": components,
        "edge_manifold": nonmanifold_edges == 0,
        "vertex_manifold": nonmanifold_vertices == 0,
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "nonmanifold_vertices": nonmanifold_vertices,
        "degenerate_faces": degenerate,
        "duplicate_faces": duplicate,
        "euler_characteristic": mesh.vertex_count - len(incidents) + mesh.face_count,
        "quad_count": mesh.face_count,
        "vertex_count": mesh.vertex_count,
    }


def _triangulate(quads: Mesh) -> Mesh:
    if quads.faces.shape[1] == 3:
        return quads
    faces = np.vstack((quads.faces[:, [0, 1, 2]], quads.faces[:, [0, 2, 3]]))
    return Mesh(quads.vertices, faces)


def surface_metrics(
    source: Mesh, quad: Mesh, *, sample_count: int = 2048, seed: int = 947
) -> dict:
    diagonal = bbox_diagonal(source)
    source_points = area_samples(source, sample_count, seed)[0]
    quad_points = area_samples(quad, sample_count, seed + 1)[0]
    s_to_q = surface_distances(source_points, quad) / diagonal
    q_to_s = surface_distances(quad_points, source) / diagonal

    def stats(values):
        return {
            "rms": float(np.sqrt(np.mean(values**2))),
            "p95": float(np.percentile(values, 95)),
            "sample_max": float(values.max()),
        }

    a, b = stats(s_to_q), stats(q_to_s)
    return {
        "source_to_quad": a,
        "quad_to_source": b,
        "symmetric_rms_bbox": float(np.sqrt((a["rms"] ** 2 + b["rms"] ** 2) / 2)),
        "symmetric_p95_bbox": max(a["p95"], b["p95"]),
        "symmetric_max_bbox": max(a["sample_max"], b["sample_max"]),
        "sample_count_per_direction": sample_count,
        "seed": seed,
        "definition": "area sampling; maximum of directional P95; sample_max is not exact Hausdorff",
    }


def quality_metrics(quad: Mesh, source: Mesh | None = None) -> dict[str, float]:
    """Return planarity and robust corner scaled Jacobians.

    A single source normal at the quad centroid is ambiguous when the quad
    crosses a sharp model edge.  The signed test therefore combines:

    * the intrinsic Newell-normal orientation, which detects concave or
      self-intersecting corners; and
    * the best alignment against nearby source-face normals, which detects a
      globally reversed winding without using the wrong side of a crease.
    """
    planarity = []
    jacobians = []
    face_minima = []
    intrinsic_jacobians = []
    source_jacobians = []
    source_tree = None
    source_normals = None
    if source is not None:
        source_normals, _, source_centers = triangle_geometry(source)
        source_tree = cKDTree(source_centers)
    for face in quad.faces:
        points = quad.vertices[face]
        if len(points) != 4:
            continue
        edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        scale = max(float(edge_lengths.mean()), 1e-30)
        normal = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
        if np.linalg.norm(normal) <= 1e-30:
            normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = max(float(np.linalg.norm(normal)), 1e-30)
        normal = normal / norm
        centered = points - points.mean(axis=0)
        planarity.append(float(np.max(np.abs(centered @ normal))) / scale)
        nearby_normals = np.empty((0, 3), dtype=float)
        if source_tree is not None and source_normals is not None:
            count = min(24, len(source_normals))
            _, nearby_ids = source_tree.query(points.mean(axis=0), k=count)
            nearby_normals = source_normals[np.atleast_1d(nearby_ids).astype(int)]
            if np.dot(nearby_normals, normal).max() < 0.0:
                normal = -normal
        for corner in range(len(points)):
            before = points[(corner - 1) % 4] - points[corner]
            after = points[(corner + 1) % len(points)] - points[corner]
            cross = np.cross(after, before)
            denominator = max(np.linalg.norm(before) * np.linalg.norm(after), 1e-30)
            intrinsic = float(np.dot(cross, normal) / denominator)
            if len(nearby_normals):
                source_aligned = max(
                    float(np.dot(cross, candidate) / denominator)
                    for candidate in nearby_normals
                )
            else:
                source_aligned = intrinsic
            intrinsic_jacobians.append(intrinsic)
            source_jacobians.append(source_aligned)
            corner_value = min(intrinsic, source_aligned)
            jacobians.append(corner_value)
        face_minima.append(min(jacobians[-4:]))
    p = np.asarray(planarity)
    j = np.asarray(jacobians)
    if not len(p) or not len(j):
        return {
            "planarity_mean": math.inf,
            "planarity_p95": math.inf,
            "planarity_max": math.inf,
            "scaled_jacobian_min": -math.inf,
            "scaled_jacobian_p05": -math.inf,
            "scaled_jacobian_mean": -math.inf,
            "inverted_or_concave_corners": 0,
            "negative_quad_count": 0,
            "nonpositive_quad_count": 0,
            "intrinsic_scaled_jacobian_min": -math.inf,
            "intrinsic_scaled_jacobian_p05": -math.inf,
            "source_aligned_scaled_jacobian_min": -math.inf,
            "source_aligned_scaled_jacobian_p05": -math.inf,
        }
    intrinsic = np.asarray(intrinsic_jacobians)
    source_aligned = np.asarray(source_jacobians)
    face_values = np.asarray(face_minima)
    return {
        "planarity_mean": float(p.mean()),
        "planarity_p95": float(np.percentile(p, 95)),
        "planarity_max": float(p.max()),
        "scaled_jacobian_min": float(j.min()),
        "scaled_jacobian_p05": float(np.percentile(j, 5)),
        "scaled_jacobian_mean": float(j.mean()),
        "inverted_or_concave_corners": int(np.count_nonzero(j <= 0.0)),
        "negative_quad_count": int(np.count_nonzero(face_values < 0.0)),
        "nonpositive_quad_count": int(np.count_nonzero(face_values <= 0.0)),
        "intrinsic_scaled_jacobian_min": float(intrinsic.min()),
        "intrinsic_scaled_jacobian_p05": float(np.percentile(intrinsic, 5)),
        "source_aligned_scaled_jacobian_min": float(source_aligned.min()),
        "source_aligned_scaled_jacobian_p05": float(np.percentile(source_aligned, 5)),
    }


def _edge_samples(
    source: Mesh, quad: Mesh
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = unique_edges(quad)
    vectors = quad.vertices[edges[:, 1]] - quad.vertices[edges[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    midpoints = 0.5 * (quad.vertices[edges[:, 0]] + quad.vertices[edges[:, 1]])
    _, _, centers = triangle_geometry(source)
    face_ids = cKDTree(centers).query(midpoints)[1].astype(int)
    return vectors, lengths, face_ids


def field_alignment_metrics(
    source: Mesh, quad: Mesh, field: DirectionField
) -> dict[str, float]:
    vectors, _, face_ids = _edge_samples(source, quad)
    errors = np.asarray(
        [
            fourfold_error(vector, field.pd1[face], field.pd2[face])[0]
            for vector, face in zip(vectors, face_ids)
        ]
    )
    return {
        "mean_degrees": float(errors.mean()),
        "median_degrees": float(np.median(errors)),
        "p95_degrees": float(np.percentile(errors, 95)),
    }


def size_response_metrics(
    source: Mesh, quad: Mesh, rho: np.ndarray, size: np.ndarray
) -> dict[str, object]:
    _, lengths, face_ids = _edge_samples(source, quad)
    ratios = lengths / size[face_ids]
    log_density = np.log(rho[face_ids])
    negative_log_length = -np.log(np.maximum(lengths, 1e-30))
    correlation = (
        spearmanr(log_density, negative_log_length).statistic
        if np.ptp(log_density) > 1.0e-14 and np.ptp(negative_log_length) > 1.0e-14
        else math.nan
    )
    bins = np.asarray(np.quantile(rho[face_ids], [0, 1 / 3, 2 / 3, 1]), dtype=float)
    regions: list[dict[str, object]] = []
    for index in range(3):
        mask = (rho[face_ids] >= bins[index]) & (
            rho[face_ids] <= bins[index + 1]
            if index == 2
            else rho[face_ids] < bins[index + 1]
        )
        regions.append(
            {
                "rho_range": [float(bins[index]), float(bins[index + 1])],
                "edge_count": int(mask.sum()),
                "median_edge_length": (
                    float(np.median(lengths[mask])) if mask.any() else None
                ),
            }
        )
    return {
        "definition": "actual unique quad edge length / local target size at the nearest source face",
        "ratio_median": float(np.median(ratios)),
        "ratio_p05": float(np.percentile(ratios, 5)),
        "ratio_p95": float(np.percentile(ratios, 95)),
        "log_ratio_rmse": float(
            np.sqrt(np.mean(np.log(np.maximum(ratios, 1e-30)) ** 2))
        ),
        "spearman_log_rho_vs_negative_log_edge_length": (
            float(correlation) if np.isfinite(correlation) else None
        ),
        "density_regions": regions,
    }


def _nearest_polyline_segment(
    point: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    midpoint_tree: cKDTree | None = None,
    max_half_length: float | None = None,
) -> tuple[float, int]:
    candidate_ids: np.ndarray | None = None
    if midpoint_tree is not None and len(starts) > 1:
        # Start with one segment to obtain an upper bound.  A segment whose
        # midpoint is farther away than ``bound + half_length`` cannot improve
        # that bound (triangle inequality), so this radius query remains exact
        # while avoiding a full scan for ordinary, similarly sized quad edges.
        _, seed_id = midpoint_tree.query(point, k=1)
        seed_id = int(seed_id)
        seed_start = starts[seed_id]
        seed_vector = ends[seed_id] - seed_start
        seed_parameter = float(
            np.dot(point - seed_start, seed_vector)
            / max(float(np.dot(seed_vector, seed_vector)), 1.0e-30)
        )
        seed_closest = seed_start + np.clip(seed_parameter, 0.0, 1.0) * seed_vector
        bound = float(np.linalg.norm(seed_closest - point))
        radius_padding = (
            max_half_length
            if max_half_length is not None
            else 0.5 * float(np.linalg.norm(ends - starts, axis=1).max(initial=0.0))
        )
        raw_ids = midpoint_tree.query_ball_point(point, bound + radius_padding)
        candidate_values = np.asarray(raw_ids, dtype=int)
        if candidate_values.size == 0:
            candidate_values = np.asarray([seed_id], dtype=int)
        candidate_ids = candidate_values
        local_starts = starts[candidate_ids]
        local_ends = ends[candidate_ids]
    else:
        local_starts = starts
        local_ends = ends
    vectors = local_ends - local_starts
    denominator = np.maximum(np.einsum("ij,ij->i", vectors, vectors), 1.0e-30)
    parameter = np.einsum("ij,ij->i", point - local_starts, vectors) / denominator
    closest = local_starts + np.clip(parameter, 0.0, 1.0)[:, None] * vectors
    distances = np.linalg.norm(closest - point, axis=1)
    local_index = int(np.argmin(distances))
    index = (
        int(candidate_ids[local_index]) if candidate_ids is not None else local_index
    )
    return float(distances[local_index]), index


def _arc_groups(structures: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if "selected_arcs" in structures:
        selected = structures.get("selected_arcs", ())
        if not isinstance(selected, (list, tuple)):
            return {"selected": []}
        return {
            "selected": [
                dict(row.get("arc", row))
                for row in selected
                if isinstance(row, Mapping) and isinstance(row.get("arc", row), Mapping)
            ]
        }
    raw_groups = structures.get("roles", {})
    if not isinstance(raw_groups, Mapping):
        return {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for name, raw_arcs in raw_groups.items():
        if not isinstance(raw_arcs, (list, tuple)):
            continue
        groups[str(name)] = [dict(arc) for arc in raw_arcs if isinstance(arc, Mapping)]
    return groups


def structure_metrics(
    source: Mesh, quad: Mesh, roles: dict[str, object], h0: float
) -> dict[str, object]:
    quad_edges = unique_edges(quad)
    starts = quad.vertices[quad_edges[:, 0]]
    ends = quad.vertices[quad_edges[:, 1]]
    midpoint_tree = cKDTree(0.5 * (starts + ends))
    max_half_length = 0.5 * float(
        np.linalg.norm(ends - starts, axis=1).max(initial=0.0)
    )
    result: dict[str, object] = {}
    for role, arcs in _arc_groups(roles).items():
        distances: list[float] = []
        angles: list[float] = []
        for arc in arcs:
            for u, v in arc.get("mesh_edges", []):
                tangent = source.vertices[int(v)] - source.vertices[int(u)]
                for alpha in np.linspace(0.0, 1.0, 5):
                    point = (1.0 - alpha) * source.vertices[
                        int(u)
                    ] + alpha * source.vertices[int(v)]
                    distance, edge_id = _nearest_polyline_segment(
                        point, starts, ends, midpoint_tree, max_half_length
                    )
                    direction = ends[edge_id] - starts[edge_id]
                    cosine = np.clip(
                        abs(np.dot(tangent, direction))
                        / max(
                            np.linalg.norm(tangent) * np.linalg.norm(direction), 1e-30
                        ),
                        0,
                        1,
                    )
                    distances.append(float(distance))
                    angles.append(float(np.degrees(np.arccos(cosine))))
        d = np.asarray(distances)
        a = np.asarray(angles)
        result[role] = {
            "sample_count": len(d),
            "coverage_within_h0": float(np.mean(d <= h0)) if len(d) else None,
            "alignment_median_bbox": (
                float(np.median(d) / bbox_diagonal(source)) if len(d) else None
            ),
            "alignment_p95_bbox": (
                float(np.percentile(d, 95) / bbox_diagonal(source)) if len(d) else None
            ),
            "edge_flow_angle_mean_degrees": float(a.mean()) if len(a) else None,
        }
    return result


def evaluate_run(
    source_path: str | Path,
    quad_path: str | Path,
    miq_dir: str | Path,
    field: DirectionField,
    roles: dict[str, object],
    rho: np.ndarray,
    size: np.ndarray,
    h0: float,
    output: str | Path | None = None,
    gate_profile: str = "moderate",
) -> dict[str, object]:
    if gate_profile not in GATE_PROFILES:
        raise ValueError(
            f"unknown gate profile {gate_profile!r}; "
            f"expected one of {sorted(GATE_PROFILES)}"
        )
    profile = GATE_PROFILES[gate_profile]
    source = load_obj(source_path)
    quad = load_obj(quad_path, require_triangles=False)
    report: dict[str, Any] = {
        "schema_version": "weak-layout.evaluation.v3",
        "miq": miq_metrics(miq_dir, expected_faces=source.face_count),
        "topology": topology_metrics(quad),
        "surface": surface_metrics(source, quad),
        "quality": quality_metrics(quad, source),
        "structure": structure_metrics(source, quad, roles, h0),
        "size_fidelity": size_response_metrics(source, quad, rho, size),
        "field_alignment": field_alignment_metrics(source, quad, field),
    }
    # Backward-compatible alias for legacy reports and comparison scripts.
    report["size_response"] = report["size_fidelity"]
    topology = report["topology"]
    source_topology = topology_metrics(source)
    report["source_topology"] = source_topology
    topology["matches_source_euler"] = (
        topology["euler_characteristic"] == source_topology["euler_characteristic"]
    )
    surface = report["surface"]
    quality = report["quality"]
    uv = report["miq"]
    field_metrics = report["field_alignment"]
    structure_groups = report["structure"]
    structure_samples = sum(
        int(row.get("sample_count", 0)) for row in structure_groups.values()
    )
    structure_coverage = [
        float(row["coverage_within_h0"])
        for row in structure_groups.values()
        if row.get("coverage_within_h0") is not None
    ]
    category_gates = {
        "topology": bool(
            topology["pure_quad"]
            and topology["connected"]
            and topology["edge_manifold"]
            and topology["vertex_manifold"]
            and topology["degenerate_faces"] == 0
            and topology["duplicate_faces"] == 0
            and topology["matches_source_euler"]
            and (topology["closed"] or not profile.require_closed_topology)
        ),
        "surface": bool(
            surface["symmetric_p95_bbox"] <= profile.surface_p95_bbox
            and surface["symmetric_max_bbox"] <= profile.surface_max_bbox
        ),
        "uv": bool(
            uv["data_valid"]
            and (
                profile.name != "strict"
                or (uv["near_zero_uv_faces"] == 0 and uv["orientation_flips"] == 0)
            )
        ),
        "structure": bool(
            not profile.gate_structure
            or structure_samples == 0
            or (
                structure_coverage
                and min(structure_coverage) >= profile.minimum_structure_coverage
            )
        ),
        "field": bool(
            field_metrics["mean_degrees"] <= profile.maximum_field_mean_degrees
        ),
        "quality": bool(
            quality["scaled_jacobian_min"] > profile.minimum_scaled_jacobian
            and quality["scaled_jacobian_p05"] >= profile.minimum_scaled_jacobian_p05
            and quality["planarity_p95"] <= profile.maximum_planarity_p95
        ),
    }
    strict_profile = GATE_PROFILES["strict"]
    strict_category_gates = {
        "topology": bool(
            topology["pure_quad"]
            and topology["connected"]
            and topology["edge_manifold"]
            and topology["vertex_manifold"]
            and topology["degenerate_faces"] == 0
            and topology["duplicate_faces"] == 0
            and topology["matches_source_euler"]
            and (topology["closed"] or not strict_profile.require_closed_topology)
        ),
        "surface": bool(
            surface["symmetric_p95_bbox"] <= strict_profile.surface_p95_bbox
            and surface["symmetric_max_bbox"] <= strict_profile.surface_max_bbox
        ),
        "uv": bool(uv["near_zero_uv_faces"] == 0 and uv["orientation_flips"] == 0),
        "structure": bool(
            structure_samples == 0
            or (
                structure_coverage
                and min(structure_coverage) >= strict_profile.minimum_structure_coverage
            )
        ),
        "field": bool(
            field_metrics["mean_degrees"] <= strict_profile.maximum_field_mean_degrees
        ),
        "quality": bool(
            quality["scaled_jacobian_min"] > strict_profile.minimum_scaled_jacobian
            and quality["scaled_jacobian_p05"]
            >= strict_profile.minimum_scaled_jacobian_p05
            and quality["planarity_p95"] <= strict_profile.maximum_planarity_p95
        ),
    }
    validity = {
        "topology": category_gates["topology"],
        "orientation": bool(
            quality["scaled_jacobian_min"] > 0.0
            and quality["inverted_or_concave_corners"] == 0
        ),
        "uv_data": bool(uv["data_valid"]),
    }
    validity["basic_valid"] = all(validity.values())
    report["validation_limitations"] = [
        "Global self-intersections are not certified.",
        "Surface errors use finite area samples and quads split along diagonal 0--2.",
    ]
    report["gate_profile"] = profile.name
    report["profile_thresholds"] = {
        "surface_p95_bbox": profile.surface_p95_bbox,
        "surface_max_bbox": profile.surface_max_bbox,
        "minimum_scaled_jacobian": profile.minimum_scaled_jacobian,
        "minimum_scaled_jacobian_p05": profile.minimum_scaled_jacobian_p05,
        "maximum_planarity_p95": profile.maximum_planarity_p95,
        "maximum_field_mean_degrees": profile.maximum_field_mean_degrees,
        "minimum_structure_coverage": profile.minimum_structure_coverage,
        "gate_structure": profile.gate_structure,
        "require_closed_topology": profile.require_closed_topology,
    }
    report["validity"] = validity
    report["category_gates"] = category_gates
    report["strict_category_gates"] = strict_category_gates
    report["failure_categories"] = [
        name for name, passed in category_gates.items() if not passed
    ]
    report["strict_failure_categories"] = [
        name for name, passed in strict_category_gates.items() if not passed
    ]
    report["gates"] = dict(category_gates)
    report["success"] = all(category_gates.values())
    report["strict_success"] = all(strict_category_gates.values())
    if output:
        Path(output).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report
