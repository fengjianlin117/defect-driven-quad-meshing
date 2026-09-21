"""Per-arc geometry, field, and size-realizability evidence.

The source label records where an arc came from; it is not a backend role.
Every selected arc is ultimately exported as a hard edge.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from weak_layout_pipeline.direction_field.adapter import DirectionField
from weak_layout_pipeline.direction_field.compatibility import audit_graph_compatibility
from weak_layout_pipeline.pipeline.mesh import (
    Mesh,
    bbox_diagonal,
    canonical_edge,
    edge_topology,
    triangle_geometry,
)


def arc_id(arc: Mapping[str, Any], ordinal: int = 0) -> str:
    return str(arc.get("arc_id", arc.get("id", f"arc-{ordinal:06d}")))


def arc_edges(arc: Mapping[str, Any]) -> list[tuple[int, int]]:
    return [
        canonical_edge(int(edge[0]), int(edge[1]))
        for edge in arc.get("mesh_edges", arc.get("edges", ()))
    ]


def arc_vertices(arc: Mapping[str, Any]) -> list[int]:
    return [int(value) for value in arc.get("mesh_vertices", arc.get("vertices", ()))]


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not len(values):
        return 0.0
    order = np.argsort(values, kind="stable")
    values = values[order]
    cumulative = np.cumsum(weights[order])
    index = min(
        int(np.searchsorted(cumulative, q * float(cumulative[-1]), side="left")),
        len(values) - 1,
    )
    return float(values[index])


def _geometry_measurements(
    mesh: Mesh,
    arc: Mapping[str, Any],
    incidents: Mapping[tuple[int, int], list[tuple[int, int]]],
) -> dict[str, float | bool]:
    normals, _, _ = triangle_geometry(mesh)
    lengths: list[float] = []
    angles: list[float] = []
    for edge in arc_edges(arc):
        length = float(np.linalg.norm(mesh.vertices[edge[1]] - mesh.vertices[edge[0]]))
        rows = incidents.get(edge, [])
        if len(rows) == 2:
            angle = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.dot(normals[rows[0][0]], normals[rows[1][0]]), -1.0, 1.0
                        )
                    )
                )
            )
        else:
            angle = 0.0
        lengths.append(length)
        angles.append(angle)
    length_array = np.asarray(lengths, dtype=float)
    angle_array = np.asarray(angles, dtype=float)
    total = float(length_array.sum())
    mean = float(np.average(angle_array, weights=length_array)) if total else 0.0
    p90 = _weighted_quantile(angle_array, length_array, 0.90) if total else 0.0
    sharp_fraction = (
        float(length_array[angle_array >= 30.0].sum() / total) if total else 0.0
    )
    return {
        "length": total,
        "normalized_length": total / max(bbox_diagonal(mesh), 1.0e-30),
        "length_weighted_mean_dihedral_deg": mean,
        "length_weighted_p90_dihedral_deg": p90,
        "sharp_length_fraction": sharp_fraction,
        "open_boundary": any(
            len(incidents.get(edge, ())) == 1 for edge in arc_edges(arc)
        ),
    }


def _nested_evidence(arc: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = arc.get("evidence")
    if not isinstance(raw, Mapping):
        return {}
    boundary = raw.get("region_pair_boundary")
    return boundary if isinstance(boundary, Mapping) else raw


def _persistence_value(arc: Mapping[str, Any], nested: Mapping[str, Any]) -> float:
    raw: Any = nested.get("persistence")
    if isinstance(raw, Mapping):
        raw = raw.get("persistence", raw.get("endpoint_agreement_fraction"))
    if not isinstance(raw, (int, float)):
        evidence = arc.get("evidence")
        raw = (
            evidence.get("boundary_stability", 0.0)
            if isinstance(evidence, Mapping)
            else 0.0
        )
    return float(np.clip(float(raw), 0.0, 1.0))


def _mandatory_reasons(
    arc: Mapping[str, Any], geometry: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    constraint = str(arc.get("constraint", "")).lower()
    raw_evidence = arc.get("evidence")
    evidence: Mapping[str, Any] = (
        raw_evidence if isinstance(raw_evidence, Mapping) else {}
    )
    origin = str(evidence.get("mandatory_origin") or "").lower()
    if "user" in constraint or "user" in origin:
        reasons.append("user_specified_boundary")
    elif "mandatory" in constraint and not origin:
        # Unlabelled explicit input constraints remain protected. Automatic
        # geometry classifications carry an origin and are only candidates.
        reasons.append("explicit_mandatory_constraint")
    if bool(geometry.get("open_boundary")):
        reasons.append("open_boundary")
    return sorted(set(reasons))


def _arc_face_weights(
    mesh: Mesh,
    arc: Mapping[str, Any],
    incidents: Mapping[tuple[int, int], list[tuple[int, int]]],
) -> dict[int, float]:
    weights: dict[int, float] = defaultdict(float)
    for edge in arc_edges(arc):
        length = float(np.linalg.norm(mesh.vertices[edge[1]] - mesh.vertices[edge[0]]))
        faces = sorted({int(row[0]) for row in incidents.get(edge, ())})
        if not faces:
            continue
        share = length / len(faces)
        for face in faces:
            weights[face] += share
    return dict(weights)


def arc_capacity(
    mesh: Mesh,
    arc: Mapping[str, Any],
    rho: np.ndarray,
    h0: float,
    incidents: Mapping[tuple[int, int], list[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    topology = incidents if incidents is not None else edge_topology(mesh)[0]
    weights = _arc_face_weights(mesh, arc, topology)
    value = float(
        sum(
            length * math.exp(math.log(max(float(rho[face]), 1.0e-30)))
            for face, length in weights.items()
        )
        / h0
    )
    target = max(1, int(round(value)))
    residual = abs(value - target)
    return {
        "value": value,
        "target_integer": target,
        "absolute_residual": residual,
        "normalized_residual": min(1.0, 2.0 * residual),
        "face_weights": [
            {"face_id": face, "arc_length_share": weights[face]}
            for face in sorted(weights)
        ],
    }


def _segment_segment_distance(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> float:
    """Exact Euclidean distance between two closed 3-D line segments."""

    first = first_end - first_start
    second = second_end - second_start
    offset = first_start - second_start
    aa = float(np.dot(first, first))
    bb = float(np.dot(first, second))
    cc = float(np.dot(second, second))
    dd = float(np.dot(first, offset))
    ee = float(np.dot(second, offset))
    denominator = aa * cc - bb * bb
    epsilon = 1.0e-30

    if aa <= epsilon and cc <= epsilon:
        return float(np.linalg.norm(offset))
    if aa <= epsilon:
        first_parameter = 0.0
        second_parameter = float(np.clip(ee / max(cc, epsilon), 0.0, 1.0))
    elif cc <= epsilon:
        second_parameter = 0.0
        first_parameter = float(np.clip(-dd / max(aa, epsilon), 0.0, 1.0))
    else:
        if denominator <= epsilon:
            first_parameter = 0.0
        else:
            first_parameter = float(
                np.clip((bb * ee - cc * dd) / denominator, 0.0, 1.0)
            )
        second_parameter = (bb * first_parameter + ee) / cc
        if second_parameter < 0.0:
            second_parameter = 0.0
            first_parameter = float(np.clip(-dd / aa, 0.0, 1.0))
        elif second_parameter > 1.0:
            second_parameter = 1.0
            first_parameter = float(np.clip((bb - dd) / aa, 0.0, 1.0))
    separation = offset + first_parameter * first - second_parameter * second
    return float(np.linalg.norm(separation))


def _polyline_geometry(
    mesh: Mesh, arc: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    edges = arc_edges(arc)
    if not edges:
        empty = np.empty((0, 3))
        return empty, empty, empty, np.full(3, math.inf), np.full(3, -math.inf)
    starts = mesh.vertices[[edge[0] for edge in edges]]
    ends = mesh.vertices[[edge[1] for edge in edges]]
    points = np.vstack((starts, ends, 0.5 * (starts + ends)))
    return starts, ends, points, points.min(axis=0), points.max(axis=0)


def _prepared_polyline_distance(
    left: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    right: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> float:
    left_starts, left_ends, left_points, _, _ = left
    right_starts, right_ends, right_points, _, _ = right
    if not len(left_starts) or not len(right_starts):
        return math.inf
    del left_points, right_points
    closest = math.inf
    for left_start, left_end in zip(left_starts, left_ends):
        for right_start, right_end in zip(right_starts, right_ends):
            closest = min(
                closest,
                _segment_segment_distance(left_start, left_end, right_start, right_end),
            )
            if closest <= 1.0e-15:
                return 0.0
    return float(closest)


def corridor_conflicts(
    mesh: Mesh,
    rows: list[dict[str, Any]],
    size: np.ndarray,
    *,
    minimum_capacity: float = 2.0,
) -> dict[str, list[dict[str, Any]]]:
    """Return close, disjoint arc pairs whose corridor cannot fit two cells."""

    incidents, _ = edge_topology(mesh)
    faces = {
        row["arc_id"]: sorted(_arc_face_weights(mesh, row["arc"], incidents))
        for row in rows
    }
    geometry = {row["arc_id"]: _polyline_geometry(mesh, row["arc"]) for row in rows}
    vertices = {row["arc_id"]: set(arc_vertices(row["arc"])) for row in rows}
    result: dict[str, list[dict[str, Any]]] = {row["arc_id"]: [] for row in rows}
    for left_index, left in enumerate(rows):
        left_vertices = vertices[left["arc_id"]]
        for right in rows[left_index + 1 :]:
            right_vertices = vertices[right["arc_id"]]
            # Meeting at an explicit graph vertex is a junction, not a corridor.
            if left_vertices & right_vertices:
                continue
            local_faces = faces[left["arc_id"]] + faces[right["arc_id"]]
            local_size = (
                float(np.median(size[local_faces])) if local_faces else math.inf
            )
            left_geometry = geometry[left["arc_id"]]
            right_geometry = geometry[right["arc_id"]]
            bbox_gap = np.maximum(
                0.0,
                np.maximum(
                    left_geometry[3] - right_geometry[4],
                    right_geometry[3] - left_geometry[4],
                ),
            )
            if float(np.linalg.norm(bbox_gap)) >= minimum_capacity * local_size:
                continue
            distance = _prepared_polyline_distance(left_geometry, right_geometry)
            capacity = distance / max(local_size, 1.0e-30)
            if not math.isfinite(capacity) or capacity >= minimum_capacity:
                continue
            severity = float(
                np.clip((minimum_capacity - capacity) / minimum_capacity, 0.0, 1.0)
            )
            for source, target in ((left, right), (right, left)):
                result[source["arc_id"]].append(
                    {
                        "other_arc_id": target["arc_id"],
                        "distance": distance,
                        "local_target_size": local_size,
                        "capacity": capacity,
                        "minimum_capacity": minimum_capacity,
                        "severity": severity,
                    }
                )
    for values in result.values():
        values.sort(key=lambda item: (-item["severity"], item["other_arc_id"]))
    return result


def compute_arc_evidence(
    mesh: Mesh,
    field: DirectionField,
    graph: Mapping[str, Any],
    *,
    rho: np.ndarray | None = None,
    size: np.ndarray | None = None,
    h0: float | None = None,
) -> dict[str, Any]:
    """Compute complete, deterministic evidence for every candidate arc."""

    incidents, _ = edge_topology(mesh)
    compatibility = audit_graph_compatibility(mesh, field, dict(graph))
    by_id = {str(row["arc_id"]): row for row in compatibility["arcs"]}
    rows: list[dict[str, Any]] = []
    for ordinal, raw_arc in enumerate(graph.get("arcs", ())):
        arc = dict(raw_arc)
        identifier = arc_id(arc, ordinal)
        geometry = _geometry_measurements(mesh, arc, incidents)
        nested = _nested_evidence(arc)
        persistence = _persistence_value(arc, nested)
        primitive_transition = bool(nested.get("primitive_transition", False))
        geometry_quality = float(
            np.clip(
                nested.get(
                    "utility",
                    0.55 * float(arc.get("confidence", 0.0))
                    + 0.20 * min(float(geometry["normalized_length"]) / 0.10, 1.0)
                    + 0.15
                    * min(
                        float(geometry["length_weighted_mean_dihedral_deg"]) / 30.0, 1.0
                    )
                    + 0.10 * float(primitive_transition),
                ),
                0.0,
                1.0,
            )
        )
        field_row = by_id.get(identifier, {})
        mean_degrees = field_row.get("mean_degrees")
        field_error = 1.0 if mean_degrees is None else float(mean_degrees) / 45.0
        mandatory_reasons = _mandatory_reasons(arc, geometry)
        row: dict[str, Any] = {
            "arc_id": identifier,
            "state": "mandatory" if mandatory_reasons else "optional",
            "mandatory_reasons": mandatory_reasons,
            "source": str(arc.get("source", "unknown")),
            "arc": arc,
            "geometry": {
                **geometry,
                "primitive_transition": primitive_transition,
                "quality": geometry_quality,
                "persistence": persistence,
                "preset_support": (
                    arc.get("evidence", {}).get("preset_support", {})
                    if isinstance(arc.get("evidence"), Mapping)
                    else {}
                ),
            },
            "field": {
                "mean_degrees": mean_degrees,
                "p95_degrees": field_row.get("p95_degrees"),
                "normalized_error": field_error,
                "branch_consistency": field_row.get("branch_consistency"),
                "sample_count": field_row.get("sample_count", 0),
            },
            "realizability": {
                "capacity": None,
                "corridor_conflicts": [],
                "corridor_conflict": 0.0,
                "error": 0.0,
            },
        }
        if rho is not None and h0 is not None:
            row["realizability"]["capacity"] = arc_capacity(
                mesh, arc, np.asarray(rho, dtype=float), float(h0), incidents
            )
        rows.append(row)

    if rho is not None and size is not None and h0 is not None:
        conflicts = corridor_conflicts(mesh, rows, np.asarray(size, dtype=float))
        for row in rows:
            capacity = row["realizability"]["capacity"]
            values = conflicts[row["arc_id"]]
            corridor_error = max(
                (float(item["severity"]) for item in values), default=0.0
            )
            capacity_error = float(capacity["normalized_residual"]) if capacity else 0.0
            row["realizability"].update(
                {
                    "corridor_conflicts": values,
                    "corridor_conflict": corridor_error,
                    "error": float(np.clip(capacity_error + corridor_error, 0.0, 2.0)),
                }
            )

    return {
        "schema_version": "weak-layout.arc-evidence.v2",
        "arc_count": len(rows),
        "field_error_definition": "length-weighted fourfold tangent error / 45 degrees",
        "capacity_definition": "integral(exp(u)/h0 ds), with edge length shared across incident faces",
        "arcs": rows,
    }


__all__ = [
    "arc_capacity",
    "arc_edges",
    "arc_id",
    "arc_vertices",
    "compute_arc_evidence",
    "corridor_conflicts",
]
