"""Fourfold graph-to-field compatibility diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weak_layout_pipeline.direction_field.adapter import DirectionField
from weak_layout_pipeline.pipeline.mesh import Mesh, canonical_edge, edge_topology


def fourfold_error(
    tangent: np.ndarray, pd1: np.ndarray, pd2: np.ndarray
) -> tuple[float, int]:
    tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-30)
    values = np.clip(np.abs([np.dot(tangent, pd1), np.dot(tangent, pd2)]), 0.0, 1.0)
    values[np.abs(values - 1.0) < 1e-12] = 1.0
    angles = np.degrees(np.arccos(values))
    branch = int(np.argmin(angles))
    return float(min(45.0, angles[branch])), branch


def _arc_edges(arc: dict[str, object]) -> list[tuple[int, int]]:
    source = arc.get("mesh_edges") or arc.get("edges") or []
    return [(int(edge[0]), int(edge[1])) for edge in source]


def audit_graph_compatibility(
    mesh: Mesh,
    field: DirectionField,
    graph: dict[str, object],
    *,
    low_alignment_degrees: float = 20.0,
) -> dict[str, object]:
    incidents, _ = edge_topology(mesh)
    reports = []
    for ordinal, arc in enumerate(graph.get("arcs", [])):
        samples: list[tuple[float, int, float, int]] = []
        total_length = 0.0
        for u, v in _arc_edges(arc):
            vector = mesh.vertices[v] - mesh.vertices[u]
            length = float(np.linalg.norm(vector))
            if length <= 1e-14:
                continue
            total_length += length
            face_ids = sorted(
                {face for face, _ in incidents.get(canonical_edge(u, v), [])}
            )
            for face in face_ids:
                tangent = (
                    vector - np.dot(vector, field.normals[face]) * field.normals[face]
                )
                if np.linalg.norm(tangent) <= 1e-14:
                    continue
                error, branch = fourfold_error(
                    tangent, field.pd1[face], field.pd2[face]
                )
                samples.append((error, branch, length / max(len(face_ids), 1), face))
        errors = np.asarray([sample[0] for sample in samples], dtype=float)
        weights = np.asarray([sample[2] for sample in samples], dtype=float)
        branches = np.asarray([sample[1] for sample in samples], dtype=int)
        face_rows = []
        for face in sorted({sample[3] for sample in samples}):
            selected = [sample for sample in samples if sample[3] == face]
            face_rows.append(
                {
                    "face_id": face,
                    "mean_degrees": float(
                        np.average(
                            [sample[0] for sample in selected],
                            weights=[sample[2] for sample in selected],
                        )
                    ),
                }
            )
        if len(errors):
            order = np.argsort(errors, kind="stable")
            cumulative = np.cumsum(weights[order]) / weights.sum()
            quantile = lambda q: float(
                errors[order[min(np.searchsorted(cumulative, q), len(errors) - 1)]]
            )
            preferred = int(
                np.bincount(branches, weights=weights, minlength=2).argmax()
            )
            consistency = float(weights[branches == preferred].sum() / weights.sum())
            low_length = float(weights[errors > low_alignment_degrees].sum())
            stats = {
                "mean_degrees": float(np.average(errors, weights=weights)),
                "median_degrees": quantile(0.5),
                "p95_degrees": quantile(0.95),
                "max_degrees": float(errors.max()),
            }
        else:
            preferred, consistency, low_length = None, 0.0, total_length
            stats = {
                key: None
                for key in (
                    "mean_degrees",
                    "median_degrees",
                    "p95_degrees",
                    "max_degrees",
                )
            }
        reports.append(
            {
                "arc_id": str(arc.get("arc_id", arc.get("id", f"arc-{ordinal:06d}"))),
                "source": arc.get("source", "unknown"),
                "confidence": float(arc.get("confidence", 0.0)),
                "sample_count": len(samples),
                "arc_length": total_length,
                "preferred_field_branch": preferred,
                "branch_consistency": consistency,
                "low_alignment_threshold_degrees": low_alignment_degrees,
                "low_alignment_length": low_length,
                "low_alignment_fraction": (
                    low_length / total_length if total_length else 0.0
                ),
                "face_error_degrees": face_rows,
                **stats,
            }
        )
    finite = [row["mean_degrees"] for row in reports if row["mean_degrees"] is not None]
    return {
        "schema_version": "weak-layout.graph-field-compatibility.v1",
        "angle_definition": "min(angle(t,+/-PD1),angle(t,+/-PD2)) in [0,45] degrees",
        "arc_count": len(reports),
        "mean_of_arc_means_degrees": float(np.mean(finite)) if finite else None,
        "arcs": reports,
    }


def write_compatibility(path: str | Path, report: dict[str, object]) -> None:
    Path(path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
