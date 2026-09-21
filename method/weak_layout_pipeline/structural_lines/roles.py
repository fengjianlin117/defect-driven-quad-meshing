"""Deterministic role classification for already-selected semantic arcs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROLE_KEYS = ("hard_feature", "soft_structure", "field_guide")
PRESERVED_KEYS = (
    "arc_id", "id", "source", "vertices", "mesh_vertices", "edges", "mesh_edges",
    "endpoints", "closed", "confidence", "constraint", "evidence", "incident_faces",
    "edge_incident_faces", "length",
)


def role_for_arc(arc: dict[str, object]) -> str:
    source = str(arc.get("source", "")).lower()
    constraint = str(arc.get("constraint", "")).lower()
    evidence = arc.get("evidence") if isinstance(arc.get("evidence"), dict) else {}
    mandatory = bool(evidence.get("mandatory_origin")) or "mandatory" in constraint
    sharp = any(token in constraint for token in ("sharp", "hard"))
    if source == "geometry" and (mandatory or sharp):
        return "hard_feature"
    if source == "geometry":
        return "soft_structure"
    return "field_guide"


def _normalized_arc(arc: dict[str, object], ordinal: int) -> dict[str, object]:
    result = {key: arc[key] for key in PRESERVED_KEYS if key in arc}
    result["arc_id"] = str(arc.get("arc_id", arc.get("id", f"arc-{ordinal:06d}")))
    result["source"] = str(arc.get("source", "unknown"))
    result["confidence"] = float(arc.get("confidence", 0.0))
    result["closed"] = bool(arc.get("closed", False))
    result["mesh_vertices"] = list(arc.get("mesh_vertices", arc.get("vertices", [])))
    result["mesh_edges"] = list(arc.get("mesh_edges", arc.get("edges", [])))
    result["incident_faces"] = list(arc.get("incident_faces", []))
    result["role"] = role_for_arc(arc)
    return result


def classify_roles(graph: dict[str, object], output_dir: str | Path | None = None) -> dict[str, object]:
    arcs = [_normalized_arc(arc, i) for i, arc in enumerate(graph.get("arcs", []))]
    grouped = {role: [arc for arc in arcs if arc["role"] == role] for role in ROLE_KEYS}
    edges = {
        role: sorted({tuple(sorted(map(int, edge))) for arc in grouped[role] for edge in arc["mesh_edges"]})
        for role in ROLE_KEYS
    }
    summary = {
        "schema_version": "weak-layout.structural-role-summary.v1",
        "arc_count": len(arcs),
        "edge_count_by_role": {role: len(edges[role]) for role in ROLE_KEYS},
        "arc_count_by_role": {role: len(grouped[role]) for role in ROLE_KEYS},
        "source_count": dict(sorted(Counter(arc["source"] for arc in arcs).items())),
        "classification_rule": {
            "hard_feature": "geometry arc with mandatory/sharp/hard evidence",
            "soft_structure": "other selected geometry arc",
            "field_guide": "selected non-geometry (normally field-derived) arc",
        },
    }
    singularity_audit = graph.get("singularity_audit") if isinstance(graph.get("singularity_audit"), dict) else {}
    singularity_rows = singularity_audit.get("singularities", [])
    singularity_vertices = sorted({
        int(row["vertex"] if isinstance(row, dict) else row)
        for row in singularity_rows
        if (isinstance(row, dict) and "vertex" in row) or isinstance(row, (int, float))
    })
    result = {
        "schema_version": "weak-layout.structural-roles.v1",
        "arcs": arcs,
        "roles": grouped,
        "edges": {role: [list(edge) for edge in edges[role]] for role in ROLE_KEYS},
        "summary": summary,
        "singularity_vertices": singularity_vertices,
    }
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "structural_roles.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (target / "role_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (target / "soft_structure_arcs.json").write_text(json.dumps(grouped["soft_structure"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (target / "field_guide_arcs.json").write_text(json.dumps(grouped["field_guide"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (target / "hard_feature_edges.txt").open("w", encoding="utf-8", newline="\n") as stream:
            for u, v in edges["hard_feature"]:
                stream.write(f"{u} {v}\n")
    return result


def write_all_selected_edges(path: str | Path, roles: dict[str, object]) -> None:
    all_edges = sorted({tuple(edge) for values in roles["edges"].values() for edge in values})
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for u, v in all_edges:
            stream.write(f"{u} {v}\n")
