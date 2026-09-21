"""Deterministic whole-arc selection without backend trials."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from weak_layout_pipeline.pipeline.mesh import Mesh, edge_topology
from weak_layout_pipeline.structural_lines.evidence import arc_edges, arc_vertices

MAX_STRUCTURE_DEGREE = 4
MAX_EDGE_FRACTION = 0.18
MAX_CYCLE_FRACTION = 0.025
MAX_CAPACITY_RESIDUAL = 0.85


def arc_utility(row: Mapping[str, Any], config: Any) -> float:
    geometry = row["geometry"]
    field = row["field"]
    realizability = row["realizability"]
    return float(
        float(geometry["quality"])
        + float(config.lambda_persistence) * float(geometry["persistence"])
        - float(config.lambda_field) * float(field["normalized_error"])
        - float(config.lambda_realizability) * float(realizability["error"])
    )


def _network_topology(edges: Iterable[tuple[int, int]]) -> dict[str, int]:
    canonical = set(edges)
    degrees: Counter[int] = Counter()
    adjacency: dict[int, set[int]] = defaultdict(set)
    for u, v in canonical:
        degrees[u] += 1
        degrees[v] += 1
        adjacency[u].add(v)
        adjacency[v].add(u)
    seen: set[int] = set()
    components = 0
    for start in sorted(adjacency):
        if start in seen:
            continue
        components += 1
        seen.add(start)
        queue = deque([start])
        while queue:
            for neighbor in adjacency[queue.popleft()]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    cycle_rank = len(canonical) - len(adjacency) + components if canonical else 0
    return {
        "edge_count": len(canonical),
        "vertex_count": len(adjacency),
        "component_count": components,
        "cycle_rank": max(0, cycle_rank),
        "maximum_degree": max(degrees.values(), default=0),
    }


def _illegal_intersection(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_vertices = arc_vertices(left)
    right_vertices = arc_vertices(right)
    shared = set(left_vertices) & set(right_vertices)
    if not shared:
        return False
    left_endpoints = {left_vertices[0], left_vertices[-1]}
    right_endpoints = {right_vertices[0], right_vertices[-1]}
    return bool(shared - (left_endpoints & right_endpoints))


def _conflicting_selected_arc(
    row: Mapping[str, Any], selected_ids: set[str]
) -> str | None:
    for conflict in row["realizability"].get("corridor_conflicts", ()):
        other = str(conflict["other_arc_id"])
        if other in selected_ids and float(conflict["severity"]) > 0.0:
            return other
    return None


def select_arcs(
    mesh: Mesh,
    evidence: Mapping[str, Any],
    config: Any,
    *,
    stage: str,
) -> dict[str, Any]:
    """Select mandatory arcs first, then utility-ranked feasible optionals."""

    rows = [dict(row) for row in evidence.get("arcs", ())]
    for row in rows:
        row["utility"] = arc_utility(row, config)
    mandatory = sorted(
        (row for row in rows if row["state"] == "mandatory"),
        key=lambda row: row["arc_id"],
    )
    optional = sorted(
        (row for row in rows if row["state"] != "mandatory"),
        key=lambda row: (-float(row["utility"]), row["arc_id"]),
    )
    mesh_edge_count = len(edge_topology(mesh)[0])
    edge_budget = max(16, int(MAX_EDGE_FRACTION * mesh_edge_count))
    cycle_budget = max(2, int(MAX_CYCLE_FRACTION * mesh.vertex_count))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_edges: set[tuple[int, int]] = set()
    edge_owner: dict[tuple[int, int], str] = {}
    warnings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def accept(row: dict[str, Any]) -> None:
        selected.append(row)
        selected_ids.add(str(row["arc_id"]))
        for edge in arc_edges(row["arc"]):
            selected_edges.add(edge)
            edge_owner.setdefault(edge, str(row["arc_id"]))

    for row in mandatory:
        duplicate = sorted(set(arc_edges(row["arc"])) & set(edge_owner))
        illegal = [
            other["arc_id"]
            for other in selected
            if _illegal_intersection(row["arc"], other["arc"])
        ]
        accept(row)
        topology = _network_topology(selected_edges)
        if duplicate or illegal or topology["maximum_degree"] > MAX_STRUCTURE_DEGREE:
            warnings.append(
                {
                    "arc_id": row["arc_id"],
                    "reason": "mandatory_topology_conflict_preserved",
                    "duplicate_edges": [list(edge) for edge in duplicate],
                    "illegal_intersections": illegal,
                    "topology": topology,
                }
            )

    for row in optional:
        identifier = str(row["arc_id"])
        reason: str | None = None
        details: dict[str, Any] = {}
        edges = set(arc_edges(row["arc"]))
        duplicates = sorted(edges & set(edge_owner))
        if float(row["utility"]) <= 0.0:
            reason = "nonpositive_utility"
        elif duplicates:
            reason = "nonmanifold_or_duplicate_structure_edge"
            details["edges"] = [list(edge) for edge in duplicates]
        else:
            illegal = [
                other["arc_id"]
                for other in selected
                if _illegal_intersection(row["arc"], other["arc"])
            ]
            if illegal:
                reason = "illegal_arc_intersection"
                details["other_arc_ids"] = illegal
        if reason is None and stage != "initial":
            conflict = _conflicting_selected_arc(row, selected_ids)
            capacity = row["realizability"].get("capacity")
            if conflict is not None:
                reason = "insufficient_corridor_capacity"
                details["other_arc_id"] = conflict
            elif (
                capacity is not None
                and float(capacity["normalized_residual"]) > MAX_CAPACITY_RESIDUAL
            ):
                reason = "integer_capacity_residual"
                details["normalized_residual"] = capacity["normalized_residual"]
        if reason is None:
            trial = selected_edges | edges
            topology = _network_topology(trial)
            if topology["edge_count"] > edge_budget:
                reason = "edge_budget"
            elif topology["maximum_degree"] > MAX_STRUCTURE_DEGREE:
                reason = "maximum_structure_vertex_degree"
            elif topology["cycle_rank"] > cycle_budget:
                reason = "cycle_budget"
            if reason is not None:
                details.update(
                    {
                        "trial_topology": topology,
                        "edge_budget": edge_budget,
                        "cycle_budget": cycle_budget,
                        "maximum_degree": MAX_STRUCTURE_DEGREE,
                    }
                )
        if reason is None:
            accept(row)
        else:
            rejected.append(
                {
                    "arc_id": identifier,
                    "state": "optional",
                    "utility": row["utility"],
                    "reason": reason,
                    **details,
                }
            )

    selected.sort(key=lambda row: row["arc_id"])
    selected_ids = {str(row["arc_id"]) for row in selected}
    hard_edges = sorted({edge for row in selected for edge in arc_edges(row["arc"])})
    return {
        "schema_version": "weak-layout.arc-selection.v2",
        "stage": stage,
        "whole_arc_only": True,
        "backend_semantics": "every selected mandatory or optional arc is hard",
        "budgets": {
            "edge_count": edge_budget,
            "cycle_rank": cycle_budget,
            "maximum_structure_vertex_degree": MAX_STRUCTURE_DEGREE,
        },
        "input_arc_count": len(rows),
        "selected_arc_count": len(selected),
        "selected_mandatory_count": sum(
            row["state"] == "mandatory" for row in selected
        ),
        "selected_optional_count": sum(row["state"] != "mandatory" for row in selected),
        "selected_arc_ids": sorted(selected_ids),
        "selected_arcs": selected,
        "hard_edges": [list(edge) for edge in hard_edges],
        "topology": _network_topology(hard_edges),
        "rejected": sorted(rejected, key=lambda row: row["arc_id"]),
        "warnings": warnings,
    }


def remove_fallback_arc(
    selection: Mapping[str, Any]
) -> tuple[dict[str, Any], str | None]:
    """Remove exactly one lowest-utility conflicting optional arc, if any."""

    selected = [dict(row) for row in selection.get("selected_arcs", ())]
    optional = [row for row in selected if row["state"] != "mandatory"]
    if not optional:
        return dict(selection), None
    conflicting = [
        row for row in optional if float(row["realizability"].get("error", 0.0)) > 0.0
    ]
    if not conflicting:
        return dict(selection), None
    removed = min(conflicting, key=lambda row: (float(row["utility"]), row["arc_id"]))
    retained = [row for row in selected if row["arc_id"] != removed["arc_id"]]
    hard_edges = sorted({edge for row in retained for edge in arc_edges(row["arc"])})
    result = dict(selection)
    result.update(
        {
            "stage": "fallback",
            "selected_arcs": retained,
            "selected_arc_ids": sorted(row["arc_id"] for row in retained),
            "selected_arc_count": len(retained),
            "selected_mandatory_count": sum(
                row["state"] == "mandatory" for row in retained
            ),
            "selected_optional_count": sum(
                row["state"] != "mandatory" for row in retained
            ),
            "hard_edges": [list(edge) for edge in hard_edges],
            "topology": _network_topology(hard_edges),
            "fallback_removed": {
                "arc_id": removed["arc_id"],
                "utility": removed["utility"],
                "reason": "lowest_utility_conflicting_optional_arc",
            },
        }
    )
    return result, str(removed["arc_id"])


def write_hard_edges(path: str | Path, selection: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for u, v in selection.get("hard_edges", ()):
            stream.write(f"{int(u)} {int(v)}\n")


__all__ = [
    "MAX_STRUCTURE_DEGREE",
    "arc_utility",
    "remove_fallback_arc",
    "select_arcs",
    "write_hard_edges",
]
