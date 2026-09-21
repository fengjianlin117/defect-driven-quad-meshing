"""Sparse feature constraints selected from omissions of a fixed baseline.

The score predicts usefulness; it is not a proof of realized backend gain.
See docs/BASELINE_GUIDED_SELECTION.md for the implemented equations.
"""

from __future__ import annotations

import math
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from weak_layout_pipeline.pipeline.mesh import (
    edge_topology,
    triangle_geometry,
    unique_edges,
)
from weak_layout_pipeline.pipeline.sampling import area_samples, surface_distances
from weak_layout_pipeline.structural_lines.evidence import (
    compute_arc_evidence,
    arc_edges,
)
from weak_layout_pipeline.structural_lines.selection import _network_topology


def _chains(edges):
    """Maximal edge-disjoint chains; degree-two vertices do not split a feature."""
    remaining = set(edges)
    neighbors = defaultdict(list)
    for a, b in sorted(remaining):
        neighbors[a].append(b)
        neighbors[b].append(a)

    def walk(a, b):
        ids = [a, b]
        remaining.remove(tuple(sorted((a, b))))
        while len(neighbors[b]) == 2:
            nxt = next(
                (v for v in sorted(neighbors[b]) if tuple(sorted((b, v))) in remaining),
                None,
            )
            if nxt is None:
                break
            remaining.remove(tuple(sorted((b, nxt))))
            ids.append(nxt)
            b = nxt
            if b == ids[0]:
                break
        return ids

    result = []
    for a in sorted(neighbors):
        if len(neighbors[a]) == 2:
            continue
        for b in sorted(neighbors[a]):
            if tuple(sorted((a, b))) in remaining:
                result.append(walk(a, b))
    while remaining:
        result.append(walk(*min(remaining)))
    return result


def feature_graph(mesh, *, sharp_degrees=30.0):
    """V3 scope: actual sharp mesh edges and protected input boundary chains.

    No primitive-segmentation boundaries or field-completion arcs are silently
    promoted to geometric features. General smooth ridges are out of scope.
    """
    normals = triangle_geometry(mesh)[0]
    incidents = edge_topology(mesh)[0]
    sharp, border = [], []
    for edge, rows in sorted(incidents.items()):
        if len(rows) == 1:
            border.append(edge)
        elif len(rows) == 2:
            angle = np.degrees(
                np.arccos(np.clip(normals[rows[0][0]] @ normals[rows[1][0]], -1, 1))
            )
            if angle >= sharp_degrees:
                sharp.append(edge)
    arcs = []
    for role, edges in (("boundary", border), ("feature", sharp)):
        for ids in _chains(edges):
            identifier = f"{role}-{len(arcs):05d}"
            pairs = [list(sorted((a, b))) for a, b in zip(ids[:-1], ids[1:])]
            arcs.append(
                dict(
                    arc_id=identifier,
                    id=identifier,
                    source="geometry",
                    mesh_vertices=ids,
                    vertices=ids,
                    mesh_edges=pairs,
                    edges=pairs,
                    closed=ids[0] == ids[-1],
                    confidence=1.0,
                    constraint="hard_mandatory" if role == "boundary" else "candidate",
                    evidence={
                        "mandatory_origin": (
                            "open_boundary" if role == "boundary" else None
                        )
                    },
                )
            )
    return {
        "schema_version": "weak-layout.sharp-chains.v1",
        "sharp_degrees": sharp_degrees,
        "arcs": arcs,
        "scope": "maximal sharp-edge chains; no remeshing-invariance guarantee",
    }


def selection_record(rows, **metadata):
    hard = sorted({e for row in rows for e in arc_edges(row["arc"])})
    return dict(
        schema_version="weak-layout.baseline-selection.v1",
        selected_arcs=rows,
        selected_arc_ids=[row["arc_id"] for row in rows],
        selected_arc_count=len(rows),
        selected_mandatory_count=sum(row["state"] == "mandatory" for row in rows),
        selected_optional_count=sum(row["state"] != "mandatory" for row in rows),
        hard_edges=[list(e) for e in hard],
        topology=_network_topology(hard),
        **metadata,
    )


def _vertex_graph(mesh):
    edges = unique_edges(mesh)
    lengths = np.linalg.norm(
        mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]], axis=1
    )
    return coo_matrix(
        (
            np.tile(lengths, 2),
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(mesh.vertex_count, mesh.vertex_count),
    ).tocsr()


def _curve_samples(mesh, arc, h0):
    ids = arc["mesh_vertices"]
    starts = mesh.vertices[ids[:-1]]
    vectors = mesh.vertices[ids[1:]] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    total = float(lengths.sum())
    count = max(4, min(512, int(math.ceil(2 * total / h0))))
    position = (np.arange(count) + 0.5) * total / count
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    segment = np.minimum(
        np.searchsorted(cumulative[1:], position, side="right"), len(lengths) - 1
    )
    t = (position - cumulative[segment]) / lengths[segment]
    return (
        starts[segment] + t[:, None] * vectors[segment],
        vectors[segment] / lengths[segment, None],
    )


def feature_miss(mesh, quad, arc, h0):
    points, tangents = _curve_samples(mesh, arc, h0)
    edges = unique_edges(quad)
    starts = quad.vertices[edges[:, 0]]
    vectors = quad.vertices[edges[:, 1]] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    unit = vectors / np.maximum(lengths[:, None], 1e-30)
    best = []
    for p, tangent in zip(points, tangents):
        t = np.clip(
            np.einsum("ij,ij->i", p - starts, vectors) / np.maximum(lengths**2, 1e-30),
            0,
            1,
        )
        distance = np.linalg.norm(p - starts - t[:, None] * vectors, axis=1)
        sine = np.sqrt(np.maximum(0, 1 - np.clip(unit @ tangent, -1, 1) ** 2))
        best.append(min(1.0, float(np.min(distance / (0.25 * h0) + sine))))
    return float(np.mean(best))


def greedy_coverage(responses, costs, *, mandatory=(), feasible=None, limit=12):
    """Max-coverage minus modular cost, greedy positive marginal additions."""
    n, samples = responses.shape
    selected = list(mandatory)
    remaining = set(range(n)) - set(selected)
    coverage = responses[selected].max(axis=0) if selected else np.zeros(samples)
    trace = []
    while remaining and len(selected) - len(mandatory) < limit:
        options = []
        for i in sorted(remaining):
            if feasible is not None and not feasible(i, selected):
                continue
            gain = float(np.maximum(responses[i] - coverage, 0).mean() - costs[i])
            options.append((gain, -i, i))
        if not options:
            break
        gain, _, i = max(options)
        if gain <= 1e-12:
            break
        selected.append(i)
        remaining.remove(i)
        coverage = np.maximum(coverage, responses[i])
        trace.append(dict(index=i, marginal_gain=gain))
    return selected, trace


def select_from_baseline(mesh, field, graph, baseline_quad, h0, config):
    evidence = compute_arc_evidence(mesh, field, graph)
    rows = evidence["arcs"]
    area = float(triangle_geometry(mesh)[1].sum())
    points, faces, bary = area_samples(mesh, config.selection_samples, 1729)
    errors = np.clip(surface_distances(points, baseline_quad) / (0.1 * h0), 0, 1)
    mesh_graph = _vertex_graph(mesh)
    responses, costs, distances, lengths, excluded = [], [], [], [], {}
    for i, row in enumerate(rows):
        length = float(row["geometry"]["length"])
        lengths.append(length)
        distance = dijkstra(
            mesh_graph,
            directed=False,
            indices=np.unique(row["arc"]["mesh_vertices"]),
            min_only=True,
        )
        distances.append(distance)
        # Vertex-edge shortest paths interpolated over each source triangle.
        # This is a discrete surface-distance proxy, not a continuous geodesic.
        local = np.sum(distance[mesh.faces[faces]] * bary, axis=1)
        kernel = np.maximum(0.0, 1 - local / h0)
        miss = feature_miss(mesh, baseline_quad, row["arc"], h0)
        field_mean = float(row["field"]["mean_degrees"] or 0.0)
        field_p95 = row["field"].get("p95_degrees", row["field"].get("p90_degrees"))
        confidence = min(
            1.0, row["geometry"]["length_weighted_mean_dihedral_deg"] / 60.0
        )
        confidence *= math.exp(-((field_mean / 20.0) ** 2))
        if row["state"] == "mandatory":
            confidence = 1.0
        elif field_p95 is None or field_p95 > config.maximum_field_p95:
            excluded[i] = "field_p95"
        response = confidence * kernel * (0.7 * errors + 0.3 * miss)
        cost = config.sparsity_weight * (h0 * length + h0 * h0) / area
        responses.append(response)
        costs.append(cost)
        row["baseline"] = dict(
            feature_miss=miss,
            support_sample_count=int(np.count_nonzero(kernel)),
            confidence=confidence,
            standalone_benefit=float(response.mean()),
            complexity_cost=cost,
            standalone_gain=float(response.mean() - cost),
        )
        row["utility"] = row["baseline"]["standalone_gain"]
    mandatory = [i for i, row in enumerate(rows) if row["state"] == "mandatory"]
    budget = config.length_budget_factor * math.sqrt(area)

    def feasible(i, selected):
        if i in excluded:
            return False
        optional = [j for j in selected if j not in mandatory]
        if sum(lengths[j] for j in optional) + lengths[i] > budget:
            return False
        edges = {e for j in selected + [i] for e in arc_edges(rows[j]["arc"])}
        if _network_topology(edges)["maximum_degree"] > 4:
            return False
        vertices = set(rows[i]["arc"]["mesh_vertices"])
        for j in selected:
            other = set(rows[j]["arc"]["mesh_vertices"])
            # Shared endpoints are junctions; redundancy beyond junctions is
            # still charged by max-coverage. No blanket Euclidean thin-wall test.
            if vertices & other:
                continue
            if (
                float(distances[i][list(other)].min())
                < config.minimum_separation_cells * h0
            ):
                return False
        return True

    matrix = np.asarray(responses) if rows else np.zeros((0, len(points)))
    chosen, trace = greedy_coverage(
        matrix,
        np.asarray(costs),
        mandatory=mandatory,
        feasible=feasible,
        limit=config.maximum_optional_curves,
    )
    for item in trace:
        item["arc_id"] = rows[item["index"]]["arc_id"]
    rejected = []
    for i, row in enumerate(rows):
        if i not in chosen:
            reason = excluded.get(
                i,
                (
                    "set_constraint"
                    if not feasible(i, chosen)
                    else (
                        "curve_count_budget"
                        if len(chosen) - len(mandatory)
                        >= config.maximum_optional_curves
                        else "nonpositive_marginal_gain"
                    )
                ),
            )
            rejected.append(dict(arc_id=row["arc_id"], reason=reason))
    selection = selection_record(
        [rows[i] for i in chosen],
        stage="baseline_guided",
        rejected=rejected,
        budgets=dict(
            optional_curve_count=config.maximum_optional_curves,
            optional_total_length=budget,
        ),
        greedy_trace=trace,
    )
    evidence["sampling"] = dict(
        count=config.selection_samples, seed=1729, corridor_width=h0
    )
    return selection, evidence
