"""Read-only reference-feature diagnostics, not a selector or acceptance gate.

Reuses the project's sharp-edge and exact point/triangle distance code. Axial
grouping is an explicitly supplied diagnostic frame, not a universal semantic
feature detector. All raw sharp edges survive in either rings or residuals.
"""

from collections import defaultdict
import math
import numpy as np
from weak_layout_pipeline.pipeline.mesh import (
    edge_topology,
    triangle_geometry,
    unique_edges,
)
from weak_layout_pipeline.pipeline.sampling import surface_distances
from weak_layout_pipeline.structural_lines.baseline_guided import feature_graph, _chains


def edge_set(curves):
    return {tuple(sorted(e)) for c in curves for e in c["mesh_edges"]}


def curve_record(mesh, ids, identifier, origin):
    xyz = mesh.vertices[ids]
    lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    return dict(
        id=identifier,
        mesh_vertices=list(map(int, ids)),
        mesh_edges=[list(sorted((int(a), int(b)))) for a, b in zip(ids[:-1], ids[1:])],
        closed=ids[0] == ids[-1],
        length=float(lengths.sum()),
        bbox_min=xyz.min(0).tolist(),
        bbox_max=xyz.max(0).tolist(),
        origin=origin,
    )


def build_inventory(mesh, axis, sharp_degrees=30.0, tolerance=1e-5):
    """Group coplanar transverse sharp edges without vertical-junction splits.

    Also include boundaries of planar transverse face components. These provide
    an independent check against a single dihedral cutoff, not CAD ground truth.
    tolerance is relative to the source bbox diagonal.
    """
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    xyz = mesh.vertices
    eps = tolerance * np.linalg.norm(np.ptp(xyz, axis=0))
    coord = xyz @ axis
    raw = feature_graph(mesh, sharp_degrees=sharp_degrees)
    reference = edge_set(raw["arcs"])
    n, _, _ = triangle_geometry(mesh)
    incidents, _ = edge_topology(mesh)
    transverse = np.abs(n @ axis) >= np.cos(np.radians(1.0))
    planar_edges = set()
    for e, rows in incidents.items():
        # Union of coplanar transverse faces: internal tessellation edges vanish.
        if sum(bool(transverse[f]) for f, _ in rows) == 1:
            planar_edges.add(e)
    all_edges = reference | planar_edges
    flat = [e for e in sorted(all_edges) if abs(coord[e[0]] - coord[e[1]]) <= eps]
    layers = []
    for e in sorted(flat, key=lambda e: (float(coord[list(e)].mean()), e)):
        level = float(coord[list(e)].mean())
        if not layers or level - layers[-1][0] > eps:
            layers.append((level, []))
        layers[-1][1].append(e)
    used = set()
    curves = []
    for level, edges in layers:
        for ids in _chains(edges):
            if ids[0] != ids[-1]:
                continue
            c = curve_record(
                mesh, ids, f"ring-{len(curves):02d}", "transverse_geometry"
            )
            c["plane_level"] = level
            c["raw_sharp_edge_count"] = len(edge_set([c]) & reference)
            curves.append(c)
            used |= edge_set([c])
    # Never discard unresolved or non-ring geometry.
    for i, ids in enumerate(_chains(all_edges - used)):
        curves.append(curve_record(mesh, ids, f"residual-{i:03d}", "residual_geometry"))
    for c in curves:
        ce = edge_set([c])
        c["raw_arc_ids"] = [a["arc_id"] for a in raw["arcs"] if ce & edge_set([a])]
    threshold_scan = []
    for angle in (15.0, 20.0, 25.0, 30.0, 35.0, 45.0, 60.0):
        es = edge_set(feature_graph(mesh, sharp_degrees=angle)["arcs"])
        threshold_scan.append(
            dict(
                angle=angle,
                edge_count=len(es),
                edges_outside_primary_inventory=[
                    list(e) for e in sorted(es - all_edges)
                ],
            )
        )
    return dict(
        schema="reference-diagnostics.v1",
        axis=axis.tolist(),
        grouping_tolerance=eps,
        sharp_degrees=sharp_degrees,
        raw_arc_count=len(raw["arcs"]),
        raw_sharp_edges=len(reference),
        additional_transverse_edges=len(planar_edges - reference),
        unrepresented_raw_edges=len(reference - edge_set(curves)),
        curves=curves,
        threshold_scan=threshold_scan,
        semantic_completeness="not certified; raw edges covered, smooth ridges not extracted",
    )


def sample_curve(mesh, curve, h, samples_per_h=32):
    xyz = mesh.vertices[curve["mesh_vertices"]]
    vectors = np.diff(xyz, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 0):
        raise ValueError("zero-length reference segment")
    total = lengths.sum()
    count = max(64, math.ceil(total / h * samples_per_h))
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    positions = (np.arange(count) + 0.5) * total / count
    ids = np.minimum(
        np.searchsorted(cumulative[1:], positions, side="right"), len(lengths) - 1
    )
    t = (positions - cumulative[ids]) / lengths[ids]
    return (
        xyz[ids] + t[:, None] * vectors[ids],
        vectors[ids] / lengths[ids, None],
        total / count,
    )


def segment_query(points, segments, tangents=None, max_angle=15.0):
    """Exact nearest segment; nearest eligible aligned segment is separate."""
    a = segments[:, 0]
    v = segments[:, 1] - a
    l2 = np.einsum("ij,ij->i", v, v)
    unit = v / np.maximum(np.sqrt(l2)[:, None], 1e-30)
    distances = []
    angles = []
    aligned = []
    match = []
    for first in range(0, len(points), 64):
        p = points[first : first + 64]
        delta = p[:, None] - a
        t = np.clip(np.einsum("pki,ki->pk", delta, v) / np.maximum(l2, 1e-30), 0, 1)
        d = np.linalg.norm(delta - t[:, :, None] * v, axis=2)
        nearest = d.argmin(1)
        distances.extend(d[np.arange(len(p)), nearest])
        if tangents is not None:
            an = np.degrees(
                np.arccos(np.clip(abs(tangents[first : first + 64] @ unit.T), 0, 1))
            )
            angles.extend(an[np.arange(len(p)), nearest])
            eligible = (an <= max_angle) & (l2[None] > 1e-30)
            ad = np.where(eligible, d, np.inf)
            mid = ad.argmin(1)
            aligned.extend(ad[np.arange(len(p)), mid])
            match.extend(mid)
    return (
        np.asarray(distances),
        np.asarray(angles),
        np.asarray(aligned),
        np.asarray(match, int),
    )


def longest_gap(covered, closed=False):
    covered = np.asarray(covered, bool)
    if not len(covered) or covered.all():
        return 0
    if not covered.any():
        return len(covered)
    miss = ~covered
    if closed:
        miss = np.r_[miss, miss]
    best = now = 0
    for value in miss:
        now = now + 1 if value else 0
        best = max(best, now)
    return min(best, len(covered))


def stats(values):
    x = np.asarray(values, float)
    if not len(x):
        return dict(count=0, rms=None, p95=None, maximum=None)
    return dict(
        count=len(x),
        rms=float(np.sqrt(np.mean(x * x))),
        p95=float(np.percentile(x, 95)),
        maximum=float(x.max()),
    )


def graph_components(edges):
    adj = defaultdict(set)
    for a, b in edges:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    unseen = set(adj)
    parts = []
    while unseen:
        stack = [min(unseen)]
        unseen.remove(stack[0])
        part = []
        while stack:
            a = stack.pop()
            part.append(a)
            for b in adj[a]:
                if b in unseen:
                    unseen.remove(b)
                    stack.append(b)
        parts.append(part)
    return adj, parts


def loop_witness(source, quad, curve, h, distance_fraction=0.1, angle=15.0):
    """Sufficient, deliberately conservative witness; absence is inconclusive.

    An output edge is eligible only if BOTH ends and its midpoint lie near the
    reference and its tangent is aligned at the midpoint. A degree-two connected
    component must also cover every reference sample. No topological or geometric
    success is inferred from sample coverage alone.
    """
    if not curve["closed"]:
        return dict(status="not_applicable")
    ref = source.vertices[np.array(curve["mesh_edges"])]
    qe = unique_edges(quad)
    seg = quad.vertices[qe]
    d0 = segment_query(seg[:, 0], ref)[0]
    d1 = segment_query(seg[:, 1], ref)[0]
    v = seg[:, 1] - seg[:, 0]
    length = np.linalg.norm(v, axis=1)
    dm, an, _, _ = segment_query(
        seg.mean(1), ref, v / np.maximum(length[:, None], 1e-30)
    )
    mask = (
        (np.maximum.reduce([d0, d1, dm]) <= distance_fraction * h)
        & (an <= angle)
        & (length > 1e-14)
    )
    eligible = qe[mask]
    adj, parts = graph_components(eligible)
    p, t, _ = sample_curve(source, curve, h)
    cycles = []
    for part in parts:
        if len(part) < 3 or any(len(adj[v]) != 2 for v in part):
            continue
        ps = set(part)
        ed = np.array([e for e in eligible if int(e[0]) in ps])
        _, _, ad, _ = segment_query(p, quad.vertices[ed], t, max_angle=angle)
        if not np.all(ad <= distance_fraction * h):
            continue
        ids = _chains({tuple(map(int, e)) for e in ed})[0]
        rxyz = source.vertices[curve["mesh_vertices"]]
        qxyz = quad.vertices[ids]
        origin = rxyz[:-1].mean(0)
        _, _, vt = np.linalg.svd(rxyz[:-1] - origin, full_matrices=False)
        normal = vt[-1]
        basis = vt[:2]

        def area(x):
            uv = (x - origin) @ basis.T
            return float(
                abs(np.sum(uv[:-1, 0] * uv[1:, 1] - uv[1:, 0] * uv[:-1, 1])) / 2
            )

        ar = area(rxyz)
        aq = area(qxyz)
        cycles.append(
            dict(
                vertex_ids=list(map(int, ids)),
                edge_count=len(ed),
                reference_polygon_area=ar,
                output_projected_polygon_area=aq,
                projected_area_relative_change=aq / ar - 1 if ar else None,
                plane_deviation_max=float(np.abs((qxyz - origin) @ normal).max()),
            )
        )
    return dict(
        status="witness_found" if cycles else "not_demonstrated",
        eligible_edges=int(mask.sum()),
        components=len(parts),
        cycles=cycles,
        limitation="Sufficient sampled witness only; projected area is not analytic circle area.",
    )


def diagnose_curve(source, quad, curve, h):
    p, t, ds = sample_curve(source, curve, h)
    d, an, aligned, match = segment_query(p, quad.vertices[unique_edges(quad)], t)
    surf = surface_distances(p, quad)
    sensitivity = []
    for fraction in (0.05, 0.1, 0.2):
        ok = aligned <= fraction * h
        sensitivity.append(
            dict(
                distance_h=fraction,
                angle_degrees=15.0,
                coverage=float(ok.mean()),
                longest_gap_absolute=longest_gap(ok, curve["closed"]) * ds,
                longest_gap_fraction=longest_gap(ok, curve["closed"]) / len(ok),
            )
        )
    return dict(
        id=curve["id"],
        label=curve.get("label", curve["id"]),
        closed=curve["closed"],
        length=curve["length"],
        sample_spacing=ds,
        sample_count=len(p),
        surface_distance=stats(surf),
        edge_distance=stats(d),
        nearest_edge_angle_degrees=stats(an),
        edge_distance_p95_h=float(np.percentile(d, 95) / h),
        surface_distance_p95_h=float(np.percentile(surf, 95) / h),
        sensitivity=sensitivity,
        loop=loop_witness(source, quad, curve, h),
        # Minimal arrays for plots and gap localization; no acceptance flag.
        samples=dict(
            points=p.tolist(),
            surface_distance=surf.tolist(),
            aligned_covered=(aligned <= 0.1 * h).tolist(),
        ),
    )


def planar_sections(mesh, axis, level, tolerance=1e-8):
    """Intersect the triangulated surface with a non-coplanar diagnostic plane.

    Returns closed polygonal sections and incomplete chains separately. Coplanar
    triangles are reported, not silently converted into arbitrary contours.
    """
    from weak_layout_pipeline.pipeline.sampling import triangles

    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    anchor = np.eye(3)[np.argmin(abs(axis))]
    b1 = np.cross(axis, anchor)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(axis, b1)
    points = []
    lookup = {}
    edges = set()
    coplanar = 0

    def vertex(p):
        key = tuple(np.rint(p / tolerance).astype(np.int64))
        if key not in lookup:
            lookup[key] = len(points)
            points.append(p)
        return lookup[key]

    for tri in triangles(mesh):
        signed = tri @ axis - level
        if np.all(abs(signed) <= tolerance):
            coplanar += 1
            continue
        intersections = []
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            da, db = signed[k], signed[(k + 1) % 3]
            if abs(da) <= tolerance:
                intersections.append(a)
            if da * db < 0 and abs(da) > tolerance and abs(db) > tolerance:
                intersections.append(a + (b - a) * da / (da - db))
        ids = sorted({vertex(p) for p in intersections})
        if len(ids) == 2 and ids[0] != ids[1]:
            edges.add(tuple(ids))
    out = []
    for ids in _chains(edges):
        p = np.array(points)[ids]
        uv = p @ np.array([b1, b2]).T
        closed = ids[0] == ids[-1]
        area = (
            float(abs(np.sum(uv[:-1, 0] * uv[1:, 1] - uv[1:, 0] * uv[:-1, 1])) / 2)
            if closed
            else None
        )
        out.append(
            dict(
                closed=closed,
                points=p.tolist(),
                area=area,
                perimeter=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                bbox_min=p.min(0).tolist(),
                bbox_max=p.max(0).tolist(),
            )
        )
    return dict(
        level=level, axis=axis.tolist(), coplanar_triangle_count=coplanar, contours=out
    )
