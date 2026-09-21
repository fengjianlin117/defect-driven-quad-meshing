"""Lossless reference graph packaging, independent of model labels and backend.

Candidate curves are groups, not disjoint constraints. Atomic arcs partition
source edges and preserve every group by membership. No feature is accepted or
discarded based on the downstream size or on a backend run.
"""
from collections import defaultdict, Counter
import numpy as np
from weak_layout_pipeline.pipeline.mesh import edge_topology, triangle_geometry
from geometric_inventory import build_geometric_inventory, continuation_chains
from weak_layout_pipeline.structural_lines.baseline_guided import feature_graph, _chains
from feature_diagnostics import edge_set, curve_record


def canonical(e):
    return tuple(sorted(map(int, e)))


def build_reference_graph(mesh):
    inv = build_geometric_inventory(mesh)
    strong = edge_set(inv['curves'])
    weak = edge_set(feature_graph(mesh, sharp_degrees=20.)['arcs']) - strong
    curves = []
    for c in inv['curves']:
        curves.append(dict(c, layer='main', status='geometric_candidate'))
    for i, ids in enumerate(continuation_chains(mesh, weak)[0]):
        c = curve_record(mesh, ids, f'weak-{i:04d}', 'supplementary_20_to_30_degrees')
        curves.append(dict(c, layer='weak', status='uncertain_candidate',
                           grouping_evidence=['weak_edge_continuation']))
    membership = defaultdict(list)
    for c in curves:
        for e in edge_set([c]):
            membership[e].append(c['id'])
    buckets = defaultdict(set)
    graph_neighbors = defaultdict(set)
    for e, members in membership.items():
        buckets[tuple(sorted(members))].add(e)
        a, b = e
        graph_neighbors[a].add(b); graph_neighbors[b].add(a)
    def split_at_junctions(ids):
        # A group can pass straight through a branch belonging to another group.
        # The graph must nevertheless expose that shared vertex as an arc end.
        if ids[0] == ids[-1]:
            branch = next((i for i, v in enumerate(ids[:-1]) if len(graph_neighbors[v]) != 2), None)
            if branch is None:
                return [ids]
            core = ids[:-1]
            core = core[branch:] + core[:branch]
            ids = core + [core[0]]
        cuts = [0] + [i for i in range(1, len(ids)-1) if len(graph_neighbors[ids[i]]) != 2] + [len(ids)-1]
        return [ids[a:b+1] for a,b in zip(cuts[:-1], cuts[1:])]
    atoms = []
    for members, edges in sorted(buckets.items()):
        for ids in [part for path in _chains(edges) for part in split_at_junctions(path)]:
            atom = curve_record(mesh, ids, f'arc-{len(atoms):04d}', 'unique_source_edge_partition')
            atom.update(group_ids=list(members), layer='main' if canonical(ids[:2]) in strong else 'weak')
            atoms.append(atom)
    edge_to_atom = {e: a['id'] for a in atoms for e in edge_set([a])}
    for c in curves:
        c['atomic_arc_ids'] = sorted({edge_to_atom[e] for e in edge_set([c])})
        p = mesh.vertices[c['mesh_vertices'][:-1] if c['closed'] else c['mesh_vertices']]
        if len(p) >= 3:
            # Descriptor only; no label or inclusion decision depends on it.
            _, _, vt = np.linalg.svd(p - p.mean(axis=0), full_matrices=False)
            c['best_fit_plane_max_distance'] = float(np.max(abs((p - p.mean(axis=0)) @ vt[-1])))
    incidents, _ = edge_topology(mesh)
    normals = triangle_geometry(mesh)[0]
    edges = []
    neighbors = defaultdict(set)
    for e, members in sorted(membership.items()):
        a, b = e
        neighbors[a].add(b); neighbors[b].add(a)
        fs = [int(f) for f, _ in incidents[e]]
        angle = 180. if len(fs) == 1 else float(np.degrees(np.arccos(np.clip(normals[fs[0]] @ normals[fs[1]], -1, 1))))
        edges.append(dict(vertices=list(e), adjacent_faces=fs, dihedral_degrees=angle,
                          length=float(np.linalg.norm(mesh.vertices[a] - mesh.vertices[b])),
                          layer='main' if e in strong else 'weak', group_ids=sorted(members), atomic_arc_id=edge_to_atom[e]))
    endpoints = {int(v) for a in atoms for v in (a['mesh_vertices'][0], a['mesh_vertices'][-1])}
    nodes = [dict(mesh_vertex=v, position=mesh.vertices[v].tolist(), degree=len(neighbors[v]),
                  role='junction' if len(neighbors[v]) > 2 else 'endpoint_or_group_boundary_or_cycle_anchor') for v in sorted(endpoints)]
    graph = dict(schema='reference-graph.v1', index_base=0, extraction=dict(main_dihedral_degrees=30.,
                 supplementary_dihedral_degrees=20., supplied_axis=False, supplied_labels=False, backend_used=False),
                 groups=curves, atomic_arcs=atoms, nodes=nodes, source_edges=edges,
                 ambiguous_continuations=inv['ambiguous_continuations'],
                 limitations=['original input geometry only; noise robustness unresolved',
                              '20-degree evidence floor is not a completeness certificate',
                              'smooth ridges and valleys not extracted',
                              'geometric candidates are not automatically required constraints',
                              'overlapping groups must use edge or arc unions for benefit accounting'])
    graph['audit'] = audit_reference_graph(mesh, graph)
    return graph


def audit_reference_graph(mesh, graph):
    """Fail closed on invalid source indices, repeated edges or lost group paths."""
    incidents = edge_topology(mesh)[0]
    atoms = {a['id']: a for a in graph['atomic_arcs']}
    errors = []
    counts = Counter(e for a in atoms.values() for e in edge_set([a]))
    declared = {canonical(e['vertices']) for e in graph['source_edges']}
    if set(counts) != declared or any(n != 1 for n in counts.values()):
        errors.append('atomic arcs do not partition source edges exactly once')
    if len(declared) != len(graph['source_edges']):
        errors.append('duplicate edge records')
    degree = Counter(v for e in declared for v in e)
    for a in atoms.values():
        if any(degree[v] != 2 for v in a['mesh_vertices'][1:-1]):
            errors.append(a['id'] + ': graph junction hidden inside arc')
    if declared - set(incidents):
        errors.append('non-source edge')
    for c in graph['groups'] + graph['atomic_arcs']:
        ids = c['mesh_vertices']
        walked = [canonical(e) for e in zip(ids[:-1], ids[1:])]
        if len(walked) != len(set(walked)) or set(walked) != edge_set([c]):
            errors.append(c['id'] + ': invalid or repeated path edges')
        if c['closed'] != (ids[0] == ids[-1]):
            errors.append(c['id'] + ': closure mismatch')
        unique_vertices = ids[:-1] if c['closed'] else ids
        if len(unique_vertices) != len(set(unique_vertices)):
            errors.append(c['id'] + ': self-touching vertex path')
    for c in graph['groups']:
        assembled = set().union(*(edge_set([atoms[a]]) for a in c['atomic_arc_ids']))
        if assembled != edge_set([c]):
            errors.append(c['id'] + ': group cannot be reconstructed from atoms')
    if errors:
        raise ValueError(errors)
    return dict(passed=True, errors=[], unique_source_edges=len(declared), atomic_arcs=len(atoms),
                groups=len(graph['groups']), closed_groups=sum(c['closed'] for c in graph['groups']),
                duplicate_atomic_edges=0, lost_group_edges=0, invented_source_edges=0,
                scope='source topology and grouping consistency; no semantic completeness or spatial self-intersection certificate')
