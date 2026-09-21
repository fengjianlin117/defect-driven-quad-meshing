"""Rebuild a reference and initial proposal, without training or backend calls."""
import argparse
import json
from pathlib import Path
import numpy as np
from project_paths import load_case
from weak_layout_pipeline.pipeline.mesh import load_obj
from reference_graph import build_reference_graph
from verify_snapshot import first_differences
from defect_selection_lexicographic import ProposalContext
from conflict_selection import EqualityAudit
from geometry_budget import propose
from protection_feedback import simple_initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    plan = load_case(args.model)
    if args.out.exists():
        parser.error('Output already exists; choose a new directory.')
    mesh = load_obj(plan['source'])
    graph = build_reference_graph(mesh)
    frozen = json.loads(Path(plan['graph']).read_text())
    differences = first_differences(json.loads(json.dumps(graph)), frozen)
    if differences:
        raise RuntimeError('Rebuilt reference differs from the frozen reference: ' + '; '.join(differences))
    baseline = json.loads(Path(plan['baseline_record']).read_text())
    folder = Path(plan['source']).parent / 'baseline'
    audit = EqualityAudit(mesh, np.loadtxt(folder/'miq_uv.txt', skiprows=1),
        np.loadtxt(folder/'miq_fuv.txt', skiprows=1, dtype=int),
        np.loadtxt(folder/'miq_combed_PD1.txt')[:, -3:],
        np.loadtxt(folder/'miq_combed_PD2.txt')[:, -3:])
    groups = {g['id']: {tuple(sorted(e['vertices'])) for e in graph['source_edges']
                       if e['layer'] == 'main' and g['id'] in e['group_ids']}
              for g in graph['groups'] if g['layer'] == 'main'}
    selection = simple_initial(groups, baseline['edge_defects'], audit)
    density, sizes, budget, requests = propose(mesh, graph, plan['reference_h'], baseline['quads'])
    args.out.mkdir(parents=True, exist_ok=False)
    for filename, value in [('graph.json', graph), ('selection.json', selection), ('budget.json', budget)]:
        (args.out/filename).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    np.savetxt(args.out/'allocated.rho', density)
    np.savetxt(args.out/'vertex_sizes.txt', sizes)
    summary = dict(model=args.model, source_triangles=mesh.face_count,
                   reference_matches_frozen=True, baseline_quads=baseline['quads'],
                   target_quads=budget['target_quads'],
                   selected_groups=selection['selected_group_ids'],
                   diagnostic_source='frozen baseline edge_defects; not newly measured',
                   new_backend_calls=0, output=str(args.out.resolve()))
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
