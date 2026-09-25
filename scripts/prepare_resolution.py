"""Regenerate an adequate unconstrained baseline before diagnosing defects."""
import math
from pathlib import Path
import shutil
import subprocess
import json
import numpy as np
from coordinator import read, save, sha
from resolution_budget import count_in_band
from neutral_evaluation import evaluate
from weak_layout_pipeline.pipeline.mesh import load_obj, bbox_diagonal
from weak_layout_pipeline.backend.runner import run_backend


def geometry_check(path, checker):
    checks = []
    for diagonal in [0, 1]:
        try:
            result = subprocess.run([str(checker), str(path), str(diagonal)],
                                    capture_output=True, text=True, timeout=30)
            row = json.loads(result.stdout) if result.returncode == 0 else dict(completed=False)
            row.update(returncode=result.returncode, stderr=result.stderr)
        except Exception as exc:
            row = dict(completed=False, error=repr(exc))
        checks.append(row)
    return dict(checks=checks, clear=all(c.get('completed') is True
                and c.get('intersection_pair_count') == 0
                and c.get('exact_degenerate_triangles') == 0 for c in checks))


def prepare_resolution(plan, out, policy, miq, qex, checker):
    """At most three solves. Failure never silently restores the coarse baseline."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'policy.json', policy)
    mesh = load_obj(plan['source'])
    graph = read(plan['graph'])
    source_audit = geometry_check(Path(plan['source']), checker)
    save(out/'source_intersection.json', source_audit)
    provenance = {str(p): sha(p) for p in [plan['source'], plan['pd1'], plan['pd2'],
                  plan['graph'], plan['baseline_record'], miq, qex, checker]}
    save(out/'input_hashes.json', provenance)
    summary = dict(policy=policy, attempts=[], new_backend_calls=0, success=False)
    save(out/'summary.json', summary)
    if not source_audit['clear']:
        raise RuntimeError('Source geometry check failed/incomplete; see resolution/source_intersection.json')
    density = out/'uniform.rho'
    np.savetxt(density, np.ones(mesh.face_count), fmt='%.17g')
    old_count = read(plan['baseline_record'])['quads']
    g = plan['gsize'] * math.sqrt(policy['target_quads']/old_count)
    for index in range(policy['max_baseline_attempts']):
        folder = out/f'attempt_{index}'
        folder.mkdir()
        print('BASELINE', plan['model'], 'attempt', index, 'target', policy['target_quads'], flush=True)
        backend = run_backend(mesh=plan['source'], pd1=plan['pd1'], pd2=plan['pd2'],
                              density=density, gsize=g, output_dir=folder/'backend',
                              miq_executable=miq, qex_executable=qex,
                              stiffness_iterations=0, timeout_seconds=60)
        summary['new_backend_calls'] += 1
        row = dict(attempt=index, gsize=g, process_success=backend['success'])
        measured = None
        if backend['success']:
            try:
                quad = load_obj(folder/'backend/quad.obj', require_triangles=False)
                h = bbox_diagonal(mesh)/g
                measured = evaluate(mesh, quad, graph, h)
                save(folder/'baseline_record.json', measured)
                audit = geometry_check(folder/'backend/quad.obj', checker)
                save(folder/'intersection.json', audit)
                row.update(quads=measured['quads'], basic_output_pass=measured['basic_output_pass'],
                           strict_output_pass=measured['basic_output_pass'] and audit['clear'],
                           count_matched=count_in_band(measured['quads'], policy), reference_h=h)
            except Exception as exc:
                row['evaluation_error'] = repr(exc)
        summary['attempts'].append(row)
        save(folder/'result.json', row)
        save(out/'summary.json', summary)
        print('BASELINE_RESULT', plan['model'], row, flush=True)
        if row.get('strict_output_pass') and row.get('count_matched'):
            inputs = out/'inputs'
            inputs.mkdir()
            updated = dict(plan)
            for key, name in [('source', 'source.obj'), ('pd1', 'PD1.txt'),
                              ('pd2', 'PD2.txt'), ('graph', 'graph.json')]:
                shutil.copy2(plan[key], inputs/name)
                updated[key] = str(inputs/name)
            shutil.copytree(folder/'backend', inputs/'baseline')
            shutil.copy2(folder/'baseline_record.json', inputs/'baseline_record.json')
            updated.update(baseline_record=str(inputs/'baseline_record.json'),
                           density=str(density), gsize=g, reference_h=h,
                           target_quads=policy['target_quads'],
                           max_quads=policy['max_actual_quads'],
                           min_quads=policy['min_actual_quads'], resolution_policy=policy)
            save(out/'plan.json', updated)
            summary.update(success=True, selected_attempt=index, actual_quads=measured['quads'])
            save(out/'summary.json', summary)
            return updated, summary
        actual = row.get('quads')
        # No count measurement => no defensible calibration. Do not repeat the
        # same deterministic failed solve or proceed with the archived baseline.
        if not actual or row.get('count_matched'):
            break
        aim = min(policy['target_quads'], .98*policy['max_actual_quads'])
        g *= math.sqrt(aim/actual)
    raise RuntimeError('No valid baseline within the requested count band; '
                       'see resolution/summary.json. No coarse fallback exported.')
