"""Run current method on a bundled model with explicit native executables."""
import argparse
import json
from pathlib import Path
import shutil
from project_paths import load_case
from conditional_coordinator import ConditionalCoordinator
from coordinator_v5 import CoordinatorV5
from coordinator import read
from resolution_budget import estimate_resolution
from prepare_resolution import prepare_resolution
from weak_layout_pipeline.pipeline.mesh import load_obj


def executable(value):
    path = Path(value).expanduser()
    if not path.is_file():
        found = shutil.which(value)
        if not found:
            raise argparse.ArgumentTypeError(f'Executable not found: {value}')
        path = Path(found)
    return str(path.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--miq', required=True, type=executable)
    parser.add_argument('--qex', required=True, type=executable)
    parser.add_argument('--checker', required=True, type=executable)
    parser.add_argument('--strategy', choices=['conditional', 'always-feedback'], default='conditional')
    parser.add_argument('--resolution', choices=['auto', 'archived'], default='auto',
                        help='auto rebuilds the starting baseline; archived reproduces the old baseline policy')
    parser.add_argument('--target-quads', type=int, help='Starting uniform baseline target (default: max(512, old baseline, source triangles/2))')
    parser.add_argument('--max-quads', type=int, help='Actual count cap throughout the run (auto: 8192; archived: 2048)')
    args = parser.parse_args()
    plan = load_case(args.model)
    out = args.out.resolve()
    if out.exists():
        parser.error('Output already exists; choose a new directory.')
    cap = args.max_quads if args.max_quads is not None else (8192 if args.resolution == 'auto' else 2048)
    if cap <= 0:
        parser.error('--max-quads must be positive.')
    if args.resolution == 'archived' and args.target_quads is not None:
        parser.error('--target-quads requires --resolution auto.')
    policy = None
    if args.resolution == 'auto':
        try:
            policy = estimate_resolution(load_obj(plan['source']).face_count,
                                         read(plan['baseline_record'])['quads'], args.target_quads, cap)
        except ValueError as exc:
            parser.error(str(exc))
    out.mkdir(parents=True)
    prep_summary = dict(new_backend_calls=0)
    if policy is not None:
        try:
            plan, prep_summary = prepare_resolution(plan, out/'resolution', policy, args.miq, args.qex, args.checker)
        except Exception as exc:
            prep_path = out/'resolution/summary.json'
            if prep_path.is_file():
                prep_summary = read(prep_path)
            (out/'run_summary.json').write_text(json.dumps(dict(model=args.model, status='resolution_failed',
                resolution=policy, error=str(exc), final_quad=None,
                new_backend_calls=prep_summary['new_backend_calls']), indent=2)+'\n')
            raise
    else:
        plan.update(max_quads=cap, min_quads=0)
    cls = ConditionalCoordinator if args.strategy == 'conditional' else CoordinatorV5
    coordinator = cls(plan, out/'method', args.miq, args.qex, args.checker)
    decision = coordinator.run()
    # The archived baseline metrics keep provenance references. Resolve the actual output here.
    chosen = decision['recommended_id']
    row = next((r for r in coordinator.rows if r['id'] == chosen), None)
    quad = coordinator.inp/'quad.obj' if chosen == 'baseline' else Path(row['quad_path']) if row else None
    if quad and quad.is_file():
        (out/'final').mkdir()
        shutil.copy2(quad, out/'final/quad.obj')
    summary = dict(model=args.model, strategy=args.strategy, recommended_id=chosen,
                   status='completed' if chosen else 'no_valid_recommendation',
                   resolution=policy or dict(policy='archived', max_actual_quads=cap),
                   baseline_quads=coordinator.base['quads'], initial_target_quads=coordinator.target,
                   actual_quads=row.get('quads') if row else None,
                   baseline_backend_calls=prep_summary['new_backend_calls'],
                   method_backend_calls=coordinator.native,
                   new_backend_calls=prep_summary['new_backend_calls']+coordinator.native,
                   logical_attempts=coordinator.attempts,
                   final_quad=str(out/'final/quad.obj') if quad and quad.is_file() else None)
    (out/'run_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
