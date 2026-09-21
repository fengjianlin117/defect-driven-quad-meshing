"""Run current method on a bundled model with explicit native executables."""
import argparse
import json
from pathlib import Path
import shutil
from project_paths import load_case
from conditional_coordinator import ConditionalCoordinator
from coordinator_v5 import CoordinatorV5


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
    args = parser.parse_args()
    plan = load_case(args.model)
    out = args.out.resolve()
    if out.exists():
        parser.error('Output already exists; choose a new directory.')
    cls = ConditionalCoordinator if args.strategy == 'conditional' else CoordinatorV5
    coordinator = cls(plan, out, args.miq, args.qex, args.checker)
    decision = coordinator.run()
    # The archived baseline metrics keep provenance references. Resolve the actual output here.
    chosen = decision['recommended_id']
    row = next((r for r in coordinator.rows if r['id'] == chosen), None)
    quad = coordinator.inp/'quad.obj' if chosen == 'baseline' else Path(row['quad_path']) if row else None
    if quad and quad.is_file():
        (out/'final').mkdir()
        shutil.copy2(quad, out/'final/quad.obj')
    summary = dict(model=args.model, strategy=args.strategy, recommended_id=chosen,
                   actual_quads=row.get('quads') if row else None,
                   new_backend_calls=coordinator.native, logical_attempts=coordinator.attempts,
                   final_quad=str(out/'final/quad.obj') if quad and quad.is_file() else None)
    (out/'run_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
