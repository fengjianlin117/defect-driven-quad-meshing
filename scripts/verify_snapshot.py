"""Verify exported hashes and all bundled references; no native or GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from project_paths import ROOT, models, load_case
from weak_layout_pipeline.pipeline.mesh import load_obj
from reference_graph import build_reference_graph


def first_differences(actual, expected, path='$', out=None):
    out = [] if out is None else out
    if len(out) >= 8 or actual == expected:
        return out
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            out.append(f'{path}: dictionary keys differ')
        for key in actual.keys() & expected.keys():
            first_differences(actual[key], expected[key], f'{path}.{key}', out)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            out.append(f'{path}: length {len(actual)} vs {len(expected)}')
        for index, (left, right) in enumerate(zip(actual, expected)):
            first_differences(left, right, f'{path}[{index}]', out)
    else:
        out.append(f'{path}: {actual!r} vs {expected!r}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hashes-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'experiments/manifests/publication_files.json').read_text())
    failures = []
    for row in manifest['files']:
        file = (ROOT/row['path']).resolve()
        if not file.is_relative_to(ROOT) or not file.is_file():
            failures.append(row['path']+': missing/invalid')
        elif hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            failures.append(row['path']+': hash changed')
    if failures:
        raise SystemExit('\n'.join(failures))
    checked = []
    if not args.hashes_only:
        for item in models():
            plan = load_case(item['model'])
            mesh = load_obj(plan['source'])
            graph = json.loads(json.dumps(build_reference_graph(mesh)))
            if graph != json.loads(Path(plan['graph']).read_text()):
                raise RuntimeError(f'Reference mismatch: {item["model"]}: ' + '; '.join(first_differences(graph, json.loads(Path(plan['graph']).read_text()))))
            fields = [np.loadtxt(plan[k]) for k in ['pd1', 'pd2']]
            for field in fields:
                if field.shape not in [(mesh.face_count, 3), (mesh.face_count, 4)] or not np.isfinite(field).all():
                    raise RuntimeError(f'Field shape/value mismatch: {item["model"]}')
                if field.shape[1] == 4 and not np.array_equal(field[:, 0], np.arange(mesh.face_count)):
                    raise RuntimeError(f'Face index mismatch: {item["model"]}')
            checked.append(item['model'])
    print(json.dumps(dict(files_verified=len(manifest['files']), reference_graphs_identical=checked,
                          new_backend_calls=0, training_runs=0), indent=2))


if __name__ == '__main__':
    try:
        main()
    except (Exception, SystemExit) as error:
        if isinstance(error, SystemExit) and not error.code:
            raise
        message = str(error).replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
        print(f'::error file=scripts/verify_snapshot.py,title=Snapshot verification::{message}')
        raise
