"""Repository-relative input resolution for the collaboration snapshot."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'method'))


def models():
    return json.loads((ROOT / 'experiments/manifests/models.json').read_text())


def load_case(model):
    candidates = [p for p in models() if p['model'] == model]
    if len(candidates) != 1:
        raise ValueError(f'Unknown or duplicate model {model!r}')
    plan = dict(candidates[0])
    for key in ['source', 'pd1', 'pd2', 'graph', 'baseline_record', 'density']:
        path = (ROOT / plan[key]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Invalid or missing {key}: {path}')
        plan[key] = str(path)
    return plan
