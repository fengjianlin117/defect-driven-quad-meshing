"""Explicit maintainer operation, after reviewing changes to published materials."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'experiments/manifests/publication_files.json'
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache', 'outputs', 'build'}


def main():
    rows = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path == TARGET or any(part in SKIP for part in path.relative_to(ROOT).parts):
            continue
        raw = path.read_bytes()
        rows.append(dict(path=str(path.relative_to(ROOT)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(dict(format=1, scope='Published files; excludes this manifest and local build/output/cache directories.', files=rows), indent=2)+'\n')
    print(f'Recorded {len(rows)} files. Review this diff before committing.')


if __name__ == '__main__':
    main()
