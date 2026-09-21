"""Immutable subprocess recording for adaptive MIQ and frozen QEx."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(argv: list[str], cwd: Path, stem: Path, timeout: int) -> dict[str, object]:
    started = time.perf_counter()
    try:
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        process = subprocess.run(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            env=env,
        )
        returncode = process.returncode
        stdout, stderr = process.stdout, process.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout, stderr = exc.stdout or b"", exc.stderr or b""
        timed_out = True
    except OSError as exc:
        returncode, stdout, stderr, timed_out = 127, b"", str(exc).encode(), False
    stem.with_suffix(".stdout.log").write_bytes(stdout)
    stem.with_suffix(".stderr.log").write_bytes(stderr)
    record = {
        "argv": argv,
        "shell_command": shlex.join(argv),
        "cwd": str(cwd),
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": str(stem.with_suffix(".stdout.log")),
        "stderr": str(stem.with_suffix(".stderr.log")),
    }
    stem.with_suffix(".json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def backend_capabilities(executable: str | Path) -> dict:
    try:
        probe = subprocess.run(
            [str(Path(executable).resolve()), "--capabilities"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        result = json.loads(probe.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError(
            "rebuild the adaptive MIQ v2 adapter; capability probe failed"
        ) from exc
    if (
        not isinstance(result, dict)
        or not result.get("density_aware")
        or not result.get("stiffness_iterations")
    ):
        raise ValueError(
            "the baseline-guided pipeline requires the adaptive MIQ v2 adapter"
        )
    return result


def run_backend(
    *,
    mesh: str | Path,
    pd1: str | Path,
    pd2: str | Path,
    density: str | Path,
    gsize: float,
    output_dir: str | Path,
    miq_executable: str | Path,
    qex_executable: str | Path,
    hard_edges: str | Path | None = None,
    timeout_seconds: int = 1800,
    stiffness_iterations: int | None = None,
) -> dict[str, object]:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"backend output already exists: {output}")
    output.mkdir(parents=True)
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "mesh": mesh,
            "pd1": pd1,
            "pd2": pd2,
            "density": density,
            "miq_executable": miq_executable,
            "qex_executable": qex_executable,
        }.items()
    }
    if hard_edges is not None:
        paths["hard_edges"] = Path(hard_edges).resolve()
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    prefix = output / "miq"
    miq_argv = [
        str(paths["miq_executable"]),
        str(paths["mesh"]),
        str(paths["pd1"]),
        str(paths["pd2"]),
        str(paths["density"]),
        str(prefix),
        str(gsize),
    ]
    if "hard_edges" in paths and paths["hard_edges"].stat().st_size:
        miq_argv.append(str(paths["hard_edges"]))
    if stiffness_iterations is not None:
        if (
            not isinstance(stiffness_iterations, int)
            or not 0 <= stiffness_iterations <= 100
        ):
            raise ValueError("stiffness_iterations must be an integer in [0,100]")
        miq_argv.extend(["--stiffness-iterations", str(stiffness_iterations)])
    miq_record = _run(miq_argv, output, output / "miq_command", timeout_seconds)
    qex_record: dict[str, object] = {"returncode": None, "skipped": True}
    if miq_record["returncode"] == 0:
        qex_argv = [
            str(paths["qex_executable"]),
            str(paths["mesh"]),
            str(prefix),
            str(output / "quad.obj"),
            "0",
            "0.02",
            "50",
        ]
        qex_record = _run(qex_argv, output, output / "qex_command", timeout_seconds)
        qex_record["skipped"] = False
    record = {
        "schema_version": "weak-layout.backend-run.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gsize": gsize,
        "stiffness_iterations": stiffness_iterations,
        "thread_policy": "OMP/OPENBLAS/MKL_NUM_THREADS=1",
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "miq": miq_record,
        "qex": qex_record,
        "success": miq_record["returncode"] == 0
        and qex_record.get("returncode") == 0
        and (output / "quad.obj").is_file(),
    }
    (output / "backend_run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record
