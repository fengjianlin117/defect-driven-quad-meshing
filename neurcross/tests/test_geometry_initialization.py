from argparse import ArgumentParser
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import re
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


@contextmanager
def _raises(expected, match=None):
    try:
        yield
    except expected as error:
        if match is not None and re.search(match, str(error)) is None:
            raise AssertionError(
                "exception {!r} does not match {!r}".format(str(error), match))
        return
    raise AssertionError("{} was not raised".format(expected.__name__))


class _PytestCompatibility:
    """The small raises subset keeps these tests runnable without pytest."""

    @staticmethod
    def raises(expected, match=None):
        return _raises(expected, match=match)


pytest = _PytestCompatibility()

ROOT = Path(__file__).resolve().parents[1]
QUAD_MESH = ROOT / "quad_mesh"
sys.path.insert(0, str(QUAD_MESH))

import geometry_initialization as geometry
import quad_mesh_args
import convergence_stop as convergence


def _write_mesh(path):
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")


def _write_target(path, mesh_path, mesh_sha=None, face_order=None,
                  face_count=None, vertex_count=None, z4=None,
                  constraint_target_z4=None, include_constraint_target=True):
    mesh_sha = geometry.sha256_file(mesh_path) if mesh_sha is None else mesh_sha
    face_order = np.array([0], dtype=np.int64) if face_order is None else face_order
    face_count = (np.array(1, dtype=np.int64)
                  if face_count is None else face_count)
    vertex_count = (np.array(3, dtype=np.int64)
                    if vertex_count is None else vertex_count)
    z4 = (np.array([1.0 + 0.0j], dtype=np.complex128)
          if z4 is None else z4)
    constraint_target_z4 = (
        np.array([1.0 + 0.0j], dtype=np.complex128)
        if constraint_target_z4 is None else constraint_target_z4)
    payload = {
        "schema_version": np.array(geometry.GEOMETRY_INIT_SCHEMA),
        "mesh_sha256": np.array(mesh_sha), "face_count": face_count,
        "vertex_count": vertex_count, "face_order": face_order,
        "field": np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float64),
        "confidence": np.array([0.75], dtype=np.float64),
        "constrained": np.array([True], dtype=np.bool_), "z4": z4,
    }
    if include_constraint_target:
        payload["constraint_target_z4"] = constraint_target_z4
    np.savez(path, **payload)


def _args(**updates):
    values = dict(
        angle_init="random", geometry_init_file=None, load_path=None,
        geometry_prefit_lr=1e-3, geometry_prefit_max_steps=500,
        geometry_prefit_target_deg=5.0, research_checkpoint_steps=[],
        research_trace=False, research_trace_path=None, auto_min_steps=1500,
        research_publication_logdir=None,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def test_cli_defaults_are_inert_and_list_is_accepted():
    parser = quad_mesh_args.add_args(ArgumentParser())
    defaults = parser.parse_args([])
    assert defaults.angle_init == "random"
    assert defaults.geometry_init_file is None
    assert defaults.research_checkpoint_steps == []
    assert defaults.research_trace is False
    assert defaults.research_publication_logdir is None
    configured = parser.parse_args([
        "--angle_init", "geometry", "--geometry_init_file", "target.npz",
        "--geometry_prefit_lr", "0.0003", "--geometry_prefit_max_steps", "20",
        "--geometry_prefit_target_deg", "4", "--research_checkpoint_steps", "100", "1500",
        "--research_trace", "--research_trace_path", "/tmp/trace.jsonl",
        "--research_publication_logdir", "/published/training/shape",
    ])
    assert configured.research_checkpoint_steps == [100, 1500]
    assert configured.research_publication_logdir == "/published/training/shape"


def test_default_model_state_key_and_shape_contract():
    from models import Network_predict_angle

    net = Network_predict_angle(
        in_dim=3, angle_in_dim=12, decoder_hidden_dim=256, nl="sine",
        decoder_n_hidden_layers=4, init_type="siren",
        sphere_init_params=[1.6, 0.1], udf=False)
    rows = [
        "{}:{}:{}".format(name, tuple(value.shape), value.dtype)
        for name, value in net.state_dict().items()
    ]
    signature = hashlib.sha256(
        (chr(10).join(rows) + chr(10)).encode("utf-8")).hexdigest()
    # Frozen from the isolated formal-source snapshot be7f0905.
    assert len(rows) == 418
    assert signature == "50d0b48cb40c76c061784a960546d9361dee0d478b74d32aabf2e2554e0fbe81"


def test_research_contract_rejects_conflicts_and_unreachable_steps():
    assert geometry.validate_research_contract(_args(), "auto", 10000, 10000) == ()
    with pytest.raises(ValueError, match="requires --geometry_init_file"):
        geometry.validate_research_contract(_args(angle_init="geometry"), "auto", 10000, 10000)
    with pytest.raises(ValueError, match="mutually exclusive"):
        geometry.validate_research_contract(
            _args(angle_init="geometry", geometry_init_file="x.npz", load_path="model.pt"),
            "auto", 10000, 10000)
    with pytest.raises(ValueError, match="skipped by termination"):
        geometry.validate_research_contract(
            _args(research_checkpoint_steps=[1750]), "auto", 10000, 10000)
    with pytest.raises(ValueError, match="requires --research_trace"):
        geometry.validate_research_contract(
            _args(research_trace_path="trace.jsonl"), "auto", 10000, 10000)
    assert geometry.validate_research_contract(
        _args(research_checkpoint_steps=[1500, 1000]), "auto", 10000, 10000
    ) == (1000, 1500)


def test_publication_paths_are_lexical_inert_and_nonoverlapping(tmp_path):
    actual = tmp_path / "attempt" / "training" / "shape"
    publication = tmp_path / "published" / "training" / "shape"
    assert not publication.exists()
    forbidden = AssertionError("publication mapper performed filesystem I/O")
    with patch.object(Path, "stat", side_effect=forbidden):
        with patch.object(Path, "mkdir", side_effect=forbidden):
            with patch.object(Path, "resolve", side_effect=forbidden):
                mapper = geometry.build_research_publication_paths(
                    actual, str(publication))
                mapped = mapper.publication_reference(
                    actual / "discarded" / ".." /
                    "research_checkpoints" /
                    "step_00001500" / "field.txt")
                mapper.validate_trace_path(actual / "custom" / "trace.jsonl")
    assert mapper.enabled is True
    assert mapped == (
        publication / "research_checkpoints" /
        "step_00001500" / "field.txt")
    assert not publication.exists()

    with pytest.raises(ValueError, match="outside the actual"):
        mapper.publication_reference(actual.parent / "foreign.txt")
    with pytest.raises(ValueError, match="inside the actual"):
        mapper.validate_trace_path(tmp_path / "external" / "trace.jsonl")

    for overlapping in (actual, actual / "nested", actual.parent):
        with pytest.raises(ValueError, match="must not overlap"):
            geometry.build_research_publication_paths(
                actual, str(overlapping))
    with pytest.raises(ValueError, match="actual shape logdir must be absolute"):
        geometry.build_research_publication_paths(
            Path("relative-actual"), str(publication))
    with pytest.raises(ValueError, match="must be an absolute path"):
        geometry.build_research_publication_paths(
            actual, "relative-publication")
    with pytest.raises(ValueError, match="non-empty absolute"):
        geometry.build_research_publication_paths(actual, "")

    disabled = geometry.build_research_publication_paths(
        Path("relative-actual"), None)
    relative_reference = Path("relative-actual") / "artifact.json"
    assert disabled.enabled is False
    assert disabled.publication_reference(
        relative_reference) == relative_reference
    disabled.validate_trace_path(Path("external-trace.jsonl"))


def test_geometry_npz_and_companion_bind_exact_mesh(tmp_path):
    mesh = tmp_path / "mesh.obj"
    target_path = tmp_path / "geometry_init.npz"
    _write_mesh(mesh)
    _write_target(target_path, mesh)
    target = geometry.load_geometry_initialization(target_path, mesh)
    assert target.face_count == 1
    assert target.mesh_sha256 == geometry.sha256_file(mesh)
    assert np.array_equal(target.tangents, [[1.0, 0.0, 0.0]])

    companion_path = tmp_path / "geometry_init.json"
    companion_path.write_text(json.dumps({
        "schema_version": geometry.GEOMETRY_INIT_SCHEMA,
        "mesh": {"path": str(mesh), "sha256": target.mesh_sha256,
                 "vertex_count": 3, "face_count": 1},
        "field": {"npz": target_path.name, "txt": "geometry_init_field.txt",
                  "shape": [1, 6], "dtype": "float64", "coordinates": "world",
                  "face_order": "obj", "columns": []},
        "confidence": {"shape": [1], "range": [0, 1]},
    }), encoding="utf-8")
    assert geometry.load_geometry_initialization(companion_path, mesh).npz_path == target_path


def test_geometry_target_rejects_sha_and_face_order(tmp_path):
    mesh = tmp_path / "mesh.obj"
    target_path = tmp_path / "geometry_init.npz"
    _write_mesh(mesh)
    _write_target(target_path, mesh, mesh_sha="0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        geometry.load_geometry_initialization(target_path, mesh)
    _write_target(target_path, mesh, face_order=np.array([1], dtype=np.int64))
    with pytest.raises(ValueError, match="identity"):
        geometry.load_geometry_initialization(target_path, mesh)


def test_geometry_target_requires_complete_strict_bundle(tmp_path):
    mesh = tmp_path / "mesh.obj"
    target_path = tmp_path / "geometry_init.npz"
    _write_mesh(mesh)

    _write_target(target_path, mesh, include_constraint_target=False)
    with pytest.raises(ValueError, match="constraint_target_z4"):
        geometry.load_geometry_initialization(target_path, mesh)

    _write_target(
        target_path, mesh, face_count=np.array(1.0, dtype=np.float64))
    with pytest.raises(ValueError, match="integer NPZ scalar"):
        geometry.load_geometry_initialization(target_path, mesh)

    _write_target(
        target_path, mesh,
        vertex_count=np.array([3], dtype=np.int64))
    with pytest.raises(ValueError, match="integer NPZ scalar"):
        geometry.load_geometry_initialization(target_path, mesh)

    _write_target(
        target_path, mesh, z4=np.array([0.5 + 0.0j], dtype=np.complex128))
    with pytest.raises(ValueError, match="z4 must contain unit complex"):
        geometry.load_geometry_initialization(target_path, mesh)

    _write_target(
        target_path, mesh,
        constraint_target_z4=np.array([2.0 + 0.0j], dtype=np.complex128))
    with pytest.raises(ValueError, match="constraint_target_z4 must contain unit complex"):
        geometry.load_geometry_initialization(target_path, mesh)


class _ToyDataset:
    def __init__(self):
        self.points = np.array([[0.0, 0.0, 0.0], [0.3, 0.2, 0.0]], dtype=np.float32)
        self.mnfld_n = np.tile([[0.0, 0.0, 1.0]], (2, 1)).astype(np.float32)
        self.vector_u = np.tile([[1.0, 0.0, 0.0]], (2, 1)).astype(np.float32)
        self.vector_v = np.tile([[0.0, 1.0, 0.0]], (2, 1)).astype(np.float32)


class _ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(3, 1)
        self.decoder_angle = nn.Sequential(nn.Linear(12, 8), nn.Tanh(), nn.Linear(8, 1))


def _toy_target(tmp_path):
    angle = 0.2
    alpha = np.tile([[np.cos(angle), np.sin(angle), 0.0]], (2, 1))
    beta = np.tile([[-np.sin(angle), np.cos(angle), 0.0]], (2, 1))
    return geometry.GeometryInitializationTarget(
        tmp_path / "source.npz", tmp_path / "source.npz", tmp_path / "mesh.obj",
        "0" * 64, "1" * 64, 2, 4, np.concatenate((alpha, beta), axis=1).astype(np.float64),
        np.ones(2, dtype=np.float64), np.ones(2, dtype=np.bool_),
        np.ones(2, dtype=np.complex128),
        np.ones(2, dtype=np.complex128))


def test_prefit_changes_only_angle_and_restores_rng_and_flags(tmp_path):
    random.seed(19)
    np.random.seed(23)
    torch.manual_seed(29)
    net = _ToyNetwork()
    net.train()
    net.decoder.eval()
    decoder_before = {name: value.detach().clone() for name, value in net.decoder.named_parameters()}
    angle_before = [value.detach().clone() for value in net.decoder_angle.parameters()]
    flags_before = [module.training for module in net.modules()]
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    report = geometry.prefit_geometry_angle_decoder(
        net, _toy_target(tmp_path), _ToyDataset(), "cpu",
        learning_rate=1e-2, max_steps=4, target_degrees=0.0)

    assert report["completed_updates"] == 4
    assert report["siren_and_non_angle_parameters_unchanged"] is True
    assert report["elapsed_seconds"] > 0.0
    assert all(torch.equal(value, decoder_before[name])
               for name, value in net.decoder.named_parameters())
    assert any(not torch.equal(after, before)
               for after, before in zip(net.decoder_angle.parameters(), angle_before))
    assert [module.training for module in net.modules()] == flags_before
    assert random.getstate() == python_before
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_before[0] and np.array_equal(after_numpy[1], numpy_before[1])
    assert after_numpy[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_checkpoint_is_atomic_and_no_replace(tmp_path):
    field = np.array([[1, 0, 0, 0, 1, 0], [0, 1, 0, -1, 0, 0]], dtype=np.float32)
    field_path, report_path, report = geometry.write_research_checkpoint(
        tmp_path, 1500, field, {"probe": {"score": 1.25}})
    assert field_path.is_file() and report_path.is_file()
    assert report["cross_field"]["rows"] == 2
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["artifact_commit"]["status"] == "complete"
    with pytest.raises(FileExistsError):
        geometry.write_research_checkpoint(tmp_path, 1500, field, {})


def test_research_field_provenance_and_default_terminal_inertness():
    mesh_sha256 = "a" * 64
    checkpoint = geometry.build_research_field_provenance(
        seed=3627473, mesh_sha256=mesh_sha256, grid_res=2,
        angle_init="random", field_state="step_1500")
    assert checkpoint == {
        "seed": 3627473,
        "mesh_sha256": mesh_sha256,
        "grid_res": 2,
        "angle_init": "random",
        "field_state": "step_1500",
    }
    assert "geometry_initialization" not in checkpoint

    baseline = {"schema_version": 1, "completed_updates": 10}
    unchanged = dict(baseline)
    result = geometry.attach_terminal_research_provenance(
        unchanged, enabled=False, seed=3627473,
        mesh_sha256=mesh_sha256, grid_res=2, angle_init="random",
        termination="automatic")
    assert result is unchanged
    assert unchanged == baseline
    assert "research_provenance" not in unchanged

    automatic_report = {}
    geometry.attach_terminal_research_provenance(
        automatic_report, enabled=True, seed=3627473,
        mesh_sha256=mesh_sha256, grid_res=2, angle_init="random",
        termination="automatic")
    automatic = automatic_report["research_provenance"]
    assert automatic["field_state"] == "automatic_terminal"
    assert automatic["termination"] == "automatic"
    assert automatic["automatic_terminal"] is True
    assert "geometry_initialization" not in automatic

    fixed_report = {}
    geometry.attach_terminal_research_provenance(
        fixed_report, enabled=True, seed=3627473,
        mesh_sha256=mesh_sha256, grid_res=2, angle_init="random",
        termination="fixed_steps")
    fixed = fixed_report["research_provenance"]
    assert fixed["field_state"] == "fixed_steps_terminal"
    assert fixed["automatic_terminal"] is False
    assert fixed["field_state"] != "automatic_terminal"

    evidence = {"schema_version": "prefit.v1", "completed_updates": 7}
    geometry_checkpoint = geometry.build_research_field_provenance(
        seed=3627473, mesh_sha256=mesh_sha256, grid_res=2,
        angle_init="geometry", field_state="step_1500",
        geometry_initialization=evidence)
    assert geometry_checkpoint["geometry_initialization"] == evidence
    assert geometry_checkpoint["geometry_initialization"] is not evidence
    with pytest.raises(ValueError, match="requires initialization evidence"):
        geometry.build_research_field_provenance(
            seed=3627473, mesh_sha256=mesh_sha256, grid_res=2,
            angle_init="geometry", field_state="step_1500")
    with pytest.raises(ValueError, match="cannot carry geometry"):
        geometry.build_research_field_provenance(
            seed=3627473, mesh_sha256=mesh_sha256, grid_res=2,
            angle_init="random", field_state="step_1500",
            geometry_initialization=evidence)
    with pytest.raises(ValueError, match="does not match termination"):
        geometry.build_research_field_provenance(
            seed=3627473, mesh_sha256=mesh_sha256, grid_res=2,
            angle_init="random", field_state="automatic_terminal",
            termination="fixed_steps")

    source = (QUAD_MESH / "train_quad_mesh.py").read_text(encoding="utf-8")
    assert "field_state='step_{}'.format(completed_steps)" in source
    assert "attach_terminal_research_provenance(" in source
    assert "research_opt_in = bool(" in source


def test_convergence_and_terminal_atomic_regression(tmp_path):
    config = convergence.AutoStopConfig(
        min_steps=3, check_interval=1, loss_relative_tolerance=0.10)
    stopper = convergence.ConvergenceStopper(
        config, face_areas=np.array([1.0, 2.0]), auto_enabled=True)
    field = np.array([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
    ], dtype=np.float32)
    decisions = [
        stopper.observe(step, field, score)
        for step, score in ((1, 1.00), (2, 0.98), (3, 1.01))
    ]
    assert [row["eligible"] for row in decisions] == [False, False, True]
    assert decisions[-1]["passes"] is True
    report = stopper.report()
    assert len(report["decisions"]) == 3
    assert report["auto_disabled_reason"] is None

    field_path, report_path, written = convergence.write_final_artifacts(
        tmp_path, field, {"completed_updates": 3})
    expected_sha256 = hashlib.sha256(field_path.read_bytes()).hexdigest()
    assert field_path.is_file() and report_path.is_file()
    assert written["terminal_cross_field"]["rows"] == 2
    assert written["terminal_cross_field"]["sha256"] == expected_sha256.upper()
    assert json.loads(report_path.read_text(encoding="utf-8"))[
        "terminal_cross_field"]["sha256"] == expected_sha256.upper()
    with pytest.raises(FileExistsError):
        convergence.write_final_artifacts(tmp_path, field, {})

    research_logdir = tmp_path / "research"
    research_logdir.mkdir()
    research_field_path, research_report_path, research_written = (
        convergence.write_final_artifacts(
            research_logdir, field, {"completed_updates": 3},
            lowercase_research_sha256=True))
    research_sha256 = hashlib.sha256(
        research_field_path.read_bytes()).hexdigest()
    assert research_written[
        "terminal_cross_field"]["sha256"] == research_sha256
    assert json.loads(research_report_path.read_text(encoding="utf-8"))[
        "terminal_cross_field"]["sha256"] == research_sha256


def test_structured_trace_is_explicit_and_no_replace(tmp_path):
    disabled = geometry.StructuredResearchTrace(None)
    disabled.record("ignored")
    registered = []
    original_register = geometry.atexit.register
    path = tmp_path / "trace.jsonl"
    trace = None
    try:
        geometry.atexit.register = lambda callback: registered.append(callback)
        trace = geometry.StructuredResearchTrace(path, {"seed": 7})
        trace.record("step", step=1, loss=np.float32(2.0))
        assert len(registered) == 1
        registered[0]()
        assert trace._stream.closed
    finally:
        geometry.atexit.register = original_register
        if trace is not None:
            trace.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["trace_start", "step"]
    with pytest.raises(FileExistsError):
        geometry.StructuredResearchTrace(path)

    with pytest.raises(ValueError, match="training log"):
        geometry.validate_research_output_destinations(
            tmp_path, [], tmp_path / "out.log")
    with pytest.raises(ValueError, match="reserved artifact directory"):
        geometry.validate_research_output_destinations(
            tmp_path, [], tmp_path / "convergence_stop" / "trace.jsonl")

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "research_checkpoints").write_text("not a directory", encoding="utf-8")
    # Inert defaults must not inspect an unused research destination.
    geometry.validate_research_output_destinations(blocked, [], None)
    with pytest.raises(NotADirectoryError, match="checkpoint parent"):
        geometry.validate_research_output_destinations(blocked, [1500], None)

    linked = tmp_path / "linked"
    linked.mkdir()
    linked_target = tmp_path / "linked-target"
    linked_target.mkdir()
    os.symlink(str(linked_target), str(linked / "research_checkpoints"))
    with pytest.raises(NotADirectoryError, match="real directory"):
        geometry.validate_research_output_destinations(linked, [1500], None)

    lexical = tmp_path / "lexical"
    lexical.mkdir()
    broken_trace = lexical / "broken_trace.jsonl"
    os.symlink(str(lexical / "missing-target"), str(broken_trace))
    with pytest.raises(FileExistsError, match="already exists"):
        geometry.validate_research_output_destinations(
            lexical, [], broken_trace)



def test_entrypoint_validates_before_side_effects_and_prefits_before_main_adam():
    source = (QUAD_MESH / "train_quad_mesh.py").read_text(encoding="utf-8")
    validation = source.index("research_checkpoint_steps = validate_research_contract")
    target_load = source.index("geometry_initialization = load_geometry_initialization")
    input_hash = source.index("research_input_sha256 =")
    mkdir = source.index("os.makedirs(logdir")
    dataloader = source.index("torch.utils.data.DataLoader")
    gpu_move = source.index("net.to(device)")
    prefit = source.index("geometry_prefit_report = prefit_geometry_angle_decoder")
    main_adam = source.index("optimizer = optim.Adam(net.parameters()")
    publication = source.index(
        "publication_paths = build_research_publication_paths")
    optimizer_step = source.index("optimizer.step()")
    completed_increment = source.index("completed_steps += 1")
    schedule_update = source.index("criterion.update_morse_weight")
    probe_due = source.index("automatic_probe_due =")
    assert validation < target_load < input_hash < mkdir < dataloader < gpu_move
    assert prefit < main_adam
    assert source.count("optimizer.step()") == 1
    assert validation < publication < target_load
    assert optimizer_step < completed_increment < schedule_update < probe_due


def test_publication_reference_wiring_is_exact_and_research_only():
    source = (QUAD_MESH / "train_quad_mesh.py").read_text(encoding="utf-8")
    assert source.count("publication_paths.publication_reference(") == 7
    assert "publication_paths.validate_trace_path(research_trace_path)" in source
    assert "or publication_paths.enabled)" in source
    assert "lowercase_research_sha256=research_opt_in" in source
    assert source.index("publication_paths = build_research_publication_paths") < \
        source.index("validate_final_artifact_destination(logdir)")


def test_probe_exception_boundary_and_terminal_fallback_provenance():
    source = (QUAD_MESH / "train_quad_mesh.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    def call_names(nodes):
        names = set()
        for root in nodes:
            for node in ast.walk(root):
                if isinstance(node, ast.Call):
                    function = node.func
                    if isinstance(function, ast.Name):
                        names.add(function.id)
                    elif isinstance(function, ast.Attribute):
                        names.add(function.attr)
        return names

    probe_tries = [
        node for node in ast.walk(tree)
        if (isinstance(node, ast.Try)
            and {"run_fixed_probe", "observe"}.issubset(call_names(node.body)))
    ]
    assert len(probe_tries) == 1
    assert "run_fixed_probe" in call_names(probe_tries[0].body)
    io_calls = {"record", "write_research_checkpoint", "log_string"}
    assert call_names(probe_tries[0].body).isdisjoint(io_calls)
    handler_source = " ".join(
        ast.get_source_segment(source, handler) or ""
        for handler in probe_tries[0].handlers)
    assert "if checkpoint_due:" in handler_source and "raise RuntimeError" in handler_source
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            assert "write_research_checkpoint" not in call_names(node.body)
            assert "record" not in call_names(node.body)

    compact = " ".join(source.split())
    assert ("terminal_observation['probe'][ "
            "'terminal_direction_field_fallback'] = field_metrics") in compact
    assert "'checkpoint_artifact_participates': False" in source
    assert "'coincident_fixed_probe_observation_participates'" in source


def test_research_timing_covers_initialization_and_each_automatic_probe():
    source = (QUAD_MESH / "train_quad_mesh.py").read_text(encoding="utf-8")
    argument_parse = source.index("args = quad_mesh_args.get_args()")
    timing_origin = source.index(
        "end_to_end_started_monotonic = time.monotonic()")
    target_load = source.index(
        "geometry_initialization = load_geometry_initialization")
    prefit = source.index(
        "geometry_prefit_report = prefit_geometry_angle_decoder")
    assert argument_parse < timing_origin < target_load < prefit
    automatic_event = source.index("'automatic_probe', step=completed_steps")
    automatic_elapsed = source.index(
        "elapsed_seconds=time.monotonic() - end_to_end_started_monotonic",
        automatic_event)
    assert automatic_elapsed > automatic_event
    assert "terminal_report['end_to_end_elapsed_seconds']" in source
    assert "'end_to_end_elapsed_seconds_at_probe'" in source


class GeometryInitializationTests(unittest.TestCase):
    def _with_tmp_path(self, callback):
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as directory:
            callback(Path(directory))

    def test_cli(self):
        test_cli_defaults_are_inert_and_list_is_accepted()

    def test_default_model_contract(self):
        test_default_model_state_key_and_shape_contract()

    def test_contract(self):
        test_research_contract_rejects_conflicts_and_unreachable_steps()

    def test_publication_paths(self):
        self._with_tmp_path(test_publication_paths_are_lexical_inert_and_nonoverlapping)

    def test_npz_and_companion(self):
        self._with_tmp_path(test_geometry_npz_and_companion_bind_exact_mesh)

    def test_target_rejections(self):
        self._with_tmp_path(test_geometry_target_rejects_sha_and_face_order)

    def test_strict_bundle_rejections(self):
        self._with_tmp_path(test_geometry_target_requires_complete_strict_bundle)

    def test_prefit_transaction(self):
        self._with_tmp_path(test_prefit_changes_only_angle_and_restores_rng_and_flags)

    def test_checkpoint(self):
        self._with_tmp_path(test_checkpoint_is_atomic_and_no_replace)

    def test_convergence_atomic(self):
        self._with_tmp_path(test_convergence_and_terminal_atomic_regression)

    def test_trace(self):
        self._with_tmp_path(test_structured_trace_is_explicit_and_no_replace)

    def test_entrypoint_ordering(self):
        test_entrypoint_validates_before_side_effects_and_prefits_before_main_adam()

    def test_publication_wiring(self):
        test_publication_reference_wiring_is_exact_and_research_only()

    def test_research_provenance(self):
        test_research_field_provenance_and_default_terminal_inertness()

    def test_probe_boundary(self):
        test_probe_exception_boundary_and_terminal_fallback_provenance()

    def test_research_timing(self):
        test_research_timing_covers_initialization_and_each_automatic_probe()


if __name__ == "__main__":
    unittest.main()
