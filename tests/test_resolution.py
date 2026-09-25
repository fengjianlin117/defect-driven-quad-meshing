import math
from pathlib import Path
import numpy as np
import pytest
from resolution_budget import estimate_resolution, count_in_band
from coordinator_v5 import strict_decision
from coordinator import read, save
from weak_layout_pipeline.pipeline.mesh import Mesh
import prepare_resolution as preparation
from protection_feedback import size_repair
from coordination import size_operation


def test_large_sources_no_longer_start_at_512_and_small_sources_stay_moderate():
    assert estimate_resolution(5248, 481)['target_quads'] == 2624
    assert estimate_resolution(4640, 513)['target_quads'] == 2320
    assert estimate_resolution(1024, 498)['target_quads'] == 512
    assert estimate_resolution(1024, 900)['target_quads'] == 900
    assert estimate_resolution(5000, 500, 1800)['target_quads'] == 1800
    with pytest.raises(ValueError, match='exceeds cap'):
        estimate_resolution(20000, 500)
    for bad in [0, -1, True, 2500.5]:
        with pytest.raises(ValueError):
            estimate_resolution(5000, 500, bad)


def test_count_band_includes_actual_cap_and_floor():
    policy = estimate_resolution(5000, 500)
    assert not count_in_band(500, policy)
    assert count_in_band(2250, policy)
    assert count_in_band(2750, policy)
    assert not count_in_band(2751, policy)
    policy = estimate_resolution(5000, 500, 8192)
    assert not count_in_band(8193, policy)


def test_strict_recommendation_cannot_choose_coarse_but_better_scoring_output():
    base = dict(id='baseline', quads=2500, process_success=True,
                basic_output_pass=True, strict_output_pass=True,
                all_main_deficit=2., symmetric_rms_h=.01, protected_deficit=1.)
    coarse = dict(base, id='coarse', quads=500, all_main_deficit=0.)
    rows = [base, coarse]
    assert strict_decision(rows, 8192)['recommended_id'] == 'coarse'
    decision = strict_decision(rows, 8192, 2250)
    assert decision['recommended_id'] == 'baseline'
    assert 'actual_count_below_resolution_floor' in decision['audit'][1]['reasons']


@pytest.mark.parametrize('protected', [False, True])
def test_local_size_feedback_respects_new_cap_above_2048(protected):
    mesh = Mesh(np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]]), np.array([[0, 1, 2]]))
    graph = dict(groups=[])
    edge = dict(vertices=[0, 1], layer='main', length_h=1., surface_loss=.1,
                unaligned_fraction=1., surface_distance_p95_h=.2)
    baseline = dict(quads=3000, edge_defects=[edge])
    parent = dict(edge_defects=[dict(edge, surface_loss=.9)])
    args = [mesh, graph, 1., baseline, parent, np.ones(1), 3000, 2.]
    result = (size_repair(*args, set(), max_quads=8192) if protected
              else size_operation(*args, max_quads=8192))
    assert result is not None and 3000 <= result['target'] <= 8192


def fixture_preparation(tmp_path, monkeypatch, counts):
    inputs = tmp_path/'old'
    inputs.mkdir()
    plan = dict(model='synthetic', gsize=10., reference_h=.1)
    for key in ['source', 'pd1', 'pd2', 'graph', 'baseline_record']:
        path = inputs/key
        path.write_text('{}')
        plan[key] = str(path)
    save(plan['baseline_record'], dict(quads=500))
    mesh = Mesh(np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]]), np.array([[0, 1, 2]]))
    monkeypatch.setattr(preparation, 'load_obj', lambda *a, **k: mesh)
    monkeypatch.setattr(preparation, 'geometry_check', lambda *a: dict(clear=True))
    calls = []
    def backend(**kwargs):
        folder = Path(kwargs['output_dir'])
        folder.mkdir()
        (folder/'quad.obj').write_text('mesh')
        calls.append(kwargs)
        return dict(success=True)
    values = iter(counts)
    monkeypatch.setattr(preparation, 'run_backend', backend)
    monkeypatch.setattr(preparation, 'evaluate', lambda *a: dict(quads=next(values), basic_output_pass=True))
    return plan, calls


def test_baseline_count_correction_remeasures_and_rebases_before_method(tmp_path, monkeypatch):
    plan, calls = fixture_preparation(tmp_path, monkeypatch, [1000, 2480])
    policy = estimate_resolution(5000, 500)
    updated, summary = preparation.prepare_resolution(plan, tmp_path/'new', policy,
                                                       plan['pd1'], plan['pd2'], plan['source'])
    assert len(calls) == 2
    assert calls[1]['gsize']/calls[0]['gsize'] == pytest.approx(math.sqrt(2.5))
    assert updated['reference_h'] == pytest.approx(math.sqrt(2)/calls[1]['gsize'])
    assert updated['min_quads'] == 2250 and updated['max_quads'] == 8192
    assert read(updated['baseline_record'])['quads'] == 2480
    assert (Path(updated['source']).parent/'baseline/quad.obj').is_file()
    assert summary['success'] and summary['new_backend_calls'] == 2
    assert read(plan['baseline_record'])['quads'] == 500


def test_persistently_undersized_output_stops_instead_of_coarse_fallback(tmp_path, monkeypatch):
    plan, calls = fixture_preparation(tmp_path, monkeypatch, [500, 500, 500])
    with pytest.raises(RuntimeError, match='No coarse fallback'):
        preparation.prepare_resolution(plan, tmp_path/'new', estimate_resolution(5000, 500),
                                       plan['pd1'], plan['pd2'], plan['source'])
    summary = read(tmp_path/'new/summary.json')
    assert len(calls) == 3 and summary['success'] is False
    assert not (tmp_path/'new/inputs').exists()
