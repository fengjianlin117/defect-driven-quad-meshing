import copy
import pytest
from test_decision_v2 import ToyContext, fixture
from conflict_selection import EqualityAudit
from defect_driven_selection import select_defect_driven, needs_repair


class Admissible:
    def check(self, edges):
        return dict(admissible=True, forced_zero_triangles=[], forced_zero_triangle_count=0)


def context():
    c = ToyContext()
    c.records = {e: dict(unaligned_fraction=1., surface_distance_p95_h=.2) for e in c.all_edges}
    return c


def test_every_needed_compatible_group_can_be_selected():
    r = select_defect_driven(context(), Admissible())
    assert len(r['selected_group_ids']) == 3
    assert r['observed_selected_length_fraction'] == 1
    assert r['budget_fraction'] is None and not r['unaddressed_deficient_edges']


def test_small_continuous_residual_does_not_force_good_features():
    c = context()
    c.records = {e: dict(unaligned_fraction=0., surface_distance_p95_h=.01) for e in c.all_edges}
    # The continuous ranking score can remain positive inside tolerance.
    assert all(c.loss.values())
    r = select_defect_driven(c, Admissible())
    assert not r['selected_edges']
    assert all(g['status'] == 'no_diagnosed_deficiency' for g in r['group_diagnostics'])


def test_only_diagnosed_group_is_needed_no_fill_to_quota():
    c = context()
    for e in [(1,2), (2,3)]:
        c.records[e] = dict(unaligned_fraction=0., surface_distance_p95_h=.01)
    assert select_defect_driven(c, Admissible())['selected_group_ids'] == ['a']


def test_conflict_is_deferred_without_erasing_reference_or_demand():
    c = context(); original = copy.deepcopy(c.__dict__)
    r = select_defect_driven(c, EqualityAudit(*fixture()))
    assert r['selected_group_ids'] == ['a','c']
    assert r['unaddressed_deficient_edges'] == [[1,2]]
    assert next(g for g in r['group_diagnostics'] if g['group_id']=='b')['status']=='deferred_combination_conflict'
    assert c.group_edges == original['group_edges'] and c.records == original['records']
    assert r['screening']['admissible']


def test_shared_deficient_edges_do_not_add_redundant_group():
    c = context(); c.group_edges['duplicate'] = {(0,1)}
    r = select_defect_driven(c, Admissible())
    assert next(g for g in r['group_diagnostics'] if g['group_id']=='duplicate')['status']=='covered_by_other_selected_groups'


def test_whole_group_retains_good_segment_next_to_bad_segment():
    c = context(); c.group_edges = {'chain': {(0,1),(1,2)}}; c.all_edges = {(0,1),(1,2)}
    c.records[(1,2)] = dict(unaligned_fraction=0., surface_distance_p95_h=0.)
    r = select_defect_driven(c, Admissible())
    assert r['selected_edges'] == [[0,1],[1,2]]


def test_empty_graph_has_no_artificial_requirement():
    c = context(); c.all_edges=set(); c.group_edges={}; c.total_length=0.
    assert select_defect_driven(c, Admissible())['selected_edges']==[]


def test_invalid_diagnostic_is_not_mistaken_for_well_represented():
    for r in [dict(unaligned_fraction=float('nan'),surface_distance_p95_h=0.),
              dict(unaligned_fraction=0.,surface_distance_p95_h=-1.)]:
        with pytest.raises(ValueError): needs_repair(r)
