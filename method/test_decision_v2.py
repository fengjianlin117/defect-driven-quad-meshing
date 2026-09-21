from types import SimpleNamespace
import itertools
import numpy as np
import pytest
from conflict_selection import EqualityAudit,select
from constraint_preflight import precheck
from protected_decision import decide_protected

def fixture():
    mesh=SimpleNamespace(vertices=np.array([[0.,0,0],[1.,.1,0],[2.,0,0],[3.,.1,0]]),faces=np.array([[0,1,2],[0,2,3]]),face_count=2,vertex_count=4)
    uv=mesh.vertices[:,:2].copy();fuv=mesh.faces.copy();x=np.tile([1.,0,0],(2,1));y=np.tile([0.,1,0],(2,1))
    return mesh,uv,fuv,x,y

def test_transitive_triangle_witness_and_order_invariance():
    args=fixture();audit=EqualityAudit(*args);hard=[(0,1),(1,2),(2,3)]
    r=audit.check(hard);assert r['forced_zero_triangles']==[0,1]
    assert r==audit.check(hard[::-1])
    old=precheck(*args,hard);assert old['transitive_extra_triangles']==[1]

def test_compiled_audit_matches_original_all_subsets():
    args=fixture();audit=EqualityAudit(*args);edges=list(audit.links)
    for flags in itertools.product([False,True],repeat=len(edges)):
        hard=[e for e,keep in zip(edges,flags) if keep]
        assert audit.check(hard)['forced_zero_triangles']==precheck(*args,hard)['forced_zero_triangles']

def test_seam_copies_are_not_silently_welded():
    mesh,uv,fuv,x,y=fixture();seamed=np.array([[0,1,2],[3,4,5]])
    audit=EqualityAudit(mesh,mesh.vertices[mesh.faces].reshape(-1,3)[:,:2],seamed,x,y)
    assert audit.check([(0,1),(1,2),(2,3)])['forced_zero_triangles']==[0]

def test_unknown_source_edge_rejected():
    with pytest.raises(ValueError):EqualityAudit(*fixture()).check([(1,3)])

class ToyContext:
    def __init__(self):
        self.mesh=fixture()[0];self.group_edges={'a':{(0,1)},'b':{(1,2)},'c':{(2,3)}};self.groups={k:{} for k in self.group_edges}
        self.all_edges=set().union(*self.group_edges.values());self.length={e:1. for e in self.all_edges};self.loss={e:1. for e in self.all_edges};self.field_risk={e:0. for e in self.all_edges};self.total_length=3.
    def residual_profile(self,selected):return tuple(sorted([float(bool(es-selected)) for es in self.group_edges.values()],reverse=True))
    def size(self,chosen):return None,dict(resolution_pressure=dict(adapted_max=0.,adapted_mean=0.),predicted_area_budget_ratio=1.)

def test_screening_preserves_reference_and_uses_other_candidate():
    ctx=ToyContext();r=select(ctx,EqualityAudit(*fixture()),1.)
    assert r['selected_group_ids']==['a','c']
    assert r['rejected_hard_groups'][0]['group_id']=='b'
    assert ctx.all_edges=={(0,1),(1,2),(2,3)} and r['screening']['admissible']

def test_empty_and_all_feature_controls_explicit():
    ctx=ToyContext();audit=EqualityAudit(*fixture())
    assert select(ctx,audit,.5,'no_features')['selected_edges']==[]
    assert select(ctx,audit,.5,'all_features')['budget_fraction']==1.
    assert not select(ctx,audit,.5,'all_features')['screening']['admissible']

def row(name,loss=10.,rms=.1,protected=3.,valid=True,n=500):
    return dict(id=name,all_main_deficit=loss,symmetric_rms_h=rms,protected_deficit=protected,process_success=True,basic_output_pass=valid,quads=n)

def test_surface_regression_is_tradeoff_not_default():
    r=decide_protected([row('baseline'),row('feature_gain',loss=5.,rms=.2)],2048)
    assert r['recommended_id']=='baseline' and r['unrestricted_feature_priority_id']=='feature_gain'
    assert 'feature_gain' in r['tradeoff_ids']

def test_unselected_regression_is_recorded_even_if_rms_improves():
    r=decide_protected([row('baseline'),row('bad',loss=5.,rms=.05,protected=4.)],2048)
    assert r['recommended_id']=='baseline'
    assert 'baseline_regression_protected_deficit' in r['audit'][1]['reasons']

def test_safe_improvement_selected_and_cap_still_holds():
    r=decide_protected([row('baseline'),row('safe',loss=7.,rms=.08,protected=2.),row('over',loss=1.,rms=.02,protected=1.,n=2049)],2048)
    assert r['recommended_id']=='safe'

def test_invalid_baseline_no_silent_acceptance():
    assert decide_protected([row('baseline',valid=False),row('worse',rms=.2)],2048)['recommended_id'] is None
    with pytest.raises(ValueError):decide_protected([row('candidate')],2048)

def test_missing_protected_metric_not_silently_zero():
    bad=row('bad',loss=1.);bad.pop('protected_deficit')
    assert decide_protected([row('baseline'),bad],2048)['recommended_id']=='baseline'
