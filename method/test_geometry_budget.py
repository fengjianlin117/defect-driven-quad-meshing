import numpy as np
import pytest
from weak_layout_pipeline.pipeline.mesh import Mesh
from geometry_budget import propose,requests


def strip(scale=1.):
    mesh=Mesh(scale*np.array([[0.,0,0],[2,0,0],[2,.1,0],[0,.1,0]]),np.array([[0,1,2],[0,2,3]]))
    graph=dict(groups=[dict(id='lower',layer='main',mesh_vertices=[0,1]),dict(id='upper',layer='main',mesh_vertices=[3,2])])
    return mesh,graph


def test_unselected_parallel_features_both_generate_demands_and_count():
    mesh,graph=strip();rho,hv,p,req=propose(mesh,graph,.1,128)
    assert p['request_diagnostics']['all_main_group_ids']==['lower','upper']
    assert len(p['request_diagnostics']['close_pairs'])==1
    assert np.allclose(req['width_caps'],.05)
    assert p['target_quads']==512 and np.allclose(rho,2.)
    assert p['allocation']['unmet_request_vertices']==0


def test_resource_cap_keeps_requests_and_reports_shortfall():
    mesh,graph=strip();rho,hv,p,req=propose(mesh,graph,.1,128,dict(max_quads=256))
    assert p['target_quads']==256 and p['resource_cap_active']
    assert p['uncapped_target_quads']==512 and np.allclose(rho,np.sqrt(2.))
    assert p['allocation']['unmet_request_vertices']==4
    assert np.allclose(req['requested_vertex_sizes'],.05)


def test_rigid_transform_and_units_do_not_change_budget():
    mesh,graph=strip();a=propose(mesh,graph,.1,128)
    rot=np.array([[0,0,1],[1,0,0],[0,1,0.]])
    moved=Mesh(13*mesh.vertices@rot+7,mesh.faces)
    b=propose(moved,graph,1.3,128)
    assert np.allclose(a[0],b[0]) and a[2]['target_quads']==b[2]['target_quads']
    assert np.allclose(a[1]*13,b[1])


def test_flat_empty_graph_does_not_invent_refinement():
    mesh,_=strip();rho,hv,p,req=propose(mesh,dict(groups=[]),.1,128)
    assert p['target_quads']==128 and np.allclose(rho,1.)
    assert not p['request_diagnostics']['close_pairs']


def test_smooth_curvature_requests_exist_without_feature_selection():
    # Closed octagonal prism: side dihedral 45 degrees, admitted by an explicit
    # smooth threshold of 60 for this analytic coarse-cylinder fixture.
    n=8;t=np.arange(n)*2*np.pi/n
    v=np.array([[np.cos(a),np.sin(a),z] for z in [0.,2.] for a in t]+[[0,0,0],[0,0,2]])
    f=[]
    for i in range(n):
        j=(i+1)%n;f.extend([[i,j,n+j],[i,n+j,n+i],[2*n,j,i],[2*n+1,n+i,n+j]])
    mesh=Mesh(v,np.array(f));graph=dict(groups=[])
    r=requests(mesh,graph,.5,dict(smooth_dihedral_degrees=60.))
    assert r['diagnostics']['curvature_request_vertices']>0
    assert r['diagnostics']['width_request_vertices']==0
    assert r['diagnostics']['maximum_graph_slope']<=.5+1e-12
    rho,hv,p,_=propose(mesh,graph,.5,128,dict(smooth_dihedral_degrees=60.))
    e=r['edges'];assert np.max(abs(hv[e[:,0]]-hv[e[:,1]])/r['edge_lengths'])<=.5+1e-12
    assert p['allocation']['actual_area_proxy']==pytest.approx(p['target_quads']/128)


def test_floor_is_visible_instead_of_changing_raw_requirement():
    mesh,graph=strip();_,_,p,r=propose(mesh,graph,1.,128)
    assert p['request_diagnostics']['floor_limited_vertices']==4
    assert p['unclipped_geometric_count_estimate']>p['floored_geometric_count_estimate']
    assert np.allclose(r['requested_vertex_sizes'],.05)
