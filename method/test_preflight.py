import numpy as np
import pytest
from types import SimpleNamespace
from constraint_preflight import precheck
from resource_budget import limit_density,redistribute_density


def example():
    # A target triangle plus four constraint faces; no target edge is constrained.
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[2,1,0],[1,2,0],
                [3,0,0],[3,2,0],[2,3,0],[0,3,0]],float)
    f=np.array([[0,1,2],[0,3,5],[3,1,6],[1,4,7],[4,2,8]])
    x=np.tile([1.,0,0],(len(f),1));y=np.tile([0.,1,0],(len(f),1))
    for i in range(1,len(f)):
        t=v[f[i,1]]-v[f[i,0]];t/=np.linalg.norm(t)
        y[i]=t;x[i]=np.cross(t,[0,0,1])
    hard=[[0,3],[3,1],[1,4],[4,2]]
    return SimpleNamespace(vertices=v,faces=f),v[:,:2].copy(),f.copy(),x,y,hard


def test_transitive_collapse_and_independent_equality_solution():
    args=example();r=precheck(*args)
    assert 0 in r['transitive_extra_triangles'] and r['direct_local_count']==0
    # Independently project random U values using a constraint matrix nullspace.
    from scipy.linalg import null_space
    mat=np.zeros((4,len(args[1])))
    for row,(a,b) in zip(mat,args[-1]):row[a]=1;row[b]=-1
    z=null_space(mat)@np.arange(null_space(mat).shape[1])
    assert np.ptp(z[[0,1,2]])<1e-12


def test_cut_separates_transitive_chain():
    mesh,uv,fuv,x,y,hard=example()
    # Separate every face corner, as independent cut vertices.
    cut=np.arange(fuv.size).reshape(fuv.shape)
    r=precheck(mesh,uv[fuv].reshape(-1,2),cut,x,y,hard)
    assert r['forced_zero_triangle_count']==0


def test_frame_swap_and_rigid_motion_do_not_change_collapse():
    mesh,uv,fuv,x,y,hard=example();a=precheck(mesh,uv,fuv,x,y,hard)
    b=precheck(mesh,uv[:,::-1],fuv,y,x,hard)
    rot=np.array([[0,0,1],[1,0,0],[0,1,0.]])
    moved=SimpleNamespace(vertices=mesh.vertices@rot+3,faces=mesh.faces)
    c=precheck(moved,uv,fuv,x@rot,y@rot,hard)
    assert a['forced_zero_triangles']==b['forced_zero_triangles']==c['forced_zero_triangles']
    assert precheck(mesh,uv,fuv,x,y,[])['forced_zero_triangle_count']==0


def test_weighted_budget_is_maximal_and_preserves_order():
    rho=np.array([1.,2.,4.]);areas=np.array([10.,2.,1.])
    result,r=limit_density(rho,areas,2.)
    assert areas@(result**2)/areas.sum()==pytest.approx(2.)
    assert np.all(result>=1) and np.all(result<=rho) and np.all(np.diff(result)>0)
    alpha=r['retained_size_relief_fraction']
    larger=1/(1-(alpha+1e-6)+(alpha+1e-6)/rho)
    assert areas@(larger**2)/areas.sum()>2.
    assert np.array_equal(limit_density(rho,areas,100)[0],rho)
    assert np.allclose(limit_density(rho,areas,1)[0],1)


def test_interpolation_commutes_with_face_min_and_reduces_gradation():
    hv=np.array([.25,.45,.7,1.]);faces=np.array([[0,1,2],[1,2,3]])
    rho=1/hv[faces].min(1)
    result,r=limit_density(rho,np.ones(2),2.)
    alpha=r['retained_size_relief_fraction'];limited=1-alpha+alpha*hv
    assert np.allclose(result,1/limited[faces].min(1))
    assert np.allclose(np.diff(limited),alpha*np.diff(hv))


@pytest.mark.parametrize('rho,areas,cap',[([.5],[1],2),([2],[0],2),([2],[1],.5),([2],[1],np.inf)])
def test_invalid_budget_inputs(rho,areas,cap):
    with pytest.raises(ValueError):limit_density(rho,areas,cap)


def test_fixed_budget_redistributes_and_preserves_gradation():
    full=np.array([1.,2.,4.]);areas=np.array([1.,1.,1.])
    result,r=redistribute_density(full,areas,2.)
    assert areas@(result*result)/areas.sum()==pytest.approx(2.)
    assert result[0]<1 and result[-1]>np.sqrt(2.)
    assert r['size_gradation_bound']<=.5+1e-12
    # Directly verify the implied vertex-size affine transform.
    h=1/full;allocated=1/result
    assert np.allclose(np.diff(allocated),np.diff(h)*r['contrast_fraction']/r['global_density_scale'])
    assert r['coarsened_area_fraction']==pytest.approx(1/3)


def test_uniform_demand_has_no_fictitious_redistribution_advantage():
    for demand in [1.,2.5,4.]:
        result,r=redistribute_density(np.full(7,demand),np.arange(1,8),2.)
        assert np.allclose(result,np.sqrt(2.))
        assert r['density_min']==pytest.approx(r['density_max'])

