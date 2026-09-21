"""Source-wide geometric size demand and an explicit moderate-count budget.

No selected-feature set or constrained output is an input. Geometry proxies
do not certify approximation error or remove field/integer compatibility risk.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from weak_layout_pipeline.pipeline.mesh import unique_edges,triangle_geometry,edge_topology
from defect_selection_v2 import graded_envelope

CONFIG=dict(strip_rows=2.,gradation_slope=.5,minimum_h_fraction=.25,maximum_h_fraction=2.,
            sagitta_h_fraction=.05,smooth_dihedral_degrees=30.,max_quads=2048,
            budget_policy='Estimate count from source-wide graded demand, retain at least the original baseline count, cap explicitly; never delete features to hide unmet demand.',
            limitations=['Graph distances can overestimate geodesic separation.',
                         'Touching groups and within-group narrow parts require further treatment.',
                         'Normal-change curvature and sagitta are geometric proxies, not surface-error bounds.',
                         'Count is calibrated from one baseline, not a guarantee of extracted count.',
                         'Field and integer compatibility remain separate risks.'])


def requests(mesh,graph,h,config=None):
    c=dict(CONFIG,**(config or {}));h=float(h)
    if not np.isfinite(h) or h<=0:raise ValueError('Positive finite reference h required')
    if not (0<c['minimum_h_fraction']<=1<=c['maximum_h_fraction'] and c['strip_rows']>0 and c['gradation_slope']>=0 and c['sagitta_h_fraction']>0):
        raise ValueError('Invalid size policy')
    edges=unique_edges(mesh);lengths=np.linalg.norm(mesh.vertices[edges[:,0]]-mesh.vertices[edges[:,1]],axis=1)
    if np.any(lengths<=0):raise ValueError('Positive source edge lengths required')
    upper=c['maximum_h_fraction']*h
    widths=np.full(mesh.vertex_count,upper);curvature=np.full(mesh.vertex_count,upper)
    groups={g['id']:np.unique(g['mesh_vertices']) for g in graph['groups'] if g['layer']=='main'}
    adj=coo_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(mesh.vertex_count,mesh.vertex_count)).tocsr()
    distances={gid:dijkstra(adj,directed=False,indices=ids,min_only=True) for gid,ids in groups.items()}
    pairs=[];touching=[];keys=sorted(groups)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            if np.intersect1d(groups[a],groups[b]).size:touching.append([a,b]);continue
            separation=float(distances[a][groups[b]].min())
            if separation>=c['strip_rows']*upper:continue
            for own,other in [(a,b),(b,a)]:
                ids=groups[own];widths[ids]=np.minimum(widths[ids],distances[other][ids]/c['strip_rows'])
            pairs.append(dict(a=a,b=b,minimum_separation_h=separation/h,selected_status_used=False))
    normals,areas,centers=triangle_geometry(mesh);face_curvature=np.zeros(mesh.face_count)
    # Exclude sharp creases from the smooth-surface curvature proxy. They remain
    # in the feature graph, with full width requirements and downstream checks.
    for fi,fj,(a,b) in edge_topology(mesh)[1]:
        angle=float(np.arccos(np.clip(normals[fi]@normals[fj],-1,1)))
        if angle<1e-8 or angle>=np.radians(c['smooth_dihedral_degrees']):continue
        edge=mesh.vertices[b]-mesh.vertices[a];edge/=np.linalg.norm(edge)
        n=normals[fi]+normals[fj];n/=np.linalg.norm(n);across=np.cross(n,edge)
        separation=abs(float((centers[fj]-centers[fi])@across))
        if separation<=1e-12*h:continue
        k=angle/separation
        face_curvature[fi]=max(face_curvature[fi],k);face_curvature[fj]=max(face_curvature[fj],k)
    active=face_curvature>0
    face_caps=np.full(mesh.face_count,upper)
    face_caps[active]=np.minimum(upper,np.sqrt(8*c['sagitta_h_fraction']*h/face_curvature[active]))
    np.minimum.at(curvature,mesh.faces.ravel(),np.repeat(face_caps,3))
    requested=np.minimum(widths,curvature)
    # Preserve raw requests; the numerical floor never changes feature membership.
    raw=graded_envelope(np.maximum(requested,1e-12*h),edges,lengths,c['gradation_slope'])
    hv=graded_envelope(np.maximum(requested,c['minimum_h_fraction']*h),edges,lengths,c['gradation_slope'])
    return dict(vertex_sizes=hv,raw_vertex_sizes=raw,requested_vertex_sizes=requested,width_caps=widths,curvature_caps=curvature,
                edges=edges,edge_lengths=lengths,areas=areas,config=c,
                diagnostics=dict(all_main_group_ids=keys,close_pairs=pairs,touching_pairs=touching,
                                 width_request_vertices=int((widths<upper-1e-12*h).sum()),
                                 curvature_request_vertices=int((curvature<upper-1e-12*h).sum()),
                                 floor_limited_vertices=int((requested<c['minimum_h_fraction']*h-1e-12*h).sum()),
                                 maximum_graph_slope=float(np.max(abs(hv[edges[:,0]]-hv[edges[:,1]])/lengths))))


def allocate(mesh,h,req,target_area_ratio):
    """Maximize demand contrast in an affine size family at fixed count proxy."""
    if not np.isfinite(target_area_ratio) or target_area_ratio<1:raise ValueError('Target ratio must be finite and >=1')
    desired=req['vertex_sizes']/h;upper=req['config']['maximum_h_fraction'];areas=req['areas'];weights=areas/areas.sum()
    def family(alpha):
        hv=upper*(1-alpha)+alpha*desired
        rho=1/hv[mesh.faces].min(axis=1);k=float(np.sqrt(target_area_ratio/(weights@(rho*rho))))
        return hv/k,rho*k,k
    def feasible(alpha):
        hv,_,k=family(alpha)
        return alpha<=k+1e-14 and hv.max()<=upper+1e-14
    if feasible(1.):alpha=1.
    else:
        lo,hi=0.,1.
        for _ in range(60):
            mid=(lo+hi)/2
            if feasible(mid):lo=mid
            else:hi=mid
        alpha=lo
    normalized,rho,k=family(alpha);hv=normalized*h
    demand=req['requested_vertex_sizes'];miss=hv>demand+1e-10*h
    return rho,hv,dict(target_area_ratio=float(target_area_ratio),actual_area_proxy=float(weights@(rho*rho)),
                       retained_contrast=alpha,scale=k,gradation_upper_bound=req['config']['gradation_slope']*alpha/k,
                       min_size_h=float(normalized.min()),max_size_h=float(normalized.max()),
                       unmet_request_vertices=int(miss.sum()),max_relative_request_shortfall=float(np.maximum(0,1-demand/hv).max()),
                       coarsened_area_fraction=float(weights[rho<1-1e-12].sum()))


def propose(mesh,graph,h,baseline_quads,config=None):
    req=requests(mesh,graph,h,config);c=req['config']
    if baseline_quads<=0 or c['max_quads']<baseline_quads:raise ValueError('Budget cap must accommodate the positive baseline count')
    areas=req['areas'];weights=areas/areas.sum()
    full_rho=h/req['vertex_sizes'][mesh.faces].min(axis=1)
    raw_rho=h/req['raw_vertex_sizes'][mesh.faces].min(axis=1)
    full_ratio=float(weights@(full_rho*full_rho));raw_ratio=float(weights@(raw_rho*raw_rho))
    uncapped=max(int(baseline_quads),int(np.ceil(baseline_quads*full_ratio-1e-10)))
    target=min(uncapped,int(c['max_quads']))
    rho,hv,allocation=allocate(mesh,h,req,target/baseline_quads)
    result=dict(config=c,reference_h=float(h),baseline_quads=int(baseline_quads),target_quads=target,
                unclipped_geometric_count_estimate=float(baseline_quads*raw_ratio),
                floored_geometric_count_estimate=float(baseline_quads*full_ratio),uncapped_target_quads=uncapped,
                resource_cap_active=target<uncapped,request_diagnostics=req['diagnostics'],allocation=allocation,
                interpretation='Source-wide size-demand estimate and explicit resource decision; not a calibrated optimal count or predicted backend success.')
    return rho,hv,result,req
