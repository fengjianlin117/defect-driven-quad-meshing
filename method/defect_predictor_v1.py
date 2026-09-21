"""Baseline-only repair opportunity and unique-edge budget selection prototype.

This is a conditional geometric surrogate, NOT a learned prediction of actual
MIQ/QEx success. No output of a constrained candidate is an input to scoring.
"""
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from weak_layout_pipeline.pipeline.mesh import unique_edges, triangle_geometry
from weak_layout_pipeline.pipeline.sampling import surface_distances
from feature_diagnostics import segment_query, edge_set

CONFIG = dict(distance_fraction_h=.1, tangent_degrees=15., edge_samples_per_h=32,
              minimum_samples_per_edge=4, salience='sin(dihedral/2)^2',
              loss='0.5*min(surface_distance/(0.1h),1)+0.5*unaligned_fraction',
              selection_budget_fractions=[.25,.5,.75], main_candidates_only=True,
              gradation_slope=.5, strip_rows=1., minimum_h_fraction=.25,
              note='Research defaults, not calibrated acceptance tolerances or guaranteed error bounds.')


def measure_edges(source, output, graph, h):
    points=[]; tangents=[]; slices=[]
    for row in graph['source_edges']:
        a,b=source.vertices[row['vertices']]
        vec=b-a; length=np.linalg.norm(vec)
        n=max(CONFIG['minimum_samples_per_edge'], math.ceil(length/h*CONFIG['edge_samples_per_h']))
        first=len(points)
        points.extend(a+((np.arange(n)+.5)/n)[:,None]*vec)
        tangents.extend(np.repeat((vec/length)[None],n,axis=0))
        slices.append(slice(first,first+n))
    if not points:
        return []
    p=np.asarray(points); t=np.asarray(tangents)
    distance,_,aligned,_=segment_query(p,output.vertices[unique_edges(output)],t,CONFIG['tangent_degrees'])
    surface=surface_distances(p,output)
    rows=[]
    for record,sl in zip(graph['source_edges'],slices):
        weight=float(np.sin(np.radians(record['dihedral_degrees'])/2)**2)
        surface_loss=float(np.minimum(surface[sl]/(CONFIG['distance_fraction_h']*h),1).mean())
        missing=float((aligned[sl]>CONFIG['distance_fraction_h']*h).mean())
        cost=record['length']/h
        loss=.5*(surface_loss+missing)
        rows.append(dict(vertices=record['vertices'],layer=record['layer'],group_ids=record['group_ids'],
                         sample_count=sl.stop-sl.start,length_h=cost,crease_weight=weight,
                         surface_loss=surface_loss,unaligned_fraction=missing,
                         edge_distance_p95_h=float(np.percentile(distance[sl],95)/h),
                         surface_distance_p95_h=float(np.percentile(surface[sl],95)/h),
                         weighted_defect=cost*weight*loss,
                         potential_loss_reduction_if_perfectly_repaired=cost*weight*loss))
    return rows


def union_score(edges, records):
    es={tuple(sorted(e)) for e in edges}
    lookup={tuple(r['vertices']):r for r in records}
    if es-set(lookup):
        raise ValueError('Candidate contains unmeasured edges')
    return dict(potential_reduction=sum(lookup[e]['weighted_defect'] for e in es),
                length_h=sum(lookup[e]['length_h'] for e in es),unique_edges=len(es))


def select_groups(graph, records, fraction):
    """Greedy marginal reduction / added length with exact edge-union accounting.

    Full groups are indivisible. No semantic labels, axes, output reruns, assumed
    resolution infeasibility, or group-count bonus enters this calculation.
    """
    groups=[c for c in graph['groups'] if c['layer']=='main']
    all_edges=edge_set(groups)
    budget=union_score(all_edges,records)['length_h']*fraction
    selected=set(); chosen=[]; trace=[]
    remaining={c['id']:edge_set([c]) for c in groups}
    while remaining:
        options=[]
        used=union_score(selected,records)['length_h']
        for gid,es in remaining.items():
            gain=union_score(es-selected,records)
            if gain['length_h']<=1e-12 or gain['potential_reduction']<=1e-12:
                continue
            if used+gain['length_h']>budget+1e-10:
                continue
            options.append((gain['potential_reduction']/gain['length_h'],gain['potential_reduction'],gid,gain))
        if not options:
            break
        _,_,gid,gain=max(options,key=lambda x:(x[0],x[1],x[2]))
        chosen.append(gid);selected|=remaining.pop(gid)
        trace.append(dict(group_id=gid,marginal=gain,cumulative=union_score(selected,records)))
    return dict(budget_fraction=fraction,budget_length_h=budget,selected_group_ids=chosen,
                selected_edges=[list(e) for e in sorted(selected)],trace=trace,
                predicted=union_score(selected,records),all_main=union_score(all_edges,records),
                interpretation='Conditional defect removal opportunity; actual response, protection of unselected features, and feasibility unproven.')


def size_for_selection(mesh,graph,selection,h):
    """Conservative edge-path grading for disjoint selected closed curves.

    One-cell strip request is a sizing hypothesis, not an integer feasibility
    certificate. Current prototype does not infer sizing at touching curves.
    Refinement floor exposes unresolved requests rather than deleting curves.
    """
    es=unique_edges(mesh)
    length=np.linalg.norm(mesh.vertices[es[:,0]]-mesh.vertices[es[:,1]],axis=1)
    adjacency=coo_matrix((np.r_[length,length],(np.r_[es[:,0],es[:,1]],np.r_[es[:,1],es[:,0]])),
                         shape=(mesh.vertex_count,mesh.vertex_count)).tocsr()
    chosen=[c for c in graph['groups'] if c['id'] in selection['selected_group_ids'] and c['closed']]
    distances={c['id']:dijkstra(adjacency,directed=False,indices=np.unique(c['mesh_vertices']),min_only=True) for c in chosen}
    seed_h={c['id']:h for c in chosen};pairs=[]
    for i,a in enumerate(chosen):
        av=set(a['mesh_vertices'])
        for b in chosen[i+1:]:
            if av & set(b['mesh_vertices']):
                pairs.append(dict(a=a['id'],b=b['id'],status='touching_curves_sizing_not_resolved'))
                continue
            d=distances[a['id']][np.unique(b['mesh_vertices'])]
            separation=float(d.min())
            if not np.isfinite(separation):
                pairs.append(dict(a=a['id'],b=b['id'],status='disconnected_surface_components'))
                continue
            requested=min(h,separation/CONFIG['strip_rows'])
            applied=max(requested,CONFIG['minimum_h_fraction']*h)
            seed_h[a['id']]=min(seed_h[a['id']],applied)
            seed_h[b['id']]=min(seed_h[b['id']],applied)
            pairs.append(dict(a=a['id'],b=b['id'],edge_path_separation_h=separation/h,
                              requested_h_fraction=requested/h,applied_h_fraction=applied/h,
                              request_clipped=applied>requested+1e-12,
                              status='sizing_hypothesis_not_feasibility_proof'))
    hv=np.full(mesh.vertex_count,h)
    for c in chosen:
        hv=np.minimum(hv,seed_h[c['id']]+CONFIG['gradation_slope']*distances[c['id']])
    hf=hv[mesh.faces].min(axis=1)
    rho=h/hf
    areas=triangle_geometry(mesh)[1]
    return rho,dict(pairs=pairs,seeds=[dict(group_id=g,h_fraction=value/h) for g,value in seed_h.items()],
                    density_min=float(rho.min()),density_max=float(rho.max()),
                    predicted_area_budget_ratio=float(areas@(rho*rho)/areas.sum()),
                    unresolved_clipped_requests=sum(p.get('request_clipped',False) for p in pairs),
                    limitations=['only disjoint closed groups supply size requests',
                                 'source-edge paths upper-bound shortest surface distances; may miss narrow passages',
                                 'field singularities, integer periods and UV injectivity not certified',
                                 'density floor may leave an explicitly reported size request unmet'])
