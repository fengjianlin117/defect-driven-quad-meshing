"""Bounded measured feedback operations. Full reference geometry is immutable."""
import math
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from defect_driven_selection import needs_repair
from geometry_budget import requests,allocate
from defect_selection_v2 import graded_envelope
from weak_layout_pipeline.pipeline.mesh import unique_edges

def key(row):return tuple(sorted(row['vertices']))
def loss(row):return row['length_h']*.5*(row['surface_loss']+row['unaligned_fraction'])

def constraint_operation(group_edges,chosen,records,audit,max_screenings=64):
    records={key(r):r for r in records if r['layer']=='main'}
    deficient={e for e,r in records.items() if needs_repair(r)}
    union=lambda gs:set().union(*(group_edges[g] for g in gs))
    selected=union(chosen);options=[];trace=[];count=0
    remaining=sorted((g for g in group_edges if g not in chosen and group_edges[g]&deficient),
        key=lambda g:(-sum(loss(records[e]) for e in group_edges[g]-selected),g))
    for add in remaining:
        if count>=max_screenings:break
        count+=1;single=audit.check(group_edges[add])
        if not single['admissible']:
            trace.append(dict(add=add,status='within_group_fixed_frame_conflict',risk=single));continue
        if count>=max_screenings:break
        count+=1;risk=audit.check(selected|group_edges[add])
        combinations=[(None,chosen+[add],risk)] if risk['admissible'] else []
        if not risk['admissible']:
            trace.append(dict(add=add,status='union_conflict',risk=risk))
            for remove in sorted(chosen,key=lambda g:(sum(loss(records[e]) for e in group_edges[g]),g))[:16]:
                if count>=max_screenings:break
                count+=1;groups=[g for g in chosen if g!=remove]+[add];risk=audit.check(union(groups))
                if risk['admissible']:combinations.append((remove,groups,risk))
        for remove,groups,risk in combinations:
            es=union(groups);options.append(dict(add=add,remove=remove,groups=groups,risk=risk,
                selected_edges=[list(e) for e in sorted(es)],displaced_edges=[list(e) for e in sorted(selected-es)],
                conditional_residual=sum(loss(records[e]) for e in deficient-es)))
    best=min(options,key=lambda x:(x['conditional_residual'],len(x['displaced_edges']),x['add'],x['remove'] or '')) if options else None
    return dict(proposal=best,alternatives=options,trace=trace,screening_calls=count,
        budget_exhausted=count>=max_screenings,unaddressed_deficient_edges=[list(e) for e in sorted(deficient-selected)])

def size_operation(mesh,graph,h,baseline,parent,density,target,step,prior_requested=None,max_quads=2048):
    if not math.isfinite(step) or step<=1:raise ValueError('Feedback density multiplier must exceed one')
    b={key(r):r for r in baseline['edge_defects'] if r['layer']=='main'}
    active=[r for r in parent['edge_defects'] if r['layer']=='main' and loss(r)>loss(b[key(r)])+1e-9 and needs_repair(r)]
    if not active:return None
    req=requests(mesh,graph,h);requested=req['requested_vertex_sizes'].copy()
    if prior_requested is not None:requested=np.minimum(requested,prior_requested)
    incident=defaultdict(list)
    for f,face in enumerate(mesh.faces):
        for a,z in zip(face,np.roll(face,-1)):incident[tuple(sorted((int(a),int(z))))].append(f)
    for row in active:
        e=key(row);local=h/float(np.max(density[incident[e]]))/math.sqrt(step)
        requested[list(e)]=np.minimum(requested[list(e)],local)
    req['requested_vertex_sizes']=requested
    req['vertex_sizes']=graded_envelope(np.maximum(requested,.25*h),req['edges'],req['edge_lengths'],.5)
    req['raw_vertex_sizes']=graded_envelope(np.maximum(requested,1e-12*h),req['edges'],req['edge_lengths'],.5)
    weights=req['areas']/req['areas'].sum()
    raw_count=float(baseline['quads']*(weights@((h/req['raw_vertex_sizes'][mesh.faces].min(axis=1))**2)))
    floored_count=float(baseline['quads']*(weights@((h/req['vertex_sizes'][mesh.faces].min(axis=1))**2)))
    new_target=min(max_quads,max(target,math.ceil(floored_count-1e-10)))
    rho,hv,allocation=allocate(mesh,h,req,new_target/baseline['quads'])
    return dict(density=rho,requested=requested,target=new_target,vertex_sizes=hv,
        audit=dict(trigger_edges=[r['vertices'] for r in active],density_step=step,raw_count_estimate=raw_count,
            floored_count_estimate=floored_count,allocation=allocation,cap_active=new_target<math.ceil(floored_count-1e-10)))

def failure_removal(mesh,output,group_edges,chosen,context):
    """One reversible group removal; spatial association is not a causal proof."""
    if not chosen:return None
    associations={g:0 for g in chosen};boundary_points=[]
    if output is not None:
        incidence=defaultdict(int)
        for face in output.faces:
            for a,b in zip(face,np.roll(face,-1)):incidence[tuple(sorted((int(a),int(b))))]+=1
        boundary_points=[output.vertices[list(e)].mean(0) for e,n in incidence.items() if n==1]
    es=sorted(set().union(*(group_edges[g] for g in chosen)))
    if boundary_points and es:
        midpoints=np.array([mesh.vertices[list(e)].mean(0) for e in es]);_,nearest=cKDTree(midpoints).query(boundary_points)
        for i in np.atleast_1d(nearest):
            for g in chosen:
                if es[int(i)] in group_edges[g]:associations[g]+=1
    rows=[]
    for g in chosen:
        length=sum(context.length[e] for e in group_edges[g])
        risk=sum(context.length[e]*context.field_risk[e] for e in group_edges[g])/max(length,1e-30)
        rows.append(dict(group=g,boundary_associations=associations[g],field_mismatch=risk,
            displaced_baseline_deficit=sum(context.length[e]*context.loss[e] for e in group_edges[g])))
    remove=min(rows,key=lambda r:(-r['boundary_associations'],-r['field_mismatch'],r['displaced_baseline_deficit'],r['group']))['group']
    return dict(remove=remove,groups=[g for g in chosen if g!=remove],ranking=rows,boundary_points=len(boundary_points),
        reason='Try removal near measured output boundary; if no boundary data, try highest field-mismatch group. This is a bounded hypothesis, not a diagnosed cause.',
        reference_policy='Every displaced edge remains in the common reference and all size requests.')

def common_metrics(row,baseline,initial_selected):
    if 'edge_defects' not in row:return {}
    b={key(r):r for r in baseline['edge_defects'] if r['layer']=='main'}
    after={key(r):r for r in row['edge_defects'] if r['layer']=='main'}
    if b.keys()!=after.keys():raise ValueError('Reference edge identity changed')
    return dict(protected_deficit=sum(loss(r) for e,r in after.items() if e not in initial_selected),
        common_positive_regression=sum(max(0.,loss(r)-loss(b[e])) for e,r in after.items()),
        newly_deficient_edges=sum(needs_repair(r) and not needs_repair(b[e]) for e,r in after.items()),
        newly_deficient_length_h=sum(r['length_h'] for e,r in after.items() if needs_repair(r) and not needs_repair(b[e])),
        unresolved_deficient_edges=[list(e) for e,r in after.items() if needs_repair(r)])
