"""Align residual constraint triggers with the existing aggregate safeguard.

This changes proposals, not acceptance thresholds or source references.
"""
from coordination import key,loss
from defect_driven_selection import needs_repair
from protected_decision import no_worse
import math
from collections import defaultdict
import numpy as np
from geometry_budget import requests,allocate
from defect_selection_v2 import graded_envelope

def repair_edges(records,baseline,initial_selected,include_protection=True):
    now={key(r):r for r in records if r['layer']=='main'}
    before={key(r):r for r in baseline if r['layer']=='main'}
    if now.keys()!=before.keys():raise ValueError('Reference edge identity changed')
    protected=set(before)-set(initial_selected)
    original={e for e,r in now.items() if needs_repair(r)}
    old_sum=sum(loss(before[e]) for e in protected);new_sum=sum(loss(now[e]) for e in protected)
    failed=not no_worse(new_sum,old_sum)
    regressed={e for e in protected if not no_worse(loss(now[e]),loss(before[e]))} if failed and include_protection else set()
    return original|regressed,dict(original_deficient_edges=[list(e) for e in sorted(original)],
        protection_trigger_edges=[list(e) for e in sorted(regressed)],baseline_protected_deficit=old_sum,parent_protected_deficit=new_sum,
        aggregate_protection_failed=failed,include_protection=include_protection,
        policy='When the fixed initial complement violates its unchanged aggregate safeguard, allow its individually regressed edges to trigger whole-group repair even below initial diagnostic thresholds.')

def constraint_repair(group_edges,chosen,records,audit,baseline,initial_selected,include_protection=True,max_screenings=64):
    required,triggers=repair_edges(records,baseline,initial_selected,include_protection)
    rows={key(r):r for r in records if r['layer']=='main'}
    union=lambda groups:set().union(*(group_edges[g] for g in groups))
    selected=union(chosen);options=[];trace=[];count=0
    remaining=sorted((g for g in group_edges if g not in chosen and group_edges[g]&required),
        key=lambda g:(-sum(loss(rows[e]) for e in group_edges[g]-selected),g))
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
            for remove in sorted(chosen,key=lambda g:(sum(loss(rows[e]) for e in group_edges[g]),g))[:16]:
                if count>=max_screenings:break
                count+=1;groups=[g for g in chosen if g!=remove]+[add];risk=audit.check(union(groups))
                if risk['admissible']:combinations.append((remove,groups,risk))
        for remove,groups,risk in combinations:
            es=union(groups);options.append(dict(add=add,remove=remove,groups=groups,risk=risk,
                selected_edges=[list(e) for e in sorted(es)],displaced_edges=[list(e) for e in sorted(selected-es)],
                conditional_residual=sum(loss(rows[e]) for e in required-es)))
    best=min(options,key=lambda x:(x['conditional_residual'],len(x['displaced_edges']),x['add'],x['remove'] or '')) if options else None
    return dict(proposal=best,alternatives=options,trace=trace,screening_calls=count,budget_exhausted=count>=max_screenings,
        unaddressed_required_edges=[list(e) for e in sorted(required-selected)],trigger_audit=triggers)

def simple_initial(group_edges,records,audit):
    rows={key(r):r for r in records if r['layer']=='main'};selected=set();chosen=[];trace=[]
    order=sorted((g for g,es in group_edges.items() if any(needs_repair(rows[e]) for e in es)),
        key=lambda g:(-sum(loss(rows[e]) for e in group_edges[g])/max(sum(rows[e]['length_h'] for e in group_edges[g]),1e-30),g))
    for group in order:
        risk=audit.check(selected|group_edges[group]);trace.append(dict(group=group,risk=risk))
        if risk['admissible']:chosen.append(group);selected|=group_edges[group]
    return dict(selected_group_ids=chosen,selected_edges=[list(e) for e in sorted(selected)],trace=trace,order=order,
        policy='Whole diagnostically needed groups, descending length-weighted average deficit, greedy fixed-frame compatibility; no quota.')

def size_repair(mesh,graph,h,baseline,parent,density,target,step,initial_selected,include_protection=True,prior_requested=None,max_quads=2048):
    if not math.isfinite(step) or step<=1:raise ValueError('Feedback density multiplier must exceed one')
    required,triggers=repair_edges(parent['edge_defects'],baseline['edge_defects'],initial_selected,include_protection)
    before={key(r):r for r in baseline['edge_defects'] if r['layer']=='main'}
    active=[r for r in parent['edge_defects'] if r['layer']=='main' and key(r) in required and loss(r)>loss(before[key(r)])+1e-9]
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
            floored_count_estimate=floored_count,allocation=allocation,cap_active=new_target<math.ceil(floored_count-1e-10),trigger_audit=triggers))
