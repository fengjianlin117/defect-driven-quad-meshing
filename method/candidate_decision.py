"""Transparent provisional decision from a bounded set of measured outputs.

This is a geometry-priority default, not a universal shape acceptance standard.
UV warnings never act as a final-output gate; partial outcomes remain in audit.
"""
import math


def count_correction_target(actual,target,cap):
    """One correction trigger also covers over-cap counts within the 5% band."""
    if any(not math.isfinite(float(x)) or x<=0 for x in [actual,target,cap]):raise ValueError('Positive finite counts required')
    if abs(actual/target-1)>.05 or actual>cap:return min(target,.98*cap)
    return None


def decide(candidates,max_quads):
    if not isinstance(max_quads,int) or max_quads<1:raise ValueError('Positive integer count cap required')
    if len({r['id'] for r in candidates})!=len(candidates):raise ValueError('Unique candidate ids required')
    eligible=[];audit=[]
    for r in candidates:
        reasons=[]
        if not r.get('process_success',False):reasons.append('generation_or_evaluation_unavailable')
        if r.get('basic_output_pass') is not True:reasons.append('final_output_basic_check_failed_or_missing')
        n=r.get('quads')
        if not isinstance(n,int) or isinstance(n,bool) or n<=0:reasons.append('invalid_quad_count')
        elif n>max_quads:reasons.append('actual_count_exceeds_resource_cap')
        for key in ['all_main_deficit','symmetric_rms_h']:
            if not isinstance(r.get(key),(int,float)) or not math.isfinite(r[key]) or r[key]<0:reasons.append('invalid_'+key)
        audit.append(dict(id=r['id'],eligible=not reasons,reasons=reasons))
        if not reasons:eligible.append(r)
    keys=['all_main_deficit','symmetric_rms_h','quads']
    front=[]
    for r in eligible:
        dominated=any(all(s[k]<=r[k] for k in keys) and any(s[k]<r[k] for k in keys) for s in eligible)
        if not dominated:front.append(r['id'])
    best=min(eligible,key=lambda r:(r['all_main_deficit'],r['symmetric_rms_h'],r['quads'],r['id'])) if eligible else None
    return dict(recommended_id=best['id'] if best else None,eligible_ids=[r['id'] for r in eligible],pareto_ids=front,audit=audit,
                policy='Among basic-valid outputs within actual count cap, minimize measured whole-reference feature deficit, then surface RMS, then count. Retain the Pareto alternatives and every failed/partial result.',
                limitation='Provisional geometry-priority default. No global intersection, application suitability, or absolute shape-tolerance certificate. Per-feature/region regressions still require inspection.')
