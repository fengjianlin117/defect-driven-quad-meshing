"""Conservative default among measured outputs; keep every geometric tradeoff."""
import math
from candidate_decision import decide

def no_worse(value,reference):
    return math.isfinite(value) and value<=reference+1e-9*max(1.,abs(reference))

def decide_protected(candidates,max_quads,min_quads=0):
    previous=decide(candidates,max_quads)
    bases=[r for r in candidates if r['id']=='baseline']
    if len(bases)!=1:raise ValueError('Exactly one measured baseline is required')
    base=bases[0];keys=['all_main_deficit','symmetric_rms_h','protected_deficit']
    if any(not isinstance(base.get(k),(int,float)) or not math.isfinite(base[k]) or base[k]<0 for k in keys):raise ValueError('Baseline comparison metrics required')
    accepted=[];audit=[]
    for r in candidates:
        reasons=[]
        if r['id'] not in previous['eligible_ids']:reasons.append('basic_or_resource_checks_failed')
        if r.get('quads',0)<min_quads:reasons.append('actual_count_below_resolution_floor')
        for key in keys:
            v=r.get(key)
            if not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:reasons.append('missing_or_invalid_'+key)
            elif not no_worse(v,base[key]):reasons.append('baseline_regression_'+key)
        audit.append(dict(id=r['id'],conservative_eligible=not reasons,reasons=reasons))
        if not reasons:accepted.append(r)
    best=min(accepted,key=lambda r:(r['all_main_deficit'],r['symmetric_rms_h'],r['quads'],r['id'])) if accepted else None
    return dict(recommended_id=best['id'] if best else None,eligible_ids=[r['id'] for r in accepted],audit=audit,
                unrestricted_feature_priority_id=previous['recommended_id'],tradeoff_ids=previous['pareto_ids'],
                policy='Prefer basic-valid, within-cap candidates with no measured regression in whole-feature deficit, surface RMS, or aggregate protected-feature deficit relative to the baseline. Same fixed protected edge set across candidates. Keep every alternative.',
                tolerance='1e-9 relative/absolute numerical slack only; not a geometric acceptance tolerance.',
                limitation='Measured aggregate safeguards, not per-feature preservation, uncertainty bounds, CAD tolerance or a pre-generation guarantee.')
