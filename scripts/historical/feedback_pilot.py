"""Frozen one-step measured-residual sizing experiment on three seen models.

Both fixed-proxy-count and demand-estimated-count arms are specified before any
new solver output. This pilot isolates sizing; it is not the final method.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys, argparse, math, shutil, importlib.util, time
import numpy as np
from audit_evidence import ROOT, OLD, OUT, read, save, sha
PILOT=OUT/'feedback_pilot_v1'
def modules():
    sys.path.insert(0,str(PILOT/'method'))
    global load_obj, requests, allocate, graded_envelope, run_backend, evaluate, deficit, count_correction_target, decide_protected, area_samples
    from weak_layout_pipeline.pipeline.mesh import load_obj
    from geometry_budget import requests, allocate
    from defect_selection_v2 import graded_envelope
    from weak_layout_pipeline.backend.runner import run_backend
    from candidate_decision import count_correction_target
    from protected_decision import decide_protected
    from weak_layout_pipeline.pipeline.sampling import area_samples
    spec=importlib.util.spec_from_file_location('frozen_evaluator',PILOT/'evaluator.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    evaluate,deficit=m.evaluate,m.deficit

def freeze():
    PILOT.mkdir(exist_ok=False)
    shutil.copytree(OLD/'defect_driven_v3/method',PILOT/'method',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    shutil.copy2(OLD/'defect_driven_v3/executor.py',PILOT/'evaluator.py')
    modules()
    protocol=dict(role='seen development mechanism pilot',models=['B49','B53','B65'],
        parent='v3 allocated_0 for every model, without selecting best output',
        trigger='Common reference edge has positive deficit regression >1e-9 and remains diagnostically deficient in parent output.',
        local_step='Request h_local/sqrt(2) at triggered endpoints, retain every source-wide geometric demand; reapply existing .25h numerical floor and .5 graph gradation.',
        arms=['same_budget','demand_budget'],max_backend_calls=12,max_calls_per_arm=2,max_quads=2048,
        timeout_per_backend_stage_seconds=60,stiffening=0,threads=1,
        correction='At most one inherited count correction per arm; no extra attempt for structural failure alone.',
        evaluation='Same frozen feature and surface samples and tolerances; full reference positive regression additionally reported; old protected set fixed from parent selection for both arms.',
        stopping='One feedback step only, two arms and at most one correction each; retain all intermediates and failures.',
        no_new_validation=True,limitations=['Local density doubling is a development step-size hypothesis, not calibrated CAD tolerance.',
            'Equal area proxy does not imply equal actual face count.', 'No claim that sizing repairs within-group fixed-frame conflict.'])
    save(PILOT/'protocol.json',protocol)
    plans=[]
    for p in read(OUT/'audit/source_plans.json'):
        if p['model'] not in protocol['models']:continue
        d=PILOT/p['model'];d.mkdir();mesh=load_obj(p['source']);graph=read(p['graph']);h=p['reference_h']
        loc=read(OUT/f'localization/{p["model"]}.json')
        parent_case=next(c for c in loc['cases'] if c['id']=='defect_driven__allocated_0')
        triggers=[r for r in parent_case['edges'] if r['delta']>1e-9 and r['candidate_deficient']]
        req=requests(mesh,graph,h);raw=req['requested_vertex_sizes'].copy()
        for r in triggers:
            requested=r['local_requested_size_h_min']*h/math.sqrt(2)
            ids=r['vertices'];raw[ids]=np.minimum(raw[ids],requested)
        req['requested_vertex_sizes']=raw
        req['vertex_sizes']=graded_envelope(np.maximum(raw,.25*h),req['edges'],req['edge_lengths'],.5)
        req['raw_vertex_sizes']=graded_envelope(np.maximum(raw,1e-12*h),req['edges'],req['edge_lengths'],.5)
        base=read(p['baseline_record']);areas=req['areas'];weights=areas/areas.sum()
        ratio=float(weights@((h/req['vertex_sizes'][mesh.faces].min(axis=1))**2))
        demand=max(p['target_quads'],math.ceil(base['quads']*ratio-1e-10))
        arms=[]
        for kind,target in [('same_budget',p['target_quads']),('demand_budget',min(2048,demand))]:
            rho,hv,stats=allocate(mesh,h,req,target/base['quads'])
            np.savetxt(d/f'{kind}.rho',rho,fmt='%.17g')
            np.savetxt(d/f'{kind}.vertex_h',hv,fmt='%.17g')
            arms.append(dict(kind=kind,target_quads=target,density=str(d/f'{kind}.rho'),allocation=stats))
        plan=dict(**p,arms=arms,trigger_edges=triggers,demand_count_uncapped=demand,
            parent_result=str(OLD/f'defect_driven_v3/development/{p["model"]}/defect_driven__allocated_0/result.json'))
        save(d/'plan.json',plan);plans.append(plan)
        print('FROZEN',p['model'],'triggers',len(triggers),'targets',[(a['kind'],a['target_quads']) for a in arms],flush=True)
    save(PILOT/'plans.json',plans)
    shutil.copy2(Path(__file__),PILOT/'feedback_pilot.py')
    shutil.copy2(Path(__file__).parent/'audit_evidence.py',PILOT/'audit_evidence.py')
    files={str(p):sha(p) for p in PILOT.rglob('*') if p.is_file()}
    for p in plans:
        for key in ['source','pd1','pd2','graph','baseline_record','parent_result']:files[p[key]]=sha(p[key])
        ep=p['methods']['defect_driven']['edges'];files[ep]=sha(ep)
    bins=ROOT/'research_runs/2026-09-19_selection_v3_tools/build/bin'
    for binary in ['miq_adaptive','qex_adapter']:files[str(bins/binary)]=sha(bins/binary)
    save(PILOT/'frozen_hashes.json',files)

def verify():
    assert all(Path(p).is_file() and sha(p)==h for p,h in read(PILOT/'frozen_hashes.json').items())

def amend():
    """Freeze execution-only deduplication before outputs exist."""
    assert not (PILOT/'runs').exists()
    target=PILOT/'execution_amendment.json'
    assert not target.exists()
    shutil.copy2(Path(__file__),PILOT/'runner_v1_1.py')
    save(target,dict(reason='All three models have identical density and target in the two preregistered arms; run identical inputs once.',
        rule='Within each model alias arms only if density SHA256 and target both match; no extra backend calls or evidence count.',
        source=str(PILOT/'runner_v1_1.py'),sha256=sha(PILOT/'runner_v1_1.py'),before_any_backend_output=True))

def run():
    verify();amendment=read(PILOT/'execution_amendment.json');assert sha(amendment['source'])==amendment['sha256']
    modules();dest=PILOT/'runs';dest.mkdir(exist_ok=False);calls=0;results=[];aliases=[]
    bins=ROOT/'research_runs/2026-09-19_selection_v3_tools/build/bin'
    for p in read(PILOT/'plans.json'):
        name=p['model'];mesh=load_obj(p['source']);graph=read(p['graph']);base=read(p['baseline_record']);h=p['reference_h']
        selected={tuple(e) for e in read(p['methods']['defect_driven']['selection'])['selected_edges']}
        all_edges={tuple(sorted(r['vertices'])) for r in base['edge_defects'] if r['layer']=='main'};protected=all_edges-selected
        before={tuple(sorted(r['vertices'])):r for r in base['edge_defects'] if r['layer']=='main'}
        points=area_samples(mesh,4096,190919)[0]
        candidates=[dict(id='baseline',process_success=True,**base,protected_deficit=deficit(base['edge_defects'],protected))]
        parent=read(p['parent_result']);parent['id']='parent_allocated_0';parent['protected_deficit']=deficit(parent['edge_defects'],protected);candidates.append(parent)
        seen_arms={}
        for arm in p['arms']:
            signature=(sha(arm['density']),arm['target_quads'])
            if signature in seen_arms:
                aliases.append(dict(model=name,arm=arm['kind'],identical_to=seen_arms[signature],density_sha256=signature[0],target_quads=signature[1]))
                save(dest/'arm_aliases.json',aliases)
                continue
            seen_arms[signature]=arm['kind']
            g=p['gsize']
            for index in range(2):
                case=f'{name}__{arm["kind"]}_{index}';folder=dest/case;folder.mkdir();calls+=1;assert calls<=12
                started=time.perf_counter()
                backend=run_backend(mesh=Path(p['source']),pd1=Path(p['pd1']),pd2=Path(p['pd2']),density=Path(arm['density']),
                    hard_edges=Path(p['methods']['defect_driven']['edges']),gsize=g,miq_executable=bins/'miq_adaptive',qex_executable=bins/'qex_adapter',
                    output_dir=folder/'backend',stiffness_iterations=0,timeout_seconds=60)
                r=dict(id=case,model=name,arm=arm['kind'],target_quads=arm['target_quads'],gsize=g,process_success=backend['success'],
                    quad_path=str(folder/'backend/quad.obj'),miq_seconds=backend['miq']['elapsed_seconds'],qex_seconds=backend['qex'].get('elapsed_seconds'))
                if backend['success']:
                    try:
                        r.update(evaluate(mesh,load_obj(folder/'backend/quad.obj',require_triangles=False),graph,h,folder/'backend',points))
                        r['protected_deficit']=deficit(r['edge_defects'],protected)
                        r['common_reference_positive_regression']=sum(max(0.,x['length_h']*.5*(x['surface_loss']+x['unaligned_fraction'])-before[tuple(sorted(x['vertices']))]['length_h']*.5*(before[tuple(sorted(x['vertices']))]['surface_loss']+before[tuple(sorted(x['vertices']))]['unaligned_fraction'])) for x in r['edge_defects'] if x['layer']=='main')
                    except Exception as exc:r.update(process_success=False,evaluation_error=repr(exc))
                r['total_seconds']=time.perf_counter()-started;save(folder/'result.json',r);candidates.append(r)
                compact={k:v for k,v in r.items() if k not in ['edge_defects','quality','topology','review','uv']};results.append(compact)
                save(dest/'summary.json',results);print('RESULT',case,r.get('quads'),r.get('basic_output_pass'),r.get('all_main_deficit'),r.get('protected_deficit'),flush=True)
                n=r.get('quads');target=count_correction_target(n,arm['target_quads'],2048) if n else None
                if index or target is None:break
                g*=math.sqrt(target/n)
        decision=decide_protected(candidates,2048);save(dest/f'{name}_decision.json',decision)
        print('DECISION',name,decision['recommended_id'],flush=True);verify()
    save(dest/'integrity.json',dict(native_calls=calls,aliased_arms=len(aliases),completed=True,frozen_inputs_unchanged=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','amend','run']);args=parser.parse_args()
    {'freeze':freeze,'amend':amend,'run':run}[args.stage]()
