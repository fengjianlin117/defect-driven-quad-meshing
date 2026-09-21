"""Freeze one method bundle and the complete seen-model ablation matrix."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import argparse,sys,shutil,copy
from audit_evidence import ROOT,OLD,OUT,read,save,sha
D=OUT/'coordinator_v4_development';W=Path(__file__).resolve().parents[1]
VARIANTS=['full','no_size','no_constraint','no_recovery','initial_only']

def freeze():
    D.mkdir(exist_ok=False)
    shutil.copytree(OLD/'defect_driven_v3/method',D/'method',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for p in (W/'method_v4').glob('*.py'):shutil.copy2(p,D/'method'/p.name)
    neutral=(W/'scripts/common_evaluation.py').read_text()
    for line in ['import sys\n','from audit_evidence import OLD\n',"sys.path.insert(0,str(OLD/'defect_driven_v3/method'))\n"]:neutral=neutral.replace(line,'')
    (D/'method/neutral_evaluation.py').write_text(neutral)
    plans=read(OUT/'audit/source_plans.json')
    for p in plans:
        name=p['model'];cache=list(p['cache_records'])
        cache+=list(map(str,(OLD/f'defect_driven_v3/development/{name}').glob('*/result.json')))
        for stage in ['feedback_pilot_v1','feedback_matched_controls_v1','constraint_feedback_pilot_v1']:
            cache+=list(map(str,(OUT/stage/'runs').glob(name+'__*/result.json')))
        p['verified_cache_records']=sorted(set(cache))
    save(D/'plans.json',plans)
    save(D/'protocol.json',dict(role='All twelve are seen development; no heldout claim',variants=VARIANTS,
        max_logical_attempts_per_model_variant=12,max_rounds=2,max_quad_count=2048,density_step=2,
        max_native_calls_upper_bound=12*len(plans)*len(VARIANTS),
        fairness='Same frozen initial proposal, original fields, full reference, solver/time caps and decision gate. Ablations may stop early when their operation is disabled. Count real calls, exact-input cache reuse and logical attempts separately.',
        protection='Fixed complement of initial defect-driven set within each model; full common per-edge diagnostics also retained. No acceptance threshold relaxed.',
        cache='Only equal hashes for source, both fields, both binaries, density, edges and exact IEEE float gsize; previous same 60s/stiffening=0 protocol. Reuse is not independent evidence.',
        remaining_scientific_gaps=['shared-reference application tolerance','full CAD dimensions','global intersection','unseen family validation','field seed stability']))
    shutil.copy2(Path(__file__),D/'run_coordinator_v4.py');shutil.copy2(Path(__file__).parent/'audit_evidence.py',D/'audit_evidence.py')
    hashes={str(p):sha(p) for p in D.rglob('*') if p.is_file()}
    for p in plans:
        for key in ['source','pd1','pd2','graph','baseline_record']:hashes[p[key]]=sha(p[key])
        for f in (Path(p['source']).parent/'baseline').iterdir():
            if f.is_file():hashes[str(f)]=sha(f)
        for rpath in p['verified_cache_records']:
            hashes[rpath]=sha(rpath);r=read(rpath);q=Path(r['quad_path']);b=q.parent/'backend_run.json';hashes[str(b)]=sha(b)
            if q.is_file():hashes[str(q)]=sha(q)
    bins=ROOT/'research_runs/2026-09-19_selection_v3_tools/build/bin'
    for binary in ['miq_adaptive','qex_adapter']:hashes[str(bins/binary)]=sha(bins/binary)
    save(D/'frozen_hashes.json',hashes);print('FROZEN_V4',len(plans),len(hashes),flush=True)

def cache_for(p,miq,qex):
    result={};prefix=(sha(p['source']),sha(p['pd1']),sha(p['pd2']),sha(miq),sha(qex))
    for path in p['verified_cache_records']:
        r=read(path);record=read(Path(r['quad_path']).parent/'backend_run.json');a=record['artifacts']
        if tuple(a[k]['sha256'] for k in ['mesh','pd1','pd2','miq_executable','qex_executable'])!=prefix:continue
        if record['stiffness_iterations']!=0:continue
        signature=(*prefix,a['density']['sha256'],a['hard_edges']['sha256'],float(record['gsize']).hex(),0,60)
        result[signature]=dict(record=r,path=path)
    return result

def run():
    frozen=read(D/'frozen_hashes.json');assert all(sha(p)==h for p,h in frozen.items())
    sys.path.insert(0,str(D/'method'));from coordinator import Coordinator
    out=D/'runs';out.mkdir(exist_ok=True);summary=[]
    bins=ROOT/'research_runs/2026-09-19_selection_v3_tools/build/bin';miq=bins/'miq_adaptive';qex=bins/'qex_adapter'
    for p in read(D/'plans.json'):
        cache=cache_for(p,miq,qex)
        for variant in VARIANTS:
            dest=out/p['model']/variant
            if (dest/'complete.json').exists():
                for record_path in dest.glob('*/result.json'):
                    r=read(record_path)
                    signature=(sha(p['source']),sha(p['pd1']),sha(p['pd2']),sha(miq),sha(qex),sha(r['density_path']),sha(r['edges_path']),float(r['gsize']).hex(),0,60)
                    cache[signature]=dict(record=r,path=str(record_path))
                summary.append(dict(model=p['model'],**read(dest/'complete.json')));save(out/'summary.json',summary)
                print('VERIFIED_COMPLETED_SKIP',p['model'],variant,flush=True)
                continue
            if dest.exists():raise RuntimeError('Incomplete output must be inspected before resuming: '+str(dest))
            c=Coordinator(p,dest,miq,qex,variant,cache)
            # Recomputed proposal must reproduce the frozen no-quota method.
            expected=read(p['methods']['defect_driven']['selection']);assert c.selection['selected_edges']==expected['selected_edges']
            assert c.target==p['target_quads']
            decision=c.run();summary.append(dict(model=p['model'],**read(dest/'complete.json')));save(out/'summary.json',summary)
            print('MODEL_VARIANT_DONE',p['model'],variant,decision['recommended_id'],c.native,c.reused,flush=True)
        assert all(sha(path)==h for path,h in frozen.items())
    save(out/'integrity.json',dict(models=12,variants=5,native_calls=sum(r['native_calls'] for r in summary),
        reused_attempts=sum(r['reused_attempts'] for r in summary),logical_attempts=sum(r['attempts'] for r in summary),complete=True,frozen_unchanged=True))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args();{'freeze':freeze,'run':run}[args.stage]()
