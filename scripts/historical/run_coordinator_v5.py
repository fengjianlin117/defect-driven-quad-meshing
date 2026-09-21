"""Freeze and run the next version of the same research method."""
from pathlib import Path
import sys,subprocess,shutil
from audit_evidence import ROOT,OUT,read,save,sha
import strong_static_controls as static
D=OUT/'coordinator_v5_development';M=OUT/'coordinator_v4_development'
W=Path('research://windows-delivery/paper_stage_20260920')
VARIANTS=['full','no_protection_feedback','no_size','no_constraint','no_recovery','initial_only']

def extra_records(p):
    paths=[f for f in (OUT/'strong_static_controls_v1/runs').glob(p['model']+'__*/result.json')]
    paths+=list((OUT/'protection_trigger_pilot_v1a/runs').glob(p['model']+'__*/result.json'))
    return paths

def cache_for(p):
    cache=static.historic(p)
    prefix=tuple(sha(p[k]) for k in ['source','pd1','pd2'])+(sha(static.BINS/'miq_adaptive'),sha(static.BINS/'qex_adapter'))
    for path in extra_records(p):
        r=read(path);b=read(Path(r['quad_path']).parent/'backend_run.json');a=b['artifacts']
        if tuple(a[k]['sha256'] for k in ['mesh','pd1','pd2','miq_executable','qex_executable'])!=prefix or b['stiffness_iterations']!=0:continue
        sig=prefix+(a['density']['sha256'],a['hard_edges']['sha256'],float(b['gsize']).hex(),0,60)
        cache[sig]=dict(record=r,path=str(path))
    return cache

def freeze():
    D.mkdir(exist_ok=False);shutil.copytree(M/'method',D/'method',ignore=shutil.ignore_patterns('__pycache__'))
    for f in (W/'method_v5').glob('*.py'):shutil.copy2(f,D/'method'/f.name)
    tests=[]
    for name in ['test_coordination.py','test_protection_feedback.py','test_v5_decision.py']:
        r=subprocess.run([sys.executable,'-B',str(D/'method'/name)],capture_output=True,text=True)
        tests.append(dict(file=name,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    save(D/'tests.json',tests);assert all(r['returncode']==0 for r in tests),tests
    plans=read(M/'plans.json');save(D/'plans.json',plans)
    save(D/'protocol.json',dict(role='Same primary research method, next frozen development version; all 12 inputs seen.',
        changes=['Use the simpler mean-deficit initial ordering, equal selected sets on all 12 prior development cases.',
                 'Apply the existing exact triangle-soup check to parents and final decisions.',
                 'Allow unchanged aggregate-protection violations to trigger constraint and size repair.',
                 'Scan unexpanded parents when a parent has no operation, with explicit total visit and solve caps.'],
        variants=VARIANTS,max_logical_attempts_per_cell=12,max_parent_visits=12,productive_rounds=2,
        max_actual_quads=2048,density_step=2.,max_native_calls_upper_bound=12*12*len(VARIANTS),
        acceptance='Original aggregate defect/RMS/fixed-complement safeguards unchanged; strict geometry additionally required. No new engineering tolerance.',
        cache='Exact source/field/binary/density/edge hashes and exact gsize, stiffness0, per-stage60s. Reuse is not independent evidence.',
        scope='No family-heldout evidence and no presumption that main method beats one-shot proposal controls. Full matrix retained.'))
    shutil.copy2(Path(__file__),D/'run_coordinator_v5.py')
    hashes={str(f):sha(f) for f in D.rglob('*') if f.is_file()};hashes.update(read(M/'frozen_hashes.json'))
    for p in plans:
        for path in list(map(Path,static.records(p)))+extra_records(p):
            hashes[str(path)]=sha(path);r=read(path);q=Path(r['quad_path']);b=q.parent/'backend_run.json'
            hashes[str(b)]=sha(b)
            if q.exists():hashes[str(q)]=sha(q)
    hashes[str(static.geom.BIN)]=sha(static.geom.BIN);save(D/'frozen_hashes.json',hashes)
    print('V5_FROZEN',len(plans),len(VARIANTS),len(hashes),flush=True)

def run():
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    sys.path.insert(0,str(D/'method'));from coordinator_v5 import CoordinatorV5
    out=D/'runs';out.mkdir(exist_ok=False);summary=[];geomcache={}
    for p in read(D/'plans.json'):
        cache=cache_for(p)
        for variant in VARIANTS:
            c=CoordinatorV5(p,out/p['model']/variant,static.BINS/'miq_adaptive',static.BINS/'qex_adapter',static.geom.BIN,variant,cache,geomcache)
            assert c.selection['selected_edges']==read(M/'runs'/p['model']/'full/initial_proposal.json')['selection']['selected_edges']
            decision=c.run();summary.append(dict(model=p['model'],**read(c.out/'complete.json')));save(out/'summary.json',summary)
            print('V5_DONE',p['model'],variant,decision['recommended_id'],c.native,c.reused,flush=True)
        assert all(sha(path)==h for path,h in read(D/'frozen_hashes.json').items())
    save(out/'integrity.json',dict(models=12,variants=len(VARIANTS),native_calls=sum(r['native_calls'] for r in summary),reused_attempts=sum(r['reused_attempts'] for r in summary),
        logical_attempts=sum(r['attempts'] for r in summary),complete=True,frozen_unchanged=True))

if __name__=='__main__':globals()[sys.argv[1]]()
