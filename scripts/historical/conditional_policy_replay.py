"""Freeze and execute conditional policy using only exact-input historical results."""
from pathlib import Path
import sys,shutil
from audit_evidence import OUT,read,save,sha
import run_coordinator_v5 as previous
import strong_static_controls as static
D=OUT/'conditional_policy_development_v1'
V=OUT/'coordinator_v5_development'
W=Path('research://windows-delivery/paper_stage_20260920')

def freeze():
    D.mkdir(exist_ok=False)
    shutil.copytree(V/'method',D/'method',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(W/'method_conditional/conditional_coordinator.py',D/'method')
    shutil.copy2(Path(__file__),D)
    save(D/'plans.json',read(V/'plans.json'))
    save(D/'protocol.json',dict(role='User-directed conditional activation development; all twelve inputs seen.',
        intent='Existing good baseline -> retain. Defective baseline -> one-shot feature/size proposal. No accepted initial proposal -> bounded feedback.',
        unchanged='All defect/quality/protection thresholds, constraint/sizing operators, two rounds, twelve total attempts, geometry checks and max2048 count.',
        evidence='Execute new control flow against exact cached native results; zero new native solves allowed. Compare recommendation and logical call count with always-feedback v5. This is not an independent stochastic repetition.',
        anticipated_tradeoff='Early acceptance may lose later quality improvement. Report that loss and saved calls; do not assume equal quality.',
        not_final_freeze='Final selection, sensitivities and family-heldout registration remain pending.'))
    hashes=read(V/'frozen_hashes.json')
    for f in D.rglob('*'):
        if f.is_file():hashes[str(f)]=sha(f)
    for f in (V/'runs').rglob('*'):
        if f.is_file() and f.name in ['result.json','summary.json','decision.json','complete.json']:
            hashes[str(f)]=sha(f)
    save(D/'frozen_hashes.json',hashes)
    print('CONDITIONAL_FROZEN',len(hashes),flush=True)

def run():
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    sys.path.insert(0,str(D/'method'))
    from conditional_coordinator import ConditionalCoordinator
    out=D/'runs';out.mkdir(exist_ok=False);rows=[];geocache={}
    for p in read(D/'plans.json'):
        cache=previous.cache_for(p)
        prefix=tuple(sha(p[k]) for k in ['source','pd1','pd2'])+(sha(static.BINS/'miq_adaptive'),sha(static.BINS/'qex_adapter'))
        for path in (V/'runs'/p['model']).glob('*/*/result.json'):
            r=read(path);b=read(Path(r['quad_path']).parent/'backend_run.json');a=b['artifacts']
            assert tuple(a[k]['sha256'] for k in ['mesh','pd1','pd2','miq_executable','qex_executable'])==prefix
            sig=prefix+(a['density']['sha256'],a['hard_edges']['sha256'],float(b['gsize']).hex(),0,60)
            cache[sig]=dict(record=r,path=str(path))
        class ReplayOnly(ConditionalCoordinator):
            def signature(self,*args):
                sig=super().signature(*args)
                if sig not in self.cache:raise RuntimeError('Uncached input: stop before native call')
                return sig
        c=ReplayOnly(p,out/p['model'],static.BINS/'miq_adaptive',static.BINS/'qex_adapter',static.geom.BIN,cache=cache,geometry_cache=geocache)
        dec=c.run();assert c.native==0
        selected=next(r for r in c.rows if r['id']==dec['recommended_id'])
        full=read(V/'runs'/p['model']/'full/complete.json')
        prev=next(r for r in read(V/'runs'/p['model']/'full/summary.json')['candidates'] if r['id']==full['recommended'])
        if c.exit_stage=='feedback_or_bounded_fallback':
            assert selected['id']==prev['id']
            assert selected['all_main_deficit']==prev['all_main_deficit']
            assert selected['symmetric_rms_h']==prev['symmetric_rms_h']
        rows.append(dict(model=p['model'],exit_stage=c.exit_stage,selected=selected['id'],quads=selected['quads'],
            deficit=selected['all_main_deficit'],rms=selected['symmetric_rms_h'],strict=selected['strict_output_pass'],
            logical_attempts=c.attempts,always_feedback_attempts=full['attempts'],
            always_feedback_selected=prev['id'],always_feedback_quads=prev['quads'],always_feedback_deficit=prev['all_main_deficit'],always_feedback_rms=prev['symmetric_rms_h'],
            decision_path=str(c.out/'decision.json')))
        save(out/'summary.json',rows)
        print('CONDITIONAL',p['model'],c.exit_stage,selected['id'],c.attempts,full['attempts'],flush=True)
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    save(out/'integrity.json',dict(complete=True,frozen_unchanged=True,native_calls=0,logical_attempts=sum(r['logical_attempts'] for r in rows),always_feedback_attempts=sum(r['always_feedback_attempts'] for r in rows)))

if __name__=='__main__':globals()[sys.argv[1]]()
