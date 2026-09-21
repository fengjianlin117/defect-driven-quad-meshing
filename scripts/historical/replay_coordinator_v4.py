"""One pre-registered cache-free end-to-end v4 coordinator replay."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,shutil
from audit_evidence import OUT,ROOT,read,save,sha
D=OUT/'coordinator_v4_cachefree_replay';M=OUT/'coordinator_v4_development'

def main():
    assert all(sha(p)==h for p,h in read(M/'frozen_hashes.json').items())
    D.mkdir(exist_ok=False)
    p=next(p for p in read(M/'plans.json') if p['model']=='B53')
    save(D/'input.json',p)
    save(D/'protocol.json',dict(model='B53',variant='full',cache='empty; all logical requests are real calls unless identical inputs recur within this run',
        reason='B53 exercises initial proposal, count correction, constraint feedback, size feedback, failed intermediates and final protective decision.',
        expected='Same final recommendation and every geometric artifact hash; nonidentical results are reported, not suppressed.',
        seed_scope='Exact deterministic-input replay only; not independent field or random-seed stability.',max_logical_attempts=12))
    shutil.copy2(Path(__file__),D/'replay_coordinator_v4.py')
    save(D/'frozen_hashes.json',{str(f):sha(f) for f in D.iterdir() if f.is_file()})
    sys.path.insert(0,str(M/'method'));from coordinator import Coordinator
    binaries=ROOT/'research_runs/2026-09-19_selection_v3_tools/build/bin'
    c=Coordinator(p,D/'run',binaries/'miq_adaptive',binaries/'qex_adapter','full',{})
    decision=c.run();rows=[]
    before=M/'runs/B53/full'
    for path in sorted((D/'run').glob('*/result.json')):
        now=read(path);old=read(before/path.parent.name/'result.json');np=Path(now['quad_path']);op=Path(old['quad_path'])
        rows.append(dict(id=now['id'],process_same=now['process_success']==old['process_success'],
            quad_presence_same=np.is_file()==op.is_file(),quad_identical=sha(np)==sha(op) if np.is_file() and op.is_file() else None,
            metrics_same=all(now.get(k)==old.get(k) for k in ['quads','basic_output_pass','all_main_deficit','symmetric_rms_h','protected_deficit','common_positive_regression','newly_deficient_edges']),
            old_record=str(before/path.parent.name/'result.json'),new_record=str(path)))
    save(D/'comparison.json',rows)
    save(D/'integrity.json',dict(native_calls=c.native,reused_attempts=c.reused,attempts=c.attempts,
        recommendation_identical=decision['recommended_id']==read(before/'decision.json')['recommended_id'],
        all_geometric_outputs_identical=all(r['quad_presence_same'] and r['quad_identical'] is not False for r in rows),
        all_metrics_identical=all(r['metrics_same'] and r['process_same'] for r in rows),
        frozen_unchanged=all(sha(p)==h for p,h in read(M/'frozen_hashes.json').items()),independent_new_models=0))
    print('CACHEFREE_REPLAY',read(D/'integrity.json'),flush=True)

if __name__=='__main__':main()
