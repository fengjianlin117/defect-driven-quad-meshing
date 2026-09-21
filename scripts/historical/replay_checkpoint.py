"""Replay one new method result and one external result into fresh directories."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,subprocess,time,signal,shutil
from audit_evidence import ROOT,OLD,OUT,read,save,sha
sys.path.insert(0,str(OLD/'defect_driven_v3/method'))
from weak_layout_pipeline.backend.runner import run_backend
D=OUT/'checkpoint_replay_v1'
def main():
    D.mkdir(exist_ok=False)
    own=OUT/'feedback_pilot_v1/runs/B49__same_budget_1/backend'
    external=OUT/'quadriflow_development_v1/runs/B32__sharp_0'
    save(D/'protocol.json',dict(cases=[str(own/'backend_run.json'),str(external/'backend_run.json')],
        purpose='Two exact-input reproducibility checks, not independent models or stability across seeds.',max_backend_attempts=2))
    old=read(own/'backend_run.json');art=old['artifacts']
    for a in art.values():assert sha(a['path'])==a['sha256']
    start=time.perf_counter()
    result=run_backend(mesh=Path(art['mesh']['path']),pd1=Path(art['pd1']['path']),pd2=Path(art['pd2']['path']),
        density=Path(art['density']['path']),hard_edges=Path(art['hard_edges']['path']),gsize=old['gsize'],
        miq_executable=Path(art['miq_executable']['path']),qex_executable=Path(art['qex_executable']['path']),
        output_dir=D/'B49_feedback',stiffness_iterations=old['stiffness_iterations'],timeout_seconds=60)
    first=dict(case='B49_feedback',success=result['success'],elapsed_seconds=time.perf_counter()-start,
        quad_byte_equal=result['success'] and sha(D/'B49_feedback/quad.obj')==sha(own/'quad.obj'),original_record=str(own/'backend_run.json'))
    save(D/'B49_replay.json',first)
    old=read(external/'backend_run.json');argv=old['argv'].copy()
    assert sha(argv[0])==old['binary_sha256'] and sha(argv[argv.index('-i')+1])==old['source_sha256']
    folder=D/'B32_external';folder.mkdir();argv[argv.index('-o')+1]=str(folder/'quad.obj')
    env=dict(os.environ,PATH=str(OUT/'external/QuadriFlow/build_sat')+os.pathsep+os.environ['PATH'])
    with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
        started=time.perf_counter();process=subprocess.Popen(argv,cwd=folder,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=process.wait(timeout=120)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();code=124
    success=code==0 and (folder/'quad.obj').exists()
    second=dict(case='B32_external',success=success,returncode=code,argv=argv,elapsed_seconds=time.perf_counter()-started,
        quad_byte_equal=success and sha(folder/'quad.obj')==sha(external/'quad.obj'),original_record=str(external/'backend_run.json'))
    save(D/'B32_replay.json',second);save(D/'summary.json',[first,second])
    shutil.copy2(Path(__file__),D/'replay_checkpoint.py')
    print('REPLAY',[(x['case'],x['success'],x['quad_byte_equal']) for x in [first,second]],flush=True)
if __name__=='__main__':main()
