"""Frozen external-method development matrix with explicit solver variants."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys, argparse, subprocess, signal, time, math, shutil
from audit_evidence import ROOT, OLD, OUT, read, save, sha
from common_evaluation import evaluate
from weak_layout_pipeline.pipeline.mesh import load_obj

Q=OUT/'external/QuadriFlow';D=OUT/'quadriflow_development_v1'
def freeze():
    D.mkdir(exist_ok=False)
    # Check backend-neutral metrics on a real frozen output before the new matrix.
    p=read(OUT/'audit/source_plans.json')[0]
    old=read(OLD/'defect_driven_v3/development/B49/defect_driven__allocated_0/result.json')
    r=evaluate(load_obj(p['source']),load_obj(old['quad_path'],require_triangles=False),read(p['graph']),p['reference_h'])
    equal={k:abs(r[k]-old[k])<1e-12 for k in ['all_main_deficit','symmetric_rms_h','source_p95_h','output_p95_h']}
    assert all(equal.values()) and r['basic_output_pass']==old['basic_output_pass']
    save(D/'evaluator_equivalence.json',dict(source_record=str(OLD/'defect_driven_v3/development/B49/defect_driven__allocated_0/result.json'),metrics_equal=equal,basic_checks_equal=True))
    protocol=dict(role='all 12 seen development inputs; no generalization estimate',
        variants={'default':[],'sharp':['-sharp'],'sharp_sat':['-sharp','-sat']},
        seed=3627473,timeout_total_seconds_per_attempt=120,threads=1,max_attempts_per_variant=2,max_attempts=72,
        target='Same source-geometry nominal targets as frozen v3.',
        count_correction='At most one desired-faces adjustment round using target/actual if error >5%; stop on process failure, retain invalid output.',
        target_cap=2048,actual_cap=2048,preprocessing='None outside unmodified official algorithm; same triangle input.',
        cost='Record complete external runtime including its field computation; MIQ-stage runtime is conditional on cached NeurCross fields and not end-to-end comparable.',
        timeout_comparison='120s is the sum of two historical 60s backend stage caps; this does not equalize internal algorithms.',
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Q,text=True).strip(),
        build='Release, BUILD_OPENMP=OFF, BUILD_TBB=OFF, BUILD_LOG=ON; bundled MapleCOMSPS_LRB minisat in local PATH; no system install.')
    save(D/'protocol.json',protocol)
    plans=read(OUT/'audit/source_plans.json');save(D/'plans.json',plans)
    for name in ['external_quadriflow.py','common_evaluation.py','audit_evidence.py']:shutil.copy2(Path(__file__).parent/name,D/name)
    hashes={str(x):sha(x) for x in D.iterdir() if x.is_file()}
    for p in plans:
        for key in ['source','graph','baseline_record']:hashes[p[key]]=sha(p[key])
    for binary in [Q/'build/quadriflow',Q/'build_sat/minisat']:hashes[str(binary)]=sha(binary)
    for x in (OLD/'defect_driven_v3/method').rglob('*'):
        if x.is_file() and '__pycache__' not in x.parts:hashes[str(x)]=sha(x)
    save(D/'frozen_hashes.json',hashes)

def run():
    hashes=read(D/'frozen_hashes.json');assert all(sha(p)==h for p,h in hashes.items())
    protocol=read(D/'protocol.json');out=D/'runs';out.mkdir(exist_ok=False);rows=[];calls=0
    env=dict(os.environ,PATH=str(Q/'build_sat')+os.pathsep+os.environ['PATH'])
    for p in read(D/'plans.json'):
        mesh=load_obj(p['source']);graph=read(p['graph']);base=read(p['baseline_record'])
        br={tuple(sorted(x['vertices'])):x for x in base['edge_defects'] if x['layer']=='main'}
        for kind,flags in protocol['variants'].items():
            requested=p['target_quads']
            for index in range(2):
                case=f'{p["model"]}__{kind}_{index}';dest=out/case;dest.mkdir();calls+=1;assert calls<=72
                argv=[str(Q/'build/quadriflow'),'-i',p['source'],'-o',str(dest/'quad.obj'),'-f',str(requested),'-seed',str(protocol['seed']),*flags]
                started=time.perf_counter();timed_out=False
                with (dest/'stdout.log').open('wb') as stdout,(dest/'stderr.log').open('wb') as stderr:
                    process=subprocess.Popen(argv,cwd=dest,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                    try:code=process.wait(timeout=protocol['timeout_total_seconds_per_attempt'])
                    except subprocess.TimeoutExpired:
                        timed_out=True;os.killpg(process.pid,signal.SIGKILL);process.wait();code=124
                elapsed=time.perf_counter()-started
                save(dest/'backend_run.json',dict(argv=argv,cwd=str(dest),returncode=code,timed_out=timed_out,elapsed_seconds=elapsed,
                    source_sha256=sha(p['source']),binary_sha256=sha(Q/'build/quadriflow'),sat_binary_sha256=sha(Q/'build_sat/minisat'),seed=protocol['seed'],threads=1))
                r=dict(id=case,model=p['model'],variant=kind,target_quads=p['target_quads'],requested_faces=requested,
                    elapsed_seconds=elapsed,process_success=code==0 and (dest/'quad.obj').exists(),returncode=code,timed_out=timed_out,quad_path=str(dest/'quad.obj'))
                if r['process_success']:
                    try:
                        r.update(evaluate(mesh,load_obj(dest/'quad.obj',require_triangles=False),graph,p['reference_h']))
                        r['within_count_cap']=r['quads']<=2048
                        r['common_reference_positive_regression']=sum(max(0.,x['length_h']*.5*(x['surface_loss']+x['unaligned_fraction'])-br[tuple(sorted(x['vertices']))]['length_h']*.5*(br[tuple(sorted(x['vertices']))]['surface_loss']+br[tuple(sorted(x['vertices']))]['unaligned_fraction'])) for x in r['edge_defects'] if x['layer']=='main')
                    except Exception as exc:r.update(process_success=False,evaluation_error=repr(exc))
                save(dest/'result.json',r);rows.append({k:v for k,v in r.items() if k not in ['edge_defects','quality','topology','review']});save(out/'summary.json',rows)
                print('EXTERNAL',case,r.get('quads'),r.get('basic_output_pass'),r.get('all_main_deficit'),round(elapsed,2),flush=True)
                actual=r.get('quads')
                if index or not actual or abs(actual/p['target_quads']-1)<=.05:break
                requested=min(2048,max(1,round(requested*p['target_quads']/actual)))
    assert all(sha(p)==h for p,h in hashes.items())
    save(out/'integrity.json',dict(native_calls=calls,complete=True,frozen_inputs_unchanged=True))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args();{'freeze':freeze,'run':run}[args.stage]()
