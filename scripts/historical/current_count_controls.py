"""Frozen latest-output-count controls, with all fixed conditions retained."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import copy, math, shutil, subprocess, signal, time, sys
from audit_evidence import OUT, read, save, sha
import strong_static_controls as static

D = OUT/'current_count_controls_v1'
Q = OUT/'external/QuadriFlow'
OLD_STATIC = OUT/'strong_static_controls_v1'
V5 = OUT/'coordinator_v5_development'
original_records = static.records

def records(p):
    paths = original_records(p)
    paths += [str(f) for f in (OLD_STATIC/'runs').glob(p['model']+'__*/result.json')]
    paths += [str(f) for f in (V5/'runs'/p['model']).glob('*/*/result.json')]
    return sorted(set(paths))

static.records = records

def freeze():
    assert read(V5/'runs/integrity.json')['complete']
    D.mkdir(exist_ok=False)
    refs = {r['model']:r for r in read(OUT/'coordinator_v5_report_v1/recommendations.json') if r['variant']=='full'}
    original = {p['model']:p for p in read(static.M/'plans.json')}
    caches = {name:static.historic(p) for name,p in original.items()}
    plans = []
    for old in read(OLD_STATIC/'plans.json'):
        p = copy.deepcopy(old); target = refs[p['model']]['quads']
        anchors = [v for sig,v in caches[p['model']].items()
                   if sig[5]==sha(p['density']) and sig[6]==sha(p['edges']) and v['record'].get('quads')]
        anchor = min(anchors,key=lambda a:(abs(math.log(a['record']['quads']/target)),a['path'])) if anchors else None
        p.update(target_quads=target, gsize=anchor['gsize']*math.sqrt(target/anchor['record']['quads']) if anchor else p['gsize']*math.sqrt(target/p['target_quads']),
                 anchor_record=anchor['path'] if anchor else None, coordinator_record=refs[p['model']]['decision_path'])
        plans.append(p)
    save(D/'plans.json',plans); save(D/'references.json',refs)
    external_plans = [dict(p,target_quads=refs[p['model']]['quads']) for p in original.values()]
    save(D/'external_plans.json',external_plans)
    save(D/'protocol.json',dict(
        role='Seen development comparison only; final method and family-heldout protocol still pending.',
        question='Quality relative to unconstrained same-field backend and official QuadriFlow at current recommended actual counts.',
        static_conditions='none/all/one-shot defect-selected constraints x uniform/source-allocated sizes; identical source, fixed field and binaries.',
        external_variants={'default':[],'sharp':['-sharp'],'sharp_sat':['-sharp','-sat']},
        primary='Keep all 12 models and every condition. Pick strict-valid output closest to target actual Q within each condition, ties earliest. Compare D and RMS only when pair actual counts differ <=5%. Baseline fallback is flagged and not attributed to our modules.',
        secondary='Report all attempts as quality/count points; one-shot vs feedback is an ablation, not a prerequisite for the primary contribution.',
        correction='At most one count correction per cell; static inherited correction, external proportional requested_faces*target/actual. Never tune quality or select a different configuration after results.',
        max_static_calls=144,max_external_calls=72,static_stage_seconds=60,external_attempt_seconds=120,
        actual_cap=2048,seed=3627473,threads=1,
        validity='Same basic checks and exact triangle-soup intersection/degeneracy check under both diagonals. No method-specific protection gate for comparative quality.',
        cost='Static and external max2 attempts, feedback max12. External runtime includes own field; static time conditional on supplied field. Cached calls count logically but not as new independent runs.',
        cache='Exact inputs, binaries, flags, seed and numeric controls only; every reused result linked. Evaluations retain full common source reference.'))
    for f in [Path(__file__),Path(static.__file__)]:shutil.copy2(f,D/f.name)
    hashes = read(V5/'frozen_hashes.json')
    for f in D.iterdir():
        if f.is_file():hashes[str(f)]=sha(f)
    for p in original.values():
        for path in records(p):
            hashes[path]=sha(path);r=read(path);q=Path(r['quad_path']);b=q.parent/'backend_run.json'
            hashes[str(b)]=sha(b)
            if q.exists():hashes[str(q)]=sha(q)
    for f in (OUT/'quadriflow_development_v1/runs').glob('*/*'):
        if f.name in ['result.json','backend_run.json','quad.obj']:hashes[str(f)]=sha(f)
    for f in [Q/'build/quadriflow',Q/'build_sat/minisat']:
        hashes[str(f)]=sha(f)
    for r in refs.values():hashes[r['decision_path']]=sha(r['decision_path'])
    save(D/'frozen_hashes.json',hashes)
    print('CURRENT_COUNT_FROZEN',len(plans),len(external_plans)*3,len(hashes),flush=True)

def static_run():
    static.D=D
    static.run()

def ext_signature(source, variant, requested, seed):
    return (sha(source),sha(Q/'build/quadriflow'),sha(Q/'build_sat/minisat'),variant,requested,seed,1,120)

def external_run():
    hashes=read(D/'frozen_hashes.json');assert all(sha(p)==h for p,h in hashes.items())
    protocol=read(D/'protocol.json');out=D/'external_runs';out.mkdir(exist_ok=False)
    cache={};geocache={};rows=[];native=0;reused=0
    for path in (OUT/'quadriflow_development_v1/runs').glob('*/result.json'):
        r=read(path);b=read(path.parent/'backend_run.json')
        sig=(b['source_sha256'],b['binary_sha256'],b['sat_binary_sha256'],r['variant'],r['requested_faces'],b['seed'],b['threads'],120)
        cache[sig]=(r,str(path))
    env=dict(os.environ,PATH=str(Q/'build_sat')+os.pathsep+os.environ['PATH'])
    for p in read(D/'external_plans.json'):
        mesh=static.load_obj(p['source']);graph=read(p['graph']);base=read(p['baseline_record'])
        selected={tuple(e) for e in read(V5/'runs'/p['model']/'full/initial_proposal.json')['selection']['selected_edges']}
        for variant,flags in protocol['external_variants'].items():
            requested=p['target_quads']
            for index in range(2):
                name=f"{p['model']}__{variant}_{index}";folder=out/name;folder.mkdir()
                sig=ext_signature(p['source'],variant,requested,protocol['seed'])
                if sig in cache:
                    r=copy.deepcopy(cache[sig][0]);r.update(reused=True,source_record=cache[sig][1]);reused+=1
                else:
                    native+=1;assert native<=72
                    argv=[str(Q/'build/quadriflow'),'-i',p['source'],'-o',str(folder/'quad.obj'),'-f',str(requested),'-seed',str(protocol['seed']),*flags]
                    start=time.perf_counter();timed_out=False
                    with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
                        process=subprocess.Popen(argv,cwd=folder,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                        try:code=process.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            timed_out=True;os.killpg(process.pid,signal.SIGKILL);process.wait();code=124
                    elapsed=time.perf_counter()-start
                    save(folder/'backend_run.json',dict(argv=argv,cwd=str(folder),returncode=code,timed_out=timed_out,elapsed_seconds=elapsed,source_sha256=sig[0],binary_sha256=sig[1],sat_binary_sha256=sig[2],seed=protocol['seed'],threads=1))
                    r=dict(process_success=code==0 and (folder/'quad.obj').is_file(),quad_path=str(folder/'quad.obj'),returncode=code,timed_out=timed_out,elapsed_seconds=elapsed,reused=False)
                    if r['process_success']:
                        try:r.update(static.evaluate(mesh,static.load_obj(r['quad_path'],require_triangles=False),graph,p['reference_h']))
                        except Exception as exc:r.update(process_success=False,evaluation_error=repr(exc))
                r.update(id=name,model=p['model'],variant=variant,attempt=index,target_quads=p['target_quads'],requested_faces=requested)
                r.update(static.common_metrics(r,base,selected));q=Path(r['quad_path']);checks=[]
                if q.is_file():
                    h=sha(q)
                    if h not in geocache:geocache[h]=[static.geom.check(q,d) for d in [0,1]]
                    checks=geocache[h]
                clear=len(checks)==2 and all(x.get('completed') is True and x.get('intersection_pair_count')==0 and x.get('exact_degenerate_triangles')==0 for x in checks)
                save(folder/'intersection.json',checks)
                r.update(strict_output_pass=r.get('process_success') is True and r.get('basic_output_pass') is True and clear,intersection_clear=clear)
                save(folder/'result.json',r);cache[sig]=(r,str(folder/'result.json'))
                rows.append({k:v for k,v in r.items() if k not in ['edge_defects','quality','review','topology','uv']});save(out/'summary.json',rows)
                print('CURRENT_EXTERNAL',name,r.get('quads'),r['strict_output_pass'],r.get('all_main_deficit'),'reuse',r['reused'],flush=True)
                actual=r.get('quads')
                if index or not actual or abs(actual/p['target_quads']-1)<=.05:break
                requested=min(2048,max(1,round(requested*p['target_quads']/actual)))
    assert all(sha(p)==h for p,h in hashes.items())
    save(out/'integrity.json',dict(complete=True,frozen_unchanged=True,native_calls=native,reused_attempts=reused,logical_attempts=len(rows)))

if __name__=='__main__':globals()[sys.argv[1]]()
