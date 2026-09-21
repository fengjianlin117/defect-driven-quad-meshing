"""Frozen count-matched static controls testing whether coordination is necessary."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,math,copy,shutil
import numpy as np
from audit_evidence import ROOT,OUT,read,save,sha
D=OUT/'strong_static_controls_v1';M=OUT/'coordinator_v4_development'
sys.path.insert(0,str(M/'method'))
from coordinator import Coordinator
from coordination import common_metrics,loss,key
from defect_driven_selection import needs_repair
from weak_layout_pipeline.backend.runner import run_backend
from weak_layout_pipeline.pipeline.mesh import load_obj
from neutral_evaluation import evaluate
from candidate_decision import count_correction_target
import audit_intersections as geom
geom.BIN=OUT/'intersection_soup_audit_v2/checker'
BINS=ROOT/'research_runs/2026-09-19_selection_v3_tools/build/bin'

def records(p):
    paths=list(p['verified_cache_records'])+list(map(str,(M/'runs'/p['model']).glob('*/*/result.json')))
    return sorted(set(paths))

def signature(p,edges,density,g):
    return tuple(sha(p[k]) for k in ['source','pd1','pd2'])+(sha(BINS/'miq_adaptive'),sha(BINS/'qex_adapter'),sha(density),sha(edges),float(g).hex(),0,60)

def historic(p):
    cache={}
    for path in records(p):
        r=read(path);b=read(Path(r['quad_path']).parent/'backend_run.json');a=b['artifacts']
        prefix=tuple(sha(p[k]) for k in ['source','pd1','pd2'])+(sha(BINS/'miq_adaptive'),sha(BINS/'qex_adapter'))
        if tuple(a[k]['sha256'] for k in ['mesh','pd1','pd2','miq_executable','qex_executable'])!=prefix or b['stiffness_iterations']!=0:continue
        sig=prefix+(a['density']['sha256'],a['hard_edges']['sha256'],float(b['gsize']).hex(),0,60)
        cache[sig]=dict(record=r,path=path,gsize=b['gsize'])
    return cache

def freeze():
    assert all(sha(p)==h for p,h in read(M/'frozen_hashes.json').items())
    D.mkdir(exist_ok=False);plans=[]
    recommendations={r['model']:r for r in read(OUT/'checkpoint_03/recommendations.json') if r['variant']=='full'}
    for p in read(M/'plans.json'):
        folder=D/p['model'];folder.mkdir();c=Coordinator(p,folder/'proposal_only',BINS/'miq_adaptive',BINS/'qex_adapter')
        rows={key(r):r for r in c.base['edge_defects'] if r['layer']=='main'};chosen=[];selected=set();trace=[]
        order=sorted((g for g,es in c.groups.items() if any(needs_repair(rows[e]) for e in es)),
            key=lambda g:(-sum(loss(rows[e]) for e in c.groups[g])/max(sum(rows[e]['length_h'] for e in c.groups[g]),1e-30),g))
        for g in order:
            audit=c.audit.check(selected|c.groups[g]);trace.append(dict(group=g,admissible=audit['admissible'],audit=audit))
            if audit['admissible']:chosen.append(g);selected|=c.groups[g]
        save(folder/'simple_selection.json',dict(order=order,selected_groups=chosen,trace=trace,
            rule='Whole groups with at least one diagnostically deficient edge, descending length-weighted mean deficit, greedy equality compatibility. No quota or percentage.'))
        selections={'none':[],'all':list(c.groups),'simple_defect':chosen};cache=historic(p)
        for name,groups in selections.items():
            ep=folder/(name+'.edges');es=sorted(set().union(*(c.groups[g] for g in groups)));np.savetxt(ep,np.asarray(es,int).reshape(-1,2),fmt='%d')
            for kind in ['uniform','allocated']:
                density=p['density'][kind];target=recommendations[p['model']]['quads']
                anchors=[v for sig,v in cache.items() if sig[5]==sha(density) and sig[6]==sha(ep) and v['record'].get('quads')]
                anchor=min(anchors,key=lambda a:(abs(math.log(a['record']['quads']/target)),a['path'])) if anchors else None
                g=anchor['gsize']*math.sqrt(target/anchor['record']['quads']) if anchor else p['gsize']*math.sqrt(target/p['target_quads'])
                plans.append(dict(model=p['model'],condition=name,size=kind,source=p['source'],pd1=p['pd1'],pd2=p['pd2'],graph=p['graph'],baseline_record=p['baseline_record'],
                    reference_h=p['reference_h'],target_quads=target,gsize=g,edges=str(ep),density=density,
                    anchor_record=anchor['path'] if anchor else None,coordinator_record=recommendations[p['model']]['record'],
                    initial_selected_edges=[list(e) for e in sorted(c.initial_selected)]))
    save(D/'plans.json',plans)
    save(D/'protocol.json',dict(question='Does bounded measured coordination improve upon strong static choices at comparable actual output counts?',
        data='All twelve previously seen development models; no exclusions, no heldout claim.',
        coordinator='Frozen v4 output for each model, including baseline fallback and the invalid B31 recommendation; no v5 adaptation during this study.',
        controls='none/all/simple_defect whole-group constraints x uniform/original source-wide allocated sizes; simple method has no quota and keeps the same equality compatibility screen.',
        target='Every model uses its frozen v4 recommended actual face count, including failures. Nominal targets do not imply matching.',
        limits='One initial call and at most one inherited count correction per cell, 60s each backend stage, one thread, stiffness=0, max actual Q=2048. All outputs retained.',
        evaluation='Identical full source references, samples, diagnostic thresholds, basic checks and both CGAL triangle-soup triangulations. Report newly deficient edges, positive edge regressions and CAD proxies. No method-specific unselected-set gate determines the comparison.',
        primary='Within each static cell choose the strict-valid <=2048 output closest in actual Q to target; ties use earliest attempt. Call count-matched only within 5%. Compare deficit and RMS separately; report both-improved/both-worse/tradeoff/unmatched/invalid. Do not pick whichever of six cells makes the coordinator look best.',
        secondary='Report every static condition separately and the static quality/count envelope. Static methods use at most 2 calls versus coordinator at most 12; include this cost advantage.',
        max_native_calls=144,seen_data_hypothesis_test=True))
    shutil.copy2(Path(__file__),D/'strong_static_controls.py')
    hashes={str(f):sha(f) for f in D.rglob('*') if f.is_file()}
    for p in plans:
        for k in ['source','pd1','pd2','graph','baseline_record','density','coordinator_record']:hashes[p[k]]=sha(p[k])
    for p in read(M/'plans.json'):
        for path in records(p):
            hashes[path]=sha(path);r=read(path);b=Path(r['quad_path']).parent/'backend_run.json';hashes[str(b)]=sha(b)
            if Path(r['quad_path']).exists():hashes[r['quad_path']]=sha(r['quad_path'])
    for f in [BINS/'miq_adaptive',BINS/'qex_adapter',geom.BIN]:hashes[str(f)]=sha(f)
    hashes.update(read(M/'frozen_hashes.json'));save(D/'frozen_hashes.json',hashes)
    print('STRONG_STATIC_FROZEN',len(plans),len(hashes),flush=True)

def run():
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    out=D/'runs';out.mkdir(exist_ok=False);rows=[];native=0;reused=0;geocache={}
    original={p['model']:p for p in read(M/'plans.json')};caches={name:historic(p) for name,p in original.items()}
    for p in read(D/'plans.json'):
        cache=caches[p['model']];mesh=load_obj(p['source']);graph=read(p['graph']);base=read(p['baseline_record']);g=p['gsize'];selected={tuple(e) for e in p['initial_selected_edges']}
        for index in range(2):
            name=f"{p['model']}__{p['condition']}__{p['size']}_{index}";folder=out/name;folder.mkdir();sig=signature(p,p['edges'],p['density'],g)
            if sig in cache:r=copy.deepcopy(cache[sig]['record']);reused+=1;r.update(reused=True,source_record=cache[sig]['path'])
            else:
                native+=1;assert native<=144
                b=run_backend(mesh=Path(p['source']),pd1=Path(p['pd1']),pd2=Path(p['pd2']),density=Path(p['density']),hard_edges=Path(p['edges']),gsize=g,
                    miq_executable=BINS/'miq_adaptive',qex_executable=BINS/'qex_adapter',output_dir=folder/'backend',stiffness_iterations=0,timeout_seconds=60)
                r=dict(process_success=b['success'],reused=False,quad_path=str(folder/'backend/quad.obj'),miq_seconds=b['miq']['elapsed_seconds'],qex_seconds=b['qex'].get('elapsed_seconds'))
                if b['success']:
                    try:r.update(evaluate(mesh,load_obj(r['quad_path'],require_triangles=False),graph,p['reference_h']))
                    except Exception as exc:r.update(process_success=False,evaluation_error=repr(exc))
            r.update(id=name,model=p['model'],condition=p['condition'],size=p['size'],attempt=index,gsize=g,target_quads=p['target_quads'],edges=p['edges'],density=p['density'])
            r.update(common_metrics(r,base,selected));q=Path(r['quad_path']);checks=[]
            if q.is_file():
                h=sha(q)
                if h not in geocache:geocache[h]=[geom.check(q,d) for d in [0,1]]
                checks=geocache[h]
            clear=len(checks)==2 and all(x.get('completed') is True and x.get('intersection_pair_count')==0 and x.get('exact_degenerate_triangles')==0 for x in checks)
            save(folder/'intersection.json',checks);r.update(strict_output_pass=r.get('process_success') is True and r.get('basic_output_pass') is True and clear,intersection_clear=clear)
            save(folder/'result.json',r);cache[sig]=dict(record=r,path=str(folder/'result.json'),gsize=g)
            rows.append({k:v for k,v in r.items() if k not in ['edge_defects','quality','review','topology','uv']});save(out/'summary.json',rows)
            print('STATIC',name,r.get('quads'),r['strict_output_pass'],r.get('all_main_deficit'),'reuse',r['reused'],flush=True)
            corrected=count_correction_target(r['quads'],p['target_quads'],2048) if r.get('quads') else None
            if index or corrected is None:break
            g*=math.sqrt(corrected/r['quads'])
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    save(out/'integrity.json',dict(native_calls=native,reused_attempts=reused,logical_attempts=len(rows),frozen_unchanged=True,complete=True))

if __name__=='__main__':globals()[sys.argv[1]]()
