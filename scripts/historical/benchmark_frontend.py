"""Measure actual feature/diagnosis/proposal overhead, with no backend calls."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import sys,time,json,subprocess,shutil,platform
import numpy as np
from audit_evidence import OUT,read,save,sha
D=OUT/'frontend_cost_v1';M=OUT/'coordinator_v5_development'

def freeze():
    D.mkdir(exist_ok=False)
    save(D/'plans.json',read(M/'plans.json'))
    save(D/'protocol.json',dict(question='What is the added cost of obtaining the feature graph, diagnosing an existing output, selecting constraints and allocating sizes?',
        models='All 12 seen source models; three fresh Python subprocesses per model, in fixed model order.',
        threads=1,repeats=3,backend_calls=0,
        timing='perf_counter wall time and process_time CPU, separate input loading, graph reconstruction, edge diagnosis, compatibility setup, simple no-quota constraint selection, source-wide sizing. Imports are excluded but subprocess wall time is also recorded.',
        evaluation='An additional full output evaluation is timed separately; it overlaps the edge-diagnosis work and is not added to front-end stage sum.',
        reuse='No graph, edge diagnosis or proposal cache reused inside timed stages. Existing field and baseline geometry are inputs. OS file-cache state is uncontrolled and acknowledged.',
        scope='Conditional on an available direction field and baseline output. Does not time NeurCross or extra MIQ/QEx solves. Three repeats estimate local runtime variability, not independent geometric samples.'))
    shutil.copy2(Path(__file__),D/'benchmark_frontend.py')
    hashes={str(f):sha(f) for f in D.iterdir() if f.is_file()};hashes.update(read(M/'frozen_hashes.json'))
    save(D/'frozen_hashes.json',hashes)

def worker(name,repeat):
    sys.path.insert(0,str(M/'method'))
    from weak_layout_pipeline.pipeline.mesh import load_obj
    from reference_graph import build_reference_graph
    from defect_predictor_v1 import measure_edges
    from conflict_selection import EqualityAudit
    from protection_feedback import simple_initial
    from geometry_budget import propose
    from neutral_evaluation import evaluate
    p=next(p for p in read(D/'plans.json') if p['model']==name);times={}
    def timed(label,fn):
        start=time.perf_counter();cpu=time.process_time();v=fn()
        times[label]=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu);return v
    def inputs():
        base=Path(p['source']).parent/'baseline'
        return (load_obj(p['source']),load_obj(base/'quad.obj',require_triangles=False),
            np.loadtxt(base/'miq_uv.txt',skiprows=1),np.loadtxt(base/'miq_fuv.txt',skiprows=1,dtype=int),
            np.loadtxt(base/'miq_combed_PD1.txt')[:,-3:],np.loadtxt(base/'miq_combed_PD2.txt')[:,-3:])
    mesh,base,uv,fuv,x,y=timed('load_inputs',inputs)
    graph=timed('feature_graph',lambda:build_reference_graph(mesh))
    records=timed('edge_diagnosis',lambda:measure_edges(mesh,base,graph,p['reference_h']))
    audit=timed('compatibility_setup',lambda:EqualityAudit(mesh,uv,fuv,x,y))
    def selection():
        groups={g['id']:{tuple(sorted(e['vertices'])) for e in graph['source_edges'] if e['layer']=='main' and g['id'] in e['group_ids']} for g in graph['groups'] if g['layer']=='main'}
        return simple_initial(groups,records,audit)
    selected=timed('constraint_selection',selection)
    sizing=timed('count_and_size_allocation',lambda:propose(mesh,graph,p['reference_h'],base.face_count))
    # Independent diagnostic validation of the measured objects, outside timers.
    original=read(p['graph']);before=read(p['baseline_record'])['edge_defects']
    canonical=lambda gg:sorted((tuple(e['vertices']),e['layer'],tuple(e['group_ids'])) for e in gg['source_edges'])
    assert canonical(graph)==canonical(original)
    mapped={tuple(r['vertices']):r for r in before}
    max_difference=max(abs(r[k]-mapped[tuple(r['vertices'])][k]) for r in records for k in ['surface_loss','unaligned_fraction','surface_distance_p95_h'])
    assert max_difference<1e-10,max_difference
    expected=read(OUT/'coordinator_v4_development/runs'/name/'full/initial_proposal.json')['selection']['selected_edges']
    assert selected['selected_edges']==expected
    full=timed('full_output_evaluation_separate',lambda:evaluate(mesh,base,graph,p['reference_h']))
    stages=[k for k in times if k!='full_output_evaluation_separate']
    return dict(model=name,repeat=repeat,source_triangles=mesh.face_count,baseline_quads=base.face_count,feature_edges=len(records),
        timings=times,frontend_wall_seconds=sum(times[k]['wall_seconds'] for k in stages),frontend_cpu_seconds=sum(times[k]['cpu_seconds'] for k in stages),
        graph_edges_equal=True,diagnosis_max_difference=max_difference,selection_equal=True,target_quads=sizing[2]['target_quads'],baseline_deficit=full['all_main_deficit'])

def run():
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    folder=D/'runs';folder.mkdir(exist_ok=False);rows=[]
    for p in read(D/'plans.json'):
        for repeat in range(3):
            start=time.perf_counter();r=subprocess.run([sys.executable,'-B',str(D/'benchmark_frontend.py'),'worker',p['model'],str(repeat)],capture_output=True,text=True)
            save(folder/(p['model']+'_'+str(repeat)+'_process.json'),dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,wall_seconds=time.perf_counter()-start))
            if r.returncode:raise RuntimeError(r.stderr)
            row=json.loads(r.stdout);rows.append(row);save(folder/(p['model']+'_'+str(repeat)+'.json'),row);save(folder/'summary.json',rows)
        print('FRONTEND',p['model'],[round(r['frontend_wall_seconds'],4) for r in rows if r['model']==p['model']],flush=True)
    env=dict(python=sys.version,platform=platform.platform(),cpuinfo=Path('/proc/cpuinfo').read_text(),threads={k:os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    save(D/'environment.json',env)
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    save(folder/'integrity.json',dict(models=12,repeats=3,measured_runs=len(rows),backend_calls=0,frozen_unchanged=True,complete=True))

if __name__=='__main__':
    if sys.argv[1]=='worker':print(json.dumps(worker(sys.argv[2],int(sys.argv[3])),allow_nan=False))
    else:globals()[sys.argv[1]]()
