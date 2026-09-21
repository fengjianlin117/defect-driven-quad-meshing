"""Prespecified diagnostic grid evaluated on fixed outputs; no solver tuning."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,math,argparse,shutil
from pathlib import Path
import numpy as np
from audit_evidence import ROOT,OLD,OUT,read,save,sha
sys.path.insert(0,str(OLD/'defect_driven_v3/method'))
from weak_layout_pipeline.pipeline.mesh import load_obj,unique_edges
from weak_layout_pipeline.pipeline.sampling import surface_distances
from feature_diagnostics import segment_query
D=OUT/'diagnostic_sensitivity_v1'

def freeze():
    D.mkdir(exist_ok=False)
    sources={p['model']:p for p in read(OUT/'audit/source_plans.json')}
    cases=[]
    for row in read(OUT/'checkpoint_01/candidate_comparison.json'):
        p=sources[row['model']];r=read(row['record'])
        quad=r.get('quad_path') or str(Path(p['source']).parent/'baseline/quad.obj')
        cases.append(dict(model=row['model'],condition=row['condition'],record=row['record'],source=p['source'],graph=p['graph'],h=p['reference_h'],quad=quad))
    save(D/'protocol.json',dict(role='Seen-output measurement sensitivity only; no reruns, no parameter selection',
        distance_fractions_h=[.05,.1,.2],angles_degrees=[10.,15.,20.],samples_per_h=32,minimum_samples=4,
        condition_manifest='Exactly the fixed candidate set in checkpoint_01, including all three pilot models; no added candidate after measurements.',
        interpretation='An octave around the inherited distance resolution and +/-5 degrees are diagnostic robustness probes, not calibrated CAD tolerances.',
        outputs='Full reference deficit; length-weighted unaligned fraction; per-edge positive regression; newly deficient reference length; geometry distance percentiles.',
        nominal_check='At .1h/15deg reproduce recorded all_main_deficit within 1e-8 before interpreting other settings.'))
    save(D/'cases.json',cases);shutil.copy2(Path(__file__),D/'diagnostic_sensitivity.py')
    hashes={str(p):sha(p) for p in D.iterdir() if p.is_file()}
    for c in cases:
        for key in ['record','source','graph','quad']:hashes[c[key]]=sha(c[key])
    for p in (OLD/'defect_driven_v3/method').rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:hashes[str(p)]=sha(p)
    save(D/'frozen_hashes.json',hashes)

def measure(c):
    source=load_obj(c['source']);output=load_obj(c['quad'],require_triangles=False);graph=read(c['graph']);h=c['h']
    points=[];tangents=[];slices=[];metadata=[]
    for r in graph['source_edges']:
        if r['layer']!='main':continue
        a,b=source.vertices[r['vertices']];v=b-a;length=float(np.linalg.norm(v));n=max(4,math.ceil(length/h*32));start=len(points)
        points.extend(a+((np.arange(n)+.5)/n)[:,None]*v);tangents.extend(np.repeat((v/length)[None],n,axis=0));slices.append(slice(start,start+n));metadata.append(r)
    p=np.asarray(points);t=np.asarray(tangents);surface=surface_distances(p,output)/h
    aligned={angle:segment_query(p,output.vertices[unique_edges(output)],t,angle)[2]/h for angle in [10.,15.,20.]}
    # Preserve raw distances so the diagnostic grid is reproducible without any backend call.
    np.savez_compressed(D/'raw'/(c['model']+'_'+c['condition']+'.npz'),surface_h=surface,aligned_10_h=aligned[10.],aligned_15_h=aligned[15.],aligned_20_h=aligned[20.],
        edge_vertices=np.asarray([r['vertices'] for r in metadata]),sample_bounds=np.asarray([[s.start,s.stop] for s in slices]))
    rows=[]
    for threshold in [.05,.1,.2]:
        for angle in [10.,15.,20.]:
            edges=[]
            for r,sl in zip(metadata,slices):
                missing=float((aligned[angle][sl]>threshold).mean());sloss=float(np.minimum(surface[sl]/threshold,1).mean());cost=r['length']/h
                p95=float(np.percentile(surface[sl],95));edges.append(dict(vertices=r['vertices'],length_h=cost,deficit=cost*.5*(sloss+missing),
                    unaligned=missing,deficient=missing>0 or p95>threshold,surface_p95_h=p95))
            row=dict(model=c['model'],condition=c['condition'],distance_h=threshold,angle=angle,edges=edges,
                all_main_deficit=sum(x['deficit'] for x in edges),
                unaligned_length_fraction=sum(x['length_h']*x['unaligned'] for x in edges)/sum(x['length_h'] for x in edges))
            if threshold==.1 and angle==15.:
                expected=read(c['record'])['all_main_deficit'];assert abs(expected-row['all_main_deficit'])<1e-8,(c,expected,row['all_main_deficit'])
            rows.append(row)
    return rows

def run():
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items());(D/'raw').mkdir(exist_ok=False);(D/'measurements').mkdir(exist_ok=False)
    all_rows=[]
    for c in read(D/'cases.json'):
        rows=measure(c);save(D/'measurements'/(c['model']+'_'+c['condition']+'.json'),rows);all_rows.extend(rows)
        print('SENSITIVITY',c['model'],c['condition'],'nominal_reproduced',flush=True)
    bases={(r['model'],r['distance_h'],r['angle']):r for r in all_rows if r['condition']=='B'}
    for r in all_rows:
        b={tuple(x['vertices']):x for x in bases[r['model'],r['distance_h'],r['angle']]['edges']}
        r['positive_regression']=sum(max(0.,x['deficit']-b[tuple(x['vertices'])]['deficit']) for x in r['edges'])
        r['newly_deficient_length_h']=sum(x['length_h'] for x in r['edges'] if x['deficient'] and not b[tuple(x['vertices'])]['deficient'])
        r['newly_deficient_edge_count']=sum(x['deficient'] and not b[tuple(x['vertices'])]['deficient'] for x in r['edges'])
    summary=[{k:v for k,v in r.items() if k!='edges'} for r in all_rows];save(D/'summary.json',summary)
    pairs=[]
    for model in ['B49','B53','B65']:
        for feedback in ['S','C']:
            if not any(r['model']==model and r['condition']==feedback for r in summary):continue
            grid=[]
            for threshold in [.05,.1,.2]:
                for angle in [10.,15.,20.]:
                    get=lambda cond:next(r for r in summary if r['model']==model and r['condition']==cond and r['distance_h']==threshold and r['angle']==angle)
                    a,s=get('A'),get(feedback)
                    grid.append(dict(distance_h=threshold,angle=angle,feedback_deficit=s['all_main_deficit'],allocated_control_deficit=a['all_main_deficit'],
                        feedback_lower_deficit=s['all_main_deficit']<a['all_main_deficit'],feedback_newly_deficient_edges=s['newly_deficient_edge_count'],
                        feedback_positive_regression=s['positive_regression']))
            pairs.append(dict(model=model,feedback=feedback,lower_deficit_settings=sum(x['feedback_lower_deficit'] for x in grid),settings=9,grid=grid))
    save(D/'paired_summary.json',pairs)
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    save(D/'integrity.json',dict(cases=len(read(D/'cases.json')),settings_per_case=9,nominal_metrics_reproduced=True,frozen_unchanged=True,new_backend_calls=0,
        note='No inference that alternative thresholds validate CAD tolerances; no tuning or output replacement performed.'))
    print('SENSITIVITY_COMPLETE',[(x['model'],x['feedback'],x['lower_deficit_settings']) for x in pairs],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args();{'freeze':freeze,'run':run}[args.stage]()
