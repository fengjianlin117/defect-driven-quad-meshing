"""Development controls for all three feedback-pilot models, not only success."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse, importlib.util, math, shutil, time
from pathlib import Path
import numpy as np
from audit_evidence import ROOT, OLD, OUT, read, save, sha
import feedback_pilot as pilot
DEST=OUT/'feedback_matched_controls_v1'

def freeze():
    DEST.mkdir(exist_ok=False);plans=[]
    results=read(pilot.PILOT/'runs/summary.json')
    for p in read(pilot.PILOT/'plans.json'):
        target=next(r['quads'] for r in results if r['id']==p['model']+'__same_budget_1')
        for kind in ['uniform','allocated']:
            parent=read(OLD/f'defect_driven_v3/development/{p["model"]}/defect_driven__{kind}_0/result.json')
            plans.append(dict(model=p['model'],kind=kind,target_quads=target,
                gsize=p['gsize']*math.sqrt(target/parent['quads']),source=p['source'],pd1=p['pd1'],pd2=p['pd2'],
                reference_h=p['reference_h'],graph=p['graph'],baseline_record=p['baseline_record'],
                density=p['density'][kind],hard_edges=p['methods']['defect_driven']['edges']))
    save(DEST/'protocol.json',dict(role='post-pilot seen development control, no heldout inference',
        targets='For every pilot model use the face count of its final count-corrected feedback output, including failures.',
        conditions='Frozen v3 uniform and source-geometry allocated size; same feature edges, fields, output evaluator and samples.',
        count_calibration='Initialize gsize by sqrt(target/old actual), at most one inherited 5% count correction.',
        max_native_calls=12,per_stage_timeout_seconds=60,threads=1,stiffening=0,max_quads=2048,
        interpretation='Report all actual counts and full errors; a match requires <=5% actual count difference. No equality assumed from nominal target.'))
    save(DEST/'plans.json',plans)
    shutil.copy2(Path(__file__),DEST/'matched_controls.py')
    hashes={str(p):sha(p) for p in DEST.iterdir() if p.is_file()}
    for p in plans:
        for k in ['source','pd1','pd2','graph','baseline_record','density','hard_edges']:hashes[p[k]]=sha(p[k])
    save(DEST/'frozen_hashes.json',hashes)

def run():
    assert all(sha(p)==h for p,h in read(DEST/'frozen_hashes.json').items());pilot.verify();pilot.modules()
    d=DEST/'runs';d.mkdir(exist_ok=False);rows=[];calls=0
    bins=ROOT/'research_runs/2026-09-19_selection_v3_tools/build/bin'
    for p in read(DEST/'plans.json'):
        mesh=pilot.load_obj(p['source']);graph=read(p['graph']);base=read(p['baseline_record']);h=p['reference_h']
        all_edges={tuple(sorted(r['vertices'])) for r in base['edge_defects'] if r['layer']=='main'}
        selected={tuple(e) for e in np.loadtxt(p['hard_edges'],dtype=int).reshape(-1,2)};protected=all_edges-selected
        before={tuple(sorted(r['vertices'])):r for r in base['edge_defects'] if r['layer']=='main'}
        points=pilot.area_samples(mesh,4096,190919)[0];g=p['gsize']
        for index in range(2):
            case=f'{p["model"]}__{p["kind"]}_{index}';folder=d/case;folder.mkdir();calls+=1;assert calls<=12
            result=pilot.run_backend(mesh=Path(p['source']),pd1=Path(p['pd1']),pd2=Path(p['pd2']),density=Path(p['density']),
                hard_edges=Path(p['hard_edges']),gsize=g,miq_executable=bins/'miq_adaptive',qex_executable=bins/'qex_adapter',
                output_dir=folder/'backend',stiffness_iterations=0,timeout_seconds=60)
            r=dict(id=case,model=p['model'],size=p['kind'],target_quads=p['target_quads'],gsize=g,process_success=result['success'],
                quad_path=str(folder/'backend/quad.obj'),miq_seconds=result['miq']['elapsed_seconds'],qex_seconds=result['qex'].get('elapsed_seconds'))
            if result['success']:
                try:
                    r.update(pilot.evaluate(mesh,pilot.load_obj(folder/'backend/quad.obj',require_triangles=False),graph,h,folder/'backend',points))
                    r['protected_deficit']=pilot.deficit(r['edge_defects'],protected)
                    r['actual_count_within_5_percent']=abs(r['quads']/p['target_quads']-1)<=.05
                    r['common_reference_positive_regression']=sum(max(0.,x['length_h']*.5*(x['surface_loss']+x['unaligned_fraction'])-before[tuple(sorted(x['vertices']))]['length_h']*.5*(before[tuple(sorted(x['vertices']))]['surface_loss']+before[tuple(sorted(x['vertices']))]['unaligned_fraction'])) for x in r['edge_defects'] if x['layer']=='main')
                except Exception as exc:r.update(process_success=False,evaluation_error=repr(exc))
            save(folder/'result.json',r);rows.append({k:v for k,v in r.items() if k not in ['edge_defects','quality','topology','review','uv']});save(d/'summary.json',rows)
            print('CONTROL',case,r.get('quads'),r.get('basic_output_pass'),r.get('all_main_deficit'),r.get('protected_deficit'),flush=True)
            target=pilot.count_correction_target(r['quads'],p['target_quads'],2048) if r.get('quads') else None
            if index or target is None:break
            g*=math.sqrt(target/r['quads'])
    assert all(sha(p)==h for p,h in read(DEST/'frozen_hashes.json').items())
    save(d/'integrity.json',dict(native_calls=calls,complete=True,frozen_inputs_unchanged=True))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args();{'freeze':freeze,'run':run}[args.stage]()
