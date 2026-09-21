"""Bounded whole-group repair proposals from actual residual deficiencies.

Keep complete references and geometric requests. If an added group conflicts
with itself, do not waste replacement searches. Otherwise consider at most 16
single-group replacements and retain their explicit displaced demands.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys, argparse, shutil
from pathlib import Path
import numpy as np
from audit_evidence import ROOT, OLD, OUT, read, save, sha
import feedback_pilot as pilot
sys.path.insert(0,str(OLD/'defect_driven_v3/method'))
from conflict_selection import EqualityAudit
from defect_driven_selection import needs_repair
D=OUT/'constraint_feedback_pilot_v1'

def propose(group_edges,chosen,records,audit,max_screenings=64):
    deficient={tuple(sorted(r['vertices'])) for r in records if r['layer']=='main' and needs_repair(r)}
    losses={tuple(sorted(r['vertices'])):r['length_h']*.5*(r['surface_loss']+r['unaligned_fraction']) for r in records if r['layer']=='main'}
    union=lambda gs:set().union(*(group_edges[g] for g in gs))
    selected=union(chosen);candidates=[];trace=[];calls=0
    ordered=sorted([g for g in group_edges if g not in chosen and group_edges[g]&deficient],
        key=lambda g:(-sum(losses[e] for e in group_edges[g]-selected),g))
    for add in ordered:
        if calls>=max_screenings:trace.append(dict(add=add,status='audit_budget_exhausted'));break
        calls+=1;standalone=audit.check(group_edges[add])
        if not standalone['admissible']:
            trace.append(dict(add=add,status='deferred_within_group_fixed_frame_conflict',screen=standalone));continue
        if calls>=max_screenings:trace.append(dict(add=add,status='audit_budget_exhausted'));break
        calls+=1;risk=audit.check(selected|group_edges[add])
        if risk['admissible']:
            candidates.append(dict(add=add,remove=None,groups=chosen+[add],screen=risk));continue
        trace.append(dict(add=add,status='union_conflict_try_bounded_replacements',screen=risk))
        removals=sorted(chosen,key=lambda g:(sum(losses[e] for e in group_edges[g]),g))[:16]
        for remove in removals:
            if calls>=max_screenings:break
            groups=[g for g in chosen if g!=remove]+[add];calls+=1;screen=audit.check(union(groups))
            if screen['admissible']:candidates.append(dict(add=add,remove=remove,groups=groups,screen=screen))
    for c in candidates:
        es=union(c['groups']);c['conditional_residual']=sum(losses[e] for e in deficient-es)
        c['displaced_edges']=[list(e) for e in sorted(selected-es)]
        c['selected_edges']=[list(e) for e in sorted(es)]
    # One candidate only per model, to bound actual solves. It is a conditional
    # ranking, never a promise that adding a constraint repairs its defects.
    best=min(candidates,key=lambda c:(c['conditional_residual'],len(c['displaced_edges']),c['add'],c['remove'] or '')) if candidates else None
    return dict(proposal=best,alternatives=candidates,trace=trace,screening_calls=calls,
        newly_needed_unselected_groups=ordered,scope='One measured feedback operation; exact fixed-frame screen, no integer feasibility guarantee.')

def freeze():
    D.mkdir(exist_ok=False);pilot.modules();plans=[];audits=[]
    for p in read(OUT/'audit/source_plans.json'):
        name=p['model'];r=read(OLD/f'defect_driven_v3/development/{name}/defect_driven__allocated_0/result.json')
        if not r.get('basic_output_pass'):
            audits.append(dict(model=name,status='no_basic_valid_parent_for_constraint_feedback'));continue
        mesh=pilot.load_obj(p['source']);graph=read(p['graph']);selection=read(p['methods']['defect_driven']['selection']);inp=Path(p['source']).parent/'baseline'
        groups={g['id']:{tuple(sorted(e['vertices'])) for e in graph['source_edges'] if g['id'] in e['group_ids']} for g in graph['groups'] if g['layer']=='main'}
        audit=EqualityAudit(mesh,np.loadtxt(inp/'miq_uv.txt',skiprows=1),np.loadtxt(inp/'miq_fuv.txt',skiprows=1,dtype=int),np.loadtxt(inp/'miq_combed_PD1.txt')[:,-3:],np.loadtxt(inp/'miq_combed_PD2.txt')[:,-3:])
        result=propose(groups,selection['selected_group_ids'],r['edge_defects'],audit);result['model']=name;audits.append(result)
        if result['proposal']:
            dest=D/name;dest.mkdir();ep=dest/'selected.edges';np.savetxt(ep,np.asarray(result['proposal']['selected_edges'],int),fmt='%d')
            plans.append(dict(**p,new_edges=str(ep),feedback=result['proposal']))
            print('CONSTRAINT_FROZEN',name,result['proposal']['add'],result['proposal']['remove'],flush=True)
    save(D/'diagnostics.json',audits);save(D/'plans.json',plans)
    save(D/'protocol.json',dict(role='seen development one-operation feedback pilot',
        parent='v3 allocated_0 only when basic valid; all 12 registered, no output-driven cherry picking',
        selection='One highest conditional opportunity feasible group addition/replacement; at most 64 fixed-frame checks and 16 removals per addition.',
        max_backend_calls=2*len(plans),sizes=['uniform','allocated'],count_correction=False,timeout_per_stage=60,max_actual_quads=2048,
        evaluation='Keep old full reference and original v3 unselected set for comparability; report all displaced groups and residual needs.',
        limitation='One-step pilot; unchanged aggregate protection is not per-edge tolerance certification.'))
    shutil.copy2(Path(__file__),D/'constraint_feedback_pilot.py')
    hashes={str(p):sha(p) for p in D.rglob('*') if p.is_file()}
    for p in plans:
        for key in ['source','pd1','pd2','graph','baseline_record']:hashes[p[key]]=sha(p[key])
        for density in p['density'].values():hashes[density]=sha(density)
    save(D/'frozen_hashes.json',hashes)

def run():
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items());pilot.verify();pilot.modules()
    out=D/'runs';out.mkdir(exist_ok=False);rows=[];calls=0
    bins=ROOT/'research_runs/2026-09-19_selection_v3_tools/build/bin'
    for p in read(D/'plans.json'):
        mesh=pilot.load_obj(p['source']);graph=read(p['graph']);h=p['reference_h'];base=read(p['baseline_record'])
        original_selected={tuple(e) for e in read(p['methods']['defect_driven']['selection'])['selected_edges']}
        protected={tuple(sorted(r['vertices'])) for r in base['edge_defects'] if r['layer']=='main'}-original_selected
        points=pilot.area_samples(mesh,4096,190919)[0];candidates=[dict(id='baseline',process_success=True,**base,protected_deficit=pilot.deficit(base['edge_defects'],protected))]
        for kind in ['uniform','allocated']:
            case=p['model']+'__'+kind;dest=out/case;dest.mkdir();calls+=1
            b=pilot.run_backend(mesh=Path(p['source']),pd1=Path(p['pd1']),pd2=Path(p['pd2']),density=Path(p['density'][kind]),hard_edges=Path(p['new_edges']),
                gsize=p['gsize'],miq_executable=bins/'miq_adaptive',qex_executable=bins/'qex_adapter',output_dir=dest/'backend',stiffness_iterations=0,timeout_seconds=60)
            r=dict(id=case,model=p['model'],size=kind,process_success=b['success'],quad_path=str(dest/'backend/quad.obj'),
                miq_seconds=b['miq']['elapsed_seconds'],qex_seconds=b['qex'].get('elapsed_seconds'))
            if b['success']:
                try:
                    r.update(pilot.evaluate(mesh,pilot.load_obj(dest/'backend/quad.obj',require_triangles=False),graph,h,dest/'backend',points))
                    r['protected_deficit']=pilot.deficit(r['edge_defects'],protected)
                except Exception as exc:r.update(process_success=False,evaluation_error=repr(exc))
            save(dest/'result.json',r);candidates.append(r);rows.append({k:v for k,v in r.items() if k not in ['edge_defects','quality','topology','review','uv']});save(out/'summary.json',rows)
            print('CONSTRAINT_RESULT',case,r.get('quads'),r.get('basic_output_pass'),r.get('all_main_deficit'),r.get('protected_deficit'),flush=True)
        save(out/(p['model']+'_decision.json'),pilot.decide_protected(candidates,2048))
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    save(out/'integrity.json',dict(native_calls=calls,complete=True,frozen_inputs_unchanged=True))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args();{'freeze':freeze,'run':run}[args.stage]()
