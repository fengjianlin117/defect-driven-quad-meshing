"""Registered all-model screen and fixed-input constraint-trigger experiment."""
from pathlib import Path
import sys,shutil,copy,subprocess
import numpy as np
from audit_evidence import ROOT,OUT,read,save,sha
import strong_static_controls as static
D=OUT/'protection_trigger_pilot_v1a';M=OUT/'coordinator_v4_development'
W=Path('research://windows-delivery/paper_stage_20260920')

def freeze():
    D.mkdir(exist_ok=False);shutil.copytree(M/'method',D/'method',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['protection_feedback.py','test_protection_feedback.py']:shutil.copy2(W/'method_v5'/name,D/'method'/name)
    tested=subprocess.run([sys.executable,'-B',str(D/'method/test_protection_feedback.py')],capture_output=True,text=True)
    save(D/'tests.json',dict(returncode=tested.returncode,stdout=tested.stdout,stderr=tested.stderr));assert tested.returncode==0
    sys.path.insert(0,str(D/'method'));from protection_feedback import constraint_repair
    audits={r['sha256']:r for r in read(OUT/'intersection_soup_audit_v2/summary.json')}
    plans=[];diagnostics=[]
    for p in read(M/'plans.json'):
        c=static.Coordinator(p,D/p['model'],static.BINS/'miq_adaptive',static.BINS/'qex_adapter')
        for kind in ['uniform','allocated']:
            path=M/'runs'/p['model']/'full'/('initial_'+kind+'_0')/'result.json';r=read(path)
            if not r.get('basic_output_pass') or not Path(r['quad_path']).exists() or not audits[sha(r['quad_path'])]['clear_both_triangulations']:
                diagnostics.append(dict(model=p['model'],size=kind,status='no_strict_valid_parent',parent=str(path)));continue
            original=constraint_repair(c.groups,r['selected_groups'],r['edge_defects'],c.audit,c.base['edge_defects'],c.initial_selected,False)
            updated=constraint_repair(c.groups,r['selected_groups'],r['edge_defects'],c.audit,c.base['edge_defects'],c.initial_selected,True)
            before=original['proposal'];after=updated['proposal']
            changed=after is not None and (before is None or after['selected_edges']!=before['selected_edges'])
            diagnostics.append(dict(model=p['model'],size=kind,parent=str(path),status='changed_proposal' if changed else 'unchanged_or_unavailable',original=original,updated=updated))
            if changed:
                ep=D/p['model']/(kind+'_repair.edges');np.savetxt(ep,np.asarray(after['selected_edges'],int),fmt='%d')
                item=dict(p)
                item.update(size=kind,parent_record=str(path),parent=r,edges=str(ep),density=r['density_path'],gsize=r['gsize'],
                    initial_selected_edges=[list(e) for e in sorted(c.initial_selected)],proposal=after)
                plans.append(item)
    save(D/'diagnostics.json',diagnostics);save(D/'plans.json',plans)
    save(D/'protocol.json',dict(question='Can an unchanged aggregate-protection violation trigger useful missing constraint repair below the original diagnostic cutoff?',
        population='Every one of the 12 seen models, both initial uniform_0/allocated_0 v4 parents; pre-registered strict-valid parent rule, including all no-op and conflict cases.',
        intervention='Only the hard-edge set changes from the parent. Exact mesh, field files, density bytes, gsize, binaries, limits and samples fixed. No count correction.',
        comparison='Original trigger versus diagnostic OR individually regressed protected edges when their unchanged aggregate safeguard fails. No acceptance threshold change.',
        max_backend_requests=len(plans),registered_parents=24,independent_new_models=0,
        reporting='Every trigger screen, inherited failure, changed group and actual output count retained. This pilot does not prove full-method generalization.'))
    shutil.copy2(Path(__file__),D/'protection_trigger_pilot.py')
    hashes={str(f):sha(f) for f in D.rglob('*') if f.is_file()};hashes.update(read(M/'frozen_hashes.json'))
    for p in plans:
        for k in ['parent_record','density','edges']:hashes[p[k]]=sha(p[k])
        hashes[p['parent']['quad_path']]=sha(p['parent']['quad_path'])
    hashes[str(static.geom.BIN)]=sha(static.geom.BIN);save(D/'frozen_hashes.json',hashes)
    print('PROTECTION_TRIGGER_FROZEN',len(diagnostics),len(plans),[(p['model'],p['size'],p['proposal']['add']) for p in plans],flush=True)

def run():
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    out=D/'runs';out.mkdir(exist_ok=False);rows=[];native=0;reused=0
    sys.path.insert(0,str(D/'method'));from protected_decision import decide_protected
    for p in read(D/'plans.json'):
        folder=out/(p['model']+'__'+p['size']);folder.mkdir();cache=static.historic(p)
        sig=static.signature(p,p['edges'],p['density'],p['gsize'])
        if sig in cache:r=copy.deepcopy(cache[sig]['record']);reused+=1;r.update(reused=True,source_record=cache[sig]['path'])
        else:
            native+=1;b=static.run_backend(mesh=Path(p['source']),pd1=Path(p['pd1']),pd2=Path(p['pd2']),density=Path(p['density']),hard_edges=Path(p['edges']),gsize=p['gsize'],
                miq_executable=static.BINS/'miq_adaptive',qex_executable=static.BINS/'qex_adapter',output_dir=folder/'backend',stiffness_iterations=0,timeout_seconds=60)
            r=dict(process_success=b['success'],reused=False,quad_path=str(folder/'backend/quad.obj'))
            if b['success']:r.update(static.evaluate(static.load_obj(p['source']),static.load_obj(r['quad_path'],require_triangles=False),read(p['graph']),p['reference_h']))
        base=read(p['baseline_record']);selected={tuple(e) for e in p['initial_selected_edges']};r.update(static.common_metrics(r,base,selected));r.update(id='repair',model=p['model'],size=p['size'])
        checks=[static.geom.check(r['quad_path'],i) for i in [0,1]] if Path(r['quad_path']).exists() else []
        r['strict_output_pass']=r.get('basic_output_pass') is True and len(checks)==2 and all(x.get('completed') is True and x.get('intersection_pair_count')==0 and x.get('exact_degenerate_triangles')==0 for x in checks)
        save(folder/'intersection.json',checks);save(folder/'result.json',r)
        baseline=dict(id='baseline',process_success=True,**base,**static.common_metrics(base,base,selected))
        decision=decide_protected([baseline,r],2048);save(folder/'decision.json',decision)
        parent=p['parent'];row=dict(model=p['model'],size=p['size'],add=p['proposal']['add'],remove=p['proposal']['remove'],parent_record=p['parent_record'],record=str(folder/'result.json'),
            exact_fixed_inputs=dict(source_sha256=sha(p['source']),pd1_sha256=sha(p['pd1']),pd2_sha256=sha(p['pd2']),density_sha256=sha(p['density']),gsize_hex=float(p['gsize']).hex()),
            parent={k:parent.get(k) for k in ['quads','all_main_deficit','symmetric_rms_h','protected_deficit','newly_deficient_edges']},
            repair={k:r.get(k) for k in ['quads','all_main_deficit','symmetric_rms_h','protected_deficit','newly_deficient_edges','common_positive_regression','strict_output_pass']},
            passes_unchanged_protection='repair' in decision['eligible_ids'],reused=r['reused'])
        rows.append(row);save(out/'summary.json',rows);print('PROTECTION_TRIGGER_RESULT',row,flush=True)
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    save(out/'integrity.json',dict(native_calls=native,reused_requests=reused,complete=True,frozen_unchanged=True))

if __name__=='__main__':globals()[sys.argv[1]]()
