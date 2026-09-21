"""Version 4: bounded constraint/size feedback for a fixed source and field.

Input JSON names mesh/field/graph/baseline artifacts and a calibrated gsize.
No model identity enters operations. Every model uses the same finite policy.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import argparse,json,hashlib,math,copy,time
import numpy as np
from coordination import constraint_operation,size_operation,failure_removal,common_metrics
from conflict_selection import EqualityAudit
from defect_driven_selection import select_defect_driven
from defect_selection_lexicographic import ProposalContext
from geometry_budget import propose,requests,allocate
from protected_decision import decide_protected
from candidate_decision import count_correction_target
from weak_layout_pipeline.pipeline.mesh import load_obj
from weak_layout_pipeline.backend.runner import run_backend
from neutral_evaluation import evaluate

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
POLICY=dict(max_attempts=12,feedback_rounds=2,max_screenings_per_operation=64,max_actual_quads=2048,
    timeout_per_backend_stage=60,threads=1,stiffening=0,density_step=2.,
    initial='Uniform and source-wide allocated density with defect-driven whole-group selection; at most one count correction per branch.',
    parent='Lowest whole-reference deficit among unexpanded basic-valid generated candidates; surface RMS and id break ties. Keep baseline and every candidate.',
    valid_feedback='One bounded constraint addition/replacement and one regressed-deficiency size proposal from the same measured parent per round.',
    recovery='When there is no basic-valid generated candidate: one group-removal hypothesis, and at most one 2x count probe in the entire run.',
    stopping='Stop after two feedback rounds, twelve logical attempts, or no new operation; all corrections and aliases count toward the attempt cap.',
    protection='Retain original v3 aggregate decision as primary developmental gate; use the initial selected set for its fixed unselected complement in every candidate.',
    common_reference='Also report positive per-edge regression and newly deficient reference length on the same full set for all alternatives.',
    limitations='No calibrated CAD tolerance, per-edge preservation, global intersection guarantee, global integer feasibility or optimal subset guarantee.')

class Coordinator:
    def __init__(self,plan,out,miq,qex,variant='full',cache=None,step=2.):
        if variant not in ['full','no_size','no_constraint','no_recovery','initial_only']:raise ValueError('Unknown ablation')
        self.p=plan;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.miq=Path(miq);self.qex=Path(qex);self.variant=variant;self.step=step
        self.mesh=load_obj(plan['source']);self.graph=read(plan['graph']);self.base=read(plan['baseline_record']);self.h=plan['reference_h']
        self.points=None;self.groups={g['id']:{tuple(sorted(e['vertices'])) for e in self.graph['source_edges'] if e['layer']=='main' and g['id'] in e['group_ids']} for g in self.graph['groups'] if g['layer']=='main'}
        inp=Path(plan['source']).parent/'baseline';self.inp=inp
        x=np.loadtxt(plan['pd1'])[:,-3:];y=np.loadtxt(plan['pd2'])[:,-3:]
        self.ctx=ProposalContext(self.mesh,self.graph,self.base['edge_defects'],x,y,self.h)
        self.audit=EqualityAudit(self.mesh,np.loadtxt(inp/'miq_uv.txt',skiprows=1),np.loadtxt(inp/'miq_fuv.txt',skiprows=1,dtype=int),np.loadtxt(inp/'miq_combed_PD1.txt')[:,-3:],np.loadtxt(inp/'miq_combed_PD2.txt')[:,-3:])
        self.selection=select_defect_driven(self.ctx,self.audit);self.initial_selected={tuple(e) for e in self.selection['selected_edges']}
        rho,hv,self.budget,self.req=propose(self.mesh,self.graph,self.h,self.base['quads'])
        self.target=self.budget['target_quads'];self.initial_density=dict(uniform=np.full(self.mesh.face_count,np.sqrt(self.target/self.base['quads'])),allocated=rho)
        if np.allclose(rho,self.initial_density['uniform'],atol=1e-12,rtol=0):self.initial_density['allocated']=self.initial_density['uniform'].copy()
        self.rows=[dict(id='baseline',process_success=True,**self.base,**common_metrics(self.base,self.base,self.initial_selected))]
        self.states={};self.trace=[];self.cache=cache if cache is not None else {};self.native=0;self.attempts=0;self.reused=0;self.count_probe_used=False
        self.prefix=(sha(plan['source']),sha(plan['pd1']),sha(plan['pd2']),sha(miq),sha(qex))
        save(self.out/'initial_proposal.json',dict(selection=self.selection,budget=self.budget))
        save(self.out/'policy.json',dict(**POLICY,variant=variant,actual_density_step=step))

    def signature(self,density,edges,g):
        return (*self.prefix,sha(density),sha(edges),float(g).hex(),0,60)

    def attempt(self,name,groups,density,g,target,requested=None):
        if self.attempts>=12:return None
        folder=self.out/name;folder.mkdir();ep=folder/'hard.edges';rp=folder/'density.rho'
        es=sorted(set().union(*(self.groups[x] for x in groups)));np.savetxt(ep,np.asarray(es,int).reshape(-1,2),fmt='%d');np.savetxt(rp,density,fmt='%.17g')
        sig=self.signature(rp,ep,g);self.attempts+=1
        if sig in self.cache:
            cached=self.cache[sig];row=copy.deepcopy(cached['record']);self.reused+=1
            row.update(reused=True,source_record=cached['path'])
        else:
            self.native+=1
            b=run_backend(mesh=Path(self.p['source']),pd1=Path(self.p['pd1']),pd2=Path(self.p['pd2']),density=rp,hard_edges=ep,gsize=g,
                miq_executable=self.miq,qex_executable=self.qex,output_dir=folder/'backend',stiffness_iterations=0,timeout_seconds=60)
            row=dict(process_success=b['success'],reused=False,quad_path=str(folder/'backend/quad.obj'),
                miq_seconds=b['miq']['elapsed_seconds'],qex_seconds=b['qex'].get('elapsed_seconds'),backend_record=str(folder/'backend/backend_run.json'))
            if b['success']:
                try:row.update(evaluate(self.mesh,load_obj(folder/'backend/quad.obj',require_triangles=False),self.graph,self.h))
                except Exception as exc:row.update(process_success=False,evaluation_error=repr(exc))
        row.update(id=name,gsize=g,target_quads=target,selected_groups=list(groups),density_path=str(rp),edges_path=str(ep))
        row.update(common_metrics(row,self.base,self.initial_selected));save(folder/'result.json',row)
        self.states[name]=dict(groups=list(groups),density=np.asarray(density).copy(),g=g,target=target,requested=requested)
        self.rows.append(row);self.cache[sig]=dict(record=row,path=str(folder/'result.json'))
        print('COORD',self.p['model'],self.variant,name,row.get('quads'),row.get('basic_output_pass'),row.get('all_main_deficit'),'reuse',row['reused'],flush=True)
        self.flush();return row

    def branch(self,name,groups,density,g,target,requested=None):
        first=self.attempt(name+'_0',groups,density,g,target,requested)
        if first and first.get('quads'):
            corrected=count_correction_target(first['quads'],target,2048)
            if corrected is not None:self.attempt(name+'_1',groups,density,g*math.sqrt(corrected/first['quads']),target,requested)

    def flush(self):
        save(self.out/'summary.json',dict(candidates=[{k:r.get(k) for k in ['id','process_success','quads','basic_output_pass','all_main_deficit','symmetric_rms_h','protected_deficit','common_positive_regression','newly_deficient_edges','reused','source_record','quad_path']} for r in self.rows],
            attempts=self.attempts,native_calls=self.native,reused_attempts=self.reused,trace=self.trace))

    def run(self):
        groups=self.selection['selected_group_ids']
        for kind in ['uniform','allocated']:self.branch('initial_'+kind,groups,self.initial_density[kind],self.p['gsize'],self.target)
        expanded=set()
        if self.variant!='initial_only':
            for round_id in range(2):
                if self.attempts>=12:break
                valid=[r for r in self.rows[1:] if r.get('basic_output_pass') and r.get('quads',2049)<=2048 and r['id'] not in expanded]
                before_attempts=self.attempts
                if valid:
                    parent=min(valid,key=lambda r:(r['all_main_deficit'],r['symmetric_rms_h'],r['id']));expanded.add(parent['id']);state=self.states[parent['id']]
                    if self.variant!='no_constraint':
                        result=constraint_operation(self.groups,state['groups'],parent['edge_defects'],self.audit)
                        self.trace.append(dict(round=round_id,operation='constraint',parent=parent['id'],**result))
                        if result['proposal']:
                            proposal=result['proposal'];self.branch(f'r{round_id}_constraint',proposal['groups'],state['density'],state['g'],state['target'],state['requested'])
                    if self.variant!='no_size':
                        effective_density=state['density']*(state['g']/self.p['gsize'])
                        result=size_operation(self.mesh,self.graph,self.h,self.base,parent,effective_density,state['target'],self.step,state['requested'])
                        self.trace.append(dict(round=round_id,operation='size',parent=parent['id'],audit=result['audit'] if result else None))
                        if result is not None:self.branch(f'r{round_id}_size',state['groups'],result['density'],self.p['gsize'],result['target'],result['requested'])
                elif not any(r.get('basic_output_pass') and r.get('quads',2049)<=2048 for r in self.rows[1:]) and self.variant!='no_recovery':
                    measured=[r for r in self.rows[1:] if r.get('edge_defects') and r['id'] not in expanded]
                    parent=min(measured,key=lambda r:(len(r.get('review',{}).get('hard_failure_reasons',[])),r['all_main_deficit'],r['id'])) if measured else self.rows[-1]
                    expanded.add(parent['id']);state=self.states[parent['id']]
                    qpath=Path(parent.get('quad_path','/nonexistent'))
                    output=load_obj(qpath,require_triangles=False) if qpath.is_file() and parent.get('edge_defects') else None
                    result=failure_removal(self.mesh,output,self.groups,state['groups'],self.ctx)
                    self.trace.append(dict(round=round_id,operation='failure_removal',parent=parent['id'],proposal=result))
                    if result:self.branch(f'r{round_id}_remove',result['groups'],state['density'],state['g'],state['target'],state['requested'])
                    if not self.count_probe_used and state['target']<2048:
                        self.count_probe_used=True;target=min(2048,2*state['target']);rho,_,stats=allocate(self.mesh,self.h,self.req,target/self.base['quads'])
                        self.trace.append(dict(round=round_id,operation='count_probe',parent=parent['id'],target=target,allocation=stats))
                        self.branch(f'r{round_id}_count',state['groups'],rho,self.p['gsize'],target)
                else:break
                if self.attempts==before_attempts:
                    self.trace.append(dict(round=round_id,stop='no_new_operation_from_available_parent'));break
        decision=decide_protected(self.rows,2048);self.flush();save(self.out/'decision.json',decision)
        save(self.out/'complete.json',dict(attempts=self.attempts,native_calls=self.native,reused_attempts=self.reused,
            recommended=decision['recommended_id'],all_candidates_retained=True,reference_edges=len(self.ctx.all_edges),variant=self.variant))
        return decision

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--miq',required=True);parser.add_argument('--qex',required=True);parser.add_argument('--variant',default='full')
    a=parser.parse_args();Coordinator(read(a.input),a.out,a.miq,a.qex,a.variant).run()
