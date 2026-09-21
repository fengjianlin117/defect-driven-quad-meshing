"""Bounded v5 coordinator: complete geometric gate and scan past no-op parents."""
import argparse, json, subprocess, time
from pathlib import Path
from coordinator import Coordinator, POLICY, read, save, sha
from coordination import constraint_operation, size_operation, failure_removal
from geometry_budget import allocate
from protected_decision import decide_protected
from weak_layout_pipeline.pipeline.mesh import load_obj
from protection_feedback import constraint_repair,size_repair,simple_initial
from coordination import common_metrics

def strict_decision(rows, cap):
    result=decide_protected(rows,cap)
    eligible=set(result['eligible_ids'])
    for row in result['audit']:
        candidate=next(c for c in rows if c['id']==row['id'])
        if candidate.get('strict_output_pass') is not True:
            row['reasons'].append('independent_intersection_or_basic_check_failed_or_incomplete')
            row['conservative_eligible']=False;eligible.discard(row['id'])
    accepted=[r for r in rows if r['id'] in eligible]
    best=min(accepted,key=lambda r:(r['all_main_deficit'],r['symmetric_rms_h'],r['quads'],r['id'])) if accepted else None
    result.update(recommended_id=best['id'] if best else None,eligible_ids=[r['id'] for r in accepted],
        geometric_gate='Basic checks AND both CGAL triangle-soup diagonal embeddings completed with no intersection and no exact degeneracy. Missing checks reject.',
        aggregate_limitation='Inherited fixed-complement aggregate protection still permits per-edge regression; no engineering tolerance or bilinear embedding guarantee.')
    return result

class CoordinatorV5(Coordinator):
    def __init__(self,plan,out,miq,qex,checker,variant='full',cache=None,geometry_cache=None,step=2.):
        self.requested_variant=variant
        super().__init__(plan,out,miq,qex,'full' if variant in ['stop_first_noop','no_protection_feedback'] else variant,cache,step)
        self.selection=simple_initial(self.groups,self.base['edge_defects'],self.audit)
        self.initial_selected={tuple(e) for e in self.selection['selected_edges']}
        self.rows[0].update(common_metrics(self.base,self.base,self.initial_selected))
        save(self.out/'initial_proposal.json',dict(selection=self.selection,budget=self.budget))
        self.checker=Path(checker);self.geometry_cache=geometry_cache if geometry_cache is not None else {}
        self.checker_sha=sha(checker);self.geometry_checks=0;self.geometry_reused=0
        source_audit=self.check_geometry(Path(plan['source']),self.out/'source_intersection.json')
        self.source_geometry_clear=source_audit['clear']
        self.rows[0].update(self.decorate(self.rows[0],self.inp/'quad.obj',self.out/'baseline_intersection.json'))
        policy=dict(POLICY)
        policy.update(version=5,variant=variant,actual_density_step=step,
            initial='Whole needed groups ordered by length-weighted average deficit, fixed-frame compatibility, no quota; uniform and source-wide allocated initial densities.',
            parent='Lowest whole-reference deficit among unexpanded strict-valid generated candidates; RMS and id break ties.',
            stopping='Two productive rounds, twelve logical attempts, twelve parent visits, or no unexpanded actionable candidate; skip no-op parents except the registered stop_first_noop ablation.',
            protection='Unchanged v4 fixed-complement aggregate safeguards, with the strict final geometry gate applied to every candidate including baseline.',
            complete_geometry_gate='Source and baseline audited; all candidates are audited even if basic-invalid. Parent and final recommendation require basic validity plus clear exact triangle-soup checks for both diagonals.',
            scan_policy='Within each of two productive rounds, visit unexpanded valid generated parents ordered by full deficit/RMS/id; a no-op parent is recorded and skipped. At most 12 parent visits and 12 logical backend attempts overall.',
            no_reference_change='All source edges and sizing requests survive skipped or removed constraints.',
            protection_feedback='The unchanged fixed-complement safeguard can also trigger constraint and size repair for its individually regressed edges, even below the initial diagnostic cutoff. Disabled only by no_protection_feedback.',
            source_failure='If source geometry audit fails or is incomplete, stop without backend calls or a recommendation.',
            checker_sha256=self.checker_sha)
        save(self.out/'policy.json',policy)

    def check_geometry(self,path,destination):
        start=time.perf_counter();checks=[]
        if not path.is_file():
            result=dict(clear=False,completed=False,reason='missing_geometry',path=str(path))
        else:
            key=(self.checker_sha,sha(path))
            if key in self.geometry_cache:
                checks=self.geometry_cache[key];self.geometry_reused+=1;reused=True
            else:
                reused=False
                for diagonal in [0,1]:
                    self.geometry_checks+=1
                    try:
                        p=subprocess.run([str(self.checker),str(path),str(diagonal)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
                        check=json.loads(p.stdout) if p.returncode==0 else dict(completed=False)
                        check.update(returncode=p.returncode,stderr=p.stderr)
                    except Exception as exc:check=dict(completed=False,error=repr(exc))
                    checks.append(check)
                self.geometry_cache[key]=checks
            result=dict(path=str(path),sha256=key[1],checker_sha256=key[0],checks=checks,reused=reused,
                completed=all(c.get('completed') is True for c in checks),
                clear=all(c.get('completed') is True and c.get('intersection_pair_count')==0 and c.get('exact_degenerate_triangles')==0 for c in checks))
        result['elapsed_seconds']=time.perf_counter()-start;save(destination,result);return result

    def decorate(self,row,path,destination):
        audit=self.check_geometry(path,destination)
        return dict(strict_output_pass=row.get('process_success') is True and row.get('basic_output_pass') is True and audit['clear'],
            intersection_clear=audit['clear'],intersection_record=str(destination))

    def attempt(self,*args,**kwargs):
        row=super().attempt(*args,**kwargs)
        if row is not None:
            folder=self.out/row['id'];row.update(self.decorate(row,Path(row['quad_path']),folder/'intersection.json'))
            save(folder/'result.json',row);self.flush()
        return row

    def flush(self):
        super().flush()
        data=read(self.out/'summary.json')
        for a,b in zip(data['candidates'],self.rows):
            a.update({k:b.get(k) for k in ['strict_output_pass','intersection_clear','intersection_record']})
        save(self.out/'summary.json',data)

    def run(self):
        groups=self.selection['selected_group_ids'];expanded=set();visits=0
        if self.source_geometry_clear:
            for kind in ['uniform','allocated']:self.branch('initial_'+kind,groups,self.initial_density[kind],self.p['gsize'],self.target)
        else:self.trace.append(dict(stop='source_geometry_failed_or_incomplete'))
        if self.source_geometry_clear and self.variant!='initial_only':
            for round_id in range(2):
                productive=False
                while self.attempts<12 and visits<12:
                    valid=[r for r in self.rows[1:] if r.get('strict_output_pass') and r.get('quads',2049)<=2048 and r['id'] not in expanded]
                    before=self.attempts
                    if valid:
                        parent=min(valid,key=lambda r:(r['all_main_deficit'],r['symmetric_rms_h'],r['id']))
                        expanded.add(parent['id']);visits+=1;state=self.states[parent['id']]
                        if self.variant!='no_constraint':
                            result=constraint_repair(self.groups,state['groups'],parent['edge_defects'],self.audit,self.base['edge_defects'],self.initial_selected,self.requested_variant!='no_protection_feedback')
                            self.trace.append(dict(round=round_id,operation='constraint',parent=parent['id'],**result))
                            if result['proposal']:
                                self.branch(f'r{round_id}_constraint',result['proposal']['groups'],state['density'],state['g'],state['target'],state['requested'])
                        if self.variant!='no_size':
                            result=size_repair(self.mesh,self.graph,self.h,self.base,parent,state['density']*(state['g']/self.p['gsize']),state['target'],self.step,self.initial_selected,self.requested_variant!='no_protection_feedback',state['requested'])
                            self.trace.append(dict(round=round_id,operation='size',parent=parent['id'],audit=result['audit'] if result else None))
                            if result is not None:self.branch(f'r{round_id}_size',state['groups'],result['density'],self.p['gsize'],result['target'],result['requested'])
                    elif not any(r.get('strict_output_pass') and r.get('quads',2049)<=2048 for r in self.rows[1:]) and self.variant!='no_recovery':
                        measured=[r for r in self.rows[1:] if r.get('edge_defects') and r['id'] not in expanded]
                        unexpanded=[r for r in self.rows[1:] if r['id'] not in expanded]
                        if not unexpanded:break
                        parent=min(measured,key=lambda r:(len(r.get('review',{}).get('hard_failure_reasons',[])),r['all_main_deficit'],r['id'])) if measured else unexpanded[-1]
                        expanded.add(parent['id']);visits+=1;state=self.states[parent['id']]
                        path=Path(parent.get('quad_path','/nonexistent'))
                        output=load_obj(path,require_triangles=False) if path.is_file() and parent.get('edge_defects') else None
                        result=failure_removal(self.mesh,output,self.groups,state['groups'],self.ctx)
                        self.trace.append(dict(round=round_id,operation='failure_removal',parent=parent['id'],proposal=result))
                        if result:self.branch(f'r{round_id}_remove',result['groups'],state['density'],state['g'],state['target'],state['requested'])
                        if not self.count_probe_used and state['target']<2048:
                            self.count_probe_used=True;target=min(2048,2*state['target']);rho,_,stats=allocate(self.mesh,self.h,self.req,target/self.base['quads'])
                            self.trace.append(dict(round=round_id,operation='count_probe',parent=parent['id'],target=target,allocation=stats))
                            self.branch(f'r{round_id}_count',state['groups'],rho,self.p['gsize'],target)
                    else:break
                    if self.attempts>before:productive=True;break
                    self.trace.append(dict(round=round_id,parent=parent['id'],no_operation=True,action='stop' if self.requested_variant=='stop_first_noop' else 'scan_next_parent'))
                    if self.requested_variant=='stop_first_noop':break
                if not productive:break
        decision=strict_decision(self.rows,2048)
        if not self.source_geometry_clear:decision.update(recommended_id=None,eligible_ids=[],input_rejected=True)
        self.flush();save(self.out/'decision.json',decision)
        save(self.out/'complete.json',dict(attempts=self.attempts,native_calls=self.native,reused_attempts=self.reused,
            recommended=decision['recommended_id'],all_candidates_retained=True,reference_edges=len(self.ctx.all_edges),variant=self.requested_variant,
            source_geometry_clear=self.source_geometry_clear,parent_visits=visits,geometry_checks=self.geometry_checks,geometry_reused=self.geometry_reused))
        return decision

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--miq',required=True);p.add_argument('--qex',required=True);p.add_argument('--checker',required=True)
    p.add_argument('--variant',default='full');p.add_argument('--step',type=float,default=2.)
    a=p.parse_args();CoordinatorV5(read(a.input),a.out,a.miq,a.qex,a.checker,a.variant,step=a.step).run()
