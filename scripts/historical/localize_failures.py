"""Localize measured regressions on immutable shared reference edges."""
from pathlib import Path
import sys, json, itertools
import numpy as np
from audit_evidence import ROOT, OLD, OUT, read, save
sys.path.insert(0,str(OLD/'defect_driven_v3/method'))
from weak_layout_pipeline.pipeline.mesh import load_obj, edge_topology
from conflict_selection import EqualityAudit
from defect_driven_selection import needs_repair

def edge(r): return tuple(sorted(r['vertices']))
def loss(r): return r['length_h']*.5*(r['surface_loss']+r['unaligned_fraction'])
def main():
    dest=OUT/'localization';dest.mkdir(exist_ok=False)
    rows=[]
    for p in read(OUT/'audit/source_plans.json'):
        name=p['model']; mesh=load_obj(p['source']); graph=read(p['graph'])
        base=read(p['baseline_record']); b={edge(r):r for r in base['edge_defects'] if r['layer']=='main'}
        selection=read(p['methods']['defect_driven']['selection']);selected={tuple(e) for e in selection['selected_edges']}
        group_edges={g['id']:{edge(r) for r in b.values() if g['id'] in r['group_ids']} for g in graph['groups'] if g['layer']=='main'}
        inp=Path(p['source']).parent/'baseline'
        audit=EqualityAudit(mesh,np.loadtxt(inp/'miq_uv.txt',skiprows=1),np.loadtxt(inp/'miq_fuv.txt',skiprows=1,dtype=int),np.loadtxt(inp/'miq_combed_PD1.txt')[:,-3:],np.loadtxt(inp/'miq_combed_PD2.txt')[:,-3:])
        conflicts=[]
        for rejection in selection['rejected_hard_groups']:
            add=rejection['group_id'];chosen=selection['selected_group_ids']; variants=[]
            for remove in [None]+chosen:
                new=[g for g in chosen if g!=remove]+[add]
                risk=audit.check(set().union(*(group_edges[g] for g in new)))
                variants.append(dict(remove=remove,add=add,groups=new,**risk))
            witness=list(chosen)+[add]
            for g in sorted(chosen):
                test=[x for x in witness if x!=g]
                if not audit.check(set().union(*(group_edges[x] for x in test)))['admissible']:witness=test
            conflicts.append(dict(deferred=add,irreducible_group_witness=witness,
                witness_check=audit.check(set().union(*(group_edges[x] for x in witness))),single_replacements=variants))
        cases=[]
        summary=read(OLD/f'defect_driven_v3/development/{name}/result.json')
        for c in summary['cases']:
            r=read(OLD/f'defect_driven_v3/development/{name}'/c['id']/'result.json')
            if 'edge_defects' not in r:
                cases.append(dict(id=c['id'],process_success=False,source_record=r.get('source_record')));continue
            after={edge(x):x for x in r['edge_defects'] if x['layer']=='main'}
            assert set(after)==set(b)
            edge_rows=[]
            rho=np.loadtxt(p['density'][c['density_kind']]);topo=edge_topology(mesh)[0]
            for e,br in b.items():
                ar=after[e]; delta=loss(ar)-loss(br)
                face_ids=[f for f,_ in topo[e]]
                edge_rows.append(dict(vertices=list(e),group_ids=br['group_ids'],selected=e in selected,
                    baseline_deficient=needs_repair(br),candidate_deficient=needs_repair(ar),
                    before=loss(br),after=loss(ar),delta=delta,
                    before_unaligned=br['unaligned_fraction'],after_unaligned=ar['unaligned_fraction'],
                    before_surface_p95_h=br['surface_distance_p95_h'],after_surface_p95_h=ar['surface_distance_p95_h'],
                    local_requested_size_h_min=float(1/rho[face_ids].max()),local_requested_size_h_max=float(1/rho[face_ids].min())))
            groups=[]
            for gid,es in group_edges.items():
                subset=[x for x in edge_rows if tuple(x['vertices']) in es]
                groups.append(dict(group_id=gid,selected=gid in selection['selected_group_ids'],
                    before=sum(x['before'] for x in subset),after=sum(x['after'] for x in subset),
                    delta=sum(x['delta'] for x in subset),regressed_edges=sum(x['delta']>1e-9 for x in subset),
                    newly_deficient_edges=sum(not x['baseline_deficient'] and x['candidate_deficient'] for x in subset)))
            case=dict(id=c['id'],quads=r['quads'],basic_output_pass=r['basic_output_pass'],
                all_main_deficit=r['all_main_deficit'],symmetric_rms_h=r['symmetric_rms_h'],
                hard_failure_reasons=r['review']['hard_failure_reasons'],
                common_reference_positive_regression=sum(max(0,x['delta']) for x in edge_rows),
                unselected_positive_regression=sum(max(0,x['delta']) for x in edge_rows if not x['selected']),
                unselected_net_regression=sum(x['delta'] for x in edge_rows if not x['selected']),
                newly_deficient_edges=sum(not x['baseline_deficient'] and x['candidate_deficient'] for x in edge_rows),
                groups=groups,edges=sorted(edge_rows,key=lambda x:-x['delta']))
            assert abs(sum(x['after'] for x in edge_rows)-r['all_main_deficit'])<1e-8
            cases.append(case)
        record=dict(model=name,role='seen development',conflicts=conflicts,cases=cases)
        save(dest/f'{name}.json',record)
        compact=dict(model=name,conflicts=[dict(deferred=x['deferred'],witness=x['irreducible_group_witness'],
            successful_single_removals=[v['remove'] for v in x['single_replacements'] if v['admissible']]) for x in conflicts],
            cases=[{k:v for k,v in x.items() if k not in ['edges','groups']} for x in cases])
        rows.append(compact)
        print(json.dumps(compact,ensure_ascii=False),flush=True)
    save(dest/'summary.json',rows)
if __name__=='__main__':main()
