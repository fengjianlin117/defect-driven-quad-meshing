"""CAD prototype: source-wide budget, frozen proposals, bounded output selection.

Requires an indexed NeurCross frame and its unconstrained baseline. Supply MIQ
and QEx to validate; otherwise only compute the source/baseline proposal.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import argparse,json
import numpy as np
from proposal_cli import field
from reference_graph import build_reference_graph
from defect_predictor_v1 import measure_edges
from defect_selection_lexicographic import ProposalContext
from geometry_budget import propose
from candidate_decision import count_correction_target
from protected_decision import decide_protected
from conflict_selection import EqualityAudit
from defect_driven_selection import select_defect_driven
from constraint_preflight import precheck
from evaluation_review import review_mesh
from weak_layout_pipeline.pipeline.mesh import load_obj,triangle_geometry,bbox_diagonal
from weak_layout_pipeline.pipeline.evaluate import topology_metrics,quality_metrics,miq_metrics
from weak_layout_pipeline.pipeline.sampling import area_samples,surface_distances
from weak_layout_pipeline.backend.runner import run_backend,sha256

def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))

def decide(candidates,cap):
    if not candidates:return dict(policy='Conservative measured baseline non-regression; every tradeoff retained.')
    return decide_protected(candidates,cap)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['mesh','pd1','pd2','baseline','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--h',type=float,required=True)
    parser.add_argument('--max-quads',type=int,default=2048)
    parser.add_argument('--miq',type=Path);parser.add_argument('--qex',type=Path)
    parser.add_argument('--four-paths',action='store_true',help='Validate uniform/allocated crossed with no-lines/lines; otherwise validate only joint.')
    parser.add_argument('--timeout',type=float,default=60.)
    a=parser.parse_args()
    if not np.isfinite(a.h) or a.h<=0 or a.max_quads<1 or not np.isfinite(a.timeout) or a.timeout<=0:parser.error('Invalid numeric policy')
    if bool(a.miq)!=bool(a.qex):parser.error('Supply both MIQ and QEx or neither')
    mesh=load_obj(a.mesh);baseline=load_obj(a.baseline,require_triangles=False);source_top=topology_metrics(mesh)
    if not all(source_top[k] for k in ['closed','connected','edge_manifold','vertex_manifold']) or source_top['degenerate_faces'] or source_top['duplicate_faces']:
        raise ValueError('Prototype requires a closed connected manifold triangular source without repeated/degenerate faces')
    if baseline.face_count>a.max_quads:raise ValueError('Resource cap is smaller than the supplied baseline')
    x,y=field(a.pd1,mesh),field(a.pd2,mesh);normals=triangle_geometry(mesh)[0]
    frame_error=max(float(abs(np.linalg.norm(x,axis=1)-1).max()),float(abs(np.linalg.norm(y,axis=1)-1).max()),
                    float(abs((x*normals).sum(1)).max()),float(abs((y*normals).sum(1)).max()),float(abs((x*y).sum(1)).max()))
    if frame_error>1e-5:raise ValueError('Frame must be unit, tangent, orthogonal, and source-face indexed')
    a.out.mkdir(parents=True,exist_ok=False)
    files=[a.mesh,a.pd1,a.pd2,a.baseline]
    hashes={str(p.resolve()):sha256(p) for p in files}
    hashes.update({str(p.resolve()):sha256(p) for p in Path(__file__).parent.glob('*.py')})
    graph=build_reference_graph(mesh);before=measure_edges(mesh,baseline,graph,a.h)
    context=ProposalContext(mesh,graph,before,x,y,a.h)
    required=[a.baseline.parent/n for n in ['miq_uv.txt','miq_fuv.txt','miq_combed_PD1.txt','miq_combed_PD2.txt']]
    if not all(p.exists() for p in required):raise ValueError('V3 requires matching baseline UV, cut and combed-frame artifacts')
    baseline_uv=np.loadtxt(required[0],skiprows=1);baseline_fuv=np.loadtxt(required[1],skiprows=1,dtype=int)
    baseline_x=np.loadtxt(required[2])[:,-3:];baseline_y=np.loadtxt(required[3])[:,-3:]
    equality_audit=EqualityAudit(mesh,baseline_uv,baseline_fuv,baseline_x,baseline_y)
    selection=select_defect_driven(context,equality_audit)
    hard_set={tuple(sorted(e)) for e in selection['selected_edges']}
    hashes.update({str(p.resolve()):sha256(p) for p in required})
    rho,hv,budget,req=propose(mesh,graph,a.h,baseline.face_count,dict(max_quads=a.max_quads))
    save(a.out/'graph.json',graph);save(a.out/'baseline_edge_defects.json',before)
    risk=dict(status='unavailable',reason='Matching baseline cut/combed-frame files were not supplied alongside the baseline; absence is not a zero-risk prediction.')
    base_dir=a.baseline.parent
    aux=[base_dir/n for n in ['miq_uv.txt','miq_fuv.txt','miq_combed_PD1.txt','miq_combed_PD2.txt']]
    if all(p.exists() for p in aux):
        uv=np.loadtxt(aux[0],skiprows=1);fuv=np.loadtxt(aux[1],skiprows=1,dtype=int)
        cx=np.loadtxt(aux[2])[:,-3:];cy=np.loadtxt(aux[3])[:,-3:]
        if fuv.shape!=(mesh.face_count,3) or fuv.min()<0 or fuv.max()>=len(uv) or cx.shape!=x.shape or cy.shape!=y.shape:
            raise ValueError('Baseline cut/frame artifacts do not match source dimensions')
        risk=dict(status='evaluated',**precheck(mesh,uv,fuv,cx,cy,selection['selected_edges']))
        hashes.update({str(p.resolve()):sha256(p) for p in aux})
    target=budget['target_quads'];gsize=bbox_diagonal(mesh)/a.h
    save(a.out/'proposal.json',dict(selection=selection,budget=budget,combination_risk=risk,gsize=gsize,frame_error=frame_error,
                                  pre_backend_inputs_only=True,unselected_protection='All main features contribute sizing demand regardless of hard-constraint selection.'))
    np.savetxt(a.out/'selected.edges',np.array(selection['selected_edges'],int).reshape(-1,2),fmt='%d')
    (a.out/'none.edges').write_text('')
    uniform=np.full(mesh.face_count,np.sqrt(target/baseline.face_count))
    if np.allclose(rho,uniform,atol=1e-12,rtol=0):rho=uniform.copy()
    np.savetxt(a.out/'allocated.rho',rho,fmt='%.17g');np.savetxt(a.out/'uniform.rho',uniform,fmt='%.17g')
    np.savetxt(a.out/'vertex_sizes.txt',hv,fmt='%.17g')
    np.savez_compressed(a.out/'requests.npz',requested=req['requested_vertex_sizes'],width=req['width_caps'],curvature=req['curvature_caps'],raw=req['raw_vertex_sizes'])
    points=area_samples(mesh,4096,190919)[0]
    def diagnose(q,directory,records=None):
        topology=topology_metrics(q);quality=quality_metrics(q,mesh)
        uv=miq_metrics(directory,expected_faces=mesh.face_count) if (directory/'miq_uv.txt').exists() else None
        review=review_mesh(q,mesh,topology,quality,uv)
        if records is None:records=measure_edges(mesh,q,graph,a.h)
        deficit=sum(r['length_h']*.5*(r['surface_loss']+r['unaligned_fraction']) for r in records if r['layer']=='main')
        sd=surface_distances(points,q)/a.h;qd=surface_distances(area_samples(q,4096,190920)[0],mesh)/a.h
        protected_deficit=sum(r['length_h']*.5*(r['surface_loss']+r['unaligned_fraction']) for r in records if r['layer']=='main' and tuple(sorted(r['vertices'])) not in hard_set)
        return dict(quads=q.face_count,basic_output_pass=review['basic_output_checks_pass'],all_main_deficit=deficit,protected_deficit=protected_deficit,
                    symmetric_rms_h=float(np.sqrt((np.mean(sd*sd)+np.mean(qd*qd))/2)),source_p95_h=float(np.percentile(sd,95)),output_p95_h=float(np.percentile(qd,95)),
                    topology=topology,quality=quality,review=review,uv=uv,edge_defects=records)
    base=diagnose(baseline,base_dir,before);save(a.out/'baseline_diagnosis.json',base)
    candidates=[dict(id='baseline',process_success=True,**base)];native=0;signatures={}
    arms=['uniform_nolines','uniform_lines','allocated_nolines','allocated_lines'] if a.four_paths else ['allocated_lines']
    protocol=dict(arms=arms,max_native_runs=2*len(arms),timeout_per_native_stage=a.timeout,stiffening=0,
                  count_correction='At most once per arm if target error >5% OR actual count exceeds cap. Correct toward min(target, 0.98*cap). Retain every attempt.',
                  decision_policy=decide([],a.max_quads)['policy'])
    save(a.out/'validation_plan.json',protocol)
    for p in a.out.iterdir():
        if p.is_file():hashes[str(p.resolve())]=sha256(p)
    if a.miq:hashes.update({str(p.resolve()):sha256(p) for p in [a.miq,a.qex]})
    save(a.out/'frozen_hashes.json',hashes) # All proposals/protocol frozen before new constrained outputs.
    def run(arm,index,g):
        nonlocal native
        name=f'{arm}_{index}';folder=a.out/name;folder.mkdir()
        density=a.out/(arm.split('_')[0]+'.rho');edges=a.out/('none.edges' if arm.endswith('_nolines') else 'selected.edges')
        sig=(sha256(density),sha256(edges),round(g,12));row=dict(id=name,gsize=g)
        if sig in signatures:
            other=signatures[sig];row.update(other);row.update(id=name,alias_of=other['id'])
        else:
            native+=1;assert native<=protocol['max_native_runs']
            backend=run_backend(mesh=a.mesh.resolve(),pd1=a.pd1.resolve(),pd2=a.pd2.resolve(),density=density.resolve(),hard_edges=edges.resolve(),gsize=g,
                                miq_executable=a.miq.resolve(),qex_executable=a.qex.resolve(),output_dir=(folder/'backend').resolve(),stiffness_iterations=0,timeout_seconds=a.timeout)
            row.update(process_success=backend['success'],miq_seconds=backend['miq']['elapsed_seconds'],qex_seconds=backend['qex'].get('elapsed_seconds'))
            if backend['success']:
                try:row.update(diagnose(load_obj(folder/'backend/quad.obj',require_triangles=False),folder/'backend'))
                except Exception as exc:row.update(evaluation_error=repr(exc),process_success=False)
            signatures[sig]=row
        save(folder/'result.json',row);candidates.append(row)
        print('RESULT',name,row.get('quads'),row.get('basic_output_pass'),flush=True)
        return row
    if a.miq:
        for arm in arms:
            first=run(arm,0,gsize);n=first.get('quads')
            corrected_target=count_correction_target(n,target,a.max_quads) if n else None
            if corrected_target is not None:
                run(arm,1,float(gsize*np.sqrt(corrected_target/n)))
    result=dict(proposal_only=not bool(a.miq),native_runs=native,target_quads=target,decision=decide(candidates,a.max_quads))
    # Full diagnostic records remain in per-attempt files; the index is compact.
    result['candidates']=[{k:r.get(k) for k in ['id','process_success','quads','basic_output_pass','all_main_deficit','symmetric_rms_h','protected_deficit','alias_of']} for r in candidates]
    assert all(sha256(p)==h for p,h in hashes.items())
    result['protected_hash_count']=len(hashes);save(a.out/'result.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
