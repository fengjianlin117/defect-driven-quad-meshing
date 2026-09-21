"""Reusable frozen proposal + optional bounded MIQ/QEx validation.

Run from a bundle containing the frozen helper modules and weak_layout_pipeline.
NeurCross field generation remains a separate remote-GPU step.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import argparse,json
import numpy as np
from reference_graph import build_reference_graph
from defect_predictor_v1 import measure_edges
from defect_selection_lexicographic import ProposalContext,CONFIG
from evaluation_review import review_mesh
from weak_layout_pipeline.pipeline.mesh import load_obj,triangle_geometry,bbox_diagonal
from weak_layout_pipeline.pipeline.evaluate import topology_metrics,quality_metrics,miq_metrics
from weak_layout_pipeline.pipeline.sampling import area_samples,surface_distances
from weak_layout_pipeline.backend.runner import sha256,run_backend

def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))

def field(path,mesh):
    a=np.loadtxt(path,ndmin=2)
    if a.shape!=(mesh.face_count,4) or not np.array_equal(a[:,0],np.arange(mesh.face_count)) or not np.isfinite(a).all():
        raise ValueError('Expected one indexed finite direction per source face')
    return a[:,1:]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['mesh','pd1','pd2','baseline','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--h',type=float,required=True,help='Physical h used for the supplied baseline')
    p.add_argument('--fraction',type=float,default=.5)
    p.add_argument('--miq',type=Path);p.add_argument('--qex',type=Path)
    a=p.parse_args()
    if not np.isfinite(a.h) or a.h<=0 or not 0<=a.fraction<=1:p.error('h>0 and 0<=fraction<=1 required')
    if bool(a.miq)!=bool(a.qex):p.error('Supply both --miq and --qex, or neither')
    inputs=[a.mesh,a.pd1,a.pd2,a.baseline]; hashes={str(x.resolve()):sha256(x) for x in inputs}
    mesh=load_obj(a.mesh);baseline=load_obj(a.baseline,require_triangles=False)
    top=topology_metrics(mesh)
    if not (top['closed'] and top['connected'] and top['edge_manifold'] and top['vertex_manifold']):raise ValueError('Prototype scope: closed connected manifold source')
    x=field(a.pd1,mesh);y=field(a.pd2,mesh);normals,_,_=triangle_geometry(mesh)
    error=max(float(np.abs(np.linalg.norm(x,axis=1)-1).max()),float(np.abs(np.linalg.norm(y,axis=1)-1).max()),
              float(np.abs((x*normals).sum(1)).max()),float(np.abs((y*normals).sum(1)).max()),float(np.abs((x*y).sum(1)).max()))
    if error>1e-5:raise ValueError('Direction frame is not unit, tangent, and orthogonal')
    a.out.mkdir(parents=True,exist_ok=False)
    graph=build_reference_graph(mesh);save(a.out/'graph.json',graph)
    before=measure_edges(mesh,baseline,graph,a.h);save(a.out/'baseline_edge_defects.json',before)
    rho,plan=ProposalContext(mesh,graph,before,x,y,a.h).select(a.fraction)
    plan.update(reference_h=a.h,gsize=bbox_diagonal(mesh)/a.h,config=CONFIG)
    save(a.out/'proposal.json',plan);np.savetxt(a.out/'selected.edges',np.array(plan['selected_edges'],int).reshape(-1,2),fmt='%d')
    np.savetxt(a.out/'density.rho',rho,fmt='%.17g')
    def diagnose(q,uv=None):
        t=topology_metrics(q);quality=quality_metrics(q,mesh)
        ps=area_samples(mesh,4096,190919)[0];pq=area_samples(q,4096,190920)[0]
        sd=surface_distances(ps,q)/a.h;qd=surface_distances(pq,mesh)/a.h
        edges=measure_edges(mesh,q,graph,a.h)
        deficit=sum(r['length_h']*.5*(r['surface_loss']+r['unaligned_fraction']) for r in edges if r['layer']=='main')
        return dict(review=review_mesh(q,mesh,t,quality,uv),topology=t,quality=quality,all_main_deficit=deficit,
                    symmetric_rms_h=float(np.sqrt((np.mean(sd*sd)+np.mean(qd*qd))/2)),source_p95_h=float(np.percentile(sd,95)),output_p95_h=float(np.percentile(qd,95)),
                    acceptance='Descriptive diagnostics; no universal absolute shape tolerance or global intersection certificate.')
    b=diagnose(baseline);save(a.out/'baseline_diagnosis.json',b)
    result=dict(proposal_only=not bool(a.miq),selected_groups=len(plan['selected_group_ids']),selected_edges=len(plan['selected_edges']),field_frame_error=error)
    if a.miq:
        hashes.update({str(t.resolve()):sha256(t) for t in [a.miq,a.qex]})
        backend=run_backend(mesh=a.mesh.resolve(),pd1=a.pd1.resolve(),pd2=a.pd2.resolve(),density=(a.out/'density.rho').resolve(),hard_edges=(a.out/'selected.edges').resolve(),
                            gsize=plan['gsize'],miq_executable=a.miq.resolve(),qex_executable=a.qex.resolve(),output_dir=(a.out/'backend').resolve(),stiffness_iterations=0,timeout_seconds=60)
        result['backend_success']=backend['success']
        if backend['success']:
            q=load_obj(a.out/'backend/quad.obj',require_triangles=False);uv=miq_metrics(a.out/'backend',expected_faces=mesh.face_count)
            after=diagnose(q,uv);save(a.out/'candidate_diagnosis.json',after)
            result.update(basic_output_checks_pass=after['review']['basic_output_checks_pass'],net_feature_deficit_reduction=b['all_main_deficit']-after['all_main_deficit'],
                          surface_rms_h_change=after['symmetric_rms_h']-b['symmetric_rms_h'])
    assert all(sha256(p)==h for p,h in hashes.items())
    save(a.out/'input_hashes.json',hashes);save(a.out/'result.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
