"""Source-registered loop diameter and coaxial spacing proxies (not B-Rep tolerances)."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,math,argparse,shutil
from pathlib import Path
import numpy as np
from audit_evidence import OUT,OLD,read,save,sha
sys.path.insert(0,str(OLD/'defect_driven_v3/method'))
from weak_layout_pipeline.pipeline.mesh import load_obj,unique_edges
from feature_diagnostics import sample_curve,segment_query
D=OUT/'cad_dimension_proxies_v1'

def circle(points):
    center=points.mean(0);_,s,vt=np.linalg.svd(points-center,full_matrices=False);normal=vt[2]
    if normal[np.argmax(abs(normal))]<0:normal=-normal
    uv=(points-center)@vt[:2].T
    design=np.c_[2*uv,np.ones(len(uv))]
    coeff,_,rank,_=np.linalg.lstsq(design,(uv*uv).sum(1),rcond=None)
    rad2=coeff[2]+coeff[0]**2+coeff[1]**2
    if rank<3 or rad2<=0:return None
    radius=float(np.sqrt(rad2));c=center+coeff[:2]@vt[:2]
    return dict(center=c.tolist(),normal=normal.tolist(),radius=radius,
        max_plane_residual=float(np.max(abs((points-center)@normal))),
        max_radial_residual=float(np.max(abs(np.linalg.norm(uv-coeff[:2],axis=1)-radius))))

def register():
    D.mkdir(exist_ok=False);models=[]
    for p in read(OUT/'audit/source_plans.json'):
        mesh=load_obj(p['source']);graph=read(p['graph']);diagonal=float(np.linalg.norm(np.ptp(mesh.vertices,axis=0)));loops=[];excluded=[]
        for g in graph['groups']:
            if g['layer']!='main':continue
            ids=np.unique(g['mesh_vertices']);fit=circle(mesh.vertices[ids]) if g['closed'] and len(ids)>=8 else None
            if fit is None or fit['max_plane_residual']>1e-5*diagonal or fit['max_radial_residual']>1e-4*diagonal:
                excluded.append(dict(group=g['id'],reason='not_closed_8_vertex_planar_circular_consistent_loop',fit=fit));continue
            points,tangents,_=sample_curve(mesh,g,p['reference_h']);samplefit=circle(points)
            loops.append(dict(group=g['id'],vertex_fit=fit,sample_fit=samplefit,sample_count=len(points),curve=g))
        pairs=[]
        for i,a in enumerate(loops):
            for b in loops[i+1:]:
                ca,cb=np.array(a['vertex_fit']['center']),np.array(b['vertex_fit']['center']);axis=np.array(a['vertex_fit']['normal']);other=np.array(b['vertex_fit']['normal'])
                delta=cb-ca;distance=abs(float(delta@axis));offaxis=float(np.linalg.norm(delta-(delta@axis)*axis))
                if abs(axis@other)>=np.cos(np.radians(1)) and offaxis<=1e-4*diagonal and distance>1e-5*diagonal:
                    pairs.append(dict(a=a['group'],b=b['group'],axis=axis.tolist(),source_spacing=distance))
        models.append(dict(model=p['model'],source=p['source'],graph=p['graph'],h=p['reference_h'],diagonal=diagonal,loops=loops,pairs=pairs,excluded_groups=excluded))
        print('DIMENSION_REGISTERED',p['model'],'loops',len(loops),'pairs',len(pairs),flush=True)
    save(D/'source_registry.json',models)
    save(D/'protocol.json',dict(role='Geometry-based CAD dimension proxies registered from all 12 sources only.',
        qualification='Closed main feature loop, >=8 distinct vertices, fitted-plane max residual <=1e-5D, fitted-circle radial max residual <=1e-4D.',
        caveat='Circular consistency of a tessellated loop does not prove an analytic CAD circle. No exact B-Rep, manufacturing tolerance or millimetre unit is asserted.',
        dimensions='Diameter fitted to fixed source-curve samples; coaxial loop center spacing along source normal. Also report whole-object axis-aligned spans.',
        output_association='Use all uniform-perimeter source samples and their nearest output segment satisfying the inherited 15-degree tangent condition. Never remove poorly matched points to improve fit.',
        missingness='If any tangent-aligned correspondence is unavailable, report unavailable; otherwise report coverage <=0.1h and distance p95 alongside fits, without gating away poor fits.',
        units='Original input units plus normalization by source bbox diagonal D and reference h; no units invented.',
        source_sampling='At least 64 samples per loop and 32 per h; compare source-sample fits to avoid a vertex-only circle versus polygon-chord bias.'))
    shutil.copy2(Path(__file__),D/'cad_dimension_proxies.py')
    hashes={str(p):sha(p) for p in D.iterdir() if p.is_file()}
    for m in models:
        for key in ['source','graph']:hashes[m[key]]=sha(m[key])
    save(D/'source_frozen_hashes.json',hashes)

def run():
    assert (OUT/'coordinator_v4_development/runs/integrity.json').exists()
    assert all(sha(p)==h for p,h in read(D/'source_frozen_hashes.json').items());dest=D/'measurements';dest.mkdir(exist_ok=False)
    models={m['model']:m for m in read(D/'source_registry.json')};cases=[]
    for row in read(OUT/'checkpoint_01/candidate_comparison.json'):
        r=read(row['record']);m=models[row['model']];cases.append(dict(model=row['model'],condition='pilot_'+row['condition'],quad=r.get('quad_path') or str(Path(m['source']).parent/'baseline/quad.obj')))
    for row in read(OUT/'coordinator_v4_development/runs/summary.json'):
        model=row['model'];variant=row['variant'];folder=OUT/'coordinator_v4_development/runs'/model/variant
        recommended=read(folder/'decision.json')['recommended_id'];m=models[model]
        quad=str(Path(m['source']).parent/'baseline/quad.obj') if recommended=='baseline' else read(folder/recommended/'result.json')['quad_path']
        cases.append(dict(model=model,condition='coordinator_'+variant,quad=quad))
    save(D/'cases_before_measurement.json',cases);cache={};rows=[]
    for c in cases:
        m=models[c['model']];signature=(m['model'],sha(c['quad']))
        if signature not in cache:
            source=load_obj(m['source']);output=load_obj(c['quad'],require_triangles=False);segments=output.vertices[unique_edges(output)];loops=[]
            for g in m['loops']:
                points,tangents,_=sample_curve(source,g['curve'],m['h']);_,_,distance,indices=segment_query(points,segments,tangents,15.)
                row=dict(group=g['group'],source_sample_fit=g['sample_fit'],source_vertex_fit=g['vertex_fit'])
                if not np.isfinite(distance).all():row.update(available=False,reason='missing_tangent_aligned_correspondence')
                else:
                    chosen=segments[indices];a=chosen[:,0];v=chosen[:,1]-a;t=np.clip(np.einsum('ij,ij->i',points-a,v)/np.maximum(np.einsum('ij,ij->i',v,v),1e-30),0,1)
                    matched=a+t[:,None]*v;fit=circle(matched)
                    row.update(available=fit is not None,output_fit=fit,coverage_within_01h=float(np.mean(distance<=.1*m['h'])),aligned_distance_p95_h=float(np.percentile(distance,95)/m['h']))
                    if fit is not None:
                        error=2*(fit['radius']-g['sample_fit']['radius']);row.update(diameter_error=error,diameter_error_over_D=error/m['diagonal'],diameter_error_over_h=error/m['h'])
                loops.append(row)
            lookup={r['group']:r for r in loops};pairs=[]
            for p in m['pairs']:
                a,b=lookup[p['a']],lookup[p['b']];row=dict(**p,available=a['available'] and b['available'])
                if row['available']:
                    actual=abs(float((np.array(b['output_fit']['center'])-np.array(a['output_fit']['center']))@np.array(p['axis'])))
                    ref=abs(float((np.array(b['source_sample_fit']['center'])-np.array(a['source_sample_fit']['center']))@np.array(p['axis'])))
                    row.update(output_spacing=actual,source_sample_spacing=ref,spacing_error=actual-ref,spacing_error_over_D=(actual-ref)/m['diagonal'])
                pairs.append(row)
            span_error=np.ptp(output.vertices,axis=0)-np.ptp(source.vertices,axis=0)
            cache[signature]=dict(loops=loops,pairs=pairs,bbox_span_error=span_error.tolist(),bbox_span_error_over_D=(span_error/m['diagonal']).tolist())
        row=dict(**c,quad_sha256=signature[1],**cache[signature]);save(dest/(c['model']+'_'+c['condition']+'.json'),row);rows.append(row)
    save(D/'summary.json',rows);save(D/'integrity.json',dict(case_references=len(cases),unique_geometry=len(cache),source_frozen_unchanged=True))
    print('DIMENSIONS_COMPLETE',len(cases),len(cache),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','run']);a=p.parse_args();globals()[a.stage]()
