"""Independent adjacency/path certificate for fixed-frame equality collapse."""
from collections import defaultdict, deque
from pathlib import Path
import sys
import numpy as np
from audit_evidence import OLD, OUT, read, save, sha
sys.path.insert(0,str(OLD/'defect_driven_v3/method'))
from weak_layout_pipeline.pipeline.mesh import load_obj

def path(adj,start,end):
    todo=deque([start]);parents={start:None}
    while todo:
        at=todo.popleft()
        if at==end:
            route=[]
            while parents[at] is not None:
                prev,e=parents[at];route.append(dict(from_uv=prev,to_uv=at,source_edge=list(e)));at=prev
            return route[::-1]
        for nxt,e in adj[at]:
            if nxt not in parents:parents[nxt]=(at,e);todo.append(nxt)
    return None

def main():
    target=OUT/'constraint_certificates';target.mkdir(exist_ok=False);rows=[]
    src=Path('research://backend-dependencies/libs/libigl/include/igl/copyleft/comiso/miq.cpp')
    save(target/'backend_rule.json',dict(source=str(src),sha256=sha(src),function='addSharpEdgeConstraint',line=1145,
        rule='Choose coordinate offset=1 when abs(t dot normalized PD1)>abs(t dot normalized PD2), else offset=0; impose equal coordinate and integer-round both endpoints.',
        certificate_scope='Only coordinate equality; round/seam/global integer feasibility are not certified.'))
    for p in read(OUT/'audit/source_plans.json'):
        loc=read(OUT/f'localization/{p["model"]}.json')
        if not loc['conflicts']:continue
        mesh=load_obj(p['source']);graph=read(p['graph']);inp=Path(p['source']).parent/'baseline'
        fuv=np.loadtxt(inp/'miq_fuv.txt',skiprows=1,dtype=int);pd1=np.loadtxt(inp/'miq_combed_PD1.txt')[:,-3:];pd2=np.loadtxt(inp/'miq_combed_PD2.txt')[:,-3:]
        for conflict in loc['conflicts']:
            gid=conflict['deferred'];edges={tuple(sorted(r['vertices'])) for r in graph['source_edges'] if gid in r['group_ids']}
            adj=[defaultdict(list),defaultdict(list)]
            for fi,face in enumerate(mesh.faces):
                for j in range(3):
                    a,b=int(face[j]),int(face[(j+1)%3]);e=tuple(sorted((a,b)))
                    if e not in edges:continue
                    t=mesh.vertices[b]-mesh.vertices[a];t/=np.linalg.norm(t)
                    axis=1 if abs(t@(pd1[fi]/np.linalg.norm(pd1[fi])))>abs(t@(pd2[fi]/np.linalg.norm(pd2[fi]))) else 0
                    u,v=int(fuv[fi,j]),int(fuv[fi,(j+1)%3]);adj[axis][u].append((v,e));adj[axis][v].append((u,e))
            triangles=[]
            for fi,face in enumerate(fuv):
                for axis in range(2):
                    p01=path(adj[axis],int(face[0]),int(face[1]));p02=path(adj[axis],int(face[0]),int(face[2]))
                    if p01 is not None and p02 is not None:
                        witnesses=sorted({tuple(x['source_edge']) for x in p01+p02})
                        triangles.append(dict(source_triangle=fi,source_vertices=mesh.faces[fi].tolist(),uv_vertices=face.tolist(),
                            constant_coordinate=axis,paths=[p01,p02],witness_source_edges=[list(e) for e in witnesses],
                            conclusion='All three vertices have the same UV coordinate on this axis, so the affine UV triangle has zero signed area.'))
                        break
            expected=conflict['witness_check']['forced_zero_triangles'];assert [x['source_triangle'] for x in triangles]==expected
            row=dict(model=p['model'],group=gid,group_edge_count=len(edges),triangles=triangles,
                independently_matches_union_find=True,scope='fixed cut and combed axes only')
            save(target/f'{p["model"]}_{gid}.json',row);rows.append(row)
            print('CERTIFICATE',p['model'],gid,'edges',len(edges),'triangles',len(triangles),'witness_sizes',[len(x['witness_source_edges']) for x in triangles],flush=True)
    save(target/'summary.json',rows)
if __name__=='__main__':main()
