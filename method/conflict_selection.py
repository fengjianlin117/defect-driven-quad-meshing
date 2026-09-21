"""Greedy whole-group selection with exact equality-closure collapse screening.

Screening concerns a fixed baseline cut/combed frame. It is a sufficient reason
to avoid a hard-constraint set, never a guarantee about resulting quad meshes.
Reference geometry, feature importance and source-wide sizes are not deleted.
"""
import numpy as np
from weak_layout_pipeline.pipeline.mesh import edge_topology,triangle_geometry

class EqualityAudit:
    def __init__(self,mesh,uv,fuv,pd1,pd2):
        self.fuv=np.asarray(fuv,int);self.n=len(uv);self.links={};self.cache={}
        for fi,face in enumerate(mesh.faces):
            for j in range(3):
                a,b=int(face[j]),int(face[(j+1)%3]);edge=tuple(sorted((a,b)))
                t=mesh.vertices[b]-mesh.vertices[a];t=t/np.linalg.norm(t)
                axis=int(abs(t@(pd1[fi]/np.linalg.norm(pd1[fi])))>abs(t@(pd2[fi]/np.linalg.norm(pd2[fi]))))
                self.links.setdefault(edge,[]).append((axis,int(fuv[fi,j]),int(fuv[fi,(j+1)%3])))
    def check(self,edges):
        key=frozenset(tuple(sorted(map(int,e))) for e in edges)
        if key in self.cache:return self.cache[key]
        unknown=key-self.links.keys()
        if unknown:raise ValueError('Hard constraint is not a source edge')
        parents=np.tile(np.arange(self.n),(2,1))
        def find(axis,i):
            while parents[axis,i]!=i:parents[axis,i]=parents[axis,parents[axis,i]];i=parents[axis,i]
            return i
        for edge in sorted(key):
            for axis,a,b in self.links[edge]:parents[axis,find(axis,a)]=find(axis,b)
        classes=np.array([[find(axis,i) for i in range(self.n)] for axis in (0,1)])
        collapsed=np.flatnonzero(np.any(np.all(classes[:,self.fuv]==classes[:,self.fuv[:,0]][:,:,None],axis=2),axis=0)).tolist()
        result=dict(forced_zero_triangle_count=len(collapsed),forced_zero_triangles=collapsed,
                    admissible=len(collapsed)==0,assumption='Fixed baseline cut and combed axes; seam translations and full integer feasibility omitted.')
        self.cache[key]=result;return result

def select(context,audit,fraction,method='conflict_aware'):
    if not 0<=fraction<=1:raise ValueError('fraction must be in [0,1]')
    if method not in ['legacy_defect','conflict_aware','mean_defect','sharpness','all_features','no_features']:raise ValueError('unknown selection')
    if method=='legacy_defect':
        result=context.select(fraction)[1];result.update(method=method,screening=audit.check(result['selected_edges']));return result
    selected=set();chosen=[];trace=[];rejected=[];blocked=set();budget=fraction*context.total_length
    normals=triangle_geometry(context.mesh)[0];topology=edge_topology(context.mesh)[0]
    sharpness={}
    for edge in context.all_edges:
        faces=[f for f,_ in topology[edge]]
        sharpness[edge]=float(np.arccos(np.clip(normals[faces[0]]@normals[faces[1]],-1,1))) if len(faces)==2 else np.pi
    if method=='all_features':chosen=sorted(context.groups);selected=set(context.all_edges)
    elif method!='no_features':
        while True:
            used=sum(context.length[e] for e in selected);options=[]
            for gid,edges in context.group_edges.items():
                if gid in chosen or gid in blocked:continue
                added=edges-selected;cost=sum(context.length[e] for e in added)
                if cost<1e-12 or used+cost>budget+1e-10:continue
                gain=sum(context.length[e]*context.loss[e] for e in added)
                if method=='conflict_aware':
                    if gain<1e-12:continue
                    _,sizing=context.size(chosen+[gid]);pressure=sizing['resolution_pressure']
                    field=sum(context.length[e]*context.field_risk[e] for e in added)/cost
                    key=(context.residual_profile(selected|edges),round(pressure['adapted_max'],12),round(pressure['adapted_mean'],12),round(field,12),round(sizing['predicted_area_budget_ratio'],12),round(cost,12),gid)
                else:
                    score=(gain/cost) if method=='mean_defect' else sum(context.length[e]*sharpness[e] for e in added)/cost
                    key=(-round(score,12),round(cost,12),gid)
                options.append((key,gid,added,gain,cost))
            if not options:break
            accepted=None
            for key,gid,added,gain,cost in sorted(options,key=lambda x:x[0]):
                risk=audit.check(selected|context.group_edges[gid])
                if method=='conflict_aware' and not risk['admissible']:
                    # Coordinate equality constraints only accumulate: an already
                    # collapsing union cannot become noncollapsing by adding groups.
                    rejected.append(dict(group_id=gid,with_groups=chosen.copy(),risk=risk));blocked.add(gid);continue
                accepted=(gid,added,gain,cost,risk);break
            if accepted is None:break
            gid,added,gain,cost,risk=accepted;chosen.append(gid);selected.update(added)
            trace.append(dict(group_id=gid,conditional_opportunity=gain,added_length_h=cost,screening=risk))
    return dict(method=method,budget_fraction=1. if method=='all_features' else fraction,budget_length_h=context.total_length if method=='all_features' else budget,
                selected_group_ids=chosen,selected_edges=[list(e) for e in sorted(selected)],selected_length_h=sum(context.length[e] for e in selected),
                conditional_opportunity=sum(context.length[e]*context.loss[e] for e in selected),conditional_residual_profile=context.residual_profile(selected),
                screening=audit.check(selected),trace=trace,rejected_hard_groups=rejected,
                reference_policy='All original main features remain in the reference graph and source-wide size demand; rejected groups are not deleted.',
                limit='Greedy whole-group heuristic, not maximum-benefit optimization or global feasibility certification.')
