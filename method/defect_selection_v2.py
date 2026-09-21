"""Pre-backend joint constraint/size proposals; heuristic, not a feasibility proof.

No model names, axes, historical annotations, or constrained outputs are inputs
to the objective. Feature importance is deliberately left equal within the
already accepted main graph. Size envelopes reuse classical grading principles.
"""
import heapq
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from weak_layout_pipeline.pipeline.mesh import unique_edges, triangle_geometry, edge_topology
from feature_diagnostics import edge_set

CONFIG=dict(strip_rows=2.,gradation_slope=.5,minimum_h_fraction=.25,
            constraint_budget_fractions=[.25,.5,.75],size_cost_weight=1.,field_risk_weight=1.,
            interpretation='Pre-backend heuristic priority; conditional opportunity is not certified realized benefit.',
            importance='Equal main-feature edge importance; no additional dihedral multiplier.',
            sizing_source='Persson, Mesh size functions for implicit geometries and PDE-based gradient limiting; discrete edge-graph envelope, not a reproduction of the medial-axis/PDE solver.')

def graded_envelope(caps, edges, lengths, slope):
    """Pointwise largest edge-graph Lipschitz field bounded by the supplied caps."""
    h=np.asarray(caps,float).copy()
    if not np.isfinite(h).all() or np.any(h<=0):raise ValueError('positive finite caps required')
    adj=[[] for _ in h]
    for (a,b),length in zip(edges,lengths):
        adj[a].append((b,slope*length));adj[b].append((a,slope*length))
    queue=[(float(x),i) for i,x in enumerate(h)];heapq.heapify(queue)
    while queue:
        value,a=heapq.heappop(queue)
        if value>h[a]:continue
        for b,cost in adj[a]:
            proposed=value+cost
            if proposed<h[b]:h[b]=proposed;heapq.heappush(queue,(proposed,b))
    return h

class ProposalContext:
    def __init__(self,mesh,graph,records,pd1,pd2,h):
        self.mesh,self.graph,self.h=mesh,graph,float(h)
        self.groups={g['id']:g for g in graph['groups'] if g['layer']=='main'}
        self.group_edges={k:edge_set([g]) for k,g in self.groups.items()}
        self.all_edges=set().union(*self.group_edges.values()) if self.groups else set()
        self.records={tuple(sorted(e['vertices'])):e for e in records}
        self.length={e:self.records[e]['length_h'] for e in self.all_edges}
        self.loss={e:.5*(self.records[e]['surface_loss']+self.records[e]['unaligned_fraction']) for e in self.all_edges}
        self.total_length=sum(self.length.values())
        self.total_opportunity=sum(self.length[e]*self.loss[e] for e in self.all_edges)
        self.edges=unique_edges(mesh)
        self.edge_lengths=np.linalg.norm(mesh.vertices[self.edges[:,0]]-mesh.vertices[self.edges[:,1]],axis=1)
        self.areas=triangle_geometry(mesh)[1]
        adj=coo_matrix((np.r_[self.edge_lengths,self.edge_lengths],(np.r_[self.edges[:,0],self.edges[:,1]],np.r_[self.edges[:,1],self.edges[:,0]])),shape=(mesh.vertex_count,mesh.vertex_count)).tocsr()
        self.vertices={k:np.unique(g['mesh_vertices']) for k,g in self.groups.items()}
        self.dist={k:dijkstra(adj,directed=False,indices=v,min_only=True) for k,v in self.vertices.items()}
        self.pairs=[];self.touching=[]
        keys=sorted(self.groups)
        for i,a in enumerate(keys):
            for b in keys[i+1:]:
                if np.intersect1d(self.vertices[a],self.vertices[b]).size:
                    self.touching.append((a,b));continue
                separation=float(self.dist[a][self.vertices[b]].min())
                if separation<CONFIG['strip_rows']*self.h:
                    self.pairs.append(dict(a=a,b=b,edge_path_separation_h=separation/self.h))
        topology=edge_topology(mesh)[0];self.field_risk={};self.field_angle={}
        for e in self.all_edges:
            tangent=mesh.vertices[e[1]]-mesh.vertices[e[0]];tangent/=np.linalg.norm(tangent)
            angles=[]
            for f,_ in topology[e]:
                x=pd1[f]/np.linalg.norm(pd1[f]);y=pd2[f]/np.linalg.norm(pd2[f])
                cosine=np.clip(max(abs(tangent@x),abs(tangent@y)),0,1)
                angles.append(float(np.arccos(cosine)))
            angle=max(angles);self.field_angle[e]=np.degrees(angle)
            self.field_risk[e]=float(np.sin(2*angle)**2)
        self.size_cache={}

    def size(self,chosen):
        key=frozenset(chosen)
        if key in self.size_cache:return self.size_cache[key]
        requested=np.full(self.mesh.vertex_count,self.h);pair_rows=[]
        for pair in self.pairs:
            a,b=pair['a'],pair['b']
            if a not in key and b not in key:continue
            for own,other in [(a,b),(b,a)]:
                ids=self.vertices[own]
                requested[ids]=np.minimum(requested[ids],self.dist[other][ids]/CONFIG['strip_rows'])
            pair_rows.append(dict(pair,kind='both_selected' if a in key and b in key else 'unselected_neighbor_protection'))
        floor=CONFIG['minimum_h_fraction']*self.h
        clipped=requested<floor-1e-12*self.h
        applied=np.maximum(requested,floor)
        hv=graded_envelope(applied,self.edges,self.edge_lengths,CONFIG['gradation_slope'])
        rho=self.h/hv[self.mesh.faces].min(axis=1)
        boundary_slope=float(np.max(np.abs(hv[self.edges[:,0]]-hv[self.edges[:,1]])/self.edge_lengths))
        result=dict(density_min=float(rho.min()),density_max=float(rho.max()),
                    predicted_area_budget_ratio=float(self.areas@(rho*rho)/self.areas.sum()),
                    requested_min_h_fraction=float(requested.min()/self.h),applied_min_h_fraction=float(hv.min()/self.h),
                    clipped_request_vertices=int(clipped.sum()),unmet_request_vertices=int((hv>requested+1e-10*self.h).sum()),
                    maximum_edge_graph_slope=boundary_slope,pairs=pair_rows,
                    unresolved_touching_pairs=sum(a in key or b in key for a,b in self.touching),
                    status='unresolved_size_requests' if clipped.any() else 'graph_size_caps_satisfied; backend_feasibility_unproven',
                    limitations=['Edge paths upper-bound surface geodesics and can miss narrow regions.',
                                 'Touching curve junctions, singularity layout, integer periods and injectivity are not certified.',
                                 'Face density uses minimum vertex size; edge-graph grading is not a continuum gradient certificate.',
                                 'Refining an unselected neighboring curve does not guarantee preservation.'])
        self.size_cache[key]=(rho,result);return rho,result

    def select(self,fraction):
        budget=fraction*self.total_length;chosen=[];selected=set();trace=[]
        current_size=1.
        while True:
            used=sum(self.length[e] for e in selected);options=[]
            for gid,group_edges in self.group_edges.items():
                if gid in chosen:continue
                added=group_edges-selected;cost=sum(self.length[e] for e in added)
                if cost<1e-12 or used+cost>budget+1e-10:continue
                gain=sum(self.length[e]*self.loss[e] for e in added)
                if gain<1e-12:continue
                field=sum(self.length[e]*self.field_risk[e] for e in added)/cost
                _,sizing=self.size(chosen+[gid])
                extra_size=max(0.,sizing['predicted_area_budget_ratio']-current_size)
                denominator=cost/self.total_length+CONFIG['size_cost_weight']*extra_size
                score=(gain/max(self.total_opportunity,1e-30))/denominator/(1+CONFIG['field_risk_weight']*field)
                options.append((score,gain,gid,cost,field,sizing))
            if not options:break
            score,gain,gid,cost,field,sizing=max(options,key=lambda x:(x[0],x[1],x[2]))
            chosen.append(gid);selected|=self.group_edges[gid];current_size=sizing['predicted_area_budget_ratio']
            trace.append(dict(group_id=gid,priority=score,conditional_incremental_opportunity=gain,added_length_h=cost,
                              mean_field_mismatch_risk=field,predicted_area_budget_ratio=current_size,
                              unresolved_request_vertices=sizing['unmet_request_vertices']))
        rho,sizing=self.size(chosen)
        selected_length=sum(self.length[e] for e in selected)
        field=sum(self.length[e]*self.field_risk[e] for e in selected)/max(selected_length,1e-30)
        opportunity=sum(self.length[e]*self.loss[e] for e in selected)
        groups=[]
        for gid,es in self.group_edges.items():
            length=sum(self.length[e] for e in es)
            deficit=sum(self.length[e]*self.loss[e] for e in es)/max(length,1e-30)
            groups.append(dict(group_id=gid,selected=gid in chosen,length_h=length,mean_baseline_deficit=deficit,
                               field_angle_max_degrees=max(self.field_angle[e] for e in es),
                               residual_conditional_deficit=sum(self.length[e]*self.loss[e] for e in es-selected)/max(length,1e-30)))
        return rho,dict(budget_fraction=fraction,budget_length_h=budget,selected_group_ids=chosen,
                        selected_edges=[list(e) for e in sorted(selected)],selected_length_h=selected_length,
                        trace=trace,size=sizing,group_diagnostics=groups,
                        predicted=dict(conditional_opportunity=opportunity,
                                       field_discounted_opportunity=opportunity/(1+field),
                                       mean_field_mismatch_risk=field,total_main_deficit=self.total_opportunity),
                        guarantees=dict(integer_feasible=False,nondegenerate=False,unselected_features_preserved=False),
                        empty_graph_policy='No line candidates; retain full surface and mesh diagnostics, never automatic acceptance.',
                        interpretation=CONFIG['interpretation'])
