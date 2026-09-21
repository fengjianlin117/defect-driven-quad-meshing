"""Prioritize the largest residual feature deficits before comparing costs.

The zero-after-constraint hypothesis is conditional, not an actual repair model.
Size and field risks remain explicit; no historical feature identities are read.
"""
import numpy as np
from defect_selection_v2 import ProposalContext as SizeContext, CONFIG as SIZE_CONFIG, graded_envelope
CONFIG=dict(SIZE_CONFIG,objective='Lexicographically minimize sorted length-normalized per-group residual defects under a unique-edge length budget.',
            cost_role='For equal geometric residual profiles compare remaining size pressure AFTER adaptation, field mismatch, then area/line resource costs.',
            sizing_role='An enabling operation: explicitly measure resolution-pressure relief; refinement demand never directly divides geometric benefit.',
            remaining_limit='Feasibility risks only break equal-benefit ties and are reported; global integer/injectivity feasibility remains unknown.')

class ProposalContext(SizeContext):
    def size(self,chosen):
        key=frozenset(chosen)
        if not hasattr(self,'relief_cache'):self.relief_cache={}
        if key in self.relief_cache:return self.relief_cache[key]
        rho,base=super().size(chosen);result=dict(base)
        requested=np.full(self.mesh.vertex_count,self.h)
        for pair in self.pairs:
            a,b=pair['a'],pair['b']
            if a not in key and b not in key:continue
            for own,other in [(a,b),(b,a)]:
                ids=self.vertices[own]
                requested[ids]=np.minimum(requested[ids],self.dist[other][ids]/SIZE_CONFIG['strip_rows'])
        applied=np.maximum(requested,SIZE_CONFIG['minimum_h_fraction']*self.h)
        hv=graded_envelope(applied,self.edges,self.edge_lengths,SIZE_CONFIG['gradation_slope'])
        active=requested<self.h-1e-12*self.h
        weights=np.zeros(self.mesh.vertex_count)
        for edge,length in self.length.items():
            for vertex in edge:weights[vertex]+=.5*length
        weights*=active
        before=np.maximum(0.,1.-requested/self.h)
        after=np.maximum(0.,1.-requested/hv)
        weight_sum=float(weights.sum())
        mean_before=float(weights@before/max(weight_sum,1e-30))
        mean_after=float(weights@after/max(weight_sum,1e-30))
        result['resolution_pressure']=dict(uniform_mean=mean_before,adapted_mean=mean_after,
                                          uniform_max=float(before.max()),adapted_max=float(after.max()),
                                          mean_relief=mean_before-mean_after,
                                          relief_fraction=(mean_before-mean_after)/mean_before if mean_before>0 else None,
                                          interpretation='Geometric two-row sizing proxy only; does not predict removal of field, integer-period, UV or final-quad conflicts.')
        result['resource_accounting']=dict(additional_predicted_area_ratio=max(0.,result['predicted_area_budget_ratio']-1.),
                                           role='Report and budget separately; not a direct penalty on feature importance.')
        self.relief_cache[key]=(rho,result)
        return rho,result

    def residual_profile(self,selected):
        profile=[]
        for es in self.group_edges.values():
            length=sum(self.length[e] for e in es)
            value=sum(self.length[e]*self.loss[e] for e in es-selected)/max(length,1e-30)
            profile.append(round(value,12))
        return tuple(sorted(profile,reverse=True))

    def select(self,fraction):
        budget=fraction*self.total_length;chosen=[];selected=set();trace=[]
        baseline_profile=self.residual_profile(selected)
        while True:
            used=sum(self.length[e] for e in selected);options=[]
            for gid,es in self.group_edges.items():
                if gid in chosen:continue
                added=es-selected;cost=sum(self.length[e] for e in added)
                if cost<1e-12 or used+cost>budget+1e-10:continue
                gain=sum(self.length[e]*self.loss[e] for e in added)
                if gain<1e-12:continue
                profile=self.residual_profile(selected|es)
                _,sizing=self.size(chosen+[gid])
                field=sum(self.length[e]*self.field_risk[e] for e in added)/cost
                remaining=sizing['resolution_pressure']
                key=(profile,round(remaining['adapted_max'],12),round(remaining['adapted_mean'],12),round(field,12),
                     round(sizing['predicted_area_budget_ratio'],12),round(cost,12),gid)
                options.append((key,gid,cost,gain,field,sizing))
            if not options:break
            key,gid,cost,gain,field,sizing=min(options,key=lambda x:x[0])
            chosen.append(gid);selected|=self.group_edges[gid]
            trace.append(dict(group_id=gid,residual_profile=list(key[0]),conditional_incremental_opportunity=gain,
                              added_length_h=cost,mean_field_mismatch_risk=field,
                              predicted_area_budget_ratio=sizing['predicted_area_budget_ratio'],
                              resolution_pressure=sizing['resolution_pressure'],
                              unresolved_request_vertices=sizing['unmet_request_vertices']))
        rho,sizing=self.size(chosen);selected_length=sum(self.length[e] for e in selected)
        field=sum(self.length[e]*self.field_risk[e] for e in selected)/max(selected_length,1e-30)
        opportunity=sum(self.length[e]*self.loss[e] for e in selected)
        groups=[]
        for gid,es in self.group_edges.items():
            length=sum(self.length[e] for e in es)
            groups.append(dict(group_id=gid,selected=gid in chosen,length_h=length,
                               mean_baseline_deficit=sum(self.length[e]*self.loss[e] for e in es)/max(length,1e-30),
                               field_angle_max_degrees=max(self.field_angle[e] for e in es),
                               residual_conditional_deficit=sum(self.length[e]*self.loss[e] for e in es-selected)/max(length,1e-30)))
        return rho,dict(budget_fraction=fraction,budget_length_h=budget,selected_group_ids=chosen,
                        selected_edges=[list(e) for e in sorted(selected)],selected_length_h=selected_length,
                        trace=trace,size=sizing,group_diagnostics=groups,baseline_residual_profile=baseline_profile,
                        conditional_residual_profile=self.residual_profile(selected),
                        predicted=dict(conditional_opportunity=opportunity,field_discounted_opportunity=opportunity/(1+field),
                                       mean_field_mismatch_risk=field,total_main_deficit=self.total_opportunity),
                        guarantees=dict(integer_feasible=False,nondegenerate=False,unselected_features_preserved=False),
                        empty_graph_policy='No line candidates; retain full surface and mesh diagnostics, never automatic acceptance.',
                        interpretation=CONFIG['interpretation'],remaining_limit=CONFIG['remaining_limit'])
