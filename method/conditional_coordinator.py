"""Demand-driven use of the frozen defect/size proposal and bounded feedback."""
from coordinator_v5 import CoordinatorV5, strict_decision, save, read
from defect_driven_selection import needs_repair

class InitialProposalAccepted(Exception):
    """Internal control flow: the complete initial proposal phase suffices."""

class ConditionalCoordinator(CoordinatorV5):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.requested_variant != 'full':
            raise ValueError('Conditional policy currently supports the full frozen modules only')
        self.exit_stage = None
        policy=read(self.out/'policy.json')
        policy.update(activation='Inspect baseline source-feature defects. If none need repair and baseline is eligible, retain it without backend calls. Otherwise generate the existing one-shot proposal phase; invoke bounded feedback only if that phase has no recommended nonbaseline candidate.',
                      initial_success='Use the unchanged strict_decision gate and ranking, with only baseline and initial-phase outputs available; no threshold tuned for this policy.',
                      initial_phase_cost='One defect/size proposal, uniform and allocated diagnostic branches with at most one count correction each; one-shot does not mean exactly one backend invocation.',
                      feedback_cost='Same two productive rounds and twelve total logical attempt cap as frozen v5.',
                      status='New development policy prompted by user narrative; not an unseen validation or the final method freeze.')
        save(self.out/'policy.json',policy)

    def finish_early(self,stage):
        self.exit_stage=stage
        decision=strict_decision(self.rows,self.max_quads,self.min_quads)
        self.trace.append(dict(stop=stage,activation_policy='conditional'))
        self.flush();save(self.out/'decision.json',decision)
        save(self.out/'complete.json',dict(attempts=self.attempts,native_calls=self.native,reused_attempts=self.reused,
            recommended=decision['recommended_id'],all_candidates_retained=True,reference_edges=len(self.ctx.all_edges),
            variant='conditional',exit_stage=stage,source_geometry_clear=self.source_geometry_clear,parent_visits=0,
            geometry_checks=self.geometry_checks,geometry_reused=self.geometry_reused))
        return decision

    def branch(self,name,*args,**kwargs):
        super().branch(name,*args,**kwargs)
        if name=='initial_allocated':
            decision=strict_decision(self.rows,self.max_quads,self.min_quads)
            if decision['recommended_id'] not in [None,'baseline']:
                raise InitialProposalAccepted()

    def run(self):
        deficient=any(needs_repair(r) for r in self.base['edge_defects'] if r['layer']=='main')
        base_decision=strict_decision(self.rows,self.max_quads,self.min_quads)
        if self.source_geometry_clear and not deficient and base_decision['recommended_id']=='baseline':
            return self.finish_early('baseline_no_diagnostic_feature_defect')
        try:
            decision=super().run()
        except InitialProposalAccepted:
            return self.finish_early('initial_proposal_accepted')
        self.exit_stage='feedback_or_bounded_fallback' if self.source_geometry_clear else 'source_rejected'
        complete=read(self.out/'complete.json')
        complete.update(variant='conditional',exit_stage=self.exit_stage)
        save(self.out/'complete.json',complete)
        return decision
