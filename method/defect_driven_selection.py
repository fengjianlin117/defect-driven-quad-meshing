"""Baseline-diagnosed need, whole-group proposals, no feature-length quota.

The existing diagnostic resolution (0.1 h, 15 degrees) is a research default,
not a calibrated CAD tolerance. Selection is a one-shot conditional proposal;
adding a constraint is not evidence that its defect has actually been repaired.
"""
import math
from defect_predictor_v1 import CONFIG as DIAGNOSTIC


def needs_repair(record):
    missing = float(record['unaligned_fraction'])
    distance = float(record['surface_distance_p95_h'])
    if not math.isfinite(missing) or not 0 <= missing <= 1:
        raise ValueError('Invalid measured alignment deficiency')
    if not math.isfinite(distance) or distance < 0:
        raise ValueError('Invalid measured surface distance')
    return missing > 0 or distance > DIAGNOSTIC['distance_fraction_h']


def select_defect_driven(context, audit):
    deficient = {e for e in context.all_edges if needs_repair(context.records[e])}
    selected, chosen, blocked = set(), [], set()
    trace, rejected = [], []
    # Retain the previous prioritization and size tie-breaks; replace only the
    # eligibility/stopping rule. No quota parameter exists in this interface.
    while True:
        options = []
        for gid, edges in context.group_edges.items():
            if gid in chosen or gid in blocked:
                continue
            needed = (edges - selected) & deficient
            if not needed:
                continue
            added = edges - selected
            cost = sum(context.length[e] for e in added)
            _, sizing = context.size(chosen + [gid])
            pressure = sizing['resolution_pressure']
            field = sum(context.length[e] * context.field_risk[e] for e in added) / cost
            key = (context.residual_profile(selected | edges),
                   round(pressure['adapted_max'], 12), round(pressure['adapted_mean'], 12),
                   round(field, 12), round(sizing['predicted_area_budget_ratio'], 12),
                   round(cost, 12), gid)
            options.append((key, gid, added, needed))
        if not options:
            break
        accepted = False
        for _, gid, added, needed in sorted(options, key=lambda r: r[0]):
            risk = audit.check(selected | context.group_edges[gid])
            if not risk['admissible']:
                blocked.add(gid)
                rejected.append(dict(group_id=gid, with_groups=chosen.copy(), risk=risk,
                                     reason='forced_zero_triangle_on_baseline_cut_and_axes'))
                continue
            chosen.append(gid)
            selected.update(added)
            trace.append(dict(group_id=gid, newly_addressed_deficient_edges=len(needed),
                              added_length_h=sum(context.length[e] for e in added),
                              conditional_opportunity=sum(context.length[e]*context.loss[e] for e in added),
                              screening=risk))
            accepted = True
            break
        if not accepted:
            break
    diagnostics = []
    for gid, edges in sorted(context.group_edges.items()):
        need = edges & deficient
        status = ('selected' if gid in chosen else
                  'no_diagnosed_deficiency' if not need else
                  'covered_by_other_selected_groups' if not need-selected else
                  'deferred_combination_conflict')
        diagnostics.append(dict(group_id=gid, status=status, deficient_edge_count=len(need),
                                unaddressed_deficient_edge_count=len(need-selected)))
    return dict(method='defect_driven', budget_fraction=None, budget_length_h=None,
                selected_group_ids=chosen, selected_edges=[list(e) for e in sorted(selected)],
                selected_length_h=sum(context.length[e] for e in selected),
                observed_selected_length_fraction=sum(context.length[e] for e in selected)/context.total_length if context.total_length else 0.,
                deficient_edges=[list(e) for e in sorted(deficient)],
                unaddressed_deficient_edges=[list(e) for e in sorted(deficient-selected)],
                group_diagnostics=diagnostics, trace=trace, rejected_hard_groups=rejected,
                conditional_residual_profile=context.residual_profile(selected), screening=audit.check(selected),
                stopping_reason='No additional diagnostically deficient group passes the fixed-frame combination screen.',
                diagnostic_policy=dict(distance_fraction_h=DIAGNOSTIC['distance_fraction_h'],
                                       tangent_degrees=DIAGNOSTIC['tangent_degrees'],
                                       predicate='At least one sampled point lacks a tangent-aligned edge within 0.1 h, or source-edge surface-distance p95 exceeds 0.1 h.',
                                       origin='Unchanged pre-existing research diagnostic resolution; not fitted to new outputs or calibrated CAD tolerances.'),
                reference_policy='Keep all reference features and all source-wide sizing requests, including deferred groups.',
                limit='Whole-group greedy one-shot proposal; predicted coverage is not measured repair. No global feasibility or optimal subset guarantee.')
