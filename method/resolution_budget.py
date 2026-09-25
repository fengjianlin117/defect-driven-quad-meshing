"""Explicit starting resolution, separate from defect-driven local allocation.

Source face count is a conservative sampling heuristic, not a fidelity bound.
Historical coordinators retain their old defaults unless a new plan supplies
the count policy; the public generation entry point uses this policy by default.
"""
import math


def estimate_resolution(triangles, baseline_quads, target_quads=None, max_quads=8192):
    for name, value in [('triangles', triangles), ('baseline_quads', baseline_quads),
                        ('max_quads', max_quads)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if target_quads is not None and (isinstance(target_quads, bool)
            or not isinstance(target_quads, int) or target_quads <= 0):
        raise ValueError('target_quads must be a positive integer')
    target = target_quads if target_quads is not None else max(512, baseline_quads, (triangles + 1)//2)
    if target > max_quads:
        raise ValueError(f'Requested/estimated {target} quads exceeds cap {max_quads}; '
                         'set --max-quads higher or choose an explicit --target-quads.')
    return dict(policy='source_half_v1' if target_quads is None else 'explicit_target',
                source_triangles=triangles, archived_baseline_quads=baseline_quads,
                target_quads=target, min_actual_quads=math.ceil(.9*target),
                max_actual_quads=max_quads, count_tolerance_fraction=.10,
                max_baseline_attempts=3,
                limitation='Triangle count depends on input tessellation; this starting '
                           'resolution is not an optimal count or an approximation-error guarantee.')


def count_in_band(actual, policy):
    target = policy['target_quads']
    return (policy['min_actual_quads'] <= actual <= policy['max_actual_quads']
            and abs(actual/target - 1) <= policy['count_tolerance_fraction'] + 1e-12)
