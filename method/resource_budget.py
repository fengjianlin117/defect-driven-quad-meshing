"""Separate density resource limits from the choice/importance of feature lines."""
import numpy as np


def redistribute_density(full_density, face_areas, target_area_ratio, gradation_slope=.5):
    """Allocate a fixed area proxy, permitting coarsening away from requests.

    rho = k / (1-alpha+alpha/rho_full); k fixes integral(rho^2).
    The corresponding size field has slope <= alpha/k times the input bound.
    Choose maximal alpha in [0,1] with alpha/k <=1. This is a one-parameter
    allocation family, not a globally optimal field or a feasibility guarantee.
    """
    # Reuse input validation; inputs represent the original refinement demands.
    limit_density(full_density,face_areas,target_area_ratio)
    if not np.isfinite(gradation_slope) or gradation_slope<0:
        raise ValueError('Nonnegative finite input gradation bound required')
    full=np.asarray(full_density,dtype=float);areas=np.asarray(face_areas,dtype=float)
    weights=areas/areas.sum()
    def family(alpha):
        base=1/(1-alpha+alpha/full)
        k=float(np.sqrt(target_area_ratio/(weights@(base*base))))
        return k*base,k
    alpha=1.;_,k=family(alpha)
    if alpha>k:
        low,high=0.,1.
        for _ in range(60):
            mid=(low+high)/2;_,k=family(mid)
            if mid<=k:low=mid
            else:high=mid
        alpha=low
    result,k=family(alpha)
    return result,dict(target_predicted_area_ratio=float(target_area_ratio),
                      allocated_predicted_area_ratio=float(weights@(result*result)),
                      contrast_fraction=alpha,global_density_scale=k,
                      size_gradation_bound=float(gradation_slope*alpha/k),
                      coarsened_area_fraction=float(weights[result<1-1e-12].sum()),
                      density_min=float(result.min()),density_max=float(result.max()),
                      unmet_original_face_request_fraction=float(weights[result<full-1e-12].sum()),
                      feature_selection_unchanged=True,
                      interpretation='Fixed integral budget with spatial redistribution; finite family and geometric grading only, actual count and constraint feasibility unproven.')


def limit_density(full_density, face_areas, max_area_ratio):
    """Retain the largest common size-relief fraction fitting an area proxy.

    h_alpha=(1-alpha)*h_baseline + alpha*h_requested.  Since alpha is common,
    taking the minimum vertex size commutes with this interpolation. Thus this
    also preserves (and reduces) the original edge-graph size gradient bound.
    This is an estimate of resource use, not a bound on extracted quad count.
    """
    full=np.asarray(full_density,dtype=float); areas=np.asarray(face_areas,dtype=float)
    if full.ndim!=1 or full.size==0 or areas.shape!=full.shape:
        raise ValueError('Matching nonempty one-dimensional density and area required')
    if not np.isfinite(full).all() or np.any(full<1) or not np.isfinite(areas).all() or np.any(areas<=0):
        raise ValueError('Finite density >=1 and positive finite areas required')
    if not np.isfinite(max_area_ratio) or max_area_ratio<1:
        raise ValueError('Finite area ratio >=1 required')
    weights=areas/areas.sum()
    def density(alpha):return 1/(1-alpha+alpha/full)
    def ratio(alpha):return float(weights@(density(alpha)**2))
    original=ratio(1.)
    if original<=max_area_ratio:alpha=1.
    else:
        low,high=0.,1.
        for _ in range(60):
            mid=(low+high)/2
            if ratio(mid)<=max_area_ratio:low=mid
            else:high=mid
        alpha=low
    result=density(alpha)
    return result,dict(max_predicted_area_ratio=float(max_area_ratio),full_predicted_area_ratio=original,
                      realized_predicted_area_ratio=ratio(alpha),retained_size_relief_fraction=alpha,
                      extra_size_shortfall_max_h=float(((1-alpha)*(1-1/full)).max()),
                      feature_selection_unchanged=True,
                      interpretation='Common size interpolation; unfulfilled sizing demand is explicit. Actual quad count must be checked separately.')
