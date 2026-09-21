"""Independent final-output review. UV and projected quality are diagnostics.

For a bilinear patch p=a+bu+cv+duv, its area vector is affine:
J(u,v)=b x c + u(b x d) + v(d x c). Minimize its squared norm
on the unit square using the interior least-squares solution and four edges.
This is a local rank test, NOT a global intersection certificate.
"""
import numpy as np
from collections import defaultdict


def bilinear_minimum(points):
    p = np.asarray(points, float)
    scale = max(float(np.max(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1))), 1e-150)
    p = (p-p[0])/scale
    b,c = p[1],p[3]
    d = p[2]-b-c
    j = np.cross(b,c)
    m = np.column_stack((np.cross(b,d),np.cross(d,c)))
    candidates = [np.array([u,v],float) for u in (0,1) for v in (0,1)]
    uv = np.linalg.lstsq(m,-j,rcond=None)[0]
    if np.all(uv>=0) and np.all(uv<=1): candidates.append(uv)
    for fixed_axis in (0,1):
        free = 1-fixed_axis
        den = float(m[:,free]@m[:,free])
        for value in (0.,1.):
            uv = np.zeros(2); uv[fixed_axis]=value
            uv[free] = np.clip(-float(m[:,free]@(j+value*m[:,fixed_axis]))/den,0,1) if den>0 else 0
            candidates.append(uv)
    norms = [float(np.linalg.norm(j+m@x)) for x in candidates]
    k = int(np.argmin(norms))
    return norms[k],candidates[k].tolist()


def review_mesh(mesh, source, topology, quality, uv=None):
    incidents = defaultdict(list)
    for fid,face in enumerate(mesh.faces):
        for a,b in zip(face,np.roll(face,-1)):
            incidents[tuple(sorted((int(a),int(b))))].append((fid,1 if a<b else -1))
    orientation_conflicts = sum(len(v)==2 and v[0][1]==v[1][1] for v in incidents.values())
    used = np.unique(mesh.faces)
    used_euler = len(used)-len(incidents)+mesh.face_count
    source_edges = {tuple(sorted((int(a),int(b)))) for f in source.faces for a,b in zip(f,np.roll(f,-1))}
    source_euler = len(np.unique(source.faces))-len(source_edges)+source.face_count
    minima=[]; singular=[]
    for fid,face in enumerate(mesh.faces):
        if len(face)!=4: continue
        value,where = bilinear_minimum(mesh.vertices[face]); minima.append(value)
        if value<=1e-10: singular.append(dict(face=fid,minimum_normalized_area=value,uv=where))
    failures=[]
    for key in ('pure_quad','closed','connected','edge_manifold','vertex_manifold'):
        if not topology[key]: failures.append(key)
    for key in ('degenerate_faces','duplicate_faces'):
        if topology[key]: failures.append(key)
    if source_euler!=used_euler: failures.append('euler_mismatch')
    if orientation_conflicts: failures.append('inconsistent_face_orientation')
    if singular: failures.append('bilinear_near_singular_cells')
    warnings=[]
    if quality['nonpositive_quad_count']: warnings.append('nonpositive_projected_corner_quality')
    if uv and (uv.get('near_zero_uv_faces',0) or uv.get('orientation_flips',0)): warnings.append('intermediate_uv_defects')
    return dict(schema='final-output-review.v1',basic_output_checks_pass=not failures,
                hard_failure_reasons=failures,warnings=warnings,
                used_vertex_euler=used_euler,source_euler=source_euler,
                orientation_conflict_edges=orientation_conflicts,
                bilinear_minimum_normalized_area=min(minima) if minima else None,
                bilinear_near_singular_count=len(singular),bilinear_near_singular_faces=singular,
                old_nonpositive_projected_quad_count=quality['nonpositive_quad_count'],
                interpretation='Basic topology and local patch rank only. Projected negative corners remain quality warnings. No global self-intersection, outward orientation, Hausdorff or application suitability certificate.')
