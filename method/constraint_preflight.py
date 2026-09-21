"""Coordinate-equality preflight extracted without repository path side effects."""
import numpy as np

def precheck(mesh,uv,fuv,pd1,pd2,hard):
    hard={tuple(sorted(map(int,e))) for e in hard}
    parents=[np.arange(len(uv)),np.arange(len(uv))]
    def find(axis,i):
        p=parents[axis]
        while p[i]!=i:p[i]=p[p[i]];i=p[i]
        return i
    local=[];pins=[set(),set()];face_axes={}
    for fi,face in enumerate(mesh.faces):
        for j in range(3):
            a,b=int(face[j]),int(face[(j+1)%3])
            if tuple(sorted((a,b))) not in hard:continue
            t=mesh.vertices[b]-mesh.vertices[a];t/=np.linalg.norm(t)
            axis=int(abs(t@(pd1[fi]/np.linalg.norm(pd1[fi])))>abs(t@(pd2[fi]/np.linalg.norm(pd2[fi]))))
            x,y=int(fuv[fi,j]),int(fuv[fi,(j+1)%3])
            parents[axis][find(axis,x)]=find(axis,y)
            pins[axis].update((x,y));face_axes.setdefault(fi,[]).append(axis)
    classes=np.array([[find(axis,i) for i in range(len(uv))] for axis in (0,1)])
    collapsed=np.flatnonzero(np.any(np.all(classes[:,fuv]==classes[:,fuv[:,0]][:,:,None],axis=2),axis=0))
    direct=[fi for fi,axes in face_axes.items() if len(axes)!=len(set(axes))]
    forced_edges=set()
    for fi,face in enumerate(mesh.faces):
        for j in range(3):
            a,b=fuv[fi,j],fuv[fi,(j+1)%3]
            if np.array_equal(classes[:,a],classes[:,b]):forced_edges.add(tuple(sorted((int(face[j]),int(face[(j+1)%3])))))
    spans=[];minimum_sq=0.;rounding_sq=0.;count=0
    for axis in (0,1):
        groups={}
        for i in pins[axis]:groups.setdefault(int(classes[axis,i]),[]).append(i)
        for ids in groups.values():
            values=uv[ids,axis];mean=values.mean();spans.append(float(np.ptp(values)))
            minimum_sq+=float(np.sum((values-mean)**2));rounding_sq+=float(np.sum((values-np.round(mean))**2));count+=len(ids)
    return dict(forced_zero_triangle_count=len(collapsed),forced_zero_triangles=collapsed.tolist(),direct_local_count=len(direct),
                transitive_extra_triangles=sorted(map(int,set(collapsed)-set(direct))),forced_zero_edge_count=len(forced_edges),
                locked_class_span_p95=float(np.percentile(spans,95)) if spans else 0.,locked_class_span_max=max(spans,default=0.),
                minimum_alignment_displacement_rms=float(np.sqrt(minimum_sq/max(count,1))),
                independent_class_integer_displacement_rms=float(np.sqrt(rounding_sq/max(count,1))),
                assumptions='Baseline cut topology and combed axes retained. Only coordinate equality closure; seam translations and full global constraints omitted. Zero detections do not prove feasibility.')


