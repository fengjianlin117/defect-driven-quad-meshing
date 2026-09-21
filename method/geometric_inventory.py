"""Axis-free reference candidates. Sharp-edge evidence is retained, not semantics.

Reuses source mesh topology and sharp-edge extraction. Tangent continuation and
planar patch boundaries provide alternative groupings; overlaps are explicit.
No model names, supplied axes, feature labels, or backend outputs are used.
"""
from collections import defaultdict
import math
import numpy as np
from weak_layout_pipeline.pipeline.mesh import edge_topology,triangle_geometry
from weak_layout_pipeline.structural_lines.baseline_guided import feature_graph,_chains
from feature_diagnostics import edge_set,curve_record

def continuation_chains(mesh,edges,max_turn_degrees=60.,ambiguity_margin_degrees=5.):
    """Pair only mutual, unambiguous least-turn continuations at a junction.

    Degree-two vertices retain the full curve, including corners. At branches,
    a near tie terminates the chain instead of inventing a semantic choice.
    """
    edges=set(edges);neighbors=defaultdict(list)
    for a,b in sorted(edges):neighbors[a].append(b);neighbors[b].append(a)
    pairs={};ambiguous=[]
    for v,ns in neighbors.items():
        if len(ns)==2:
            pairs[(ns[0],v)]=ns[1];pairs[(ns[1],v)]=ns[0]
            continue
        choices={}
        for a in ns:
            incoming=mesh.vertices[v]-mesh.vertices[a]
            incoming=incoming/np.linalg.norm(incoming)
            ranks=[]
            for b in ns:
                if a==b:continue
                outgoing=mesh.vertices[b]-mesh.vertices[v]
                outgoing=outgoing/np.linalg.norm(outgoing)
                angle=float(np.degrees(np.arccos(np.clip(incoming@outgoing,-1,1))))
                ranks.append((angle,b))
            ranks.sort()
            if not ranks or ranks[0][0]>max_turn_degrees:continue
            if len(ranks)>1 and ranks[1][0]-ranks[0][0]<ambiguity_margin_degrees:
                ambiguous.append(dict(vertex=int(v),incoming_vertex=int(a),best_turn=ranks[0][0],second_turn=ranks[1][0]))
                continue
            choices[a]=ranks[0][1]
        for a,b in choices.items():
            if choices.get(b)==a:pairs[(a,v)]=b
    remaining=edges.copy();paths=[]
    def walk(a,b):
        ids=[a,b];remaining.remove(tuple(sorted((a,b))))
        while (a,b) in pairs:
            c=pairs[(a,b)];edge=tuple(sorted((b,c)))
            if edge not in remaining:break
            remaining.remove(edge);ids.append(c);a,b=b,c
            if b==ids[0]:break
        return ids
    starts=sorted((a,b) for a,b in list(edges)+[(b,a) for a,b in edges] if (b,a) not in pairs)
    for a,b in starts:
        if tuple(sorted((a,b))) in remaining:paths.append(walk(a,b))
    while remaining:paths.append(walk(*min(remaining)))
    return paths,ambiguous

def planar_patch_boundaries(mesh,reference,angle_degrees=1.,distance_relative=1e-5):
    """Connected nearly coplanar source faces, tested against a fixed seed plane.

    Only complete boundary loops already supported by raw feature edges are
    promoted to candidate groups. Single-triangle patches are not promoted.
    """
    normals,areas,centers=triangle_geometry(mesh)
    incidents,adjacency=edge_topology(mesh)
    adjacent=defaultdict(list)
    for a,b,e in adjacency:adjacent[a].append(b);adjacent[b].append(a)
    available=set(range(mesh.face_count));patches=[];loops=[]
    tolerance=distance_relative*np.linalg.norm(np.ptp(mesh.vertices,axis=0))
    cosine=np.cos(np.radians(angle_degrees))
    while available:
        first=min(available);available.remove(first);stack=[first];faces=[first]
        normal=normals[first];origin=centers[first]
        while stack:
            f=stack.pop()
            for g in sorted(adjacent[f]):
                if g not in available:continue
                if normals[g]@normal<cosine:continue
                if np.max(abs((mesh.vertices[mesh.faces[g]]-origin)@normal))>tolerance:continue
                available.remove(g);faces.append(g);stack.append(g)
        if len(faces)<2:continue
        group=set(faces)
        boundary={e for e,rows in incidents.items() if sum(f in group for f,_ in rows)==1}
        accepted=[]
        for ids in _chains(boundary):
            ce={tuple(sorted((a,b))) for a,b in zip(ids[:-1],ids[1:])}
            if ids[0]==ids[-1] and ce<=reference:
                loops.append((ids,len(patches)));accepted.append(len(loops)-1)
        patches.append(dict(id=len(patches),faces=sorted(faces),area=float(areas[faces].sum()),
            normal=normal.tolist(),boundary_edges=[list(e) for e in sorted(boundary)],
            fully_supported_closed_boundaries=accepted))
    return loops,patches

def build_geometric_inventory(mesh,sharp_degrees=30.,max_turn_degrees=60.,ambiguity_margin_degrees=5.):
    raw=feature_graph(mesh,sharp_degrees=sharp_degrees)
    reference=edge_set(raw['arcs'])
    paths,ambiguous=continuation_chains(mesh,reference,max_turn_degrees,ambiguity_margin_degrees)
    loops,patches=planar_patch_boundaries(mesh,reference)
    candidates={}
    def add(ids,method,patch=None):
        edges=frozenset(tuple(sorted((int(a),int(b)))) for a,b in zip(ids[:-1],ids[1:]))
        if edges not in candidates:
            c=curve_record(mesh,ids,'pending','axis_free_geometry')
            c['grouping_evidence']=[];c['planar_patch_ids']=[]
            candidates[edges]=c
        c=candidates[edges]
        if method not in c['grouping_evidence']:c['grouping_evidence'].append(method)
        if patch is not None:c['planar_patch_ids'].append(patch)
    # Closed tangent curves and patch loops first; remaining edges stay as chains.
    for ids in paths:
        if ids[0]==ids[-1]:add(ids,'tangent_continuation')
    for ids,patch in loops:add(ids,'planar_patch_boundary',patch)
    represented=set().union(*candidates.keys()) if candidates else set()
    residual,_=continuation_chains(mesh,reference-represented,max_turn_degrees,ambiguity_margin_degrees)
    for ids in residual:add(ids,'residual_continuation')
    normals=triangle_geometry(mesh)[0];incidents=edge_topology(mesh)[0]
    curves=[]
    for i,(ce,c) in enumerate(sorted(candidates.items(),key=lambda item:tuple(sorted(item[0])))):
        c['id']=f'geometry-{i:04d}'
        angles=[];lengths=[];boundary=0
        for a,b in sorted(ce):
            rows=incidents[(a,b)]
            lengths.append(float(np.linalg.norm(mesh.vertices[b]-mesh.vertices[a])))
            if len(rows)==1:angles.append(180.);boundary+=1
            else:angles.append(float(np.degrees(np.arccos(np.clip(normals[rows[0][0]]@normals[rows[1][0]],-1,1)))))
        c['dihedral_degrees_min']=min(angles)
        c['dihedral_degrees_length_mean']=float(np.average(angles,weights=lengths))
        c['source_boundary_edge_count']=boundary
        c['semantic_status']='geometric_candidate_not_certified_design_feature'
        curves.append(c)
    counts=defaultdict(int)
    for c in curves:
        for e in edge_set([c]):counts[e]+=1
    return dict(schema='axis-free-reference.v1',sharp_degrees=sharp_degrees,
        max_junction_turn_degrees=max_turn_degrees,ambiguity_margin_degrees=ambiguity_margin_degrees,
        raw_edge_count=len(reference),raw_chain_count=len(raw['arcs']),curves=curves,
        unrepresented_raw_edges=[list(e) for e in sorted(reference-edge_set(curves))],
        overlapping_reference_edges=[dict(edge=list(e),candidate_count=n) for e,n in sorted(counts.items()) if n>1],
        ambiguous_continuations=ambiguous,planar_patches=patches,
        limitations=['dihedral sensitivity remains','smooth ridges not extracted','patches require near planarity','overlapping curves are alternative/group evidence, not independent constraints','no semantic completeness guarantee'])
