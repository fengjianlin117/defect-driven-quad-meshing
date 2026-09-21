"""Independent CGAL audit of both triangulations of every registered quad output."""
from pathlib import Path
import argparse,subprocess,shutil,json,time
from audit_evidence import OUT,OLD,read,save,sha
BIN=OUT/'external/cgal_local/check_self_intersections'
D=OUT/'intersection_audit_v1'

def check(p,diagonal):
    start=time.perf_counter()
    try:
        result=subprocess.run([str(BIN),str(p),str(diagonal)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
        row=json.loads(result.stdout) if result.returncode==0 else dict(completed=False)
        return dict(**row,returncode=result.returncode,stderr=result.stderr,elapsed_seconds=time.perf_counter()-start)
    except Exception as exc:return dict(completed=False,error=repr(exc),elapsed_seconds=time.perf_counter()-start)

def fixtures():
    dest=OUT/'intersection_checker_tests';dest.mkdir(exist_ok=False);rows=[]
    cases={
        'separated':('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 0 1 1\nf 1 2 3\nf 4 5 6\n',0),
        'crossing':('v -2 -2 0\nv 2 -2 0\nv 0 2 0\nv 0 -1 -1\nv 0 -1 1\nv 0 1 0\nf 1 2 3\nf 4 5 6\n',1),
        'shared_vertex_only':('v 0 0 0\nv 1 0 0\nv 0 1 0\nv -1 0 0\nv 0 -1 0\nf 1 2 3\nf 1 4 5\n',0),
        'adjacent_edge':('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 -1 0\nf 1 2 3\nf 2 1 4\n',0),
        'coplanar_overlap':('v 0 0 0\nv 2 0 0\nv 0 2 0\nv .2 .2 0\nv 2.2 .2 0\nv .2 2.2 0\nf 1 2 3\nf 4 5 6\n',1),
        'quad':('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n',0)}
    for name,(text,expected) in cases.items():
        p=dest/(name+'.obj');p.write_text(text)
        for diagonal in [0,1]:
            r=check(p,diagonal);r.update(name=name,expected_pairs=expected)
            r['test_pass']=r.get('completed') is True and r['intersection_pair_count']==expected
            rows.append(r)
    save(dest/'results.json',rows)
    assert all(r['test_pass'] for r in rows),rows
    print('INTERSECTION_FIXTURES',len(rows),'passed',flush=True)

def freeze():
    assert (OUT/'coordinator_v4_development/runs/integrity.json').exists()
    D.mkdir(exist_ok=False);cases=[]
    for p in read(OUT/'audit/source_plans.json'):
        cases.append(dict(model=p['model'],role='source',path=p['source']))
        cases.append(dict(model=p['model'],role='baseline',path=str(Path(p['source']).parent/'baseline/quad.obj')))
    for stage in ['coordinator_v4_development','quadriflow_development_v1','feedback_pilot_v1','feedback_matched_controls_v1','constraint_feedback_pilot_v1']:
        for p in (OUT/stage/'runs').rglob('result.json'):
            r=read(p)
            if not r.get('quad_path') or not Path(r['quad_path']).is_file():continue
            cases.append(dict(model=r.get('model') or p.relative_to(OUT/stage/'runs').parts[0],role=stage,record=str(p),path=r['quad_path'],basic_output_pass=r.get('basic_output_pass')))
    save(D/'cases.json',cases)
    save(D/'protocol.json',dict(scope='CGAL 5.6 exact-predicate/construction intersection audit of a piecewise triangular embedding of parsed double coordinates.',
        triangulations='Each quad is split by both alternative diagonals in separate runs; retain both results. This is not a certificate for curved bilinear patches.',
        invalid_topology='Report inability to insert faces as incomplete, never as no intersection.',
        degenerate='Report exact degeneracy separately; retain all detected face pairs and original polygon indices.',
        source='https://doc.cgal.org/5.6/Polygon_mesh_processing/group__PMP__intersection__grp.html',
        expected='Run on all completed registered candidate files, including structurally invalid candidates and every source/baseline, not only recommended results.',
        dedup='Same SHA256 geometry checked once per diagonal; all referring records preserved.',timeout_per_check=30))
    shutil.copy2(Path(__file__),D/'audit_intersections.py')
    hashes={str(p):sha(p) for p in D.iterdir() if p.is_file()};hashes[str(BIN)]=sha(BIN)
    for c in cases:hashes[c['path']]=sha(c['path'])
    save(D/'frozen_hashes.json',hashes)

def run():
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items());out=D/'results';out.mkdir(exist_ok=False);cache={};rows=[]
    for c in read(D/'cases.json'):
        h=sha(c['path'])
        if h not in cache:
            checks=[check(c['path'],i) for i in [0,1]];cache[h]=checks;save(out/(h+'.json'),checks)
            print('INTERSECTION',c['model'],c['role'],[(r.get('completed'),r.get('intersection_pair_count')) for r in checks],flush=True)
        checks=cache[h];rows.append(dict(**c,sha256=h,result=str(out/(h+'.json')),
            both_completed=all(r.get('completed') is True for r in checks),
            clear_both_triangulations=all(r.get('completed') is True and r.get('intersection_pair_count')==0 and r.get('exact_degenerate_triangles')==0 for r in checks)))
        save(D/'summary.json',rows)
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    save(D/'integrity.json',dict(case_references=len(rows),unique_geometry=len(cache),checks=2*len(cache),frozen_unchanged=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fixtures','freeze','run']);a=p.parse_args();globals()[a.stage]()
