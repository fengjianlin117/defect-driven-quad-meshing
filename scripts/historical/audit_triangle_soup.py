"""Second, topology-independent audit; preserve original Surface_mesh results."""
from pathlib import Path
import sys, subprocess, shutil
from audit_evidence import OUT,read,save,sha
D=OUT/'intersection_soup_audit_v2'
PRE=OUT/'external/cgal_local'

def build():
    D.mkdir(exist_ok=False)
    src=(PRE/'check_self_intersections.cpp').read_text()
    src=src.replace('#include <CGAL/Surface_mesh.h>','#include <array>')
    src=src.replace('using Mesh=CGAL::Surface_mesh<K::Point_3>;','')
    src=src.replace('Mesh mesh;std::vector<Mesh::Vertex_index> verts;std::vector<int> original;','std::vector<K::Point_3> verts;std::vector<std::array<size_t,3>> triangles;std::vector<int> original;')
    src=src.replace('verts.push_back(mesh.add_vertex(K::Point_3(x,y,z)));','verts.emplace_back(x,y,z);')
    start=src.index('          auto a=verts[face[0]]')
    end=src.index('        }++polygon;',start)
    src=src[:start]+'''          size_t a=face[0],b=face[j],c=face[j+1];
          if(CGAL::collinear(verts[a],verts[b],verts[c]))++degenerate;
          triangles.push_back({a,b,c});original.push_back(polygon);
'''+src[end:]
    src=src.replace('std::vector<std::pair<Mesh::Face_index,Mesh::Face_index>> pairs;','std::vector<std::pair<size_t,size_t>> pairs;')
    src=src.replace('CGAL::Polygon_mesh_processing::self_intersections<CGAL::Sequential_tag>(mesh,std::back_inserter(pairs));','CGAL::Polygon_mesh_processing::triangle_soup_self_intersections<CGAL::Sequential_tag>(verts,triangles,std::back_inserter(pairs));')
    src=src.replace('pairs[i].first.idx(),b=pairs[i].second.idx()','pairs[i].first,b=pairs[i].second')
    (D/'checker.cpp').write_text(src)
    argv=['g++','-O2','-std=c++17','-I',str(PRE/'prefix/usr/include'),'-I',str(PRE/'prefix/usr/include/x86_64-linux-gnu'),str(D/'checker.cpp'),'-L',str(PRE/'prefix/usr/lib/x86_64-linux-gnu'),'-lmpfr','-lgmp','-o',str(D/'checker')]
    with (D/'build.log').open('w') as stream:r=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
    save(D/'build.json',dict(argv=argv,returncode=r.returncode))
    if r.returncode:print((D/'build.log').read_text()[-5000:]);raise SystemExit(r.returncode)
    shutil.copy2(Path(__file__),D/'audit_triangle_soup.py')
    save(D/'protocol.json',dict(reason='Surface_mesh cannot represent every alternate-diagonal triangulation. Audit identical geometry without face insertion using the official CGAL triangle-soup API.',
        rule='Shared point IDs or shared edge IDs alone are not intersections. Coincident geometry with different IDs remains detectable. Degenerate triangles also appear as self-pairs.',
        comparison='All registered v1 geometry; require identical pair sets wherever Surface_mesh completed. All v1 results remain unchanged.',
        scope='Exact constructions on parsed-double triangular embeddings, not a curved bilinear patch certificate.',
        source='https://doc.cgal.org/5.6/Polygon_mesh_processing/group__PMP__intersection__grp.html'))
    save(D/'frozen_hashes.json',{str(p):sha(p) for p in D.iterdir() if p.is_file()})
    print('SOUP_BUILT',sha(D/'checker'),flush=True)

def run():
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    import audit_intersections as old
    old.BIN=D/'checker'
    tests=[]
    for r in read(OUT/'intersection_checker_tests/results.json'):
        q=old.check(OUT/'intersection_checker_tests'/(r['name']+'.obj'),r['diagonal'])
        assert q['completed'] and q['intersection_pair_count']==r['expected_pairs'],(r,q)
        tests.append(dict(name=r['name'],**q))
    save(D/'fixture_results.json',tests)
    out=D/'results';out.mkdir(exist_ok=False);cache={};rows=[];parity=[]
    for c in read(OUT/'intersection_audit_v1/summary.json'):
        h=sha(c['path']);assert h==c['sha256']
        if h not in cache:
            checks=[old.check(c['path'],i) for i in [0,1]];cache[h]=checks;save(out/(h+'.json'),checks)
            prior=read(c['result'])
            for i,(a,b) in enumerate(zip(prior,checks)):
                if a.get('completed'):
                    aa={tuple(sorted(p['triangles'])) for p in a['pairs']};bb={tuple(sorted(p['triangles'])) for p in b.get('pairs',[])}
                    parity.append(dict(sha256=h,diagonal=i,identical=b.get('completed') is True and aa==bb))
            print('SOUP',c['model'],[(q.get('completed'),q.get('intersection_pair_count')) for q in checks],flush=True)
        checks=cache[h]
        row=dict(c);row.update(v1_result=c['result'],v1_clear=c['clear_both_triangulations'],result=str(out/(h+'.json')),
            both_completed=all(q.get('completed') is True for q in checks),
            clear_both_triangulations=all(q.get('completed') is True and q['intersection_pair_count']==0 and q['exact_degenerate_triangles']==0 for q in checks))
        rows.append(row)
    save(D/'summary.json',rows);save(D/'parity.json',parity)
    assert all(r['identical'] for r in parity)
    assert all(sha(p)==h for p,h in read(D/'frozen_hashes.json').items())
    save(D/'integrity.json',dict(case_references=len(rows),unique_geometry=len(cache),checks=2*len(cache),parity_checks=len(parity),parity_all_equal=True,fixtures=len(tests),frozen_unchanged=True))

if __name__=='__main__':globals()[sys.argv[1]]()
