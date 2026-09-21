from pathlib import Path
import subprocess,shutil
from audit_evidence import OUT,save,sha
D=OUT/'external/cgal_local'
for archive in D.glob('*.deb'):subprocess.run(['dpkg-deb','-x',str(archive),str(D/'prefix')],check=True)
shutil.copy2(Path(__file__).parent/'check_self_intersections.cpp',D/'check_self_intersections.cpp')
argv=['g++','-O2','-std=c++17','-I',str(D/'prefix/usr/include'),'-I',str(D/'prefix/usr/include/x86_64-linux-gnu'),str(D/'check_self_intersections.cpp'),'-L',str(D/'prefix/usr/lib/x86_64-linux-gnu'),'-lmpfr','-lgmp','-o',str(D/'check_self_intersections')]
with (D/'build_v2.log').open('w') as stream:r=subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT)
save(D/'build_record.json',dict(argv=argv,returncode=r.returncode,packages={p.name:sha(p) for p in D.glob('*.deb')}))
if r.returncode:print((D/'build_v2.log').read_text()[-5000:]);raise SystemExit(r.returncode)
print('CHECKER_BUILT',sha(D/'check_self_intersections'))
