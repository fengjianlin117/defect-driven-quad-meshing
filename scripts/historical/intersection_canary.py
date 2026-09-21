from pathlib import Path
from audit_evidence import OUT,read,save,sha
from audit_intersections import check,BIN
D=OUT/'intersection_canary_v1';D.mkdir(exist_ok=False)
plans={p['model']:p for p in read(OUT/'audit/source_plans.json')};cases=[]
for r in read(OUT/'coordinator_v4_development/runs/summary.json'):
    if r['variant']!='full':continue
    p=plans[r['model']];base=str(Path(p['source']).parent/'baseline/quad.obj')
    cases.extend([dict(model=r['model'],role='source',path=p['source']),dict(model=r['model'],role='baseline',path=base)])
    case=OUT/'coordinator_v4_development/runs'/r['model']/'full'/r['recommended']/'result.json'
    cases.append(dict(model=r['model'],role='full_recommendation',path=base if r['recommended']=='baseline' else read(case)['quad_path']))
save(D/'cases_before_check.json',cases);save(D/'input_hashes.json',{c['path']:sha(c['path']) for c in cases})
rows=[]
for c in cases:
    row=dict(**c,checks=[check(c['path'],i) for i in [0,1]]);rows.append(row)
    print(c['model'],c['role'],[(r.get('completed'),r.get('intersection_pair_count')) for r in row['checks']],flush=True)
save(D/'results.json',rows)
save(D/'integrity.json',dict(binary_sha256=sha(BIN),case_count=len(cases),scope='Canary on every model completed at registration time; final all-case audit still required.'))
