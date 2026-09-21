"""Read-only audit of frozen evidence; write only to the new paper stage."""
from pathlib import Path
import json, hashlib, subprocess, sys, datetime

ROOT = Path('research://source-repository')
OLD = ROOT/'research_runs/2026-09-20_cad_decision_v2'
OUT = ROOT/'research_runs/2026-09-20_paper_stage_v1'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, data): p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))

def main():
    OUT.mkdir(exist_ok=False)
    (OUT/'audit').mkdir()
    status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=ROOT,text=True)
    (OUT/'audit/starting_git_status.txt').write_text(status)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    save(OUT/'audit/starting_tracked_hashes.json',{p:sha(ROOT/p) for p in tracked if p and (ROOT/p).is_file()})
    expected=read(OLD/'final_verified_hashes.json')
    print('HASH_SCHEMA',type(expected).__name__,len(expected),list(expected)[:3],flush=True)
    changed=[];missing=[]
    for p,h in expected.items():
        if not Path(p).is_file(): missing.append(p)
        elif sha(p)!=h: changed.append(p)
    save(OUT/'audit/integrity.json',dict(expected_files=len(expected),changed=changed,missing=missing,
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_passed=not(changed or missing)))
    assert not(changed or missing), (changed,missing)
    plans=read(OLD/'defect_driven_v3/proposals/index.json')
    print('PLAN_EXAMPLE',json.dumps(plans[0],ensure_ascii=False),flush=True)
    save(OUT/'audit/source_plans.json',plans)
    for name in ['B49','B53','B65','B31','B32','B43']:
        plan=next(x for x in plans if x['model']==name)
        selection=read(plan['methods']['defect_driven']['selection'])
        base=read(plan['baseline_record'])
        summary=read(OLD/f'defect_driven_v3/development/{name}/result.json')
        print('MODEL_SCHEMA',name,'base',list(base),'case',summary['cases'][0],flush=True)
        print('SELECTION',name,selection['selected_group_ids'],selection['rejected_hard_groups'],flush=True)
    print('AUDIT_PASSED',len(expected),str(OUT),flush=True)
if __name__=='__main__':main()
