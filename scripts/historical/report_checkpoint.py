"""Generate auditable stage report and figures from retained raw records."""
from pathlib import Path
import sys,json,csv,shutil,subprocess
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_evidence import ROOT,OLD,OUT,read,save,sha
WINDOWS=Path('research://windows-delivery/paper_stage_20260920')

def compact(row):return {k:v for k,v in row.items() if k not in ['edge_defects','quality','review','uv','topology']}
def main():
    d=OUT/'checkpoint_01';d.mkdir(exist_ok=False)
    plans={p['model']:p for p in read(OUT/'audit/source_plans.json')};rows=[]
    stages=['feedback_pilot_v1','feedback_matched_controls_v1','constraint_feedback_pilot_v1','quadriflow_development_v1']
    stages_count={s:read(OUT/s/'runs/integrity.json')['native_calls'] for s in stages}
    for stage in stages:
        for r in read(OUT/stage/'runs/summary.json'):
            row=dict(stage=stage,**r);row['source_record']=str(OUT/stage/'runs'/r['id']/'result.json');rows.append(row)
    save(d/'all_new_attempts.json',rows)
    keys=['stage','model','id','source_record','process_success','quads','basic_output_pass','all_main_deficit','symmetric_rms_h','protected_deficit','common_reference_positive_regression','quad_path']
    with (d/'all_new_attempts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    external={}
    for variant in ['default','sharp','sharp_sat']:
        groups=defaultdict(list)
        for r in rows:
            if r['stage']=='quadriflow_development_v1' and r['variant']==variant:groups[r['model']].append(r)
        terminal=[v[-1] for v in groups.values()]
        external[variant]=dict(registered_models=len(groups),attempts=sum(map(len,groups.values())),
            terminal_process_failure=sum(not r['process_success'] for r in terminal),
            terminal_basic_pass=sum(r.get('basic_output_pass') is True for r in terminal),
            any_basic_pass=sum(any(r.get('basic_output_pass') is True for r in v) for v in groups.values()),
            terminal_within_5_percent=sum(bool(r.get('quads')) and abs(r['quads']/r['target_quads']-1)<=.05 for r in terminal),
            terminal_basic_and_within_5_percent=sum(r.get('basic_output_pass') is True and abs(r['quads']/r['target_quads']-1)<=.05 for r in terminal))
    save(d/'external_summary.json',external)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,2,figsize=(12,10),layout='constrained');compare=[]
    colors=['#64748b','#b0bac8','#60a5fa','#2563eb','#f59e0b','#059669']
    for ri,name in enumerate(['B49','B53','B65']):
        p=plans[name];base=read(p['baseline_record']);selection=read(p['methods']['defect_driven']['selection']);selected={tuple(e) for e in selection['selected_edges']}
        protected={tuple(sorted(r['vertices'])) for r in base['edge_defects'] if r['layer']=='main'}-selected
        base['protected_deficit']=sum(r['length_h']*.5*(r['surface_loss']+r['unaligned_fraction']) for r in base['edge_defects'] if tuple(sorted(r['vertices'])) in protected and r['layer']=='main')
        candidates=[('B',base,str(p['baseline_record'])),('P',read(OLD/f'defect_driven_v3/development/{name}/defect_driven__allocated_0/result.json'),str(OLD/f'defect_driven_v3/development/{name}/defect_driven__allocated_0/result.json'))]
        for label,stage,case in [('U','feedback_matched_controls_v1',name+'__uniform_0'),('A','feedback_matched_controls_v1',name+'__allocated_0'),('S','feedback_pilot_v1',name+'__same_budget_1'),('C','constraint_feedback_pilot_v1',name+'__allocated')]:
            path=OUT/stage/'runs'/case/'result.json'
            if path.exists():candidates.append((label,read(path),str(path)))
        for label,r,path in candidates:compare.append(dict(model=name,condition=label,record=path,**{k:r.get(k) for k in ['quads','all_main_deficit','symmetric_rms_h','protected_deficit','basic_output_pass']}))
        for ci,key in enumerate(['all_main_deficit','symmetric_rms_h']):
            ax=axs[ri,ci];vals=[r[key] for _,r,_ in candidates]
            bars=ax.bar(np.arange(len(vals)),vals,color=colors[:len(vals)],width=.66)
            ax.set_xticks(np.arange(len(vals)),[f'{label}\n{r["quads"]}Q' for label,r,_ in candidates]);ax.set_title(name+(' | whole-reference deficit' if ci==0 else ' | symmetric surface RMS / h'))
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.set_ylim(0,max(vals)*1.22)
            for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)*.025,f'{v:.2f}' if ci==0 else f'{v:.4f}',ha='center',fontsize=8)
    fig.suptitle('Seen development only: measured feedback and count-matched controls\nB baseline; P v3 parent; U uniform control; A allocated control; S size feedback; C constraint feedback',fontsize=12)
    fig.savefig(d/'feedback_comparison.png',dpi=160);fig.savefig(d/'feedback_comparison.svg');plt.close(fig)
    save(d/'candidate_comparison.json',compare)
    unchanged=[p for p,h in read(OLD/'final_verified_hashes.json').items() if not Path(p).is_file() or sha(p)!=h]
    tracked=[p for p,h in read(OUT/'audit/starting_tracked_hashes.json').items() if sha(ROOT/p)!=h]
    assert not unchanged and not tracked
    save(d/'integrity.json',dict(old_protected_files=4613,changed_old_files=unchanged,changed_starting_tracked_files=tracked,
        new_native_calls=stages_count,total_new_native_calls=sum(stages_count.values()),new_independent_heldout_models=0,goal_complete=False))
    text=['# 论文阶段检查点01：已执行的证据核对与机制实验','',
        'Goal仍为active；论文方法、实验与复现验收尚未完成。所有结果均来自已见开发数据，新增未见模型数为0。','',
        '## 核验与研究发现','',
        '- 旧4613个保护文件再次核验无变更；本轮开始时的仓库跟踪文件亦无变更。',
        '- B49/B53/B31的4个暂缓组各自单独造成固定框架压扁，独立邻接路径证书与原并查集检查一致；不是4个组间冲突实例。',
        '- B31的旧未选合计改善掩盖逐边退步：新增共同参考逐边正退步指标，尚未据此调整接受阈值。',
        '- 尺寸反馈与约束反馈都有开发正例，也都有失败。不存在普遍修复或保证成功的结论。','',
        '## 候选与公平性检查','',
        '|模型|条件|实际Q|全参考缺陷|RMS/h|固定旧未选缺陷|基本检查|','|---|---|---:|---:|---:|---:|---|']
    for r in compare:text.append(f'|{r["model"]}|{r["condition"]}|{r["quads"]}|{r["all_main_deficit"]:.4f}|{r["symmetric_rms_h"]:.5f}|{r["protected_deficit"]:.4f}|{r["basic_output_pass"]}|')
    text+=['','B=基线；P=v3 allocated_0；U/A=均匀/原几何尺寸对照；S=尺寸反馈的最后一次数量校正；C=新增约束的allocated候选。图中展示这些固定条件，不代表完整失败率；全部尝试另存CSV/JSON。',
        '', 'B49三种对照均为528Q：S全缺陷11.1532，A为14.0606，U为15.2638。B53尺寸反馈512Q并未优于497Q原分配对照的全缺陷，保护仍失败；新增约束431Q候选通过原保护，均匀520Q候选结构失败。B65尺寸与约束反馈改善误差但均未通过原保护。','',
        '## 外部对照（预先冻结12模型、三种配置）','',
        '|QuadriFlow配置|调用|终次基本通过/12|任一中间基本通过/12|终次基本通过且数量误差≤5%/12|终次过程失败/12|','|---|---:|---:|---:|---:|---:|']
    for k,r in external.items():text.append(f'|{k}|{r["attempts"]}|{r["terminal_basic_pass"]}|{r["any_basic_pass"]}|{r["terminal_basic_and_within_5_percent"]}|{r["terminal_process_failure"]}|')
    text+=['','采用官方revision与本地SAT实现；默认/保尖锐/保尖锐+SAT都保留。B49 SAT失败为原实现断言；B32 sharp在516Q基本通过；不把外部方法的部分失败视为本方法优越证明。外部耗时包含自身场计算，当前MIQ耗时复用NeurCross场，两者不能作为端到端耗时直接比较。',
        '',f'本检查点新增后端调用共{sum(stages_count.values())}次：'+', '.join(f'{s}={n}' for s,n in stages_count.items())+'。两臂完全相同的尺寸计划已在输出前去重，不增加独立证据。',
        '', '## 下一阶段仍需解决','',
        '诊断阈值和步长敏感性；将尺寸/新增约束反馈组合为明确预算的统一方法；结构失败后的有预算回退；共同参考保护与几何公差；CAD关键尺寸；全局相交与稳定性；冻结后模型族隔离测试；更完整外部质量—面数曲线；完整论文、图表和环境重放。不得在这些工作完成前标记goal完成。',
        '', '## 证据入口','',
        '新WSL证据根：`research://source-repository/research_runs/2026-09-20_paper_stage_v1`。',
        '各试验目录有protocol/plan、frozen_hashes、result、backend_run、stdout/stderr与原始OBJ。`all_new_attempts.json`每行提供绝对source_record；`candidate_comparison.json`对应本表。',
        '原始文献及核验表在Windows本目录的literature与LITERATURE_AUDIT.md。']
    (d/'CHECKPOINT_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    (WINDOWS/'checkpoint_01').mkdir(exist_ok=False)
    for p in d.iterdir():shutil.copy2(p,WINDOWS/'checkpoint_01'/p.name)
    for name in ['CLAIM_EVIDENCE_MATRIX.md','GOAL_ACCEPTANCE.md','LITERATURE_AUDIT.md']:shutil.copy2(WINDOWS/name,OUT/name)
    shutil.copytree(WINDOWS/'scripts',OUT/'scripts',dirs_exist_ok=True)
    print('CHECKPOINT',sum(stages_count.values()),external,flush=True)
if __name__=='__main__':main()
