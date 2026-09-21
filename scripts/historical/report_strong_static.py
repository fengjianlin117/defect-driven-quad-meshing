"""Apply the pre-registered comparison rule to every static cell."""
from pathlib import Path
from collections import Counter
import shutil,csv,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_evidence import OUT,read,save,sha
D=OUT/'strong_static_controls_v1';W=Path('research://windows-delivery/paper_stage_20260920')

def main():
    integrity=read(D/'runs/integrity.json');assert integrity['complete']
    dest=OUT/'strong_static_report_v1';dest.mkdir(exist_ok=False)
    raw=read(D/'runs/summary.json');refs={r['model']:r for r in read(OUT/'checkpoint_03/recommendations.json') if r['variant']=='full'}
    audits={r['sha256']:r for r in read(OUT/'intersection_soup_audit_v2/summary.json')};rows=[]
    for p in read(D/'plans.json'):
        c=refs[p['model']];group=[r for r in raw if r['model']==p['model'] and r['condition']==p['condition'] and r['size']==p['size']]
        valid=[r for r in group if r['strict_output_pass'] and r['quads']<=2048]
        picked=min(valid,key=lambda r:(abs(r['quads']/p['target_quads']-1),r['attempt'])) if valid else None
        matched=picked is not None and abs(picked['quads']/p['target_quads']-1)<=.05
        coord_ok=c['basic_output_pass'] and audits[sha(c['quad'])]['clear_both_triangulations']
        status='coordinator_invalid' if not coord_ok else 'static_no_valid' if not picked else 'count_unmatched' if not matched else None
        if status is None:
            ds=c['all_main_deficit']-picked['all_main_deficit'];rs=c['symmetric_rms_h']-picked['symmetric_rms_h']
            epsd=1e-9*max(1,c['all_main_deficit'],picked['all_main_deficit']);epsr=1e-9*max(1,c['symmetric_rms_h'],picked['symmetric_rms_h'])
            status='equivalent' if abs(ds)<=epsd and abs(rs)<=epsr else 'coordinator_dominates' if ds<=epsd and rs<=epsr else 'static_dominates' if ds>=-epsd and rs>=-epsr else 'tradeoff'
        rows.append(dict(model=p['model'],condition=p['condition'],size=p['size'],status=status,coord_is_baseline=c['recommended']=='baseline',
            coordinator_record=c['record'],coordinator_valid=bool(coord_ok),coordinator_quads=c['quads'],coordinator_deficit=c['all_main_deficit'],coordinator_rms=c['symmetric_rms_h'],
            static_record=str(D/'runs'/picked['id']/'result.json') if picked else None,static_quads=picked['quads'] if picked else None,
            static_deficit=picked['all_main_deficit'] if picked else None,static_rms=picked['symmetric_rms_h'] if picked else None,
            within_5_percent=matched,logical_attempts=len(group),native_calls=sum(not r['reused'] for r in group)))
    save(dest/'comparisons.json',rows)
    with (dest/'comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    labels={'coordinator_invalid':'协调结果无效','static_no_valid':'静态无有效输出','count_unmatched':'未匹配数量','coordinator_dominates':'协调两指标占优','static_dominates':'静态两指标占优','tradeoff':'两指标取舍','equivalent':'等同'}
    summary={}
    for condition in ['none','all','simple_defect']:
        for size in ['uniform','allocated']:
            key=condition+'_'+size;rr=[r for r in rows if r['condition']==condition and r['size']==size]
            summary[key]=dict(all_models=dict(Counter(r['status'] for r in rr)),nonbaseline_models=dict(Counter(r['status'] for r in rr if not r['coord_is_baseline'])))
    save(dest/'summary.json',dict(integrity=integrity,conditions=summary,interpretation='Each condition retained; 72 cells are not 72 independent models. Dominance refers only to feature deficit and surface RMS within the registered 5% actual-count window; ignores other objectives and solve cost.'))
    text=['# 强静态对照：协调方法是否值得保留','',
        '这是对核心论文假设的开发实验，未修改v4方法、评价阈值或数据身份。全部12个模型、三种约束选择×两种尺寸，共72格；每格至多两次求解。','',
        '每格取严格有效输出中实际面数最接近冻结v4推荐数量者，平局取较早尝试；差异≤5%才比较。占优仅指全参考缺陷和表面RMS两指标至少一项更好、另一项不更差，不意味着CAD公差、计算成本或所有指标更好。原始失效与未匹配格均保留。','',
        '|静态条件|协调占优|静态占优|取舍|等同|静态无有效输出|数量未匹配|协调无效|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,value in summary.items():
        a=value['all_models'];text.append('|'+key+'|'+'|'.join(str(a.get(k,0)) for k in ['coordinator_dominates','static_dominates','tradeoff','equivalent','static_no_valid','count_unmatched','coordinator_invalid'])+'|')
    text+=['','表中包含B43/B34/B65的基线回退；其表现不能归因于反馈贡献。另存summary.json的nonbaseline_models分组。B31协调结果无效，不计为胜出。一个模型的多个静态格不是独立样本，也不合并成“72次胜率”。','',
        '|模型|静态约束|尺寸|协调Q|静态Q|协调缺陷|静态缺陷|协调RMS/h|静态RMS/h|比较|','|---|---|---|---:|---:|---:|---:|---:|---:|---|']
    def fmt(x):return '—' if x is None else f'{x:.5g}'
    for r in rows:text.append(f"|{r['model']}{'*' if r['coord_is_baseline'] else ''}|{r['condition']}|{r['size']}|{r['coordinator_quads']}|{fmt(r['static_quads'])}|{fmt(r['coordinator_deficit'])}|{fmt(r['static_deficit'])}|{fmt(r['coordinator_rms'])}|{fmt(r['static_rms'])}|{labels[r['status']]}|")
    text+=['','*表示协调方法回退基线。未匹配的数字仅作记录，不作同面数胜负判断。所有中间尝试见runs/summary.json。','',
        f"本实验新增后端调用{integrity['native_calls']}，复用{integrity['reused_attempts']}，逻辑尝试{integrity['logical_attempts']}。静态每格最多2次；协调最多12次。这是静态方法的真实成本优势，不能因协调输出较好而忽略。",'',
        '这些结果应决定后续是否保留复杂策略。只在少数正例成立的机制不能写成普遍优越性；全约束在相容模型上的强表现需要保留。外部同实际面数曲线、未见模型族测试、随机稳定性和工程尺寸验证仍未完成。','']
    (dest/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    colors={'none':'#64748b','all':'#d97706','simple_defect':'#2563eb'}
    fig,axes=plt.subplots(3,4,figsize=(15,10),layout='constrained')
    for ax,(name,c) in zip(axes.flat,refs.items()):
        for p in [r for r in raw if r['model']==name and r.get('quads') and r.get('all_main_deficit') is not None]:
            ax.scatter(p['quads'],p['all_main_deficit'],color=colors[p['condition']],marker=('o' if p['size']=='uniform' else '^') if p['strict_output_pass'] else 'x',s=27,alpha=.75)
        valid=audits[sha(c['quad'])]['clear_both_triangulations'];ax.scatter(c['quads'],c['all_main_deficit'],marker='*' if valid else 'X',color='#dc2626',s=130,zorder=5)
        ax.axvspan(.95*c['quads'],1.05*c['quads'],color='#ef4444',alpha=.06);ax.set_title(name+(' (baseline fallback)' if c['recommended']=='baseline' else ''));ax.set_xlabel('Actual quads');ax.set_ylabel('Feature deficit');ax.grid(alpha=.15)
    fig.suptitle('All 12 seen models: static controls at v4 output-count targets\nGray none; orange all; blue simple defect. Circle uniform; triangle allocated; x invalid. Red v4 (X invalid).',fontsize=12)
    fig.savefig(dest/'static_quality_count.png',dpi=150);fig.savefig(dest/'static_quality_count.svg');plt.close(fig)
    shutil.copytree(dest,W/'strong_static_report_v1');shutil.copy2(dest/'RESULTS.md',W/'manuscript/强静态对照结果附页.md')
    print('STRONG_STATIC_REPORT',summary,flush=True)

if __name__=='__main__':main()
