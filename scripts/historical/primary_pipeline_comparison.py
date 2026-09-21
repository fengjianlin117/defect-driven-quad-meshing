"""Organize existing factorial evidence around the no-constraint pipeline baseline."""
from pathlib import Path
from collections import Counter
import shutil,csv
from audit_evidence import OUT,read,save,sha
D=OUT/'primary_pipeline_comparison_v1';W=Path('research://windows-delivery/paper_stage_20260920')

def main():
    D.mkdir(exist_ok=False)
    raw=read(OUT/'strong_static_controls_v1/runs/summary.json')
    refs={r['model']:r for r in read(OUT/'checkpoint_03/recommendations.json') if r['variant']=='full'}
    audits={r['sha256']:r for r in read(OUT/'intersection_soup_audit_v2/summary.json')}
    def pick(name,condition,size):
        cells=[r for r in raw if r['model']==name and r['condition']==condition and r['size']==size and r['strict_output_pass'] and r['quads']<=2048]
        return min(cells,key=lambda r:(abs(r['quads']/r['target_quads']-1),r['attempt'])) if cells else None
    rows=[]
    names={'size_only':'仅尺寸模块候选','constraints_only':'仅缺陷约束候选','joint_one_shot':'约束＋尺寸一次性候选','joint_feedback':'约束＋尺寸＋反馈默认输出'}
    for name,ref in refs.items():
        base=pick(name,'none','uniform')
        full=read(ref['record']);full['strict_output_pass']=full['basic_output_pass'] and audits[sha(ref['quad'])]['clear_both_triangulations']
        choices={'size_only':pick(name,'none','allocated'),'constraints_only':pick(name,'simple_defect','uniform'),
            'joint_one_shot':pick(name,'simple_defect','allocated'),'joint_feedback':full}
        for label,c in choices.items():
            valid=c is not None and c.get('strict_output_pass') is True and c['quads']<=2048
            match=bool(valid and base and abs(c['quads']/base['quads']-1)<=.05)
            status='module_invalid' if not valid else 'baseline_unavailable' if not base else 'count_unmatched' if not match else None
            if status is None:
                dd=c['all_main_deficit']-base['all_main_deficit'];dr=c['symmetric_rms_h']-base['symmetric_rms_h']
                ed=1e-9*max(1,c['all_main_deficit'],base['all_main_deficit']);er=1e-9
                status='equivalent' if abs(dd)<=ed and abs(dr)<=er else 'module_dominates' if dd<=ed and dr<=er else 'baseline_dominates' if dd>=-ed and dr>=-er else 'tradeoff'
            path=ref['record'] if label=='joint_feedback' else str(OUT/'strong_static_controls_v1/runs'/c['id']/'result.json') if c else None
            row=dict(model=name,module=label,status=status,valid=valid,within_5_percent=match,
                feedback_baseline_fallback=label=='joint_feedback' and ref['recommended']=='baseline',
                module_record=path,baseline_record=str(OUT/'strong_static_controls_v1/runs'/base['id']/'result.json') if base else None,
                module_quads=c['quads'] if c else None,baseline_quads=base['quads'] if base else None,
                module_deficit=c['all_main_deficit'] if c else None,baseline_deficit=base['all_main_deficit'] if base else None,
                module_rms=c['symmetric_rms_h'] if c else None,baseline_rms=base['symmetric_rms_h'] if base else None,
                common_positive_regression=c.get('common_positive_regression',0 if ref['recommended']=='baseline' else None) if c else None,
                newly_deficient_edges=c.get('newly_deficient_edges',0 if ref['recommended']=='baseline' else None) if c else None)
            rows.append(row)
    save(D/'comparisons.json',rows)
    with (D/'comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats={label:dict(Counter(r['status'] for r in rows if r['module']==label)) for label in names}
    save(D/'summary.json',stats)
    save(D/'interpretation.json',dict(primary='Comparison to the same indexed fixed field and MIQ/QEx backend without hard feature constraints or spatially varying density.',
        timing='Reanalysis after seeing development results, motivated by the clarified research question; not preregistered heldout inference.',
        selection='Unchanged selection of the closest-count valid result within each previously frozen cell. No choice based on feature error.',
        counts='Direct method-versus-baseline actual-count difference <=5%, not two separate comparisons to a shared target.',
        one_shot_scope='These cells use a shared count target derived previously from v4 output. They test module value conditional on that target, not the standalone accuracy of automatic count estimation.',
        recommendation_scope='The first three rows concern strict-geometrically-valid candidates, not candidates certified by the inherited aggregate protection gate. Per-edge regressions remain visible. Feedback row is the original frozen default; fallbacks marked.',
        main_method='One research pipeline with source feature graph, defect diagnosis, constraints and sizing; feedback is an optional incremental module, not a prerequisite for the pipeline contribution.',
        generalization='All 12 models seen; no statement about every mainstream method or all CAD families.'))
    labels={'module_dominates':'本文两指标改善','baseline_dominates':'基线两指标更好','tradeoff':'取舍','equivalent':'等同','module_invalid':'本文输出无效/无可用候选','baseline_unavailable':'基线不可用','count_unmatched':'实际数量未匹配'}
    text=['# 主比较：本文模块相对方向场＋后端的贡献','',
        '研究主线只有一条。主比较是本文方案与同一固定方向场＋MIQ/QEx后端的无硬特征、均匀尺寸输出；一次性提案与反馈之间的比较属于模块消融。',
        '', '以下为已见开发数据的重新组织，沿用先前冻结的72格试验，不伪称新的预注册验证。比较直接要求两方实际面数相差≤5%。缺陷与RMS至少一项改善、另一项不退步才称两指标改善。',
        '', '|本文方案|两指标改善|基线更好|取舍|等同|本文无效/不可用|数量未匹配|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,statsrow in stats.items():text.append('|'+names[name]+'|'+'|'.join(str(statsrow.get(k,0)) for k in ['module_dominates','baseline_dominates','tradeoff','equivalent','module_invalid','count_unmatched'])+'|')
    text+=['','每行均保留全部12模型；不能把48格当独立样本。一次性候选已通过统一几何检查，但可能不满足原未选集合合计保护。反馈行使用原默认推荐，其基线回退另行标记，不把回退的优势归因于新增模块。',
        '', '数量目标沿用此前v4实际输出建立的共同数量目标。因此前三行检验“在给定数量下的模块价值”，不证明一次性方案能独立估计出最佳数量。自动数量估计、原默认推荐与完整成本要另列。',
        '', '|模型|本文方案|本文Q|无约束Q|本文缺陷|无约束缺陷|本文RMS/h|无约束RMS/h|结果|',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    def fmt(x):return '—' if x is None else f'{x:.5g}'
    for r in rows:text.append(f"|{r['model']}{'*' if r['feedback_baseline_fallback'] else ''}|{names[r['module']]}|{fmt(r['module_quads'])}|{fmt(r['baseline_quads'])}|{fmt(r['module_deficit'])}|{fmt(r['baseline_deficit'])}|{fmt(r['module_rms'])}|{fmt(r['baseline_rms'])}|{labels[r['status']]}|")
    text+=['','*反馈行回退到原基线。原基线与新数量校准的无约束结果可不同；这种差异不能作为反馈收益。完整共同参考逐边退步见JSON。','',
        '合理主张：若约束/尺寸主体在公平对照下改善质量，即可支持主体的效果贡献；反馈只需作为增量收益和成本消融，不要求其普遍击败一次性提案才能使整项研究成立。方法新颖性、未见族验证和适用边界仍须独立论证。','']
    (D/'PRIMARY_RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
    shutil.copytree(D,W/'primary_pipeline_comparison_v1');print('PRIMARY_RESULTS',stats,flush=True)

if __name__=='__main__':main()
