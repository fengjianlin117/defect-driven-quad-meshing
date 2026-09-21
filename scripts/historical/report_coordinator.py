"""Summarize frozen v4 ablations and independent audits without changing decisions."""
from pathlib import Path
import csv, shutil, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_evidence import ROOT, OLD, OUT, read, save, sha
W=Path('research://windows-delivery/paper_stage_20260920')
D=OUT/'checkpoint_03'

def main():
    matrix=OUT/'coordinator_v4_development'; mi=read(matrix/'runs/integrity.json')
    intersections=read(OUT/'intersection_audit_v1/summary.json')
    ii=read(OUT/'intersection_audit_v1/integrity.json')
    dims=read(OUT/'cad_dimension_proxies_v1/summary.json')
    di=read(OUT/'cad_dimension_proxies_v1/integrity.json')
    byhash={r['sha256']:r for r in intersections}
    plans={p['model']:p for p in read(matrix/'plans.json')}
    sys.path.insert(0,str(matrix/'method'))
    from coordination import common_metrics
    rows=[];attempts=[]
    for r in read(matrix/'runs/summary.json'):
        p=plans[r['model']]; folder=matrix/'runs'/r['model']/r['variant']
        base=read(p['baseline_record']); selected={tuple(e) for e in read(folder/'initial_proposal.json')['selection']['selected_edges']}
        chosen=r['recommended'];path=p['baseline_record'] if chosen=='baseline' else str(folder/chosen/'result.json')
        rec=read(path);rec.update(common_metrics(rec,base,selected))
        quad=rec.get('quad_path') or str(Path(p['source']).parent/'baseline/quad.obj')
        dm=next(d for d in dims if d['model']==r['model'] and d['condition']=='coordinator_'+r['variant'])
        loops=[x for x in dm['loops'] if x['available']];pairs=[x for x in dm['pairs'] if x['available']]
        rows.append(dict(**r,record=path,quad=quad,baseline_quads=base['quads'],baseline_deficit=base['all_main_deficit'],
            baseline_rms_h=base['symmetric_rms_h'],count_ratio=rec['quads']/base['quads'],deficit_ratio=rec['all_main_deficit']/max(base['all_main_deficit'],1e-30),
            **{k:rec.get(k) for k in ['quads','all_main_deficit','symmetric_rms_h','basic_output_pass','protected_deficit','common_positive_regression','newly_deficient_edges','newly_deficient_length_h']},
            triangulated_intersection_clear=byhash[sha(quad)]['clear_both_triangulations'],
            circular_loops_registered=len(dm['loops']),circular_loops_available=len(loops),coaxial_pairs_registered=len(dm['pairs']),coaxial_pairs_available=len(pairs),
            max_abs_diameter_error_over_D=max((abs(x['diameter_error_over_D']) for x in loops),default=None),
            min_loop_coverage=min((x['coverage_within_01h'] for x in loops),default=None),
            max_abs_spacing_error_over_D=max((abs(x['spacing_error_over_D']) for x in pairs),default=None),
            max_abs_bbox_span_error_over_D=max(map(abs,dm['bbox_span_error_over_D']))))
        for f in sorted(folder.glob('*/result.json')):
            a=read(f);attempts.append(dict(model=r['model'],variant=r['variant'],record=str(f),**{k:a.get(k) for k in ['id','process_success','basic_output_pass','quads','target_quads','all_main_deficit','symmetric_rms_h','reused','source_record','common_positive_regression','newly_deficient_edges','quad_path']}))
    D.mkdir(exist_ok=False)
    save(D/'recommendations.json',rows);save(D/'all_attempts.json',attempts)
    for name,data in [('recommendations',rows),('all_attempts',attempts)]:
        with (D/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    stats={}
    for v in ['full','no_size','no_constraint','no_recovery','initial_only']:
        rr=[r for r in rows if r['variant']==v];aa=[a for a in attempts if a['variant']==v]
        stats[v]=dict(models=len(rr),nonbaseline=sum(r['recommended']!='baseline' for r in rr),
            attempts=len(aa),native=sum(r['native_calls'] for r in rr),reused=sum(r['reused_attempts'] for r in rr),
            attempt_basic_failure=sum(a['basic_output_pass'] is not True for a in aa),
            recommended_with_new_deficiencies=sum((r['newly_deficient_edges'] or 0)>0 for r in rr),
            recommended_intersection_clear=sum(r['triangulated_intersection_clear'] for r in rr))
    save(D/'ablation_summary.json',stats)
    failed=[r for r in intersections if not r['clear_both_triangulations']]
    save(D/'intersection_nonclear_cases.json',failed)
    unchanged=[p for p,h in read(OLD/'final_verified_hashes.json').items() if not Path(p).is_file() or sha(p)!=h]
    tracked=[p for p,h in read(OUT/'audit/starting_tracked_hashes.json').items() if sha(ROOT/p)!=h]
    assert not unchanged and not tracked
    save(D/'integrity.json',dict(old_protected_files=4613,changed_old_files=unchanged,changed_starting_tracked_files=tracked,
        matrix=mi,intersection=ii,dimensions=di,total_backend_calls_to_here=73+mi['native_calls'],new_independent_heldout_models=0,goal_complete=False))
    full=[r for r in rows if r['variant']=='full'];x=np.arange(len(full));fig,axs=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    axs[0].bar(x,[r['deficit_ratio'] for r in full],color='#2563eb');axs[0].set_ylabel('Deficit / baseline');axs[0].axhline(1,color='black',lw=.8)
    axs[1].bar(x,[r['count_ratio'] for r in full],color='#b45309');axs[1].set_ylabel('Actual quads / baseline');axs[1].axhline(1,color='black',lw=.8)
    for ax in axs:ax.set_xticks(x,[r['model'] for r in full]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Frozen v4 recommendations on 12 seen development models\nBaseline fallback retained; quality gains must be read with actual count cost')
    fig.savefig(D/'v4_quality_count.png',dpi=160);fig.savefig(D/'v4_quality_count.svg');plt.close(fig)
    text=['# 检查点03：统一协调方法、消融与独立几何审计','',
        'Goal保持active；12个模型均为已见开发数据，新增未见测试数为0。v4是冻结的开发方法版本，不是已满足投稿验收的最终方法。','',
        '## 全方法推荐结果','',
        '|模型|推荐|基线Q→实际Q|全参考缺陷 基线→推荐|RMS/h|新增缺陷边|逐边正退步|两种三角剖分无相交且无退化|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in full:text.append(f"|{r['model']}|{r['recommended']}|{r['baseline_quads']}→{r['quads']}|{r['baseline_deficit']:.4f}→{r['all_main_deficit']:.4f}|{r['symmetric_rms_h']:.5f}|{r['newly_deficient_edges']}|{r['common_positive_regression']:.4f}|{r['triangulated_intersection_clear']}|")
    text+=['','推荐沿用原保护规则：结构、全参考缺陷、表面RMS与固定初始未选集合合计均不劣于基线。合计保护不保证逐边不退步；新增缺陷列与正退步列不能省略。新增相交审计只作独立审查，未事后改变冻结决策。','',
        '## 消融与成本','',
        '|版本|非基线推荐/12|逻辑尝试|新增后端调用|精确输入复用|逻辑尝试结构或过程失败|推荐出现新缺陷的模型数|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for v,s in stats.items():text.append(f"|{v}|{s['nonbaseline']}|{s['attempts']}|{s['native']}|{s['reused']}|{s['attempt_basic_failure']}|{s['recommended_with_new_deficiencies']}|")
    text+=['','no_size仅关闭有效输出之后的局部尺寸反馈；no_constraint仅关闭有效输出之后的增补/替换约束反馈。初始选择、初始尺寸与失败恢复仍按冻结规则执行。no_recovery关闭失败分支；initial_only仅运行两种初始尺寸与数量校正。','',
        '调用顺序先full后消融，因此新增调用数不能直接当各版本独立运行成本；复用项是相同后端输入的重复请求，不增加独立证据。逻辑失败统计也包括复用失败。所有中间输出保留。','',
        f"矩阵新增真实调用{mi['native_calls']}，逻辑尝试{mi['logical_attempts']}，复用{mi['reused_attempts']}。加检查点01/02累计真实调用{73+mi['native_calls']}。",'',
        '## 独立几何检查','',
        f"CGAL EPECK检查{ii['case_references']}个文件引用、{ii['unique_geometry']}份独特几何、{ii['checks']}次三角剖分；非clear引用{len(failed)}，详情包含全部相交面与无法完成的原因。检查包括源模型、基线、全部可读中间输出及外部方法，未仅选成功结果。",'',
        '每个四边形分别沿两种对角线剖分。结果仅说明解析后的三角片嵌入；不是曲面双线性四边形的全局单射保证。6个人工几何夹具×2剖分通过。','',
        '## CAD尺寸代理','',
        '从源几何预先登记近圆闭合主特征环与同轴环对。报告拟合直径、轴向环间距和包围盒跨度误差，并保留切向对应覆盖率。单位是输入原单位及D/h归一化，不能解释为毫米或工程合格公差。','',
        '|模型|可测/登记圆环|最大直径误差/D|最低环覆盖率|可测/登记环对|最大间距误差/D|',
        '|---|---:|---:|---:|---:|---:|']
    def fmt(x):return '不适用' if x is None else f'{x:.6g}'
    for r in full:text.append(f"|{r['model']}|{r['circular_loops_available']}/{r['circular_loops_registered']}|{fmt(r['max_abs_diameter_error_over_D'])}|{fmt(r['min_loop_coverage'])}|{r['coaxial_pairs_available']}/{r['coaxial_pairs_registered']}|{fmt(r['max_abs_spacing_error_over_D'])}|")
    text+=['','圆环一致性不证明真实解析CAD圆；同模型多个环对不是独立样本。B43大量环对不能增加模型数。对照原始B-Rep尺寸和真实工程公差仍未完成。','',
        '## 已知缺口和下一步','',
        '- B32/B33恢复依赖明显增加面数；需实际面数匹配的质量—数量曲线，不能仅报告缺陷降低。',
        '- B43仍失败；B65仍回退。最佳父候选无新操作时停止，可能漏掉其他未扩展父候选，需在新的开发版本中分析并完整重测，不能修改本版本冻结结果。',
        '- 原保护合计仍允许逐边退步。共同参考应用容差、步骤/数量敏感性、场和随机种子稳定性尚缺。',
        '- 全部冻结后才可登记并运行家族隔离的新测试；尚未启动。完整论文、可迁移环境、投稿前主张审查仍未完成。','',
        '旧4613个保护文件及进入本阶段时已有跟踪文件均再次验证无变更。执行器的汇总重复键错误单独记录在execution_amendment.json；已完成模型安全恢复，方法快照不变。','']
    (D/'CHECKPOINT_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    shutil.copytree(D,W/'checkpoint_03');print('CHECKPOINT_03',stats,flush=True)

if __name__=='__main__':main()
