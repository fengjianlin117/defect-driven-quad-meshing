from pathlib import Path
import statistics,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_evidence import OUT,read,save,sha
W=Path('research://windows-delivery/paper_stage_20260920')
D=OUT/'frontend_cost_report_v1'

def main():
    rows=read(OUT/'frontend_cost_v1/runs_v1_1/summary.json');assert len(rows)==36
    D.mkdir(exist_ok=False);models=list(dict.fromkeys(r['model'] for r in rows));stages=[k for k in rows[0]['timings'] if k!='full_output_evaluation_separate'];table=[]
    for model in models:
        rr=[r for r in rows if r['model']==model]
        table.append(dict(model=model,source_triangles=rr[0]['source_triangles'],baseline_quads=rr[0]['baseline_quads'],
            median_frontend_seconds=statistics.median(r['frontend_wall_seconds'] for r in rr),
            min_frontend_seconds=min(r['frontend_wall_seconds'] for r in rr),max_frontend_seconds=max(r['frontend_wall_seconds'] for r in rr),
            stages={k:statistics.median(r['timings'][k]['wall_seconds'] for r in rr) for k in stages},
            full_output_evaluation_separate_seconds=statistics.median(r['timings']['full_output_evaluation_separate']['wall_seconds'] for r in rr)))
    stats=dict(model_median_range=[min(r['median_frontend_seconds'] for r in table),max(r['median_frontend_seconds'] for r in table)],
        all_run_range=[min(r['frontend_wall_seconds'] for r in rows),max(r['frontend_wall_seconds'] for r in rows)],
        median_of_model_medians=statistics.median(r['median_frontend_seconds'] for r in table),
        stage_medians={k:statistics.median(r['stages'][k] for r in table) for k in stages},
        full_evaluation_median=statistics.median(r['full_output_evaluation_separate_seconds'] for r in table))
    save(D/'per_model.json',table);save(D/'summary.json',stats)
    names=['Input loading','Feature graph','Edge diagnosis','Compatibility setup','Constraint selection','Count and sizing']
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained');bottom=np.zeros(12)
    for k,name in zip(stages,names):
        values=np.array([r['stages'][k] for r in table]);ax.bar(models,values,bottom=bottom,label=name);bottom+=values
    ax.set_ylabel('Seconds (median of 3 fresh processes)');ax.set_title('Core front-end stages on the 12 seen CAD models\nExisting field and baseline supplied; no backend solve or field training included');ax.legend(ncol=3,fontsize=8);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(D/'frontend_cost.png',dpi=160);fig.savefig(D/'frontend_cost.svg');plt.close(fig)
    text=['# 特征图、诊断与提案的成本测量','',
        f"12模型，每模型3个全新Python进程；36次核心阶段合计范围{stats['all_run_range'][0]:.3f}–{stats['all_run_range'][1]:.3f}秒。每模型中位数的中位数为{stats['median_of_model_medians']:.3f}秒。",'',
        '计时包括输入读取、从源网格重新构图、重新测量基线特征缺陷、相容性初始化、简单无配额选组、整体数量和尺寸分配。没有复用这些阶段的计算结果；进程启动/导入耗时单独在process记录中，操作系统文件缓存状态未控制。',
        '', '范围严格限定为上述核心函数。现有协调驱动还包含候选记录、检查、额外上下文和后端调用；这里不是完整CLI端到端时间，也不包括NeurCross训练、生成基线或新增MIQ/QEx调用。',
        '', '|模型|源三角面|核心阶段中位数(s)|三次范围(s)|另测完整输出评价(s)|','|---|---:|---:|---:|---:|']
    for r in table:text.append(f"|{r['model']}|{r['source_triangles']}|{r['median_frontend_seconds']:.4f}|{r['min_frontend_seconds']:.4f}–{r['max_frontend_seconds']:.4f}|{r['full_output_evaluation_separate_seconds']:.4f}|")
    text+=['','完整输出评价独立计时，含缺陷测量，与核心阶段有重叠，不重复相加。三次计时只是同机器运行波动，不增加独立模型样本。','',
        '可支持的表述：在这些1024–4640面源网格上，核心特征处理与提案合计为亚秒到约1.4秒量级，作为已有场/基线之上的处理开销较小。不可据此宣称整个方法零开销、任意规模都快，或反馈12次调用与一次提案成本相同。','',
        '计时输出已核验重建参考边、重新诊断指标和约束集合与原记录一致。初次子进程因缺少相邻辅助模块而在计时前退出；原失败记录保留，补齐同一模块后在runs_v1_1运行，未改变方法或协议。','']
    (D/'COST_REPORT.md').write_text('\n'.join(text),encoding='utf-8');shutil.copytree(D,W/'frontend_cost_report_v1')
    manuscript=(W/'manuscript/论文初稿_v0.2.md').read_text()
    manuscript=manuscript.replace('内部研究初稿v0.2，2026-09-20；已纳入完整强静态对照','内部研究初稿v0.3，2026-09-20；主比较为同一方向场＋后端的无约束流水线')
    manuscript=manuscript.replace('本文研究的问题是：在复用固定方向场、限制后端求解次数的条件下，实测输出能否帮助协调硬约束和尺寸需求，并获得优于简单静态选择的质量—面数取舍？',
        '本文研究的首要问题是：在复用固定方向场和后端的条件下，从源几何构图、诊断输出缺陷并生成约束与尺寸需求，能否以较小的前端开销改善原无约束流水线的质量—面数取舍？反馈是进一步检验增量收益的模块，其优于一次性提案不是整个方法主体成立的前提。')
    manuscript=manuscript.replace('复杂协调是否值得保留为最终方法，由本稿新增的强静态对照及后续独立验证决定。','约束与尺寸主体的效果由无约束主比较检验；反馈和排序等模块是否值得保留，由消融及成本比较决定。')
    block=f'''### 5.2 相对无约束流水线的主比较

使用同一固定方向场和MIQ/QEx后端的均匀尺寸、无硬特征输出作为主要基线。联合一次性候选在6个模型上同时改善全参考缺陷和表面RMS，另有1个模型表现为两指标取舍、3个模型没有严格有效候选、2个模型实际数量未匹配。全部12个已见模型保留在分母中。比较直接要求双方实际面数相差不超过5%，没有用“双方分别接近同一目标”代替直接数量匹配。

该结果支持主体方案在开发数据上的效果贡献，不依赖反馈必须优于一次性提案。仅尺寸模块在2例两指标改善、5例由无约束均匀基线两指标占优，说明单独分配尺寸也不是普遍改进。仅缺陷约束候选有3例两指标改善，并伴随无效和数量未匹配条件；联合处理的实际意义需要结合完整逐模型表判断。

这些一次性候选来自此前冻结的共同数量目标矩阵，检验的是给定数量下的模块价值，不证明自动数量估计独立获得最佳预算。候选通过统一几何检查，但可能不满足原合计保护；候选质量、逐边退步和默认推荐接受率分别报告。反馈默认输出的8例两指标改善中包含1例原基线回退，不能全部归因于反馈。所有逐行记录见primary_pipeline_comparison_v1。

### 5.3 核心前端处理开销

在全部12个源模型上，每模型运行3个全新Python进程，重新构建特征图、测量基线边缺陷、初始化相容性检查、选择约束并估计数量和尺寸。核心阶段合计为{stats['all_run_range'][0]:.3f}–{stats['all_run_range'][1]:.3f}秒，每模型中位数的中位数为{stats['median_of_model_medians']:.3f}秒。输入读取计入，导入及进程启动单独保存。图和诊断没有使用结果缓存，输出与冻结参考逐项核对。

该测量支持适用规模下核心前端开销较小的表述，条件是方向场和基线输出已提供。它不包含场训练、额外MIQ/QEx求解或完整协调驱动的所有记录与检查工作。另测完整输出评价的跨模型中位数为{stats['full_evaluation_median']:.3f}秒，其中包含与边诊断重叠的工作，不与前述合计重复相加。操作系统文件缓存未控制，三次重复不视为独立几何样本。

'''
    # Keep original subsection identifiers for historical traceability, then renumber.
    manuscript=manuscript.replace('### 5.2 已完成的协调与消融',block+'### 5.4 已完成的协调与消融')
    for before,after in [('### 5.3 同面数','### 5.5 同面数'),('### 5.4 几何','### 5.6 几何'),('### 5.5 外部','### 5.7 外部'),('### 5.6 强静态','### 5.8 强静态')]:manuscript=manuscript.replace(before,after)
    manuscript=manuscript.replace('结果见第5.6节及独立结果附页','结果见第5.8节及独立结果附页')
    manuscript=manuscript.replace('## 8 结论\n\n','## 8 结论\n\n本文主体的主要效果证据是相对同一无约束场＋后端流水线的改善，而非反馈必须击败一次性方案。开发数据已给出联合提案的质量收益及约一秒量级核心前端开销的证据；反馈作为可选增量模块，同时报告收益、失败和额外调用成本。\n\n')
    (W/'manuscript/论文初稿_v0.3.md').write_text(manuscript,encoding='utf-8')
    snapshot=OUT/'manuscript_v0_3';snapshot.mkdir(exist_ok=False)
    for f in ['论文初稿_v0.3.md','定量主张证据映射.json']:shutil.copy2(W/'manuscript'/f,snapshot/f)
    shutil.copy2(OUT/'primary_pipeline_comparison_v1/comparisons.json',snapshot/'primary_comparisons.json')
    shutil.copy2(D/'per_model.json',snapshot/'frontend_timings.json');save(snapshot/'hashes.json',{str(f):sha(f) for f in snapshot.iterdir() if f.is_file()})
    print('FRONTEND_COST',stats,flush=True)

if __name__=='__main__':main()
