"""Publish completed development results without changing frozen experiment evidence."""
from pathlib import Path
import shutil
from audit_evidence import OUT, read, save, sha

W = Path('research://windows-delivery/paper_stage_20260920')
D = OUT / 'coordinator_v5_report_v1'
R = OUT / 'coordinator_v5_development/runs'

def main():
    integrity = read(R / 'integrity.json')
    assert integrity['complete'] and integrity['frozen_unchanged']
    complete = read(R / 'summary.json')
    assert len(complete) == 72
    D.mkdir(exist_ok=False)
    rows = []
    for cell in complete:
        folder = R / cell['model'] / cell['variant']
        decision = read(folder / 'decision.json')
        candidates = read(folder / 'summary.json')['candidates']
        candidate = next(c for c in candidates if c['id'] == decision['recommended_id'])
        assert candidate['strict_output_pass']
        rows.append(dict(model=cell['model'], variant=cell['variant'], **candidate,
                         attempts=cell['attempts'], native_calls=cell['native_calls'],
                         decision_path=str(folder / 'decision.json'),
                         decision_sha256=sha(folder / 'decision.json')))
    save(D / 'recommendations.json', rows)
    stats = {v:dict(cells=sum(r['variant']==v for r in rows),
                   nonbaseline=sum(r['variant']==v and r['id']!='baseline' for r in rows))
             for v in dict.fromkeys(r['variant'] for r in rows)}
    save(D / 'summary.json', dict(execution=integrity, variants=stats))
    lines = ['# 同一方法的反馈模块：完成开发矩阵', '',
             '主比较仍是本文约束与尺寸处理相对同一固定方向场＋无约束后端。反馈属于可选增强，反馈是否胜过一次性提案不是主体贡献成立的前提。', '',
             f"12已见模型×6配置完成；{integrity['logical_attempts']}次逻辑尝试，{integrity['native_calls']}次新增后端调用，{integrity['reused_attempts']}次精确输入复用。复用不增加独立证据；实验前冻结文件未变。", '',
             '保留原缺陷、RMS和固定未选集合合计保护，另要求两种三角剖分均通过相交检查；保护条件没有放宽。全模型保留，不因失败移除。', '',
             '|模型|推荐|实际Q|缺陷D|RMS/h|保护集合D|', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['variant']=='full':
            lines.append(f"|{r['model']}|{r['id']}|{r['quads']}|{r['all_main_deficit']:.6f}|{r['symmetric_rms_h']:.6f}|{r['protected_deficit']:.6f}|")
    lines += ['', '完整配置12个推荐均通过上述几何条件，其中10个为新增候选、2个回退到原基线。该数字是输出选择结果，不是10次公平同面数胜出。', '',
              'B65完整反馈得到454Q、D=1.348284、RMS/h=0.009835、保护集合D=0.297606，满足原基线保护上限0.492564。关闭新增保护退步触发时回退到原基线；保留触发但关闭尺寸反馈时得到500Q、D=1.542108。这支持触发修正的开发证据，不证明必须同时使用所有反馈模块。B65旧全约束分配对照仍有更低RMS，不能宣称当前反馈全面支配它。', '',
              'B31不再推荐旧有相交的493Q结果，新推荐535Q、D=36.916077、RMS/h=0.026951，通过两种三角剖分检查。此处数量已变化，仍需新数量下公平无约束及外部对照。B43、B34回退；B73的缺陷变化很小，不应渲染为重大收益。', '',
              '一次性联合提案相对无约束的6例两指标改善来自先前冻结的共同数量矩阵，与本次反馈矩阵分开报告。新反馈结果不能直接替换旧主比较而沿用旧面数匹配结论。', '',
              '全部12模型仍是已见开发数据；方法族外验证、完整算法敏感性、场稳定性、完整运行成本和新颖性论证尚待完成。', '']
    (D/'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    shutil.copytree(D, W/D.name)

    # Correct a prose range typo in a new report; keep original measurements/report intact.
    old_cost = OUT/'frontend_cost_report_v1'
    cost = OUT/'frontend_cost_report_v1_1'
    shutil.copytree(old_cost, cost)
    sizes = [r['source_triangles'] for r in read(cost/'per_model.json')]
    content = (cost/'COST_REPORT.md').read_text(encoding='utf-8')
    content = content.replace('1024–4640', f'{min(sizes)}–{max(sizes)}')
    content += '\n文字勘误：旧报告结尾源面数上限误写4640，逐模型表和原始计时正确；上限应为5248。计时与实验数据未改。\n'
    (cost/'COST_REPORT.md').write_text(content, encoding='utf-8')
    shutil.copytree(cost, W/cost.name)

    manuscript = (W/'manuscript/论文初稿_v0.3.md').read_text(encoding='utf-8')
    manuscript = manuscript.replace('内部研究初稿v0.3', '内部研究初稿v0.4')
    start = manuscript.index('## 摘要')
    end = manuscript.index('关键词：', start)
    abstract = '''## 摘要

在有限面数下，方向场与参数化后端生成的CAD四边形网格可能遗漏几何特征。本文研究一种复用已有方向场与后端的缺陷驱动处理方法：从源网格建立特征图，以基线输出诊断约束需求，并联合确定特征约束、整体数量与局部尺寸；还提供有限次数的实测反馈，以处理残余缺陷和失败。一次性提案与反馈属于同一方法的不同配置。

在12个已见开发模型的共同数量对照中，一次性联合提案有6例在双方实际面数相差不超过5%时同时改善全参考特征缺陷与表面RMS；其余为1例取舍、3例无有效候选和2例数量未匹配。每模型3次独立进程计时显示，构图、诊断、相容性初始化、约束选择和数量尺寸分配等核心阶段合计0.180–1.395秒。该时间不含方向场生成、基线求解和额外后端调用。新增反馈开发矩阵在保持原保护条件下解决了部分候选接受缺口，但其增量收益和求解成本需分别评价。这些结果为主体方案的效果与较低核心处理成本提供了开发证据；跨模型族泛化、最终方法新颖性及完整成本尚待验证。

'''
    manuscript = manuscript[:start]+abstract+manuscript[end:]
    manuscript = manuscript.replace('当前贡献定位是一个可复现的协调框架及其机制实验', '当前贡献定位是复用方向场与后端的缺陷驱动约束和尺寸处理，以及其质量、成本和适用边界实验')
    manuscript = manuscript.replace('### 5.4 已完成的协调与消融', '### 5.4 前一版协调与消融（冻结历史结果）')
    marker = '\n## 6 '
    pos = manuscript.find(marker)
    assert pos >= 0
    addition = '''
### 5.9 当前反馈实现的补充开发结果

后续冻结的12模型×6配置矩阵完成397次逻辑尝试，其中21次新增后端调用、376次精确输入复用。修正使原固定保护集合的实测退步也能触发修复，并在无可操作候选时检查其他父候选；原合计保护阈值未放宽。完整配置输出中10例为新增候选、2例回退，12例均通过基本检查与两种三角剖分相交检查。这是推荐统计，不能解释为10例同数量胜出。

B65得到454Q、D=1.348284、RMS/h=0.009835，满足原保护；关闭新增保护触发则仍回退。B31得到通过几何检查的535Q结果，替代旧有相交候选。新输出数量变化，因此不直接混入第5.2节历史数量对照。反馈的作用是补充主体提案的适用能力；它并非主体贡献成立的必要条件。完整逐配置推荐及原始记录映射见coordinator_v5_report_v1。

'''
    manuscript = manuscript[:pos]+addition+manuscript[pos:]
    (W/'manuscript/论文初稿_v0.4.md').write_text(manuscript, encoding='utf-8')
    snapshot = OUT/'manuscript_v0_4'
    snapshot.mkdir(exist_ok=False)
    shutil.copy2(W/'manuscript/论文初稿_v0.4.md', snapshot)
    shutil.copy2(D/'recommendations.json', snapshot)
    shutil.copy2(OUT/'primary_pipeline_comparison_v1/comparisons.json', snapshot/'primary_comparisons.json')
    shutil.copy2(cost/'per_model.json', snapshot/'frontend_timings.json')
    save(snapshot/'hashes.json', {str(f):sha(f) for f in snapshot.iterdir() if f.is_file()})
    print(dict(execution=integrity, variants=stats, source_triangle_range=[min(sizes),max(sizes)]))

if __name__=='__main__':
    main()
