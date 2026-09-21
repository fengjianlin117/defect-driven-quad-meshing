"""Integrate completed strong controls into a new manuscript revision and checkpoint."""
from pathlib import Path
import json,shutil
from audit_evidence import ROOT,OLD,OUT,read,save,sha
W=Path('research://windows-delivery/paper_stage_20260920')

def main():
    report=read(OUT/'strong_static_report_v1/summary.json');integ=report['integrity']
    dest=OUT/'checkpoint_04';dest.mkdir(exist_ok=False)
    manuscript=(W/'manuscript/论文初稿_v0.1.md').read_text()
    manuscript=manuscript.replace('完整数量对照和随机稳定性仍未完成','外部同数量对照和随机稳定性仍未完成')
    manuscript=manuscript.replace('取决于正在进行的强静态对照','由本稿新增的强静态对照及后续独立验证决定')
    manuscript=manuscript.replace('强对照结果由正在执行的冻结矩阵自动汇总到独立结果附页，完成前不在本文预填胜率。','72格强对照已完成，结果见第5.6节及独立结果附页；全部中间尝试均保留。')
    manuscript=manuscript.replace('**内部研究初稿，2026-09-20。**','**内部研究初稿v0.2，2026-09-20；已纳入完整强静态对照。**')
    section='''### 5.6 强静态数量对照与贡献判断

静态对照共72格，实际产生{attempts}次逻辑尝试，其中{native}次新增后端调用、{reused}次精确复用。实际数量差不超过5%的有效输出按预注册规则比较特征缺陷和表面RMS；至少一项更好、另一项不更差记为两指标占优。该比较不包含求解成本优势，也不代表所有几何指标均更好。

|静态条件|协调两指标占优|静态两指标占优|取舍|等同|静态无有效输出|数量未匹配|协调无效|
|---|---:|---:|---:|---:|---:|---:|---:|
'''.format(attempts=integ['logical_attempts'],native=integ['native_calls'],reused=integ['reused_attempts'])
    names={'none_uniform':'无约束/均匀','none_allocated':'无约束/分配','all_uniform':'全约束/均匀','all_allocated':'全约束/分配','simple_defect_uniform':'简单缺陷/均匀','simple_defect_allocated':'简单缺陷/分配'}
    for key,s in report['conditions'].items():
        a=s['all_models'];section+='|'+names[key]+'|'+'|'.join(str(a.get(k,0)) for k in ['coordinator_dominates','static_dominates','tradeoff','equivalent','static_no_valid','count_unmatched','coordinator_invalid'])+'|\n'
    section+='''
每行分母为12个已见模型，六行不能合并为72个独立样本。B43、B34、B65的协调输出为基线回退，不能把其数值优势归因于反馈；B31协调输出无效，不能计为胜出。完整六条件同时保留，避免只选弱对手。

相对“简单缺陷/分配”条件，协调在B49、B53、B68、B61的缺陷和RMS两指标上占优；B33、B65则由静态方法占优；B70、B73相同；B34体现两指标取舍；B32、B43的静态条件无有效输出；B31的协调结果无效。B49和B53是精确同面数比较，B68和B61在5%数量窗口内。

B33是重要反例：全约束/分配和简单缺陷/分配均在883面取得缺陷2.6636、RMS/h为0.007159，优于协调918面的4.8835和0.007794。此前观察到的恢复不证明恢复策略优于在类似预算下运行静态方法。

B65的全约束/分配在478面取得缺陷1.4310、RMS/h为0.009551，显著优于协调回退的490面、缺陷77.5958、RMS/h为0.059513。该静态候选的固定未选集合缺陷近零，逐边正退步和新增诊断缺陷均为零，也满足原合计保护；因此无需为此例放宽接受阈值。当前流程没有生成这一候选，暴露的是反馈提案的缺口。

进一步比较初始选择，在全部12例中，简单长度加权平均缺陷排序与原初始策略选出的整组集合完全一致。本数据不支持复杂初始排序的独立贡献。本文应将贡献定位收敛到实测反馈与有限预算协调，而不能以更复杂的排序作为已证实优势。

这些结果支持“部分实例有同数量反馈增益”，同时反驳“当前协调流程普遍优于简单静态策略”。静态每格最多2次调用，协调最多12次，其额外成本仍需与收益权衡。全约束在部分相容模型上的优势必须保留，最终方案应由进一步的统一机制验证决定。

'''
    manuscript=manuscript.replace('## 6 讨论与失败分析',section+'## 6 讨论与失败分析')
    manuscript=manuscript.replace('只有强静态对照和数量曲线才能区分这些因素。','新增强静态对照已说明B33的恢复不优于静态数量校准；完整数量曲线仍需检验更广的取舍。')
    manuscript=manuscript.replace('强静态协议与结果见 `strong_static_controls_v1`。','强静态协议与结果见 `strong_static_controls_v1`，预注册比较见 `strong_static_report_v1/comparisons.json`。')
    (W/'manuscript/论文初稿_v0.2.md').write_text(manuscript,encoding='utf-8')
    shutil.copytree(W/'manuscript',dest/'manuscript')
    shutil.copytree(W/'method_v5',dest/'unvalidated_method_v5_draft')
    for name in ['strong_static_controls.py','report_strong_static.py','manuscript_evidence.py','checkpoint_04_paper.py']:
        (dest/'scripts').mkdir(exist_ok=True);shutil.copy2(W/'scripts'/name,dest/'scripts'/name)
    changed=[p for p,h in read(OLD/'final_verified_hashes.json').items() if not Path(p).exists() or sha(p)!=h]
    tracked=[p for p,h in read(OUT/'audit/starting_tracked_hashes.json').items() if sha(ROOT/p)!=h]
    assert not changed and not tracked
    assert all(sha(p)==h for p,h in read(OUT/'strong_static_controls_v1/frozen_hashes.json').items())
    save(dest/'integrity.json',dict(old_protected_files=4613,changed_old_files=changed,changed_starting_tracked_files=tracked,
        strong_static=integ,total_native_calls=119+integ['native_calls'],new_heldout_models=0,goal_status='active',manuscript='v0.2 internal complete-structure draft; not submission ready'))
    text=f'''# 检查点04：论文贡献检验和初稿（按用户最新反馈调整优先级）

Goal保持active。上一轮是实质进展；本轮用户明确指出工作过度集中于审计/有效性，询问论文是否已有支撑。已回答：有机制正例和可写材料，但核心优越性/新颖性/泛化未充分建立，不是仅剩最后完善。

## 本轮实际完成

- 完整强静态对照：12已见模型×无/全/简单缺陷约束×均匀/分配尺寸=72格；{integ['logical_attempts']}次逻辑尝试、{integ['native_calls']}次新增后端调用、{integ['reused_attempts']}次复用。72格不是72独立模型。
- 全部条件在执行前冻结；以v4推荐实际Q为目标，每格至多2次尝试。统一基本+双对角线相交检查；按数量最接近选静态输出，仅≤5%才比较；保留全部失效和未匹配格。
- 简单缺陷/分配对照：协调在B49/B53/B68/B61两指标占优；静态在B33/B65占优；B70/B73相同；B34取舍；B32/B43静态无有效结果；B31协调无效。
- B33静态883Q D2.6636 RMS.007159，优于协调918Q D4.8835 RMS.007794。恢复成功并不能证明复杂恢复优于加预算的简单策略。
- B65全约束分配478Q D1.4310 RMS.009551，固定旧未选缺陷≈0、正退步0、新缺陷0。它满足原保护，不需放松阈值；当前协调没有生成这种候选。
- 全12模型的简单平均缺陷初始排序与v4选择的组集合完全一样：复杂初始排序收益无证据。
- 写成完整结构的中文论文工作稿v0.1，随后v0.2纳入完整对照表、正反例、算法、相关工作、实验、讨论、结论和参考文献。12组定量主张映射已生成；强对照独立表每行有source_record。
- 旧4613保护文件和进入阶段时的原有跟踪文件未变。累计真实后端调用{119+integ['native_calls']}。新未见模型仍0。

## 首读交付

Windows paper_stage_20260920/manuscript/论文初稿_v0.2.md；strong_static_report_v1/RESULTS.md；manuscript/贡献定位与执行优先级.md；manuscript/B65反馈缺口研究记录.md。
图：strong_static_report_v1/static_quality_count.png/svg（已查看）；比较表JSON/CSV和所有格的原始尝试保留。
WSL根仍为 research://source-repository/research_runs/2026-09-20_paper_stage_v1。
新目录strong_static_controls_v1、strong_static_report_v1、manuscript_v0_1、checkpoint_04。

## 当前状态与接下来真正有价值的工作

所有本轮计算已完成；不要恢复旧session65325或重跑已有输出目录。

1. 先做B65固定父候选场/密度/gsize、仅改变保护退步触发约束的受控实验。研究假设：反馈只处理超过初始0.1h/15度阈值的边，但保护还要求连续缺陷不退步，触发与接受目标不一致。应研究由实测保护退步触发修复，不需要放宽接受规则，也不能无条件强加全约束。当前数量校准强静态对照改变gsize，尚不是纯单变量机制证明。
2. 根据结果简化初始排序，统一有限反馈和新几何条件，再冻结下一方法版本与必要消融/敏感性。不要继续扩展独立审计项目作为主产出。每项新试验要能改变论文贡献判断。
3. method_v5/coordinator_v5.py只有工作草案（strict gate+跳过无操作父候选），未测试/冻结/运行。本轮在用户反馈后没有把它当新方法继续大跑，也未混入当前对照。草案仍有继承v4的触发/保护不一致待处理，不能直接当完成品。
4. 外部QuadriFlow尚需与方法实际Q一致的质量曲线；完整步长/数量/保护敏感性、场/种子稳定性仍缺。最终方法冻结后再运行模型族隔离的新测试，12旧模型永久已见。
5. 论文已有结构完整草稿不等于投稿完成；近年文献查新、最终新颖性收敛、未见验证、可迁移复现包和投稿前逐条主张审查尚未完成。允许结果反驳初始主张，不能承诺普遍优越或录用。

用户未要求停止goal。期刊/期限/算力仍是可选待答信息，不重复追问，不据此暂停。没有对外投稿、发消息、commit或push。
'''
    (dest/'CONTINUE.md').write_text(text,encoding='utf-8')
    save(dest/'snapshot_hashes.json',{str(f):sha(f) for f in dest.rglob('*') if f.is_file()})
    shutil.copytree(dest,W/'checkpoint_04');shutil.copy2(dest/'CONTINUE.md',W/'ACTIVE_RESEARCH_STATE.md')
    print('PAPER_CHECKPOINT',read(dest/'integrity.json'),flush=True)

if __name__=='__main__':main()
