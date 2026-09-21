"""Preserve continuation state, executable source snapshot, and complete integrity audit."""
from pathlib import Path
import shutil
from audit_evidence import ROOT,OLD,OUT,read,save,sha
W=Path('research://windows-delivery/paper_stage_20260920')

def main():
    dest=OUT/'checkpoint_03_final';dest.mkdir(exist_ok=False)
    replay=read(OUT/'coordinator_v4_cachefree_replay/integrity.json')
    matrix=read(OUT/'coordinator_v4_development/runs/integrity.json')
    intersection=read(OUT/'intersection_review_v2/summary.json')
    unchanged=[p for p,h in read(OLD/'final_verified_hashes.json').items() if not Path(p).is_file() or sha(p)!=h]
    tracked=[p for p,h in read(OUT/'audit/starting_tracked_hashes.json').items() if sha(ROOT/p)!=h]
    assert not unchanged and not tracked
    frozen_audits={}
    for folder,filename in [('coordinator_v4_development','frozen_hashes.json'),('intersection_audit_v1','frozen_hashes.json'),('intersection_soup_audit_v2','frozen_hashes.json'),('cad_dimension_proxies_v1','source_frozen_hashes.json'),('coordinator_v4_cachefree_replay','frozen_hashes.json')]:
        bad=[p for p,h in read(OUT/folder/filename).items() if not Path(p).is_file() or sha(p)!=h]
        assert not bad,(folder,bad);frozen_audits[folder]=dict(changed=bad,entries=len(read(OUT/folder/filename)))
    shutil.copytree(W/'scripts',dest/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(W/'method_v4',dest/'working_method_source',ignore=shutil.ignore_patterns('__pycache__'))
    save(dest/'integrity.json',dict(old_protected_files=4613,changed_old_files=unchanged,changed_starting_tracked_files=tracked,frozen_audits=frozen_audits,
        experimental_native_calls=71+matrix['native_calls'],replay_native_calls=2+replay['native_calls'],total_native_calls=73+matrix['native_calls']+replay['native_calls'],
        new_heldout_models=0,goal_status='active',replay=replay,intersection=intersection))
    text='''# 检查点03执行状态与继续入口

Goal仍active，未达到投稿前验收。完整范围继续以最初两份交接文件与GOAL_ACCEPTANCE.md为准，不因本检查点缩减。

## 已完成且可审查

- 冻结v4统一方法：两轮反馈、最多12次逻辑尝试、2048Q上限；12模型×5版本全部完成。284次逻辑尝试，38次新增后端调用，246次精确输入复用。消融标签只关闭特定反馈操作，详见METHOD_V4_SCOPE.md。
- full按旧合计保护产生9/12非基线推荐，initial_only为5/12。B32、B33恢复到1043Q、918Q，面数成本必须报告；B43和B65仍回退，B34本来误差很低。
- 两种四边形三角剖分的独立CGAL审计：353文件引用、149份独特几何、298次检查。第二版triangle-soup检查全部完成，与第一版已完成的288次相交对完全一致。18份独特几何有相交，其中7份旧基本检查已通过。
- B31推荐493Q在替代对角线下出现两对相交三角片（四边形348、492），不能视为合格结果。12基线均通过新审计。v4非基线推荐中仅8/12同时通过新相交审计；此为事后审查，不是改写冻结方法。
- 外部QuadriFlow终次同时基本通过且无相交：default12/12、sharp5/12、sharp_sat6/12；再要求数量误差≤5%为7/12、4/12、5/12。不得据此作不公平优越性结论。
- 源几何登记近圆闭环和同轴间距；77个候选引用、31份独特输出完成尺寸代理测量。报告覆盖率与单位限制，无工程公差保证、无精确B-Rep认证。
- B53完整v4流程关闭历史缓存重放8次调用，所有OBJ、指标与推荐逐项一致。6个协调单元测试、12个CGAL夹具检查和3个圆拟合测试通过；未声称旧软件包的全部测试通过。
- 累计真实后端调用119：开发实验109，精确重放10。新增未见模型0。旧4613文件与开始时已修改的跟踪文件再次核验无变更。

## 证据入口

- Windows checkpoint_03/CHECKPOINT_REPORT.md：矩阵、数量代价、尺寸表；recommendations.json/csv和all_attempts.json/csv可追溯每行。
- intersection_review_v2/REVIEW.md：必须与检查点03一起读，解决其不完整相交检查，并保存B31证据和图。
- WSL coordinator_v4_development：冻结方法、plans/protocol/frozen_hashes、全部60个模型版本的原始尝试。
- 初始执行器在B49结束后因重复summary键退出；runner_execution_v1_1.py仅修复汇总和完整结果恢复，execution_amendment.json留痕。方法快照没有改变。不要覆盖旧runner。
- WSL intersection_audit_v1与intersection_soup_audit_v2：第一版Surface_mesh和第二版三角片集合审计均保留。CGAL本地依赖在external/cgal_local，无系统安装。
- WSL cad_dimension_proxies_v1：source_registry、协议、全部测量与对应覆盖率。
- WSL coordinator_v4_cachefree_replay：8次关闭历史缓存重放及逐输出比较。
- WSL checkpoint_03_final：当前所有脚本副本、方法工作源副本、完整散列审计。

WSL证据根：research://source-repository/research_runs/2026-09-20_paper_stage_v1。
所有本检查点后台进程已完成；不要按旧session ID恢复。已有输出目录故意拒绝覆盖；新实验使用新目录。

## 紧接着应解决的研究缺口

1. 将两种三角剖分无相交/退化正式纳入下一版本统一有效性判定，所有来源一致执行。源/基线也要核验；无法检查不能当通过。保留旧版本结果作为反例。
2. 分析v4最佳父候选无新操作即停止的缺口，尤其B65；下一版本考虑有预算遍历未扩展父候选。不要按模型身份定制，不要事后放宽保护。变更后重新冻结完整开发矩阵。
3. 原合计保护仍允许6个full推荐出现新缺陷；需要共同参考的应用容差与逐边保护政策研究。不能为了提高推荐率选择阈值。CAD尺寸目前只是代理。
4. 实际面数匹配的无/全/简单约束、尺寸、数量控制与外部质量—面数曲线；步长、轮数、数量、诊断及保护敏感性。现有no_size/no_constraint不是从初始化起彻底移除模块的消融。
5. 场估计/随机种子稳定性与端到端成本；寻找具有B-Rep和几何家族信息的新数据。方法最终冻结后才登记并运行家族隔离未见测试，12个旧模型永久为已见。
6. 补完整一手文献核验、新颖性收敛、完整论文、图表、可迁移复现包和投稿前逐项主张审查。当前不具备投稿证据，不能标记goal完成。

目标期刊、截止时间、额外算力为可选待答信息；不因此暂停，不重复追问。未对外投稿、发消息、提交commit或push。
'''
    (dest/'CONTINUE.md').write_text(text,encoding='utf-8')
    save(dest/'snapshot_hashes.json',{str(f):sha(f) for f in dest.rglob('*') if f.is_file()})
    shutil.copytree(dest,W/'checkpoint_03_final')
    shutil.copy2(dest/'CONTINUE.md',W/'ACTIVE_RESEARCH_STATE.md')
    print('CHECKPOINT_03_FINAL',read(dest/'integrity.json'),flush=True)

if __name__=='__main__':main()
