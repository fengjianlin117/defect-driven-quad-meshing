# 实验结果索引

|目录|内容|解释边界|
|---|---|---|
|primary_comparison|旧72格共同数量矩阵重组|一次性主体6改善/1取舍/3不可用/2未匹配；不是新v5矩阵|
|strong_static|无约束、全特征和缺陷选择等强静态比较|保留全部开发模型与数量匹配规则|
|coordinator_v5|12×6配置、推荐和消融|10非基线推荐、2回退，不等于10同Q胜出|
|conditional_policy|无需干预、初始接受、反馈的条件控制|缓存执行；早停有质量/数量取舍|
|processing_cost|核心特征处理与提案计时|不包含NeurCross、基线及额外后端|
|records|各阶段原始JSON汇总、候选、decision等|保留数值与历史身份；路径为公开来源标识|

最新数量对照在 records/current_count_controls_v1/runs/summary.json 和 external_runs/summary.json。全部计算已结束，综合报告尚未生成。其协议见 experiments/protocols/current_count_controls_v1。

records 中的原始 result/summary 可能重复引用同一次后端输出；复用不增加独立实验量。全体模型属于开发集。完整原生中间文件与所有候选OBJ未全部导入本仓库，人工网格在 examples 和 viewer。
