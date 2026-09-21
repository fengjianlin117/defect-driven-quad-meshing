# 历史与当前入口

本仓库从原研究项目的工作区独立导出，保留了未提交但已用于研究的源码状态；source_git_commit 和逐文件来源见 experiments/manifests/export_sources.json。不是原仓库完整Git历史迁移。

旧顶层 structural_graph/ 及其几何图/场补线流程不属于当前方法依赖，未搬入新仓库。当前 method 使用 frozen conditional_policy_development_v1/method 的依赖闭包和对应测试。

旧固定50%边长预算已经取消。旧 v3 为一次性提案；随后 v4 有界协调、v5 保护退步触发/几何检查、当前条件策略逐步形成。当前辅助文件仍可能以历史版本命名，但主入口始终由 README 和 scripts/run_pipeline.py 指明。

results/primary_comparison 是旧共同数量矩阵的重新组织，反馈行含历史v4身份。results/coordinator_v5 与 results/conditional_policy 单独保存新结果，不相互替换。docs/history 中更早验收清单只反映当时状态；以 docs/current_status.md 为当前摘要。
