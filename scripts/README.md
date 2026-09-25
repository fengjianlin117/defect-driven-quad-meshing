# 脚本入口

从仓库根目录运行，不需要手动设置method的PYTHONPATH：

- verify_snapshot.py：校验最终发布清单，重建12模型参考图并检查方向场索引。
- inspect_model.py：在新输出目录生成参考图和初始约束/尺寸提案，使用冻结基线诊断，不调用后端。
- run_pipeline.py：默认按源三角面数的一半（至少512且不低于旧基线）重新生成基线，校正实际面数，再运行条件流程或完整v5。支持 --target-quads、--max-quads；--resolution archived 显式恢复历史基线策略。
- prepare_resolution.py：上述入口使用的基线生成和校正模块；保存所有尝试，新基线的参数化和重新测量的缺陷一起传入方法。
- refresh_manifest.py：维护者在审阅发布材料变更后显式刷新清单；不要用来掩盖输入或结果变化。
- historical/：原实验编排、图表和报告源码。依赖历史工作区，research://路径只作来源标识；未承诺这些脚本可在独立仓库原样运行。

各可运行脚本支持 --help。新增方法或报告脚本请说明输入、输出、新求解预算和是否依赖历史缓存。
