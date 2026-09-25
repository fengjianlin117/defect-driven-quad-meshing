# 运行与复现说明

## 首版能复现到哪一层

本仓库提供当前 Python 方法源码、45份人工检查 OBJ、12模型的源网格/方向场/基线/参考图、中间参数化文件、实验方案、原始数值记录与论文材料。可在 CPU 上重建全部参考图和初始提案、运行方法测试，并在准备好原生后端后执行当前单模型流程。

不声称已完成从干净机器自动重训全部方向场、重跑全部历史矩阵的验收。历史完整工作目录、训练权重、第三方编译树以及所有候选网格不都在此仓库；results/records 保留的是逐次 JSON 数值和选择记录。原始冻结数据仍由维护者保存。

## CPU 环境

本次核验环境为 Python 3.12.3、NumPy 1.26.4、SciPy 1.11.4、pytest 9.1.1。用仓库 requirements-core.txt 创建独立环境。首次核对执行：

```bash
python scripts/verify_snapshot.py
python scripts/inspect_model.py --model B49 --out outputs/B49-inspection
python -m pytest
```

inspect_model 重建源特征参考、读入冻结的基线逐边诊断并计算初始选择和尺寸提案；它没有重新测量基线缺陷或运行后端。完整运行入口和输出解释见 README。

## 当前单模型输入契约

experiments/manifests/models.json 中的路径以仓库根目录为基准，scripts/project_paths.py 转成绝对路径供未改动的研究方法使用。

每个模型目录 experiments/inputs/<ID>/ 包含：

|文件|要求|
|---|---|
|source.obj|原三角网格，不改变顶点/面的编号顺序|
|PD1.txt、PD2.txt|与源三角面逐行对应的方向向量；当前保存格式含面索引|
|graph.json|同一源网格的完整特征参考|
|baseline_record.json|无硬特征基线的质量与逐边缺陷记录|
|baseline/quad.obj|该记录对应的基线网格|
|baseline/miq_uv.txt、miq_fuv.txt|基线参数化 UV 和三角形索引|
|baseline/miq_combed_PD1.txt、miq_combed_PD2.txt|组合检查需要的基线梳理方向场|
|uniform.rho、allocated.rho|冻结提案的尺寸记录；当前协调器会重新生成提案|

manifest 中 reference_h 是固定诊断参考尺度，gsize 是该基线的后端尺度。添加新模型必须为同一网格准备这些一致的材料，并在新 manifest/协议中注册；不能仅替换 source.obj。

```bash
python scripts/run_pipeline.py --model B49 \
  --miq /absolute/path/miq_adaptive \
  --qex /absolute/path/qex_adapter \
  --checker /absolute/path/check_self_intersections \
  --out outputs/B49-run-001
```

默认 `--resolution auto --strategy conditional`。自动分辨率起始目标为 `max(512, 历史基线面数, ceil(源三角面数/2))`，例如 B43 的 5248 个三角面对应 2624 个目标四边形，B73 的 4640 个三角面对应 2320 个。输入三角剖分密度会影响该估计；它是避免明显过度简化的工程默认值，不是保真误差界或最优面数。

可指定 `--target-quads 2500 --max-quads 8192`。目标指起始基线预算，后续几何分配和失败恢复仍可以增加面数；上限适用于实际输出、数量校正、局部尺寸反馈和最终选择。默认上限8192。若估计目标超过上限，程序要求显式调整参数，不静默截断。

生成前审查源网格。无硬特征基线先按历史实际面数校准尺度，最多三次后端调用，通过 `g_new = g_old * sqrt(target/actual)` 校正数量。基线要求实际面数在目标±10%内、且不超过上限，并通过原有基本检查和两个对角划分的相交检查。重算 `h=bbox/gsize` 和基线逐边缺陷，使用新基线对应的 UV、切缝索引和梳理方向场进入完整约束流程。整个新运行在同一个新 h 下比较；与历史结果比较时必须另用共同评价尺度。

后续最终推荐至少达到起始目标的90%。生成不到合格基线时保存失败记录并报错，不导出旧的低面数回退。有合格新基线而约束候选失败时，可以回退到新基线；这不算约束修复成功。

`--strategy always-feedback` 显式执行完整 v5。输出包括 `resolution/` 的基线尝试、输入散列和新计划，`method/` 的 policy/candidate/trace/decision/complete 记录，以及根目录 `run_summary.json`。有推荐时 `final/quad.obj` 是最终结果。输出目录必须不存在。

`--resolution archived` 使用原来的冻结基线和2048默认上限；不接受 `--target-quads`，用于复现历史低预算方法。`inspect_model.py` 仍只诊断冻结基线，不代表自动分辨率生成结果。

后端保持单线程与60秒/阶段设置。自动分辨率最多新增3次基线后端调用，方法最多12次逻辑尝试，分别计数。程序执行失败、超时或预算用尽均保留结果。跨编译器/依赖版本的新运行不保证与冻结二进制逐字节一致；请记录新二进制散列并比较指标。

## 原始记录与公开导出

results/records 下保留各阶段逐次结果和完整汇总，数值不因公开整理而更改。公开副本中的原本地绝对路径替换为 research://source-repository/... 等来源标识，这些是可追溯标识，不是可访问URL，也不是可直接执行的路径。

experiments/manifests/export_sources.json 保存原文件来源标识、原 SHA256、导出 SHA256 及变换类型。部分发布适配（如 CMake 源文件位置、README 和可移植入口）在 docs/validation.md 说明。publication_files.json 核对最终发布文件。

scripts/historical 是原实验/报告代码，含多个阶段的依赖假设；还原全部历史缓存需要原研究归档。不要把 research:// 标识直接传给历史脚本运行，也不要把缓存调用数当作独立重跑。

## 大型/外部资产

不提交虚拟环境、GPU训练权重、系统依赖安装包、第三方库整仓库、原生二进制和完整重复历史压缩包。NeurCross 可用其独立环境运行，本次 CPU 核验复用已保存方向场，不启动新的 GPU 训练。
