# 首次协作发布核验

日期：2026-09-21。本次是现有研究成果整理与可移植入口检查，没有开展新的方法实验或GPU训练。

## 已验证

- 从当前冻结方法导出实际依赖闭包与相应方法测试，排除旧顶层 structural_graph 及其旧工作流。
- 原工作区依赖审计中，拦截旧包导入后重建12份参考图，与冻结 graph.json 全部完全一致；原始核对结果在 experiments/manifests/dependency_audit.json。
- 在独立发布目录运行 `python -m pytest -q`：54 passed。
- B49执行新的inspect_model入口成功：1052源三角面，参考图一致；基线499Q，初始目标499Q，无后端求解。
- B34执行新的条件流程入口成功：保留483Q基线，0逻辑尝试、0新增MIQ/QEx调用，最终OBJ复制成功。该检查使用现有原生后端和几何检查器，不冒充重新编译后的全矩阵验证。
- 随后从发布目录成功编译miq_adaptive、qex_adapter和triangle-soup几何检查器；MIQ能力查询确认density_aware=true。使用新编译的三个程序再执行B34，仍为483Q、0新增MIQ/QEx求解，最终OBJ与保存基线逐字节一致。这仍不是完整矩阵重跑。
- 独立发布目录中12份参考图全部重建一致；逐面场行数及0起始面索引通过检查。
- Python文件完成语法解析；公开导出保留数值，替换原机器部署路径。

最终发布文件由 experiments/manifests/publication_files.json 记录SHA256；`scripts/verify_snapshot.py` 核对散列和12模型的参考图、方向场行数/索引。该清单排除自身、Git目录、本机输出、构建和缓存目录。

## 发布适配与来源

- 新增 scripts/project_paths.py、inspect_model.py、run_pipeline.py、verify_snapshot.py、refresh_manifest.py，负责相对路径与检查/单模型入口；method算法内容保留，仅constraint_preflight.py及test_preflight.py的CRLF换行为LF，以保证跨平台Git散列一致。
- backend/CMakeLists.txt 将adaptive C++源文件位置改为发布目录，并修复空缓存变量阻止QEx库自动查找的问题；不改动C++算法。
- geometry_checks 导出的是已用于v5的 intersection_soup_audit_v2/checker.cpp，而不是旧Surface_mesh检查器。
- NeurCross的UNet.py从已有远程源码比较副本补齐；输入参数改为明确传入网格，输出默认改为相对路径。未重新训练。
- 报告/历史脚本中的私人部署路径替换为research://来源标识，原文件散列与导出文件散列分别保留。个别发布后路径适配还会反映在publication_files中。

## 尚未验证的范围

完整GPU重训、所有历史缓存矩阵在新机器的重放、跨平台二进制一致性和未见模型泛化未在本次发布中验证。仓库提供协作起点与已有证据，不把上传成功等同于论文验收完成。
