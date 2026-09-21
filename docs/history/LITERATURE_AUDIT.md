# 一手文献核验与贡献边界

核验日期：2026-09-20。已下载5篇原始论文并保存URL、SHA256、页数与文本。检索摘要只是入口；下列机制判断已核对论文正文或官方实现。此表仍需扩充近年工作，不能视作完整新颖性查新结论。

| 文献 | 原始资料 / 官方代码 | 核对位置与相关机制 | 对本研究的影响 |
|---|---|---|---|
| Bommes, Zimmer, Kobbelt. Mixed-Integer Quadrangulation. TOG 28(3), 2009 | [作者机构论文页](https://www.graphics.rwth-aachen.de/publication/0344/) | 稀疏方向/对齐约束、十字场、全局无缝混合整数参数化 | 这些步骤均非本研究新贡献；需明确复用后端 |
| Ebke, Bommes, Campen, Kobbelt. QEx: Robust Quad Mesh Extraction. TOG 32(6), 2013 | [机构记录与DOI](https://publications.rwth-aachen.de/record/238081) | 从含退化/翻折参数化中稳健提取；论文首图及算法 | 不能因为UV警告直接丢弃最终有效候选，也不能将QEx鲁棒性归因于我们的筛选 |
| Gehre, Lim, Kobbelt. Adapting Feature Curve Networks to a Prescribed Scale. CGF 35(2), 2016 | [DOI](https://doi.org/10.1111/cgf.12834)、本地gehre2016.pdf | 第5.3.2节、式(1)–(9)：带冲突约束的二元特征边选择；第6节限制；整曲线可共享变量 | “按尺度选择相容特征集合”已有直接先例。本研究可能的区别是保留完整参考并由实测残差协调求解，但这尚待实验和更全面查新支持 |
| Pietroni, Nuvoli, Alderighi, Cignoni, Tarini. Reliable Feature-Line Driven Quad-Remeshing. TOG 40(4), 2021 | [论文网站](https://www.quadmesh.cloud/)、[官方QuadWild](https://github.com/nicopietroni/quadwild) | 第2.3节讨论特征约束/参数化冲突；图2；第6节离散边数优化；第8.1节限制 | 强相关CAD基线。采用可含T结点的中间布局和最终一致纯四边形细分；不能称其“无任何失败保证”。官方构建依赖Gurobi及Qt，当前环境尚未发现配置 |
| Dong et al. NeurCross: A Neural Approach to Computing Cross Fields for Quad Mesh Generation. TOG 44(4), 2025 | [DOI](https://doi.org/10.1145/3731159)、[官方实现](https://github.com/QiujieDong/NeurCross) | 官方发表版本作者和标题与早期预印本不同；官方README确认libigl/libQEx提取 | 复用场需注明发表版本与本地实际代码/种子。场学习、SDF与提取器本身不是我们的贡献 |
| Huang et al. QuadriFlow: A Scalable and Robust Method for Quadrangulation. CGF, 2018 | [作者项目页](https://stanford.edu/~jingweih/papers/quadriflow/)、[官方实现](https://github.com/hjwdzh/QuadriFlow) | 官方CLI有目标面数、sharp、SAT及seed选项；默认不开sharp | 可运行外部基线，须同时报告默认和sharp。本阶段已独立构建revision `810b7a0967c35b0dc85b4464e3835e26a756c967`；尚未以该构建输出作性能结论 |

尚待完整核读的一手相关工作：[Gmsh面向复杂CAD的准结构四边形流水线](https://arxiv.org/abs/2103.04652)、[2025年Messy Grid Preserving Maps提取分析](https://arxiv.org/abs/2507.15404)、[TopGen预印本](https://arxiv.org/abs/2603.10606)。不能仅列参考文献就声称已完成比较。还需核对经典尺寸渐变、整数网格有效性与近期尺寸可控场方法。

本轮可继续检验的主张是：**在固定场和有限后端预算下，用实测几何缺陷定位约束及尺寸需求，并记录未满足需求、保留中间候选和全部共同参考，以改善质量—面数取舍。** 该句目前是待检验主张，不能写成已普遍优于已有方法的论文结论。
