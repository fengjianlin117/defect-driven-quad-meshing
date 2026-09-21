# Defect-driven Quad Meshing

**面向类 CAD 三角网格，通过输出缺陷诊断、结构约束选择和尺寸协调，改善四边形网格的特征表达。**

本仓库是共同开发和论文研究的工作仓库。当前版本整理于 **2026-09-21**，包含已实现的方法、12 个开发模型、冻结结果、查看器和论文工作稿。它尚不是已经完成未见模型验证的最终论文发布版。

## 新协作者从哪里开始

| 你想做什么 | 从这里开始 |
|---|---|
| 理解研究问题和方法 | 本页流程、[方法说明](docs/method.md) |
| 直接检查生成网格 | 下载/克隆后打开 [viewer/index.html](viewer/index.html)，或查看 [静态图册](viewer/atlas.html) |
| 浏览数字与失败案例 | [结果索引](results/README.md)、[当前研究状态](docs/current_status.md) |
| 在 CPU 上检查代码和已有输入 | 下方快速开始；不需要 GPU 或 C++ 后端 |
| 配置后端并运行一个模型 | [复现与运行说明](docs/reproduction.md)、[后端构建](backend/README.md) |
| 修改方法或参与论文 | [协作约定](CONTRIBUTING.md)、[论文入口](paper/README.md) |

## 方法解决什么问题

已有方向场经 MIQ/QEx 生成的四边形网格，可能仍有棱线遗漏、网格边未沿特征对齐、窄区域分辨率不足等问题。我们比较源几何特征与实际输出，再决定需要哪些显式约束、整体多少四边形以及局部尺寸如何分配。

```text
源三角网格 ── NeurCross ──> 逐面方向场 PD1/PD2
                                  │
                                  v
                     无特征约束的 MIQ/QEx 基线
                                  │
源几何特征参考 ─────────────> 基线输出缺陷诊断
                                  │
              ┌───────────────────┴────────────────────┐
              │ 已足够好                               │ 需要干预
              v                                        v
          保留基线                  按需结构约束＋整体数量/局部尺寸提案
                                                       │
                                            生成并检查初始候选
                                                       │
                          ┌────────────────────────────┴─────────┐
                          │ 接受                                 │ 不接受
                          v                                      v
                         停止                         有限预算反馈，再接受或回退
```

**一次性提案是主体，反馈是需要时才使用的增强。**一次性阶段可包含两个尺寸分支和数量校正，并不等于只调用一次后端。新增约束、尺寸和反馈模块不训练布局预测网络；NeurCross 仍是学习式方向场前端，不能把整条流程称为完全无学习。

## 目录与职责

```text
method/          当前特征诊断、约束选择、尺寸分配、v5及条件协调器
neurcross/       独立的方向场生成前端及原许可证；已有场可直接复用
backend/         MIQ/QEx适配器与最终网格相交检查的C++源码
scripts/         可移植的检查/单模型入口；historical/保存原实验脚本
experiments/     12模型输入、逐面方向场、基线中间文件、协议和清单
results/         主比较、v5、条件策略、成本、逐次原始数值记录
examples/        按模型组织的原始三角网格及三阶段结果，共45份OBJ
viewer/          离线交互查看器、静态图册与来源记录
paper/           完整中文工作稿v0.4、证据映射和历史草稿
docs/            方法、复现、研究状态与版本边界
```

旧项目的顶层 `structural_graph/` **不属于当前方法依赖，未纳入本仓库**。当前特征参考由 `method/reference_graph.py`、`geometric_inventory.py` 和配套的 `weak_layout_pipeline` 基础模块构建。该基础包只保留当前方法和对应测试需要的模块，没有搬入旧的场补线工作流。

主要修改入口：

| 文件 | 职责 |
|---|---|
| `method/reference_graph.py`、`geometric_inventory.py` | 源几何特征和曲线组 |
| `method/feature_diagnostics.py`、`defect_predictor_v1.py` | 特征位置偏差与方向覆盖测量 |
| `method/defect_driven_selection.py`、`conflict_selection.py` | 根据缺陷选组并检查一类组合冲突 |
| `method/geometry_budget.py` | 整体数量、局部尺寸需求及预算分配 |
| `method/protection_feedback.py`、`coordination.py` | 约束和尺寸反馈操作 |
| `method/coordinator_v5.py` | 有界反馈与严格输出检查 |
| `method/conditional_coordinator.py` | 无需干预、一次性接受、反馈/回退的控制流程 |

部分文件名保留了历史版本号，因为当前实现仍实际调用其通用函数；这不表示默认方法回退到旧版本。

## 快速开始：先检查已有结果与代码

建议 Linux / WSL、Python 3.12。当前 CPU 依赖已单独列出，**无需先安装 PyTorch 或 CUDA**。

```bash
git clone https://github.com/fengjianlin117/defect-driven-quad-meshing.git
cd defect-driven-quad-meshing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt

# 核对发布文件散列、12模型参考图及方向场形状，不调用原生后端
python scripts/verify_snapshot.py

# 查看一个模型，并将重建参考图与缺陷驱动的初始提案写入新目录
python scripts/inspect_model.py --model B49 --out outputs/B49-inspection

# 方法回归测试
python -m pytest
```

`outputs/` 已在 `.gitignore` 中；输出目录必须是新目录，重复命令请换一个目录名。

查看网格可直接双击 `viewer/index.html`；也可运行下面命令后在浏览器打开 `http://127.0.0.1:8000`：

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory viewer
```

GitHub 文件页显示 HTML 源码，不会直接运行这个查看器；克隆到本地打开即可。

## 生成新结果

先按 [backend/README.md](backend/README.md) 配置 MIQ、QEx 和相交检查器，再运行：

```bash
python scripts/run_pipeline.py --model B49 \
  --miq build/backend/bin/miq_adaptive \
  --qex build/backend/bin/qex_adapter \
  --checker build/geometry/check_self_intersections \
  --out outputs/B49-conditional
```

这个入口使用仓库内该模型已经准备好的方向场、基线与参数化中间文件，运行当前条件策略。它**会执行后端求解**；它不是从任意 OBJ 自动完成全部前处理的通用产品入口。接入新模型需要准备对应输入契约，详见 [复现说明](docs/reproduction.md)。

`scripts/historical/` 是原实验编排与报告源码的可审阅导出，仍依赖历史实验目录；部署路径已替换为 `research://...` 标识。不要把这些历史脚本直接当作上述可移植入口运行。完整历史目录、远程训练检查点与原生二进制未整体搬入 Git。

## 当前结果怎么解读

全部 12 个模型都是**已见开发数据**，不能作为当前方法的未见泛化证据。

| 已完成工作 | 结果与边界 |
|---|---|
| 旧共同数量矩阵的一次性联合候选 | 6例两指标改善、1例取舍、3例无有效候选、2例数量未匹配；为旧矩阵重组 |
| v5完整反馈配置 | 10个非基线推荐、2个回退；不是10次同面数胜出 |
| 当前按需条件策略 | 1例无需介入、4例一次性接受、7例进入反馈，其中6例新推荐、1例回退 |
| 条件策略的逻辑尝试 | 83降至65，减少18次；精确缓存执行，不等于21.7%时间加速 |
| 核心特征处理与提案 | 0.180–1.395秒；不包含方向场、基线、额外后端和完整驱动耗时 |
| 最新数量对照 | 静态87次、外部51次尝试已完成；综合比较报告仍待整理 |

查看器显示的是**基线／有效初始候选／完整v5反馈**，最后一列不是条件早停输出。B32、B43、B33没有有效初始候选，保留空位；展示中的部分初始候选未通过原保护。B68/B61等早停存在质量或数量取舍，不能只报告节约调用数。

## 共同开发

1. 从 `main` 创建自己的功能分支，一次 PR 聚焦一个方法、文档或实验问题。
2. 修改方法时先运行 `pytest`，并使用新输出目录；不覆盖已有冻结结果。
3. PR 说明修改原因、涉及模型、实际面数、评价指标、运行代价和验证结果。
4. 新实验记录代码提交、输入散列、参数、种子、后端版本、全部失败与复用次数。
5. 提案、有效候选、最终接受结果、历史结果分别标明，不把复用计作独立样本。

完整约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。公开仓库可通过 fork + PR 协作；直接写权限由仓库所有者在 GitHub 中邀请指定协作者，本次未自动添加任何人。

## 当前限制、引用与来源

最终新颖性论证、未见模型族验证、完整成本、最新数量比较和最终论文复现包仍待完成；相容性筛查与尺寸加密不保证解决所有整数、方向场或几何冲突。完整论文最新为 [v0.4](paper/论文初稿_v0.4.md)，尚未纳入全部最新结果。

NeurCross、MIQ/QEx、CGAL、Three.js 和模型来源分别注明于 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。各第三方许可证保留；本次没有擅自为整个仓库指定一个覆盖所有代码与数据的统一许可证。

这是合作研究快照；科学结论以冻结记录及其版本身份为准，完整性与移植核验见 [发布验证](docs/validation.md)。
