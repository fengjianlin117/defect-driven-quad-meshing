# NeurCross方向场前端

此目录是既有 NeurCross 集成源码，不是本次新增缺陷驱动方法。它负责从三角网格生成逐面交叉方向场；MIQ/QEx提取和本文约束/尺寸处理在其他目录。已提供12模型的 PD1/PD2，检查和使用这些场无需启动GPU训练。

上游：[QiujieDong/NeurCross](https://github.com/QiujieDong/NeurCross)，论文 *NeurCross: A Neural Approach to Computing Cross Fields for Quad Mesh Generation*，DOI：[10.1145/3731159](https://doi.org/10.1145/3731159)。原AGPLv3许可见 [LICENSE](LICENSE)。

本集成包含自动停止、可重复检查点、几何初始化兼容和最终场导出。原本地目录缺失但模型仍导入的 models/UNet.py，已从此前保存的远程源码比较副本补回，来源及散列保存在 experiments/manifests/export_sources.json。不能因为它缺失就删掉当前网络中的引用；它确实参与角度预测。

当前CPU发布验证不包含重新训练，GPU环境需单独配置。依赖包括兼容CUDA的PyTorch、NumPy、SciPy、trimesh、torchinfo、timm；部分可选路径使用torch_kmeans。远程既有环境清单如已导出，见 experiments/environment。

入口为 quad_mesh/train_quad_mesh.py，参数见 quad_mesh/quad_mesh_args.py。传入明确的 --data_path、--logdir、--model_name、--seed 等参数；默认训练轮次及收敛策略不应为了新结果临时修改。工作目录设为 neurcross/quad_mesh，并将 neurcross 目录加入 PYTHONPATH。首次运行应先核对环境和源码来源；不要把本说明理解为已经完成独立GPU重训验收。

发布整理只调整机器专用默认路径与补齐已有依赖，没有重训或改变论文方法。输出方向场必须保持与源三角网格相同的面顺序，再交给method/backend使用。
