# 第三方来源与授权说明

本仓库整理研究代码和实验材料，没有为所有文件统一授予一个新的许可证。第三方内容按其原许可证使用，保留其作者与引用；协作者不要把公开可见等同于所有内容都采用MIT等统一授权。

|组件|来源与说明|
|---|---|
|NeurCross|https://github.com/QiujieDong/NeurCross ，原许可证复制在 neurcross/LICENSE（AGPLv3）；集成修改及来源散列有记录|
|libigl / CoMISo / libQEx|MIQ及四边形提取所需外部依赖；本仓库提供适配器源码，不重新打包这些库；取得依赖时保留各自许可|
|CGAL|https://www.cgal.org/ ，相交检查源码调用其Polygon Mesh Processing API；库不随仓库打包|
|Three.js r160|https://github.com/mrdoob/three.js ，viewer/assets 中的预存客户端脚本；MIT许可随assets保存|
|模型输入|来自 tue-alga/meshes 的 MAMBO 命名模型，固定revision f7dd0cb2258d6cd3dadbeef056a1d664d462a6ee；上游 https://github.com/tue-alga/meshes ，逐模型来源与散列见 experiments/manifests|

输入几何及其衍生网格不因本仓库发布而改变原始来源或再使用条件。本仓库没有为这些模型另行授予宽泛再许可。论文引用时同时说明原数据来源、所用固定revision和预处理；不要把输入形状当成本项目原创模型。

NeurCross论文：Qiujie Dong 等，*NeurCross: A Neural Approach to Computing Cross Fields for Quad Mesh Generation*，ACM TOG 2025，DOI 10.1145/3731159。

本研究方法尚无最终已发表题名与DOI；引用此开发版本请记录仓库URL和实际commit，不虚构出版信息。
