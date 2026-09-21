# 原生后端构建

本目录提供源码，不包含现有机器上的二进制。当前方法调用 miq_adaptive、qex_adapter 和 triangle-soup 几何检查器。

## MIQ/QEx

需要 C++17、CMake≥3.20、Eigen、BLAS/LAPACK、OpenMP、OpenMesh，以及 libigl、CoMISo（含GMM头文件）和libQEx。首先按各项目文档取得依赖并构建libQEx；依赖目录布局为：

```text
deps/
├── libs/libigl/include/
├── libs/CoMISo/
├── libs/libQEx/interfaces/c/
└── build/libqex/libQExStatic.a
```

```bash
cmake -S backend -B build/backend -DCMAKE_BUILD_TYPE=Release \
  -DMIQ_BACKEND_ROOT=/absolute/path/to/deps
cmake --build build/backend --target miq_adaptive qex_adapter -j2
build/backend/bin/miq_adaptive --capabilities
```

Eigen/OpenMesh或QEx位于其他位置时，可传 EIGEN3_INCLUDE_DIR、OPENMESH_INCLUDE_DIR、OPENMESH_LIBRARY_DIR、LIBQEX_STATIC_LIBRARY。构建脚本做了发布目录适配，并修复空缓存变量阻止QEx库自动查找的问题；C++算法与原研究版本一致。

## 几何检查器

使用 CGAL 5.6系列的 triangle_soup_self_intersections API，以及GMP/MPFR。它对应v5冻结实验使用的 triangle-soup 源码，不是旧 Surface_mesh 版本。

```bash
cmake -S backend/geometry_checks -B build/geometry -DCMAKE_BUILD_TYPE=Release
cmake --build build/geometry -j2
build/geometry/check_self_intersections experiments/inputs/B34/source.obj 0
```

CGAL未安装到系统目录时，用 CMAKE_PREFIX_PATH 或 CGAL_DIR 指向其安装位置。完整流程会分别检查两种对角剖分；手动只运行对角0不能替代完整检查。

系统库与第三方源码许可证由其上游提供，本仓库不复制整套依赖。第一次构建请保存版本/commit和编译日志；新编译二进制不保证与历史文件逐字节一致。
