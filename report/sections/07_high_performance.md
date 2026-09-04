# 7. 高性能计算

## 7.1 设计

C++17 内核（`cpp/src/bs_kernels.cpp`）覆盖批量 BS 价格+Greeks、批量 IV、
GBM MC（单线程/多线程）、情景重估与组合 VaR/Euler 贡献。绑定采用 **ctypes**
（替代 pybind11，调用时释放 GIL、无 Python 头文件依赖），多线程用
`std::thread`。每个内核都有与 Python 参考实现的一致性测试（1e-9 量级）。

## 7.2 加速比（必须区分两种口径）

1. **独立单价格基准**（同公式 100 万次，单线程 -O2，2026-09-04 实测）：
   C++ 0.0149s vs Python 0.6068s，**约 40.8x**；
2. **批量内核 vs Python 标量循环**（100 万笔相同样本，2026-09-04 实测，
   `outputs/testing/benchmark_baseline.json`）：Python 0.592s vs
   C++ 0.157s，**约 3.8x**，价格差 8.2e-11。

两种口径不矛盾：独立基准测量纯算术热点；批量内核还包括数组内存搬运与
六列 Greeks 输出。面试/文档一律区分口径，不做夸大。

## 7.3 诚实发现

- 内存带宽受限的简单批量定价，多线程收益有限（基准中如实标注）；
- 数值一致性优先于速度：不启用 -ffast-math / -march=native，编译参数写入
  文档，保证 C++ 与 Python 结果可对拍。

Linux 下 CI 用 g++ -O2 -fPIC 编译 `.so` 并运行同一套一致性测试。
