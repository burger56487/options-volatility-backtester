# 高性能计算（C++）

## 职责

把批量定价、IV 反解、蒙特卡洛、情景重估与组合 VaR 热点用 C++17 实现，
通过 ctypes 绑定（计算段释放 GIL），多线程用 `std::thread`。

## 内核

`cpp/src/bs_kernels.cpp`：批量 BS 价格+Greeks、批量 IV、GBM MC（单线程/
多线程）、情景 PnL、组合 VaR/Euler 贡献、曲面批量重定价。

## 构建

Linux：

```bash
g++ -shared -O2 -w -std=c++17 -fPIC -pthread \
    cpp/src/bs_kernels.cpp -o outputs/bs_kernels.so
```

Windows（zig c++ 或 MinGW，输出 `outputs/bs_kernels.dll`）与 macOS 命令见
`docs/CXX_BENCH.md`。Python 侧自动探测 `.dll` / `.so`。

## 一致性测试与基准

```bash
PYTHONPATH=. python -m pytest -q tests/pricing/test_cpp_backend.py -m cpp
PYTHONPATH=. python scripts/run_benchmark_baseline.py
```

实测（2026-09-04）：独立单价格基准 ~40.8x；批量内核 vs Python 标量循环
~3.8x（价格差 8.2e-11）。两种口径在报告与 README 中严格区分，不做夸大。

## 诚实发现

内存带宽受限的简单批量定价多线程收益有限；为数值一致性不启用
-ffast-math/-march=native，编译参数写入文档。
