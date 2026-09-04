# C++ 性能基准（需要你在有编译器的机器上跑）

## 实测结果（2026-09-04，本机，单线程 -O2）

```text
C++ (zig c++ 0.16.0): paths=1000000 seconds=0.014869
Python 3.x 标量循环:   paths=1000000 seconds=0.606757
加速比 ≈ 40.8x
两端价格输出均为 7.7215522304（数值一致）
```

编译器：Zig 0.16.0（自带 clang/LLVM，-O2 -std=c++17）。

## 批量内核基准（2026-09-04，outputs/cpp_evidence.json）

- 批量 BSM 价格+Greeks（20 万笔，同一随机样本，单线程）：
  C++ 0.024s vs Python 111.5s，**加速约 4680×**；
  价格/Greeks 与 Python 最大绝对差 ~1e-13（机器精度级）。
- 批量隐含波动率反解（3000 笔，±15% 价内，牛顿+二分，tol=1e-8）：
  C++ 0.012s vs Python 6.88s，**加速约 557×**；最大绝对差 8e-7，0 失败。
- 批量蒙特卡洛（100 万路径，单线程 GBM）：C++ 0.040s vs numpy 向量化
  Python 0.065s，加速约 1.6×；两者价格差约 2 个标准误以内
  （numpy 基线已是向量化实现，加速空间在并行与批量场景）。

实现：`cpp/src/bs_kernels.cpp`（extern "C" 批量 BSM/Greeks/IV），
`src/pricing/cpp_backend.py`（ctypes 后端切换层，含 Python 回退），
`scripts/cpp_verify_bench.py`（批量一致性 + 计时）。

## 任务 15 收尾（2026-09-04）

- 批量 MC（单线程 + 4 线程并行，`mc_gbm_parallel`）；并行与单线程结果
  在 3 个合并标准误内一致。
- 情景重估 `scenario_pnl`：批量 spot/vol 冲击逐期权 PnL，与 Python 一致 1e-10。
- 组合 VaR/Euler `portfolio_var`：与 Python `linear_risk_contributions`
  一致 1e-9，贡献和 = VaR。
- 曲面批量重定价 `cpp_surface_prices`：VolSurface 节点 → C++ 批量 BSM，
  与 Python 逐点一致 1e-9。
- 端到端演示 `benchmarks/cpp_backend_demo.py`：27 个曲面节点 -10% 标普/vol+5
  冲击合计 PnL −106.32；VaR 23.26 且 Euler 贡献和等于 VaR。
- 绑定方式：ctypes（调用时释放 GIL，无 Python 头文件依赖），替代 pybind11。

## Python 基线

```bash
python benchmarks/python_bs_loop.py
# 输出: paths=1000000 seconds=... avg=...
```

## C++ 内核

```bash
# Linux/macOS (g++)
g++ -O2 -std=c++17 cpp/benchmark/black_scholes_bench.cpp -o /tmp/bs_bench
/tmp/bs_bench

# Windows: 用 Visual Studio 的 Developer Command Prompt 或 MinGW g++
g++ -O2 -std=c++17 cpp/benchmark/black_scholes_bench.cpp -o bs_bench.exe
bs_bench.exe
```

## GitHub Actions（免本地安装）

仓库里加一个 ubuntu workflow 跑 `g++ -O2` 编译上述文件并执行，
把 C++ 秒数写入日志即可得到可复现加速比。

两边输出的 `avg` 应一致（参考值 7.721552230287969），
`加速比 = python_seconds / cpp_seconds`。
