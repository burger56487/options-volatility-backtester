# C++ 性能基准（需要你在有编译器的机器上跑）

## 实测结果（2026-09-04，本机，单线程 -O2）

```text
C++ (zig c++ 0.16.0): paths=1000000 seconds=0.014869
Python 3.x 标量循环:   paths=1000000 seconds=0.606757
加速比 ≈ 40.8x
两端价格输出均为 7.7215522304（数值一致）
```

编译器：Zig 0.16.0（自带 clang/LLVM，-O2 -std=c++17）。

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
