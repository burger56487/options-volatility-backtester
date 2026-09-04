# C++ 性能基准（需要你在有编译器的机器上跑）

本机（开发沙箱）没有 C++ 编译器，因此加速比无法在这里测量。
仓库已准备好等价的单线程基准，编译运行后把两端秒数相除即得加速比。

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
