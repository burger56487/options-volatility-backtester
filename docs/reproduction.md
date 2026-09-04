# 复现指南

> 原则：照着做能复现 README/报告里的核心数字；本文件中的命令都已在仓库
> 环境下验证过。

## 环境

- Python 3.12（依赖锁定：numpy 2.5 要求 ≥3.12）；
- 可选：C++ 编译器（g++/zig）、Docker（平台部署）。

## 1. 安装

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 2. 测试与验证报告

```bash
PYTHONPATH=. python -m pytest -q -m "not perf"
PYTHONPATH=. python scripts/run_validation_report.py
```

预期：全量测试通过（本机 300+），`outputs/testing/validation_report.json`
中 `all_passed=true`（若 C++ 未编译，相关用例会 skip 而非失败）。

## 3. 核心回测数字

```bash
PYTHONPATH=. python scripts/run_account_rolling_demo.py
```

预期输出：`trades: 17`、`total pnl: -8038.77`、
`reconciliation failures: 0`（输出目录内 `summary.json` 的最大对账差异
约 3.456e-11）。

严格样本外与波动率过滤：

```bash
PYTHONPATH=. python scripts/run_strict_evaluation.py
PYTHONPATH=. python scripts/run_volatility_filtered_backtest.py
PYTHONPATH=. python scripts/run_regime_sensitivity_analysis.py
```

## 4. 曲面与 VaR 证据

```bash
PYTHONPATH=. python scripts/evidence_run.py
```

输出 `outputs/evidence_20260904.json`，关键值：

| 指标 | 预期值 |
|---|---|
| SVI 分到期 RMSE | 3.9e-5~1.7e-4（6 个到期日） |
| VaR 回测 | 1194 天 / 87 例外 |
| Kupiec p | ≈0.0007 |
| Christoffersen p | ≈0.78 |

## 5. C++ 内核（可选）

```bash
g++ -shared -O2 -w -std=c++17 -fPIC -pthread \
    cpp/src/bs_kernels.cpp -o outputs/bs_kernels.so
PYTHONPATH=. python -m pytest -q tests/pricing/test_cpp_backend.py -m cpp
```

预期：7 个一致性测试通过；`scripts/run_benchmark_baseline.py` 生成
`outputs/testing/benchmark_baseline.json`（加速比随硬件变化，不承诺固定值；
本机参考 ~3.8x）。

## 6. 图表与报告

```bash
PYTHONPATH=. python scripts/generate_report_figures.py
```

在 `report/figures/` 生成 8 张标准图；研究报告见 `report/`。

## 数字来源与诚实声明

所有“预期值”都来自仓库 `outputs/` 中已提交/可再生的运行结果。样本量
（6–33 笔）与合成期权报价等限制在报告第 10 章与 README“局限性”节如实
披露；若环境/数据不同，数字可能变化，请以当次运行输出为准。
