# 8. 验证体系

## 8.1 测试分层

75 个测试文件、300+ 测试覆盖：单元（定价/Greeks/仓储）、集成
（账户+执行+风控、API+DB）、数值正确性、属性测试（Hypothesis：无套利边界、
看跌看涨平价等恒等式）、可复现性（同 seed 结果一致）、golden 回归
（重算并比对已记录结果）、模型验证套件与性能回归（perf，默认不跑）。

## 8.2 模型验证套件

`scripts/run_validation_report.py` 生成 `outputs/testing/validation_report.*`，
包含：CRR/CN 收敛、MC 标准误按 1/√N 下降、Heston/Merton 退化、SVI 校准
质量、真实 SPY 的 VaR 回测。任一项失败 CI 即红。

## 8.3 VaR 回测（真实 SPY 日收益）

60 日滚动 95% 历史 VaR，回测 1194 天：

| 指标 | 值 |
|---|---|
| 实际例外 | 87 |
| 期望例外（5%） | 59.7 |
| Kupiec p | 0.00066（覆盖被拒绝） |
| Christoffersen p | 0.78（无聚类证据） |

## 8.4 CI 与覆盖率

- GitHub Actions：`lint`（ruff + mypy on validation）、`test`
  （pytest + 覆盖率门槛 80%、C++ 编译+测试、验证报告/覆盖率 artifact、
  PostgreSQL 服务容器）；
- 实测覆盖率约 86%（本地 86.4%，Linux 85.1%）；
- 性能回归通过 workflow_dispatch 手动运行。
