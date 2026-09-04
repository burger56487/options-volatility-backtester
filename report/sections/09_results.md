# 9. 结果与讨论

## 9.1 核心结果表（全部来自仓库运行结果）

| 实验 | 关键数字 | 来源 |
|---|---|---|
| 账户引擎滚动跨式 | 17 笔，PnL −8038.77，对账最大差异 3.5e-11 | `outputs/account_rolling_*` |
| legacy 滚动（无过滤） | 33 笔，PnL −8038.86，Sharpe-like −1.74，回撤 −8.08% | `outputs/volatility_filtered_backtest/baseline_summary.json` |
| 波动率过滤阈值 1.30 | 6 笔，PnL +1357.48，Sharpe-like +0.32，回撤 −2.22% | `outputs/regime_sensitivity/...csv` |
| 严格样本外（阈值 1.10） | 测试段 5 笔，PnL +1731.79，年化 Sharpe 1.28，Bonferroni α=0.0125 | `outputs/strict_evidence2/test_metrics.json` |
| SVI 校准 | 6 个到期日 RMSE 3.9e-5~1.7e-4 | `outputs/evidence_20260904.json` |
| VaR 回测 | 87/1194 例外，Kupiec p=0.0007，Christoffersen p=0.78 | 同上 |
| C++ 单价格基准 | 约 40.8x（1M 次 -O2） | `docs/CXX_BENCH.md` |
| C++ 批量 vs Python 标量 | 约 3.8x（价格差 8e-11） | `outputs/testing/benchmark_baseline.json` |
| 测试/覆盖率 | 300+ 测试；86% | CI artifact |

## 9.2 负结果的科学解释（最值得讲的部分）

无条件系统性买入长跨式在样本内产生 **−8.04%** 回报（Sharpe-like −1.74）。
这不是系统 bug，而是期权经济学的直接结果：买方需要实现波动率超过隐含波动率
加上 Theta 衰减、买卖价差与对冲成本。一个天真的买波动率策略若在成本模型中
稳定盈利，反而说明仿真有缺陷。

引入波动率状态过滤（仅当短期已实现波动率相对长期抬升时入场）后，同一
候选样本下 Sharpe-like 从 −1.74 改善：阈值 1.30 时为 **+0.32**（6 笔，
回撤收窄到 −2.22%）；阈值 1.10 时 11 笔 +298 PnL。严格样本外（参数只在
训练/验证段选择、测试段锁定）测试段 5 笔取得 +1731.79。

## 9.3 不确定性度量与诚实声明

- 样本很小：滚动交易 6–33 笔、测试段 5 笔，统计功效低，置信区间宽
  （legacy 33 笔的 Sharpe-like 自助区间宽约 6.10）；
- 过滤改善方向与期权理论一致，但小样本下无法排除偶然性；不声称
  “统计显著”；
- 单标的（SPY）、合成历史期权报价、参数化订单流——这些限制在
  “局限性”章逐条说明。
