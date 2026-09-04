# 5. 动态对冲

## 5.1 策略族

- 固定频率 Delta 对冲（日频/周频等）；
- Delta 阈值对冲（可配置阈值 + 风险限额比例）；
- 成本感知对冲（比较“不调仓的风险损失”与“调仓成本”，生成无交易区间）；
- RL 对冲（状态含价格/时间/波动率/Greeks/持仓/成本，动作=目标对冲头寸）。

## 5.2 实验

`outputs/sensitivity/hedging_sensitivity_results.csv` 记录不同再平衡阈值、
佣金/滑点假设下的对冲误差、换手、成本与尾部损失；`outputs/figures/` 下
有对应的 PnL/换手/成本敏感性图。关键结论：

- 更频繁的对冲降低跟踪误差但线性推高交易成本，存在明显的成本-风险前沿；
- 波动率错配（定价 σ vs 实现 σ）对单笔跨式 PnL 的影响可被
  Gamma/Theta 归因量化解释。

详见 `src/hedging/`、`tests/hedging/` 与
`scripts/run_sensitivity_analysis.py`。
