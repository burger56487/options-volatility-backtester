# 动态对冲（Hedging）

## 职责

比较不同 Delta/风险对冲策略在离散交易、交易成本与模型错配下的表现。

## 策略

- 固定频率对冲；
- Delta 阈值对冲（含风险限额比例）；
- 成本感知对冲（无交易区间）；
- RL 对冲（`src/hedging/rl_env.py`）。

## 运行与输出

```bash
PYTHONPATH=. python scripts/run_delta_hedging_demo.py
PYTHONPATH=. python scripts/run_sensitivity_analysis.py
```

输出：`outputs/sensitivity/hedging_sensitivity_results.csv` 与
`outputs/figures/spy_hedging_sensitivity_*.png`。

## 关键结论

更频繁的对冲降低跟踪误差但线性推高成本，存在成本-风险前沿；波动率错配的
影响可被 Gamma/Theta 归因量化解释（`src/backtest/pnl_attribution.py`）。

## 局限

标的对冲为简化执行假设（无停牌/借券冲击反馈等）；RL 对冲为研究性实验。
