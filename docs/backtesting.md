# 回测框架（Backtesting）

## 职责

严格信息时点的策略回测：从信号生成到成交、持仓估值、融资、成本、风险与
逐笔对账的完整链路。

## 时间线与前视控制

- 第 t 日信号只用截至 t−1 日收盘可用的数据；
- 特征带 JSON sidecar 声明可用时间，审计模块拒绝后视特征；
- 样本外：训练/验证/测试三区段，测试集被 `evaluation/test_evaluation_log.json`
  锁定；跨边界交易单独记录。

## 账户与执行

`src/portfolio/`（现金账本、持仓、估值、PnL 桥）、`src/execution/`
（盘口、佣金/滑点/冲击、陈旧报价拒绝、部分成交）、`src/financing/`、
`src/risk/`（限额与交易前检查）。

## 运行

```bash
PYTHONPATH=. python scripts/run_account_rolling_demo.py   # 账户引擎（17 笔）
PYTHONPATH=. python scripts/run_strict_evaluation.py      # 严格样本外
PYTHONPATH=. python scripts/run_volatility_filtered_backtest.py
```

实测：账户引擎 17 笔总 PnL −8038.77，逐笔对账最大差异 3.456e-11；
legacy close-to-close（33 笔）−8038.86（计费口径差异约 0.09）。

## 局限

历史期权报价为合成；成交为规则模拟；保证金为研究性近似。
