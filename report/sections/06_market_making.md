# 6. 做市仿真

## 6.1 设计

事件驱动：客户订单按点过程到达（Poisson 基准 / Hawkes 强度路径），每个事件
经过固定顺序的处理链（到达 → 状态更新 → 报价/成交判定 → 库存/现金更新 →
风险检查 → 盯市/对账），同一时间戳下用序列号保证顺序确定，避免“用成交后
信息生成成交前报价”的前视。

## 6.2 多合约 Greeks 感知报价

`greeks_book.py` 把跨行权价/期限的净 Delta/Gamma/Vega 聚合到风险桶：
Delta 采用 A–S 保留价偏移，Gamma/Vega 按各腿对净风险的贡献加宽价差，
并按 0–7/7–30/30–90/90+ 交易日分桶。

## 6.3 策略与基准

- 固定价差；
- 库存偏斜（A–S 扩展）；
- Greeks 偏斜；
- 动态规划（ODE 基准，指数成交闭式交叉验证，V*(0,0)=1.90299 对拍）；
- 事件驱动 RL：Algorithm 1（连续时间 Linear-MC，闭式 Critic）与
  Algorithm 2（离散时间 AC 基准），Linear-Pair Actor。

`study.py` 保证同一订单流/同一 seed/多种子聚合的公平对照，输出
`comparison.csv`、`aggregated.csv`（含置信区间）与图表。

## 6.4 边界

订单流与成交概率是参数化研究模型，不代表真实交易所撮合；README 与
`market_making_study_*/summary.json` 均有明确 disclaimer。
