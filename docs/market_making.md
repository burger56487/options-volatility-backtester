# 做市仿真（Market Making）

## 职责

多合约期权做市研究：在客户订单异步到达、执行成本与风险限额下比较
固定价差、库存偏斜、Greeks 偏斜、DP 与事件驱动 RL。

## 模块

- `src/market_making/greeks_book.py`：多合约 Greeks 感知报价（风险桶）；
- `src/market_making/simulator.py`：事件仿真（含亏损熔断）；
- `src/market_making/orderflow.py`：Poisson/Hawkes 订单流；
- `src/market_making/dp.py`：ODE/有限状态 DP 基准（V*(0,0)=1.90299 对拍）；
- `src/market_making/intensity_env.py` + `rl.py`：事件驱动 RL
  （连续时间 Linear-MC + 闭式 Critic、离散 AC 基准）；
- `src/market_making/study.py`：同订单流/同 seed/多种子公平对照；
- `src/backtest/account_market_making.py`：把报价策略接入账户引擎。

## 运行

```bash
PYTHONPATH=. python scripts/run_market_making_study.py
PYTHONPATH=. python scripts/run_account_market_making_demo.py
```

## 边界

订单流与成交概率是参数化研究模型，不代表真实交易所撮合；README/输出文件
均有 disclaimer。
