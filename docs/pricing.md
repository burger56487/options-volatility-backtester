# 定价引擎（Pricing）

## 职责

统一的期权定价框架：六种模型 + 局部波动率，共享 `PricingRequest` /
`PricingResult`，并附带收敛、退化、恒等式与属性测试。

## 模型与文件

- Black–Scholes–Merton：`src/pricing/black_scholes.py`
- CRR 二叉树（欧式/美式）：`src/pricing/binomial.py`
- 有限差分（显式/隐式/Crank–Nicolson）：`src/pricing/finite_difference.py`
- 蒙特卡洛（对偶 + 控制变量）：`src/pricing/monte_carlo.py`
- Heston（CF + 全截断 MC + Fourier 半解析）：`src/pricing/heston.py`
- Heston 校准：`src/pricing/heston_calibration.py`
- Merton（级数 + compound-Poisson MC）：`src/pricing/merton.py`
- Dupire 局部波动率：`src/pricing/local_vol.py`
- 统一注册/对比：`src/pricing/registry.py`、`src/pricing/calibration.py`

## 示例

```python
from src.pricing.black_scholes import option_price
price = option_price(
    spot=100.0, strike=100.0, time_to_expiry=0.5,
    risk_free_rate=0.04, volatility=0.25,
    option_type="call", dividend_yield=0.01,
)
```

命令行演示：

```bash
PYTHONPATH=. python scripts/run_pricing_demo.py
```

## 验证（2026-09-04 实测）

- CRR 100→800 步误差 1.74e-2→2.18e-3（步数翻倍误差约减半）；
- Crank–Nicolson（400 步）误差 3.5e-2；
- Merton 零跳跃级数解与 BS 误差 <1e-9；
- Heston σ_v→0 半解析定价精确回到 BS（误差 0）；
- Heston 合成曲面校准：RMSE 2.6e-4，参数恢复至 4 位有效数字。

实现期间修复了两个真实数值 bug（Heston CF 缺失对数分支项；Euler 使用
步末 v 导致的 ρσ/2 漂移偏差），报告第 2 章有完整说明。

## 局限

有限差分以欧式为主（美式提前行权用 CRR 处理）；MC 尚未含 Sobol/桥；
Heston 校准基于合成曲面（单快照不足以识别随机波动率参数）。
