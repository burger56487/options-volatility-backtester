# 波动率与曲面（Volatility）

## 职责

把原始行情/期权快照变成可研究、可校准的波动率曲面：数据清洗、无套利检查、
隐含波动率反解、SVI 校准与曲面插值。

## 数据管线

`src/market_data/pipeline.py` 依次执行字段/类型检查、日期标准化、OHLC
逻辑检查、期权报价检查、相对价差检查、欧式无套利边界、重复记录隔离、
质量报告与血缘哈希。

## 真实链（快照分析）

```bash
PYTHONPATH=. python scripts/download_real_spy_option_snapshot.py   # 需网络
PYTHONPATH=. python scripts/evidence_run.py                        # 校准证据
```

已提交快照（`outputs/real_option_chain/`）：1500 条报价 → 872 条活跃
（OI>0 且 |log-moneyness|≤0.15）→ 734 条成功反解 IV（138 条反解失败，
与无套利违规一致）。

## SVI

总方差 w(k)=σ²T，逐到期校准（`src/volatility_surface/calibration.py`）。
实测 6 个到期日 RMSE 3.9e-5~1.7e-4（`outputs/evidence_20260904.json`）。
蝶式/日历无套利检查见 `src/volatility_surface/arbitrage.py`。

## 局限

单一快照；SPY 期权为美式，当前用欧式边界近似；跨日稳定性/随机波动率识别
列为未来工作。
