OPTIONS VOLATILITY TRADING AND DYNAMIC HEDGING BACKTESTER

期权波动率交易与动态对冲研究平台

基于真实标的行情与合成历史期权报价的可复现研究平台，支持期权定价、隐含波动率、
动态 Delta 对冲、交易成本分析、风险指标与参数敏感性研究。

![Python Tests](https://github.com/burger56487/options-volatility-backtester/actions/workflows/tests.yml/badge.svg)

Overview

This is a Python research project for studying delta-hedged long straddle
strategies using real SPY historical prices and a transparent synthetic option
pricing framework.

The project includes Black-Scholes-Merton pricing, Greeks, implied volatility
recovery, realised volatility analysis, synthetic implied volatility surfaces,
ATM long straddle construction, dynamic delta hedging, transactions costs,
single-trade backtesting, rolling backtesting, volatility-regime filters, and
parameter sensitivity analysis.

Important disclaimer:

This repository is an educational derivatives-research project. It does not
provide investment advice, trading recommendations, or evidence that a strategy
will be profitable in real markets.

================================================================
RESEARCH QUESTIONS
================================================================

This project investigates the following questions.

How do option premium, theta decay, gamma exposure, and transaction costs affect
a delta-hedged long straddle?

How does delta-rebalance frequency affect hedge turnover, transaction costs, and
profit and loss?

How sensitive is strategy performance to underlying slippage assumptions?

Can a realised-volatility regime filter reduce unconditional long-volatility
losses and drawdowns?

How robust are results across different volatility-ratio thresholds?

================================================================
DATA AND MODELLING ASSUMPTIONS
================================================================

Underlying data:

Underlying asset: SPY ETF.
Frequency: Daily adjusted OHLCV prices.
Source: yfinance.
Sample period: 2021-01-01 to 2026-01-01.

Synthetic option chain:

Historical SPY option quotes are not used in this project. Option prices are
generated from a transparent synthetic implied-volatility surface.

The synthetic surface combines:

Blended 20-day, 60-day, and 252-day realised volatility.
A variance-risk premium.
An upward maturity term structure.
Volatility smile curvature.
Negative put skew.
Synthetic bid-ask spreads.

This design makes assumptions inspectable and reproducible. However, the
results should not be interpreted as a historical real-option-market backtest.

================================================================
MAIN FEATURES
================================================================

Option pricing and Greeks:

Black-Scholes-Merton European call and put pricing.
Delta, Gamma, Vega, Theta, and Rho.
Continuous dividend yield support.
Put-call parity validation.
Implied-volatility recovery using Newton-Raphson with bisection fallback.
No-arbitrage option-price bounds.

Market data and volatility analytics:

SPY daily adjusted-price download and validation.
OHLCV quality checks.
Date sorting and duplicate handling.
Simple and logarithmic returns.
20-day, 60-day, and 252-day annualised realised volatility.
Blended realised-volatility inputs for synthetic option pricing.

Synthetic option market:

Strike grid from 80 percent to 120 percent moneyness.
30, 60, 90, and 180 day expiries.
Synthetic implied-volatility smile and term structure.
Bid, ask, mid, spread, and Greeks for calls and puts.
ATM straddle selection.

Strategy and delta hedging:

Long ATM call plus put straddle.
Long option positions entered at ask prices.
Daily option marking using synthetic implied volatility.
Dynamic underlying delta hedging.
Configurable delta rebalance threshold.
Integer or fractional share hedging.
Commission and slippage modelling.
Hedge trade logs, turnover, and transaction-cost tracking.

Backtesting and research:

Single-trade delta-hedged long-straddle backtest.
Non-overlapping rolling backtest.
Transaction-cost and delta-threshold sensitivity analysis.
Volatility-regime filtered backtest.
Volatility-regime threshold sensitivity analysis.
Profit and loss, drawdown, VaR, Expected Shortfall, win rate, and Sharpe-like
metrics.

================================================================
PROJECT STRUCTURE
================================================================

src/pricing/

black_scholes.py
implied_volatility.py

src/market_data/

underlying_data.py
realized_volatility.py
synthetic_option_chain.py

src/strategy/

long_straddle.py
delta_hedging.py

src/backtest/

long_straddle_backtest.py
rolling_backtest.py
sensitivity_analysis.py
volatility_filter.py
regime_sensitivity.py

scripts/

download_spy_data.py
run_pricing_demo.py
run_implied_volatility_demo.py
run_volatility_demo.py
generate_synthetic_option_chain.py
run_long_straddle_demo.py
run_delta_hedging_demo.py
run_single_trade_backtest.py
run_sensitivity_analysis.py
run_rolling_backtest.py
run_volatility_filtered_backtest.py
run_regime_sensitivity_analysis.py

tests/

Automated pytest test suite for pricing, data validation, volatility analytics,
option-chain construction, straddle positions, delta hedging, backtesting, and
sensitivity analysis.

data/

Raw and processed SPY price data and synthetic option-chain outputs.

outputs/

Backtest CSV files, JSON summaries, and research figures.

================================================================
INSTALLATION
================================================================

Create and activate a virtual environment.

python -m venv .venv
source .venv/bin/activate

Install project dependencies.

python -m pip install --upgrade pip
pip install -r requirements.txt

For tests, model-validation reports and coverage, install the dev extras.

pip install -r requirements-dev.txt

================================================================
RUN TESTS
================================================================

Run the complete automated test suite (the fast ``perf`` marker is excluded
by default because wall-clock benchmarks are machine-dependent).

python -m pytest -q -m "not perf"

Run a specific group.

python -m pytest -q -m "property or golden or model_validation"

Generate the model validation report.

PYTHONPATH=. python scripts/run_validation_report.py

This writes:

outputs/testing/validation_report.json
outputs/testing/validation_report.md

Record the C++/Python speedup baseline (opt-in, needs a compiled backend).

PYTHONPATH=. python scripts/run_benchmark_baseline.py

The test suite covers:

Black-Scholes pricing and Greeks.
Put-call parity.
Implied-volatility recovery and no-arbitrage bounds.
Price-data validation.
Realised-volatility calculations.
Synthetic option-chain construction.
Long-straddle construction and valuation.
Delta hedging, turnover, and transaction costs.
Single-trade and rolling backtests.
Transaction-cost sensitivity.
Volatility-regime filtering.
Volatility-regime threshold sensitivity.

================================================================
TESTS, MODEL VALIDATION AND CI
================================================================

The project uses a layered test setup: unit and integration tests across
pricing, volatility-surface, backtest, execution, portfolio, risk and
market-making modules; Hypothesis property tests for no-arbitrage bounds and
put-call parity; reproducibility tests for seeded Monte Carlo; golden
regression tests that re-execute key results on committed data; and a
deterministic model-validation suite (``src/validation/suite.py``) covering
convergence, degeneracy (Heston/Merton back to Black-Scholes), SVI
calibration quality and Kupiec/Christoffersen VaR backtesting.

CI (``.github/workflows/tests.yml``) runs two parallel jobs:

- ``lint``: ruff error-level checks plus mypy on the validation module.
- ``test``: the full pytest suite with coverage, C++ backend build and tests,
  and the model-validation report; reports are uploaded as artifacts.

Line coverage is measured against ``src`` and gated at 80% (the measured value
is around 86%). CI runs Python 3.12 only: the pinned NumPy 2.5 release line
requires Python >= 3.12, so a multi-version matrix is intentionally not
enabled. Performance benchmarks run on demand via ``-m perf`` locally or a
manual ``workflow_dispatch`` run in CI, because absolute timings are not stable
on shared runners; the C++/Python speedup floor test is kept conservative for
the same reason.

Scope: this is research-grade quality assurance for correctness and
reproducibility. It does not include production monitoring or alerting, chaos
or fault-injection engineering, security penetration testing, large-scale
load testing, or production deployment/rollback pipelines.

================================================================
REPRODUCE THE RESEARCH WORKFLOW
================================================================

Step 1. Download and validate SPY data.

PYTHONPATH=. python scripts/download_spy_data.py

Step 2. Calculate realised volatility.

PYTHONPATH=. python scripts/run_volatility_demo.py

This creates:

data/processed/spy_daily_with_realized_volatility.csv
outputs/figures/spy_realized_volatility.png

Step 3. Generate a synthetic option chain.

PYTHONPATH=. python scripts/generate_synthetic_option_chain.py

This creates:

data/processed/spy_synthetic_option_chain_latest.csv
outputs/figures/spy_synthetic_volatility_smile.png
outputs/figures/spy_synthetic_term_structure.png

Step 4. Run a single delta-hedged long straddle.

PYTHONPATH=. python scripts/run_single_trade_backtest.py

Step 5. Run delta-hedging threshold and transaction-cost sensitivity analysis.

PYTHONPATH=. python scripts/run_sensitivity_analysis.py

Step 6. Run the non-overlapping rolling backtest.

PYTHONPATH=. python scripts/run_rolling_backtest.py

Step 7. Compare baseline and volatility-filtered strategy results.

PYTHONPATH=. python scripts/run_volatility_filtered_backtest.py

Step 8. Run volatility-regime threshold sensitivity analysis.

PYTHONPATH=. python scripts/run_regime_sensitivity_analysis.py

================================================================
ILLUSTRATIVE RESEARCH RESULTS
================================================================

Unconditional rolling long straddle:

The non-overlapping rolling backtest used 30-day ATM long straddles, synthetic
implied volatility, daily delta hedging, 1 basis point underlying slippage, and
a 5-share delta rebalance threshold.

Number of trades: 33.
Total profit and loss: -8,038.86.
Return on illustrative initial capital: -8.04 percent.
Win rate: 24.24 percent.
Portfolio maximum drawdown: -8.08 percent.
Sharpe-like ratio: -1.74.
Total hedge costs: 533.11.

The unconditional strategy lost money under the stated assumptions. This is
consistent with the economics of systematically buying options. Realised path
volatility must exceed option premium, theta decay, bid-ask spread, and dynamic
hedging costs.

Volatility regime filter:

The regime filter only allows entry when the following condition is met.

20-day realised volatility divided by 252-day realised volatility is greater
than or equal to a specified threshold.

Threshold 0.90:
17 trades.
Total profit and loss: -2,371.96.
Maximum drawdown: -5.43 percent.
Sharpe-like ratio: -0.76.

Threshold 1.00:
13 trades.
Total profit and loss: -1,573.25.
Maximum drawdown: -4.27 percent.
Sharpe-like ratio: -0.50.

Threshold 1.05:
12 trades.
Total profit and loss: -769.71.
Maximum drawdown: -4.27 percent.
Sharpe-like ratio: -0.36.

Threshold 1.10:
11 trades.
Total profit and loss: 298.41.
Maximum drawdown: -3.21 percent.
Sharpe-like ratio: -0.04.

Threshold 1.20:
7 trades.
Total profit and loss: 1,285.96.
Maximum drawdown: -2.22 percent.
Sharpe-like ratio: 0.30.

Threshold 1.30:
6 trades.
Total profit and loss: 1,357.48.
Maximum drawdown: -2.22 percent.
Sharpe-like ratio: 0.32.

Higher thresholds reduced trade count, turnover, and drawdown in this sample.
However, the higher-threshold results are based on only 6 to 7 trades. They
must not be treated as evidence of an optimal parameter or tradable strategy.

Delta-hedging cost sensitivity:

For one selected 30-day SPY long-straddle window:

Delta threshold 0 shares and slippage 0.5 basis points:
Final profit and loss: 594.29.
Hedge trades: 23.
Hedge costs: 14.20.

Delta threshold 0 shares and slippage 10 basis points:
Final profit and loss: 136.69.
Hedge trades: 23.
Hedge costs: 243.00.

Delta threshold 10 shares and slippage 1 basis point:
Final profit and loss: 180.46.
Hedge trades: 10.
Hedge costs: 18.60.

Delta threshold 10 shares and slippage 10 basis points:
Final profit and loss: -126.73.
Hedge trades: 10.
Hedge costs: 172.19.

The analysis demonstrates the trade-off between tight delta neutrality and
higher turnover. Under high trading-cost assumptions, transaction costs can
materially reduce or reverse apparent gamma-scalping profits.

================================================================
LIMITATIONS
================================================================

SPY is the only underlying asset in the current study.

Underlying prices are real adjusted historical daily prices, but option quotes
and implied-volatility surfaces are synthetic.

The strategy is daily-frequency and does not model intraday hedging.

The model uses simplified assumptions for bid-ask spreads, slippage, commissions,
financing, and liquidity.

The rolling study uses fixed contract quantity rather than full capital-constrained
position sizing.

The strategy does not include real margin requirements, borrow costs, corporate
actions, early exercise, assignment risk, option volume, or open interest.

The analysis explores multiple parameters and can be affected by sample-selection
bias and data-mining bias.

Results are for education and research only. They should not be used for
investment decisions.

================================================================
DEVELOPMENT PRACTICES
================================================================

Modular Python architecture.
Data validation and explicit modelling assumptions.
Automated pytest test suite.
Reproducible scripts for data generation, backtesting, and figures.
GitHub Actions continuous integration.
Transparent reporting of negative and positive results.

================================================================
FUTURE EXTENSIONS
================================================================

Use licensed historical option bid and ask data.
Add rolling out-of-sample train, validation, and test analysis.
Support overlapping option positions and capital allocation.
Model option liquidity, volume, open interest, and more realistic execution.
Add event filters and multiple underlying ETFs.
Add profit and loss attribution across Delta, Gamma, Vega, Theta, and costs.
Generate a self-contained HTML report or Streamlit dashboard.

================================================================
DISCLAIMER
================================================================

This project is provided solely for educational and research purposes. Nothing in
this repository constitutes investment advice, financial advice, an offer to buy
or sell securities, or a recommendation to use any trading strategy.

## 数据与研究边界 (DATA AND RESEARCH BOUNDARIES)

本项目是一个用于期权定价、波动率策略、动态对冲与风险分析的研究型回测平台。

### 当前数据构成

- **标的行情：真实历史数据。** 使用公开数据源获取 SPY 的日频历史行情，
  包括开盘价、最高价、最低价、收盘价、复权收盘价与成交量。
- **期权报价：合成历史数据。** 历史期权价格、隐含波动率曲面和买卖价差并非
  来自可逐笔验证的真实历史期权链，而是根据标的价格、历史波动率、期限结构、
  波动率偏斜、微笑曲率和预设交易成本规则生成。
- **交易执行：模拟成交。** 策略成交、Delta 对冲、佣金、滑点和买卖价差均由
  回测规则模拟，不代表真实交易所或经纪商的实际成交结果。

### 结果解释

当前历史回测结果主要用于研究：不同波动率假设对定价的影响、Delta 动态对冲的
离散误差、对冲频率与交易成本之间的权衡、隐含波动率与实现波动率错配引起的
模型风险，以及策略与参数设定下的敏感性和稳健性。

当前结果不应解释为：真实历史期权市场中的可实现收益、已验证的实盘交易策略、
对未来收益的承诺，或投资建议。

### 可追溯性

每次通过 `scripts/run_backtest.py` 的运行都会生成独立输出目录，包含
`run_metadata.json`（run_id、数据/执行/评估边界、Git 提交、配置哈希、随机种子、
依赖版本）、`config_snapshot.yaml`、`research_boundary.json` 与 `manifest.json`；
结果表均带 `run_id` 列。每条回测 summary 也携带 `underlying_data_type`、
`option_data_type`、`execution_type` 等字段，结果文件无需阅读 README 即可识别
数据性质。

### 后续扩展

真实期权链快照模块（`src/market_data/real_option_chain.py`）用于市场快照分析、
报价质量过滤、隐含波动率反解、偏斜与期限结构可视化，以及后续的曲面校准与
样本外研究，用于比较合成市场假设与真实市场特征的差异。

## 随机/跳跃/局部波动率 (STOCHASTIC VOL MODELS)

`src/pricing/heston.py`（特征函数 + 全截断 MC）、`src/pricing/merton.py`
（欧式解析级数 + compound-Poisson MC）与 `src/pricing/local_vol.py`
（Dupire 局部方差）用于受控研究与模型对比。重要边界：单一真实期权快照不足以
可靠识别 Heston 等随机波动率参数，跨日校准实验基于带标记的合成曲面；这些模型
不用于策略回测，也不构成实盘结论。`src/pricing/calibration.py` 提供 BS 曲面拟合
与模型对比报告入口。

## 数据质量管道 (DATA QUALITY PIPELINE)

原始市场数据不会直接进入回测。`src/market_data/pipeline.py` 对每次输入依次执行：

1. 字段与类型检查（统一 schema，`OptionType` / `DataType` 使用枚举）；
2. 日期与数值标准化；
3. 标的 OHLC 逻辑检查与非正价格/负成交量检查；
4. 期权买卖报价、到期日、合约乘数检查；
5. 相对价差检查（警告级）；
6. 欧式期权无套利边界检查；
7. 重复记录识别与去除；
8. 无效记录隔离并输出具体原因（CSV）；
9. 数据质量报告（保留率、错误/警告统计）；
10. 输入/输出文件哈希与数据血缘记录（`data_lineage.json`）。

回测入口（`scripts/run_backtest.py` 与 `scripts/run_data_pipeline.py`）先生成
`outputs/<run_id>/market_data/` 下的验证后数据与全套报告。真实 SPY 期权为美式
合约，当前欧式边界用于合成链（欧式 Black-Scholes 框架）；接入真实美式期权时
需增加 `exercise_style` 处理与提前行权规则。

## 时间序列验证与前视偏差控制 (TIME-SERIES VALIDATION)

项目采用严格的信息时点约定：第 t 日的策略信号仅使用截至第 t-1 日收盘时可获得
的数据，并在第 t 日的模拟可执行报价上成交。`src/features/` 生成的所有波动率
特征在观测后滞后一个交易日，并附带 JSON sidecar 声明可用时间规则；审计模块会
拒绝"观测截止不早于信号日"的特征表。`src/backtest/timeline.py` 提供滞后一日的
波动率过滤信号与三区段切分。

模型和策略参数仅使用训练集与验证集选择（`src/evaluation/model_selection.py`
使用带小样本/回撤/换手惩罚的综合评分），独立测试集只用于最终评价并被
`evaluation/test_evaluation_log.json` 锁定——重复评估必须显式说明原因。
`scripts/run_strict_evaluation.py` 运行完整流程并输出：

- `parameter_selection.csv`、`selected_parameters.json`；
- `train/validation/test_metrics.json`（三个区间分别报告）；
- `trade_log.csv`（每笔交易带 `split_name`）与
  `boundary_crossing_trades.csv`（跨越 train/validation 边界被剔除的交易，
  保留审计痕迹）；
- `split_summary.csv` 与 `test_evaluation_record.json`。

## 账户、资本与风险约束 (ACCOUNT & RISK CONSTRAINTS)

`src/portfolio/` 提供组合级复式现金流逻辑：订单、成交（实际成交价口径，现金只扣
佣金；价差/滑点/冲击仅作归因字段）、头寸（加权平均成本、平仓与反转的已实现 PnL）、
不可变现金账本、账户权益与每日快照、期权到期现金结算，以及独立的 PnL 桥对账
（`TotalPnL ≈ Realised + Unrealised + Financing + BorrowFee + Settlement − Fees`）。

`src/financing/` 提供现金存/贷利息、股票借券费和简化保证金估算；`src/risk/`
计算组合 Greeks（per-contract 口径）并做交易前限额检查，支持对整张订单的拒绝
与二分搜索最大可执行数量。保证金模型属于研究性近似，不代表交易所、清算机构或
经纪商的实际规则。

`src/execution/` 提供统一执行引擎：盘口快照（orderbook 或 mid-only 模式）、
佣金/滑点/市场冲击定价（实际成交价口径，成本只作归因字段）、陈旧报价拒绝、
限价单不可成交取消、受流动性约束的部分成交，以及执行质量指标与成交明细 CSV。

## 市场做市研究与事件驱动 RL (MARKET MAKING RESEARCH)

任务十四的做市研究由三层模块构成，全部为**研究仿真，不代表实盘或真实期权
行情**：

- `src/market_making/greeks_book.py`：多合约 Greeks 感知报价。按净
  Delta/Gamma/Vega 在跨行权价与到期结构上聚合风险：Delta 采用 A–S 保留价
  偏移（以各腿自身 Delta 折算到期权权利金），Gamma/Vega 按各腿对净风险的
  贡献加宽价差，并按 0–7/7–30/30–90/90+ 交易日到期桶汇总风险。
- `src/backtest/account_market_making.py`：把上述报价策略接入账户引擎。
  每日事件中 taker 以可配置概率到价成交（多头以 ask 卖出、空头以 bid 买回，
  逐笔佣金与价差归因），随后按阈值 Δ 中性动态对冲；每笔期权成交做交易前
  限额检查，日末做日亏损/回撤熔断检查；覆盖融资、期权到期结算与 PnL 桥
  对账，输出 `snapshots.csv`、`fills.csv`、`breach_log.csv` 与 `summary.json`。
- 单资产事件驱动研究栈，与论文《Continuous-time RL for market making》对应：
  - `src/market_making/intensity_env.py`：§5.1 仿真框架——Poisson 到达 +
    MNL 路由 + 报价菜单 + 库存约束；含网格版仿真用于离散时间基准；
  - `src/market_making/dp.py`：§6.2 有限状态 ODE 基准（式 6.1 后向 Euler）
    与指数成交闭式交叉验证（式 6.4，可对拍 3.42025）；
  - `src/market_making/rl.py`：§5.2 Algorithm 1（连续时间 Linear-MC）与
    Algorithm 2（离散时间 AC 基准），§5.3 Linear-Pair Actor；
  - `src/market_making/study.py`：DP/Greedy/Uniform/RL 同订单流、同 seed、
    多 seed 对照，输出 `comparison.csv`、`aggregated.csv`、
    `episode_metrics.csv`、`comparison.png` 与 `summary.json`（含置信区间）。

入口脚本：`scripts/run_market_making_study.py`（论文 Experiment I 参数，
DP V*(0,0) 可与 1.90299 对拍）与 `scripts/run_account_market_making_demo.py`。

## 模块集成状态 (INTEGRATION STATUS)

| 模块 | 是否被主回测流程调用 | 状态 |
|---|---|---|
| `src/market_data/real_option_chain.py` | 否（独立快照分析） | 已测试，待接入统一数据层 |
| `src/market_data/pipeline.py` | 演示脚本调用 | 真实历史行情尚未转统一 schema 走管道 |
| `src/features/` `src/evaluation/` | strict runner 调用 | 已接入样本外评估 |
| `src/portfolio/` `src/risk/` `src/execution/` | **是（账户引擎路径）** | 驱动单腿 Call、ATM 跨式与滚动跨式回测，逐笔 PnL 桥对账 |
| `src/backtest/rolling_backtest.py` | legacy 对照 | 老 close-to-close 路径，保留用于结果对照，不再是推荐入口 |

推荐入口：`run_rolling_account_straddle_backtest`（账户引擎驱动，每笔交易都通过
PnL 桥对账）。真实 SPY 数据上 17 笔滚动跨式总 PnL −8038.77，与 legacy 路径的
−8038.86 基本一致（差异来自计费口径细节），验证迁移等价性成立。

## STATISTICAL VALIDATION AND P&L ATTRIBUTION

The rolling backtest summary now reports:

- `sharpe_like_ratio`: per-trade mean minus risk-free rate, divided by
  per-trade volatility, scaled by `sqrt(number_of_trades)` (kept for
  backward compatibility; this is not an annualized Sharpe);
- `annualized_sharpe_estimate`: trade-level Sharpe scaled by
  `sqrt(trades_per_year_assumed)` with
  `trades_per_year_assumed = 252 / entry_spacing_trading_days`;
- moving-block-bootstrap confidence intervals for mean trade return,
  `sharpe_like_ratio`, and the annualized estimate (`src/statistics.py`).

`src/backtest/pnl_attribution.py` decomposes each trading day's P&L into
Delta, Gamma, Vega, Theta, Rho and hedge-cost contributions using the
previous day's recorded exposures, and reports the unexplained residual.
See `scripts/run_upgrade_demo.py` for an end-to-end example.

## API 服务与部署 (API & DEPLOYMENT)

`src/api/` 提供 FastAPI 服务：`GET /health`、`POST /pricing/vanilla`、
`POST /pricing/surface`（C++ 后端可用时走批量内核）、`POST /runs`（后台线程
任务，可轮询 `GET /runs/{id}`）。持久化默认 SQLite（`outputs/app.db`），
Repository 抽象便于切换 Postgres。

本地启动：

```bash
uvicorn src.api.run_server:application --reload
```

Docker：`docker compose up --build`（api 在 8000，Streamlit 看板在 8501）。
看板逻辑在 `scripts/dashboard.py`，UI 需在本地/容器运行验证。
