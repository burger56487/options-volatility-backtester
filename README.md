OPTIONS VOLATILITY TRADING AND DYNAMIC HEDGING BACKTESTER

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

================================================================
RUN TESTS
================================================================

Run the complete automated test suite.

python -m pytest -q

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


head -30 README.md
tail -20 README.md

git add README.md
git commit -m "Add options backtester documentation"
git push

