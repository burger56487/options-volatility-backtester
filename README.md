# Options Volatility Trading and Dynamic Hedging Backtester

A Python research project for studying delta-hedged long straddle strategies
using real SPY historical prices and a transparent synthetic option-pricing
framework.

Important: This is an educational derivatives-research project. It does not
provide investment advice or evidence that a strategy is profitable.

## Project Structure

src/
- pricing/black_scholes.py
- pricing/implied_volatility.py
- market_data/underlying_data.py
- market_data/realized_volatility.py
- market_data/synthetic_option_chain.py
- strategy/long_straddle.py
- strategy/delta_hedging.py
- backtest/long_straddle_backtest.py
- backtest/rolling_backtest.py
- backtest/sensitivity_analysis.py
- backtest/volatility_filter.py
- backtest/regime_sensitivity.py

Other folders: scripts, tests, data, outputs

## Installation

Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

## Run Tests

Run the complete automated test suite:

    python -m pytest -q

The test suite covers pricing, Greeks, put-call parity, implied-volatility
recovery, market-data validation, realised volatility, synthetic option-chain
construction, long-straddle valuation, delta hedging, transaction costs,
single-trade backtests, rolling backtests, and regime sensitivity analysis.

## Reproduce the Research Workflow

### 1. Download and Validate SPY Data

    PYTHONPATH=. python scripts/download_spy_data.py

### 2. Calculate Realised Volatility

    PYTHONPATH=. python scripts/run_volatility_demo.py

### 3. Generate a Synthetic Option Chain

    PYTHONPATH=. python scripts/generate_synthetic_option_chain.py

### 4. Run a Single Delta-Hedged Long Straddle

    PYTHONPATH=. python scripts/run_single_trade_backtest.py

### 5. Run Delta-Hedging Sensitivity Analysis

    PYTHONPATH=. python scripts/run_sensitivity_analysis.py

### 6. Run the Non-Overlapping Rolling Backtest

    PYTHONPATH=. python scripts/run_rolling_backtest.py

### 7. Compare Baseline and Volatility-Filtered Strategies

    PYTHONPATH=. python scripts/run_volatility_filtered_backtest.py

### 8. Run Volatility-Regime Threshold Sensitivity

    PYTHONPATH=. python scripts/run_regime_sensitivity_analysis.py

## Illustrative Research Results

### Unconditional Rolling Long Straddle

The non-overlapping rolling backtest used 30-day ATM long straddles, synthetic
implied volatility, daily delta hedging, 1 bp underlying slippage, and a
5-share delta rebalance threshold.

- Number of trades: 33
- Total P&L: -8,038.86
- Return on illustrative initial capital: -8.04 percent
- Win rate: 24.24 percent
- Portfolio maximum drawdown: -8.08 percent
- Sharpe-like ratio: -1.74
- Total hedge costs: 533.11

The unconditional strategy lost money under the stated assumptions. This is
consistent with the economics of systematically buying options: realised path
volatility must overcome option premium, theta decay, bid-ask spread, and
dynamic hedging costs.

### Volatility-Regime Threshold Sensitivity

The regime filter only allows entry when 20-day realised volatility divided by
252-day realised volatility is greater than or equal to a threshold.

- Threshold 0.90: 17 trades, total P&L -2,371.96, drawdown -5.43 percent, Sharpe-like -0.76
- Threshold 1.00: 13 trades, total P&L -1,573.25, drawdown -4.27 percent, Sharpe-like -0.50
- Threshold 1.05: 12 trades, total P&L -769.71, drawdown -4.27 percent, Sharpe-like -0.36
- Threshold 1.10: 11 trades, total P&L 298.41, drawdown -3.21 percent, Sharpe-like -0.04
- Threshold 1.20: 7 trades, total P&L 1,285.96, drawdown -2.22 percent, Sharpe-like 0.30
- Threshold 1.30: 6 trades, total P&L 1,357.48, drawdown -2.22 percent, Sharpe-like 0.32

Higher thresholds reduced trade count, turnover, and drawdown in this sample.
However, higher-threshold results are based on only 6 to 7 trades and must not
be treated as evidence of an optimal parameter or tradable strategy.

### Delta-Hedging Cost Sensitivity

For one selected 30-day SPY long-straddle window:

- Delta threshold 0 shares, slippage 0.5 bps: final P&L 594.29, 23 hedge trades, costs 14.20
- Delta threshold 0 shares, slippage 10 bps: final P&L 136.69, 23 hedge trades, costs 243.00
- Delta threshold 10 shares, slippage 1 bp: final P&L 180.46, 10 hedge trades, costs 18.60
- Delta threshold 10 shares, slippage 10 bps: final P&L -126.73, 10 hedge trades, costs 172.19

The analysis demonstrates the trade-off between tighter delta neutrality and
higher turnover. In high-cost scenarios, transaction costs can materially
reduce or reverse apparent gamma-scalping profits.

## Limitations

- SPY is the only underlying asset in the current study.
- Underlying prices are real adjusted historical daily prices, but option quotes and implied-volatility surfaces are synthetic.
- The strategy is daily-frequency and does not model intraday hedging.
- The model uses simplified assumptions for bid-ask spreads, slippage, commissions, financing, and liquidity.
- The rolling study uses fixed contract quantity rather than full capital-constrained position sizing.
- The strategy does not include real margin requirements, borrow costs, corporate actions, early exercise, assignment risk, option volume, or open interest.
- The analysis includes multiple parameter explorations and may be affected by sample-selection and data-mining bias.
- Results are educational and should not be used for investment decisions.

## Development Practices

- Modular Python architecture
- Data validation and explicit modelling assumptions
- Automated pytest test suite
- Reproducible scripts for data generation, backtesting, and figures
- GitHub Actions continuous integration
- Transparent reporting of negative and positive results

## Future Extensions

- Use licensed historical option bid-ask data
- Add rolling out-of-sample train, validation, and test analysis
- Support overlapping option positions and capital allocation
- Model option liquidity, volume, open interest, and more realistic execution
- Add event filters and multiple underlying ETFs
- Add P&L attribution across Delta, Gamma, Vega, Theta, and transaction costs
- Generate a self-contained HTML report or Streamlit dashboard

## Disclaimer

This project is provided solely for educational and research purposes. Nothing
in this repository constitutes investment advice, financial advice, an offer to
buy or sell securities, or a recommendation to use any trading strategy.
