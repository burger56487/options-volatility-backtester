import numpy as np
import pandas as pd

from src.performance.benchmark import benchmark_metrics
from src.performance.bootstrap import sharpe_ci
from src.performance.hypothesis_tests import (
    paired_daily_difference_p_value,
)
from src.performance.multiple_testing import (
    bonferroni_threshold,
    max_t_p_value,
)
from src.performance.ratios import annualised_sharpe
from src.performance.reporting import compute_performance_report
from src.performance.tail_risk import historical_var_cvar


def _equity():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, size=300)
    equity = 100_000.0 * np.cumprod(1.0 + returns)
    index = pd.bdate_range("2024-01-01", periods=300)
    return pd.Series(equity, index=index)


def test_positive_drift_yields_positive_sharpe():
    returns = pd.Series(np.full(100, 0.001))
    assert annualised_sharpe(returns) > 0


def test_tail_risk_requires_minimum_sample():
    small = pd.Series(np.full(5, -0.01))
    result = historical_var_cvar(small)
    assert result["insufficient_sample"] is True


def test_sharpe_ci_contains_point_estimate():
    equity = _equity()
    returns = equity.pct_change().dropna()
    ci = sharpe_ci(returns, n_samples=500, seed=1)
    point = annualised_sharpe(returns)
    assert ci["annualized_sharpe_ci_low"] <= point
    assert point <= ci["annualized_sharpe_ci_high"]


def test_sharpe_ci_reports_actual_block_size():
    import math

    import pandas as pd

    returns = pd.Series(range(1, 31))
    ci = sharpe_ci(returns, n_samples=50, seed=1)
    expected = max(1, int(math.ceil(30 ** (1.0 / 3.0))))
    assert ci["block_size"] == expected


def test_identical_strategies_not_significantly_different():
    returns = pd.Series(np.random.default_rng(3).normal(0, 0.01, 200))
    result = paired_daily_difference_p_value(returns, returns.copy())
    assert result["insufficient_sample"] is False
    assert result["p_value"] == 1.0


def test_multiple_testing_helpers():
    assert bonferroni_threshold(0.05, 5) == 0.01
    samples = np.random.default_rng(1).normal(0, 1, size=(5, 1000))
    p = max_t_p_value(samples, observed_max=4.0)
    assert 0.0 <= p <= 1.0


def test_performance_report_and_benchmark():
    equity = _equity()
    report = compute_performance_report(equity)
    assert "annualized_sharpe" in report
    assert "max_drawdown" in report
    bench = benchmark_metrics(
        equity.pct_change().dropna(),
        equity.pct_change().dropna(),
    )
    assert abs(bench["beta"] - 1.0) < 1e-9
