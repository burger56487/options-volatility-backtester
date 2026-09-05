"""Tests for the end-to-end analysis pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.pricing.black_scholes import option_price
from src.report.pipeline import (
    AnalysisResult,
    build_report,
    json_default,
    run_full_analysis,
    sanitize_for_json,
)


def test_json_serialization_types():
    payload = {
        "a": np.float64(1.5),
        "b": np.int64(3),
        "c": pd.Timestamp("2026-01-01"),
        "d": np.float64(np.nan),
        "e": np.array([1, 2, 3]),
    }
    loaded = json.loads(
        json.dumps(
            sanitize_for_json(payload),
            default=json_default,
        )
    )
    assert loaded["a"] == 1.5
    assert loaded["d"] is None
    assert loaded["e"] == [1, 2, 3]
    assert sanitize_for_json({"x": float("nan")}) == {"x": None}


def test_build_report_partial_and_empty():
    empty = AnalysisResult(ticker="TEST")
    assert build_report(empty)["ticker"] == "TEST"

    partial = AnalysisResult(ticker="TEST", spot=100.0)
    partial.chain_full = pd.DataFrame(
        {"quality": ["good", "good", "invalid"]}
    )
    report = build_report(partial)
    assert report["spot"] == 100.0
    assert report["data_quality"]["total"] == 3
    assert report["data_quality"]["by_quality"]["good"] == 2


def _synthetic_quotes():
    rows = []
    for t, expiry in ((0.25, "2026-01-01"), (0.5, "2026-04-01")):
        for strike in np.arange(85.0, 116.0, 5.0):
            for option_type, is_call in (
                ("call", True),
                ("put", False),
            ):
                mid = option_price(
                    spot=100.0,
                    strike=strike,
                    time_to_expiry=t,
                    risk_free_rate=0.04,
                    volatility=0.25,
                    option_type=option_type,
                    dividend_yield=0.012,
                )
                rows.append(
                    {
                        "expiry": pd.Timestamp(expiry),
                        "strike": strike,
                        "option_type": option_type,
                        "mid": mid,
                        "bid": mid - 0.005,
                        "ask": mid + 0.005,
                        "spot": 100.0,
                        "time_to_expiry": t,
                        "quality": "good",
                        "volume": 10.0,
                        "open_interest": 100.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_end_to_end_pipeline_synthetic(tmp_path):
    quotes = _synthetic_quotes()
    result = run_full_analysis(
        quotes,
        tmp_path,
        ticker="SYN",
    )
    assert result.spot > 0
    assert (tmp_path / "report.json").exists()
    assert result.forwards is not None and not result.forwards.empty
    assert result.chain_iv is not None and not result.chain_iv.empty
    assert result.skew is not None and not result.skew.empty
    assert result.term_structure is not None
    assert len(result.svi_results) == 2
    assert result.chain_greeks is not None
    assert result.liquidity is not None
    assert (tmp_path / "figures" / "gamma_heatmap.png").exists()
