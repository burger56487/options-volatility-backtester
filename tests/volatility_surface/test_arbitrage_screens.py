"""Butterfly and calendar screen tests."""

from __future__ import annotations

import pandas as pd

from src.volatility_surface.arbitrage import check_butterfly_calls


def _quote_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_equal_spacing_butterfly_matches_half_weighted_rule():
    quotes = _quote_frame(
        [
            {
                "expiry": "2026-09-08",
                "option_type": "call",
                "strike": 100.0,
                "mid": 2.0,
            },
            {
                "expiry": "2026-09-08",
                "option_type": "call",
                "strike": 101.0,
                "mid": 1.5,
            },
            {
                "expiry": "2026-09-08",
                "option_type": "call",
                "strike": 102.0,
                "mid": 1.05,
            },
        ]
    )
    violations = check_butterfly_calls(
        quotes,
        risk_free_rate=0.04,
        tolerance=1e-8,
    )
    # 0.5*(2.0 + 1.05) - 1.5 = 0.025 > 0 -> no violation.
    assert violations.empty


def test_unequal_spacing_uses_weighted_convexity():
    quotes = _quote_frame(
        [
            {
                "expiry": "2026-09-08",
                "option_type": "call",
                "strike": 100.0,
                "mid": 1.0,
            },
            {
                "expiry": "2026-09-08",
                "option_type": "call",
                "strike": 101.0,
                "mid": 0.8,
            },
            {
                "expiry": "2026-09-08",
                "option_type": "call",
                "strike": 103.0,
                "mid": 0.5,
            },
        ]
    )
    violations = check_butterfly_calls(
        quotes,
        risk_free_rate=0.04,
        tolerance=1e-8,
    )
    # Weighted: (2/3)*1.0 + (1/3)*0.5 - 0.8 = 0.033 > 0 -> no violation.
    # Unweighted average would have been 0.75 - 0.8 = -0.05 (false positive).
    assert violations.empty


def test_true_unequal_spacing_violation_is_flagged():
    quotes = _quote_frame(
        [
            {
                "expiry": "2026-09-08",
                "option_type": "call",
                "strike": 100.0,
                "mid": 1.0,
            },
            {
                "expiry": "2026-09-08",
                "option_type": "call",
                "strike": 101.0,
                "mid": 0.9,
            },
            {
                "expiry": "2026-09-08",
                "option_type": "call",
                "strike": 103.0,
                "mid": 0.5,
            },
        ]
    )
    violations = check_butterfly_calls(
        quotes,
        risk_free_rate=0.04,
        tolerance=1e-8,
    )
    assert len(violations) == 1
    assert violations.iloc[0]["strike"] == 101.0
