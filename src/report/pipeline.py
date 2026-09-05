"""End-to-end offline option-chain analysis pipeline.

Unlike the live-fetch plan draft, this repository's canonical inputs are the
committed real snapshots (a graded quote frame).  The pipeline runs stages
1-8 in memory on that frame, uses a lightweight liquidity assessment as
stage 9, and produces a structured report plus figures.

Data-flow discipline (the core review point of the draft): the *full* graded
snapshot feeds liquidity; the *quality-good* chain with implied vols feeds
skew/Greeks; every stage degrades gracefully and records warnings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.arbitrage.checks import run_all_checks
from src.greeks.chain_greeks import compute_chain_greeks
from src.implied_vol.forward import estimate_all_forwards
from src.implied_vol.skew import compute_skew_metrics
from src.implied_vol.solver import solve_chain_iv
from src.implied_vol.term_structure import compute_term_structure
from src.liquidity.analysis import assess_liquidity_state
from src.volatility_surface.svi_analysis import (
    calibrate_svi_curve,
    check_calendar_arbitrage,
)


logger = logging.getLogger(__name__)


def json_default(obj):
    """JSON encoder for numpy/Timestamp/NaN objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def sanitize_for_json(obj):
    """Recursively replace NaN values with None for strict JSON."""
    if isinstance(obj, dict):
        return {
            key: sanitize_for_json(value)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(value) for value in obj]
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


@dataclass
class AnalysisResult:
    ticker: str
    spot: float = float("nan")
    chain_full: pd.DataFrame | None = None
    forwards: pd.DataFrame | None = None
    arbitrage: object | None = None
    chain_iv: pd.DataFrame | None = None
    skew: pd.DataFrame | None = None
    curves: dict | None = None
    term_structure: object | None = None
    svi_results: list = field(default_factory=list)
    calendar_violations: int = 0
    chain_greeks: pd.DataFrame | None = None
    liquidity: object | None = None
    warnings: list[str] = field(default_factory=list)


def _save(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_full_analysis(
    quotes: pd.DataFrame,
    output_dir: str | Path,
    *,
    ticker: str = "SPY",
    forwards: pd.DataFrame | None = None,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.012,
) -> AnalysisResult:
    """Run stages 1-8 (plus liquidity) on a graded quote snapshot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = AnalysisResult(ticker=ticker)

    # ---- 1. Input / cleaning state ----
    logger.info("stage 1: quote frame check")
    if quotes is None or quotes.empty:
        result.warnings.append("清洗后数据为空")
        return result
    chain_full = quotes.copy()
    if "quality" not in chain_full.columns:
        chain_full["quality"] = "good"
        result.warnings.append(
            "输入缺少 quality 列，全部按 good 处理"
        )
    if "mid" not in chain_full.columns and {
        "bid",
        "ask",
    }.issubset(chain_full.columns):
        chain_full["mid"] = 0.5 * (
            chain_full["bid"] + chain_full["ask"]
        )
    result.chain_full = chain_full
    result.spot = float(
        pd.to_numeric(chain_full["spot"], errors="coerce").iloc[0]
    )
    _save(chain_full, output_path / "01_chain_full.csv")

    # ---- 2. Forwards ----
    try:
        logger.info("stage 2: implied forwards")
        if forwards is None or forwards.empty:
            forwards = estimate_all_forwards(chain_full)
        result.forwards = forwards
        _save(forwards, output_path / "02_forwards.csv")
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"远期估计失败: {exc}")
        logger.exception("forward stage failed")
        return result
    if forwards.empty or not forwards["valid"].astype(bool).any():
        result.warnings.append("无有效远期，后续分析受限")
        return result

    # ---- 3. Arbitrage ----
    try:
        logger.info("stage 3: no-arbitrage checks")
        result.arbitrage = run_all_checks(chain_full, forwards)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"无套利检查失败: {exc}")

    # ---- 4. Implied volatility ----
    try:
        logger.info("stage 4: implied volatility")
        chain_iv = solve_chain_iv(
            chain_full,
            forwards,
            good_only=True,
        )
        result.chain_iv = chain_iv
        _save(chain_iv, output_path / "03_implied_vol.csv")
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"隐含波动率反解失败: {exc}")
        logger.exception("IV stage failed")
        return result

    # ---- 5. Skew ----
    skew = pd.DataFrame()
    curves: dict = {}
    try:
        logger.info("stage 5: skew")
        skew, curves = compute_skew_metrics(chain_iv, forwards)
        result.skew = skew
        result.curves = curves
        _save(skew, output_path / "04_skew_metrics.csv")
        if curves:
            pd.concat(
                [
                    curve.assign(expiry=expiry)
                    for expiry, curve in curves.items()
                ],
                ignore_index=True,
            ).to_csv(output_path / "05_skew_curves.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"偏斜分析失败: {exc}")

    # ---- 6. Term structure ----
    try:
        logger.info("stage 6: term structure")
        if not skew.empty:
            term = compute_term_structure(skew)
            result.term_structure = term
            _save(term.curve, output_path / "06_term_structure.csv")
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"期限结构失败: {exc}")

    # ---- 7. SVI + calendar ----
    try:
        logger.info("stage 7: SVI calibration")
        svi_results = []
        for expiry, curve in curves.items():
            if curve.empty or "time_to_expiry" not in curve.columns:
                continue
            time_to_expiry = float(
                curve["time_to_expiry"].dropna().iloc[0]
            )
            svi_results.append(
                calibrate_svi_curve(
                    curve,
                    expiry,
                    time_to_expiry,
                )
            )
        result.svi_results = svi_results
        result.calendar_violations, _ = check_calendar_arbitrage(
            svi_results
        )
        if svi_results:
            pd.DataFrame(
                [
                    {
                        "expiry": item.expiry,
                        "time_to_expiry": item.time_to_expiry,
                        "rmse_vol": item.rmse_vol,
                        "butterfly_violations": (
                            item.butterfly_violations
                        ),
                        "valid": item.valid,
                        "params": item.params.tolist()
                        if item.converged
                        else None,
                    }
                    for item in svi_results
                ]
            ).to_csv(output_path / "07_svi_results.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"SVI 校准失败: {exc}")

    # ---- 8. Greeks ----
    try:
        logger.info("stage 8: chain Greeks")
        if curves:
            curve_frame = pd.concat(
                list(curves.values()),
                ignore_index=True,
            )
            greeks = compute_chain_greeks(
                curve_frame,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                good_only=False,
            )
            result.chain_greeks = greeks
            _save(greeks, output_path / "08_greeks.csv")
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"Greeks 计算失败: {exc}")

    # ---- 9. Liquidity ----
    try:
        logger.info("stage 9: liquidity")
        result.liquidity = assess_liquidity_state(chain_full)
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"流动性分析失败: {exc}")

    # ---- 10. Report & figures ----
    report = build_report(result)
    (output_path / "report.json").write_text(
        json.dumps(
            sanitize_for_json(report),
            indent=2,
            default=json_default,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    try:
        from src.report.figures import generate_all_figures

        generate_all_figures(result, output_path / "figures")
    except Exception as exc:  # noqa: BLE001
        result.warnings.append(f"图表生成失败: {exc}")

    logger.info("analysis complete; warnings=%d", len(result.warnings))
    return result


def build_report(result: AnalysisResult) -> dict:
    """Structured JSON report with partial-result tolerance."""
    report = {
        "ticker": result.ticker,
        "spot": result.spot,
        "warnings": result.warnings,
    }
    if result.chain_full is not None:
        quality_series = result.chain_full["quality"].astype(str)
        quality_counts = {
            grade: int((quality_series == grade).sum())
            for grade in [
                "good",
                "wide_spread",
                "low_liquidity",
                "out_of_range",
                "invalid",
            ]
        }
        report["data_quality"] = {
            "total": int(len(result.chain_full)),
            "by_quality": quality_counts,
        }
    if result.arbitrage is not None:
        arb = result.arbitrage
        report["arbitrage"] = {
            "bound_violations": arb.bound_violations,
            "parity_violations": arb.parity_violations,
            "monotonicity_violations": arb.monotonicity_violations,
            "butterfly_violations": arb.butterfly_violations,
            "calendar_violations": arb.calendar_violations,
            "total_checked": arb.total_checked,
        }
    if result.skew is not None and not result.skew.empty:
        valid_skew = result.skew[result.skew["valid"].astype(bool)]
        if not valid_skew.empty:
            front = valid_skew.iloc[0]
            report["skew_front_month"] = {
                "atm_vol": front.get("atm_vol"),
                "rr_25": front.get("rr_25"),
                "bf_25": front.get("bf_25"),
            }
    if result.term_structure is not None:
        term = result.term_structure
        report["term_structure"] = {
            "shape": term.shape,
            "front_shape": term.front_shape,
            "back_shape": term.back_shape,
            "slope": term.slope,
            "annualized_slope": term.annualized_slope,
            "calendar_violations": term.calendar_violations,
        }
    if result.svi_results:
        valid_svi = [item for item in result.svi_results if item.valid]
        rmses = [
            item.rmse_vol
            for item in valid_svi
            if np.isfinite(item.rmse_vol)
        ]
        report["svi"] = {
            "num_calibrated": int(len(result.svi_results)),
            "num_valid": int(len(valid_svi)),
            "mean_rmse_vol": (
                float(np.mean(rmses)) if rmses else None
            ),
            "cross_expiry_calendar_violations": (
                result.calendar_violations
            ),
        }
    if result.chain_greeks is not None and "gamma" in (
        result.chain_greeks.columns
    ):
        report["greeks"] = {
            "rows": int(len(result.chain_greeks)),
            "max_gamma": float(
                result.chain_greeks["gamma"].abs().max()
            ),
            "max_vega": float(
                result.chain_greeks["vega"].abs().max()
            ),
        }
    if result.liquidity is not None:
        liquidity = result.liquidity
        report["liquidity"] = {
            "state": liquidity.overall_state,
            "median_rel_spread": liquidity.median_rel_spread,
            "pct_reliable": liquidity.pct_reliable,
            "pct_good": liquidity.pct_good,
        }
    return report
