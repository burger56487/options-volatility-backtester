"""ATM forward, discount factor and implied rate/dividend estimation.

For every expiry the put-call parity relation ``C - P = D * (F - K)`` is
fitted with a strike-weighted, centred linear regression:

- pairs are built from quality-good quotes (optional hard-violation screen);
- weights are proportional to the inverse mid-noise proxy
  ``1 / (call_spread^2 + put_spread^2)`` when bid/ask quotes exist;
- the regression is centred at the mean strike and refit after MAD-based
  outlier removal (at most two passes);
- results carry explicit validity flags, warnings and used-pair counts.

The regression route is deliberately complementary to the per-ATM pair
estimates in ``market_data.real_option_chain``: it reports one coherent
discount factor and forward per expiry with diagnostics, while the pair
route is less sensitive to wing noise but only inspects the near-ATM pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


MIN_PAIRS = 4
MAX_OUTLIER_PASSES = 2
OUTLIER_MAD_FACTOR = 3.0
MIN_ABS_SPREAD_NOISE = 0.05

MIN_DISCOUNT = 0.5
MAX_DISCOUNT = 1.05
MIN_IMPLIED_RATE = -0.05
MAX_IMPLIED_RATE = 0.20
MIN_IMPLIED_DIVIDEND = -0.05
MAX_IMPLIED_DIVIDEND = 0.20
ABSURD_RATE_BOUND = 1.0


@dataclass
class ForwardEstimate:
    expiry: pd.Timestamp
    time_to_expiry: float = float("nan")
    spot: float = float("nan")
    discount_factor: float = float("nan")
    forward: float = float("nan")
    implied_rate: float = float("nan")
    implied_dividend: float = float("nan")
    r_squared: float = float("nan")
    num_pairs_total: int = 0
    num_pairs_used: int = 0
    outliers_removed: int = 0
    residual_std: float = float("nan")
    strike_mean: float = float("nan")
    warnings: list[str] = field(default_factory=list)
    valid: bool = True

    def to_dict(self) -> dict:
        output = {
            key: value
            for key, value in self.__dict__.items()
            if key != "expiry"
        }
        output["expiry"] = (
            self.expiry.isoformat()
            if isinstance(self.expiry, pd.Timestamp)
            else str(self.expiry)
        )
        return output


def _inverse_noise_weight(
    call_spread: pd.Series,
    put_spread: pd.Series,
    epsilon: float = 1e-8,
) -> pd.Series:
    """Mid-noise proxy weight with an absolute noise floor.

    A 0.01-wide penny quote has tiny dollar noise but is not a reliable
    parity observation; the floor prevents those quotes from dominating the
    regression over wider but economically meaningful near-ATM quotes.
    """
    call_noise = np.maximum(call_spread, MIN_ABS_SPREAD_NOISE)
    put_noise = np.maximum(put_spread, MIN_ABS_SPREAD_NOISE)
    return 1.0 / (call_noise**2 + put_noise**2 + epsilon)


def build_call_put_pairs(
    df: pd.DataFrame,
    good_only: bool = True,
    exclude_hard_violation: bool = True,
) -> pd.DataFrame:
    """Pair call and put mids on (expiry, strike) with quality screens."""
    work = df.copy()
    if good_only and "quality" in work.columns:
        work = work[work["quality"] == "good"]
    if (
        exclude_hard_violation
        and "hard_violation" in work.columns
    ):
        work = work[~work["hard_violation"].fillna(False)]

    required = {
        "expiry",
        "strike",
        "option_type",
        "mid",
        "time_to_expiry",
        "spot",
    }
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if "bid" in work.columns and "ask" in work.columns:
        work["spread"] = work["ask"] - work["bid"]
    else:
        work["spread"] = float("nan")

    calls = work[work["option_type"] == "call"][
        [
            "expiry",
            "strike",
            "mid",
            "spread",
            "time_to_expiry",
            "spot",
        ]
    ].rename(
        columns={
            "mid": "call_mid",
            "spread": "call_spread",
        }
    )
    puts = work[work["option_type"] == "put"][
        ["expiry", "strike", "mid", "spread"]
    ].rename(
        columns={
            "mid": "put_mid",
            "spread": "put_spread",
        }
    )
    pairs = calls.merge(puts, on=["expiry", "strike"], how="inner")
    if pairs.empty:
        return pairs

    pairs["c_minus_p"] = pairs["call_mid"] - pairs["put_mid"]
    has_spreads = (
        pairs["call_spread"].notna() & pairs["put_spread"].notna()
    )
    pairs["weight"] = _inverse_noise_weight(
        pairs["call_spread"].fillna(0.0),
        pairs["put_spread"].fillna(0.0),
    )
    pairs.loc[~has_spreads, "weight"] = 1.0
    return pairs


def _mad_scale(residuals: np.ndarray) -> float:
    if residuals.size < 2:
        return 0.0
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    scale = 1.4826 * mad
    if scale > 0:
        return scale
    return float(np.std(residuals, ddof=1))


def _weighted_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float, float]:
    """Centred weighted least squares; returns slope, intercept_orig,
    intercept_centred, x_mean."""
    x_mean = float(np.mean(x))
    centred = x - x_mean
    coefficients = np.polyfit(centred, y, 1, w=weights)
    slope = float(coefficients[0])
    intercept_centred = float(coefficients[1])
    intercept_orig = intercept_centred - slope * x_mean
    return slope, intercept_orig, intercept_centred, x_mean


def _fit_expiry(
    pairs: pd.DataFrame,
) -> ForwardEstimate:
    expiry = pairs["expiry"].iloc[0]
    time_values = pairs["time_to_expiry"].astype(float).unique()
    spot_values = pairs["spot"].astype(float).unique()
    time_to_expiry = float(time_values[0]) if len(time_values) else float("nan")
    spot = float(spot_values[0]) if len(spot_values) else float("nan")

    warnings: list[str] = []
    if len(time_values) > 1:
        warnings.append(
            "expiry 内 time_to_expiry 不一致，取首个值"
        )
    if len(spot_values) > 1:
        warnings.append("expiry 内 spot 不一致，取首个值")

    result = ForwardEstimate(
        expiry=expiry,
        time_to_expiry=time_to_expiry,
        spot=spot,
        warnings=warnings,
    )

    finite = (
        pairs["strike"].notna()
        & pairs["c_minus_p"].notna()
        & np.isfinite(pairs["strike"].astype(float))
        & np.isfinite(pairs["c_minus_p"].astype(float))
    )
    clean = pairs[finite].copy()
    total = int(len(clean))
    result.num_pairs_total = total
    if total < MIN_PAIRS:
        result.valid = False
        result.warnings.append(
            f"配对点不足（{total}<{MIN_PAIRS}），无法回归"
        )
        return result

    x = clean["strike"].to_numpy(dtype=float)
    y = clean["c_minus_p"].to_numpy(dtype=float)
    weights = clean["weight"].to_numpy(dtype=float)

    keep = np.ones(x.size, dtype=bool)
    slope = 0.0
    intercept_orig = 0.0
    x_mean = float(np.mean(x))
    residual_std = float("nan")
    for _ in range(MAX_OUTLIER_PASSES + 1):
        slope, intercept_orig, _, x_mean = _weighted_linear_fit(
            x[keep],
            y[keep],
            weights[keep],
        )
        residuals = y - (slope * x + intercept_orig)
        scale = _mad_scale(residuals[keep])
        if not np.isfinite(scale) or scale <= 0:
            break
        candidate = np.abs(residuals) <= OUTLIER_MAD_FACTOR * scale
        if candidate.sum() < MIN_PAIRS:
            break
        if np.array_equal(candidate, keep):
            break
        keep = candidate

    used = int(keep.sum())
    result.num_pairs_used = used
    result.outliers_removed = total - used
    result.strike_mean = x_mean
    if used < MIN_PAIRS:
        result.valid = False
        result.warnings.append(
            f"剔除离群点后配对不足（{used}<{MIN_PAIRS}）"
        )
        return result

    slope, intercept_orig, _, x_mean = _weighted_linear_fit(
        x[keep],
        y[keep],
        weights[keep],
    )
    result.strike_mean = x_mean

    y_used = y[keep]
    y_pred = slope * x[keep] + intercept_orig
    residuals_used = y_used - y_pred
    result.residual_std = float(np.std(residuals_used, ddof=1))
    ss_res = float(np.sum(residuals_used**2))
    ss_tot = float(np.sum((y_used - np.mean(y_used)) ** 2))
    if ss_tot > 0:
        result.r_squared = 1.0 - ss_res / ss_tot
    else:
        result.r_squared = float("nan")
        result.warnings.append("因变量恒等，R² 无定义")

    discount_factor = -slope
    result.discount_factor = discount_factor
    if not np.isfinite(discount_factor) or discount_factor <= 0:
        result.valid = False
        result.warnings.append(
            f"斜率非负/贴现因子非正（D={discount_factor:.6f}），"
            "平价回归方向错误"
        )
        return result
    if not MIN_DISCOUNT <= discount_factor <= MAX_DISCOUNT:
        result.valid = False
        result.warnings.append(
            f"贴现因子超出合理范围（{discount_factor:.4f}）"
        )

    forward = intercept_orig / discount_factor
    result.forward = forward
    if not np.isfinite(forward) or forward <= 0:
        result.valid = False
        result.warnings.append(
            f"远期非正或非有限（F={forward:.4f}）"
        )
        return result

    if not np.isfinite(time_to_expiry) or time_to_expiry <= 0:
        result.valid = False
        result.warnings.append("time_to_expiry 必须为正")
        return result
    if not np.isfinite(spot) or spot <= 0:
        result.valid = False
        result.warnings.append("spot 必须为正")
        return result

    implied_rate = -np.log(discount_factor) / time_to_expiry
    implied_dividend = (
        implied_rate
        - np.log(forward / spot) / time_to_expiry
    )
    result.implied_rate = float(implied_rate)
    result.implied_dividend = float(implied_dividend)

    if not (
        MIN_IMPLIED_RATE <= implied_rate <= MAX_IMPLIED_RATE
    ):
        result.warnings.append(
            f"隐含利率超出合理范围（{implied_rate:.4f}）"
        )
    if not (
        MIN_IMPLIED_DIVIDEND
        <= implied_dividend
        <= MAX_IMPLIED_DIVIDEND
    ):
        result.warnings.append(
            f"隐含股息超出合理范围（{implied_dividend:.4f}）"
        )
    if (
        abs(implied_rate) > ABSURD_RATE_BOUND
        or abs(implied_dividend) > ABSURD_RATE_BOUND
    ):
        result.valid = False
        result.warnings.append("利率/股息绝对值异常，判定无效")

    if not np.isnan(result.r_squared) and result.r_squared < 0.99:
        result.warnings.append(
            f"R²偏低（{result.r_squared:.4f}），平价拟合不佳"
        )
    return result


def estimate_forward_single_expiry(pairs: pd.DataFrame) -> ForwardEstimate:
    """Fit one expiry from an already-built pairs frame."""
    return _fit_expiry(pairs)


def estimate_all_forwards(
    df: pd.DataFrame,
    good_only: bool = True,
    exclude_hard_violation: bool = True,
) -> pd.DataFrame:
    """Estimate forward/discount/rate/dividend for every expiry."""
    pairs = build_call_put_pairs(
        df,
        good_only=good_only,
        exclude_hard_violation=exclude_hard_violation,
    )
    if pairs.empty:
        return pd.DataFrame()
    results = []
    for _, group in pairs.groupby("expiry"):
        results.append(_fit_expiry(group))
    frame = pd.DataFrame([result.to_dict() for result in results])
    if "expiry" in frame.columns:
        frame["expiry"] = pd.to_datetime(frame["expiry"])
    if "time_to_expiry" in frame.columns:
        frame = frame.sort_values("time_to_expiry").reset_index(drop=True)
    return frame
