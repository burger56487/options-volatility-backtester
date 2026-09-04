"""Real market option-chain snapshot pipeline.

Downloads current option chains for a listed ETF from public quote sources
(yfinance first, Cboe delayed quotes as a fallback), normalises the raw
fields, applies quote-quality filters, runs European no-arbitrage soft checks
and solves Black-Scholes implied volatility on mid quotes.

SPY options are American style. Early-exercise premiums mean the European
no-arbitrage checks are soft diagnostics and can legitimately fail on deep
in-the-money puts; they are reported, not silently dropped.
"""

from __future__ import annotations

import math
import re
from datetime import date

import pandas as pd

from src.pricing.black_scholes import OptionType
from src.pricing.implied_volatility import implied_volatility


def normalise_option_frame(
    raw: pd.DataFrame,
    expiry: str,
    snapshot_date: date,
    spot: float,
    option_type: OptionType,
) -> pd.DataFrame:
    """Normalise one yfinance call/put frame into the standard quote schema."""
    frame = raw.copy()
    frame["expiry"] = pd.to_datetime(expiry)
    frame["snapshot_date"] = pd.Timestamp(snapshot_date)
    frame["option_type"] = option_type
    frame["spot"] = float(spot)
    frame["last"] = pd.to_numeric(frame.get("lastPrice"), errors="coerce")
    frame["volume"] = pd.to_numeric(frame.get("volume"), errors="coerce")
    frame["open_interest"] = pd.to_numeric(
        frame.get("openInterest"),
        errors="coerce",
    )
    frame["bid"] = pd.to_numeric(frame.get("bid"), errors="coerce")
    frame["ask"] = pd.to_numeric(frame.get("ask"), errors="coerce")
    frame["strike"] = pd.to_numeric(frame.get("strike"), errors="coerce")
    frame["multiplier"] = 100
    return frame[
        [
            "snapshot_date",
            "expiry",
            "option_type",
            "strike",
            "bid",
            "ask",
            "last",
            "volume",
            "open_interest",
            "spot",
            "multiplier",
        ]
    ]


def fetch_option_chain_snapshot(
    symbol: str = "SPY",
    expiries: list[str] | None = None,
    max_expiries: int = 6,
    source: str = "auto",
) -> tuple[pd.DataFrame, float]:
    """Download current option-chain quotes for the symbol."""
    if source not in ("auto", "yfinance", "cboe"):
        raise ValueError(
            "source must be one of 'auto', 'yfinance' or 'cboe'."
        )

    last_error: Exception | None = None
    if source in ("auto", "yfinance"):
        try:
            return _fetch_yfinance_option_chain(
                symbol=symbol,
                expiries=expiries,
                max_expiries=max_expiries,
            )
        except Exception as exc:  # noqa: BLE001 - try the fallback source
            last_error = exc
            if source == "yfinance":
                raise

    try:
        return _fetch_cboe_option_chain(
            symbol=symbol,
            max_expiries=max_expiries,
        )
    except Exception as cboe_error:
        if last_error is not None:
            raise RuntimeError(
                "Both yfinance and Cboe fetches failed. "
                f"yfinance: {last_error}; cboe: {cboe_error}"
            ) from cboe_error
        raise


def _fetch_yfinance_option_chain(
    symbol: str,
    expiries: list[str] | None,
    max_expiries: int,
) -> tuple[pd.DataFrame, float]:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    available_expiries = list(ticker.options)
    if not available_expiries:
        raise RuntimeError(
            f"No option expiries returned for {symbol}."
        )

    if expiries is None:
        expiries = available_expiries[:max_expiries]

    invalid_expiries = [
        expiry
        for expiry in expiries
        if expiry not in available_expiries
    ]
    if invalid_expiries:
        raise ValueError(
            f"Unavailable expiries: {invalid_expiries}"
        )

    try:
        spot = float(ticker.fast_info["last_price"])
    except Exception:  # pragma: no cover - API shape varies
        history = ticker.history(period="5d")
        spot = float(history["Close"].dropna().iloc[-1])

    snapshot_date = date.today()
    frames = []
    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        frames.append(
            normalise_option_frame(
                raw=chain.calls,
                expiry=expiry,
                snapshot_date=snapshot_date,
                spot=spot,
                option_type="call",
            )
        )
        frames.append(
            normalise_option_frame(
                raw=chain.puts,
                expiry=expiry,
                snapshot_date=snapshot_date,
                spot=spot,
                option_type="put",
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["strike"].notna()].reset_index(drop=True)
    return combined, spot


def _parse_cboe_symbol(option_symbol: str) -> tuple[date, float, str]:
    """Parse an OCC-style symbol like SPY260903C00420000."""
    match = re.fullmatch(
        r"([A-Z]+)(\d{6})([CP])(\d{8})",
        option_symbol,
    )
    if match is None:
        raise ValueError(
            f"Cannot parse option symbol: {option_symbol}"
        )
    _, yymmdd, option_type_code, strike_code = match.groups()
    expiry = pd.to_datetime(yymmdd, format="%y%m%d").date()
    strike = float(int(strike_code)) / 1000.0
    option_type = "call" if option_type_code == "C" else "put"
    return expiry, strike, option_type


def _fetch_cboe_option_chain(
    symbol: str,
    max_expiries: int,
) -> tuple[pd.DataFrame, float]:
    """Fetch delayed option-chain quotes from Cboe's public JSON endpoint."""
    import json
    import urllib.request

    url = (
        "https://cdn.cboe.com/api/global/delayed_quotes/options/"
        f"{symbol}.json"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)

    data = payload["data"]
    spot = float(data["current_price"])
    snapshot_date = date.today()

    rows = []
    for record in data.get("options", []):
        expiry, strike, option_type = _parse_cboe_symbol(
            record["option"]
        )
        rows.append(
            {
                "snapshot_date": pd.Timestamp(snapshot_date),
                "expiry": pd.Timestamp(expiry),
                "option_type": option_type,
                "strike": strike,
                "bid": record.get("bid"),
                "ask": record.get("ask"),
                "last": record.get("last_trade_price"),
                "volume": record.get("volume"),
                "open_interest": record.get("open_interest"),
                "spot": spot,
                "multiplier": 100,
            }
        )

    combined = pd.DataFrame(rows)
    combined = combined[combined["expiry"] > combined["snapshot_date"]]

    if max_expiries is not None:
        selected = sorted(
            combined["expiry"].unique()
        )[:max_expiries]
        combined = combined[
            combined["expiry"].isin(selected)
        ]

    combined = combined[combined["strike"].notna()].reset_index(drop=True)
    return combined, spot


def clean_quote_frame(
    quotes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Apply quote-quality rules and return cleaned quotes with a report."""
    frame = quotes.copy()
    frame["mid"] = 0.5 * (frame["bid"] + frame["ask"])
    frame["spread"] = frame["ask"] - frame["bid"]
    frame["quote_valid"] = True
    frame["invalid_reason"] = ""

    rules = [
        (frame["bid"].isna() | frame["ask"].isna(), "missing_quote"),
        (frame["bid"] < 0, "negative_bid"),
        (frame["ask"] <= 0, "non_positive_ask"),
        (frame["ask"] < frame["bid"], "ask_below_bid"),
        (frame["mid"] <= 0, "non_positive_mid"),
        (frame["expiry"] <= frame["snapshot_date"], "expired"),
        (
            (frame["spread"] > 1.0)
            & (frame["spread"] > 0.5 * frame["mid"]),
            "wide_relative_spread",
        ),
        (frame["strike"] <= 0, "non_positive_strike"),
    ]

    for mask, reason in rules:
        newly_invalid = mask & frame["quote_valid"]
        frame.loc[newly_invalid, "quote_valid"] = False
        frame.loc[newly_invalid, "invalid_reason"] = reason

    cleaned = frame[frame["quote_valid"]].reset_index(drop=True)
    reason_counts = (
        frame.loc[~frame["quote_valid"], "invalid_reason"]
        .value_counts()
        .to_dict()
    )
    total = len(frame)
    report = {
        "total_rows": int(total),
        "kept_rows": int(len(cleaned)),
        "dropped_rows": int(total - len(cleaned)),
        "retention_rate": (
            float(len(cleaned) / total) if total else 0.0
        ),
        "drop_reasons": reason_counts,
    }
    return cleaned, report


def add_implied_volatility(
    quotes: pd.DataFrame,
    risk_free_rate: float = 0.04,
    dividend_yield: float | None = None,
) -> pd.DataFrame:
    """Add time, forward, log-moneyness and Black-Scholes implied volatility."""
    frame = quotes.copy()
    days = (frame["expiry"] - frame["snapshot_date"]).dt.days
    frame["time_to_expiry_days"] = days.astype(float)
    frame["time_to_expiry"] = days / 365.0

    if dividend_yield is None:
        dividend_yield, _ = estimate_per_expiry_dividend_yields(
            quotes=frame,
            risk_free_rate=risk_free_rate,
        )

    def _per_row_dividend(expiry) -> float:
        if isinstance(dividend_yield, dict):
            return dividend_yield.get(expiry, 0.012)
        return float(dividend_yield)

    frame["dividend_yield_used"] = frame["expiry"].apply(
        _per_row_dividend
    )
    frame["forward"] = frame["spot"] * (
        (risk_free_rate - frame["dividend_yield_used"])
        * frame["time_to_expiry"]
    ).apply(math.exp)
    frame["log_moneyness"] = (
        frame["strike"] / frame["forward"]
    ).apply(math.log)
    frame["no_arbitrage_pass"] = True
    frame["arbitrage_note"] = ""
    frame["iv"] = float("nan")
    frame["iv_error"] = ""

    lower_bounds = []
    upper_bounds = []
    for _, row in frame.iterrows():
        discounted_spot = row["spot"] * math.exp(
            -row["dividend_yield_used"] * row["time_to_expiry"]
        )
        discounted_strike = row["strike"] * math.exp(
            -risk_free_rate * row["time_to_expiry"]
        )
        if row["option_type"] == "call":
            lower = max(0.0, discounted_spot - discounted_strike)
            upper = discounted_spot
        else:
            lower = max(0.0, discounted_strike - discounted_spot)
            upper = discounted_strike
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    frame["arb_lower_bound"] = lower_bounds
    frame["arb_upper_bound"] = upper_bounds

    for index, row in frame.iterrows():
        tolerance = 1e-6
        if row["mid"] < row["arb_lower_bound"] - tolerance:
            frame.at[index, "no_arbitrage_pass"] = False
            frame.at[index, "arbitrage_note"] = (
                "below_european_lower_bound"
            )
        elif row["mid"] > row["arb_upper_bound"] + tolerance:
            frame.at[index, "no_arbitrage_pass"] = False
            frame.at[index, "arbitrage_note"] = (
                "above_european_upper_bound"
            )

        try:
            frame.at[index, "iv"] = implied_volatility(
                market_price=float(row["mid"]),
                spot=float(row["spot"]),
                strike=float(row["strike"]),
                time_to_expiry=float(row["time_to_expiry"]),
                risk_free_rate=risk_free_rate,
                option_type=row["option_type"],
                dividend_yield=float(row["dividend_yield_used"]),
            )
        except Exception as exc:  # noqa: BLE001 - record, do not fail the batch
            frame.at[index, "iv_error"] = str(exc)

    return frame


def estimate_per_expiry_dividend_yields(
    quotes: pd.DataFrame,
    risk_free_rate: float,
    fallback: float = 0.012,
) -> tuple[dict, int]:
    """Estimate per-expiry dividend yield from near-ATM put-call pairs.

    F = K + (C - P) * exp(rT), then q = r - log(F/S)/T. American-style
    early-exercise premiums make this an approximation; the nearest-to-ATM
    pair per expiry is used to limit the distortion.
    """
    frame = quotes.copy()
    frame["mid"] = 0.5 * (frame["bid"] + frame["ask"])
    estimates: dict = {}
    estimated = 0

    for expiry, group in frame.groupby("expiry"):
        spot = float(group["spot"].iloc[0])
        time_to_expiry = float(group["time_to_expiry"].iloc[0])
        best = None
        for strike, strike_group in group.groupby("strike"):
            calls = strike_group[
                strike_group["option_type"] == "call"
            ]
            puts = strike_group[
                strike_group["option_type"] == "put"
            ]
            if calls.empty or puts.empty or time_to_expiry <= 0:
                continue
            call_mid = float(calls["mid"].iloc[0])
            put_mid = float(puts["mid"].iloc[0])
            forward = float(strike) + (
                call_mid - put_mid
            ) * math.exp(risk_free_rate * time_to_expiry)
            if forward <= 0 or spot <= 0:
                continue
            dividend = risk_free_rate - (
                math.log(forward / spot) / time_to_expiry
            )
            distance = abs(float(strike) - spot)
            if best is None or distance < best[0]:
                best = (distance, dividend)
        if best is not None:
            estimates[expiry] = best[1]
            estimated += 1

    for expiry in frame["expiry"].unique():
        estimates.setdefault(expiry, fallback)
    return estimates, estimated
