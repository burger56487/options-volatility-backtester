"""Attribution data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AttributionRow:
    valuation_date: str
    instrument: str
    factor: str
    amount: float
    category: str  # greek / trade / financing / settlement / residual
    trade_or_holding: str = "holding"
    diagnostic: str = ""


@dataclass
class DailyAttribution:
    rows: list[AttributionRow] = field(default_factory=list)

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "valuation_date": row.valuation_date,
                    "instrument": row.instrument,
                    "factor": row.factor,
                    "amount": row.amount,
                    "category": row.category,
                    "trade_or_holding": row.trade_or_holding,
                    "diagnostic": row.diagnostic,
                }
                for row in self.rows
            ]
        )


def save_daily_attribution(
    attribution: DailyAttribution,
    output_directory,
) -> None:
    """Write daily attribution rows and a factor-summary CSV."""
    from pathlib import Path

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = attribution.to_dataframe()
    frame.to_csv(
        output_path / "attribution_daily.csv",
        index=False,
    )
    summary = (
        frame.groupby("factor")["amount"]
        .sum()
        .reset_index()
    )
    summary.to_csv(
        output_path / "attribution_summary.csv",
        index=False,
    )


def classify_residual(
    residual: float,
    spot_move: float,
    is_settlement_day: bool = False,
    is_trade_day: bool = False,
    tolerance: float = 1e-6,
) -> str:
    """Assign a diagnostic code to an unexplained residual."""
    if abs(residual) <= tolerance:
        return "clean"
    if is_settlement_day:
        return "settlement_jump"
    if is_trade_day:
        return "intraday_trade"
    if abs(spot_move) > 0.1:
        return "large_spot_jump"
    return "unexplained"
