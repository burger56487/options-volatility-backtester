from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.long_straddle_backtest import (
    LongStraddleBacktestConfig,
)
from src.backtest.sensitivity_analysis import (
    create_sensitivity_pivot,
    run_hedging_sensitivity_analysis,
)
from src.market_data.underlying_data import load_price_data


def save_heatmap(
    pivot: pd.DataFrame,
    title: str,
    colorbar_label: str,
    output_path: Path,
    format_string: str,
) -> None:
    """Save a simple matplotlib heatmap from a sensitivity pivot table."""
    figure, axis = plt.subplots(figsize=(9, 6))

    image = axis.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
    )

    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(
        [f"{value:g}" for value in pivot.columns]
    )

    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels(
        [f"{value:g}" for value in pivot.index]
    )

    axis.set_xlabel("Underlying Slippage (bps)")
    axis.set_ylabel("Delta Rebalance Threshold (shares)")
    axis.set_title(title)

    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = pivot.iloc[row_index, column_index]

            axis.text(
                column_index,
                row_index,
                format(value, format_string),
                ha="center",
                va="center",
                fontsize=9,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )
    colorbar.set_label(colorbar_label)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    input_path = Path("data/raw/spy_daily_adjusted.csv")
    output_directory = Path("outputs/sensitivity")
    figure_directory = Path("outputs/figures")

    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory.mkdir(parents=True, exist_ok=True)

    data = load_price_data(input_path)

    entry_date = pd.Timestamp("2025-03-03")

    base_config = LongStraddleBacktestConfig(
        days_to_expiry=30,
        quantity=1,
        multiplier=100,
        risk_free_rate=0.04,
        dividend_yield=0.012,
        commission_per_share=0.005,
        allow_fractional_shares=False,
    )

    delta_thresholds = (0.0, 1.0, 5.0, 10.0, 20.0)
    slippage_bps_values = (0.5, 1.0, 5.0, 10.0)

    print("=" * 72)
    print("DELTA-HEDGING SENSITIVITY ANALYSIS")
    print("=" * 72)
    print()
    print(f"Entry date: {entry_date.date()}")
    print(f"Delta thresholds: {delta_thresholds}")
    print(f"Slippage assumptions: {slippage_bps_values}")
    print()

    results = run_hedging_sensitivity_analysis(
        price_data=data,
        entry_date=entry_date,
        base_config=base_config,
        delta_thresholds=delta_thresholds,
        slippage_bps_values=slippage_bps_values,
    )

    output_csv = output_directory / "hedging_sensitivity_results.csv"
    results.to_csv(output_csv, index=False)

    print("Scenario results")
    print("-" * 72)
    print(
        results[
            [
                "delta_threshold",
                "slippage_bps",
                "final_pnl",
                "max_drawdown",
                "number_of_hedge_trades",
                "cumulative_hedge_turnover",
                "cumulative_hedge_costs",
                "hedge_turnover_ratio",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print()

    pnl_pivot = create_sensitivity_pivot(
        results=results,
        value_column="final_pnl",
    )

    turnover_pivot = create_sensitivity_pivot(
        results=results,
        value_column="hedge_turnover_ratio",
    )

    costs_pivot = create_sensitivity_pivot(
        results=results,
        value_column="cumulative_hedge_costs",
    )

    save_heatmap(
        pivot=pnl_pivot,
        title="Long Straddle P&L Sensitivity",
        colorbar_label="Final P&L",
        output_path=(
            figure_directory
            / "spy_hedging_sensitivity_pnl.png"
        ),
        format_string=".0f",
    )

    save_heatmap(
        pivot=turnover_pivot,
        title="Delta-Hedge Turnover Sensitivity",
        colorbar_label="Turnover Ratio",
        output_path=(
            figure_directory
            / "spy_hedging_sensitivity_turnover.png"
        ),
        format_string=".1f",
    )

    save_heatmap(
        pivot=costs_pivot,
        title="Delta-Hedging Cost Sensitivity",
        colorbar_label="Cumulative Hedge Costs",
        output_path=(
            figure_directory
            / "spy_hedging_sensitivity_costs.png"
        ),
        format_string=".1f",
    )

    best_result = results.loc[
        results["final_pnl"].idxmax()
    ]

    print("Best scenario by final P&L")
    print("-" * 72)
    print(best_result.round(6).to_string())
    print()

    print(f"Saved results to: {output_csv}")
    print(
        "Saved figures to: "
        f"{figure_directory}"
    )


if __name__ == "__main__":
    main()
