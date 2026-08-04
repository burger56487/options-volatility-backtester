from pathlib import Path

import matplotlib.pyplot as plt

from src.market_data.realized_volatility import (
    add_realized_volatility_features,
    volatility_summary,
)
from src.market_data.underlying_data import (
    load_price_data,
    save_price_data,
)


def main() -> None:
    input_path = Path("data/raw/spy_daily_adjusted.csv")
    output_path = Path(
        "data/processed/spy_daily_with_realized_volatility.csv"
    )
    figure_path = Path(
        "outputs/figures/spy_realized_volatility.png"
    )

    print("=" * 64)
    print("SPY REALISED VOLATILITY ANALYSIS")
    print("=" * 64)
    print()

    data = load_price_data(input_path)

    result = add_realized_volatility_features(
        data=data,
        price_column="Close",
        windows=(20, 60, 252),
    )

    volatility_columns = [
        "realised_vol_20d",
        "realised_vol_60d",
        "realised_vol_252d",
    ]

    summary = volatility_summary(
        data=result,
        volatility_columns=volatility_columns,
    )

    print("Realised volatility summary")
    print("-" * 64)
    print(summary.round(4).to_string())
    print()

    save_price_data(result, output_path)

    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for column in volatility_columns:
        plt.plot(
            result.index,
            result[column],
            label=column.replace(
                "realised_vol_",
                "",
            ).replace("d", "-day"),
            linewidth=1.2,
        )

    plt.title("SPY Annualised Realised Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualised Volatility")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()

    print(f"Saved processed data to: {output_path}")
    print(f"Saved figure to: {figure_path}")
    print()

    print("Latest volatility observations")
    print("-" * 64)
    print(
        result[
            ["Close"] + volatility_columns
        ]
        .tail()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
