"""Reusable charts for account snapshots, costs and Greeks."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_and_drawdown(
    snapshots: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Dual-axis equity / drawdown chart from account snapshots."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = snapshots.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(frame["timestamp"], frame["equity"], label="Equity")
    axis.set_ylabel("Equity")
    twin = axis.twinx()
    twin.plot(
        frame["timestamp"],
        frame["drawdown"],
        color="tab:red",
        label="Drawdown",
    )
    twin.set_ylabel("Drawdown")
    axis.grid(alpha=0.25)
    figure.suptitle("Account equity and drawdown")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_cost_breakdown(
    fills: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Stacked bar chart of commission / spread / slippage / impact costs."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = fills.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(
        frame["timestamp"],
        frame["commission"],
        label="Commission",
    )
    axis.bar(
        frame["timestamp"],
        frame["spread_cost"],
        bottom=frame["commission"],
        label="Spread",
    )
    axis.bar(
        frame["timestamp"],
        frame["slippage_cost"],
        bottom=frame["commission"] + frame["spread_cost"],
        label="Slippage",
    )
    axis.bar(
        frame["timestamp"],
        frame["impact_cost"],
        bottom=(
            frame["commission"]
            + frame["spread_cost"]
            + frame["slippage_cost"]
        ),
        label="Impact",
    )
    axis.set_ylabel("Cost")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.suptitle("Execution cost breakdown")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_greeks_vs_limits(
    greeks_frame: pd.DataFrame,
    limits: dict[str, float],
    output_path: str | Path,
) -> Path:
    """Plot delta/gamma/vega columns against their limits."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [name for name in ("delta", "gamma", "vega") if name in greeks_frame]
    figure, axes = plt.subplots(
        len(columns),
        1,
        figsize=(10, 3 * len(columns)),
        sharex=True,
    )
    if len(columns) == 1:
        axes = [axes]
    for axis, name in zip(axes, columns):
        axis.plot(greeks_frame.index, greeks_frame[name], label=name)
        limit = limits.get(name)
        if limit is not None:
            axis.axhline(limit, color="red", linestyle="--", label=f"limit {name}")
            axis.axhline(-limit, color="red", linestyle="--")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.suptitle("Portfolio Greeks vs risk limits")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
