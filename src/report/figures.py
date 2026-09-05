"""Matplotlib figures for the end-to-end analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.volatility_surface.svi import svi_total_variance


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def generate_all_figures(result, fig_dir: Path) -> None:
    """Generate skew, term-structure, SVI-fit and gamma heatmap figures."""
    plt = _matplotlib()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if result.curves:
        _plot_skew_curves(result, fig_dir, plt)
    if result.term_structure is not None and not (
        result.term_structure.curve.empty
    ):
        _plot_term_structure(result, fig_dir, plt)
    _plot_svi_fits(result, fig_dir, plt)
    _plot_gamma_heatmap(result, fig_dir, plt)


def _plot_skew_curves(result, fig_dir: Path, plt) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for expiry, curve in result.curves.items():
        data = curve.dropna(subset=["iv_mid"])
        if data.empty:
            continue
        axis.plot(
            data["log_moneyness"],
            data["iv_mid"],
            marker="o",
            markersize=3,
            label=str(pd.Timestamp(expiry).date()),
        )
    axis.axvline(0.0, color="grey", linestyle="--", alpha=0.5)
    axis.set_xlabel("log-moneyness ln(K/F)")
    axis.set_ylabel("implied volatility")
    axis.set_title(f"{result.ticker} skew by expiry")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(fig_dir / "skew_curves.png", dpi=120)
    plt.close(figure)


def _plot_term_structure(result, fig_dir: Path, plt) -> None:
    term = result.term_structure
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        term.curve["time_to_expiry"],
        term.curve["atm_vol"],
        marker="o",
        color="navy",
    )
    axis.set_xlabel("time to expiry (years)")
    axis.set_ylabel("ATM implied volatility")
    axis.set_title(f"{result.ticker} term structure ({term.shape})")
    figure.tight_layout()
    figure.savefig(fig_dir / "term_structure.png", dpi=120)
    plt.close(figure)


def _plot_svi_fits(result, fig_dir: Path, plt) -> None:
    if not result.svi_results or not result.curves:
        return
    for item in result.svi_results:
        if not item.valid or item.expiry not in result.curves:
            continue
        curve = result.curves[item.expiry].dropna(subset=["iv_mid"])
        if curve.empty:
            continue
        k = curve["log_moneyness"].to_numpy(dtype=float)
        iv = curve["iv_mid"].to_numpy(dtype=float)
        grid = np.linspace(float(k.min()), float(k.max()), 100)
        w_fit = svi_total_variance(grid, *item.params)
        iv_fit = np.sqrt(
            np.maximum(w_fit, 1e-12) / item.time_to_expiry
        )
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.scatter(k, iv, s=15, color="black", label="market")
        axis.plot(
            grid,
            iv_fit,
            color="red",
            label=f"SVI (RMSE={item.rmse_vol:.4f})",
        )
        axis.set_xlabel("log-moneyness")
        axis.set_ylabel("implied volatility")
        axis.set_title(f"SVI fit {pd.Timestamp(item.expiry).date()}")
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            fig_dir
            / f"svi_fit_{pd.Timestamp(item.expiry).date()}.png",
            dpi=120,
        )
        plt.close(figure)


def _plot_gamma_heatmap(result, fig_dir: Path, plt) -> None:
    from src.greeks.chain_greeks import build_greek_heatmap

    if result.chain_greeks is None or "gamma" not in (
        result.chain_greeks.columns
    ):
        return
    pivot = build_greek_heatmap(result.chain_greeks, "gamma")
    if pivot.empty:
        return
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(
        [f"{value:.3f}" for value in pivot.columns],
        rotation=45,
        fontsize=7,
    )
    tick_step = max(1, len(pivot.index) // 10)
    tick_positions = list(range(0, len(pivot.index), tick_step))
    axis.set_yticks(tick_positions)
    axis.set_yticklabels(
        [f"{pivot.index[pos]:.0f}" for pos in tick_positions],
        fontsize=7,
    )
    axis.set_xlabel("time to expiry")
    axis.set_ylabel("strike")
    axis.set_title("gamma risk heatmap")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(fig_dir / "gamma_heatmap.png", dpi=120)
    plt.close(figure)
