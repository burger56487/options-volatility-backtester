import pandas as pd

from src.performance.aggregation import aggregate_metric_runs
from src.reporting.plots import plot_parameter_stability_heatmap


def test_aggregate_metric_runs():
    runs = pd.DataFrame(
        {
            "seed": range(6),
            "sharpe": [1.0, 1.1, 0.9, 1.2, 0.8, 1.0],
        }
    )
    result = aggregate_metric_runs(runs, "sharpe")
    assert result["n_runs"] == 6
    assert abs(result["mean"] - 1.0) < 1e-9
    assert result["std"] > 0


def test_parameter_stability_heatmap_saves(tmp_path):
    frame = pd.DataFrame(
        {
            "threshold": [1.0, 1.0, 1.1, 1.1],
            "cost": [1.0, 2.0, 1.0, 2.0],
            "sharpe": [0.5, 0.4, 0.8, 0.7],
        }
    )
    path = plot_parameter_stability_heatmap(
        frame,
        x_column="threshold",
        y_column="cost",
        value_column="sharpe",
        output_path=tmp_path / "heatmap.png",
    )
    assert path.exists()
