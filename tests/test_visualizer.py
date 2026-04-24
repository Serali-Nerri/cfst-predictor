import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src.visualizer import (
    plot_error_distribution,
    plot_predictions_scatter,
    plot_residuals,
)


def test_visualizer_saves_core_evaluation_plots(tmp_path):
    y_true = pd.Series([100.0, 150.0, 200.0, 260.0])
    y_pred = np.array([98.0, 155.0, 190.0, 270.0])

    scatter_path = tmp_path / "scatter.png"
    residual_path = tmp_path / "residual.png"
    error_path = tmp_path / "error.png"

    plot_predictions_scatter(y_true, y_pred, save_path=str(scatter_path), r2_score=0.98)
    plot_residuals(y_true, y_pred, save_path=str(residual_path))
    plot_error_distribution(y_true, y_pred, save_path=str(error_path))

    assert scatter_path.is_file()
    assert residual_path.is_file()
    assert error_path.is_file()
