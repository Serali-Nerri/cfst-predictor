import numpy as np
import pandas as pd
import pytest

from src.evaluator import Evaluator


def test_calculate_metrics_raises_on_length_mismatch():
    evaluator = Evaluator()

    with pytest.raises(ValueError, match="Length mismatch"):
        evaluator.calculate_metrics(pd.Series([1.0, 2.0]), np.array([1.0]))


def test_calculate_metrics_returns_none_mape_for_all_zero_targets():
    evaluator = Evaluator()
    metrics = evaluator.calculate_metrics(pd.Series([0.0, 0.0]), np.array([0.0, 1.0]))

    assert metrics["mape"] is None
    assert metrics["n_samples"] == 2


def test_calculate_metrics_returns_expected_keys_for_normal_input():
    evaluator = Evaluator()
    metrics = evaluator.calculate_metrics(pd.Series([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0]))

    for key in ["rmse", "mae", "r2", "mse", "max_error", "cov", "n_samples"]:
        assert key in metrics
    assert metrics["n_samples"] == 3
