import numpy as np
import pandas as pd
import pytest

from src.splitting import build_regression_stratification_labels


def test_regression_stratification_tries_non_prefix_auxiliary_subsets():
    features = pd.DataFrame(
        {
            "Ac (mm^2)": [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10, 10,
                1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            ],
            "lambda_bar": [1, 1, 1, 1, 2, 2, 2, 2] * 4,
        }
    )
    target = pd.Series(
        [10] * 16 + [20] * 16,
        dtype=float,
    )
    labels, metadata = build_regression_stratification_labels(
        features=features,
        target_raw=target,
        split_config={
            "target_bins": 2,
            "auxiliary_features": [
                {"column": "Ac (mm^2)", "bins": 2},
                {"column": "lambda_bar", "bins": 2},
            ],
        },
        minimum_count=5,
    )

    used_columns = [item["column"] for item in metadata["used_auxiliary_features"]]
    candidate_subsets = metadata["candidate_subsets_tried"]

    assert labels.nunique() > 1
    assert used_columns == ["lambda_bar"]
    assert any(item["columns"] == ["lambda_bar"] for item in candidate_subsets)


def test_regression_stratification_rejects_non_finite_values():
    features = pd.DataFrame({"lambda_bar": [0.1, np.nan, 0.3, 0.4]})
    target = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=float)

    with pytest.raises(ValueError, match="Stratification values must be finite"):
        build_regression_stratification_labels(
            features=features,
            target_raw=target,
            split_config={
                "target_bins": 2,
                "auxiliary_features": [{"column": "lambda_bar", "bins": 2}],
            },
            minimum_count=2,
        )
