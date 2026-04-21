import re

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader


def test_load_data_applies_log_transform_and_preserves_raw_target(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "feat": [1.0, 2.0, 3.0],
        "Nexp (kN)": [10.0, 20.0, 40.0],
    })
    df.to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    features, target = loader.load_data(str(csv_path), "Nexp (kN)", target_transform="log")

    expected = np.log(df["Nexp (kN)"].to_numpy(dtype=float))

    assert list(features.columns) == ["feat"]
    assert np.allclose(target.to_numpy(dtype=float), expected)
    assert loader.target_raw is not None
    assert np.allclose(loader.target_raw.to_numpy(dtype=float), df["Nexp (kN)"].to_numpy(dtype=float))


def test_load_data_rejects_unsupported_target_transform(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "feat": [1.0, 2.0],
        "Nexp (kN)": [9.0, 16.0],
    })
    df.to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    with pytest.raises(ValueError, match="Unsupported target transform 'sqrt'"):
        loader.load_data(str(csv_path), "Nexp (kN)", target_transform="sqrt")


def test_load_data_without_transform_returns_original_target(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "feat": [1.0, 2.0],
        "Nexp (kN)": [11.0, 13.0],
    })
    df.to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    _, target = loader.load_data(str(csv_path), "Nexp (kN)", target_transform=None)

    assert np.allclose(
        target.to_numpy(dtype=float),
        df["Nexp (kN)"].to_numpy(dtype=float),
    )


def test_load_data_applies_boxcox_transform(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "feat": [1.0, 2.0, 3.0],
        "Nexp (kN)": [4.0, 9.0, 16.0],
    })
    df.to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    _, target = loader.load_data(str(csv_path), "Nexp (kN)", target_transform="boxcox_0.5")

    expected = (np.sqrt(df["Nexp (kN)"].to_numpy(dtype=float)) - 1.0) / 0.5
    assert np.allclose(target.to_numpy(dtype=float), expected)


def test_load_data_rejects_invalid_log_target_domain(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"feat": [1.0, 2.0], "Nexp (kN)": [0.0, 10.0]}).to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    with pytest.raises(ValueError, match="log target transform requires all target values to be finite and > 0"):
        loader.load_data(str(csv_path), "Nexp (kN)", target_transform="log")


def test_load_data_raises_when_target_column_missing(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"feat": [1.0, 2.0]}).to_csv(csv_path, index=False)

    loader = DataLoader()
    with pytest.raises(ValueError, match=re.escape("Target column 'Nexp (kN)' not found in data")):
        loader.load_data(str(csv_path), "Nexp (kN)")


def test_load_data_requires_precomputed_strength_ratio_target_columns(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "feat": [1.0, 2.0],
            "Nexp (kN)": [380.0, 304.0],
        }
    )
    df.to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    with pytest.raises(ValueError, match="Processed input is missing target-mode helper columns"):
        loader.load_data(
            str(csv_path),
            "Nexp (kN)",
            target_transform=None,
            target_mode="eta_u_over_npl",
        )


def test_load_data_uses_precomputed_r_target_columns(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "feat": [1.0, 2.0],
            "Nexp (kN)": [380.0, 304.0],
            "Npl (kN)": [380.0, 380.0],
            "eta_u": [1.0, 0.8],
            "r": [0.0, -0.2],
            "axial_flag": ["axial", "eccentric"],
            "section_family": ["square", "rectangular"],
        }
    )
    df.to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    features, target = loader.load_data(
        str(csv_path),
        "Nexp (kN)",
        target_transform=None,
        target_mode="r_over_npl",
    )

    assert "Npl (kN)" in features.columns
    assert "eta_u" not in features.columns
    assert "r" not in features.columns
    assert "axial_flag" in features.columns
    assert "section_family" in features.columns
    assert np.allclose(target.to_numpy(dtype=float), np.array([0.0, -0.2]))
    assert loader.training_target_name == "r"
