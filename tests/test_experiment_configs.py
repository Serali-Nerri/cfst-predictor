from pathlib import Path

import pytest
import yaml

from scripts.run_experiment_suite import summarize_report


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIGS = [
    REPO_ROOT / "config/experiments/raw_original_metric.yaml",
    REPO_ROOT / "config/experiments/log_original_metric.yaml",
    REPO_ROOT / "config/experiments/log_transformed_metric.yaml",
]


def test_experiment_configs_preserve_shared_selection_contract():
    for config_path in EXPERIMENT_CONFIGS:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert isinstance(config, dict)

        assert config["data"]["split"]["auxiliary_features"][0]["column"] == "lambda_bar"
        assert config["model"]["use_optuna"] is True
        assert config["model"]["selection_objective"]["metric_space"] == "original_nexp"
        assert config["model"]["selection_objective"]["rmse_normalizer"] == "mean_actual"
        assert config["evaluation"]["regime_analysis"]["reference_split"] == "train"
        assert config["evaluation"]["regime_analysis"]["sort_metric"] == "cov"
        regime_names = [regime["name"] for regime in config["evaluation"]["regime_analysis"]["regimes"]]
        assert regime_names == [
            "axiality",
            "section_family",
            "slenderness_state",
            "scale_npl",
            "eccentricity_severity",
            "confinement_level",
        ]


def test_experiment_suite_summary_schema():
    summary = summarize_report(
        {
            "config_path": "config/experiments/example.yaml",
            "output_dir": "output/experiments/example",
            "target_mode": "eta_u_over_npl",
            "report_target_column": "Nexp (kN)",
            "split_strategy_effective": "regression_stratified",
            "optuna_metric_space": "original",
            "cv_metric_space": "original",
            "target_transform": {"enabled": True, "type": "log"},
            "optuna_run_info": {
                "best_score": 0.1,
                "best_params": {"max_depth": 5},
            },
            "selection_metrics_cv": {
                "composite_objective": 0.2,
                "rmse": 120.0,
                "r2": 0.98,
                "cov": 0.13,
            },
            "cv_results": {
                "mean_cv_score": 0.3,
                "mean_cv_rmse": 130.0,
                "std_cv_rmse": 10.0,
                "mean_cv_r2": 0.97,
                "mean_cv_cov": 0.14,
            },
            "test_metrics_original_space": {
                "rmse": 140.0,
                "mae": 80.0,
                "r2": 0.981,
                "cov": 0.132,
            },
        }
    )

    assert summary["selection_basis"] == "cv_composite_objective"
    assert summary["cv_composite_score"] == pytest.approx(0.2)
    assert summary["test_r2"] == pytest.approx(0.981)
    assert summary["test_cov"] == pytest.approx(0.132)
