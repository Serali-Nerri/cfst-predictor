import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from train import (
    build_cv_splitter,
    build_model_params,
    build_sample_weights,
    build_study_name,
    get_cv_n_splits,
    make_common_artifact_payload,
    make_data_split_summary,
    make_overfitting_summary,
    make_selection_metrics_cv,
    make_serializable_cv_results,
    normalize_model_backbone,
)
from src.model_trainer import ModelTrainer


def test_build_cv_splitter_uses_config_values():
    splitter = build_cv_splitter(
        {"n_splits": 4, "shuffle": True, "random_state": 123},
        "random",
    )

    assert isinstance(splitter, KFold)
    assert splitter.n_splits == 4
    assert splitter.shuffle is True
    assert splitter.random_state == 123


def test_build_cv_splitter_ignores_random_state_when_shuffle_disabled():
    splitter = build_cv_splitter(
        {"n_splits": 3, "shuffle": False, "random_state": 999},
        "random",
    )

    assert splitter.n_splits == 3
    assert splitter.shuffle is False
    assert splitter.random_state is None


def test_build_cv_splitter_rejects_non_boolean_shuffle():
    with pytest.raises(ValueError, match="config.cv.shuffle must be a boolean"):
        build_cv_splitter({"n_splits": 5, "shuffle": "yes"}, "random")


def test_get_cv_n_splits_rejects_deprecated_n_folds():
    with pytest.raises(ValueError, match="config.cv.n_folds is deprecated"):
        get_cv_n_splits({"n_folds": 5})


def test_get_cv_n_splits_returns_integer_value():
    assert get_cv_n_splits({"n_splits": 7}) == 7


def test_xgboost_finalize_params_after_cv_uses_median_best_iteration_plus_one():
    trainer = ModelTrainer(
        params={"n_estimators": 200, "device": "cpu", "n_jobs": -1, "random_state": 42},
        backbone="xgboost",
        validation_size=0.0,
    )

    finalized_params, fold_best_iterations = trainer.finalize_params_after_cv(
        {
            "fold_details": [
                {"best_iteration": 9},
                {"best_iteration": 11},
                {"best_iteration": None},
                {"best_iteration": 13},
            ]
        }
    )

    assert finalized_params["n_estimators"] == 12
    assert fold_best_iterations == [10, 12, 14]


def test_build_sample_weights_supports_high_e_over_h_emphasis():
    features = pd.DataFrame({"e/h": [0.0, 0.05, 0.15]}, index=[10, 11, 12])

    weights, metadata = build_sample_weights(
        features,
        {
            "enabled": True,
            "strategy": "e_over_h_threshold",
            "column": "e/h",
            "threshold": 0.1,
            "base_weight": 1.0,
            "high_weight": 1.8,
        },
    )

    assert weights is not None
    assert weights.to_dict() == {10: 1.0, 11: 1.0, 12: 1.8}
    assert metadata["n_high_weight"] == 1
    assert metadata["n_base_weight"] == 2


def test_make_selection_metrics_cv_returns_expected_keys():
    result = make_selection_metrics_cv(
        {"mean_cv_score": 1.2, "mean_cv_rmse": 3.4, "mean_cv_r2": 0.95, "mean_cv_cov": 0.08}
    )

    assert result == {
        "composite_objective": 1.2,
        "rmse": 3.4,
        "mae": None,
        "r2": 0.95,
        "mape": None,
        "mu": None,
        "cov": 0.08,
        "a20_index": None,
    }


def test_make_overfitting_summary_reports_status():
    result = make_overfitting_summary(
        {"rmse": 10.0},
        {"rmse": 13.0},
        {"rmse": 0.2},
        {"rmse": 0.3},
    )

    assert result["detected"] is True
    assert result["status"] == "overfitting"
    assert result["rmse_ratio_original"] == pytest.approx(1.3)


def test_make_data_split_summary_counts_strata():
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    X_test = pd.DataFrame({"a": [4, 5]})
    train_strata = pd.Series(["s1", "s1", "s2"])
    test_strata = pd.Series(["s1", "s2"])

    result = make_data_split_summary(X_train, X_test, 0.4, train_strata, test_strata)

    assert result["n_train"] == 3
    assert result["n_test"] == 2
    assert result["n_strata_train"] == 2
    assert result["n_strata_test"] == 2


def test_make_serializable_cv_results_handles_numpy_arrays():
    result = make_serializable_cv_results(
        {"cv_scores": np.array([1.0, 2.0]), "nested": {"values": np.array([3.0, 4.0])}}
    )

    assert result == {"cv_scores": [1.0, 2.0], "nested": {"values": [3.0, 4.0]}}


def test_make_common_artifact_payload_includes_shared_sections():
    result = make_common_artifact_payload(
        context_hash="abc123",
        model_backbone="xgboost",
        params_source="config",
        final_model_params={"max_depth": 5},
        optuna_run_info=None,
        optuna_metric_space="original",
        cv_metric_space="transformed",
        selection_objective={"metric_space": "original_nexp"},
        target_metadata={"target_mode": "raw"},
        split_strategy="random",
        effective_split_strategy="random",
        stratification_metadata={"strategy": "random"},
        sample_weight_metadata={"enabled": False},
        cv_results={"mean_cv_score": 1.0, "mean_cv_rmse": 2.0, "mean_cv_r2": 0.9, "mean_cv_cov": 0.1},
        train_metrics={"rmse": 1.0},
        test_metrics={"rmse": 2.0},
        train_metrics_trans={"rmse": 0.1},
        test_metrics_trans={"rmse": 0.2},
        regime_schema={},
        train_regime_metrics={},
        test_regime_metrics={},
        final_n_estimators=123,
        fold_best_iterations=[100, 120],
    )

    assert result["context_hash"] == "abc123"
    assert result["model_backbone"] == "xgboost"
    assert result["selection_metrics_cv"] == {
        "composite_objective": 1.0,
        "rmse": 2.0,
        "mae": None,
        "r2": 0.9,
        "mape": None,
        "mu": None,
        "cov": 0.1,
        "a20_index": None,
    }
    assert result["final_n_estimators_from_cv"] == 123
    assert result["fold_best_iterations"] == [100, 120]


def test_normalize_model_backbone_defaults_to_xgboost_when_missing():
    assert normalize_model_backbone(None) == "xgboost"


def test_normalize_model_backbone_accepts_case_insensitive_xgboost():
    assert normalize_model_backbone("XGBoost") == "xgboost"


def test_normalize_model_backbone_rejects_unsupported_values():
    with pytest.raises(ValueError, match="Unsupported config.model.backbone"):
        normalize_model_backbone("unknown")


def test_build_model_params_requires_full_xgboost_param_set():
    with pytest.raises(ValueError, match="missing required keys"):
        build_model_params(
            model_config={"params": {"objective": "reg:squarederror"}},
            model_backbone="xgboost",
        )


def test_build_model_params_allows_rf_defaults_without_explicit_params():
    model_config = {}

    params = build_model_params(model_config=model_config, model_backbone="rf")

    assert params == {}


def test_build_study_name_includes_backbone_prefix():
    study_name = build_study_name(
        "data/processed/final_feature_parameters_raw.csv",
        "abc123",
        "xgboost",
    )

    assert study_name.startswith("xgboost_optimization__")
    assert study_name.endswith("__abc123")
