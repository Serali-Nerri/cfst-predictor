import sys
import types
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd
import pytest

import src.backbones.xgboost_adapter as xgboost_adapter_module
from src.model_trainer import (
    KeMLRegressor,
    ModelTrainer,
    _build_selection_objective_config,
    _calculate_regression_metrics,
    _calculate_selection_objective,
)


class RecordingSplitter:
    def __init__(self):
        self.split_called_with = None

    def split(self, X, y=None, groups=None):
        self.split_called_with = (X.copy(), None if y is None else y.copy(), groups)
        indices = np.arange(len(X))
        yield indices[:2], indices[2:]
        yield indices[2:], indices[:2]

    def get_n_splits(self, X=None, y=None, groups=None):
        return 2


class DummyTrial:
    number = 7

    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low


class DummyStudy:
    def __init__(self):
        self.trials = []
        self.best_params = {"max_depth": 3}
        self.best_value = 0.25
        self.best_trial = DummyTrial()

    def enqueue_trial(self, params):
        return None

    def optimize(self, objective, n_trials, timeout):
        trial = DummyTrial()
        self.best_value = objective(trial)
        self.trials = [trial]


class ConstantPsiRegressor:
    def __init__(self, **kwargs):
        self.best_iteration = 7
        self.constant_prediction = 1.5
        self.feature_importances_ = np.array([1.0])

    def fit(self, X_train, y_train, verbose=False, eval_set=None):
        return self

    def predict(self, X_val):
        return np.full(len(X_val), self.constant_prediction, dtype=float)


class ConstantLinearModel:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class RecordingRegressor:
    last_init_kwargs: ClassVar[dict[str, Any] | None] = None
    last_fit_kwargs: ClassVar[dict[str, Any] | None] = None
    last_fit_target: ClassVar[np.ndarray[Any, Any] | None] = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = dict(kwargs)
        self.best_iteration = 3
        self.feature_importances_ = np.array([1.0])

    def fit(self, X_train, y_train, **kwargs):
        type(self).last_fit_kwargs = dict(kwargs)
        type(self).last_fit_target = np.asarray(y_train, dtype=float)
        return self

    def predict(self, X_val):
        return np.zeros(len(X_val), dtype=float)


def test_keml_regressor_sums_linear_and_nonlinear_predictions():
    regressor = KeMLRegressor(
        linear_model=ConstantLinearModel(0.2),
        nonlinear_model=ConstantPsiRegressor(),
        linear_feature_names=["a"],
    )
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    predictions = regressor.predict(X)

    assert np.allclose(predictions, np.array([1.7, 1.7]))


def test_fit_model_passes_sample_weights_to_plain_xgb(monkeypatch):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 42},
        validation_size=0.0,
    )
    X_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0]})
    y_train = pd.Series([0.0, 0.1, 0.2])
    sample_weight = pd.Series([1.0, 2.0, 3.0])

    monkeypatch.setattr(xgboost_adapter_module.xgb, "XGBRegressor", RecordingRegressor)

    trainer._fit_model(trainer.params, X_train, y_train, sample_weight=sample_weight)

    assert RecordingRegressor.last_fit_kwargs is not None
    assert np.allclose(RecordingRegressor.last_fit_kwargs["sample_weight"], np.array([1.0, 2.0, 3.0]))


def test_fit_model_preserves_eval_settings_for_keml_branch(monkeypatch):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 42},
        target_mode="r_over_npl",
        target_transform_type=None,
        validation_size=0.0,
        use_keml_residual_split=True,
        linear_feature_names=["linear"],
        linear_ridge_alpha=1.0,
        early_stopping_rounds=9,
        eval_metric="rmse",
    )
    X_train = pd.DataFrame({"linear": [0.0, 1.0, 2.0], "other": [1.0, 1.5, 2.0]})
    y_train = pd.Series([0.0, 0.1, 0.2])
    X_val = pd.DataFrame({"linear": [3.0, 4.0], "other": [2.5, 3.0]})
    y_val = pd.Series([0.3, 0.4])

    sample_weight = pd.Series([1.0, 2.0, 3.0])
    sample_weight_eval = pd.Series([4.0, 5.0])

    monkeypatch.setattr(xgboost_adapter_module.xgb, "XGBRegressor", RecordingRegressor)

    trainer._fit_model(
        trainer.params,
        X_train,
        y_train,
        X_val,
        y_val,
        sample_weight=sample_weight,
        sample_weight_eval_set=sample_weight_eval,
    )

    assert RecordingRegressor.last_init_kwargs is not None
    assert RecordingRegressor.last_init_kwargs["early_stopping_rounds"] == 9
    assert RecordingRegressor.last_init_kwargs["eval_metric"] == "rmse"
    assert RecordingRegressor.last_fit_kwargs is not None
    assert "eval_set" in RecordingRegressor.last_fit_kwargs
    assert "sample_weight" in RecordingRegressor.last_fit_kwargs
    assert "sample_weight_eval_set" in RecordingRegressor.last_fit_kwargs
    assert np.allclose(RecordingRegressor.last_fit_kwargs["sample_weight"], np.array([1.0, 2.0, 3.0]))
    assert np.allclose(RecordingRegressor.last_fit_kwargs["sample_weight_eval_set"][0], np.array([4.0, 5.0]))
    eval_set = RecordingRegressor.last_fit_kwargs["eval_set"]
    assert isinstance(eval_set, list)
    assert len(eval_set) == 1
    eval_X, eval_y = cast(tuple[pd.DataFrame, pd.Series], eval_set[0])
    assert eval_X.equals(X_val)
    assert np.asarray(eval_y, dtype=float).shape == (2,)
    assert RecordingRegressor.last_fit_target is not None
    assert RecordingRegressor.last_fit_target.shape == (3,)


def test_fit_model_requires_same_keml_linear_features_in_validation(monkeypatch):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 42},
        target_mode="r_over_npl",
        target_transform_type=None,
        validation_size=0.0,
        use_keml_residual_split=True,
        linear_feature_names=["linear", "other_linear"],
        linear_ridge_alpha=1.0,
    )
    X_train = pd.DataFrame(
        {"linear": [0.0, 1.0, 2.0], "other_linear": [2.0, 3.0, 4.0], "other": [1.0, 1.5, 2.0]}
    )
    y_train = pd.Series([0.0, 0.1, 0.2])
    X_val = pd.DataFrame({"linear": [3.0, 4.0], "other": [2.5, 3.0]})
    y_val = pd.Series([0.3, 0.4])

    monkeypatch.setattr(xgboost_adapter_module.xgb, "XGBRegressor", RecordingRegressor)

    with pytest.raises(ValueError, match="Validation data is missing KeML linear features"):
        trainer._fit_model(trainer.params, X_train, y_train, X_val, y_val)


class InfoOnlyModel:
    pass


def test_get_model_info_uses_actual_model_type():
    trainer = ModelTrainer(params={"device": "cpu", "n_jobs": -1, "random_state": 42})
    trainer.model = InfoOnlyModel()

    info = trainer.get_model_info()

    assert info["model_type"] == "InfoOnlyModel"


def test_model_trainer_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="Unsupported model backbone"):
        ModelTrainer(
            params={"device": "cpu", "n_jobs": -1, "random_state": 42},
            backbone="unknown",
        )


def test_model_trainer_accepts_xgb_backbone_alias(monkeypatch):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 42},
        backbone="xgb",
        validation_size=0.0,
    )
    X_train = pd.DataFrame({"feature": [0.0, 1.0, 2.0]})
    y_train = pd.Series([0.0, 0.1, 0.2])

    monkeypatch.setattr(xgboost_adapter_module.xgb, "XGBRegressor", RecordingRegressor)

    trainer._fit_model(trainer.params, X_train, y_train)

    assert RecordingRegressor.last_fit_kwargs is not None


def test_cross_validate_restores_report_space_metrics_for_eta_u_target(monkeypatch):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 42},
        target_mode="eta_u_over_npl",
        target_transform_type=None,
        validation_size=0.0,
    )
    splitter = RecordingSplitter()
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "Npl (kN)": [100.0, 200.0, 100.0, 200.0],
        }
    )
    y = pd.Series([1.0, 1.0, 2.0, 2.0])
    y_report = pd.Series([100.0, 200.0, 200.0, 400.0])

    monkeypatch.setattr(xgboost_adapter_module.xgb, "XGBRegressor", ConstantPsiRegressor)

    results = trainer.cross_validate(X, y, y_report=y_report, cv=splitter)

    expected_rmse = float(np.sqrt(((50.0**2) + (100.0**2)) / 2.0))
    assert splitter.split_called_with is not None
    assert np.isclose(results["mean_cv_rmse"], expected_rmse)
    assert np.isclose(results["mean_cv_cov"], 0.0)
    assert all(detail["best_iteration"] == 7 for detail in results["fold_details"])


def test_optimize_hyperparameters_uses_report_target_and_splitter(monkeypatch, tmp_path):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 42},
        use_optuna=True,
        n_trials=1,
        optuna_timeout=1,
        target_mode="eta_u_over_npl",
        target_transform_type=None,
        validation_size=0.0,
    )
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "Npl (kN)": [100.0, 200.0, 100.0, 200.0],
        }
    )
    y = pd.Series([1.0, 1.0, 2.0, 2.0])
    y_report = pd.Series([100.0, 200.0, 200.0, 400.0])
    splitter = RecordingSplitter()

    fake_optuna = types.SimpleNamespace(
        create_study=lambda **kwargs: DummyStudy(),
        samplers=types.SimpleNamespace(TPESampler=lambda **kwargs: object()),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setattr(xgboost_adapter_module.xgb, "XGBRegressor", ConstantPsiRegressor)
    monkeypatch.setattr("src.model_trainer.save_best_params", lambda **kwargs: None)

    results = trainer.optimize_hyperparameters(
        X,
        y,
        y_report=y_report,
        cv=splitter,
        n_trials=1,
        study_name="test-study",
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
        best_params_output_path=str(tmp_path / "best_params.json"),
    )

    assert splitter.split_called_with is not None
    split_X, split_y, split_groups = splitter.split_called_with
    assert split_X.equals(X)
    assert split_y is not None
    assert split_y.equals(y)
    assert split_groups is None
    assert results["best_params"] == {"max_depth": 3}
    assert results["target_mode"] == "eta_u_over_npl"


def test_selection_objective_matches_planned_formula():
    selection_objective = _build_selection_objective_config(
        {
            "metric_space": "original_nexp",
            "rmse_normalizer": "mean_actual",
            "cov_threshold": 0.10,
            "r2_threshold": 0.99,
            "cov_weight": 2.0,
            "r2_weight": 2.0,
        }
    )
    score = _calculate_selection_objective(
        {
            "rmse": 10.0,
            "r2": 0.98,
            "cov": 0.15,
            "mean_actual": 100.0,
        },
        selection_objective,
    )

    assert score == pytest.approx(3.1)


def test_selection_objective_rejects_unsupported_metric_space():
    with pytest.raises(ValueError, match="selection_objective.metric_space"):
        _build_selection_objective_config({"metric_space": "transformed"})


def test_cross_validate_uses_requested_metric_space_for_selection(monkeypatch):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 42},
        target_mode="eta_u_over_npl",
        target_transform_type=None,
        validation_size=0.0,
    )
    splitter = RecordingSplitter()
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "Npl (kN)": [100.0, 200.0, 100.0, 200.0],
        }
    )
    y = pd.Series([1.0, 1.0, 2.0, 2.0])
    y_report = pd.Series([100.0, 200.0, 200.0, 400.0])

    monkeypatch.setattr(xgboost_adapter_module.xgb, "XGBRegressor", ConstantPsiRegressor)

    original_results = trainer.cross_validate(
        X,
        y,
        y_report=y_report,
        cv=splitter,
        metric_space="original",
    )
    transformed_results = trainer.cross_validate(
        X,
        y,
        y_report=y_report,
        cv=splitter,
        metric_space="transformed",
    )

    selection_objective = _build_selection_objective_config(
        {
            "metric_space": "original_nexp",
            "rmse_normalizer": "mean_actual",
            "cov_threshold": 0.10,
            "r2_threshold": 0.99,
            "cov_weight": 2.0,
            "r2_weight": 2.0,
        }
    )
    expected_original_scores = [
        _calculate_selection_objective(
            _calculate_regression_metrics(
                np.array([200.0, 400.0]),
                np.array([150.0, 300.0]),
            ),
            selection_objective,
        ),
        _calculate_selection_objective(
            _calculate_regression_metrics(
                np.array([100.0, 200.0]),
                np.array([150.0, 300.0]),
            ),
            selection_objective,
        ),
    ]
    expected_transformed_scores = [
        _calculate_selection_objective(
            _calculate_regression_metrics(
                np.array([2.0, 2.0]),
                np.array([1.5, 1.5]),
            ),
            selection_objective,
        ),
        _calculate_selection_objective(
            _calculate_regression_metrics(
                np.array([1.0, 1.0]),
                np.array([1.5, 1.5]),
            ),
            selection_objective,
        ),
    ]

    assert original_results["mean_cv_score"] == pytest.approx(np.mean(expected_original_scores))
    assert transformed_results["mean_cv_score"] == pytest.approx(np.mean(expected_transformed_scores))
    assert all(detail["selection_metric_space"] == "original" for detail in original_results["fold_details"])
    assert all(detail["selection_metric_space"] == "transformed" for detail in transformed_results["fold_details"])
    assert original_results["mean_cv_score"] != transformed_results["mean_cv_score"]
