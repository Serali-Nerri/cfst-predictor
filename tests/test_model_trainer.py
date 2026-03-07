import sys
import types

import numpy as np
import pandas as pd

import src.model_trainer as model_trainer_module
from src.model_trainer import ModelTrainer


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

    def optimize(self, objective, n_trials, timeout):
        trial = DummyTrial()
        self.best_value = objective(trial)
        self.trials = [trial]


def test_cross_validate_passes_splitter_through(monkeypatch):
    trainer = ModelTrainer(params={"device": "cpu", "n_jobs": -1})
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    splitter = RecordingSplitter()
    captured = {}

    def fake_cross_val_score(model, features, target, cv, scoring, n_jobs, verbose):
        captured["cv"] = cv
        captured["features"] = features
        captured["target"] = target
        return np.array([-1.0, -2.0])

    monkeypatch.setattr(model_trainer_module, "cross_val_score", fake_cross_val_score)

    results = trainer.cross_validate(X, y, cv=splitter)

    assert captured["cv"] is splitter
    assert captured["features"].equals(X)
    assert captured["target"].equals(y)
    assert results["n_folds"] == 2


def test_optimize_hyperparameters_uses_provided_splitter(monkeypatch, tmp_path):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 42},
        use_optuna=True,
        n_trials=1,
        optuna_timeout=1,
    )
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    splitter = RecordingSplitter()

    fake_optuna = types.ModuleType("optuna")
    setattr(fake_optuna, "create_study", lambda **kwargs: DummyStudy())
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    class DummyRegressor:
        def __init__(self, **kwargs):
            self.mean_ = 0.0

        def fit(self, X_train, y_train, verbose=False):
            self.mean_ = float(np.mean(y_train.to_numpy(dtype=float)))
            return self

        def predict(self, X_val):
            return np.full(len(X_val), self.mean_, dtype=float)

    monkeypatch.setattr(model_trainer_module.xgb, "XGBRegressor", DummyRegressor)
    monkeypatch.setattr(model_trainer_module, "save_best_params", lambda **kwargs: None)

    results = trainer.optimize_hyperparameters(
        X,
        y,
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


def test_optimize_hyperparameters_keeps_integer_cv_fallback(monkeypatch, tmp_path):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 99},
        use_optuna=True,
        n_trials=1,
        optuna_timeout=1,
    )
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    captured = {}

    fake_optuna = types.ModuleType("optuna")
    setattr(fake_optuna, "create_study", lambda **kwargs: DummyStudy())
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    original_kfold = model_trainer_module.KFold

    class RecordingKFold(original_kfold):
        def __init__(self, *args, **kwargs):
            captured["n_splits"] = kwargs.get("n_splits")
            captured["shuffle"] = kwargs.get("shuffle")
            captured["random_state"] = kwargs.get("random_state")
            super().__init__(*args, **kwargs)

    class DummyRegressor:
        def __init__(self, **kwargs):
            self.mean_ = 0.0

        def fit(self, X_train, y_train, verbose=False):
            self.mean_ = float(np.mean(y_train.to_numpy(dtype=float)))
            return self

        def predict(self, X_val):
            return np.full(len(X_val), self.mean_, dtype=float)

    monkeypatch.setattr(model_trainer_module, "KFold", RecordingKFold)
    monkeypatch.setattr(model_trainer_module.xgb, "XGBRegressor", DummyRegressor)
    monkeypatch.setattr(model_trainer_module, "save_best_params", lambda **kwargs: None)

    trainer.optimize_hyperparameters(
        X,
        y,
        cv=3,
        n_trials=1,
        study_name="test-study",
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
        best_params_output_path=str(tmp_path / "best_params.json"),
    )

    assert captured == {"n_splits": 3, "shuffle": True, "random_state": 99}
