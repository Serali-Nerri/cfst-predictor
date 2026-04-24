"""Random forest backbone adapter."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.backbones.base import OptunaTrialProtocol
from src.backbones.registry import register_backbone_adapter


class RandomForestBackboneAdapter:
    name = "rf"
    model_display_name = "RandomForestRegressor"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "bootstrap": True,
            "random_state": 42,
            "n_jobs": -1,
        }

    def get_optuna_center_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "n_estimators": int(params.get("n_estimators", 300)),
            "max_depth": params.get("max_depth", 12),
            "min_samples_split": int(params.get("min_samples_split", 2)),
            "min_samples_leaf": int(params.get("min_samples_leaf", 1)),
            "max_features": float(params.get("max_features", 1.0)),
        }

    def get_optuna_search_space(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        center = self.get_optuna_center_point(params)
        max_depth = center["max_depth"] if center["max_depth"] is not None else 12
        return {
            "n_estimators": {
                "kind": "int",
                "low": max(100, int(round(center["n_estimators"] * 0.5))),
                "high": min(1200, int(round(center["n_estimators"] * 2.0))),
            },
            "max_depth": {
                "kind": "int",
                "low": max(4, int(max_depth) - 4),
                "high": min(32, int(max_depth) + 6),
            },
            "min_samples_split": {
                "kind": "int",
                "low": 2,
                "high": 12,
            },
            "min_samples_leaf": {
                "kind": "int",
                "low": 1,
                "high": 8,
            },
            "max_features": {
                "kind": "float",
                "low": 0.4,
                "high": 1.0,
            },
        }

    def build_optuna_trial_params(self, trial: OptunaTrialProtocol, params: Dict[str, Any]) -> Dict[str, Any]:
        search_space = self.get_optuna_search_space(params)
        tuned_params: Dict[str, Any] = {}
        for name, spec in search_space.items():
            if spec["kind"] == "int":
                tuned_params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
            else:
                tuned_params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]))

        return {
            **tuned_params,
            "bootstrap": bool(params.get("bootstrap", True)),
            "random_state": params.get("random_state", 42),
            "n_jobs": params.get("n_jobs", -1),
        }

    def fit(
        self,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[str] = None,
        sample_weight: Optional[pd.Series] = None,
        sample_weight_eval_set: Optional[pd.Series] = None,
    ) -> Any:
        del X_val, y_val, early_stopping_rounds, eval_metric, sample_weight_eval_set
        model = RandomForestRegressor(**params)
        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.to_numpy(dtype=float)
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def finalize_after_cv(
        self,
        params: Dict[str, Any],
        cv_results: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[int]]:
        del cv_results
        return params.copy(), []


register_backbone_adapter("rf", RandomForestBackboneAdapter)
register_backbone_adapter("random_forest", RandomForestBackboneAdapter)
