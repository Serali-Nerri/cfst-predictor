"""MLP backbone adapter."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from optuna.trial import Trial
from sklearn.neural_network import MLPRegressor

from src.backbones.registry import register_backbone_adapter


class MLPBackboneAdapter:
    name = "mlp"
    model_display_name = "MLPRegressor"

    def _normalize_fit_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = params.copy()
        normalized.pop("n_jobs", None)
        hidden_layer_sizes = normalized.get("hidden_layer_sizes")
        if isinstance(hidden_layer_sizes, (list, tuple)) and len(hidden_layer_sizes) > 0:
            first_layer = int(hidden_layer_sizes[0])
            second_layer = int(hidden_layer_sizes[1]) if len(hidden_layer_sizes) > 1 else 64
        else:
            first_layer = normalized.pop("first_layer", None)
            second_layer = normalized.pop("second_layer", None)
        if first_layer is not None:
            normalized["hidden_layer_sizes"] = (
                int(first_layer),
                int(second_layer) if second_layer is not None else 64,
            )
        return normalized

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "alpha": 1e-4,
            "learning_rate_init": 1e-3,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "random_state": 42,
        }

    def get_optuna_center_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_fit_params(params)
        hidden_layer_sizes = normalized.get("hidden_layer_sizes", (128, 64))
        first_layer = int(hidden_layer_sizes[0])
        second_layer = int(hidden_layer_sizes[1]) if len(hidden_layer_sizes) > 1 else 64
        return {
            "first_layer": first_layer,
            "second_layer": second_layer,
            "alpha": float(params.get("alpha", 1e-4)),
            "learning_rate_init": float(params.get("learning_rate_init", 1e-3)),
            "max_iter": int(params.get("max_iter", 500)),
        }

    def get_optuna_search_space(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        center = self.get_optuna_center_point(params)
        return {
            "first_layer": {"kind": "int", "low": 32, "high": 256},
            "second_layer": {"kind": "int", "low": 16, "high": 128},
            "alpha": {"kind": "float", "low": 1e-6, "high": 1e-2, "log": True},
            "learning_rate_init": {"kind": "float", "low": 1e-4, "high": 5e-3, "log": True},
            "max_iter": {
                "kind": "int",
                "low": max(200, int(round(center["max_iter"] * 0.5))),
                "high": min(1000, int(round(center["max_iter"] * 1.5))),
            },
        }

    def build_optuna_trial_params(self, trial: Trial, params: Dict[str, Any]) -> Dict[str, Any]:
        search_space = self.get_optuna_search_space(params)
        first_layer = trial.suggest_int("first_layer", int(search_space["first_layer"]["low"]), int(search_space["first_layer"]["high"]))
        second_layer = trial.suggest_int("second_layer", int(search_space["second_layer"]["low"]), int(search_space["second_layer"]["high"]))
        alpha = trial.suggest_float("alpha", float(search_space["alpha"]["low"]), float(search_space["alpha"]["high"]), log=True)
        learning_rate_init = trial.suggest_float(
            "learning_rate_init",
            float(search_space["learning_rate_init"]["low"]),
            float(search_space["learning_rate_init"]["high"]),
            log=True,
        )
        max_iter = trial.suggest_int("max_iter", int(search_space["max_iter"]["low"]), int(search_space["max_iter"]["high"]))
        return {
            "hidden_layer_sizes": (first_layer, second_layer),
            "activation": params.get("activation", "relu"),
            "alpha": alpha,
            "learning_rate_init": learning_rate_init,
            "max_iter": max_iter,
            "early_stopping": bool(params.get("early_stopping", True)),
            "validation_fraction": float(params.get("validation_fraction", 0.15)),
            "random_state": params.get("random_state", 42),
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
        filtered_params = self._normalize_fit_params(params)
        model = MLPRegressor(**filtered_params)
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


register_backbone_adapter("mlp", MLPBackboneAdapter)
register_backbone_adapter("mlp_regressor", MLPBackboneAdapter)
