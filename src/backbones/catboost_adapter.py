"""CatBoost backbone adapter."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from optuna.trial import Trial

from src.backbones.registry import register_backbone_adapter


class CatBoostBackboneAdapter:
    name = "catboost"
    model_display_name = "CatBoostRegressor"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "loss_function": "RMSE",
            "iterations": 400,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": 42,
            "verbose": False,
        }

    def get_optuna_center_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "iterations": int(params.get("iterations", 400)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "depth": int(params.get("depth", 6)),
            "l2_leaf_reg": float(params.get("l2_leaf_reg", 3.0)),
        }

    def get_optuna_search_space(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        center = self.get_optuna_center_point(params)
        return {
            "iterations": {"kind": "int", "low": max(100, center["iterations"] // 2), "high": min(1500, center["iterations"] * 2)},
            "learning_rate": {"kind": "float", "low": 0.01, "high": 0.2, "log": True},
            "depth": {"kind": "int", "low": 4, "high": 10},
            "l2_leaf_reg": {"kind": "float", "low": 1.0, "high": 10.0, "log": True},
        }

    def build_optuna_trial_params(self, trial: Trial, params: Dict[str, Any]) -> Dict[str, Any]:
        search_space = self.get_optuna_search_space(params)
        return {
            "loss_function": params.get("loss_function", "RMSE"),
            "iterations": trial.suggest_int("iterations", int(search_space["iterations"]["low"]), int(search_space["iterations"]["high"])),
            "learning_rate": trial.suggest_float("learning_rate", float(search_space["learning_rate"]["low"]), float(search_space["learning_rate"]["high"]), log=True),
            "depth": trial.suggest_int("depth", int(search_space["depth"]["low"]), int(search_space["depth"]["high"])),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", float(search_space["l2_leaf_reg"]["low"]), float(search_space["l2_leaf_reg"]["high"]), log=True),
            "random_seed": params.get("random_seed", 42),
            "verbose": False,
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
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("catboost is not installed. Install with: pip install catboost") from exc

        filtered_params = params.copy()
        filtered_params.pop("n_jobs", None)

        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.to_numpy(dtype=float)
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (X_val, y_val)
            if sample_weight_eval_set is not None:
                fit_kwargs["sample_weight_eval_set"] = sample_weight_eval_set.to_numpy(dtype=float)
            if early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            if eval_metric is not None:
                fit_kwargs["verbose"] = False

        model = CatBoostRegressor(**filtered_params)
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def finalize_after_cv(
        self,
        params: Dict[str, Any],
        cv_results: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[int]]:
        del cv_results
        return params.copy(), []


register_backbone_adapter("catboost", CatBoostBackboneAdapter)
