"""LightGBM backbone adapter."""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from optuna.trial import Trial

from src.backbones.registry import register_backbone_adapter


class LightGBMBackboneAdapter:
    name = "lightgbm"
    model_display_name = "LGBMRegressor"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "objective": "regression",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

    def get_optuna_center_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "n_estimators": int(params.get("n_estimators", 400)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "num_leaves": int(params.get("num_leaves", 31)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        }

    def get_optuna_search_space(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        center = self.get_optuna_center_point(params)
        return {
            "n_estimators": {"kind": "int", "low": max(100, center["n_estimators"] // 2), "high": min(1500, center["n_estimators"] * 2)},
            "learning_rate": {"kind": "float", "low": 0.01, "high": 0.2, "log": True},
            "num_leaves": {"kind": "int", "low": 15, "high": 127},
            "subsample": {"kind": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"kind": "float", "low": 0.5, "high": 1.0},
        }

    def build_optuna_trial_params(self, trial: Trial, params: Dict[str, Any]) -> Dict[str, Any]:
        search_space = self.get_optuna_search_space(params)
        return {
            "objective": params.get("objective", "regression"),
            "n_estimators": trial.suggest_int("n_estimators", int(search_space["n_estimators"]["low"]), int(search_space["n_estimators"]["high"])),
            "learning_rate": trial.suggest_float("learning_rate", float(search_space["learning_rate"]["low"]), float(search_space["learning_rate"]["high"]), log=True),
            "num_leaves": trial.suggest_int("num_leaves", int(search_space["num_leaves"]["low"]), int(search_space["num_leaves"]["high"])),
            "subsample": trial.suggest_float("subsample", float(search_space["subsample"]["low"]), float(search_space["subsample"]["high"])),
            "colsample_bytree": trial.suggest_float("colsample_bytree", float(search_space["colsample_bytree"]["low"]), float(search_space["colsample_bytree"]["high"])),
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
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is not installed. Install with: pip install lightgbm") from exc

        fit_kwargs: Dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight.to_numpy(dtype=float)
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if sample_weight_eval_set is not None:
                fit_kwargs["eval_sample_weight"] = [sample_weight_eval_set.to_numpy(dtype=float)]
            if eval_metric is not None:
                fit_kwargs["eval_metric"] = eval_metric
            if early_stopping_rounds is not None:
                fit_kwargs["callbacks"] = []
                try:
                    from lightgbm import early_stopping
                    fit_kwargs["callbacks"].append(early_stopping(early_stopping_rounds, verbose=False))
                except ImportError:
                    pass

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def finalize_after_cv(
        self,
        params: Dict[str, Any],
        cv_results: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[int]]:
        del cv_results
        return params.copy(), []


register_backbone_adapter("lightgbm", LightGBMBackboneAdapter)
register_backbone_adapter("lgbm", LightGBMBackboneAdapter)
