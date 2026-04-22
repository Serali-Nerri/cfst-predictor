"""XGBoost backbone adapter."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from optuna.trial import Trial

from src.backbones.registry import register_backbone_adapter


class XGBoostBackboneAdapter:
    """Adapter that encapsulates current XGBoost training behavior."""

    name = "xgboost"
    model_display_name = "XGBRegressor"

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": -1,
        }

    def get_optuna_center_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "max_depth": int(params.get("max_depth", 5)),
            "learning_rate": float(params.get("learning_rate", 0.05)),
            "n_estimators": int(params.get("n_estimators", 1200)),
            "subsample": float(params.get("subsample", 0.8)),
            "colsample_bytree": float(params.get("colsample_bytree", 0.75)),
            "min_child_weight": int(params.get("min_child_weight", 10)),
            "reg_alpha": float(params.get("reg_alpha", 0.5)),
            "reg_lambda": float(params.get("reg_lambda", 2.0)),
            "gamma": float(params.get("gamma", 0.05)),
        }

    def get_optuna_search_space(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        center = self.get_optuna_center_point(params)
        return {
            "max_depth": {
                "kind": "int",
                "low": max(3, center["max_depth"] - 1),
                "high": min(8, center["max_depth"] + 2),
            },
            "learning_rate": {
                "kind": "float",
                "low": max(0.01, center["learning_rate"] / 2.5),
                "high": min(0.12, max(center["learning_rate"] * 1.8, 0.02)),
                "log": True,
            },
            "n_estimators": {
                "kind": "int",
                "low": max(800, int(round(center["n_estimators"] * 0.6))),
                "high": min(4500, int(round(center["n_estimators"] * 2.2))),
            },
            "subsample": {
                "kind": "float",
                "low": max(0.55, center["subsample"] - 0.18),
                "high": min(0.98, center["subsample"] + 0.08),
            },
            "colsample_bytree": {
                "kind": "float",
                "low": max(0.4, center["colsample_bytree"] - 0.18),
                "high": min(0.95, center["colsample_bytree"] + 0.12),
            },
            "min_child_weight": {
                "kind": "int",
                "low": max(4, int(round(center["min_child_weight"] * 0.6))),
                "high": min(40, int(round(center["min_child_weight"] * 2.0))),
            },
            "reg_alpha": {
                "kind": "float",
                "low": max(1e-3, center["reg_alpha"] / 20.0),
                "high": min(20.0, max(center["reg_alpha"] * 20.0, 1e-2)),
                "log": True,
            },
            "reg_lambda": {
                "kind": "float",
                "low": max(1e-2, center["reg_lambda"] / 15.0),
                "high": min(50.0, max(center["reg_lambda"] * 15.0, 0.1)),
                "log": True,
            },
            "gamma": {
                "kind": "float",
                "low": max(1e-4, center["gamma"] / 30.0),
                "high": min(5.0, max(center["gamma"] * 20.0, 1e-3)),
                "log": True,
            },
        }

    def build_optuna_trial_params(self, trial: Trial, params: Dict[str, Any]) -> Dict[str, Any]:
        search_space = self.get_optuna_search_space(params)
        tuned_params: Dict[str, Any] = {}

        for name, spec in search_space.items():
            if spec["kind"] == "int":
                tuned_params[name] = trial.suggest_int(
                    name, int(spec["low"]), int(spec["high"])
                )
            else:
                tuned_params[name] = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False)),
                )

        return {
            "objective": params.get("objective", "reg:squarederror"),
            **tuned_params,
            "random_state": params.get("random_state", 42),
            "tree_method": params.get("tree_method", "hist"),
            "device": params.get("device", "cpu"),
            "n_jobs": params.get("n_jobs", -1),
        }

    @staticmethod
    def _to_float_numpy(series: Optional[pd.Series]) -> Optional[np.ndarray]:
        if series is None:
            return None
        return series.to_numpy(dtype=float)

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
        model_params = params.copy()
        fit_kwargs: Dict[str, Any] = {"verbose": False}

        sample_weight_numpy = self._to_float_numpy(sample_weight)
        sample_weight_eval_numpy = self._to_float_numpy(sample_weight_eval_set)

        if sample_weight_numpy is not None:
            fit_kwargs["sample_weight"] = sample_weight_numpy

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if sample_weight_eval_numpy is not None:
                fit_kwargs["sample_weight_eval_set"] = [sample_weight_eval_numpy]
            if early_stopping_rounds is not None:
                model_params["early_stopping_rounds"] = early_stopping_rounds
        else:
            model_params.pop("early_stopping_rounds", None)

        if eval_metric is not None:
            model_params["eval_metric"] = eval_metric
        else:
            model_params.pop("eval_metric", None)

        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def finalize_after_cv(
        self,
        params: Dict[str, Any],
        cv_results: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[int]]:
        finalized_params = params.copy()
        fold_details = cv_results.get("fold_details", [])
        best_iterations = [
            int(detail["best_iteration"]) + 1
            for detail in fold_details
            if detail.get("best_iteration") is not None
        ]
        if not best_iterations:
            return finalized_params, []

        finalized_params["n_estimators"] = int(np.median(np.asarray(best_iterations, dtype=int)))
        return finalized_params, best_iterations


register_backbone_adapter("xgboost", XGBoostBackboneAdapter)
register_backbone_adapter("xgb", XGBoostBackboneAdapter)
