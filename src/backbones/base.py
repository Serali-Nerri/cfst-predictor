"""Backbone adapter contracts for model training."""

from typing import Any, Dict, List, Optional, Protocol, Tuple

import pandas as pd
from optuna.trial import Trial


class BackboneAdapter(Protocol):
    """Contract for trainable model backbones."""

    name: str
    model_display_name: str

    def get_default_params(self) -> Dict[str, Any]:
        ...

    def get_optuna_center_point(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def get_optuna_search_space(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        ...

    def build_optuna_trial_params(
        self,
        trial: Trial,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...

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
        ...

    def finalize_after_cv(
        self,
        params: Dict[str, Any],
        cv_results: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[int]]:
        ...
