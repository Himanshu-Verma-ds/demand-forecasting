from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ..features import ml_feature_columns


@dataclass
class MLBundle:
    """Container for a trained ML pipeline and its feature definitions."""

    model_name: str
    pipeline: Any
    numeric_features: list[str]
    categorical_features: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.pipeline.predict(X), 0.0, None)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "MLBundle":
        return joblib.load(path)


def _estimator(name: str, params: dict[str, Any], n_jobs: int = -1, random_seed: int = 42):
    """Create the configured regression estimator."""
    name = name.lower()

    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression_l1",
            random_state=random_seed,
            n_jobs=n_jobs,
            verbosity=-1,
            **params,
        )

    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_seed,
            n_jobs=n_jobs,
            **params,
        )

    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            loss_function="MAE",
            random_seed=random_seed,
            verbose=False,
            thread_count=n_jobs,
            **params,
        )

    raise ValueError(f"Unknown model: {name}")


def build_ml_bundle(df: pd.DataFrame, model_name: str, params: dict[str, Any], config: dict, n_jobs: int = -1) -> MLBundle:
    """Build the preprocessing and regression pipeline using config-approved features."""
    numeric, categorical = ml_feature_columns(df, config)
    random_seed = int(config["project"]["random_seed"])

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical),
        ],
        remainder="drop",
    )

    estimator = _estimator(model_name, params, n_jobs=n_jobs, random_seed=random_seed)
    pipe = Pipeline([("preprocess", pre), ("model", estimator)])

    return MLBundle(model_name=model_name.lower(), pipeline=pipe, numeric_features=numeric, categorical_features=categorical)


def default_params(name: str) -> dict[str, Any]:
    """Return default hyperparameters for the requested ML model."""
    name = name.lower()

    return {
        "lightgbm": dict(n_estimators=700, learning_rate=0.04, num_leaves=63, max_depth=-1, subsample=0.9, colsample_bytree=0.9),
        "xgboost": dict(n_estimators=700, learning_rate=0.04, max_depth=8, min_child_weight=5, subsample=0.9, colsample_bytree=0.9),
        "catboost": dict(iterations=700, learning_rate=0.04, depth=8, l2_leaf_reg=5.0),
    }[name]