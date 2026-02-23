from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load

from src.config import settings


def load_model_artifacts() -> tuple[Any, list[str]]:
    models_dir = settings.root_dir / settings.models_dir
    model_path = models_dir / "churn_model.joblib"
    cols_path = models_dir / "churn_features.joblib"

    if not model_path.exists() or not cols_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run: python -m src.modeling.train_churn_model"
        )

    model = load(model_path)
    feature_cols = load(cols_path)

    return model, feature_cols


def predict_churn_proba(features_df: pd.DataFrame) -> pd.Series:
    """
    Return churn probability for each row in features_df.
    features_df must contain the same feature columns used in training.
    """
    model, feature_cols = load_model_artifacts()
    X = features_df[feature_cols].fillna(0)
    proba = model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=features_df.index, name="churn_probability")