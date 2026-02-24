from __future__ import annotations

import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import settings


def main() -> None:
    processed_dir = settings.root_dir / settings.data_processed_dir
    features_path = processed_dir / "customer_features.csv"

    print(f"[model] Loading features from: {features_path}")
    df = pd.read_csv(features_path)

    churn_col = [c for c in df.columns if c.startswith("churn_")][0]

    feature_cols = [
        # "recency_days",  # removed to prevent target leakage
        "frequency_orders",
        "monetary_total",
        "avg_order_value",
        "avg_review_score",
        "avg_delivery_days",
    ]

    # Safety warning (no crash)
    if "recency_days" in feature_cols and churn_col.startswith("churn_"):
        print("[warning] recency_days may leak target definition. Consider removing it.")

    X = df[feature_cols].fillna(0)
    y = df[churn_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=settings.random_seed,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=settings.random_seed,
    )

    print("[model] Training RandomForest...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    importances = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )

    print("\nFeature Importances:")
    print(importances)

    # =========================
    # Save model
    # =========================

    models_dir = settings.root_dir / settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    dump(model, models_dir / "churn_model.joblib")
    dump(feature_cols, models_dir / "churn_features.joblib")

    print(f"[model] Saved model to: {models_dir / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()