import pandas as pd


def build_customer_features(transactions: pd.DataFrame, churn_window_days: int = 90) -> pd.DataFrame:
    """
    Build customer-level features for segmentation (RFM) and churn modeling.

    Churn proxy:
      churn_{window}d = True if customer has not purchased in the last `churn_window_days`
      relative to the dataset snapshot date (max purchase timestamp).
    """
    tx = transactions.copy()

    tx["order_purchase_timestamp"] = pd.to_datetime(
        tx["order_purchase_timestamp"], errors="coerce"
    )

    snapshot_date = tx["order_purchase_timestamp"].max()

    features = (
        tx.groupby("customer_unique_id")
        .agg(
            last_purchase=("order_purchase_timestamp", "max"),
            frequency_orders=("order_id", "nunique"),
            monetary_total=("revenue", "sum"),
            avg_review_score=("review_score", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
        )
        .reset_index()
    )

    features["recency_days"] = (snapshot_date - features["last_purchase"]).dt.days
    features["avg_order_value"] = features["monetary_total"] / features["frequency_orders"]

    features[f"churn_{churn_window_days}d"] = features["recency_days"] > churn_window_days

    cols = [
        "customer_unique_id",
        "recency_days",
        "frequency_orders",
        "monetary_total",
        "avg_order_value",
        "avg_review_score",
        "avg_delivery_days",
        "last_purchase",
        f"churn_{churn_window_days}d",
    ]
    return features[cols]
