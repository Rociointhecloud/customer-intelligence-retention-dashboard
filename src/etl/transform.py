import pandas as pd

from src.utils.validation import require_columns


def build_transaction_table(data: dict) -> pd.DataFrame:
    orders = data["orders"]
    order_items = data["order_items"]
    customers = data["customers"]
    payments = data["payments"]
    reviews = data["reviews"]

    # --- Schema validation (fail fast) ---
    require_columns(
        orders,
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
        ],
        "orders",
    )
    require_columns(order_items, ["order_id", "price", "freight_value"], "order_items")
    require_columns(customers, ["customer_id", "customer_unique_id"], "customers")
    require_columns(payments, ["order_id", "payment_value"], "payments")
    require_columns(reviews, ["order_id", "review_score"], "reviews")

    # --- Dates ---
    orders = orders.copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"], errors="coerce"
    )
    orders["order_delivered_customer_date"] = pd.to_datetime(
        orders["order_delivered_customer_date"], errors="coerce"
    )

    # --- Aggregate order_items to order level ---
    order_items_agg = (
        order_items.groupby("order_id", as_index=False)
        .agg(
            revenue=("price", "sum"),
            freight_value=("freight_value", "sum"),
        )
    )

    # --- Aggregate payments to order level ---
    payments_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            total_payment=("payment_value", "sum"),
        )
    )

    # --- Keep only needed review columns (order-level) ---
    reviews_small = reviews[["order_id", "review_score"]].copy()

    # --- Merge datasets ---
    df = orders.merge(order_items_agg, on="order_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(reviews_small, on="order_id", how="left")
    df = df.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    # --- Delivery time ---
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    return df