import pandas as pd


def build_transaction_table(data: dict) -> pd.DataFrame:
    orders = data["orders"]
    order_items = data["order_items"]
    customers = data["customers"]
    payments = data["payments"]
    reviews = data["reviews"]

    # Convert date columns
    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )
    orders["order_delivered_customer_date"] = pd.to_datetime(
        orders["order_delivered_customer_date"]
    )

    # Aggregate order_items to order level (revenue + freight)
    order_items_agg = (
        order_items.groupby("order_id")
        .agg(
            revenue=("price", "sum"),
            freight_value=("freight_value", "sum"),
        )
        .reset_index()
    )

    # Aggregate payments to order level
    payments_agg = (
        payments.groupby("order_id")
        .agg(
            total_payment=("payment_value", "sum"),
        )
        .reset_index()
    )

    # Merge datasets
    df = orders.merge(order_items_agg, on="order_id", how="left")
    df = df.merge(payments_agg, on="order_id", how="left")
    df = df.merge(reviews[["order_id", "review_score"]], on="order_id", how="left")
    df = df.merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )

    # Delivery time
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    return df