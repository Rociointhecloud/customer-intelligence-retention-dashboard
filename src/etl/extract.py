from pathlib import Path
import pandas as pd

from src.config import settings


def load_csv(filename: str) -> pd.DataFrame:
    path = settings.root_dir / settings.data_raw_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"[extract] Loading {filename}...")
    df = pd.read_csv(path)
    print(f"[extract] {filename} loaded with shape {df.shape}")
    return df


def load_all_raw_data() -> dict:
    data = {
        "orders": load_csv("olist_orders_dataset.csv"),
        "order_items": load_csv("olist_order_items_dataset.csv"),
        "customers": load_csv("olist_customers_dataset.csv"),
        "payments": load_csv("olist_order_payments_dataset.csv"),
        "reviews": load_csv("olist_order_reviews_dataset.csv"),
        "products": load_csv("olist_products_dataset.csv"),
        "category_translation": load_csv("product_category_name_translation.csv"),
    }

    return data