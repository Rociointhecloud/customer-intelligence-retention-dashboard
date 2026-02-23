from __future__ import annotations

from pathlib import Path

from src.analysis.segmentation import assign_rfm_segments
from src.config import settings
from src.etl.extract import load_all_raw_data
from src.etl.transform import build_transaction_table
from src.modeling.features import build_customer_features


def _has_raw_files(raw_dir: Path) -> bool:
    if not raw_dir.exists():
        return False
    files = [
        p for p in raw_dir.iterdir()
        if p.is_file() and p.name.lower() != "readme.md"
    ]
    return len(files) > 0


def main() -> None:
    settings.ensure_dirs()

    raw_dir = settings.root_dir / settings.data_raw_dir
    processed_dir = settings.root_dir / settings.data_processed_dir

    print(f"[info] App: {settings.app_title}")
    print(f"[info] Raw data dir: {raw_dir}")
    print(f"[info] Processed data dir: {processed_dir}")

    if not _has_raw_files(raw_dir):
        print(
            "\n[action] No raw data files found.\n"
            "1) Put your dataset file(s) inside 'data/raw/'\n"
            "2) Re-run: python main.py\n"
        )
        return

    print("\n[etl] Starting extraction...")
    data = load_all_raw_data()
    print(f"[etl] Loaded datasets: {list(data.keys())}")

    print("\n[etl] Building transaction table...")
    transactions = build_transaction_table(data)
    print(f"[etl] Transaction table shape: {transactions.shape}")

    out_path = processed_dir / "transactions.csv"
    transactions.to_csv(out_path, index=False)
    print(f"[etl] Saved processed transactions to: {out_path}")

    print("\n[model] Building customer features...")
    customer_features = build_customer_features(
        transactions,
        churn_window_days=settings.default_churn_window_days,
    )

    features_path = processed_dir / "customer_features.csv"
    customer_features.to_csv(features_path, index=False)
    print(f"[model] Customer features shape: {customer_features.shape}")
    print(f"[model] Saved to: {features_path}")

    print("\n[analysis] Assigning RFM segments...")
    segmented = assign_rfm_segments(customer_features)

    segments_path = processed_dir / "customer_segments.csv"
    segmented.to_csv(segments_path, index=False)

    print(f"[analysis] Segmented dataset shape: {segmented.shape}")
    print(f"[analysis] Saved to: {segments_path}")


if __name__ == "__main__":
    main()