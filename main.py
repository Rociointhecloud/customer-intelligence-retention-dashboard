from __future__ import annotations

from pathlib import Path

from src.config import settings


def _has_raw_files(raw_dir: Path) -> bool:
    if not raw_dir.exists():
        return False
    files = [p for p in raw_dir.iterdir() if p.is_file() and p.name.lower() != "readme.md"]
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
            "1) Put your dataset file(s) inside 'data/raw/' (e.g., transactions.csv)\n"
            "2) Re-run: python main.py\n"
        )
        return

    # Next step: ETL pipeline entrypoint (we will implement this in src/etl)
    print("\n[next] Raw data found. ETL pipeline will run here (to be implemented).")


if __name__ == "__main__":
    main()