from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# Load environment variables from .env if present (local dev)
load_dotenv()


def _env(key: str, default: str | None = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


@dataclass(frozen=True)
class Settings:
    # Base paths
    root_dir: Path = Path(__file__).resolve().parents[1]

    data_raw_dir: Path = Path(_env("DATA_RAW_DIR", "data/raw"))
    data_processed_dir: Path = Path(_env("DATA_PROCESSED_DIR", "data/processed"))
    models_dir: Path = Path(_env("MODELS_DIR", "models"))
    reports_dir: Path = Path(_env("REPORTS_DIR", "reports"))

    # App
    app_title: str = _env("APP_TITLE", "Customer Intelligence Dashboard")
    default_churn_window_days: int = int(_env("DEFAULT_CHURN_WINDOW_DAYS", "90"))

    # Reproducibility
    random_seed: int = int(_env("RANDOM_SEED", "42"))

    def ensure_dirs(self) -> None:
        (self.root_dir / self.data_raw_dir).mkdir(parents=True, exist_ok=True)
        (self.root_dir / self.data_processed_dir).mkdir(parents=True, exist_ok=True)
        (self.root_dir / self.models_dir).mkdir(parents=True, exist_ok=True)
        (self.root_dir / self.reports_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()