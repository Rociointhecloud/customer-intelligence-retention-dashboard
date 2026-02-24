from __future__ import annotations

from typing import Iterable
import pandas as pd


def require_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[schema] {name} missing columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )