import pandas as pd


def _qcut_score(series: pd.Series, q: int, labels: list[int], reverse: bool = False) -> pd.Series:
    """
    Quantile scoring that won't fail when there are duplicate bin edges.
    If qcut can't create q bins, it will drop duplicate edges and still return a score.
    """
    s = series.copy()

    # Use qcut with duplicates='drop' to avoid "Bin edges must be unique"
    buckets = pd.qcut(s, q=q, duplicates="drop")

    # Convert buckets to ordered codes: 0..k-1
    codes = buckets.cat.codes

    # If qcut produced fewer than q bins, remap codes to the requested label range
    # Example: got k=2 bins -> map [0,1] to lowest/highest labels
    k = buckets.cat.categories.size
    if k <= 0:
        # fallback: everything same bucket
        out = pd.Series([labels[-1]] * len(s), index=s.index)
        return out

    # Map codes (0..k-1) into labels length
    # We'll spread them across labels by rank
    scaled = (codes / max(k - 1, 1) * (len(labels) - 1)).round().astype(int)
    out = pd.Series([labels[i] for i in scaled], index=s.index)

    if reverse:
        # reverse scoring: high value -> low score (used for Recency)
        out = out.max() + out.min() - out

    return out.astype(int)


def assign_rfm_segments(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()

    # R: lower recency is better, so reverse=True
    df["R_score"] = _qcut_score(df["recency_days"], q=4, labels=[1, 2, 3, 4], reverse=True)

    # F: higher frequency is better, but it's very discrete in Olist (mostly 1)
    df["F_score"] = _qcut_score(df["frequency_orders"], q=4, labels=[1, 2, 3, 4], reverse=False)

    # M: higher monetary is better
    df["M_score"] = _qcut_score(df["monetary_total"], q=4, labels=[1, 2, 3, 4], reverse=False)

    df["RFM_score"] = (
        df["R_score"].astype(str)
        + df["F_score"].astype(str)
        + df["M_score"].astype(str)
    )

    return df