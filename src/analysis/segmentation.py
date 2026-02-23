import pandas as pd


def _qcut_score(series: pd.Series, q: int, labels: list[int], reverse: bool = False) -> pd.Series:
    """
    Quantile scoring that won't fail when there are duplicate bin edges.
    If qcut can't create q bins, it will drop duplicate edges and still return a score.
    """
    s = series.copy()

    buckets = pd.qcut(s, q=q, duplicates="drop")
    codes = buckets.cat.codes

    k = buckets.cat.categories.size
    if k <= 0:
        return pd.Series([labels[-1]] * len(s), index=s.index)

    scaled = (codes / max(k - 1, 1) * (len(labels) - 1)).round().astype(int)
    out = pd.Series([labels[i] for i in scaled], index=s.index)

    if reverse:
        out = out.max() + out.min() - out

    return out.astype(int)


def assign_rfm_segments(features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()

    # Recency → lower is better
    df["R_score"] = _qcut_score(
        df["recency_days"],
        q=4,
        labels=[1, 2, 3, 4],
        reverse=True,
    )

    # Frequency → higher is better
    df["F_score"] = _qcut_score(
        df["frequency_orders"],
        q=4,
        labels=[1, 2, 3, 4],
        reverse=False,
    )

    # Monetary → higher is better
    df["M_score"] = _qcut_score(
        df["monetary_total"],
        q=4,
        labels=[1, 2, 3, 4],
        reverse=False,
    )

    df["RFM_score"] = (
        df["R_score"].astype(str)
        + df["F_score"].astype(str)
        + df["M_score"].astype(str)
    )

    def label_segment(r: int, f: int, m: int) -> str:
        if r >= 4 and f >= 3 and m >= 3:
            return "Champions"
        if r >= 3 and f >= 3 and m >= 2:
            return "Loyal"
        if r >= 4 and f <= 2:
            return "New Customers"
        if r == 3 and f <= 2 and m <= 2:
            return "Need Attention"
        if r <= 2 and f >= 2 and m >= 2:
            return "At Risk"
        if r <= 2 and f == 1:
            return "Hibernating"
        return "Regular"

    df["segment_name"] = df.apply(
        lambda x: label_segment(x["R_score"], x["F_score"], x["M_score"]),
        axis=1,
    )

    return df