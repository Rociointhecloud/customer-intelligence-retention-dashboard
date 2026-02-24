from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import settings
from src.modeling.inference import predict_churn_proba


ROOT = settings.root_dir
PROCESSED_DIR = ROOT / settings.data_processed_dir


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------
def money(x: float) -> str:
    return f"{x:,.0f}"


def pct(x: float) -> str:
    return f"{x:.2f}%"


def last_updated(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def risk_label(p: float) -> tuple[str, str]:
    """
    Returns (label, emoji)
    """
    if p >= 0.75:
        return "High", "🔴"
    if p >= 0.45:
        return "Medium", "🟠"
    return "Low", "🟢"


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    seg_path = PROCESSED_DIR / "customer_segments.csv"
    tx_path = PROCESSED_DIR / "transactions.csv"

    if not seg_path.exists() or not tx_path.exists():
        st.error("Processed files not found. Run: python main.py")
        st.stop()

    segments = pd.read_csv(seg_path)
    tx = pd.read_csv(tx_path)

    tx["order_purchase_timestamp"] = pd.to_datetime(
        tx["order_purchase_timestamp"], errors="coerce"
    )

    return segments, tx


def main() -> None:
    st.title("📊 Customer Intelligence & Retention Dashboard")
    st.caption("Olist e-commerce (2016–2018) · Segmentation + Churn prediction (proxy label)")

    with st.spinner("Loading data..."):
        segments, tx = load_data()

    churn_col = [c for c in segments.columns if c.startswith("churn_")][0]

    # Coverage
    min_date = tx["order_purchase_timestamp"].min()
    max_date = tx["order_purchase_timestamp"].max()
    days_span = int((max_date - min_date).days)

    # Sidebar
    st.sidebar.header("Filters")
    segment_options = ["All"] + sorted(segments["segment_name"].dropna().unique().tolist())
    selected_segment = st.sidebar.selectbox("Customer segment", segment_options)

    date_range = st.sidebar.date_input(
        "Purchase date range",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    if selected_segment != "All":
        segments_filtered = segments[segments["segment_name"] == selected_segment].copy()
    else:
        segments_filtered = segments.copy()

    tx_filtered = tx[
        (tx["order_purchase_timestamp"].dt.date >= date_range[0])
        & (tx["order_purchase_timestamp"].dt.date <= date_range[1])
    ].copy()

    # KPIs (simple and readable)
    total_customers = int(segments_filtered.shape[0])
    total_orders = int(tx_filtered["order_id"].nunique())
    total_revenue = float(segments_filtered["monetary_total"].sum())
    churn_rate = float(segments_filtered[churn_col].mean() * 100)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{total_customers:,}")
    k2.metric("Delivered orders", f"{total_orders:,}")
    k3.metric("Revenue", money(total_revenue))
    k4.metric(f"Churn (proxy {churn_col.replace('churn_', '')})", pct(churn_rate))

    st.caption(
        f"Coverage: {min_date.date()} → {max_date.date()} "
        f"({days_span} days) · Data updated: {last_updated(PROCESSED_DIR / 'customer_segments.csv')}"
    )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Executive", "Segments", "Predict", "Method"]
    )

    # -----------------------------
    # Tab 1: Executive
    # -----------------------------
    with tab1:
        st.subheader("Executive summary (non-technical)")

        st.markdown(
            """
This dashboard answers 3 business questions:

1) **Who are our customers today?** (segments)  
2) **Where is revenue coming from?** (impact per segment)  
3) **Who is likely to stop buying?** (risk prediction)
            """
        )

        rev_by_seg = (
            segments_filtered.groupby("segment_name")["monetary_total"]
            .sum()
            .sort_values(ascending=False)
        )
        rev_share = (rev_by_seg / rev_by_seg.sum() * 100).round(2)

        top_seg = rev_by_seg.index[0] if len(rev_by_seg) else "N/A"
        st.info(
            f"Highest revenue segment: **{top_seg}** · "
            f"Revenue share: **{rev_share.iloc[0]:.2f}%**"
            if len(rev_share)
            else "Not enough data for revenue share."
        )

        fig = px.bar(
            rev_by_seg.reset_index(),
            x="segment_name",
            y="monetary_total",
            title="Revenue by segment",
            labels={"segment_name": "Segment", "monetary_total": "Revenue"},
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recommended actions")
        st.markdown(
            """
- **Inactives / Hibernating:** win-back with limited incentive + highlight fast-delivery products  
- **New Customers:** onboarding journey + second purchase trigger in 7–30 days  
- **Need Attention:** reduce friction (delivery / support), personalized reminders  
            """
        )

    # -----------------------------
    # Tab 2: Segments
    # -----------------------------
    with tab2:
        st.subheader("Segment distribution")

        seg_counts = segments_filtered["segment_name"].value_counts().reset_index()
        seg_counts.columns = ["segment_name", "customers"]

        fig = px.bar(
            seg_counts,
            x="segment_name",
            y="customers",
            title="Customers by segment",
            labels={"segment_name": "Segment", "customers": "Customers"},
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Segment table")
        summary = (
            segments_filtered.groupby("segment_name")
            .agg(
                customers=("customer_unique_id", "count"),
                revenue=("monetary_total", "sum"),
                churn_rate=(churn_col, "mean"),
            )
            .sort_values("revenue", ascending=False)
        )
        summary["churn_rate"] = (summary["churn_rate"] * 100).round(2)

        st.dataframe(summary, use_container_width=True)

        st.download_button(
            "Download segment summary (CSV)",
            data=summary.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="segment_summary.csv",
            mime="text/csv",
        )

    # -----------------------------
    # Tab 3: Predict
    # -----------------------------
    with tab3:
        st.subheader("Churn risk prediction (interactive)")

        st.write(
            "This is a **risk score** (0–100%). "
            "It helps prioritize outreach. It is not a guarantee."
        )

        # Predict probabilities for current filtered set
        try:
            segments_scored = segments_filtered.copy()
            segments_scored["churn_probability"] = predict_churn_proba(segments_scored)

            avg_risk = float(segments_scored["churn_probability"].mean() * 100)
            label, emoji = risk_label(avg_risk / 100)

            c1, c2 = st.columns(2)
            c1.metric("Average churn risk", pct(avg_risk))
            c2.metric("Risk level", f"{emoji} {label}")

            st.divider()

            # Distribution chart
            fig = px.histogram(
                segments_scored,
                x="churn_probability",
                nbins=20,
                title="Distribution of churn risk",
                labels={"churn_probability": "Churn risk (0–1)"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top customers to prioritize (sample)")
            top_n = st.slider("How many customers?", 10, 200, 50, step=10)
            top = segments_scored.sort_values("churn_probability", ascending=False).head(top_n)

            display_cols = [
                "customer_unique_id",
                "segment_name",
                "monetary_total",
                "avg_order_value",
                "avg_review_score",
                "avg_delivery_days",
                "churn_probability",
            ]
            top_display = top[display_cols].copy()
            top_display["churn_probability"] = (top_display["churn_probability"] * 100).round(2)

            st.dataframe(top_display, use_container_width=True)

            st.download_button(
                "Download prioritized customers (CSV)",
                data=top_display.to_csv(index=False).encode("utf-8"),
                file_name="priority_customers.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(
                "Model artifacts not found or failed to load.\n\n"
                "Run: python -m src.modeling.train_churn_model\n\n"
                f"Details: {e}"
            )

    # -----------------------------
    # Tab 4: Method
    # -----------------------------
    with tab4:
        st.subheader("Methodology (portfolio transparency)")

        st.markdown(
            f"""
**Dataset coverage:** {min_date.date()} → {max_date.date()} ({days_span} days)

**Churn label used for training (proxy):** `{churn_col} = recency_days > window`  
This is based on the last available date in the dataset (snapshot logic).

**Why we removed `recency_days` from model features:**  
Because it would *leak* the churn label definition into the model and artificially inflate performance.

**Model:** RandomForest (baseline)  
**Goal:** rank customers by risk to prioritize retention actions.
            """
        )

        st.markdown(
            """
**Limitations**
- This dataset is marketplace-heavy, many customers buy once → frequency is very discrete.
- This churn definition is a proxy, not a real cancellation/unsubscribe event.
- Better production evaluation would use time-based validation.
            """
        )


if __name__ == "__main__":
    main()