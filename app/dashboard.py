# app/dashboard.py
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# Import path fix for Streamlit Cloud (so `src.*` imports work)
# ------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import settings
from src.modeling.inference import predict_churn_proba

# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------
ROOT = settings.root_dir
PROCESSED_DIR = ROOT / settings.data_processed_dir

ASSETS_DIR = ROOT / "assets" / "branding"
FAVICON_PNG = ASSETS_DIR / "favicon.png"
BANNER_SVG = ASSETS_DIR / "banner-app.svg"
LOGO_LIGHT_SVG = ASSETS_DIR / "logo.svg"
LOGO_DARK_SVG = ASSETS_DIR / "logo-dark.svg"

CURRENCY_CODE = "BRL"
CURRENCY_SYMBOL = "R$"


# -----------------------------
# Page config (favicon real)
# -----------------------------
st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon=str(FAVICON_PNG) if FAVICON_PNG.exists() else "📊",
    layout="wide",
)


# -----------------------------
# i18n (ES/EN)
# -----------------------------
I18N = {
    "en": {
        "preferences": "Preferences",
        "language": "Language",
        "currency": "Currency",
        "filters": "Filters",
        "reset_filters": "Reset filters",
        "help": "Help",
        "segment": "Customer segment",
        "date_range": "Purchase date range",
        "all": "All",
        "active_filters": "Active filters",
        "customers": "Customers",
        "orders": "Delivered orders",
        "revenue": "Revenue",
        "churn_proxy": "Churn (proxy {window})",
        "coverage": "Coverage",
        "data_source": "Data source",
        "tabs": ["Executive", "Segments", "Predict", "Method"],
        "loading": "Loading data and computing KPIs…",
        "badge_demo": "Demo mode",
        "demo_note": "You are seeing a sample dataset (cloud-friendly).",
        "exec_bullets": [
            "Revenue is concentrated in a small set of segments.",
            "Churn is a proxy label (recency window) used for prioritization.",
            "Use Predict to size opportunity and export a priority list.",
        ],
        "cards_titles": ["What’s happening", "Why it matters", "What I’d do today"],
        "cards_exec": [
            "Revenue varies strongly by segment; some groups fund most of the business.",
            "Knowing which segments drive revenue guides retention and service improvements.",
            "Start with top revenue segments + high-risk list to plan win-back outreach.",
        ],
        "rev_by_seg": "Revenue by segment",
        "seg_dist": "Customers by segment",
        "seg_table": "Segment table",
        "download_seg": "Download segment summary (CSV)",
        "predict_intro": "This is a risk score (0–100%) to prioritize outreach. It supports decisions; it is not a guarantee.",
        "how_to_use": "How to use it: start with the suggested threshold, then adjust to your team’s capacity.",
        "decision_controls": "Decision controls",
        "suggested_threshold": "Suggested threshold",
        "risk_threshold": "Risk threshold (%)",
        "risk_threshold_help": "Customers above this threshold are considered high-risk.",
        "winback": "Expected win-back rate (%)",
        "winback_help": "Scenario: re-activate X% of high-risk customers.",
        "opp_sizing": "Opportunity sizing",
        "avg_risk": "Average churn risk",
        "high_risk_customers": "High-risk customers",
        "rev_at_risk": "Revenue at risk",
        "uplift": "Projected uplift",
        "rev_at_risk_share": "Revenue at risk share",
        "dist_risk": "Risk distribution (%)",
        "top_prioritize": "Priority customers",
        "how_many": "How many customers?",
        "download_prior": "Download priority customers (CSV)",
        "no_model": "Model artifacts not found or failed to load.",
        "run_train": "Local: run `python -m src.modeling.train_churn_model`",
        "cloud_commit": "Cloud: ensure model artifacts are committed.",
        "insight_threshold": "{pct}% of customers in current filters are above the selected threshold.",
        "insight_capacity": "Tip: If the list is too large, raise the threshold or narrow filters.",
        "help_text": {
            "proxy": "Churn here is a proxy: a customer is labeled churned if they haven’t purchased within a time window (snapshot-based).",
            "interpret": "Use risk to prioritize outreach. It’s a ranking tool, not a guarantee.",
            "export": "Export the priority list and hand it to CRM/marketing for targeted campaigns.",
        },
        "method_title": "Method (portfolio)",
        "method_bullets": [
            "Churn label is a proxy: churn_window = recency_days > window (snapshot-based).",
            "We remove recency_days from model inputs to avoid target leakage.",
            "Baseline model: RandomForest to rank customers by risk.",
        ],
        "limitations": [
            "Marketplace-heavy behavior: many customers buy once → frequency is discrete.",
            "Proxy churn ≠ real cancellation/unsubscribe.",
            "Production evaluation should use time-based validation.",
        ],
    },
    "es": {
        "preferences": "Preferencias",
        "language": "Idioma",
        "currency": "Moneda",
        "filters": "Filtros",
        "reset_filters": "Restablecer filtros",
        "help": "Ayuda",
        "segment": "Segmento de cliente",
        "date_range": "Rango de fechas de compra",
        "all": "Todos",
        "active_filters": "Filtros activos",
        "customers": "Clientes",
        "orders": "Pedidos entregados",
        "revenue": "Ingresos",
        "churn_proxy": "Churn (proxy {window})",
        "coverage": "Cobertura",
        "data_source": "Fuente de datos",
        "tabs": ["Resumen", "Segmentos", "Predicción", "Método"],
        "loading": "Cargando datos y calculando KPIs…",
        "badge_demo": "Modo demo",
        "demo_note": "Estás viendo un dataset de muestra (apto para cloud).",
        "exec_bullets": [
            "Los ingresos se concentran en pocos segmentos clave.",
            "El churn es un proxy (ventana de recencia) para priorizar.",
            "Usa Predicción para dimensionar oportunidad y exportar una lista priorizada.",
        ],
        "cards_titles": ["Qué está pasando", "Por qué importa", "Qué haría hoy"],
        "cards_exec": [
            "Los ingresos varían mucho por segmento; algunos sostienen gran parte del negocio.",
            "Saber qué segmentos aportan más ayuda a enfocar retención y mejoras operativas.",
            "Empieza por segmentos top en ingresos + lista de alto riesgo para campañas win-back.",
        ],
        "rev_by_seg": "Ingresos por segmento",
        "seg_dist": "Clientes por segmento",
        "seg_table": "Tabla por segmento",
        "download_seg": "Descargar resumen de segmentos (CSV)",
        "predict_intro": "Esto es un score de riesgo (0–100%) para priorizar acciones. Ayuda a decidir; no es una garantía.",
        "how_to_use": "Cómo usarlo: empieza con el umbral sugerido y ajusta según la capacidad del equipo.",
        "decision_controls": "Controles de decisión",
        "suggested_threshold": "Umbral sugerido",
        "risk_threshold": "Umbral de riesgo (%)",
        "risk_threshold_help": "Clientes por encima del umbral se consideran alto riesgo.",
        "winback": "Tasa esperada de win-back (%)",
        "winback_help": "Escenario: reactivar X% de los clientes de alto riesgo.",
        "opp_sizing": "Cálculo de oportunidad",
        "avg_risk": "Riesgo medio de churn",
        "high_risk_customers": "Clientes de alto riesgo",
        "rev_at_risk": "Ingresos en riesgo",
        "uplift": "Uplift proyectado",
        "rev_at_risk_share": "Share de ingresos en riesgo",
        "dist_risk": "Distribución del riesgo (%)",
        "top_prioritize": "Clientes prioritarios",
        "how_many": "¿Cuántos clientes?",
        "download_prior": "Descargar clientes priorizados (CSV)",
        "no_model": "No se encuentran los artefactos del modelo o falló la carga.",
        "run_train": "Local: ejecuta `python -m src.modeling.train_churn_model`",
        "cloud_commit": "Cloud: asegúrate de haber commiteado los artefactos del modelo.",
        "insight_threshold": "El {pct}% de los clientes filtrados está por encima del umbral seleccionado.",
        "insight_capacity": "Tip: si la lista es enorme, sube el umbral o acota filtros.",
        "help_text": {
            "proxy": "El churn aquí es un proxy: se etiqueta churn si no compra dentro de una ventana (snapshot).",
            "interpret": "El riesgo sirve para priorizar. Es un ranking, no una garantía.",
            "export": "Exporta la lista priorizada y úsala en CRM/marketing para campañas dirigidas.",
        },
        "method_title": "Método (portfolio)",
        "method_bullets": [
            "El churn es un proxy: churn_window = recency_days > window (snapshot).",
            "Quitamos recency_days del modelo para evitar leakage.",
            "Modelo base: RandomForest para rankear clientes por riesgo.",
        ],
        "limitations": [
            "Marketplace: muchos clientes compran una vez → frecuencia muy discreta.",
            "Proxy churn ≠ cancelación real.",
            "En producción, evaluar con validación temporal.",
        ],
    },
}


# -----------------------------
# CSS (pro + accesible + premium states)
# -----------------------------
def apply_css() -> None:
    st.markdown(
        """
<style>
/* Layout + spacing */
.block-container { padding-top: 1.0rem; padding-bottom: 2.25rem; max-width: 1200px; }
.section { margin-top: 0.6rem; margin-bottom: 0.9rem; }

/* Sidebar */
section[data-testid="stSidebar"] { border-right: 1px solid rgba(49,51,63,0.12); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }

/* Typography */
h1, h2, h3 { letter-spacing: -0.02em; }
.stCaption, small { color: rgba(49,51,63,0.72); }
@media (prefers-color-scheme: dark) { .stCaption, small { color: rgba(255,255,255,0.72); } }

/* Accessible focus ring */
*:focus { outline: 3px solid rgba(0, 121, 191, 0.70) !important; outline-offset: 2px; }

/* Content cards */
.card {
  border: 1px solid rgba(49,51,63,0.12);
  border-radius: 16px;
  padding: 0.9rem 1rem;
  background: #ffffff;
}
@media (prefers-color-scheme: dark) {
  .card { background: rgba(20,20,25,0.55); border: 1px solid rgba(255,255,255,0.10); }
}

/* KPI cards */
div[data-testid="stMetric"] {
  padding: 0.95rem 1rem;
  border: 1px solid rgba(49,51,63,0.12);
  border-radius: 16px;
  background: #ffffff;
}
@media (prefers-color-scheme: dark) {
  div[data-testid="stMetric"] { background: rgba(20,20,25,0.55); border: 1px solid rgba(255,255,255,0.10); }
}

/* Tabs */
div[data-baseweb="tab-list"] { gap: 0.25rem; }
div[data-baseweb="tab-list"] button {
  font-weight: 650;
  padding: 0.55rem 0.85rem;
  border-radius: 12px;
  transition: transform .06s ease, background-color .12s ease, border-color .12s ease;
}
div[data-baseweb="tab-list"] button:hover { transform: translateY(-1px); }
div[data-baseweb="tab-list"] button[aria-selected="true"] { border: 1px solid rgba(49,51,63,0.18); }
div[data-baseweb="tab-panel"] { padding-top: 0.7rem; }

/* Buttons */
.stDownloadButton button, .stButton button {
  border-radius: 12px !important;
  min-height: 44px !important;
  transition: transform .06s ease, box-shadow .12s ease;
}
.stDownloadButton button:hover, .stButton button:hover { transform: translateY(-1px); }
.stDownloadButton button:active, .stButton button:active { transform: translateY(0px); }

/* Tables */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(49,51,63,0.12);
  border-radius: 16px;
  overflow: hidden;
}

/* Banner */
[data-testid="stAppViewContainer"] .main { padding-top: 0.25rem; }
.brand-banner img { border-radius: 16px; border: 1px solid rgba(49,51,63,0.12); }

/* Badge */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.25rem 0.55rem;
  border-radius: 999px;
  border: 1px solid rgba(49,51,63,0.18);
  background: rgba(255,255,255,0.90);
  font-size: 0.85rem;
}
@media (prefers-color-scheme: dark) {
  .badge { background: rgba(20,20,25,0.55); border: 1px solid rgba(255,255,255,0.12); }
}

/* Emphasis (not only color: icons + label) */
.emph-risk { border-color: rgba(220, 38, 38, 0.25) !important; }
.emph-uplift { border-color: rgba(22, 163, 74, 0.25) !important; }

.hint { font-size: 0.92rem; opacity: 0.92; }
</style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Helpers
# -----------------------------
def brl(x: float) -> str:
    """Format BRL with R$ prefix for UI."""
    try:
        return f"{CURRENCY_SYMBOL} {float(x):,.0f}"
    except Exception:
        return "—"


def pct(x: float, decimals: int = 1) -> str:
    try:
        return f"{float(x):.{decimals}f}%"
    except Exception:
        return "—"


def last_updated(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def render_branding() -> None:
    if BANNER_SVG.exists():
        st.markdown('<div class="brand-banner">', unsafe_allow_html=True)
        st.image(str(BANNER_SVG), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    theme_base = (st.get_option("theme.base") or "light").lower()
    logo_path = LOGO_DARK_SVG if theme_base == "dark" and LOGO_DARK_SVG.exists() else LOGO_LIGHT_SVG
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=140)


def card(title: str, body: str) -> None:
    st.markdown(
        f"""
<div class="card">
  <div style="font-weight: 750; margin-bottom: 0.35rem;">{title}</div>
  <div class="hint">{body}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def compute_suggested_threshold(pcts: pd.Series) -> int:
    """Pick a threshold that yields a manageable list (70th percentile, snapped to 5)."""
    if pcts.empty:
        return 60
    q = float(pcts.quantile(0.70))
    snapped = int(round(q / 5) * 5)
    return max(10, min(95, snapped))


def clamp_date_range(min_date, max_date):
    if pd.notna(min_date) and pd.notna(max_date):
        return (min_date.date(), max_date.date())
    return None


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    seg_path = PROCESSED_DIR / "customer_segments.csv"
    tx_path = PROCESSED_DIR / "transactions.csv"

    demo_seg_path = ROOT / "data" / "demo" / "customer_segments_demo.csv"
    demo_tx_path = ROOT / "data" / "demo" / "transactions_demo.csv"

    demo_mode = False
    if seg_path.exists() and tx_path.exists():
        segments = pd.read_csv(seg_path)
        tx = pd.read_csv(tx_path)
        source = f"processed (local) · updated {last_updated(seg_path)}"
    elif demo_seg_path.exists() and demo_tx_path.exists():
        segments = pd.read_csv(demo_seg_path)
        tx = pd.read_csv(demo_tx_path)
        source = f"demo sample (cloud) · updated {last_updated(demo_seg_path)}"
        demo_mode = True
    else:
        st.error(
            "No data found.\n\n"
            "Local: run `python main.py` to generate `data/processed/*.csv`\n"
            "Cloud: commit `data/demo/*.csv` so the app can load sample data."
        )
        st.stop()

    if "order_purchase_timestamp" in tx.columns:
        tx["order_purchase_timestamp"] = pd.to_datetime(tx["order_purchase_timestamp"], errors="coerce")

    required_seg_cols = {"customer_unique_id", "segment_name", "monetary_total"}
    required_tx_cols = {"order_id", "order_purchase_timestamp"}
    if not required_seg_cols.issubset(set(segments.columns)):
        st.error(f"Segments missing columns: {sorted(required_seg_cols - set(segments.columns))}")
        st.stop()
    if not required_tx_cols.issubset(set(tx.columns)):
        st.error(f"Transactions missing columns: {sorted(required_tx_cols - set(tx.columns))}")
        st.stop()

    return segments, tx, source, demo_mode


def main() -> None:
    apply_css()
    render_branding()

    # -----------------------------
    # Preferences (persistent)
    # -----------------------------
    LANG_LABEL_TO_CODE = {"English (EN)": "en", "Español (ES)": "es"}

    if "lang" not in st.session_state:
        st.session_state.lang = "es"

    current_lang = st.session_state.lang if st.session_state.lang in I18N else "es"
    t0 = I18N[current_lang]

    st.sidebar.header(t0["preferences"])
    selected_label = st.sidebar.selectbox(
        t0["language"],
        list(LANG_LABEL_TO_CODE.keys()),
        index=list(LANG_LABEL_TO_CODE.values()).index(current_lang),
        help="Choose the language for the interface / Elige el idioma de la interfaz.",
    )
    st.session_state.lang = LANG_LABEL_TO_CODE[selected_label]

    lang = st.session_state.lang
    t = I18N[lang]

    # Reset filters (pro UX)
    if st.sidebar.button(t["reset_filters"], type="secondary"):
        st.session_state.clear()
        st.rerun()

    # Help (microcopy, no tech)
    with st.sidebar.expander(f"ℹ️ {t['help']}", expanded=False):
        st.write(f"**Churn proxy:** {t['help_text']['proxy']}")
        st.write(f"**Interpretation:** {t['help_text']['interpret']}")
        st.write(f"**Export:** {t['help_text']['export']}")

    st.sidebar.divider()

    # Badges
    st.markdown(
        f'<span class="badge">💱 {t["currency"]}: <b>{CURRENCY_CODE}</b> ({CURRENCY_SYMBOL})</span>',
        unsafe_allow_html=True,
    )

    with st.spinner(t["loading"]):
        segments, tx, data_source, demo_mode = load_data()

    if demo_mode:
        st.markdown(f'<span class="badge">🧪 {t["badge_demo"]}</span>', unsafe_allow_html=True)
        st.caption(t["demo_note"])

    churn_cols = [c for c in segments.columns if c.startswith("churn_")]
    churn_col = churn_cols[0] if churn_cols else None

    # Coverage
    min_date = tx["order_purchase_timestamp"].min()
    max_date = tx["order_purchase_timestamp"].max()
    days_span = int((max_date - min_date).days) if pd.notna(min_date) and pd.notna(max_date) else 0

    # -----------------------------
    # Filters (order: segment -> dates)
    # -----------------------------
    st.sidebar.header(t["filters"])
    segment_options = [t["all"]] + sorted(segments["segment_name"].dropna().unique().tolist())
    selected_segment = st.sidebar.selectbox(t["segment"], segment_options)

    date_range = None
    minmax = clamp_date_range(min_date, max_date)
    if minmax:
        date_range = st.sidebar.date_input(
            t["date_range"],
            value=minmax,
            min_value=minmax[0],
            max_value=minmax[1],
        )

    # Apply filters
    if selected_segment != t["all"]:
        segments_filtered = segments[segments["segment_name"] == selected_segment].copy()
    else:
        segments_filtered = segments.copy()

    tx_filtered = tx.copy()
    if date_range:
        tx_filtered = tx_filtered[
            (tx_filtered["order_purchase_timestamp"].dt.date >= date_range[0])
            & (tx_filtered["order_purchase_timestamp"].dt.date <= date_range[1])
        ].copy()

    # Active filters summary (UX)
    seg_label = selected_segment
    date_label = f"{date_range[0]} → {date_range[1]}" if date_range else "—"
    st.markdown(
        f'<span class="badge">🎛️ <b>{t["active_filters"]}:</b> {t["segment"]}: {seg_label} · {t["date_range"]}: {date_label}</span>',
        unsafe_allow_html=True,
    )

    # -----------------------------
    # KPIs (consistent)
    # -----------------------------
    total_customers = int(segments_filtered.shape[0])
    total_orders = int(tx_filtered["order_id"].nunique())
    total_revenue = float(segments_filtered["monetary_total"].sum())

    churn_label = t["churn_proxy"].format(window="—")
    churn_value = "N/A"
    if churn_col:
        churn_window = churn_col.replace("churn_", "")
        churn_label = t["churn_proxy"].format(window=churn_window)
        churn_rate = float(segments_filtered[churn_col].mean() * 100)
        churn_value = pct(churn_rate, 1)

    k1, k2, k3, k4 = st.columns(4, gap="small")
    k1.metric(t["customers"], f"{total_customers:,}")
    k2.metric(t["orders"], f"{total_orders:,}")
    k3.metric(t["revenue"], brl(total_revenue))
    k4.metric(churn_label, churn_value)

    if pd.notna(min_date) and pd.notna(max_date):
        st.caption(
            f'{t["coverage"]}: {min_date.date()} → {max_date.date()} '
            f"({days_span} days) · {t['data_source']}: {data_source}"
        )
    else:
        st.caption(f"{t['data_source']}: {data_source}")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(t["tabs"])

    # -----------------------------
    # Tab 1: Executive (5-sec story)
    # -----------------------------
    with tab1:
        st.markdown("\n".join([f"- {b}" for b in t["exec_bullets"]]))

        cA, cB, cC = st.columns(3, gap="small")
        with cA:
            card(t["cards_titles"][0], t["cards_exec"][0])
        with cB:
            card(t["cards_titles"][1], t["cards_exec"][1])
        with cC:
            card(t["cards_titles"][2], t["cards_exec"][2])

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)

        st.subheader(t["rev_by_seg"])
        rev_by_seg = segments_filtered.groupby("segment_name")["monetary_total"].sum().sort_values(ascending=False)

        max_bars = 10
        if len(rev_by_seg) > max_bars:
            top = rev_by_seg.head(max_bars)
            others = pd.Series({"Others": float(rev_by_seg.iloc[max_bars:].sum())})
            rev_plot = pd.concat([top, others])
        else:
            rev_plot = rev_by_seg

        fig = px.bar(
            rev_plot.reset_index().rename(columns={"index": "segment_name"}),
            x="segment_name",
            y="monetary_total",
            labels={"segment_name": "Segment", "monetary_total": f"{t['revenue']} ({CURRENCY_SYMBOL})"},
        )
        fig.update_layout(xaxis_tickangle=-25, margin=dict(t=10, l=10, r=10, b=10))
        fig.update_yaxes(tickprefix=f"{CURRENCY_SYMBOL} ", tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")

    # -----------------------------
    # Tab 2: Segments (decision table)
    # -----------------------------
    with tab2:
        st.subheader(t["seg_dist"])

        seg_counts = segments_filtered["segment_name"].value_counts().reset_index()
        seg_counts.columns = ["segment_name", "customers"]

        fig = px.bar(
            seg_counts,
            x="segment_name",
            y="customers",
            labels={"segment_name": "Segment", "customers": t["customers"]},
        )
        fig.update_layout(xaxis_tickangle=-25, margin=dict(t=10, l=10, r=10, b=10))
        fig.update_yaxes(tickformat=",.0f")
        st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.subheader(t["seg_table"])

        if churn_col:
            summary = (
                segments_filtered.groupby("segment_name")
                .agg(
                    customers=("customer_unique_id", "count"),
                    revenue=("monetary_total", "sum"),
                    churn_rate=(churn_col, "mean"),
                )
                .sort_values("revenue", ascending=False)
            )
            summary["churn_risk_%"] = (summary["churn_rate"] * 100).round(1)
            summary = summary.drop(columns=["churn_rate"])
        else:
            summary = (
                segments_filtered.groupby("segment_name")
                .agg(customers=("customer_unique_id", "count"), revenue=("monetary_total", "sum"))
                .sort_values("revenue", ascending=False)
            )

        display = summary.reset_index().rename(columns={"segment_name": "segment"}).copy()
        cols = ["segment", "customers", "revenue"] + (["churn_risk_%"] if "churn_risk_%" in display.columns else [])
        display = display[cols]

        display["revenue"] = display["revenue"].map(brl)
        if "churn_risk_%" in display.columns:
            display["churn_risk_%"] = display["churn_risk_%"].map(lambda x: pct(x, 1))

        st.dataframe(display, width="stretch", hide_index=True)

        st.download_button(
            t["download_seg"],
            data=summary.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="segment_summary.csv",
            mime="text/csv",
        )

    # -----------------------------
    # Tab 3: Predict (product-feel)
    # -----------------------------
    with tab3:
        st.subheader(t["tabs"][2])
        st.write(t["predict_intro"])
        st.caption(t["how_to_use"])

        try:
            segments_scored = segments_filtered.copy()
            segments_scored["churn_probability"] = predict_churn_proba(segments_scored)
            segments_scored["churn_probability_%"] = (segments_scored["churn_probability"] * 100).clip(0, 100)

            suggested = compute_suggested_threshold(segments_scored["churn_probability_%"])

            st.markdown('<div class="section"></div>', unsafe_allow_html=True)
            st.markdown(f"### {t['decision_controls']}")

            cA, cB = st.columns([1, 1], gap="small")
            with cA:
                st.markdown(
                    f'<span class="badge">✨ {t["suggested_threshold"]}: <b>{suggested}%</b></span>',
                    unsafe_allow_html=True,
                )
                threshold = st.slider(
                    t["risk_threshold"],
                    min_value=10,
                    max_value=95,
                    value=suggested,
                    step=5,
                    help=t["risk_threshold_help"],
                )
            with cB:
                uplift_rate = st.slider(
                    t["winback"],
                    min_value=1,
                    max_value=25,
                    value=5,
                    step=1,
                    help=t["winback_help"],
                )

            high_risk = segments_scored[segments_scored["churn_probability_%"] >= threshold].copy()

            st.markdown('<div class="section"></div>', unsafe_allow_html=True)
            st.markdown(f"### {t['opp_sizing']}")

            total_customers_scored = int(segments_scored.shape[0])
            high_risk_customers = int(high_risk.shape[0])
            pct_above = (high_risk_customers / total_customers_scored * 100) if total_customers_scored else 0.0

            total_revenue_scored = float(segments_scored["monetary_total"].sum())
            revenue_at_risk = float(high_risk["monetary_total"].sum())
            revenue_at_risk_share = (revenue_at_risk / total_revenue_scored * 100) if total_revenue_scored else 0.0
            projected_uplift = revenue_at_risk * (uplift_rate / 100)
            avg_risk = float(segments_scored["churn_probability_%"].mean())

            st.caption(t["insight_threshold"].format(pct=f"{pct_above:.1f}"))
            st.caption(t["insight_capacity"])

            k1, k2, k3, k4 = st.columns(4, gap="small")
            k1.metric(t["avg_risk"], pct(avg_risk, 1))
            k2.metric(t["high_risk_customers"], f"{high_risk_customers:,} / {total_customers_scored:,}")

            with k3:
                st.markdown('<div class="emph-risk">', unsafe_allow_html=True)
                st.metric(f"⚠️ {t['rev_at_risk']}", brl(revenue_at_risk))
                st.markdown("</div>", unsafe_allow_html=True)

            with k4:
                st.markdown('<div class="emph-uplift">', unsafe_allow_html=True)
                st.metric(f"⬆️ {t['uplift']}", brl(projected_uplift))
                st.markdown("</div>", unsafe_allow_html=True)

            st.caption(f"{t['rev_at_risk_share']}: **{revenue_at_risk_share:.1f}%**")

            st.markdown('<div class="section"></div>', unsafe_allow_html=True)
            st.subheader(t["dist_risk"])

            fig = px.histogram(
                segments_scored,
                x="churn_probability_%",
                nbins=20,
                labels={"churn_probability_%": "Churn risk (%)"},
            )
            fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
            fig.update_xaxes(tickformat=".0f")
            st.plotly_chart(fig, width="stretch")

            st.markdown('<div class="section"></div>', unsafe_allow_html=True)
            st.subheader(t["top_prioritize"])

            top_n = st.slider(t["how_many"], 10, 300, 50, step=10)

            display_cols = [
                "customer_unique_id",
                "segment_name",
                "churn_probability_%",
                "monetary_total",
                "avg_order_value",
                "avg_delivery_days",
                "avg_review_score",
            ]
            available_cols = [c for c in display_cols if c in segments_scored.columns]

            top = segments_scored.sort_values("churn_probability_%", ascending=False).head(top_n).copy()
            top_display = top[available_cols].copy()

            if "monetary_total" in top_display.columns:
                top_display["monetary_total"] = top_display["monetary_total"].map(brl)
            if "avg_order_value" in top_display.columns:
                top_display["avg_order_value"] = top_display["avg_order_value"].round(0).astype("Int64")
            if "avg_delivery_days" in top_display.columns:
                top_display["avg_delivery_days"] = top_display["avg_delivery_days"].round(0).astype("Int64")
            if "avg_review_score" in top_display.columns:
                top_display["avg_review_score"] = top_display["avg_review_score"].round(2)
            if "churn_probability_%" in top_display.columns:
                top_display["churn_probability_%"] = top_display["churn_probability_%"].round(1)

            st.caption("Top 10 are the highest risk in the current filters.")
            st.dataframe(top_display, width="stretch", hide_index=True)

            st.download_button(
                t["download_prior"],
                data=top[available_cols].to_csv(index=False).encode("utf-8"),
                file_name="priority_customers.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(
                f"{t['no_model']}\n\n"
                f"{t['run_train']}\n"
                f"{t['cloud_commit']}\n\n"
                f"Details: {e}"
            )

    # -----------------------------
    # Tab 4: Method (scannable bullets)
    # -----------------------------
    with tab4:
        st.subheader(t["method_title"])

        st.markdown("**Key points**")
        st.markdown("\n".join([f"- {b}" for b in t["method_bullets"]]))

        st.markdown("**Limitations**")
        st.markdown("\n".join([f"- {b}" for b in t["limitations"]]))


if __name__ == "__main__":
    main()