"""
Sentiment Analysis Page — Deep-dive sentiment visualization and trend analysis.
streamlit_app/pages/2_Sentiment_Analysis.py
"""
import sys
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

st.set_page_config(
    page_title="Sentiment Analysis — Financial Analyzer",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sentiment-positive { color: #10b981; font-weight: 600; }
    .sentiment-negative { color: #dc2626; font-weight: 600; }
    .sentiment-neutral  { color: #6b7280; font-weight: 600; }
    .insight-card {
        border-radius: 0.5rem;
        padding: 1rem 1.25rem;
        margin-bottom: 0.6rem;
        border-left: 4px solid;
    }
    .insight-positive { background:#f0fdf4; border-color:#10b981; }
    .insight-negative { background:#fff1f2; border-color:#dc2626; }
    .insight-neutral  { background:#f8fafc; border-color:#6b7280; }
    .score-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .compare-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "AAPL"


# ── Mock data generators ──────────────────────────────────────────────────────

def _sentiment_timeseries(symbol: str, days: int) -> pd.DataFrame:
    np.random.seed(hash(symbol) % 999)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    # Create a trending signal with noise
    trend = np.linspace(0.55, 0.75, days)
    noise = np.random.normal(0, 0.07, days)
    scores = np.clip(trend + noise, 0.05, 0.98)
    return pd.DataFrame({"Date": dates, "Score": scores, "Symbol": symbol})


def _compare_symbols_data(symbols: list, days: int) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        frames.append(_sentiment_timeseries(sym, days))
    return pd.concat(frames, ignore_index=True)


def _entity_sentiments(symbol: str) -> list:
    return [
        {"entity": "Earnings Performance", "score": 0.91, "direction": "Positive", "articles": 14},
        {"entity": "Product Innovation",   "score": 0.82, "direction": "Positive", "articles": 9},
        {"entity": "Management Team",      "score": 0.71, "direction": "Positive", "articles": 5},
        {"entity": "Market Competition",   "score": 0.52, "direction": "Neutral",  "articles": 11},
        {"entity": "Regulatory Outlook",   "score": 0.31, "direction": "Negative", "articles": 7},
        {"entity": "Debt & Leverage",      "score": 0.38, "direction": "Negative", "articles": 4},
    ]


def _keyword_cloud_data(symbol: str) -> dict:
    return {
        "earnings": 18, "growth": 15, "revenue": 14, "innovation": 12,
        "AI": 11, "guidance": 9, "buyback": 8, "margin": 8,
        "regulatory": 7, "competition": 6, "debt": 5, "lawsuit": 4,
        "dividend": 4, "acquisition": 3, "layoffs": 2,
    }


def _ml_vs_llm_comparison(symbol: str) -> dict:
    return {
        "ml_score": 0.73,
        "llm_score": 0.77,
        "ml_label": "POSITIVE",
        "llm_label": "Positive",
        "agreement": True,
        "ml_confidence": 0.81,
        "llm_confidence": 0.85,
        "articles_analyzed": 47,
    }


SENTIMENT_COLORS = {"Positive": "#10b981", "Negative": "#dc2626", "Neutral": "#6b7280"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Symbol")
    symbol = st.text_input("Stock Symbol", value=st.session_state.selected_symbol).upper()
    st.session_state.selected_symbol = symbol

    st.markdown("### ⚙️ Settings")
    days_back = st.slider("Analysis Period (days)", 7, 30, 14, key="sa_days")

    st.markdown("### 🔀 Compare Mode")
    compare_mode = st.checkbox("Compare multiple symbols", key="sa_compare")
    compare_symbols_raw = ""
    if compare_mode:
        compare_symbols_raw = st.text_input(
            "Additional symbols (comma-separated)",
            value="GOOGL,MSFT",
            key="sa_compare_symbols",
        )

    st.markdown("---")
    run_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📌 Model Info")
    st.info(
        "**ML Model**: DistilBERT (SST-2)\n\n"
        "**LLM**: GPT-4 / Claude\n\n"
        "Combined scoring weights:\n- ML: 50%\n- LLM: 50%"
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">💭 Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown(f"Deep sentiment intelligence for **{symbol}** · {days_back}-day window")
st.markdown("---")

# ── Top KPIs ──────────────────────────────────────────────────────────────────
analysis = _ml_vs_llm_comparison(symbol)
ts_data = _sentiment_timeseries(symbol, days_back)
current_score = float(ts_data["Score"].iloc[-1])
prev_score = float(ts_data["Score"].iloc[-7]) if days_back >= 7 else float(ts_data["Score"].iloc[0])
delta_score = current_score - prev_score

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Score",    f"{current_score:.2f}", delta=f"{delta_score:+.2f} vs 7d ago")
col2.metric("ML Model Score",   f"{analysis['ml_score']:.2f}", delta=analysis["ml_label"])
col3.metric("LLM Score",        f"{analysis['llm_score']:.2f}", delta=analysis["llm_label"])
col4.metric("Articles Analyzed", analysis["articles_analyzed"])
col5.metric(
    "Model Agreement",
    "✅ Yes" if analysis["agreement"] else "⚠️ No",
    delta="High confidence" if analysis["agreement"] else "Review needed",
)

st.markdown("---")

# ── Gauge + Trend ─────────────────────────────────────────────────────────────
col_gauge, col_trend = st.columns([1, 2])

with col_gauge:
    st.markdown("#### 🎯 Overall Sentiment Gauge")
    label = "POSITIVE" if current_score >= 0.6 else "NEGATIVE" if current_score < 0.4 else "NEUTRAL"
    label_color = SENTIMENT_COLORS.get(label.capitalize(), "#6b7280")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_score * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>{label}</b>", "font": {"color": label_color, "size": 18}},
        delta={"reference": prev_score * 100, "valueformat": ".1f"},
        number={"suffix": "%", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": label_color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 40],  "color": "#fee2e2"},
                {"range": [40, 60], "color": "#fef3c7"},
                {"range": [60, 100],"color": "#d1fae5"},
            ],
            "threshold": {
                "line": {"color": "#1e3a8a", "width": 3},
                "thickness": 0.75,
                "value": current_score * 100,
            },
        },
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_trend:
    st.markdown("#### 📈 Sentiment Trend")
    if compare_mode and compare_symbols_raw:
        compare_syms = [s.strip().upper() for s in compare_symbols_raw.split(",") if s.strip()]
        all_syms = [symbol] + compare_syms[:3]
        df_compare = _compare_symbols_data(all_syms, days_back)
        fig_trend = px.line(
            df_compare, x="Date", y="Score", color="Symbol",
            markers=True,
            color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"],
        )
    else:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=ts_data["Date"], y=ts_data["Score"],
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
            name=symbol,
        ))

    fig_trend.add_hline(y=0.60, line_dash="dot", line_color="#10b981", opacity=0.6, annotation_text="Positive")
    fig_trend.add_hline(y=0.40, line_dash="dot", line_color="#dc2626", opacity=0.6, annotation_text="Negative")
    fig_trend.update_layout(
        height=300, yaxis_range=[0, 1],
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        yaxis_title="Sentiment Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ── Entity sentiment breakdown ────────────────────────────────────────────────
col_entities, col_dist = st.columns([3, 2])

with col_entities:
    st.markdown("#### 🔬 Entity-Level Sentiment")
    entities = _entity_sentiments(symbol)
    for e in entities:
        color = SENTIMENT_COLORS[e["direction"]]
        bar_pct = e["score"] * 100
        st.markdown(f"""
        <div class="insight-card insight-{e['direction'].lower()}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span style="font-weight:600; color:#1e293b;">{e['entity']}</span>
                <span>
                    <span class="score-pill" style="background:{color}22; color:{color};">{e['direction']}</span>
                    <span style="color:#94a3b8; font-size:0.78rem; margin-left:0.5rem;">{e['articles']} articles</span>
                </span>
            </div>
            <div style="background:#e2e8f0; border-radius:4px; height:8px; overflow:hidden;">
                <div style="width:{bar_pct:.0f}%; height:100%; background:{color}; border-radius:4px;"></div>
            </div>
            <div style="text-align:right; font-size:0.78rem; color:#64748b; margin-top:2px;">{e['score']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

with col_dist:
    st.markdown("#### 📊 Score Distribution")
    scores = ts_data["Score"].tolist()
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["Very Neg", "Negative", "Neutral", "Positive", "Very Pos"]
    counts, _ = np.histogram(scores, bins=bins)
    bar_colors = ["#dc2626", "#f87171", "#6b7280", "#34d399", "#10b981"]

    fig_dist = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color=bar_colors,
        text=counts, textposition="auto",
    ))
    fig_dist.update_layout(
        height=240, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        yaxis_title="# Days",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("#### 🥧 Sentiment Mix")
    pos_days = sum(1 for s in scores if s >= 0.6)
    neg_days = sum(1 for s in scores if s < 0.4)
    neu_days = len(scores) - pos_days - neg_days
    fig_pie = go.Figure(go.Pie(
        labels=["Positive", "Neutral", "Negative"],
        values=[pos_days, neu_days, neg_days],
        hole=0.5,
        marker_colors=["#10b981", "#6b7280", "#dc2626"],
        textinfo="percent+label",
    ))
    fig_pie.update_layout(height=200, showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ── Keyword heatmap ───────────────────────────────────────────────────────────
st.markdown("#### 🏷️ Keyword Frequency & Sentiment")
kw_data = _keyword_cloud_data(symbol)
positive_kw = {"earnings", "growth", "revenue", "innovation", "AI", "guidance", "buyback", "margin", "dividend", "acquisition"}
kw_df = pd.DataFrame([
    {
        "Keyword": k,
        "Mentions": v,
        "Sentiment": "Positive" if k in positive_kw else "Negative",
        "Color": "#10b981" if k in positive_kw else "#dc2626",
    }
    for k, v in sorted(kw_data.items(), key=lambda x: -x[1])
])

fig_kw = px.bar(
    kw_df, x="Keyword", y="Mentions",
    color="Sentiment",
    color_discrete_map={"Positive": "#10b981", "Negative": "#dc2626"},
    text="Mentions",
)
fig_kw.update_layout(
    height=260, showlegend=True,
    margin=dict(l=0, r=0, t=10, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,1)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
fig_kw.update_traces(textposition="outside")
st.plotly_chart(fig_kw, use_container_width=True)

st.markdown("---")

# ── ML vs LLM comparison ──────────────────────────────────────────────────────
st.markdown("#### 🤖 ML Model vs LLM Reasoning Comparison")
col_ml, col_arrow, col_llm = st.columns([2, 1, 2])

with col_ml:
    st.markdown(f"""
    <div class="compare-card">
        <div style="font-size:0.85rem; color:#6b7280; margin-bottom:0.3rem;">🧠 DistilBERT (ML)</div>
        <div style="font-size:2.5rem; font-weight:700; color:#3b82f6;">{analysis['ml_score']:.2f}</div>
        <div style="font-size:1rem; color:#10b981; font-weight:600;">{analysis['ml_label']}</div>
        <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.3rem;">
            Confidence: {analysis['ml_confidence']:.0%}
        </div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:0.5rem;">
            Fast pattern-based classification<br>Max 512 tokens per article
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_arrow:
    st.markdown("""
    <div style="display:flex; align-items:center; justify-content:center; height:100%; padding-top:2.5rem;">
        <div style="text-align:center;">
            <div style="font-size:2rem;">⚖️</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.3rem;">Combined<br>50/50</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_llm:
    st.markdown(f"""
    <div class="compare-card">
        <div style="font-size:0.85rem; color:#6b7280; margin-bottom:0.3rem;">💬 LLM Reasoning</div>
        <div style="font-size:2.5rem; font-weight:700; color:#8b5cf6;">{analysis['llm_score']:.2f}</div>
        <div style="font-size:1rem; color:#10b981; font-weight:600;">{analysis['llm_label']}</div>
        <div style="font-size:0.78rem; color:#94a3b8; margin-top:0.3rem;">
            Confidence: {analysis['llm_confidence']:.0%}
        </div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:0.5rem;">
            Context-aware reasoning<br>Understands financial nuance
        </div>
    </div>
    """, unsafe_allow_html=True)

if analysis["agreement"]:
    st.success("✅ Both models agree on sentiment direction — higher confidence in the combined score.")
else:
    st.warning("⚠️ Models disagree on sentiment direction. Consider reviewing individual article analysis.")

# ── Custom text analyzer ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### ✏️ Analyze Custom Text")
custom_text = st.text_area(
    "Paste any financial text to get an instant sentiment score:",
    placeholder="e.g. 'The company reported record revenue and raised full-year guidance…'",
    height=100,
    key="sa_custom_text",
)
if st.button("⚡ Analyze Text", key="sa_analyze_text") and custom_text.strip():
    with st.spinner("Analyzing…"):
        # Heuristic demo scoring
        positive_words = ["record", "beat", "raised", "growth", "strong", "profit", "surge", "upgrade", "buy"]
        negative_words = ["miss", "decline", "cut", "loss", "investigation", "lawsuit", "debt", "concern", "risk"]
        p = sum(1 for w in positive_words if w in custom_text.lower())
        n = sum(1 for w in negative_words if w in custom_text.lower())
        demo_score = min(0.95, max(0.05, 0.5 + (p - n) * 0.08))
        demo_label = "Positive" if demo_score >= 0.6 else "Negative" if demo_score < 0.4 else "Neutral"
        color = SENTIMENT_COLORS[demo_label]

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Sentiment Score", f"{demo_score:.2f}")
    col_r2.metric("Label", demo_label)
    col_r3.metric("Confidence", f"{min(0.92, 0.55 + abs(demo_score - 0.5)):.0%}")
    st.markdown(f"<div style='color:{color}; font-weight:600; font-size:1.1rem;'>→ {demo_label} sentiment detected</div>", unsafe_allow_html=True)
    st.caption("💡 This is a demo heuristic. In production, text is processed by DistilBERT + LLM.")
