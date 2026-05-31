"""
Portfolio Analysis Page — multi-holding portfolio-level risk and sentiment.
streamlit_app/pages/5_Portfolio_Analysis.py
"""
import sys
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

st.set_page_config(
    page_title="Portfolio Analysis — Financial Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .score-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 0.5rem; padding: 1rem 1.25rem;
        text-align: center;
    }
    .rec-buy   { background:#f0fdf4; border-left:5px solid #10b981; border-radius:0.5rem; padding:1rem; }
    .rec-hold  { background:#fff7ed; border-left:5px solid #f59e0b; border-radius:0.5rem; padding:1rem; }
    .rec-sell  { background:#fff1f2; border-left:5px solid #dc2626; border-radius:0.5rem; padding:1rem; }
    .rec-avoid { background:#fef2f2; border-left:5px solid #991b1b; border-radius:0.5rem; padding:1rem; }
    .risk-tag {
        display:inline-block; padding:0.15rem 0.6rem; border-radius:9999px;
        font-size:0.78rem; font-weight:700;
    }
    .risk-low    { background:#d1fae5; color:#065f46; }
    .risk-medium { background:#fef3c7; color:#92400e; }
    .risk-high   { background:#fee2e2; color:#991b1b; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "portfolio_result" not in st.session_state:
    st.session_state.portfolio_result = None
if "portfolio_holdings_str" not in st.session_state:
    st.session_state.portfolio_holdings_str = "AAPL:0.40,MSFT:0.30,GOOGL:0.30"


# ===========================================================================
# Mock data generator
# ===========================================================================

_RISK_CATEGORIES = ["regulatory", "volatility", "financial", "market", "operational"]


def _mock_analysis(symbol: str) -> dict:
    rng = np.random.default_rng(abs(hash(symbol)) % (2 ** 31))
    sent = round(float(rng.uniform(0.45, 0.85)), 2)
    risk = round(float(rng.uniform(0.25, 0.70)), 2)
    comp = round(0.5 * sent + 0.5 * (1 - risk), 2)

    if sent >= 0.75 and risk <= 0.35:
        rec = "STRONG BUY"
    elif sent >= 0.60 and risk <= 0.50:
        rec = "BUY"
    elif sent >= 0.40 and risk <= 0.65:
        rec = "HOLD"
    elif sent >= 0.25:
        rec = "SELL"
    else:
        rec = "AVOID"

    n_risks = int(rng.integers(1, 4))
    cats = list(rng.choice(_RISK_CATEGORIES, size=n_risks, replace=False))
    identified_risks = [
        {
            "category": c,
            "severity": rng.choice(["LOW", "MEDIUM", "HIGH"]),
            "likelihood": round(float(rng.uniform(0.3, 0.85)), 2),
            "description": f"{c.title()} risk identified for {symbol}",
        }
        for c in cats
    ]

    return {
        "symbol": symbol,
        "recommendation": rec,
        "scores": {"sentiment_score": sent, "risk_score": risk, "composite_score": comp},
        "_risk": {"identified_risks": identified_risks, "overall_risk_score": risk},
    }


def _mock_portfolio_result(holdings: list) -> dict:
    """Generate a plausible mock portfolio result without a real LLM."""
    individual_analyses = {
        h["symbol"].upper(): _mock_analysis(h["symbol"]) for h in holdings
    }

    total_w = sum(h["weight"] for h in holdings)
    holdings_table = []
    for h in holdings:
        sym = h["symbol"].upper()
        s = individual_analyses[sym]["scores"]
        holdings_table.append({
            "symbol": sym,
            "weight": h["weight"],
            "recommendation": individual_analyses[sym]["recommendation"],
            "sentiment_score": s["sentiment_score"],
            "risk_score": s["risk_score"],
            "composite_score": s["composite_score"],
        })

    p_sent = sum(h["weight"] * h["sentiment_score"] for h in holdings_table) / total_w
    p_risk = sum(h["weight"] * h["risk_score"] for h in holdings_table) / total_w
    p_comp = round(0.5 * p_sent + 0.5 * (1 - p_risk), 4)

    if p_sent >= 0.75 and p_risk <= 0.35:
        rec = "STRONG BUY"
    elif p_sent >= 0.60 and p_risk <= 0.50:
        rec = "BUY"
    elif p_sent >= 0.40 and p_risk <= 0.65:
        rec = "HOLD"
    elif p_sent >= 0.25:
        rec = "SELL"
    else:
        rec = "AVOID"

    from collections import defaultdict
    cat_syms: dict = defaultdict(list)
    for h in holdings:
        sym = h["symbol"].upper()
        for r in individual_analyses[sym]["_risk"]["identified_risks"]:
            cat_syms[r["category"]].append(sym)

    correlated = [
        {"category": c, "symbols": list(set(ss))}
        for c, ss in cat_syms.items()
        if len(set(ss)) > 2
    ]
    concentration = [
        {"symbol": h["symbol"], "weight": h["weight"]}
        for h in holdings
        if h["weight"] > 0.25
    ]

    conf = "HIGH" if p_comp >= 0.65 else "MEDIUM" if p_comp >= 0.45 else "LOW"
    symbols_str = ", ".join(h["symbol"].upper() for h in holdings)
    summary = (
        f"Portfolio of {len(holdings)} holdings ({symbols_str}) shows weighted sentiment "
        f"of {p_sent:.2f} and risk of {p_risk:.2f}, yielding a {rec} recommendation. "
        f"{'Concentration risk present. ' if concentration else ''}"
        f"{'Correlated risks detected: ' + ', '.join(c['category'] for c in correlated) + '.' if correlated else 'No significant cross-holding risk correlation.'}"
    )

    return {
        "portfolio_sentiment_score": round(p_sent, 4),
        "portfolio_risk_score": round(p_risk, 4),
        "portfolio_composite_score": p_comp,
        "portfolio_recommendation": rec,
        "portfolio_confidence": conf,
        "concentration_risks": concentration,
        "correlated_risks": correlated,
        "holdings_table": holdings_table,
        "individual_analyses": individual_analyses,
        "executive_summary": summary,
        "analysis_date": datetime.now().isoformat(),
    }


# ===========================================================================
# Helpers
# ===========================================================================

def _parse_holdings(text: str) -> tuple[list, str | None]:
    """Parse 'AAPL:0.4,MSFT:0.3' → list of dicts or ([], error_msg)."""
    holdings = []
    try:
        for part in text.strip().split(","):
            sym, w = part.strip().split(":")
            holdings.append({"symbol": sym.strip().upper(), "weight": float(w.strip())})
    except Exception:
        return [], "Invalid format — use SYMBOL:WEIGHT pairs, e.g. AAPL:0.40,MSFT:0.30"
    total = sum(h["weight"] for h in holdings)
    if abs(total - 1.0) > 0.05:
        return holdings, f"Weights sum to {total:.2f} — they should total 1.0"
    return holdings, None


def _rec_css(rec: str) -> str:
    if "BUY" in rec:
        return "rec-buy"
    if rec == "HOLD":
        return "rec-hold"
    if rec == "SELL":
        return "rec-sell"
    return "rec-avoid"


def _risk_tag(level: str) -> str:
    css = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}.get(level, "risk-medium")
    return f'<span class="risk-tag {css}">{level}</span>'


def _gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 28, "color": color}, "valueformat": ".2f"},
        title={"text": title, "font": {"size": 13, "color": "#6b7280"}},
        gauge={
            "axis": {"range": [0, 1], "tickfont": {"size": 9}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#f8fafc",
            "borderwidth": 1,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 0.4], "color": "#fee2e2"},
                {"range": [0.4, 0.65], "color": "#fef3c7"},
                {"range": [0.65, 1], "color": "#d1fae5"},
            ],
        },
    ))
    fig.update_layout(
        height=180, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_heatmap(result: dict) -> go.Figure | None:
    """Build a risk-category × holdings presence heatmap."""
    holdings_table = result.get("holdings_table", [])
    individual = result.get("individual_analyses", {})
    if not holdings_table or not individual:
        return None

    symbols = [h["symbol"] for h in holdings_table]
    matrix = []
    for cat in _RISK_CATEGORIES:
        row = []
        for sym in symbols:
            risks = individual.get(sym, {}).get("_risk", {}).get("identified_risks", [])
            cats_in_holding = {r["category"].lower() for r in risks}
            row.append(1.0 if cat in cats_in_holding else 0.0)
        matrix.append(row)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=symbols,
        y=[c.title() for c in _RISK_CATEGORIES],
        colorscale=[[0, "#f0fdf4"], [1, "#dc2626"]],
        showscale=False,
        text=[["Present" if v else "—" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 11},
        xgap=3,
        ygap=3,
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=90, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(side="top", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


# ===========================================================================
# Sidebar
# ===========================================================================

with st.sidebar:
    st.markdown("### 📊 Portfolio")
    holdings_input = st.text_area(
        "Holdings (SYMBOL:WEIGHT, one per line or comma-separated)",
        value=st.session_state.portfolio_holdings_str.replace(",", "\n"),
        height=140,
        help="Weights must sum to 1.0. Example:\nAAPL:0.40\nMSFT:0.30\nGOOGL:0.30",
        key="pa_holdings_input",
    )
    # Normalise: accept newline or comma delimiters
    holdings_text = holdings_input.replace("\n", ",").replace(",,", ",").strip(",")
    st.session_state.portfolio_holdings_str = holdings_text

    days_back = st.slider("Look-back (days)", 1, 30, 7, key="pa_days")

    st.markdown("---")
    analyze_btn = st.button(
        "🚀 Analyze Portfolio",
        type="primary",
        use_container_width=True,
        key="pa_analyze",
    )

    if st.session_state.portfolio_result:
        r = st.session_state.portfolio_result
        st.markdown("---")
        st.success("✅ Analysis complete")
        st.markdown(f"**Recommendation:** {r.get('portfolio_recommendation', 'N/A')}")
        st.markdown(f"**Composite:** {r.get('portfolio_composite_score', 0):.2f}")


# ===========================================================================
# Header
# ===========================================================================

st.markdown('<h1 class="main-header">📊 Portfolio Analysis</h1>', unsafe_allow_html=True)
st.markdown("Aggregate risk and sentiment across a weighted portfolio of holdings.")
st.markdown("---")


# ===========================================================================
# Run analysis
# ===========================================================================

if analyze_btn:
    st.session_state.portfolio_result = None
    holdings, parse_error = _parse_holdings(holdings_text)

    if parse_error and not holdings:
        st.error(f"❌ {parse_error}")
    else:
        if parse_error:
            st.warning(f"⚠️ {parse_error}")

        with st.spinner(f"Analyzing {len(holdings)} holding(s)…"):
            result = None
            chain_error = None
            try:
                from src.chains.analysis_chain import FinancialAnalysisChain
                chain = FinancialAnalysisChain(verbose=False)
                result = chain.analyze_portfolio(holdings=holdings, days_back=days_back)
            except EnvironmentError as e:
                chain_error = str(e)
            except Exception as e:
                chain_error = str(e)

            if result is None:
                result = _mock_portfolio_result(holdings)
                # Attach individual_analyses for heatmap even in mock path
                if chain_error:
                    st.caption(
                        f"ℹ️ Using demo data (chain error: {chain_error}). "
                        "Configure an LLM key in `.env` for live analysis."
                    )

        st.session_state.portfolio_result = result

# ===========================================================================
# Render result
# ===========================================================================

result = st.session_state.portfolio_result

if result is None:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#94a3b8;">
        <div style="font-size:4rem; margin-bottom:1rem;">📊</div>
        <div style="font-size:1.25rem; font-weight:600; color:#64748b; margin-bottom:0.5rem;">
            No portfolio analyzed yet
        </div>
        <div style="font-size:0.9rem;">
            Enter your holdings in the sidebar and click <strong>Analyze Portfolio</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    rec = result.get("portfolio_recommendation", "HOLD")
    p_sent = result.get("portfolio_sentiment_score", 0.5)
    p_risk = result.get("portfolio_risk_score", 0.5)
    p_comp = result.get("portfolio_composite_score", 0.5)
    conf = result.get("portfolio_confidence", "MEDIUM")
    holdings_table = result.get("holdings_table", [])

    # ── Portfolio header banner ───────────────────────────────────────────────
    n_holdings = len(holdings_table)
    symbols_str = " · ".join(h["symbol"] for h in holdings_table)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg,#1e3a8a 0%,#3b82f6 100%);
                color:white; border-radius:0.75rem; padding:1.5rem 2rem; margin-bottom:1.25rem;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
            <div>
                <div style="font-size:0.85rem; opacity:0.8; margin-bottom:0.25rem;">
                    {n_holdings}-Holding Portfolio · {result.get('analysis_date','')[:10]}
                </div>
                <div style="font-size:1.3rem; font-weight:600; letter-spacing:0.03em;">{symbols_str}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.85rem; opacity:0.8;">Portfolio Recommendation</div>
                <div style="font-size:1.8rem; font-weight:800; letter-spacing:0.05em;">{rec}</div>
                <div style="font-size:0.8rem; opacity:0.75;">{conf} confidence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Score gauges ──────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(_gauge(p_sent, "Portfolio Sentiment", "#10b981"), use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(_gauge(p_risk, "Portfolio Risk", "#dc2626"), use_container_width=True, config={"displayModeBar": False})
    with g3:
        st.plotly_chart(_gauge(p_comp, "Composite Score", "#3b82f6"), use_container_width=True, config={"displayModeBar": False})

    # ── Executive summary ─────────────────────────────────────────────────────
    st.markdown("#### 📋 Executive Summary")
    rec_css = _rec_css(rec)
    st.markdown(f"""
    <div class="{rec_css}">
        <p style="margin:0; line-height:1.6; color:#1e293b;">
            {result.get('executive_summary', '')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── Alerts ───────────────────────────────────────────────────────────────
    conc = result.get("concentration_risks", [])
    corr = result.get("correlated_risks", [])

    if conc or corr:
        col_a, col_b = st.columns(2)
        with col_a:
            if conc:
                st.markdown("##### ⚠️ Concentration Risk")
                for c in conc:
                    st.warning(
                        f"**{c['symbol']}** represents {c['weight']:.0%} of the portfolio "
                        "(threshold: 25%)"
                    )
        with col_b:
            if corr:
                st.markdown("##### 🔗 Correlated Risks")
                for c in corr:
                    st.info(
                        f"**{c['category'].title()}** risk shared across: "
                        f"{', '.join(c['symbols'])}"
                    )
        st.markdown("")

    # ── Per-holding comparison table ──────────────────────────────────────────
    st.markdown("#### 📋 Per-Holding Comparison")
    if holdings_table:
        df = pd.DataFrame(holdings_table)
        df["weight_pct"] = df["weight"].apply(lambda w: f"{w:.0%}")
        df_display = df[["symbol", "weight_pct", "recommendation",
                          "sentiment_score", "risk_score", "composite_score"]].copy()
        df_display.columns = ["Symbol", "Weight", "Recommendation",
                               "Sentiment", "Risk", "Composite"]
        df_display[["Sentiment", "Risk", "Composite"]] = df_display[
            ["Sentiment", "Risk", "Composite"]
        ].round(3)

        # Colour-coded bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Sentiment",
            x=df["symbol"],
            y=df["sentiment_score"],
            marker_color="#10b981",
            opacity=0.85,
        ))
        fig_bar.add_trace(go.Bar(
            name="Risk",
            x=df["symbol"],
            y=df["risk_score"],
            marker_color="#dc2626",
            opacity=0.85,
        ))
        fig_bar.add_trace(go.Bar(
            name="Composite",
            x=df["symbol"],
            y=df["composite_score"],
            marker_color="#3b82f6",
            opacity=0.85,
        ))
        fig_bar.update_layout(
            barmode="group",
            height=260,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0, 1], title="Score"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,1)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # ── Risk correlation heatmap ──────────────────────────────────────────────
    st.markdown("#### 🗺️ Risk Category Heatmap")
    st.caption("Shows which risk categories are present in each holding (red = present).")
    heatmap_fig = _build_heatmap(result)
    if heatmap_fig is not None:
        st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.info("Heatmap requires individual risk data (available after live analysis).")

    # ── Weight allocation pie ─────────────────────────────────────────────────
    st.markdown("#### 🥧 Portfolio Allocation")
    if holdings_table:
        fig_pie = px.pie(
            pd.DataFrame(holdings_table),
            names="symbol",
            values="weight",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig_pie.update_layout(
            height=280,
            margin=dict(l=0, r=0, t=10, b=0),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.download_button(
        label="⬇️ Download JSON",
        data=json.dumps(
            {k: v for k, v in result.items() if not k.startswith("_")},
            indent=2,
            default=str,
        ),
        file_name=f"portfolio_analysis_{result.get('analysis_date','')[:10]}.json",
        mime="application/json",
    )
