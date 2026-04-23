"""
Report Generator Page — Full investment research report with live agent pipeline
and downloadable output.
streamlit_app/pages/4_Report_Generator.py
"""
import sys
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

st.set_page_config(
    page_title="Report Generator — Financial Analyzer",
    page_icon="📑",
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
    .report-section {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .rec-buy      { background:#f0fdf4; border-left:5px solid #10b981; border-radius:0.5rem; padding:1rem; }
    .rec-hold     { background:#fff7ed; border-left:5px solid #f59e0b; border-radius:0.5rem; padding:1rem; }
    .rec-sell     { background:#fff1f2; border-left:5px solid #dc2626; border-radius:0.5rem; padding:1rem; }
    .rec-avoid    { background:#fef2f2; border-left:5px solid #991b1b; border-radius:0.5rem; padding:1rem; }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .badge-green  { background:#d1fae5; color:#065f46; }
    .badge-red    { background:#fee2e2; color:#991b1b; }
    .badge-yellow { background:#fef3c7; color:#92400e; }
    .badge-blue   { background:#dbeafe; color:#1e40af; }
    .pipeline-step {
        display: flex;
        align-items: center;
        padding: 0.5rem 0.75rem;
        border-radius: 0.4rem;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
    }
    .step-done    { background: #f0fdf4; color: #065f46; }
    .step-active  { background: #eff6ff; color: #1e40af; }
    .step-pending { background: #f8fafc; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "AAPL"
if "rg_report" not in st.session_state:
    st.session_state.rg_report = None
if "rg_symbol" not in st.session_state:
    st.session_state.rg_symbol = None


# ── Mock report generator ─────────────────────────────────────────────────────

def _generate_mock_report(symbol: str, days: int, report_type: str,
                          include_sentiment: bool, include_risk: bool,
                          include_charts: bool) -> dict:
    np.random.seed(hash(symbol) % 333)
    sent_score = round(np.random.uniform(0.55, 0.85), 2)
    risk_score  = round(np.random.uniform(0.35, 0.65), 2)
    composite   = round(0.5 * sent_score + 0.5 * (1 - risk_score), 2)

    if sent_score >= 0.75 and risk_score <= 0.35:
        rec = "STRONG BUY"
    elif sent_score >= 0.60 and risk_score <= 0.50:
        rec = "BUY"
    elif sent_score >= 0.40 and risk_score <= 0.65:
        rec = "HOLD"
    elif sent_score >= 0.25:
        rec = "SELL"
    else:
        rec = "AVOID"

    risk_level = "LOW" if risk_score < 0.35 else "HIGH" if risk_score > 0.65 else "MEDIUM"
    conf_label = "HIGH" if sent_score > 0.7 else "MEDIUM"

    return {
        "symbol": symbol,
        "report_type": report_type,
        "analysis_date": datetime.now().isoformat(),
        "period_days": days,
        "recommendation": rec,
        "confidence": round(0.75 + sent_score * 0.1, 2),
        "confidence_label": conf_label,
        "executive_summary": (
            f"{symbol} demonstrates a {rec.lower().replace('strong buy','strongly bullish').replace('buy','bullish').replace('hold','stable').replace('sell','bearish').replace('avoid','high-risk')} "
            f"outlook over the {days}-day analysis window. Sentiment analysis across {days * 6 + 3} articles "
            f"yields a score of {sent_score:.2f}, while the multi-factor risk model places overall risk at "
            f"{risk_level.lower()} ({risk_score:.2f}). The composite investment score of {composite:.2f} "
            f"supports a {rec} recommendation with {conf_label.lower()} confidence."
        ),
        "key_positives": [
            f"Strong earnings momentum — beat expectations for {np.random.randint(2,6)} consecutive quarters",
            "Services / recurring revenue growing at double-digit rates YoY",
            "Institutional ownership increased by 3.2% this quarter",
            "Balance sheet remains robust with positive free cash flow",
        ],
        "key_negatives": [
            "EU regulatory investigation creates near-term overhang",
            f"Valuation premium of {np.random.randint(15,35)}% vs sector peers warrants caution",
            "Macro environment (rising rates) compresses growth multiples",
        ],
        "action_items": [
            "Monitor regulatory headlines — any fines above $1B would be a material risk",
            f"Review position sizing; reduce if risk score exceeds 0.70",
            "Consider covered calls at current premium levels to reduce cost basis",
        ],
        "scores": {
            "sentiment_score": sent_score,
            "risk_score": risk_score,
            "composite_score": composite,
            "confidence": round(0.75 + sent_score * 0.1, 2),
        },
        "sentiment": {
            "overall_sentiment": "POSITIVE" if sent_score >= 0.6 else "NEUTRAL" if sent_score >= 0.4 else "NEGATIVE",
            "sentiment_score": sent_score,
            "confidence": round(sent_score * 0.95, 2),
            "articles_analyzed": days * 6 + 3,
        } if include_sentiment else None,
        "risk": {
            "risk_level": risk_level,
            "overall_risk_score": risk_score,
            "identified_risks": [
                {"category": "regulatory", "severity": "MEDIUM", "likelihood": 0.60, "description": "EU antitrust investigation ongoing"},
                {"category": "volatility",  "severity": "HIGH",   "likelihood": 0.72, "description": "Elevated implied volatility vs 90d average"},
                {"category": "financial",   "severity": "LOW",    "likelihood": 0.35, "description": "Debt-to-equity slightly above sector median"},
            ],
            "alerts": [{"severity": "HIGH", "message": "Volatility alert: IV rank at 68th percentile"}] if risk_score > 0.5 else [],
            "recommendations": [
                "Implement stop-loss at 8% below entry",
                "Consider collar strategy to cap downside",
                "Rebalance if portfolio weight exceeds 5%",
            ],
        } if include_risk else None,
    }


def _recommendation_css_class(rec: str) -> str:
    if "BUY" in rec:
        return "rec-buy"
    elif rec == "HOLD":
        return "rec-hold"
    elif rec == "SELL":
        return "rec-sell"
    return "rec-avoid"


def _rec_badge_class(rec: str) -> str:
    if "BUY" in rec:
        return "badge-green"
    elif rec == "HOLD":
        return "badge-yellow"
    return "badge-red"


def _risk_badge_class(level: str) -> str:
    return {"LOW": "badge-green", "MEDIUM": "badge-yellow", "HIGH": "badge-red"}.get(level, "badge-blue")


def _render_pipeline_steps(symbol: str, stages_done: int) -> None:
    steps = [
        ("📰", "Research Agent",  "Fetching news & market data"),
        ("💭", "Sentiment Agent", "Analyzing market sentiment"),
        ("⚠️", "Risk Agent",      "Assessing risk factors"),
        ("📋", "Summary Agent",   "Synthesizing final report"),
    ]
    for i, (icon, name, desc) in enumerate(steps):
        if i < stages_done:
            css = "step-done"
            status = "✅"
        elif i == stages_done:
            css = "step-active"
            status = "⏳"
        else:
            css = "step-pending"
            status = "○"
        st.markdown(f"""
        <div class="pipeline-step {css}">
            <span style="margin-right:0.5rem;">{status}</span>
            <span style="font-weight:600;">{icon} {name}</span>
            <span style="margin-left:0.5rem; opacity:0.65;">— {desc}</span>
        </div>
        """, unsafe_allow_html=True)


def _build_markdown_report(report: dict) -> str:
    """Convert report dict to a downloadable Markdown string."""
    r = report
    s = r.get("scores", {})
    lines = [
        f"# Investment Research Report: {r['symbol']}",
        f"",
        f"**Report Type:** {r['report_type'].title()}  ",
        f"**Date:** {r['analysis_date'][:10]}  ",
        f"**Period Covered:** {r['period_days']} days  ",
        f"**Confidence:** {r['confidence_label']} ({r['confidence']:.0%})  ",
        f"",
        f"---",
        f"",
        f"## Executive Summary",
        f"",
        r['executive_summary'],
        f"",
        f"---",
        f"",
        f"## Investment Recommendation",
        f"",
        f"### **{r['recommendation']}**",
        f"",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Sentiment Score | {s.get('sentiment_score', 0):.2f} |",
        f"| Risk Score | {s.get('risk_score', 0):.2f} |",
        f"| Composite Score | {s.get('composite_score', 0):.2f} |",
        f"",
        f"---",
        f"",
        f"## Key Positives",
        f"",
    ]
    for p in r.get("key_positives", []):
        lines.append(f"- {p}")
    lines += ["", "## Key Negatives / Risks", ""]
    for n in r.get("key_negatives", []):
        lines.append(f"- {n}")

    if r.get("sentiment"):
        sent = r["sentiment"]
        lines += [
            "", "---", "",
            "## Sentiment Analysis",
            "",
            f"- **Overall Sentiment:** {sent.get('overall_sentiment', 'N/A')}",
            f"- **Score:** {sent.get('sentiment_score', 0):.2f}",
            f"- **Confidence:** {sent.get('confidence', 0):.0%}",
            f"- **Articles Analyzed:** {sent.get('articles_analyzed', 0)}",
        ]

    if r.get("risk"):
        risk = r["risk"]
        lines += [
            "", "---", "",
            "## Risk Assessment",
            "",
            f"- **Risk Level:** {risk.get('risk_level', 'N/A')}",
            f"- **Overall Risk Score:** {risk.get('overall_risk_score', 0):.2f}",
            "",
            "### Identified Risks",
            "",
        ]
        for rk in risk.get("identified_risks", []):
            lines.append(f"- **{rk['category'].title()}** ({rk['severity']}) — {rk['description']}")
        lines += ["", "### Recommendations", ""]
        for rec in risk.get("recommendations", []):
            lines.append(f"- {rec}")

    lines += [
        "", "---", "",
        "## Action Items",
        "",
    ]
    for i, a in enumerate(r.get("action_items", []), 1):
        lines.append(f"{i}. {a}")

    lines += [
        "", "---", "",
        f"*Generated by Financial News Analyzer · {r['analysis_date'][:10]}*",
    ]
    return "\n".join(lines)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Symbol")
    symbol = st.text_input("Stock Symbol", value=st.session_state.selected_symbol).upper()
    st.session_state.selected_symbol = symbol

    st.markdown("### ⚙️ Report Options")
    report_type = st.selectbox(
        "Report Type",
        ["Comprehensive", "Quick Summary", "Risk Focus", "Sentiment Focus"],
        key="rg_type",
    )
    days_back = st.slider("Analysis Period (days)", 1, 30, 7, key="rg_days")

    st.markdown("### 📦 Include Sections")
    include_sentiment = st.checkbox("Sentiment Analysis", value=True, key="rg_sent")
    include_risk      = st.checkbox("Risk Assessment",    value=True, key="rg_risk")
    include_charts    = st.checkbox("Charts & Visuals",   value=True, key="rg_charts")
    include_actions   = st.checkbox("Action Items",       value=True, key="rg_actions")

    st.markdown("### 📤 Download Format")
    dl_format = st.selectbox("Format", ["Markdown (.md)", "JSON (.json)"], key="rg_format")

    st.markdown("---")
    generate_btn = st.button(
        "🚀 Generate Report",
        type="primary",
        use_container_width=True,
        key="rg_generate",
    )

    if st.session_state.rg_report:
        st.markdown("---")
        st.success("✅ Report ready")
        st.markdown(f"**Symbol:** {st.session_state.rg_symbol}")
        st.markdown(f"**Recommendation:** {st.session_state.rg_report.get('recommendation', 'N/A')}")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">📑 Report Generator</h1>', unsafe_allow_html=True)
st.markdown("AI-generated investment research reports powered by the multi-agent pipeline.")
st.markdown("---")

# ── Generate ──────────────────────────────────────────────────────────────────
if generate_btn:
    st.session_state.rg_report = None
    st.session_state.rg_symbol = symbol

    progress_placeholder = st.empty()
    pipeline_placeholder = st.empty()

    with progress_placeholder.container():
        progress = st.progress(0, text=f"Starting analysis for {symbol}…")

    stages = [
        (f"📰 Research Agent — fetching {days_back}d of news & market data…", 25),
        ("💭 Sentiment Agent — scoring articles across ML + LLM…", 50),
        ("⚠️ Risk Agent — evaluating risk factors and alerts…", 75),
        ("📋 Summary Agent — synthesizing final report…", 95),
    ]

    # Try real chain first
    report = None
    chain_error = None
    try:
        from src.chains.analysis_chain import FinancialAnalysisChain
        import time
        for i, (msg, pct) in enumerate(stages):
            progress_placeholder.empty()
            with progress_placeholder.container():
                st.progress(pct / 100, text=msg)
            with pipeline_placeholder.container():
                _render_pipeline_steps(symbol, i)

        chain = FinancialAnalysisChain(verbose=False)
        result = chain.analyze_stock(
            symbol=symbol,
            days_back=days_back,
            include_sentiment=include_sentiment,
            include_risk=include_risk,
        )
        if "error" not in result:
            report = result
    except EnvironmentError as e:
        chain_error = str(e)
    except Exception as e:
        chain_error = str(e)

    # Fall back to mock
    if report is None:
        import time
        for i, (msg, pct) in enumerate(stages):
            progress_placeholder.empty()
            with progress_placeholder.container():
                st.progress(pct / 100, text=msg)
            with pipeline_placeholder.container():
                _render_pipeline_steps(symbol, i)
            time.sleep(0.4)
        report = _generate_mock_report(symbol, days_back, report_type.lower(),
                                       include_sentiment, include_risk, include_charts)
        if chain_error:
            st.caption(f"ℹ️ Using demo report (chain error: {chain_error}). Configure an LLM key in `.env` for live reports.")

    progress_placeholder.empty()
    pipeline_placeholder.empty()

    with progress_placeholder.container():
        st.progress(1.0, text="✅ Report complete!")
    _render_pipeline_steps(symbol, 4)

    st.session_state.rg_report = report

# ── Render report ─────────────────────────────────────────────────────────────
report = st.session_state.rg_report

if report is None:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#94a3b8;">
        <div style="font-size:4rem; margin-bottom:1rem;">📋</div>
        <div style="font-size:1.25rem; font-weight:600; color:#64748b; margin-bottom:0.5rem;">
            No report generated yet
        </div>
        <div style="font-size:0.9rem;">
            Configure your options in the sidebar and click <strong>Generate Report</strong> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── Report header ──────────────────────────────────────────────────────────
    rec = report.get("recommendation", "HOLD")
    rec_cls = _recommendation_css_class(rec)
    rec_badge = _rec_badge_class(rec)
    scores = report.get("scores", {})

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                color:white; border-radius:0.75rem; padding:1.5rem 2rem; margin-bottom:1.25rem;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
            <div>
                <div style="font-size:0.85rem; opacity:0.8; margin-bottom:0.25rem;">
                    {report.get('report_type','Comprehensive').title()} Report · {report['analysis_date'][:10]} · {report['period_days']}d
                </div>
                <div style="font-size:2rem; font-weight:700;">{report['symbol']}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.85rem; opacity:0.8;">Recommendation</div>
                <div style="font-size:1.6rem; font-weight:800; letter-spacing:0.05em;">{rec}</div>
                <div style="font-size:0.8rem; opacity:0.75;">
                    {report.get('confidence_label','N/A')} confidence · {report.get('confidence',0):.0%}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Score gauges ───────────────────────────────────────────────────────────
    if include_charts:
        col1, col2, col3 = st.columns(3)
        for col, label, key, color_pos, color_neg in [
            (col1, "Sentiment Score", "sentiment_score", "#10b981", "#dc2626"),
            (col2, "Risk Score",      "risk_score",      "#dc2626", "#10b981"),  # inverted for risk
            (col3, "Composite Score", "composite_score", "#3b82f6", "#f59e0b"),
        ]:
            val = scores.get(key, 0.5)
            bar_color = color_pos if val >= 0.5 else color_neg
            col.markdown(f"""
            <div class="report-section" style="text-align:center; padding:1rem;">
                <div style="font-size:0.8rem; color:#6b7280; margin-bottom:0.3rem;">{label}</div>
                <div style="font-size:2.2rem; font-weight:700; color:{bar_color};">{val:.2f}</div>
                <div style="background:#f1f5f9; height:6px; border-radius:3px; margin-top:0.4rem;">
                    <div style="width:{val*100:.0f}%; height:100%; background:{bar_color}; border-radius:3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Executive summary ──────────────────────────────────────────────────────
    st.markdown("#### 📋 Executive Summary")
    st.markdown(f"""
    <div class="{rec_cls}">
        <p style="margin:0; line-height:1.6; color:#1e293b;">{report.get('executive_summary','')}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # ── Positives & Negatives ──────────────────────────────────────────────────
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("##### ✅ Key Positives")
        for p in report.get("key_positives", []):
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:0.5rem; margin-bottom:0.4rem;">
                <span style="color:#10b981; font-weight:700; flex-shrink:0;">▸</span>
                <span style="color:#1e293b; font-size:0.9rem;">{p}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_neg:
        st.markdown("##### ⚠️ Key Negatives")
        for n in report.get("key_negatives", []):
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:0.5rem; margin-bottom:0.4rem;">
                <span style="color:#dc2626; font-weight:700; flex-shrink:0;">▸</span>
                <span style="color:#1e293b; font-size:0.9rem;">{n}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ── Sentiment section ──────────────────────────────────────────────────────
    if include_sentiment and report.get("sentiment"):
        sent = report["sentiment"]
        st.markdown("#### 💭 Sentiment Analysis")
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Overall",  sent.get("overall_sentiment", "N/A"))
        s_col2.metric("Score",    f"{sent.get('sentiment_score', 0):.2f}")
        s_col3.metric("Confidence", f"{sent.get('confidence', 0):.0%}")
        s_col4.metric("Articles Analyzed", sent.get("articles_analyzed", 0))

    # ── Risk section ───────────────────────────────────────────────────────────
    if include_risk and report.get("risk"):
        risk = report["risk"]
        st.markdown("#### ⚠️ Risk Assessment")
        r_col1, r_col2 = st.columns([1, 2])

        with r_col1:
            risk_level = risk.get("risk_level", "MEDIUM")
            risk_score = risk.get("overall_risk_score", 0.5)
            level_colors = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#dc2626"}
            lc = level_colors.get(risk_level, "#6b7280")
            st.markdown(f"""
            <div style="background:{lc}15; border:2px solid {lc}; border-radius:0.5rem;
                        padding:1.2rem; text-align:center;">
                <div style="color:{lc}; font-weight:800; font-size:1.6rem;">{risk_level}</div>
                <div style="font-size:2rem; font-weight:700; color:#1e293b;">{risk_score:.2f}</div>
                <div style="font-size:0.78rem; color:#6b7280;">Risk Score</div>
            </div>
            """, unsafe_allow_html=True)

        with r_col2:
            identified = risk.get("identified_risks", [])
            if identified:
                sev_colors = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#dc2626", "CRITICAL": "#7c3aed"}
                cats = [r["category"].title() for r in identified]
                sevs = [r["severity"] for r in identified]
                likes = [r.get("likelihood", 0.5) for r in identified]
                fig_risk = go.Figure()
                fig_risk.add_trace(go.Bar(
                    x=cats, y=likes,
                    marker_color=[sev_colors.get(s, "#6b7280") for s in sevs],
                    text=[f"{s}<br>{l:.0%}" for s, l in zip(sevs, likes)],
                    textposition="auto",
                ))
                fig_risk.update_layout(
                    height=180, showlegend=False,
                    yaxis=dict(range=[0, 1], title="Likelihood"),
                    margin=dict(l=0, r=0, t=5, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(248,250,252,1)",
                )
                st.plotly_chart(fig_risk, use_container_width=True)

        if risk.get("alerts"):
            for alert in risk["alerts"]:
                st.warning(f"🚨 **{alert.get('severity','')} Alert** — {alert.get('message','')}")

        if risk.get("recommendations"):
            st.markdown("**Risk Recommendations:**")
            for rec_item in risk["recommendations"]:
                st.markdown(f"• {rec_item}")

    # ── Action items ───────────────────────────────────────────────────────────
    if include_actions and report.get("action_items"):
        st.markdown("#### 📌 Action Items")
        for i, action in enumerate(report["action_items"], 1):
            st.markdown(f"""
            <div style="display:flex; align-items:flex-start; gap:0.75rem; padding:0.6rem 0.9rem;
                        background:#f8fafc; border-radius:0.4rem; margin-bottom:0.4rem; border-left:3px solid #3b82f6;">
                <span style="background:#3b82f6; color:white; border-radius:50%;
                             width:1.3rem; height:1.3rem; display:flex; align-items:center;
                             justify-content:center; font-size:0.75rem; font-weight:700;
                             flex-shrink:0;">{i}</span>
                <span style="color:#1e293b; font-size:0.9rem;">{action}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⬇️ Download Report")
    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 2])

    md_content = _build_markdown_report(report)
    filename_stem = f"{symbol}_report_{report['analysis_date'][:10]}"

    with dl_col1:
        st.download_button(
            label="📄 Download Markdown",
            data=md_content,
            file_name=f"{filename_stem}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with dl_col2:
        json_content = json.dumps(
            {k: v for k, v in report.items() if not k.startswith("_")},
            indent=2, default=str,
        )
        st.download_button(
            label="🗄️ Download JSON",
            data=json_content,
            file_name=f"{filename_stem}.json",
            mime="application/json",
            use_container_width=True,
        )

    with dl_col3:
        st.info(
            "💡 **PDF export**: Copy the Markdown content into any Markdown → PDF converter, "
            "or use `pandoc report.md -o report.pdf` from the command line."
        )
