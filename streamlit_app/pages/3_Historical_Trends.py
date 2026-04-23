"""
Historical Trends Page — Sentiment vs price correlation, rolling averages,
and vector store event search.
streamlit_app/pages/3_Historical_Trends.py
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
    page_title="Historical Trends — Financial Analyzer",
    page_icon="📈",
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
    .event-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
    }
    .event-card:hover { background: #f1f5f9; }
    .stat-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .corr-high   { color: #10b981; font-weight: 700; }
    .corr-medium { color: #f59e0b; font-weight: 700; }
    .corr-low    { color: #dc2626; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "AAPL"


# ── Mock data ─────────────────────────────────────────────────────────────────

def _price_and_sentiment(symbol: str, days: int) -> pd.DataFrame:
    """Generate correlated price + sentiment timeseries."""
    np.random.seed(hash(symbol) % 777)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    # Sentiment: smooth trend with noise
    sent_trend = np.linspace(0.55, 0.72, days)
    sentiment = np.clip(sent_trend + np.random.normal(0, 0.06, days), 0.05, 0.97)

    # Price: correlated with sentiment + independent random walk
    base_price = {"AAPL": 185, "GOOGL": 142, "MSFT": 415, "TSLA": 248, "NVDA": 870}.get(symbol, 200)
    price_changes = 0.35 * (sentiment - 0.5) + np.random.normal(0, 0.012, days)
    prices = base_price * np.cumprod(1 + price_changes)

    volume_base = np.random.randint(40_000_000, 80_000_000)
    volume = volume_base + np.random.randint(-10_000_000, 20_000_000, days)

    return pd.DataFrame({
        "Date": dates,
        "Price": np.round(prices, 2),
        "Sentiment": np.round(sentiment, 4),
        "Volume": volume.astype(int),
        "Price_Change_Pct": np.round(price_changes * 100, 2),
    })


def _rolling_stats(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()
    df[f"Price_MA{window}"] = df["Price"].rolling(window).mean()
    df[f"Sent_MA{window}"] = df["Sentiment"].rolling(window).mean()
    df["Sent_Upper"] = df[f"Sent_MA{window}"] + df["Sentiment"].rolling(window).std()
    df["Sent_Lower"] = df[f"Sent_MA{window}"] - df["Sentiment"].rolling(window).std()
    return df


MOCK_EVENTS = [
    {"date": "2024-03-30", "title": "Q4 Earnings Beat — Revenue +15% YoY",       "sentiment": 0.88, "price_impact": +2.3,  "tags": ["earnings"]},
    {"date": "2024-03-15", "title": "EU Regulatory Investigation Announced",       "sentiment": 0.22, "price_impact": -1.8,  "tags": ["regulatory"]},
    {"date": "2024-03-05", "title": "Major Analyst Upgrade to Strong Buy",         "sentiment": 0.81, "price_impact": +1.1,  "tags": ["analyst"]},
    {"date": "2024-02-20", "title": "New Product Line Unveiled at Annual Keynote", "sentiment": 0.79, "price_impact": +0.8,  "tags": ["product"]},
    {"date": "2024-02-10", "title": "CFO Comments on Macro Headwinds",             "sentiment": 0.41, "price_impact": -0.9,  "tags": ["macro"]},
    {"date": "2024-01-28", "title": "Institutional Buying Hits 5-Year High",       "sentiment": 0.74, "price_impact": +0.6,  "tags": ["institutional"]},
    {"date": "2024-01-15", "title": "Supply Chain Issues Partially Resolved",      "sentiment": 0.58, "price_impact": +0.3,  "tags": ["operations"]},
    {"date": "2023-12-20", "title": "Board Approves $10B Share Buyback Program",  "sentiment": 0.82, "price_impact": +1.5,  "tags": ["buyback"]},
]

VECTOR_STORE_RESULTS = [
    {"score": 0.94, "title": "Federal Reserve Rate Hike Impact on Tech Valuations",      "date": "2023-07-26", "symbol": "SPY", "source": "Bloomberg"},
    {"score": 0.88, "title": "Interest Rate Expectations Drive Sector Rotation",          "date": "2023-09-20", "symbol": "SPY", "source": "Reuters"},
    {"score": 0.81, "title": "Tech Earnings Season Kicks Off With Mixed Results",         "date": "2023-10-17", "symbol": "AAPL","source": "CNBC"},
    {"score": 0.76, "title": "Inflation Data Surprises Markets; Growth Stocks Pressured", "date": "2023-11-14", "symbol": "SPY", "source": "WSJ"},
]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Symbol")
    symbol = st.text_input("Stock Symbol", value=st.session_state.selected_symbol).upper()
    st.session_state.selected_symbol = symbol

    st.markdown("### ⚙️ Settings")
    days_back = st.slider("Historical Period (days)", 30, 180, 90, step=30, key="ht_days")
    ma_window = st.slider("Moving Average Window (days)", 3, 21, 7, key="ht_ma")
    show_volume = st.checkbox("Show Volume", value=True, key="ht_volume")
    show_bands = st.checkbox("Show Sentiment Bands (±1σ)", value=True, key="ht_bands")

    st.markdown("---")
    st.markdown("### 🔍 Event Search")
    event_query = st.text_input("Search historical events", placeholder="e.g. interest rate", key="ht_query")
    event_symbol_filter = st.text_input("Filter by symbol", placeholder="Leave blank for all", key="ht_event_sym")

    st.markdown("---")
    st.markdown("### 📊 Statistics")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">📈 Historical Trends</h1>', unsafe_allow_html=True)
st.markdown(f"Sentiment × price correlation for **{symbol}** · {days_back}-day window")
st.markdown("---")

# Build data
df = _price_and_sentiment(symbol, days_back)
df = _rolling_stats(df, ma_window)
corr = float(df["Price_Change_Pct"].corr(df["Sentiment"]))

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
price_return = (df["Price"].iloc[-1] / df["Price"].iloc[0] - 1) * 100
avg_sentiment = df["Sentiment"].mean()
sentiment_trend = df["Sentiment"].iloc[-7:].mean() - df["Sentiment"].iloc[:7].mean()
max_drawdown = ((df["Price"] / df["Price"].cummax()) - 1).min() * 100

col1.metric("Period Return",      f"{price_return:+.1f}%")
col2.metric("Avg Sentiment",      f"{avg_sentiment:.2f}",  delta=f"{sentiment_trend:+.2f} trend")
col3.metric("Sentiment–Price Corr", f"{corr:.2f}",
            delta="Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak")
col4.metric("Max Drawdown",       f"{max_drawdown:.1f}%")
col5.metric("Avg Daily Volume",   f"{df['Volume'].mean()/1e6:.1f}M")

# Correlation interpretation
if corr > 0.5:
    st.success(f"🔗 **Strong positive correlation ({corr:.2f})** — sentiment closely predicts price direction.")
elif corr > 0.3:
    st.info(f"📊 **Moderate correlation ({corr:.2f})** — sentiment is a useful but not decisive signal.")
elif corr > 0:
    st.warning(f"〰️ **Weak correlation ({corr:.2f})** — other factors may be dominating price action.")
else:
    st.error(f"↔️ **Negative correlation ({corr:.2f})** — sentiment and price are moving in opposite directions.")

st.markdown("---")

# ── Dual-axis: Price + Sentiment ──────────────────────────────────────────────
st.markdown("#### 📊 Price vs Sentiment Overlay")
fig = go.Figure()

# Price line (primary y)
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Price"],
    name="Price ($)", yaxis="y1",
    line=dict(color="#1e3a8a", width=2),
    mode="lines",
))
fig.add_trace(go.Scatter(
    x=df["Date"], y=df[f"Price_MA{ma_window}"],
    name=f"Price MA{ma_window}", yaxis="y1",
    line=dict(color="#93c5fd", width=1.5, dash="dash"),
    mode="lines",
))

# Sentiment area (secondary y)
fig.add_trace(go.Scatter(
    x=df["Date"], y=df["Sentiment"],
    name="Sentiment Score", yaxis="y2",
    line=dict(color="#10b981", width=1.8),
    mode="lines",
    opacity=0.8,
))
fig.add_trace(go.Scatter(
    x=df["Date"], y=df[f"Sent_MA{ma_window}"],
    name=f"Sentiment MA{ma_window}", yaxis="y2",
    line=dict(color="#6ee7b7", width=1.2, dash="dot"),
    mode="lines",
))

if show_bands:
    fig.add_trace(go.Scatter(
        x=pd.concat([df["Date"], df["Date"][::-1]]),
        y=pd.concat([df["Sent_Upper"], df["Sent_Lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(16,185,129,0.08)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±1σ Band", yaxis="y2",
        showlegend=True,
    ))

# Volume bars (secondary y, scaled down)
if show_volume:
    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["Volume"] / df["Volume"].max() * 0.15,  # scale to 0–0.15 on sentiment axis
        name="Volume (normalized)", yaxis="y2",
        marker_color="rgba(148,163,184,0.25)",
        showlegend=True,
    ))

fig.update_layout(
    height=420,
    yaxis=dict(title="Price ($)", side="left", showgrid=True, gridcolor="#f1f5f9"),
    yaxis2=dict(title="Sentiment Score", side="right", overlaying="y",
                range=[-0.05, 1.1], showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,1)",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Scatter: correlation plot ─────────────────────────────────────────────────
col_scatter, col_events = st.columns([1, 1])

with col_scatter:
    st.markdown("#### 🔵 Correlation: Sentiment vs Next-Day Return")
    df_scatter = df.copy()
    df_scatter["Next_Return"] = df_scatter["Price_Change_Pct"].shift(-1)
    df_scatter = df_scatter.dropna()

    # Manual OLS trendline via numpy (avoids statsmodels dependency)
    x_vals = df_scatter["Sentiment"].to_numpy()
    y_vals = df_scatter["Next_Return"].to_numpy()
    m, b = np.polyfit(x_vals, y_vals, 1)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_line = m * x_line + b

    fig_scatter = px.scatter(
        df_scatter, x="Sentiment", y="Next_Return",
        color="Next_Return",
        color_continuous_scale=["#dc2626", "#f1f5f9", "#10b981"],
        labels={"Sentiment": "Sentiment Score (Day T)", "Next_Return": "Price Return % (Day T+1)"},
    )
    fig_scatter.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color="#3b82f6", width=2, dash="dash"),
        name=f"Trend (slope={m:.2f})",
        showlegend=False,
    ))
    fig_scatter.update_layout(
        height=320, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(f"Pearson r = {corr:.3f} · slope = {m:.3f} · {len(df_scatter)} data points")

with col_events:
    st.markdown("#### 📌 Key Market Events")
    for ev in MOCK_EVENTS[:5]:
        impact_color = "#10b981" if ev["price_impact"] > 0 else "#dc2626"
        impact_arrow = "▲" if ev["price_impact"] > 0 else "▼"
        tags = " ".join(f"<span style='background:#eff6ff;color:#1e3a8a;font-size:0.7rem;padding:1px 5px;border-radius:4px;'>{t}</span>" for t in ev["tags"])
        st.markdown(f"""
        <div class="event-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="flex:1;">
                    <div style="font-size:0.75rem; color:#94a3b8;">{ev['date']}</div>
                    <div style="font-weight:600; color:#1e293b; font-size:0.9rem; margin:2px 0;">{ev['title']}</div>
                    <div>{tags}</div>
                </div>
                <div style="text-align:right; padding-left:0.75rem;">
                    <div style="color:{impact_color}; font-weight:700; font-size:1rem;">
                        {impact_arrow} {abs(ev['price_impact']):.1f}%
                    </div>
                    <div style="font-size:0.75rem; color:#94a3b8;">Sentiment: {ev['sentiment']:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Heatmap: sentiment by day of week / month ─────────────────────────────────
st.markdown("#### 🗓️ Sentiment Heatmap — Day of Week vs Week Number")
df["DayOfWeek"] = df["Date"].dt.day_name()
df["Week"] = df["Date"].dt.strftime("W%U")
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
df_filtered_weekdays = df[df["DayOfWeek"].isin(day_order)]

if not df_filtered_weekdays.empty:
    pivot = df_filtered_weekdays.pivot_table(
        values="Sentiment", index="DayOfWeek", columns="Week", aggfunc="mean"
    ).reindex(day_order)
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=["#dc2626", "#f1f5f9", "#10b981"],
        zmin=0, zmax=1,
        aspect="auto",
        text_auto=".2f",
    )
    fig_heat.update_layout(
        height=250, margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_showscale=True,
        xaxis_title="Week", yaxis_title="",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ── Vector store semantic search ──────────────────────────────────────────────
st.markdown("#### 🔍 Historical Event Semantic Search")
st.markdown("Search the vector store for similar past market events (powered by ChromaDB).")

search_col, _ = st.columns([2, 1])
with search_col:
    query_input = st.text_input(
        "Search query",
        value=event_query or "interest rate hikes inflation",
        placeholder="e.g. 'earnings beat guidance raised'",
        key="ht_search_input",
        label_visibility="collapsed",
    )

if st.button("🔎 Search Historical Events", key="ht_search_btn") or query_input:
    with st.spinner("Searching vector store…"):
        # Try real vector store first; fall back to mock
        results = []
        try:
            from src.utils.vector_store import VectorStore
            vs = VectorStore()
            where = {"symbol": event_symbol_filter.upper()} if event_symbol_filter.strip() else None
            results = vs.search(query=query_input, n_results=5, where=where, threshold=0.3)
        except Exception:
            results = []  # will show mock below

    if results:
        for r in results:
            meta = r.get("metadata", {})
            st.markdown(f"""
            <div class="event-card">
                <div style="display:flex; justify-content:space-between;">
                    <div>
                        <span style="background:#eff6ff;color:#1e3a8a;font-size:0.75rem;padding:1px 6px;border-radius:4px;">{meta.get('symbol','N/A')}</span>
                        <span style="color:#94a3b8; font-size:0.75rem; margin-left:0.5rem;">{meta.get('source','N/A')} · {meta.get('published_at','')[:10]}</span>
                    </div>
                    <span style="color:#3b82f6; font-size:0.78rem; font-weight:600;">Similarity: {r['score']:.2f}</span>
                </div>
                <div style="font-weight:600; color:#1e293b; margin-top:0.3rem;">{meta.get('title', r['document'][:120])}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Show mock results for demo
        filtered_mock = VECTOR_STORE_RESULTS
        if event_symbol_filter.strip():
            filtered_mock = [r for r in filtered_mock if r["symbol"] == event_symbol_filter.upper()]

        if filtered_mock:
            st.caption("📝 Demo results (run `init_db.py` and add your news data for live results)")
            for r in filtered_mock:
                st.markdown(f"""
                <div class="event-card">
                    <div style="display:flex; justify-content:space-between;">
                        <div>
                            <span style="background:#eff6ff;color:#1e3a8a;font-size:0.75rem;padding:1px 6px;border-radius:4px;">{r['symbol']}</span>
                            <span style="color:#94a3b8; font-size:0.75rem; margin-left:0.5rem;">{r['source']} · {r['date']}</span>
                        </div>
                        <span style="color:#3b82f6; font-size:0.78rem; font-weight:600;">Similarity: {r['score']:.2f}</span>
                    </div>
                    <div style="font-weight:600; color:#1e293b; margin-top:0.3rem;">{r['title']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No matching events found. Try a different query or run `python src/utils/init_db.py` to seed data.")