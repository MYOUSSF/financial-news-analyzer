"""
News Monitor Page — Real-time financial news feed with sentiment filtering.
Streamlit multi-page app: streamlit_app/pages/1_News_Monitor.py
"""
import sys
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Monitor — Financial Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared CSS (mirrors app.py) ───────────────────────────────────────────────
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
    .metric-card {
        background-color: #f8fafc;
        padding: 1.2rem 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .sentiment-positive { color: #10b981; font-weight: 600; }
    .sentiment-negative { color: #dc2626; font-weight: 600; }
    .sentiment-neutral  { color: #6b7280; font-weight: 600; }
    .news-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s;
    }
    .news-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .source-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1e3a8a;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.5rem;
        border-radius: 9999px;
        margin-right: 0.5rem;
    }
    .tag {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        font-size: 0.7rem;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        margin-right: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "AAPL"
if "nm_articles" not in st.session_state:
    st.session_state.nm_articles = []
if "nm_last_fetch" not in st.session_state:
    st.session_state.nm_last_fetch = None


# ── Helpers ───────────────────────────────────────────────────────────────────

MOCK_ARTICLES = [
    {
        "title": "{sym} Reports Record Q4 Revenue, Services Segment Drives Growth",
        "source": "Bloomberg",
        "published_at": datetime.now() - timedelta(hours=2),
        "sentiment": "Positive",
        "score": 0.88,
        "summary": "Company beat Wall Street estimates with revenue growth driven by the services segment, which surged 16% year-over-year. Analysts raised price targets following the announcement.",
        "url": "https://example.com/1",
        "tags": ["earnings", "revenue", "growth"],
    },
    {
        "title": "Analysts Upgrade {sym} to Strong Buy After Earnings Beat",
        "source": "Reuters",
        "published_at": datetime.now() - timedelta(hours=5),
        "sentiment": "Positive",
        "score": 0.81,
        "summary": "Three major investment banks raised their price targets citing stronger-than-expected margin expansion and robust forward guidance from management.",
        "url": "https://example.com/2",
        "tags": ["analyst", "upgrade", "earnings"],
    },
    {
        "title": "{sym} Faces Regulatory Scrutiny Over Market Practices in EU",
        "source": "Financial Times",
        "published_at": datetime.now() - timedelta(hours=9),
        "sentiment": "Negative",
        "score": 0.22,
        "summary": "European regulators opened a formal investigation into alleged anti-competitive practices. Shares fell 1.4% on the news. The company said it would cooperate fully with authorities.",
        "url": "https://example.com/3",
        "tags": ["regulatory", "EU", "investigation"],
    },
    {
        "title": "Institutional Investors Increase {sym} Holdings in Q3",
        "source": "WSJ",
        "published_at": datetime.now() - timedelta(hours=14),
        "sentiment": "Positive",
        "score": 0.74,
        "summary": "13-F filings reveal major funds added positions during the quarter. Institutional ownership rose to 74% of outstanding shares, signaling continued confidence.",
        "url": "https://example.com/4",
        "tags": ["institutional", "ownership", "holdings"],
    },
    {
        "title": "{sym} Supply Chain Pressures Ease as Asian Partners Ramp Production",
        "source": "CNBC",
        "published_at": datetime.now() - timedelta(hours=20),
        "sentiment": "Neutral",
        "score": 0.55,
        "summary": "Management confirmed lead times are returning to normal. Components availability improved across key product lines. Analysts view this as a modest positive for gross margins.",
        "url": "https://example.com/5",
        "tags": ["supply chain", "operations", "margins"],
    },
    {
        "title": "Global Tech Sector Faces Headwinds from Rising Bond Yields",
        "source": "MarketWatch",
        "published_at": datetime.now() - timedelta(hours=26),
        "sentiment": "Negative",
        "score": 0.31,
        "summary": "Rising 10-year treasury yields pressure growth stock valuations. Tech ETFs saw $2.1B outflows this week. Sector rotation into value names accelerating.",
        "url": "https://example.com/6",
        "tags": ["macro", "rates", "sector"],
    },
    {
        "title": "{sym} Named to S&P 500 ESG Index; Sustainability Score Improves",
        "source": "Barron's",
        "published_at": datetime.now() - timedelta(hours=31),
        "sentiment": "Positive",
        "score": 0.69,
        "summary": "The inclusion reflects improved emissions reporting and governance scores. ESG-focused funds may increase allocations as a result of the index addition.",
        "url": "https://example.com/7",
        "tags": ["ESG", "index", "sustainability"],
    },
    {
        "title": "{sym} CFO Sells $12M in Shares; Insider Activity Under Scrutiny",
        "source": "Seeking Alpha",
        "published_at": datetime.now() - timedelta(hours=38),
        "sentiment": "Negative",
        "score": 0.29,
        "summary": "The CFO exercised options and sold a portion of the resulting shares. While pre-planned, the transaction has drawn attention from retail investors monitoring insider activity.",
        "url": "https://example.com/8",
        "tags": ["insider", "selling", "CFO"],
    },
]


def _get_articles(symbol: str, source_filter: list, sentiment_filter: str, keyword: str) -> list:
    """Return filtered mock articles for the given symbol."""
    articles = [
        {**a, "title": a["title"].replace("{sym}", symbol)}
        for a in MOCK_ARTICLES
    ]
    # Source filter
    if source_filter and "All" not in source_filter:
        articles = [a for a in articles if a["source"] in source_filter]
    # Sentiment filter
    if sentiment_filter != "All":
        articles = [a for a in articles if a["sentiment"] == sentiment_filter]
    # Keyword filter
    if keyword:
        kw = keyword.lower()
        articles = [
            a for a in articles
            if kw in a["title"].lower() or kw in a["summary"].lower()
            or any(kw in t for t in a["tags"])
        ]
    return articles


def _sentiment_badge(sentiment: str) -> str:
    icons = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
    cls = f"sentiment-{sentiment.lower()}"
    return f"<span class='{cls}'>{icons.get(sentiment, '⚪')} {sentiment}</span>"


def _time_ago(dt: datetime) -> str:
    delta = datetime.now() - dt
    hours = int(delta.total_seconds() // 3600)
    if hours == 0:
        return f"{int(delta.total_seconds() // 60)}m ago"
    if hours < 24:
        return f"{hours}h ago"
    return f"{delta.days}d ago"


def _render_article(article: dict, idx: int) -> None:
    """Render a single news article card."""
    sentiment = article["sentiment"]
    score = article["score"]
    bar_color = "#10b981" if sentiment == "Positive" else "#dc2626" if sentiment == "Negative" else "#6b7280"

    tags_html = "".join(f"<span class='tag'>#{t}</span>" for t in article.get("tags", []))

    with st.container():
        st.markdown(f"""
        <div class="news-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.4rem;">
                <div style="flex:1; padding-right:1rem;">
                    <span class="source-badge">{article['source']}</span>
                    <span style="color:#94a3b8; font-size:0.78rem;">{_time_ago(article['published_at'])}</span>
                </div>
                <div style="text-align:right; min-width:100px;">
                    {_sentiment_badge(sentiment)}
                    <div style="font-size:0.78rem; color:#94a3b8; margin-top:2px;">Score: {score:.2f}</div>
                </div>
            </div>
            <div style="font-weight:600; font-size:1rem; color:#1e293b; margin-bottom:0.4rem; line-height:1.4;">
                {article['title']}
            </div>
            <div style="color:#475569; font-size:0.875rem; margin-bottom:0.5rem; line-height:1.5;">
                {article['summary']}
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>{tags_html}</div>
                <a href="{article['url']}" style="font-size:0.78rem; color:#3b82f6; text-decoration:none;">Read full article →</a>
            </div>
            <div style="margin-top:0.6rem; height:4px; background:#f1f5f9; border-radius:2px;">
                <div style="width:{score*100:.0f}%; height:100%; background:{bar_color}; border-radius:2px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Symbol")
    symbol = st.text_input(
        "Stock Symbol",
        value=st.session_state.selected_symbol,
        key="nm_symbol_input",
    ).upper()
    st.session_state.selected_symbol = symbol

    st.markdown("### ⚙️ Filters")
    source_filter = st.multiselect(
        "News Sources",
        ["All", "Bloomberg", "Reuters", "CNBC", "WSJ", "Financial Times", "MarketWatch", "Barron's", "Seeking Alpha"],
        default=["All"],
        key="nm_source_filter",
    )
    sentiment_filter = st.selectbox(
        "Sentiment",
        ["All", "Positive", "Negative", "Neutral"],
        key="nm_sentiment_filter",
    )
    sort_by = st.selectbox(
        "Sort By",
        ["Most Recent", "Highest Score", "Lowest Score"],
        key="nm_sort_by",
    )
    keyword = st.text_input("Keyword Search", placeholder="e.g. earnings, merger…", key="nm_keyword")

    st.markdown("---")
    days_back = st.slider("Date Range (days)", 1, 30, 7, key="nm_days")

    fetch_btn = st.button("🔄 Refresh Feed", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Feed Stats")
    all_articles = _get_articles(symbol, source_filter, sentiment_filter, keyword)
    pos = sum(1 for a in all_articles if a["sentiment"] == "Positive")
    neg = sum(1 for a in all_articles if a["sentiment"] == "Negative")
    neu = sum(1 for a in all_articles if a["sentiment"] == "Neutral")
    st.metric("Total Articles", len(all_articles))
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢", pos)
    col2.metric("🔴", neg)
    col3.metric("🟡", neu)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown(f'<h1 class="main-header">📰 News Monitor</h1>', unsafe_allow_html=True)
st.markdown(f"Live news feed for **{symbol}** · Last {days_back} days · {len(all_articles)} articles")

st.markdown("---")

# ── Top metrics row ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
avg_score = sum(a["score"] for a in all_articles) / max(len(all_articles), 1)
col1.metric("Avg Sentiment Score", f"{avg_score:.2f}", delta=f"+0.05 vs prev week")
col2.metric("Positive Articles", f"{pos} ({pos/max(len(all_articles),1):.0%})")
col3.metric("Negative Articles", f"{neg} ({neg/max(len(all_articles),1):.0%})")
col4.metric("Sources Tracked", len({a["source"] for a in all_articles}))

st.markdown("---")

# ── Score timeline chart ──────────────────────────────────────────────────────
with st.expander("📈 Sentiment Score Timeline", expanded=True):
    sorted_by_time = sorted(all_articles, key=lambda x: x["published_at"])
    df_timeline = pd.DataFrame([{
        "Time": a["published_at"].strftime("%m/%d %H:%M"),
        "Score": a["score"],
        "Sentiment": a["sentiment"],
        "Title": a["title"][:50] + "…",
    } for a in sorted_by_time])

    if not df_timeline.empty:
        color_map = {"Positive": "#10b981", "Negative": "#dc2626", "Neutral": "#6b7280"}
        fig = px.scatter(
            df_timeline,
            x="Time", y="Score",
            color="Sentiment",
            hover_data=["Title"],
            color_discrete_map=color_map,
            size=[12] * len(df_timeline),
        )
        fig.add_hline(y=0.6, line_dash="dot", line_color="#10b981", annotation_text="Positive threshold")
        fig.add_hline(y=0.4, line_dash="dot", line_color="#dc2626", annotation_text="Negative threshold")
        fig.update_layout(
            height=260, showlegend=True,
            yaxis_range=[0, 1],
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,1)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display.")

# ── Source distribution ───────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])
with col_left:
    st.markdown("#### 📡 Source Breakdown")
    source_counts = pd.Series([a["source"] for a in all_articles]).value_counts()
    fig_src = px.pie(
        values=source_counts.values,
        names=source_counts.index,
        hole=0.55,
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig_src.update_layout(height=220, showlegend=True, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_src, use_container_width=True)

with col_right:
    st.markdown("#### 🏷️ Top Topics")
    all_tags = [tag for a in all_articles for tag in a.get("tags", [])]
    tag_counts = pd.Series(all_tags).value_counts().head(8)
    fig_tags = go.Figure(go.Bar(
        x=tag_counts.values[::-1],
        y=tag_counts.index[::-1],
        orientation="h",
        marker_color="#3b82f6",
    ))
    fig_tags.update_layout(
        height=220, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,1)",
    )
    st.plotly_chart(fig_tags, use_container_width=True)

st.markdown("---")

# ── Article feed ──────────────────────────────────────────────────────────────
st.markdown(f"#### 📄 Articles ({len(all_articles)})")

# Sort
if sort_by == "Most Recent":
    display_articles = sorted(all_articles, key=lambda x: x["published_at"], reverse=True)
elif sort_by == "Highest Score":
    display_articles = sorted(all_articles, key=lambda x: x["score"], reverse=True)
else:
    display_articles = sorted(all_articles, key=lambda x: x["score"])

if not display_articles:
    st.info("No articles match your current filters. Try adjusting the source, sentiment, or keyword filters.")
else:
    for i, article in enumerate(display_articles):
        _render_article(article, i)
