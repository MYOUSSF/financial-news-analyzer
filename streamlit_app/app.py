"""
Financial News Analyzer - Streamlit Dashboard
Main application interface for financial news analysis and research.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Page configuration
st.set_page_config(
    page_title="Financial News Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .risk-high {
        color: #dc2626;
        font-weight: bold;
    }
    .risk-medium {
        color: #f59e0b;
        font-weight: bold;
    }
    .risk-low {
        color: #10b981;
        font-weight: bold;
    }
    .sentiment-positive {
        color: #10b981;
    }
    .sentiment-negative {
        color: #dc2626;
    }
    .sentiment-neutral {
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = 'AAPL'


def main():
    """Main application entry point."""
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1e3a8a/ffffff?text=FinAnalyzer", 
                use_column_width=True)
        st.markdown("---")
        
        st.markdown("### 🎯 Quick Actions")
        
        # Symbol input
        symbol = st.text_input(
            "Stock Symbol",
            value=st.session_state.selected_symbol,
            help="Enter a stock symbol (e.g., AAPL, GOOGL, TSLA)"
        ).upper()
        
        st.session_state.selected_symbol = symbol
        
        # Analysis period
        days_back = st.slider(
            "Analysis Period (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to look back for analysis"
        )
        
        # Analysis options
        st.markdown("### ⚙️ Analysis Options")
        
        include_sentiment = st.checkbox("Sentiment Analysis", value=True)
        include_risk = st.checkbox("Risk Assessment", value=True)
        include_trends = st.checkbox("Historical Trends", value=False)
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            run_analysis(symbol, days_back, include_sentiment, include_risk, include_trends)
        
        st.markdown("---")
        
        # Recent searches
        st.markdown("### 📋 Recent Analyses")
        if st.session_state.analysis_history:
            for item in st.session_state.analysis_history[-5:]:
                if st.button(
                    f"{item['symbol']} - {item['date']}",
                    key=f"history_{item['symbol']}_{item['date']}",
                    use_container_width=True
                ):
                    st.session_state.selected_symbol = item['symbol']
        else:
            st.info("No recent analyses")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Financial News Analyzer**
        
        AI-powered tool for analyzing financial news, sentiment, and market risks.
        
        Built with:
        - 🦜 LangChain
        - 🤖 GPT-4
        - 📊 Plotly
        - 🔥 Streamlit
        """)
    
    # Main content
    st.markdown('<h1 class="main-header">📊 Financial News Analyzer</h1>', 
               unsafe_allow_html=True)
    st.markdown("AI-powered financial research assistant with multi-agent analysis")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "📰 News Monitor",
        "💭 Sentiment Analysis",
        "⚠️ Risk Assessment",
        "📑 Report Generator"
    ])
    
    with tab1:
        show_overview_tab(symbol, days_back)
    
    with tab2:
        show_news_monitor_tab(symbol, days_back)
    
    with tab3:
        show_sentiment_tab(symbol, days_back)
    
    with tab4:
        show_risk_tab(symbol, days_back)
    
    with tab5:
        show_report_generator_tab(symbol, days_back)


def run_analysis(symbol: str, days_back: int, include_sentiment: bool, 
                include_risk: bool, include_trends: bool):
    """
    Run comprehensive analysis on a symbol.
    
    Args:
        symbol: Stock symbol
        days_back: Number of days to analyze
        include_sentiment: Include sentiment analysis
        include_risk: Include risk assessment
        include_trends: Include trend analysis
    """
    with st.spinner(f"🔍 Analyzing {symbol}..."):
        try:
            # Simulate analysis (replace with actual agent calls)
            import time
            time.sleep(2)
            
            # Add to history
            st.session_state.analysis_history.append({
                'symbol': symbol,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'sentiment': 'Positive' if include_sentiment else 'N/A',
                'risk': 'Medium' if include_risk else 'N/A'
            })
            
            st.success(f"✅ Analysis complete for {symbol}!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")


def show_overview_tab(symbol: str, days_back: int):
    """Display overview dashboard."""
    st.markdown(f"### Overview for {symbol}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Sentiment",
            value="Positive",
            delta="↑ 12%",
            help="Aggregate sentiment from news analysis"
        )
    
    with col2:
        st.metric(
            label="Risk Level",
            value="Medium",
            delta="Stable",
            help="Overall risk assessment"
        )
    
    with col3:
        st.metric(
            label="News Articles",
            value="47",
            delta="+5",
            help="Number of articles analyzed"
        )
    
    with col4:
        st.metric(
            label="Confidence Score",
            value="85%",
            delta="↑ 3%",
            help="Analysis confidence level"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment trend chart
        st.markdown("#### 📈 Sentiment Trend")
        
        dates = pd.date_range(
            end=datetime.now(), 
            periods=days_back, 
            freq='D'
        )
        sentiment_data = pd.DataFrame({
            'Date': dates,
            'Sentiment Score': [0.65, 0.70, 0.68, 0.72, 0.75, 0.73, 0.78][:days_back]
        })
        
        fig = px.line(
            sentiment_data,
            x='Date',
            y='Sentiment Score',
            markers=True
        )
        fig.update_layout(
            height=300,
            showlegend=False,
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk distribution
        st.markdown("#### ⚠️ Risk Distribution")
        
        risk_data = pd.DataFrame({
            'Category': ['Regulatory', 'Financial', 'Market', 'Operational', 'Volatility'],
            'Risk Score': [0.4, 0.6, 0.5, 0.3, 0.7]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=risk_data['Category'],
                y=risk_data['Risk Score'],
                marker_color=['#10b981', '#f59e0b', '#10b981', '#10b981', '#dc2626']
            )
        ])
        fig.update_layout(height=300, showlegend=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent insights
    st.markdown("#### 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Positive Factors:**
        - Strong Q4 earnings beat expectations
        - Positive analyst upgrades
        - Increasing institutional ownership
        - New product launches well-received
        """)
    
    with col2:
        st.warning("""
        **Risk Factors:**
        - Increased market volatility
        - Regulatory scrutiny in EU markets
        - Supply chain concerns
        - Competitive pressure
        """)


def show_news_monitor_tab(symbol: str, days_back: int):
    """Display news monitoring interface."""
    st.markdown(f"### 📰 News Monitor for {symbol}")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_filter = st.multiselect(
            "News Sources",
            ["All", "Bloomberg", "Reuters", "CNBC", "WSJ", "FT"],
            default=["All"]
        )
    
    with col2:
        sentiment_filter = st.selectbox(
            "Sentiment Filter",
            ["All", "Positive", "Negative", "Neutral"]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["Relevance", "Date", "Sentiment Score"]
        )
    
    st.markdown("---")
    
    # Mock news articles
    articles = [
        {
            "title": f"{symbol} Reports Strong Q4 Earnings, Beats Expectations",
            "source": "Bloomberg",
            "date": "2024-03-30",
            "sentiment": "Positive",
            "score": 0.85,
            "summary": "Company reports 15% revenue growth year-over-year with improved margins."
        },
        {
            "title": f"Analysts Upgrade {symbol} Following Earnings Call",
            "source": "Reuters",
            "date": "2024-03-29",
            "sentiment": "Positive",
            "score": 0.78,
            "summary": "Major banks raise price targets citing strong fundamentals and growth outlook."
        },
        {
            "title": f"{symbol} Faces Regulatory Scrutiny in Europe",
            "source": "Financial Times",
            "date": "2024-03-28",
            "sentiment": "Negative",
            "score": 0.35,
            "summary": "EU regulators investigating potential antitrust violations."
        }
    ]
    
    for article in articles:
        with st.expander(f"📄 {article['title']} - {article['date']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Source:** {article['source']}")
                st.markdown(f"**Summary:** {article['summary']}")
            
            with col2:
                sentiment_class = f"sentiment-{article['sentiment'].lower()}"
                st.markdown(
                    f"**Sentiment:** <span class='{sentiment_class}'>{article['sentiment']}</span>",
                    unsafe_allow_html=True
                )
                st.progress(article['score'])
                st.caption(f"Score: {article['score']:.2f}")


def show_sentiment_tab(symbol: str, days_back: int):
    """Display sentiment analysis interface."""
    st.markdown(f"### 💭 Sentiment Analysis for {symbol}")
    
    # Overall sentiment gauge
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col2:
        sentiment_score = 0.75
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Sentiment"},
            delta={'reference': 60},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 40], 'color': "#fee2e2"},
                    {'range': [40, 60], 'color': "#fef3c7"},
                    {'range': [60, 100], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Sentiment breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Sentiment Distribution")
        
        sentiment_dist = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [32, 10, 5]
        })
        
        fig = px.pie(
            sentiment_dist,
            values='Count',
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={
                'Positive': '#10b981',
                'Neutral': '#6b7280',
                'Negative': '#dc2626'
            }
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔑 Key Themes")
        
        themes = [
            ("Earnings Performance", 0.9, "Positive"),
            ("Product Innovation", 0.8, "Positive"),
            ("Regulatory Concerns", 0.4, "Negative"),
            ("Market Competition", 0.6, "Neutral"),
            ("Growth Outlook", 0.75, "Positive")
        ]
        
        for theme, score, sentiment in themes:
            color = {
                'Positive': '#10b981',
                'Negative': '#dc2626',
                'Neutral': '#6b7280'
            }[sentiment]
            
            st.markdown(f"**{theme}**")
            st.progress(score)
            st.markdown(
                f"<span style='color: {color}'>{sentiment} ({score:.0%})</span>",
                unsafe_allow_html=True
            )
            st.markdown("")


def show_risk_tab(symbol: str, days_back: int):
    """Display risk assessment interface."""
    st.markdown(f"### ⚠️ Risk Assessment for {symbol}")
    
    # Risk score
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        risk_score = 0.55
        risk_level = "MEDIUM"
        
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; background-color: #fef3c7; 
                    border-radius: 0.5rem; border: 2px solid #f59e0b;'>
            <h2 class='risk-medium'>Risk Level: {risk_level}</h2>
            <h1 style='font-size: 4rem; margin: 0;'>{risk_score:.0%}</h1>
            <p>Overall Risk Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk factors
    st.markdown("#### 🎯 Identified Risk Factors")
    
    risks = [
        {
            "category": "Volatility",
            "severity": "HIGH",
            "likelihood": 0.75,
            "description": "Increased price volatility in recent trading sessions"
        },
        {
            "category": "Regulatory",
            "severity": "MEDIUM",
            "likelihood": 0.60,
            "description": "Ongoing regulatory investigations in EU markets"
        },
        {
            "category": "Financial",
            "severity": "MEDIUM",
            "likelihood": 0.55,
            "description": "Elevated debt levels compared to industry average"
        },
        {
            "category": "Operational",
            "severity": "LOW",
            "likelihood": 0.35,
            "description": "Minor supply chain disruptions reported"
        }
    ]
    
    for risk in risks:
        with st.expander(f"⚠️ {risk['category']} Risk - {risk['severity']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Description:** {risk['description']}")
                st.markdown(f"**Likelihood:** {risk['likelihood']:.0%}")
            
            with col2:
                severity_colors = {
                    "CRITICAL": "#dc2626",
                    "HIGH": "#f59e0b",
                    "MEDIUM": "#f59e0b",
                    "LOW": "#10b981"
                }
                st.markdown(
                    f"<h3 style='color: {severity_colors[risk['severity']]}'>"
                    f"{risk['severity']}</h3>",
                    unsafe_allow_html=True
                )
    
    # Recommendations
    st.markdown("#### 💡 Risk Management Recommendations")
    
    st.info("""
    **Recommended Actions:**
    1. Monitor volatility closely and consider hedging strategies
    2. Stay informed on regulatory developments
    3. Review position sizing given current risk levels
    4. Implement stop-loss orders to limit downside
    5. Diversify exposure across multiple sectors
    """)


def show_report_generator_tab(symbol: str, days_back: int):
    """Display report generation interface."""
    st.markdown(f"### 📑 Investment Report Generator for {symbol}")
    
    st.markdown("""
    Generate comprehensive investment research reports combining all analysis components.
    """)
    
    # Report options
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Comprehensive Analysis", "Quick Summary", "Risk Focus", "Sentiment Focus"]
        )
        
        include_charts = st.checkbox("Include Charts and Visualizations", value=True)
        include_recommendations = st.checkbox("Include Investment Recommendations", value=True)
    
    with col2:
        output_format = st.selectbox(
            "Output Format",
            ["PDF", "Word Document", "HTML", "Markdown"]
        )
        
        email_report = st.checkbox("Email Report", value=False)
    
    # Generate button
    if st.button("📄 Generate Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            import time
            time.sleep(2)
            st.success("✅ Report generated successfully!")
            
            # Preview
            st.markdown("---")
            st.markdown("### 📋 Report Preview")
            
            st.markdown(f"""
            # Investment Research Report: {symbol}
            **Date:** {datetime.now().strftime('%B %d, %Y')}
            **Analysis Period:** {days_back} days
            
            ## Executive Summary
            
            Based on comprehensive analysis of recent market data, news sentiment, and risk factors,
            {symbol} demonstrates a **positive outlook** with **moderate risk** considerations.
            
            ### Key Findings:
            - **Sentiment Score:** 75% (Positive)
            - **Risk Level:** Medium (55%)
            - **News Coverage:** 47 articles analyzed
            - **Recommendation:** HOLD with positive outlook
            
            ## Detailed Analysis
            
            ### News Summary
            Recent news has been predominantly positive, with strong Q4 earnings beating 
            expectations and positive analyst upgrades. However, regulatory concerns in 
            European markets present a moderate risk factor.
            
            ### Sentiment Analysis
            Overall market sentiment towards {symbol} is positive, with 68% of analyzed 
            articles showing favorable sentiment. Key positive drivers include earnings 
            performance and product innovation.
            
            ### Risk Assessment
            Primary risks identified:
            1. Increased volatility (HIGH)
            2. Regulatory scrutiny (MEDIUM)
            3. Financial leverage (MEDIUM)
            
            ### Investment Recommendation
            **HOLD** with positive long-term outlook. Current risk/reward profile supports 
            maintaining positions while monitoring regulatory developments.
            
            ---
            *This report was generated by Financial News Analyzer AI System*
            """)
            
            # Download button
            st.download_button(
                label="⬇️ Download Full Report",
                data=f"Investment Report for {symbol}",
                file_name=f"{symbol}_analysis_report.pdf",
                mime="application/pdf"
            )


if __name__ == "__main__":
    main()
