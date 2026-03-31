#!/usr/bin/env python
"""
Financial News Analyzer - Quick Demo Script

This script demonstrates the core functionality of the system.
Run this to see the agents in action!
"""
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=" * 80)
print("🚀 Financial News Analyzer - Interactive Demo")
print("=" * 80)
print()

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "─" * 80)
    print(f"📋 {title}")
    print("─" * 80 + "\n")


def demo_research_agent():
    """Demonstrate the Research Agent."""
    print_section("Research Agent Demo")
    
    print("The Research Agent gathers information from multiple sources.")
    print("It can fetch news, stock data, and economic indicators.\n")
    
    # Mock demonstration
    symbol = "AAPL"
    print(f"🔍 Researching: {symbol}")
    print(f"📅 Period: Last 7 days")
    print(f"⏳ Fetching data...\n")
    
    # Simulated results
    print("✅ Research Complete!")
    print("\nFindings:")
    print("  • Found 47 news articles")
    print("  • Stock up 2.3% over period")
    print("  • Major events: Q4 earnings beat, new product launch")
    print("  • Analyst upgrades: 3 firms raised price targets")
    print("  • Trading volume: Above average (+15%)")


def demo_sentiment_agent():
    """Demonstrate the Sentiment Agent."""
    print_section("Sentiment Agent Demo")
    
    print("The Sentiment Agent analyzes market sentiment using:")
    print("  • Machine Learning models (DistilBERT)")
    print("  • LLM reasoning (GPT-4)")
    print("  • Historical context\n")
    
    print("🔍 Analyzing sentiment for AAPL...")
    print("⏳ Processing 47 articles...\n")
    
    # Simulated results
    print("✅ Sentiment Analysis Complete!")
    print("\nResults:")
    print("  📊 Overall Sentiment: POSITIVE")
    print("  💯 Sentiment Score: 0.75 (75%)")
    print("  🎯 Confidence: 85%")
    print("\n  Positive Factors:")
    print("    • Strong Q4 earnings performance")
    print("    • Positive analyst ratings and upgrades")
    print("    • New product launches well-received")
    print("\n  Negative Factors:")
    print("    • Regulatory concerns in EU markets")
    print("    • Competitive pressure in key segments")


def demo_risk_agent():
    """Demonstrate the Risk Agent."""
    print_section("Risk Agent Demo")
    
    print("The Risk Agent identifies and assesses various risk categories:")
    print("  • Regulatory risks")
    print("  • Financial risks")
    print("  • Market risks")
    print("  • Operational risks")
    print("  • Volatility risks\n")
    
    print("🔍 Assessing risks for AAPL...")
    print("⏳ Analyzing data...\n")
    
    # Simulated results
    print("✅ Risk Assessment Complete!")
    print("\nOverall Risk Level: MEDIUM (55%)")
    print("\nIdentified Risks:")
    print("\n  ⚠️  HIGH - Volatility Risk (75% likelihood)")
    print("      → Increased price volatility in recent trading")
    print("      → Recommendation: Consider hedging strategies")
    
    print("\n  ⚠️  MEDIUM - Regulatory Risk (60% likelihood)")
    print("      → Ongoing investigations in EU markets")
    print("      → Recommendation: Monitor developments closely")
    
    print("\n  ⚠️  MEDIUM - Financial Risk (55% likelihood)")
    print("      → Elevated debt levels vs. industry average")
    print("      → Recommendation: Review financial statements")
    
    print("\n  ✅ LOW - Operational Risk (35% likelihood)")
    print("      → Minor supply chain concerns")
    print("      → Recommendation: Continue monitoring")


def demo_complete_analysis():
    """Demonstrate a complete multi-agent analysis."""
    print_section("Complete Multi-Agent Analysis")
    
    print("This demonstrates the full analysis pipeline:")
    print("Research → Sentiment → Risk → Summary\n")
    
    symbol = "AAPL"
    print(f"🎯 Target: {symbol}")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"⏱️  Analysis Period: 7 days\n")
    
    print("🔄 Running multi-agent analysis...\n")
    
    stages = [
        ("Research Agent", "Gathering news and market data"),
        ("Sentiment Agent", "Analyzing market sentiment"),
        ("Risk Agent", "Identifying risk factors"),
        ("Summary Agent", "Synthesizing findings")
    ]
    
    for agent, task in stages:
        print(f"  ▶️  {agent}: {task}...")
        import time
        time.sleep(0.5)
        print(f"  ✅ {agent}: Complete")
    
    print("\n" + "=" * 80)
    print("📄 INVESTMENT RESEARCH REPORT")
    print("=" * 80)
    
    report = f"""
Symbol: {symbol}
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
Period: 7 days

EXECUTIVE SUMMARY
─────────────────
Overall Assessment: POSITIVE with MODERATE RISK
Recommendation: HOLD with positive long-term outlook
Confidence Level: 85%

KEY FINDINGS
────────────
News Coverage: 47 articles analyzed
  • Predominantly positive coverage (68%)
  • Major focus on earnings and product launches
  • Some regulatory concerns noted

Sentiment Analysis:
  • Overall Sentiment: POSITIVE (75%)
  • Confidence: 85%
  • Trend: Improving over past week

Risk Assessment:
  • Overall Risk Level: MEDIUM (55%)
  • Primary Concern: Volatility (HIGH)
  • Secondary Concerns: Regulatory (MEDIUM), Financial (MEDIUM)

DETAILED ANALYSIS
─────────────────
Recent Performance:
  • Stock up 2.3% over analysis period
  • Trading volume above average (+15%)
  • Q4 earnings beat analyst expectations

Positive Catalysts:
  ✅ Strong earnings performance
  ✅ Product innovation and launches
  ✅ Analyst upgrades and positive ratings
  ✅ Institutional buying activity

Risk Factors:
  ⚠️  Increased market volatility
  ⚠️  Regulatory scrutiny in EU
  ⚠️  Elevated debt levels
  ⚠️  Competitive market pressures

INVESTMENT RECOMMENDATION
─────────────────────────
Action: HOLD
Rationale: Current risk/reward profile supports maintaining positions.
          Strong fundamentals offset by moderate risk factors.

Risk Management:
  • Monitor volatility and consider hedging
  • Stay informed on regulatory developments
  • Review position sizing given current risk levels
  • Set stop-loss orders to limit downside

Price Targets:
  • Bull Case: Continued earnings growth and positive sentiment
  • Base Case: Stable performance with moderate volatility
  • Bear Case: Regulatory headwinds and market correction

─────────────────────────────────────────────────────────────
This report was generated by Financial News Analyzer AI System
Powered by LangChain Multi-Agent Architecture
"""
    
    print(report)


def demo_api_usage():
    """Demonstrate API usage."""
    print_section("REST API Demo")
    
    print("The system provides a comprehensive REST API for programmatic access.\n")
    
    print("Available Endpoints:")
    print("\n📊 Analysis Endpoints:")
    print("  POST /api/analyze              - Complete stock analysis")
    print("  GET  /api/stocks/{symbol}/news - Get recent news")
    print("  POST /api/sentiment/analyze    - Sentiment analysis")
    print("  POST /api/risks/detect         - Risk assessment")
    
    print("\n🔍 Search Endpoints:")
    print("  POST /api/search/semantic      - Semantic search")
    print("  GET  /api/symbols/trending     - Trending symbols")
    
    print("\n📑 Reports:")
    print("  POST /api/reports/generate     - Generate report")
    
    print("\n💡 System:")
    print("  GET  /health                   - Health check")
    print("  GET  /api/stats                - System statistics")
    
    print("\n" + "─" * 40)
    print("Example API Call:")
    print("─" * 40)
    
    api_example = """
import requests

response = requests.post(
    "http://localhost:8000/api/analyze",
    json={
        "symbol": "AAPL",
        "days_back": 7,
        "include_sentiment": True,
        "include_risk": True
    }
)

result = response.json()
print(f"Sentiment: {result['sentiment']['overall_sentiment']}")
print(f"Risk Level: {result['risk']['risk_level']}")
"""
    print(api_example)


def demo_dashboard():
    """Demonstrate dashboard features."""
    print_section("Streamlit Dashboard Demo")
    
    print("Interactive web dashboard with multiple features:\n")
    
    print("📈 Overview Tab:")
    print("  • Key metrics and KPIs")
    print("  • Sentiment trend charts")
    print("  • Risk distribution visualization")
    print("  • Recent insights and highlights")
    
    print("\n📰 News Monitor Tab:")
    print("  • Real-time news feed")
    print("  • Source filtering")
    print("  • Sentiment-based filtering")
    print("  • Article summaries with scores")
    
    print("\n💭 Sentiment Analysis Tab:")
    print("  • Overall sentiment gauge")
    print("  • Sentiment distribution charts")
    print("  • Key themes and topics")
    print("  • Historical sentiment trends")
    
    print("\n⚠️ Risk Assessment Tab:")
    print("  • Overall risk score")
    print("  • Risk factor breakdown")
    print("  • Severity and likelihood ratings")
    print("  • Risk management recommendations")
    
    print("\n📑 Report Generator Tab:")
    print("  • Customizable report templates")
    print("  • Multiple output formats (PDF, DOCX, HTML)")
    print("  • Chart and visualization inclusion")
    print("  • Email delivery option")
    
    print("\n🚀 To launch the dashboard:")
    print("   streamlit run streamlit_app/app.py")
    print("\n   Then open: http://localhost:8501")


def main():
    """Main demo function."""
    
    print("Welcome to the Financial News Analyzer demo!")
    print("\nThis system uses AI-powered multi-agent architecture to analyze")
    print("financial news, assess sentiment, and identify risks.\n")
    
    print("Built with:")
    print("  🦜 LangChain - Multi-agent orchestration")
    print("  🤖 GPT-4 - Language understanding")
    print("  📊 Plotly - Interactive visualizations")
    print("  🔥 Streamlit - Web dashboard")
    print("  ⚡ FastAPI - REST API")
    
    print("\n" + "=" * 80)
    
    while True:
        print("\n\nDemo Options:")
        print("  1. Research Agent Demo")
        print("  2. Sentiment Agent Demo")
        print("  3. Risk Agent Demo")
        print("  4. Complete Multi-Agent Analysis")
        print("  5. API Usage Demo")
        print("  6. Dashboard Features")
        print("  7. Exit")
        
        choice = input("\nSelect an option (1-7): ").strip()
        
        if choice == "1":
            demo_research_agent()
        elif choice == "2":
            demo_sentiment_agent()
        elif choice == "3":
            demo_risk_agent()
        elif choice == "4":
            demo_complete_analysis()
        elif choice == "5":
            demo_api_usage()
        elif choice == "6":
            demo_dashboard()
        elif choice == "7":
            print("\n" + "=" * 80)
            print("Thank you for exploring Financial News Analyzer!")
            print("=" * 80)
            print("\n📚 Next Steps:")
            print("  • Read the documentation in docs/")
            print("  • Try the Streamlit dashboard")
            print("  • Explore the API at /docs")
            print("  • Check out example notebooks")
            print("\n🚀 Get started:")
            print("   streamlit run streamlit_app/app.py")
            print("\n💼 Portfolio ready!")
            print("   This project demonstrates:")
            print("     ✅ LangChain multi-agent systems")
            print("     ✅ Real-world API integration")
            print("     ✅ Production-ready architecture")
            print("     ✅ Full-stack development")
            print("     ✅ ML/AI implementation")
            print("\nGood luck with your job search! 🎯\n")
            break
        else:
            print("Invalid option. Please select 1-7.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Please check your setup and try again.")
