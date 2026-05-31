#!/usr/bin/env python
"""
Financial News Analyzer — Interactive Demo

Usage:
    python demo.py               # defaults to AAPL
    python demo.py --symbol TSLA # pre-fills the symbol
"""
import argparse
import os
import sys
import textwrap
from datetime import datetime
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# LLM detection (mirrors _build_llm priority order in analysis_chain.py)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_llm() -> tuple:
    """Return (is_configured: bool, provider_name: str)."""
    if os.getenv("OPENAI_API_KEY"):
        return True, "OpenAI"
    if os.getenv("ANTHROPIC_API_KEY"):
        return True, "Anthropic"
    if os.getenv("OLLAMA_BASE_URL"):
        return True, "Ollama"
    return False, "none"


_LLM_CONFIGURED, _LLM_PROVIDER = _detect_llm()

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded mock data (used by DemoChain when no LLM is configured)
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_TEXT = (
    "Apple reported Q4 earnings with EPS of $1.46, beating the $1.39 consensus. "
    "Revenue grew 2% YoY to $89.5B. The Services segment hit a record $23.2B. "
    "Three analyst firms raised price targets following the earnings beat. "
    "Institutional buyers increased positions. An ongoing EU App Store antitrust "
    "investigation remains a regulatory risk. Trading volume ran 15% above the "
    "30-day average in the sessions after the earnings release."
)

_MOCK_RESEARCH: Dict[str, Any] = {
    "findings": _SAMPLE_TEXT,
    "sources_used": ["NewsTool", "StockTool"],
    "period_days": 7,
    "research_date": datetime.now().isoformat(),
}

_MOCK_SENTIMENT: Dict[str, Any] = {
    "overall_sentiment": "POSITIVE",
    "sentiment_score": 0.72,
    "confidence": 0.84,
    "text_count": 12,
    "dedup_count": 3,
    "key_insights": {
        "positive_factors": [
            "Strong Q4 earnings beat estimates",
            "Services revenue at record levels",
            "Institutional buying activity",
        ],
        "negative_factors": [
            "EU regulatory investigations ongoing",
            "Competitive pressure in smartphone segment",
        ],
        "market_implications": ["Momentum likely to continue short-term"],
    },
}

_MOCK_RISK: Dict[str, Any] = {
    "overall_risk_score": 0.42,
    "risk_level": "MEDIUM",
    "identified_risks": [
        {
            "category": "regulatory",
            "description": "EU antitrust investigation into App Store practices",
            "severity": "MEDIUM",
            "likelihood": 0.6,
        },
        {
            "category": "market",
            "description": "Increased competition in key smartphone markets",
            "severity": "LOW",
            "likelihood": 0.5,
        },
        {
            "category": "volatility",
            "description": "Price swings around earnings season",
            "severity": "MEDIUM",
            "likelihood": 0.55,
        },
    ],
    "alerts": [],
    "recommendations": [
        "Monitor EU regulatory developments closely",
        "Consider hedging strategies given elevated volatility",
    ],
}

_MOCK_SUMMARY: Dict[str, Any] = {
    "analysis_date": datetime.now().isoformat(),
    "period_days": 7,
    "executive_summary": (
        "Apple reported strong Q4 earnings with beats across revenue and EPS, "
        "driven by record Services performance. Market sentiment is broadly "
        "positive with analyst upgrades supporting a constructive outlook. "
        "Moderate regulatory and volatility risk are the primary concerns."
    ),
    "key_positives": [
        "Q4 EPS beat consensus by 5%",
        "Services revenue at record $23.2B",
        "Three analyst price target upgrades",
        "Above-average institutional buying",
    ],
    "key_negatives": [
        "EU App Store investigation ongoing",
        "YoY revenue growth modest at 2%",
        "Elevated near-term price volatility",
    ],
    "recommendation": "BUY",
    "recommendation_rationale": (
        "Strong fundamentals and positive sentiment outweigh moderate regulatory "
        "and volatility risks. The earnings beat and Services growth trajectory "
        "support a BUY with a 12-month horizon."
    ),
    "confidence": 0.75,
    "confidence_label": "HIGH",
    "action_items": [
        "Review position sizing relative to current volatility",
        "Monitor EU regulatory developments over the next 30 days",
        "Set a stop-loss 8% below current price",
    ],
    "scores": {
        "sentiment_score": 0.72,
        "risk_score": 0.42,
        "composite_score": 0.65,
    },
}


class DemoChain:
    """Fallback that returns hardcoded realistic results when no LLM is configured."""

    def analyze_stock(self, symbol: str, days_back: int = 7, **kwargs) -> Dict[str, Any]:
        sym = symbol.upper()
        return {
            **_MOCK_SUMMARY,
            "symbol": sym,
            "_research": {**_MOCK_RESEARCH, "symbol": sym},
            "_sentiment": {**_MOCK_SENTIMENT, "symbol": sym},
            "_risk": {**_MOCK_RISK, "symbol": sym},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline initialization
# ─────────────────────────────────────────────────────────────────────────────

def _get_chain(session: Dict[str, Any]) -> Any:
    """Return the cached chain, initializing it on first call."""
    if "chain" not in session:
        if not _LLM_CONFIGURED:
            session["chain"] = DemoChain()
        else:
            print("  Initializing analysis pipeline...", end=" ", flush=True)
            try:
                from src.chains.analysis_chain import FinancialAnalysisChain
                session["chain"] = FinancialAnalysisChain(verbose=False)
                print("ready.")
            except Exception as exc:
                print(f"\n  Pipeline init failed: {exc}")
                print("  Falling back to demo mode.")
                session["chain"] = DemoChain()
    return session["chain"]


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def _divider(title: str = "") -> None:
    print()
    print("─" * 80)
    if title:
        print(f"  {title}")
        print("─" * 80)


def _bar(score: float, width: int = 20) -> str:
    filled = max(0, min(width, int(score * width)))
    return f"[{'#' * filled}{' ' * (width - filled)}] {score:.0%}"


def _display_research(result: Dict[str, Any]) -> None:
    symbol = result.get("symbol", "?")
    _divider(f"Research Agent — {symbol}")
    date_key = "research_date" if "research_date" in result else "analysis_date"
    date_val = str(result.get(date_key, "N/A"))[:10]
    print(f"  Date:    {date_val}")
    print(f"  Period:  {result.get('period_days', 7)} days")
    sources = result.get("sources_used") or []
    print(f"  Sources: {', '.join(sources) if sources else 'N/A'}")
    print()
    findings = result.get("findings", "No findings available.")
    for line in textwrap.wrap(str(findings), width=76):
        print(f"  {line}")


def _display_sentiment(result: Dict[str, Any]) -> None:
    symbol = result.get("symbol", "?")
    _divider(f"Sentiment Agent — {symbol}")
    label = result.get("overall_sentiment", "N/A")
    score = float(result.get("sentiment_score", 0.5))
    confidence = float(result.get("confidence", 0.0))
    count = result.get("text_count", "N/A")
    dedup = int(result.get("dedup_count", 0))
    print(f"  Sentiment:      {label}")
    print(f"  Score:          {_bar(score)}")
    print(f"  Confidence:     {_bar(confidence)}")
    suffix = f"  ({dedup} duplicates removed)" if dedup else ""
    print(f"  Texts analyzed: {count}{suffix}")
    insights = result.get("key_insights", {})
    if isinstance(insights, dict):
        pos = insights.get("positive_factors", [])
        neg = insights.get("negative_factors", [])
        if pos:
            print("\n  Positive factors:")
            for item in pos:
                print(f"    + {item}")
        if neg:
            print("\n  Negative factors:")
            for item in neg:
                print(f"    - {item}")


def _display_risk(result: Dict[str, Any]) -> None:
    symbol = result.get("symbol", "?")
    _divider(f"Risk Agent — {symbol}")
    level = result.get("risk_level", "N/A")
    score = float(result.get("overall_risk_score", 0.0))
    print(f"  Risk level: {level}")
    print(f"  Risk score: {_bar(score)}")
    risks = result.get("identified_risks", [])
    if risks:
        print()
        print(f"  {'Category':<14} {'Severity':<10} {'Likelihood':<12} Description")
        print("  " + "─" * 66)
        for r in risks:
            cat = str(r.get("category", "?")).capitalize()
            sev = str(r.get("severity", "?"))
            lik = float(r.get("likelihood", 0.0))
            desc = str(r.get("description", ""))[:42]
            print(f"  {cat:<14} {sev:<10} {lik:<12.0%} {desc}")
    alerts = result.get("alerts", [])
    if alerts:
        print(f"\n  {len(alerts)} alert(s) generated")
    recs = result.get("recommendations", [])
    if recs:
        print("\n  Recommendations:")
        for rec in recs:
            print(f"    * {rec}")


def _format_report(chain: Any, result: Dict[str, Any]) -> str:
    """Format a full analysis result as Markdown using SummaryAgent when available."""
    if hasattr(chain, "summary_agent"):
        return chain.summary_agent.generate_report_markdown(result)
    # DemoChain fallback — render inline (mirrors generate_report_markdown logic)
    sym = result.get("symbol", "N/A")
    date = str(result.get("analysis_date", datetime.now().isoformat()))[:10]
    scores = result.get("scores", {})
    positives = "\n".join(f"- {p}" for p in result.get("key_positives", ["No data"]))
    negatives = "\n".join(f"- {n}" for n in result.get("key_negatives", ["No data"]))
    actions = "\n".join(
        f"{i+1}. {a}" for i, a in enumerate(result.get("action_items", ["Monitor closely"]))
    )
    return (
        f"# Investment Research Report: {sym}\n\n"
        f"**Date:** {date}  \n"
        f"**Period:** {result.get('period_days', 7)} days  \n"
        f"**Confidence:** {result.get('confidence_label', 'N/A')}"
        f" ({result.get('confidence', 0):.0%})\n\n"
        f"---\n\n"
        f"## Executive Summary\n\n{result.get('executive_summary', 'Not available.')}\n\n"
        f"---\n\n"
        f"## Key Positives\n\n{positives}\n\n"
        f"## Key Negatives / Risks\n\n{negatives}\n\n"
        f"---\n\n"
        f"## Scores\n\n"
        f"| Metric | Score |\n|---|---|\n"
        f"| Sentiment Score | {scores.get('sentiment_score', 0):.2f} |\n"
        f"| Risk Score | {scores.get('risk_score', 0):.2f} |\n"
        f"| Composite Score | {scores.get('composite_score', 0):.2f} |\n\n"
        f"---\n\n"
        f"## Investment Recommendation\n\n"
        f"**{result.get('recommendation', 'N/A')}**\n\n"
        f"{result.get('recommendation_rationale', '')}\n\n"
        f"---\n\n"
        f"## Action Items\n\n{actions}\n\n"
        f"---\n\n"
        f"*Generated by Financial News Analyzer — "
        f"{'Live Analysis' if _LLM_CONFIGURED else 'Demo Mode'}*\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Menu handlers
# ─────────────────────────────────────────────────────────────────────────────

def run_research(symbol: str, session: Dict[str, Any]) -> None:
    chain = _get_chain(session)
    if isinstance(chain, DemoChain):
        result = {**_MOCK_RESEARCH, "symbol": symbol.upper()}
        session["research"] = result
        _display_research(result)
        print("\n  (Demo mode — representative mock data)")
        return

    print(f"  Running ResearchAgent for {symbol}...")
    try:
        result = chain.research_agent.execute({"symbol": symbol, "days_back": 7})
        session["research"] = result
        _display_research(result)
    except Exception as exc:
        print(f"  ResearchAgent failed: {exc}")


def run_sentiment(symbol: str, session: Dict[str, Any]) -> None:
    chain = _get_chain(session)
    if isinstance(chain, DemoChain):
        result = {**_MOCK_SENTIMENT, "symbol": symbol.upper()}
        session["sentiment"] = result
        _display_sentiment(result)
        print("\n  (Demo mode — representative mock data)")
        return

    research = session.get("research", {})
    text = research.get("findings") or _SAMPLE_TEXT
    if not research:
        print("  No prior research in session — analyzing sample text.")
    print(f"  Running SentimentAgent for {symbol}...")
    try:
        result = chain.sentiment_agent.execute({
            "symbol": symbol,
            "text": text,
            "context": f"Financial analysis for {symbol}",
        })
        session["sentiment"] = result
        _display_sentiment(result)
    except Exception as exc:
        print(f"  SentimentAgent failed: {exc}")


def run_risk(symbol: str, session: Dict[str, Any]) -> None:
    chain = _get_chain(session)
    if isinstance(chain, DemoChain):
        result = {**_MOCK_RISK, "symbol": symbol.upper()}
        session["risk"] = result
        _display_risk(result)
        print("\n  (Demo mode — representative mock data)")
        return

    news_data = session.get("research", {})
    sentiment = session.get("sentiment", {})
    if not news_data:
        print("  No prior research in session — risk analysis will use limited context.")
    print(f"  Running RiskAgent for {symbol}...")
    try:
        result = chain.risk_agent.execute({
            "symbol": symbol,
            "news_data": news_data,
            "market_data": {},
            "sentiment": sentiment,
        })
        session["risk"] = result
        _display_risk(result)
    except Exception as exc:
        print(f"  RiskAgent failed: {exc}")


def run_full_analysis(symbol: str, days_back: int, session: Dict[str, Any]) -> None:
    chain = _get_chain(session)
    _divider(f"Full Analysis — {symbol} (last {days_back} days)")

    if isinstance(chain, DemoChain):
        print("  (Demo mode — representative mock data)\n")
        result = chain.analyze_stock(symbol, days_back)
    else:
        print("  Running pipeline: Research → Sentiment + Risk (parallel) → Summary")
        try:
            result = chain.analyze_stock(symbol, days_back)
        except Exception as exc:
            print(f"  Pipeline failed: {exc}")
            return

    session["research"] = result.get("_research", {})
    session["sentiment"] = result.get("_sentiment", {})
    session["risk"] = result.get("_risk", {})

    elapsed = result.get("_elapsed_seconds")
    if elapsed:
        print(f"  Completed in {elapsed}s\n")

    print(_format_report(chain, result))


def show_api() -> None:
    _divider("REST API Endpoints")
    print("  Analysis:")
    print("    POST /api/analyze              — Full stock analysis")
    print("    GET  /api/stocks/{symbol}/news — Recent news for a symbol")
    print("    POST /api/sentiment/analyze    — Sentiment analysis")
    print("    POST /api/risks/detect         — Risk assessment")
    print()
    print("  Search:")
    print("    POST /api/search/semantic      — Semantic search")
    print("    GET  /api/symbols/trending     — Trending symbols")
    print()
    print("  Reports:")
    print("    POST /api/reports/generate     — Generate markdown report")
    print()
    print("  System:")
    print("    GET  /health                   — Health check")
    print("    GET  /api/stats                — System statistics")
    print()
    print("  Example:")
    print()
    print("    import requests")
    print('    r = requests.post("http://localhost:8000/api/analyze",')
    print('                      json={"symbol": "AAPL", "days_back": 7,')
    print('                            "include_sentiment": True, "include_risk": True})')
    print("    print(r.json()['recommendation'])")
    print()
    print("  Start the server:")
    print("    .venv/bin/uvicorn src.api.main:app --reload --port 8000")
    print("    Docs at: http://localhost:8000/docs")


def show_dashboard() -> None:
    _divider("Streamlit Dashboard")
    print("  Pages:")
    print("    Overview     — Key metrics, sentiment trend, risk distribution")
    print("    News Monitor — Live news feed with sentiment filtering")
    print("    Sentiment    — Gauge, distribution charts, historical trends")
    print("    Risk         — Risk score breakdown, severity / likelihood table")
    print("    Reports      — Full report generation (Markdown / PDF)")
    print()
    print("  Launch:")
    print("    .venv/bin/streamlit run streamlit_app/Main.py")
    print("    Then open: http://localhost:8501")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Financial News Analyzer — Demo")
    parser.add_argument(
        "--symbol", default="AAPL", metavar="TICKER",
        help="Stock symbol to analyze (default: AAPL)",
    )
    args = parser.parse_args()
    symbol = args.symbol.upper()

    print("=" * 80)
    print("  Financial News Analyzer — Interactive Demo")
    print("=" * 80)
    print()
    print("  Stack:  LangChain multi-agent pipeline, FastAPI, Streamlit")
    print("  Agents: Research → Sentiment → Risk → Summary")
    print()
    if _LLM_CONFIGURED:
        print(f"  LLM: {_LLM_PROVIDER}  (live analysis enabled)")
    else:
        print("  No LLM key detected (OPENAI_API_KEY / ANTHROPIC_API_KEY / OLLAMA_BASE_URL).")
        print("  Running in demo mode — outputs are representative mock data.")
        print("  Add an LLM key to .env to enable live analysis.")
    print()
    print(f"  Symbol: {symbol}  (override with --symbol TSLA)")

    # Session state: chain and agent results shared across all menu options
    session: Dict[str, Any] = {}

    while True:
        print()
        print("─" * 80)
        print(f"  Symbol: {symbol}")
        print("─" * 80)
        print("    1. Research Agent       — fetch news & market data")
        print("    2. Sentiment Agent      — score market sentiment")
        print("    3. Risk Agent           — identify risk factors")
        print("    4. Full Analysis        — run complete pipeline, display report")
        print("    5. API Endpoints        — REST API reference")
        print("    6. Dashboard            — Streamlit launch instructions")
        print("    7. Change symbol")
        print("    8. Exit")

        choice = input("\n  Select (1-8): ").strip()

        if choice == "1":
            run_research(symbol, session)
        elif choice == "2":
            run_sentiment(symbol, session)
        elif choice == "3":
            run_risk(symbol, session)
        elif choice == "4":
            days_str = input("  Days back (default 7): ").strip()
            days_back = int(days_str) if days_str.isdigit() else 7
            run_full_analysis(symbol, days_back, session)
        elif choice == "5":
            show_api()
        elif choice == "6":
            show_dashboard()
        elif choice == "7":
            new_sym = input("  Enter new symbol: ").strip().upper()
            if new_sym:
                old_chain = session.get("chain")
                symbol = new_sym
                session = {}
                if old_chain is not None:
                    session["chain"] = old_chain
                print(f"  Symbol changed to {symbol}. Session state cleared.")
        elif choice == "8":
            print()
            print("=" * 80)
            print("  Thank you for exploring Financial News Analyzer!")
            print("  Next: .venv/bin/streamlit run streamlit_app/Main.py")
            print("=" * 80)
            print()
            break
        else:
            print("  Invalid choice — enter a number from 1 to 8.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    except Exception as exc:
        print(f"\nError: {exc}")
        print("Check your setup (.env, dependencies) and try again.")
