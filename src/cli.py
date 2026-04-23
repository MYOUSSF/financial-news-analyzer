"""
Financial News Analyzer — Command Line Interface

Usage examples:
    python -m src.cli analyze --symbol AAPL --days 7
    python -m src.cli report  --symbol TSLA --output report.md
    python -m src.cli monitor --symbols AAPL,GOOGL,MSFT
    python -m src.cli search  --query "interest rate hikes" --limit 5
    python -m src.cli init-db --reset
"""
import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Make src/ importable when run as `python -m src.cli`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ===========================================================================
# Sub-command handlers
# ===========================================================================

def cmd_analyze(args: argparse.Namespace) -> int:
    """Run full multi-agent analysis for a symbol."""
    from src.chains.analysis_chain import FinancialAnalysisChain

    print(f"\n🔍 Analyzing {args.symbol} — last {args.days} days…\n")
    try:
        chain = FinancialAnalysisChain(verbose=args.verbose)
        result = chain.analyze_stock(
            symbol=args.symbol,
            days_back=args.days,
            include_sentiment=not args.no_sentiment,
            include_risk=not args.no_risk,
        )
    except EnvironmentError as e:
        print(f"❌  {e}")
        return 1

    if "error" in result:
        print(f"❌  Analysis failed: {result['error']}")
        return 1

    # Print summary
    _print_header(f"Analysis Report — {result['symbol']}")
    print(f"  Date          : {result['analysis_date'][:10]}")
    print(f"  Period        : {result['period_days']} days")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Confidence    : {result['confidence_label']} ({result['confidence']:.0%})")

    if result.get("scores"):
        s = result["scores"]
        print(f"\n  Scores:")
        print(f"    Sentiment   : {s.get('sentiment_score', 'N/A'):.2f}")
        print(f"    Risk        : {s.get('risk_score', 'N/A'):.2f}")
        print(f"    Composite   : {s.get('composite_score', 'N/A'):.2f}")

    print(f"\n  Summary:\n  {result.get('executive_summary', 'N/A')}")

    if result.get("key_positives"):
        print("\n  ✅ Positives:")
        for p in result["key_positives"]:
            print(f"     • {p}")

    if result.get("key_negatives"):
        print("\n  ⚠️  Risks:")
        for n in result["key_negatives"]:
            print(f"     • {n}")

    if result.get("action_items"):
        print("\n  📌 Action Items:")
        for i, a in enumerate(result["action_items"], 1):
            print(f"     {i}. {a}")

    if args.json:
        output = {k: v for k, v in result.items() if not k.startswith("_")}
        print("\n" + json.dumps(output, indent=2, default=str))

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate and save a Markdown investment report."""
    from src.chains.analysis_chain import FinancialAnalysisChain

    print(f"\n📄 Generating report for {args.symbol}…\n")
    try:
        chain = FinancialAnalysisChain(verbose=args.verbose)
        result = chain.analyze_stock(symbol=args.symbol, days_back=args.days)
    except EnvironmentError as e:
        print(f"❌  {e}")
        return 1

    if "error" in result:
        print(f"❌  Analysis failed: {result['error']}")
        return 1

    md = chain.summary_agent.generate_report_markdown(result)

    output_path = args.output or f"{args.symbol}_report_{datetime.now().strftime('%Y%m%d')}.md"
    Path(output_path).write_text(md, encoding="utf-8")

    print(f"✅  Report saved to: {output_path}")
    if args.print:
        print("\n" + "─" * 70 + "\n")
        print(md)
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """Batch-analyze a list of symbols."""
    from src.chains.analysis_chain import FinancialAnalysisChain

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("❌  No valid symbols provided.")
        return 1

    print(f"\n📡 Monitoring {len(symbols)} symbol(s): {', '.join(symbols)}\n")
    try:
        chain = FinancialAnalysisChain(verbose=args.verbose)
        results = chain.batch_analyze(symbols=symbols, days_back=args.days)
    except EnvironmentError as e:
        print(f"❌  {e}")
        return 1

    _print_header("Monitoring Summary")
    print(f"  {'Symbol':<8} {'Recommendation':<16} {'Sentiment':>10} {'Risk':>8} {'Confidence':>12}")
    print("  " + "─" * 60)
    for r in results:
        if "error" in r:
            print(f"  {r['symbol']:<8} {'ERROR':<16}")
            continue
        s = r.get("scores", {})
        print(
            f"  {r['symbol']:<8} "
            f"{r.get('recommendation', 'N/A'):<16} "
            f"{s.get('sentiment_score', 0):>10.2f} "
            f"{s.get('risk_score', 0):>8.2f} "
            f"{r.get('confidence_label', 'N/A'):>12}"
        )
    print()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Perform semantic search over the vector store."""
    from src.utils.vector_store import VectorStore

    print(f"\n🔎 Searching: '{args.query}'\n")
    try:
        vs = VectorStore()
        where = {"symbol": args.symbol.upper()} if args.symbol else None
        results = vs.search(
            query=args.query,
            n_results=args.limit,
            where=where,
            threshold=args.threshold,
        )
    except Exception as e:
        print(f"❌  Search failed: {e}")
        return 1

    if not results:
        print("  No results found. Try lowering --threshold or running init_db.py to seed data.")
        return 0

    _print_header(f"{len(results)} Result(s)")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(f"  [{i}] Score: {r['score']:.3f}")
        print(f"       Symbol : {meta.get('symbol', 'N/A')}")
        print(f"       Source : {meta.get('source', 'N/A')}")
        print(f"       Date   : {meta.get('published_at', 'N/A')[:10]}")
        print(f"       Title  : {meta.get('title', 'N/A')[:80]}")
        if args.verbose:
            print(f"       Text   : {r['document'][:200]}…")
        print()
    return 0


def cmd_init_db(args: argparse.Namespace) -> int:
    """Initialize (or reset) the ChromaDB vector store."""
    from src.utils.init_db import init_database

    db_path = args.path or os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    init_database(db_path=db_path, seed=not args.no_seed, reset=args.reset)
    return 0


# ===========================================================================
# Argument parser
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="financial-analyzer",
        description="Financial News Analyzer — AI-powered investment research CLI",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── analyze ──────────────────────────────────────────────────────────────
    p_analyze = sub.add_parser("analyze", help="Run full multi-agent analysis for a symbol")
    p_analyze.add_argument("--symbol", "-s", required=True, help="Stock ticker (e.g. AAPL)")
    p_analyze.add_argument("--days", "-d", type=int, default=7, help="Days to look back (default: 7)")
    p_analyze.add_argument("--no-sentiment", action="store_true", help="Skip sentiment analysis")
    p_analyze.add_argument("--no-risk", action="store_true", help="Skip risk assessment")
    p_analyze.add_argument("--json", action="store_true", help="Also print full JSON output")

    # ── report ───────────────────────────────────────────────────────────────
    p_report = sub.add_parser("report", help="Generate a Markdown investment report")
    p_report.add_argument("--symbol", "-s", required=True)
    p_report.add_argument("--days", "-d", type=int, default=7)
    p_report.add_argument("--output", "-o", help="Output file path (default: <SYMBOL>_report_<DATE>.md)")
    p_report.add_argument("--print", action="store_true", help="Also print report to stdout")

    # ── monitor ──────────────────────────────────────────────────────────────
    p_monitor = sub.add_parser("monitor", help="Batch-analyze multiple symbols")
    p_monitor.add_argument("--symbols", required=True, help="Comma-separated symbols: AAPL,GOOGL,MSFT")
    p_monitor.add_argument("--days", "-d", type=int, default=7)

    # ── search ───────────────────────────────────────────────────────────────
    p_search = sub.add_parser("search", help="Semantic search over historical news")
    p_search.add_argument("--query", "-q", required=True, help="Search query")
    p_search.add_argument("--limit", "-l", type=int, default=10)
    p_search.add_argument("--threshold", "-t", type=float, default=0.4, help="Min similarity (0–1)")
    p_search.add_argument("--symbol", help="Filter by symbol")

    # ── init-db ──────────────────────────────────────────────────────────────
    p_initdb = sub.add_parser("init-db", help="Initialize the ChromaDB vector store")
    p_initdb.add_argument("--path", help="ChromaDB directory path")
    p_initdb.add_argument("--reset", action="store_true", help="Wipe existing data before init")
    p_initdb.add_argument("--no-seed", action="store_true", help="Skip seed articles")

    return parser


# ===========================================================================
# Helpers
# ===========================================================================

def _print_header(title: str) -> None:
    bar = "─" * (len(title) + 4)
    print(f"\n  ┌{bar}┐")
    print(f"  │  {title}  │")
    print(f"  └{bar}┘\n")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"

    dispatch = {
        "analyze": cmd_analyze,
        "report": cmd_report,
        "monitor": cmd_monitor,
        "search": cmd_search,
        "init-db": cmd_init_db,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
