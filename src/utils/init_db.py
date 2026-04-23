"""
Database Initialization Script — ChromaDB Vector Store Setup

Run this once before starting the application:
    python src/utils/init_db.py

What it does:
  1. Creates the ChromaDB persist directory (data/chroma_db by default)
  2. Creates the 'financial_news' collection with the cosine similarity index
  3. Seeds the collection with a small set of example market events so
     semantic search works out of the box for demos
  4. Verifies the store is queryable

Usage:
    python src/utils/init_db.py                        # default settings
    python src/utils/init_db.py --path ./my/chroma/db  # custom path
    python src/utils/init_db.py --no-seed              # skip seed data
    python src/utils/init_db.py --reset                # wipe and re-create
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure src/ is on the path when run directly
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(ROOT))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Import our wrapper (makes it easy to test in isolation too)
from src.utils.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Seed data — representative historical market events
# ---------------------------------------------------------------------------
SEED_ARTICLES = [
    {
        "title": "Federal Reserve Raises Interest Rates by 25 Basis Points",
        "content": (
            "The Federal Reserve announced a 0.25% increase in the federal funds rate, "
            "bringing it to a 22-year high. Fed Chair Jerome Powell cited persistent "
            "inflation and a strong labor market as primary reasons for the decision. "
            "Markets initially sold off but recovered as Powell signaled a potential pause."
        ),
        "symbol": "SPY",
        "source": "Reuters",
        "published_at": (datetime.utcnow() - timedelta(days=30)).isoformat(),
        "sentiment": "Negative",
        "sentiment_score": 0.35,
        "url": "https://example.com/fed-rate-hike",
    },
    {
        "title": "Apple Reports Record Q4 Revenue, iPhone Sales Beat Estimates",
        "content": (
            "Apple Inc. reported quarterly revenue of $119.6 billion, surpassing Wall "
            "Street estimates of $117.9 billion. iPhone revenue reached $69.7 billion, "
            "driven by strong demand in China and emerging markets. Services segment "
            "grew 16% year-over-year to $22.3 billion. CEO Tim Cook highlighted AI "
            "integration across the product line as a key growth driver."
        ),
        "symbol": "AAPL",
        "source": "Bloomberg",
        "published_at": (datetime.utcnow() - timedelta(days=14)).isoformat(),
        "sentiment": "Positive",
        "sentiment_score": 0.88,
        "url": "https://example.com/apple-q4-earnings",
    },
    {
        "title": "Tesla Cuts Vehicle Prices Across Major Markets, Margin Concerns Rise",
        "content": (
            "Tesla reduced prices on its Model 3 and Model Y vehicles by 5–10% in the "
            "United States, Europe, and China. The move intensifies competition in the EV "
            "sector but raises investor concerns about gross margins, which fell to 17.6% "
            "last quarter. Analysts are divided on whether the strategy will boost volume "
            "enough to offset the margin impact."
        ),
        "symbol": "TSLA",
        "source": "Financial Times",
        "published_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
        "sentiment": "Negative",
        "sentiment_score": 0.32,
        "url": "https://example.com/tesla-price-cuts",
    },
    {
        "title": "Microsoft Azure Cloud Revenue Surges 28%, AI Services Drive Growth",
        "content": (
            "Microsoft reported Azure and other cloud services revenue growth of 28% in "
            "its fiscal second quarter, beating analyst expectations of 25%. The company "
            "attributed growth to enterprise adoption of Azure OpenAI Service. Overall "
            "revenue rose 17% to $62 billion. Microsoft raised its full-year guidance, "
            "citing strong demand from enterprise customers for AI infrastructure."
        ),
        "symbol": "MSFT",
        "source": "CNBC",
        "published_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
        "sentiment": "Positive",
        "sentiment_score": 0.91,
        "url": "https://example.com/microsoft-azure-growth",
    },
    {
        "title": "SEC Launches Investigation into Crypto Exchange Over Alleged Fraud",
        "content": (
            "The Securities and Exchange Commission has opened a formal investigation into "
            "a major cryptocurrency exchange following whistleblower complaints about "
            "misappropriation of customer funds. Trading volumes on the platform fell 40% "
            "following the announcement. The crypto market broadly sold off with Bitcoin "
            "declining 8% and Ethereum dropping 11% over 24 hours."
        ),
        "symbol": "BTC",
        "source": "Wall Street Journal",
        "published_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
        "sentiment": "Negative",
        "sentiment_score": 0.12,
        "url": "https://example.com/sec-crypto-investigation",
    },
    {
        "title": "NVIDIA GPU Demand Surges as AI Data Center Buildout Accelerates",
        "content": (
            "NVIDIA reported data center revenue of $18.4 billion, up 409% year-over-year, "
            "as hyperscalers and cloud providers continued to invest heavily in AI "
            "infrastructure. The H100 and upcoming H200 GPUs remain supply-constrained, "
            "with lead times extending to 6–9 months. CEO Jensen Huang projected continued "
            "strong demand through the next two fiscal years."
        ),
        "symbol": "NVDA",
        "source": "Bloomberg",
        "published_at": (datetime.utcnow() - timedelta(days=3)).isoformat(),
        "sentiment": "Positive",
        "sentiment_score": 0.95,
        "url": "https://example.com/nvidia-data-center",
    },
    {
        "title": "Global Supply Chain Disruptions Threaten Q3 Manufacturing Outlook",
        "content": (
            "A new wave of supply chain disruptions stemming from port congestion in Asia "
            "and Red Sea shipping rerouting is threatening manufacturing schedules for "
            "electronics, automotive, and consumer goods companies. Shipping costs have "
            "risen 35% in the past quarter. Companies with significant Asian manufacturing "
            "exposure, including major tech and automotive firms, issued cautious guidance."
        ),
        "symbol": "SPY",
        "source": "Reuters",
        "published_at": (datetime.utcnow() - timedelta(days=20)).isoformat(),
        "sentiment": "Negative",
        "sentiment_score": 0.28,
        "url": "https://example.com/supply-chain-disruptions",
    },
    {
        "title": "Amazon AWS Maintains Cloud Market Leadership, Operating Income Doubles",
        "content": (
            "Amazon Web Services maintained its position as the world's largest cloud "
            "provider with 31% market share, reporting revenue of $24.2 billion, up 17% "
            "year-over-year. AWS operating income more than doubled to $9.4 billion, "
            "indicating significant margin improvement. Generative AI workloads now "
            "represent a meaningful portion of new AWS contract signings."
        ),
        "symbol": "AMZN",
        "source": "TechCrunch",
        "published_at": (datetime.utcnow() - timedelta(days=8)).isoformat(),
        "sentiment": "Positive",
        "sentiment_score": 0.82,
        "url": "https://example.com/amazon-aws-results",
    },
]


# ---------------------------------------------------------------------------
# Core init logic
# ---------------------------------------------------------------------------

def init_database(
    db_path: str,
    seed: bool = True,
    reset: bool = False,
) -> VectorStore:
    """
    Initialize the ChromaDB vector store.

    Args:
        db_path: Filesystem path for persistent ChromaDB storage.
        seed: Whether to insert example seed articles.
        reset: If True, wipe any existing data before initializing.

    Returns:
        The initialized VectorStore instance.
    """
    logger.info("=" * 60)
    logger.info("Financial News Analyzer — Database Initialization")
    logger.info("=" * 60)
    logger.info(f"ChromaDB path : {db_path}")
    logger.info(f"Seed data     : {seed}")
    logger.info(f"Reset         : {reset}")
    logger.info("")

    # 1. Create the VectorStore (creates directory + collection automatically)
    store = VectorStore(persist_directory=db_path)

    # 2. Optionally wipe existing data
    if reset and store.count() > 0:
        logger.warning(f"Resetting collection — deleting {store.count()} existing documents…")
        store.clear_collection()

    existing = store.count()
    logger.info(f"Existing documents in collection: {existing}")

    # 3. Seed with representative market events
    if seed:
        if existing > 0 and not reset:
            logger.info("Skipping seed — collection already contains documents. "
                        "Use --reset to force re-seeding.")
        else:
            logger.info(f"Seeding collection with {len(SEED_ARTICLES)} example articles…")

            for i, article in enumerate(SEED_ARTICLES, start=1):
                doc_id = store.add_news_article(**article)
                logger.info(
                    f"  [{i}/{len(SEED_ARTICLES)}] "
                    f"{article['symbol']:6s} | {article['title'][:60]}…  → {doc_id[:8]}…"
                )

            logger.info(f"Seed complete. Total documents: {store.count()}")

    # 4. Smoke-test: run a quick semantic search
    logger.info("")
    logger.info("Running smoke-test query: 'interest rate hikes inflation'")
    results = store.search("interest rate hikes inflation", n_results=3)
    if results:
        logger.info(f"  ✅  Search returned {len(results)} result(s). Top match:")
        top = results[0]
        logger.info(f"       Score    : {top['score']:.4f}")
        logger.info(f"       Symbol   : {top['metadata'].get('symbol', 'N/A')}")
        logger.info(f"       Title    : {top['metadata'].get('title', 'N/A')[:70]}")
    else:
        logger.warning("  ⚠️  Search returned no results — collection may be empty.")

    # 5. Print final stats
    stats = store.get_stats()
    logger.info("")
    logger.info("─" * 60)
    logger.info("Vector store statistics:")
    for k, v in stats.items():
        logger.info(f"  {k:<25} : {v}")
    logger.info("─" * 60)
    logger.info("✅  Database initialization complete.")

    return store


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize the ChromaDB vector store for Financial News Analyzer."
    )
    parser.add_argument(
        "--path",
        default=os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
        help="Path to the ChromaDB persist directory (default: ./data/chroma_db or $CHROMA_DB_PATH)",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Skip inserting example seed articles.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing collection data before initializing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    init_database(
        db_path=args.path,
        seed=not args.no_seed,
        reset=args.reset,
    )
