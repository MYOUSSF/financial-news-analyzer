"""
Analysis Chain — Orchestrates the full multi-agent pipeline.

Pipeline:
    ResearchAgent → SentimentAgent → RiskAgent → SummaryAgent

Usage (Python SDK):
    from src.chains.analysis_chain import FinancialAnalysisChain

    chain = FinancialAnalysisChain()
    result = chain.analyze_stock(symbol="AAPL", days_back=7)
    print(result["recommendation"])
    print(result["executive_summary"])
"""
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def _build_llm(provider: str = "auto", temperature: float = 0.3) -> Any:
    """
    Instantiate the configured LLM.

    Checks provider in order:
      1. Explicit provider argument ("openai" | "anthropic" | "ollama")
      2. OPENAI_API_KEY   → ChatOpenAI
      3. ANTHROPIC_API_KEY → ChatAnthropic
      4. OLLAMA_BASE_URL  → ChatOllama (local)

    Args:
        provider: Force a specific provider, or "auto" to detect from env.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        Instantiated LangChain chat model.

    Raises:
        EnvironmentError: If no LLM provider can be configured.
    """
    if provider in ("openai", "auto") and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info(f"LLM provider: OpenAI ({model})")
        return ChatOpenAI(model=model, temperature=temperature)

    if provider in ("anthropic", "auto") and os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        logger.info(f"LLM provider: Anthropic ({model})")
        return ChatAnthropic(model=model, temperature=temperature)

    if provider in ("ollama", "auto") and os.getenv("OLLAMA_BASE_URL"):
        from langchain_community.chat_models import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        logger.info(f"LLM provider: Ollama ({base_url}, {model})")
        return ChatOllama(base_url=base_url, model=model, temperature=temperature)

    raise EnvironmentError(
        "No LLM provider configured. Set one of: "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_BASE_URL in your .env file."
    )


class FinancialAnalysisChain:
    """
    High-level orchestrator for the four-agent financial analysis pipeline.

    The chain wires together:
    - ResearchAgent  — fetches news & market context
    - SentimentAgent — scores market sentiment
    - RiskAgent      — identifies risk factors
    - SummaryAgent   — synthesizes a final report

    Each agent's output is passed as context to the next stage.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        llm_provider: str = "auto",
        llm_temperature: float = 0.3,
        news_api_key: Optional[str] = None,
        chroma_db_path: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the analysis chain.

        Args:
            llm: Pre-built LangChain chat model. If None, one is built from env vars.
            llm_provider: "auto" | "openai" | "anthropic" | "ollama".
            llm_temperature: Sampling temperature passed to the LLM constructor.
            news_api_key: NewsAPI key (falls back to NEWSAPI_KEY env var).
            chroma_db_path: ChromaDB persist path (falls back to CHROMA_DB_PATH env var).
            verbose: Enable verbose logging in all agents.
        """
        self.verbose = verbose

        # ── LLM ──────────────────────────────────────────────────────────────
        self.llm = llm or _build_llm(provider=llm_provider, temperature=llm_temperature)

        # ── Tools ────────────────────────────────────────────────────────────
        self.tools = self._build_tools(news_api_key=news_api_key)

        # ── Agents ───────────────────────────────────────────────────────────
        self._init_agents()

        # ── Vector store (optional — used for context enrichment) ─────────
        self.vector_store = self._init_vector_store(chroma_db_path)

        logger.info("FinancialAnalysisChain initialized successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_stock(
        self,
        symbol: str,
        days_back: int = 7,
        focus_areas: str = "earnings, product launches, regulatory issues, market sentiment",
        include_sentiment: bool = True,
        include_risk: bool = True,
        context_from_vector_store: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline for a stock symbol.

        Args:
            symbol: Stock ticker (e.g. "AAPL").
            days_back: Number of days to look back for news (1–30).
            focus_areas: Comma-separated focus topics for the research agent.
            include_sentiment: Run the SentimentAgent stage.
            include_risk: Run the RiskAgent stage.
            context_from_vector_store: Enrich research with similar historical events.

        Returns:
            Dict with keys matching SummaryAgent.execute() output plus
            added convenience keys: symbol, summary, sentiment_score,
            risk_factors, recommendation, confidence.
        """
        symbol = symbol.upper()
        logger.info(f"Starting full pipeline for {symbol} (days_back={days_back})")
        pipeline_start = datetime.now()

        # ── Stage 1: Research ─────────────────────────────────────────────
        logger.info(f"[1/4] Research Agent — {symbol}")
        research_result = self.research_agent.execute({
            "symbol": symbol,
            "days_back": days_back,
            "focus_areas": focus_areas,
        })

        # Optionally enrich with vector store historical context
        if context_from_vector_store and self.vector_store:
            historical = self._fetch_historical_context(symbol, research_result)
            research_result["historical_context"] = historical

        # ── Stage 2: Sentiment ────────────────────────────────────────────
        sentiment_result: Dict[str, Any] = {}
        if include_sentiment:
            logger.info(f"[2/4] Sentiment Agent — {symbol}")
            # Feed raw research findings as the text corpus
            findings_text = research_result.get("findings", "")
            sentiment_result = self.sentiment_agent.execute({
                "symbol": symbol,
                "text": findings_text if findings_text else f"General market news for {symbol}",
                "context": f"Financial analysis for {symbol}",
            })
        else:
            logger.info("[2/4] Sentiment Agent — skipped")

        # ── Stage 3: Risk ─────────────────────────────────────────────────
        risk_result: Dict[str, Any] = {}
        if include_risk:
            logger.info(f"[3/4] Risk Agent — {symbol}")
            risk_result = self.risk_agent.execute({
                "symbol": symbol,
                "news_data": research_result,
                "market_data": {},   # extend here with live market data from stock_tool
                "sentiment": sentiment_result,
            })
        else:
            logger.info("[3/4] Risk Agent — skipped")

        # ── Stage 4: Summary ──────────────────────────────────────────────
        logger.info(f"[4/4] Summary Agent — {symbol}")
        summary_result = self.summary_agent.execute({
            "symbol": symbol,
            "research": research_result,
            "sentiment": sentiment_result,
            "risk": risk_result,
            "period_days": days_back,
        })

        # Persist the summarized article to the vector store for future context
        if self.vector_store and "error" not in summary_result:
            self._persist_to_vector_store(symbol, summary_result)

        elapsed = (datetime.now() - pipeline_start).total_seconds()
        logger.info(f"Pipeline complete for {symbol} in {elapsed:.1f}s — "
                    f"Recommendation: {summary_result.get('recommendation', 'N/A')}")

        # ── Build convenience output ──────────────────────────────────────
        return {
            **summary_result,
            # Convenience aliases used by the README SDK examples
            "summary": summary_result.get("executive_summary", ""),
            "sentiment_score": sentiment_result.get("sentiment_score", None),
            "risk_factors": [r.get("description") for r in risk_result.get("identified_risks", [])],
            # Raw agent outputs (for debugging / downstream use)
            "_research": research_result,
            "_sentiment": sentiment_result,
            "_risk": risk_result,
            "_elapsed_seconds": round(elapsed, 2),
        }

    def batch_analyze(
        self,
        symbols: List[str],
        days_back: int = 7,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run the analysis pipeline for multiple symbols sequentially.

        Args:
            symbols: List of stock tickers.
            days_back: Days to look back for each symbol.
            **kwargs: Additional keyword arguments passed to analyze_stock().

        Returns:
            List of analysis results, one per symbol.
        """
        results = []
        for symbol in symbols:
            try:
                result = self.analyze_stock(symbol=symbol, days_back=days_back, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Pipeline failed for {symbol}: {e}")
                results.append({"symbol": symbol, "error": str(e), "status": "failed"})
        return results

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Return initialization and health status of all chain components."""
        return {
            "llm": str(type(self.llm).__name__),
            "tools": [t.name for t in self.tools],
            "agents": [
                self.research_agent.get_status(),
                self.sentiment_agent.get_status(),
                self.risk_agent.get_status(),
                self.summary_agent.get_status(),
            ],
            "vector_store": self.vector_store.get_stats() if self.vector_store else None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_tools(self, news_api_key: Optional[str] = None) -> List[Any]:
        """Instantiate and return all LangChain tools."""
        tools = []
        try:
            from src.tools.news_tool import NewsTool, FinancialNewsTool
            tools.append(NewsTool(api_key=news_api_key or os.getenv("NEWSAPI_KEY")))
            tools.append(FinancialNewsTool())
            logger.info("News tools loaded.")
        except Exception as e:
            logger.warning(f"Could not load news tools: {e}")

        try:
            from src.tools.stock_tool import StockTool
            tools.append(StockTool())
            logger.info("Stock tool loaded.")
        except Exception as e:
            logger.warning(f"Could not load stock tool: {e}")

        try:
            from src.tools.economic_tool import EconomicTool
            tools.append(EconomicTool())
            logger.info("Economic tool loaded.")
        except Exception as e:
            logger.warning(f"Could not load economic tool: {e}")

        return tools

    def _init_agents(self) -> None:
        """Instantiate all four agents."""
        from src.agents.research_agent import ResearchAgent
        from src.agents.sentiment_agent import SentimentAgent
        from src.agents.risk_agent import RiskAgent
        from src.agents.summary_agent import SummaryAgent

        self.research_agent = ResearchAgent(
            llm=self.llm, tools=self.tools, verbose=self.verbose
        )
        self.sentiment_agent = SentimentAgent(
            llm=self.llm, tools=[], verbose=self.verbose
        )
        self.risk_agent = RiskAgent(
            llm=self.llm, tools=[], verbose=self.verbose
        )
        self.summary_agent = SummaryAgent(
            llm=self.llm, tools=[], verbose=self.verbose
        )

    def _init_vector_store(self, db_path: Optional[str]) -> Optional[Any]:
        """Initialize the vector store; return None on failure (non-fatal)."""
        try:
            from src.utils.vector_store import VectorStore
            return VectorStore(persist_directory=db_path)
        except Exception as e:
            logger.warning(f"Vector store unavailable (run init_db.py to set up): {e}")
            return None

    def _fetch_historical_context(
        self, symbol: str, research_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search vector store for similar past events."""
        if not self.vector_store:
            return []
        try:
            query = research_result.get("findings", f"{symbol} market news")[:200]
            return self.vector_store.search(query=query, n_results=3, threshold=0.4)
        except Exception as e:
            logger.warning(f"Vector store context fetch failed: {e}")
            return []

    def _persist_to_vector_store(
        self, symbol: str, summary: Dict[str, Any]
    ) -> None:
        """Save a summary to the vector store for future retrieval."""
        if not self.vector_store:
            return
        try:
            text = summary.get("executive_summary", "") or summary.get("full_report", "")
            if not text:
                return
            self.vector_store.add_news_article(
                title=f"{symbol} Analysis — {summary.get('recommendation', 'N/A')}",
                content=text[:2000],
                symbol=symbol,
                source="FinancialAnalysisChain",
                published_at=summary.get("analysis_date", datetime.utcnow().isoformat()),
                sentiment=summary.get("scores", {}).get("sentiment_score"),
            )
        except Exception as e:
            logger.warning(f"Could not persist summary to vector store: {e}")
