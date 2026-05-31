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
import asyncio
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

from src.agents.base import AgentExecutionError
from src.utils.cache import CacheClient

load_dotenv()


def _load_cache_config() -> Dict[str, Any]:
    """Read the ``cache`` section from agents_config.yaml; return defaults on failure."""
    try:
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "../../config/agents_config.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("cache", {})
    except Exception:
        return {}


def _make_retry_log_callback(max_retries: int):
    """Return a tenacity before_sleep callback that logs a warning on each retry."""
    def _before_sleep(retry_state) -> None:
        exc = retry_state.outcome.exception()
        n = retry_state.attempt_number
        logger.warning(
            f"LLM request failed (attempt {n}/{max_retries}): {exc}. Retrying..."
        )
    return _before_sleep


def _wrap_ollama_with_retry(llm: Any, max_retries: int = 3) -> Any:
    """
    Patch ChatOllama's invoke in-place with tenacity exponential-backoff retries.

    ChatOllama does not support a native max_retries constructor argument, so we
    wrap the instance method instead.  The original llm object is returned with
    its invoke method replaced so callers need no extra handling.
    """
    import tenacity

    original_invoke = llm.invoke

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(max_retries),
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=_make_retry_log_callback(max_retries),
        reraise=True,
    )
    def _invoke_with_retry(*args, **kwargs):
        return original_invoke(*args, **kwargs)

    llm.invoke = _invoke_with_retry
    return llm


def _build_llm(provider: str = "auto", temperature: float = 0.3) -> Any:
    """
    Instantiate the configured LLM.

    Checks provider in order:
      1. Explicit provider argument ("openai" | "anthropic" | "ollama")
      2. OPENAI_API_KEY   → ChatOpenAI
      3. ANTHROPIC_API_KEY → ChatAnthropic
      4. OLLAMA_BASE_URL  → ChatOllama (local)

    Retry behaviour:
      - OpenAI / Anthropic: max_retries passed to the constructor; LangChain
        handles exponential back-off internally.
      - Ollama: invoke() is wrapped with a tenacity retry loop (Ollama's
        ChatOllama does not accept max_retries).
      - Retry count is read from OPENAI_MAX_RETRIES (default 3).

    Args:
        provider: Force a specific provider, or "auto" to detect from env.
        temperature: Sampling temperature (lower = more deterministic).

    Returns:
        Instantiated LangChain chat model.

    Raises:
        EnvironmentError: If no LLM provider can be configured.
    """
    max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

    if provider in ("openai", "auto") and os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info(f"LLM provider: OpenAI ({model}, max_retries={max_retries})")
        return ChatOpenAI(model=model, temperature=temperature, max_retries=max_retries)

    if provider in ("anthropic", "auto") and os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")
        logger.info(f"LLM provider: Anthropic ({model}, max_retries={max_retries})")
        return ChatAnthropic(model=model, temperature=temperature, max_retries=max_retries)

    if provider in ("ollama", "auto") and os.getenv("OLLAMA_BASE_URL"):
        from langchain_community.chat_models import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        logger.info(
            f"LLM provider: Ollama ({base_url}, {model}, max_retries={max_retries})"
        )
        llm = ChatOllama(base_url=base_url, model=model, temperature=temperature)
        return _wrap_ollama_with_retry(llm, max_retries=max_retries)

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

        # ── Cache (optional — Redis with in-memory fallback) ──────────────
        cache_cfg = _load_cache_config()
        self._cache_ttl_seconds: int = int(cache_cfg.get("ttl_hours", 4)) * 3600
        self.cache: Optional[CacheClient] = self._init_cache(cache_cfg)

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
        fail_fast: bool = False,
        record_for_backtest: bool = False,
        enable_alerts: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline for a stock symbol (synchronous wrapper).

        Internally delegates to analyze_stock_async() via asyncio.run().
        Call analyze_stock_async() directly from async contexts (FastAPI, etc.).

        Args:
            symbol: Stock ticker (e.g. "AAPL").
            days_back: Number of days to look back for news (1–30).
            focus_areas: Comma-separated focus topics for the research agent.
            include_sentiment: Run the SentimentAgent stage.
            include_risk: Run the RiskAgent stage.
            context_from_vector_store: Enrich research with similar historical events.
            fail_fast: When True, halt on the first agent failure and return a
                top-level error dict.  When False (default), log the failure,
                substitute an empty dict for that stage, and continue.
            record_for_backtest: When True, persist the recommendation to the
                SQLite backtesting database after a successful pipeline run.
                Defaults to False to avoid unintended writes.
            enable_alerts: When True, dispatch HIGH/CRITICAL risk alerts via
                ``AlertDispatcher`` after the RiskAgent stage.
                Defaults to False to avoid unintended external calls.

        Returns:
            Dict with keys matching SummaryAgent.execute() output plus
            added convenience keys: symbol, summary, sentiment_score,
            risk_factors, recommendation, confidence, _elapsed_seconds.
            On fail_fast failure: {"symbol", "status": "failed", "error",
            "failed_stage"}.
        """
        return asyncio.run(
            self.analyze_stock_async(
                symbol=symbol,
                days_back=days_back,
                focus_areas=focus_areas,
                include_sentiment=include_sentiment,
                include_risk=include_risk,
                context_from_vector_store=context_from_vector_store,
                fail_fast=fail_fast,
                record_for_backtest=record_for_backtest,
                enable_alerts=enable_alerts,
            )
        )

    async def analyze_stock_async(
        self,
        symbol: str,
        days_back: int = 7,
        focus_areas: str = "earnings, product launches, regulatory issues, market sentiment",
        include_sentiment: bool = True,
        include_risk: bool = True,
        context_from_vector_store: bool = True,
        fail_fast: bool = False,
        record_for_backtest: bool = False,
        enable_alerts: bool = False,
    ) -> Dict[str, Any]:
        """
        Async version of analyze_stock().

        Stages 2 (Sentiment) and 3 (Risk) run concurrently via asyncio.gather
        after Stage 1 (Research) completes, roughly halving wall-clock time.

        Note: RiskAgent receives an empty sentiment dict because it runs in
        parallel with SentimentAgent and cannot depend on its output.
        """
        symbol = symbol.upper()

        # ── Cache check ───────────────────────────────────────────────────
        cache_key: Optional[str] = None
        if self.cache is not None:
            cache_key = self.cache.make_cache_key(
                symbol, days_back, datetime.now().strftime("%Y-%m-%d-%H")
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Cache hit for {symbol} ({cache_key})")
                return {**cached, "cache_hit": True}

        logger.info(f"Starting async pipeline for {symbol} (days_back={days_back})")
        pipeline_start = datetime.now()

        # ── Stage 1: Research ─────────────────────────────────────────────
        logger.info(f"[1/4] Research Agent — {symbol}")
        research_result: Dict[str, Any] = {}
        try:
            research_result = await self.research_agent.aexecute({
                "symbol": symbol,
                "days_back": days_back,
                "focus_areas": focus_areas,
            })
        except AgentExecutionError as exc:
            logger.error(f"ResearchAgent failed for {symbol}: {exc}")
            if fail_fast:
                return self._error_response(symbol, exc, "research")
            # fail_fast=False: continue with empty research

        if context_from_vector_store and self.vector_store and research_result:
            historical = self._fetch_historical_context(symbol, research_result)
            research_result["historical_context"] = historical

        # ── Stages 2 + 3: Sentiment and Risk — parallel ───────────────────
        logger.info(f"[2+3/4] Sentiment + Risk Agents — {symbol} (parallel)")

        async def _noop() -> Dict[str, Any]:
            return {}

        findings_text = research_result.get("findings", "")
        sentiment_coro = (
            self.sentiment_agent.aexecute({
                "symbol": symbol,
                "text": findings_text or f"General market news for {symbol}",
                "context": f"Financial analysis for {symbol}",
            })
            if include_sentiment
            else _noop()
        )
        risk_coro = (
            self.risk_agent.aexecute({
                "symbol": symbol,
                "news_data": research_result,
                "market_data": {},
                # Sentiment runs in parallel — not available yet; risk adjusts on {}
                "sentiment": {},
            })
            if include_risk
            else _noop()
        )

        raw_sentiment, raw_risk = await asyncio.gather(
            sentiment_coro, risk_coro, return_exceptions=True
        )

        sentiment_result: Dict[str, Any] = {}
        if include_sentiment:
            if isinstance(raw_sentiment, BaseException):
                logger.error(f"SentimentAgent failed for {symbol}: {raw_sentiment}")
                if fail_fast:
                    exc = (
                        raw_sentiment
                        if isinstance(raw_sentiment, AgentExecutionError)
                        else AgentExecutionError("SentimentAgent", raw_sentiment, {})
                    )
                    return self._error_response(symbol, exc, "sentiment")
            else:
                sentiment_result = raw_sentiment
        else:
            logger.info("[2/4] Sentiment Agent — skipped")

        risk_result: Dict[str, Any] = {}
        if include_risk:
            if isinstance(raw_risk, BaseException):
                logger.error(f"RiskAgent failed for {symbol}: {raw_risk}")
                if fail_fast:
                    exc = (
                        raw_risk
                        if isinstance(raw_risk, AgentExecutionError)
                        else AgentExecutionError("RiskAgent", raw_risk, {})
                    )
                    return self._error_response(symbol, exc, "risk")
            else:
                risk_result = raw_risk
        else:
            logger.info("[3/4] Risk Agent — skipped")

        # ── Alert dispatch (opt-in) ───────────────────────────────────────
        if enable_alerts and risk_result.get("alerts"):
            try:
                from src.utils.alerting import AlertDispatcher
                AlertDispatcher().dispatch(
                    alerts=risk_result["alerts"],
                    symbol=symbol,
                )
            except Exception as exc:
                logger.warning(f"Alert dispatch failed for {symbol}: {exc}")

        # ── Stage 4: Summary ──────────────────────────────────────────────
        logger.info(f"[4/4] Summary Agent — {symbol}")
        try:
            summary_result = await self.summary_agent.aexecute({
                "symbol": symbol,
                "research": research_result,
                "sentiment": sentiment_result,
                "risk": risk_result,
                "period_days": days_back,
            })
        except AgentExecutionError as exc:
            logger.error(f"SummaryAgent failed for {symbol}: {exc}")
            return self._error_response(symbol, exc, "summary")

        if self.vector_store and "error" not in summary_result:
            self._persist_to_vector_store(symbol, summary_result)

        elapsed = (datetime.now() - pipeline_start).total_seconds()
        logger.info(
            f"Async pipeline complete for {symbol} in {elapsed:.1f}s — "
            f"Recommendation: {summary_result.get('recommendation', 'N/A')}"
        )

        result = {
            **summary_result,
            "summary": summary_result.get("executive_summary", ""),
            "sentiment_score": sentiment_result.get("sentiment_score", None),
            "risk_factors": [
                r.get("description")
                for r in risk_result.get("identified_risks", [])
            ],
            "_research": research_result,
            "_sentiment": sentiment_result,
            "_risk": risk_result,
            "_elapsed_seconds": round(elapsed, 2),
        }

        # ── Cache store ───────────────────────────────────────────────────
        if cache_key is not None and "status" not in result:
            self.cache.set(cache_key, result, ttl_seconds=self._cache_ttl_seconds)
            logger.debug(f"Cached analysis for {symbol} (TTL={self._cache_ttl_seconds}s)")

        # ── Backtest recording (opt-in) ───────────────────────────────────
        if record_for_backtest and "status" not in result:
            try:
                from src.backtesting.recorder import RecommendationRecorder
                RecommendationRecorder().save(
                    symbol=symbol,
                    recommendation=result.get("recommendation", "HOLD"),
                    confidence=result.get("confidence", 0.0),
                    scores=result.get("scores", {}),
                )
            except Exception as exc:
                logger.warning(f"Backtest recording failed for {symbol}: {exc}")

        return result

    def analyze_portfolio(
        self,
        holdings: List[Dict[str, Any]],
        days_back: int = 7,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze a portfolio of holdings and return a portfolio-level report.

        Runs ``analyze_stock()`` for each holding, then passes all results to
        ``PortfolioAgent`` which aggregates scores, flags concentration and
        correlated risks, and produces a portfolio recommendation.

        Args:
            holdings: List of ``{"symbol": str, "weight": float}`` dicts.
                      Weights should sum to 1.0.
            days_back: Look-back window passed to each stock analysis.
            **kwargs: Additional keyword args forwarded to ``analyze_stock()``.

        Returns:
            Portfolio report dict from ``PortfolioAgent.execute()``.
        """
        from src.agents.portfolio_agent import PortfolioAgent

        individual_analyses: Dict[str, Any] = {}
        for holding in holdings:
            symbol = holding["symbol"].upper()
            try:
                result = self.analyze_stock(symbol=symbol, days_back=days_back, **kwargs)
                individual_analyses[symbol] = result
            except Exception as exc:
                logger.error(f"Portfolio: analysis failed for {symbol}: {exc}")
                individual_analyses[symbol] = {
                    "symbol": symbol,
                    "error": str(exc),
                    "status": "failed",
                }

        portfolio_agent = PortfolioAgent(llm=self.llm, verbose=self.verbose)
        return portfolio_agent.execute({
            "holdings": holdings,
            "individual_analyses": individual_analyses,
        })

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

    @staticmethod
    def _error_response(
        symbol: str, exc: AgentExecutionError, stage: str
    ) -> Dict[str, Any]:
        """Build a top-level error dict returned when fail_fast=True."""
        return {
            "symbol": symbol,
            "status": "failed",
            "failed_stage": stage,
            "error": str(exc),
        }

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

    def _init_cache(self, cache_cfg: Dict[str, Any]) -> Optional[CacheClient]:
        """Initialize the cache client; return None when caching is disabled."""
        if not cache_cfg.get("enabled", True):
            logger.info("Cache: disabled by configuration")
            return None
        try:
            return CacheClient(redis_url=os.getenv("REDIS_URL"))
        except Exception as exc:
            logger.warning(f"Cache initialization failed: {exc}")
            return None

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
