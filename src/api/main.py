"""
FastAPI Application — REST API for Financial News Analyzer

All endpoints are wired to the real multi-agent pipeline (lazy-initialized
on first request so the server starts fast even without API keys configured).

Start with:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Make src/ importable when running from the api/ sub-directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# ---------------------------------------------------------------------------
# Lazy global chain — initialized once on first real request
# ---------------------------------------------------------------------------
_chain = None


def get_chain():
    """Return the singleton FinancialAnalysisChain, initializing it on first call."""
    global _chain
    if _chain is None:
        from src.chains.analysis_chain import FinancialAnalysisChain
        _chain = FinancialAnalysisChain(verbose=False)
        logger.info("FinancialAnalysisChain initialized.")
    return _chain


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Financial News Analyzer API")
    # Warm up chain in background (optional — comment out to keep startup fast)
    # get_chain()
    yield
    logger.info("Shutting down Financial News Analyzer API")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Financial News Analyzer API",
    description="AI-powered financial research and analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Pydantic models
# ===========================================================================

class AnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    days_back: int = Field(7, ge=1, le=30, description="Days to analyze")
    include_sentiment: bool = Field(True)
    include_risk: bool = Field(True)
    include_trends: bool = Field(False)


class SentimentRequest(BaseModel):
    symbol: str
    text: Optional[str] = None
    articles: Optional[List[str]] = None


class RiskRequest(BaseModel):
    symbol: str
    days_back: int = Field(7, ge=1, le=30)
    news_data: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., description="Semantic search query")
    limit: int = Field(10, ge=1, le=50)
    threshold: float = Field(0.4, ge=0.0, le=1.0)
    symbol: Optional[str] = None


class ReportRequest(BaseModel):
    symbol: str
    report_type: str = Field("comprehensive", description="comprehensive | summary | risk | sentiment")
    format: str = Field("markdown", description="markdown | pdf | html")


class SentimentResponse(BaseModel):
    symbol: str
    overall_sentiment: str
    sentiment_score: float
    confidence: float
    analysis_date: str
    key_insights: Dict[str, List[str]]


class RiskResponse(BaseModel):
    symbol: str
    overall_risk_score: float
    risk_level: str
    identified_risks: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    recommendations: List[str]
    analysis_date: str


class AnalysisResponse(BaseModel):
    symbol: str
    analysis_date: str
    period_days: int
    executive_summary: str
    recommendation: str
    confidence: float
    confidence_label: str
    sentiment: Optional[SentimentResponse] = None
    risk: Optional[RiskResponse] = None
    key_positives: List[str] = []
    key_negatives: List[str] = []
    action_items: List[str] = []
    scores: Dict[str, float] = {}


class NewsArticle(BaseModel):
    title: str
    source: str
    published_at: str
    summary: str
    url: str
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None


# ===========================================================================
# Utility
# ===========================================================================

def _chain_available() -> bool:
    """Return True if at least one LLM provider is configured."""
    return any([
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("OLLAMA_BASE_URL"),
    ])


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/")
async def root():
    return {
        "message": "Financial News Analyzer API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "chain_ready": _chain_available(),
    }


@app.get("/health")
async def health_check():
    vector_store_ok = False
    try:
        from src.utils.vector_store import VectorStore
        vs = VectorStore()
        vector_store_ok = vs.count() >= 0
    except Exception:
        pass

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "operational",
            "llm_configured": _chain_available(),
            "vector_db": "operational" if vector_store_ok else "unavailable (run init_db.py)",
        },
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest):
    """Run the full multi-agent analysis pipeline for a stock symbol."""
    if not _chain_available():
        raise HTTPException(
            status_code=503,
            detail="No LLM provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_BASE_URL.",
        )

    try:
        chain = get_chain()
        result = chain.analyze_stock(
            symbol=request.symbol,
            days_back=request.days_back,
            include_sentiment=request.include_sentiment,
            include_risk=request.include_risk,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Build sentiment sub-response
        sentiment_resp: Optional[SentimentResponse] = None
        if request.include_sentiment and result.get("_sentiment"):
            s = result["_sentiment"]
            sentiment_resp = SentimentResponse(
                symbol=request.symbol,
                overall_sentiment=s.get("overall_sentiment", "NEUTRAL"),
                sentiment_score=float(s.get("sentiment_score", 0.5)),
                confidence=float(s.get("confidence", 0.5)),
                analysis_date=s.get("analysis_date", datetime.now().isoformat()),
                key_insights=s.get("key_insights", {}),
            )

        # Build risk sub-response
        risk_resp: Optional[RiskResponse] = None
        if request.include_risk and result.get("_risk"):
            r = result["_risk"]
            risk_resp = RiskResponse(
                symbol=request.symbol,
                overall_risk_score=float(r.get("overall_risk_score", 0.5)),
                risk_level=r.get("risk_level", "MEDIUM"),
                identified_risks=r.get("identified_risks", []),
                alerts=r.get("alerts", []),
                recommendations=r.get("recommendations", []),
                analysis_date=r.get("analysis_date", datetime.now().isoformat()),
            )

        return AnalysisResponse(
            symbol=result["symbol"],
            analysis_date=result["analysis_date"],
            period_days=result["period_days"],
            executive_summary=result.get("executive_summary", ""),
            recommendation=result.get("recommendation", "HOLD"),
            confidence=result.get("confidence", 0.5),
            confidence_label=result.get("confidence_label", "MEDIUM"),
            sentiment=sentiment_resp,
            risk=risk_resp,
            key_positives=result.get("key_positives", []),
            key_negatives=result.get("key_negatives", []),
            action_items=result.get("action_items", []),
            scores=result.get("scores", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stocks/{symbol}/news", response_model=List[NewsArticle])
async def get_stock_news(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Fetch and return recent news articles for a stock symbol."""
    try:
        from src.tools.news_tool import NewsTool
        tool = NewsTool(api_key=os.getenv("NEWSAPI_KEY"))
        raw = tool._run(symbol.upper())

        # Parse the formatted string back into articles
        # (In production you'd return structured data directly from the tool)
        articles = []
        current: Dict[str, str] = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit() and "." in line[:3]:
                if current:
                    articles.append(current)
                current = {"title": line.split(".", 1)[1].strip(), "source": "", "published_at": "", "summary": "", "url": ""}
            elif line.startswith("Source:"):
                current["source"] = line.replace("Source:", "").strip()
            elif line.startswith("Published:"):
                current["published_at"] = line.replace("Published:", "").strip()
            elif line.startswith("Summary:"):
                current["summary"] = line.replace("Summary:", "").strip()
            elif line.startswith("URL:"):
                current["url"] = line.replace("URL:", "").strip()
        if current and current.get("title"):
            articles.append(current)

        return [
            NewsArticle(
                title=a.get("title", ""),
                source=a.get("source", ""),
                published_at=a.get("published_at", datetime.now().isoformat()),
                summary=a.get("summary", ""),
                url=a.get("url", ""),
            )
            for a in articles[:limit]
        ]

    except Exception as e:
        logger.error(f"News fetch error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sentiment/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """Run standalone sentiment analysis on provided text or articles."""
    if not request.text and not request.articles:
        raise HTTPException(status_code=400, detail="Provide either 'text' or 'articles'.")

    if not _chain_available():
        raise HTTPException(status_code=503, detail="No LLM provider configured.")

    try:
        chain = get_chain()
        input_data: Dict[str, Any] = {"symbol": request.symbol}
        if request.text:
            input_data["text"] = request.text
        else:
            input_data["articles"] = request.articles

        result = chain.sentiment_agent.execute(input_data)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return SentimentResponse(
            symbol=result["symbol"],
            overall_sentiment=result.get("overall_sentiment", "NEUTRAL"),
            sentiment_score=float(result.get("sentiment_score", 0.5)),
            confidence=float(result.get("confidence", 0.5)),
            analysis_date=result.get("analysis_date", datetime.now().isoformat()),
            key_insights=result.get("key_insights", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risks/detect", response_model=RiskResponse)
async def detect_risks(request: RiskRequest):
    """Run standalone risk detection for a stock symbol."""
    if not _chain_available():
        raise HTTPException(status_code=503, detail="No LLM provider configured.")

    try:
        chain = get_chain()
        result = chain.risk_agent.execute({
            "symbol": request.symbol,
            "news_data": request.news_data or "",
            "market_data": {},
            "sentiment": {},
        })

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return RiskResponse(
            symbol=result["symbol"],
            overall_risk_score=float(result.get("overall_risk_score", 0.5)),
            risk_level=result.get("risk_level", "MEDIUM"),
            identified_risks=result.get("identified_risks", []),
            alerts=result.get("alerts", []),
            recommendations=result.get("recommendations", []),
            analysis_date=result.get("analysis_date", datetime.now().isoformat()),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk detection error for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/semantic")
async def semantic_search(request: SearchRequest):
    """Search historical news in the vector store using semantic similarity."""
    try:
        from src.utils.vector_store import VectorStore
        vs = VectorStore()

        where = {"symbol": request.symbol.upper()} if request.symbol else None
        results = vs.search(
            query=request.query,
            n_results=request.limit,
            where=where,
            threshold=request.threshold,
        )

        return {
            "query": request.query,
            "results": results,
            "total": len(results),
        }

    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reports/generate")
async def generate_report(request: ReportRequest):
    """Generate a full investment research report via the analysis chain."""
    if not _chain_available():
        raise HTTPException(status_code=503, detail="No LLM provider configured.")

    try:
        chain = get_chain()
        result = chain.analyze_stock(symbol=request.symbol)

        md_report = chain.summary_agent.generate_report_markdown(result)

        return {
            "symbol": request.symbol,
            "report_type": request.report_type,
            "format": request.format,
            "status": "generated",
            "content": md_report,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/symbols/trending")
async def get_trending_symbols(limit: int = Query(default=10, ge=1, le=50)):
    """Return trending symbols based on vector store news volume."""
    try:
        from src.utils.vector_store import VectorStore
        vs = VectorStore()

        # Count documents per symbol via broad search
        popular = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "SPY", "BTC", "NFLX"]
        trending = []
        for sym in popular[:limit]:
            results = vs.search_by_symbol(sym, n_results=5)
            trending.append({
                "symbol": sym,
                "news_count": len(results),
                "avg_sentiment": round(
                    sum(r["metadata"].get("sentiment_score", 0.5) for r in results) / max(len(results), 1),
                    2,
                ),
            })

        trending.sort(key=lambda x: x["news_count"], reverse=True)

        return {"symbols": trending, "updated_at": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Trending symbols error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    """Return system statistics."""
    doc_count = 0
    try:
        from src.utils.vector_store import VectorStore
        doc_count = VectorStore().count()
    except Exception:
        pass

    return {
        "vector_store_documents": doc_count,
        "chain_initialized": _chain is not None,
        "llm_configured": _chain_available(),
        "timestamp": datetime.now().isoformat(),
    }


# ===========================================================================
# Dev entry point
# ===========================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
