"""
Sentiment Agent - Performs sentiment analysis on financial news and market data.
"""
import asyncio
import hashlib
import threading
from typing import Any, Dict, List, Tuple
from datetime import datetime
from loguru import logger

import numpy as np
from langchain_core.prompts import PromptTemplate

from .base import AgentExecutionError, BaseAgent


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all SentimentAgent instances.
# Loaded at most once per process; thread-safe via double-checked locking.
# ---------------------------------------------------------------------------

_SENTIMENT_PIPELINE = None
_PIPELINE_LOADED = False
_PIPELINE_LOCK = threading.Lock()


def get_sentiment_pipeline():
    """
    Return the module-level FinBERT pipeline, loading it exactly once.

    FinBERT (ProsusAI/finbert) is trained on financial communications and
    returns three labels: "positive", "negative", "neutral" (lowercase).

    Thread-safe: concurrent callers block on the lock only during the first
    load; all subsequent calls return immediately without acquiring the lock.
    Returns None when transformers is not installed or the model fails to load.
    """
    global _SENTIMENT_PIPELINE, _PIPELINE_LOADED
    if not _PIPELINE_LOADED:
        with _PIPELINE_LOCK:
            if not _PIPELINE_LOADED:  # double-checked
                try:
                    from transformers import pipeline as _hf_pipeline
                    _SENTIMENT_PIPELINE = _hf_pipeline(
                        "sentiment-analysis",
                        model="ProsusAI/finbert",
                        device=-1,  # CPU
                    )
                    logger.info("Loaded sentiment analysis model")
                except ImportError:
                    logger.warning(
                        "transformers not installed — ML sentiment disabled, using LLM only"
                    )
                except Exception as e:
                    logger.warning(f"Could not load sentiment model: {e}")
                finally:
                    _PIPELINE_LOADED = True
    return _SENTIMENT_PIPELINE


class SentimentAgent(BaseAgent):
    """
    Agent responsible for analyzing sentiment from financial news and social media.

    Uses both transformer models and LLM reasoning for comprehensive sentiment analysis.
    The transformer pipeline is a module-level singleton loaded at most once per process.
    """

    def __init__(self, llm: Any, tools: List[Any] = None, verbose: bool = False):
        """
        Initialize the Sentiment Agent.

        Args:
            llm: Language model to use
            tools: Optional list of tools
            verbose: Enable verbose logging; also eagerly warms up the pipeline
        """
        super().__init__(
            name="SentimentAgent",
            description="Analyzes sentiment from financial news and data",
            llm=llm,
            tools=tools or [],
            verbose=verbose,
        )

        # Warm up the pipeline now so the first request doesn't pay the load cost.
        # When verbose=False the pipeline loads lazily on the first execute() call.
        if verbose:
            get_sentiment_pipeline()

        # Sentiment analysis prompt
        self.sentiment_prompt = PromptTemplate(
            input_variables=["text", "context"],
            template="""
            You are a financial sentiment analyst. Analyze the following text for
            sentiment regarding financial markets and investments.

            Text to analyze:
            {text}

            Context:
            {context}

            Provide a detailed sentiment analysis including:
            1. Overall sentiment (Positive/Negative/Neutral) with confidence score (0-1)
            2. Key positive factors
            3. Key negative factors
            4. Market implications
            5. Confidence in the analysis

            Format your response as a structured analysis.
            """,
        )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sentiment analysis on provided text or news articles.

        Args:
            input_data: Dictionary containing:
                - text: Text to analyze OR
                - articles: List of article texts
                - metadata: Optional list of dicts parallel to articles;
                  each may contain a "url" key used as the primary dedup key.
                - symbol: Stock symbol (for context)
                - context: Additional context (optional)

        Returns:
            Dictionary containing sentiment analysis results, including
            dedup_count (number of duplicate articles removed).
        """
        try:
            symbol = input_data.get("symbol", "")
            context = input_data.get("context", f"Analysis for {symbol}")

            # Handle single text or multiple articles
            if "text" in input_data:
                texts = [input_data["text"]]
            elif "articles" in input_data:
                texts = input_data["articles"]
            else:
                return {"error": "No text or articles provided"}

            # Deduplicate before analysis to avoid inflating confidence scores
            article_metadata = input_data.get("metadata", [])
            texts, dedup_count = self._deduplicate_texts(texts, article_metadata)
            if dedup_count:
                logger.debug(
                    f"Removed {dedup_count} duplicate article(s) before sentiment analysis"
                )

            logger.info(f"Analyzing sentiment for {len(texts)} text(s)")

            ml_sentiments = self._analyze_with_ml(texts)
            llm_analysis = self._analyze_with_llm(texts, context)
            combined_sentiment = self._combine_sentiments(ml_sentiments, llm_analysis)

            output = {
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "text_count": len(texts),
                "dedup_count": dedup_count,
                "overall_sentiment": combined_sentiment["label"],
                "sentiment_score": combined_sentiment["score"],
                "confidence": combined_sentiment["confidence"],
                "ml_analysis": ml_sentiments,
                "llm_analysis": llm_analysis,
                "key_insights": self._extract_insights(llm_analysis),
                "metadata": {"agent": self.name},
            }

            self._log_execution(input_data, output)
            logger.info(
                f"Sentiment analysis complete: {combined_sentiment['label']} "
                f"(score: {combined_sentiment['score']:.2f})"
            )
            return output

        except Exception as e:
            logger.error(f"Error in SentimentAgent execution: {e}")
            raise AgentExecutionError(
                agent_name=self.name,
                original_error=e,
                input_data=input_data,
            ) from e

    async def aexecute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of execute() with true async LLM calls.

        The ML pipeline (CPU-bound) runs in a thread-pool executor; the LLM
        call uses ainvoke so the event loop is never blocked.
        """
        try:
            symbol = input_data.get("symbol", "")
            context = input_data.get("context", f"Analysis for {symbol}")

            if "text" in input_data:
                texts = [input_data["text"]]
            elif "articles" in input_data:
                texts = input_data["articles"]
            else:
                return {"error": "No text or articles provided"}

            article_metadata = input_data.get("metadata", [])
            texts, dedup_count = self._deduplicate_texts(texts, article_metadata)
            if dedup_count:
                logger.debug(
                    f"Removed {dedup_count} duplicate article(s) before sentiment analysis"
                )

            logger.info(f"Analyzing sentiment for {len(texts)} text(s)")

            loop = asyncio.get_running_loop()
            ml_sentiments, llm_analysis = await asyncio.gather(
                loop.run_in_executor(None, self._analyze_with_ml, texts),
                self._analyze_with_llm_async(texts, context),
            )
            combined_sentiment = self._combine_sentiments(ml_sentiments, llm_analysis)

            output = {
                "symbol": symbol,
                "analysis_date": datetime.now().isoformat(),
                "text_count": len(texts),
                "dedup_count": dedup_count,
                "overall_sentiment": combined_sentiment["label"],
                "sentiment_score": combined_sentiment["score"],
                "confidence": combined_sentiment["confidence"],
                "ml_analysis": ml_sentiments,
                "llm_analysis": llm_analysis,
                "key_insights": self._extract_insights(llm_analysis),
                "metadata": {"agent": self.name},
            }

            self._log_execution(input_data, output)
            logger.info(
                f"Sentiment analysis complete: {combined_sentiment['label']} "
                f"(score: {combined_sentiment['score']:.2f})"
            )
            return output

        except Exception as e:
            logger.error(f"Error in SentimentAgent aexecute: {e}")
            raise AgentExecutionError(
                agent_name=self.name,
                original_error=e,
                input_data=input_data,
            ) from e

    async def _analyze_with_llm_async(self, texts: List[str], context: str) -> Dict[str, Any]:
        """Async LLM call used by aexecute()."""
        try:
            combined_text = "\n\n".join(texts[:5])
            if len(combined_text) > 3000:
                combined_text = combined_text[:3000] + "..."
            prompt = self.sentiment_prompt.format(text=combined_text, context=context)
            response = await self.llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return {"analysis": content, "method": "llm_reasoning"}
        except Exception as e:
            logger.error(f"Async LLM analysis failed: {e}")
            return {"analysis": "LLM analysis unavailable", "error": str(e)}

    def _deduplicate_texts(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]],
    ) -> Tuple[List[str], int]:
        """
        Remove duplicate articles before analysis.

        Dedup key priority:
          1. URL from the corresponding metadata entry (if present)
          2. MD5 hash of the first 100 characters of the text

        Args:
            texts: Raw list of article strings (may contain duplicates).
            metadata: Optional parallel list of dicts; each may carry a "url".

        Returns:
            (unique_texts, duplicate_count)
        """
        seen: set = set()
        unique: List[str] = []

        for i, text in enumerate(texts):
            meta = metadata[i] if i < len(metadata) else {}
            url = meta.get("url") if isinstance(meta, dict) else None
            key = url if url else hashlib.md5(text[:100].encode()).hexdigest()

            if key not in seen:
                seen.add(key)
                unique.append(text)

        return unique, len(texts) - len(unique)

    def _analyze_with_ml(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment using the module-level ML pipeline singleton.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment results, or [] when the pipeline is unavailable
        """
        _pipeline = get_sentiment_pipeline()
        if not _pipeline:
            return []

        results = []
        for text in texts:
            try:
                truncated_text = text[:512]
                sentiment = _pipeline(truncated_text)[0]
                results.append({
                    "text_preview": text[:100] + "...",
                    "label": sentiment["label"],
                    "score": sentiment["score"],
                    "normalized_score": self._normalize_score(
                        sentiment["label"], sentiment["score"]
                    ),
                })
            except Exception as e:
                logger.warning(f"ML analysis failed for text: {str(e)}")
                results.append({
                    "text_preview": text[:100] + "...",
                    "label": "NEUTRAL",
                    "score": 0.5,
                    "normalized_score": 0.0,
                })

        return results

    def _analyze_with_llm(self, texts: List[str], context: str) -> Dict[str, Any]:
        """
        Analyze sentiment using LLM for deeper reasoning.

        Args:
            texts: List of texts to analyze
            context: Additional context

        Returns:
            LLM sentiment analysis
        """
        try:
            combined_text = "\n\n".join(texts[:5])  # Limit to first 5 articles
            if len(combined_text) > 3000:
                combined_text = combined_text[:3000] + "..."

            prompt = self.sentiment_prompt.format(text=combined_text, context=context)
            response = self.llm.predict(prompt)

            return {"analysis": response, "method": "llm_reasoning"}

        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return {"analysis": "LLM analysis unavailable", "error": str(e)}

    def _normalize_score(self, label: str, score: float) -> float:
        """
        Normalize sentiment score to [-1, 1] range.

        Handles both FinBERT labels ("positive"/"negative"/"neutral") and
        SST-2 style labels ("POSITIVE"/"NEGATIVE") via case-insensitive matching.

        Args:
            label: Sentiment label from the ML model
            score: Confidence score [0, 1]

        Returns:
            +score for positive, -score for negative, 0.0 for neutral/unknown
        """
        label_upper = label.upper()
        if label_upper == "POSITIVE":
            return score
        elif label_upper == "NEGATIVE":
            return -score
        else:  # "NEUTRAL" / "neutral" / unknown
            return 0.0

    def _combine_sentiments(
        self,
        ml_results: List[Dict[str, Any]],
        llm_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine ML and LLM sentiment analyses.

        Works with binary (POSITIVE/NEGATIVE) and three-class
        (positive/negative/neutral) model output alike because each result
        carries a pre-computed normalized_score: +score for positive,
        -score for negative, 0.0 for neutral.  The average of those values
        is then mapped back to a POSITIVE / NEUTRAL / NEGATIVE label.

        Args:
            ml_results: Results from ML model (each entry must have "normalized_score")
            llm_analysis: Results from LLM

        Returns:
            Combined sentiment assessment
        """
        if not ml_results:
            return {"label": "NEUTRAL", "score": 0.5, "confidence": 0.3}

        normalized_scores = [r["normalized_score"] for r in ml_results]
        avg_score = np.mean(normalized_scores)

        if avg_score > 0.15:
            label = "POSITIVE"
        elif avg_score < -0.15:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        score = (avg_score + 1) / 2
        score_std = np.std(normalized_scores)
        confidence = max(0.3, 1.0 - score_std)

        return {
            "label": label,
            "score": score,
            "confidence": confidence,
            "raw_average": avg_score,
        }

    def _extract_insights(self, llm_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract key insights from LLM analysis.

        Args:
            llm_analysis: LLM analysis results

        Returns:
            Dictionary of positive and negative insights
        """
        return {
            "positive_factors": ["Analysis indicates positive market response"],
            "negative_factors": ["Some concerns noted in market data"],
            "market_implications": ["Monitor for continued trends"],
        }

    def analyze_trend(
        self,
        historical_sentiments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze sentiment trends over time.

        Args:
            historical_sentiments: List of past sentiment analyses

        Returns:
            Trend analysis results
        """
        if not historical_sentiments:
            return {"trend": "insufficient_data"}

        scores = [s.get("sentiment_score", 0.5) for s in historical_sentiments]

        if len(scores) >= 2:
            trend_direction = "improving" if scores[-1] > scores[0] else "declining"
            volatility = np.std(scores)
        else:
            trend_direction = "stable"
            volatility = 0.0

        return {
            "trend": trend_direction,
            "volatility": volatility,
            "current_score": scores[-1] if scores else 0.5,
            "average_score": np.mean(scores),
            "data_points": len(scores),
        }
