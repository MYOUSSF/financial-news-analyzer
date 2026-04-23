"""
Summary Agent — Synthesizes findings from Research, Sentiment, and Risk agents
into a cohesive investment research report.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

from langchain_core.prompts import PromptTemplate

from .base import BaseAgent


class SummaryAgent(BaseAgent):
    """
    Agent responsible for synthesizing multi-agent findings into structured
    investment research reports.

    It receives the outputs of the Research, Sentiment, and Risk agents and
    produces:
    - An executive summary
    - An investment recommendation (BUY / HOLD / SELL / AVOID)
    - A confidence score
    - Key action items for the investor
    """

    # Recommendation thresholds
    RECOMMENDATION_MATRIX = {
        # (sentiment_score_range, risk_score_range): recommendation
        "strong_buy":  {"sentiment_min": 0.75, "risk_max": 0.35},
        "buy":         {"sentiment_min": 0.60, "risk_max": 0.50},
        "hold":        {"sentiment_min": 0.40, "risk_max": 0.65},
        "sell":        {"sentiment_min": 0.25, "risk_max": 0.80},
        "avoid":       {"sentiment_min": 0.00, "risk_max": 1.00},
    }

    def __init__(self, llm: Any, tools: Optional[List[Any]] = None, verbose: bool = False):
        """
        Initialize the Summary Agent.

        Args:
            llm: Language model to use.
            tools: Optional tools (not required for summarization).
            verbose: Enable verbose logging.
        """
        super().__init__(
            name="SummaryAgent",
            description="Synthesizes multi-agent findings into investment research reports",
            llm=llm,
            tools=tools or [],
            verbose=verbose,
        )

        # Primary synthesis prompt
        self.synthesis_prompt = PromptTemplate(
            input_variables=[
                "symbol",
                "analysis_date",
                "period_days",
                "research_findings",
                "sentiment_analysis",
                "risk_assessment",
            ],
            template="""
You are a senior investment research analyst. You have received detailed reports from
three specialist agents. Synthesize their findings into a professional, actionable
investment research report.

=== INPUTS ===

Symbol: {symbol}
Analysis Date: {analysis_date}
Period Covered: Last {period_days} days

--- Research Agent Findings ---
{research_findings}

--- Sentiment Agent Analysis ---
{sentiment_analysis}

--- Risk Agent Assessment ---
{risk_assessment}

=== YOUR TASK ===

Produce a concise but comprehensive synthesis covering:

1. EXECUTIVE SUMMARY (3–4 sentences capturing the overall picture)
2. KEY POSITIVES (bullet list, max 5 items)
3. KEY NEGATIVES / RISKS (bullet list, max 5 items)
4. INVESTMENT RECOMMENDATION (one of: STRONG BUY / BUY / HOLD / SELL / AVOID)
   Include a one-paragraph rationale.
5. CONFIDENCE LEVEL (LOW / MEDIUM / HIGH) with a brief justification
6. IMMEDIATE ACTION ITEMS for an investor (max 3 concrete steps)

Be factual, balanced, and professional. Avoid speculation beyond what the data supports.
If data from any agent is missing or unreliable, acknowledge this in your confidence assessment.
""",
        )

        # Short-form summary prompt (for dashboard cards / API previews)
        self.short_summary_prompt = PromptTemplate(
            input_variables=["symbol", "research", "sentiment", "risk"],
            template="""
Summarize in exactly 2 sentences (no more) the investment outlook for {symbol}
based on the following data:

Research: {research}
Sentiment: {sentiment}
Risk: {risk}

Be direct and informative. Do not use bullet points.
""",
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a full synthesis report from multi-agent outputs.

        Args:
            input_data: Dictionary containing:
                - symbol (str): Stock symbol.
                - research (dict): Output from ResearchAgent.execute().
                - sentiment (dict): Output from SentimentAgent.execute().
                - risk (dict): Output from RiskAgent.execute().
                - period_days (int, optional): Analysis period in days (default: 7).

        Returns:
            Dictionary with keys:
                symbol, analysis_date, period_days,
                executive_summary, key_positives, key_negatives,
                recommendation, recommendation_rationale,
                confidence, confidence_label, action_items,
                scores (dict with sentiment_score, risk_score, composite_score),
                full_report (str), metadata (dict).
        """
        try:
            symbol = input_data.get("symbol", "").upper()
            research = input_data.get("research", {})
            sentiment = input_data.get("sentiment", {})
            risk = input_data.get("risk", {})
            period_days = input_data.get("period_days", 7)

            logger.info(f"Generating synthesis report for {symbol}")

            analysis_date = datetime.now().isoformat()

            # Derive quantitative scores
            scores = self._compute_composite_scores(sentiment, risk)

            # Get recommendation from rule-based matrix
            rule_recommendation = self._derive_recommendation(
                scores["sentiment_score"], scores["risk_score"]
            )

            # Generate LLM synthesis
            llm_output = self._synthesize_with_llm(
                symbol=symbol,
                analysis_date=analysis_date,
                period_days=period_days,
                research=research,
                sentiment=sentiment,
                risk=risk,
            )

            # Parse LLM output into structured sections
            parsed = self._parse_llm_output(llm_output)

            # Prefer LLM recommendation if it disagrees; log discrepancy
            final_recommendation = parsed.get("recommendation") or rule_recommendation
            if parsed.get("recommendation") and parsed["recommendation"] != rule_recommendation:
                logger.info(
                    f"LLM recommendation ({parsed['recommendation']}) differs from "
                    f"rule-based ({rule_recommendation}); using LLM recommendation."
                )

            output = {
                "symbol": symbol,
                "analysis_date": analysis_date,
                "period_days": period_days,
                # Core report content
                "executive_summary": parsed.get("executive_summary", llm_output[:500]),
                "key_positives": parsed.get("key_positives", []),
                "key_negatives": parsed.get("key_negatives", []),
                "recommendation": final_recommendation,
                "recommendation_rationale": parsed.get("recommendation_rationale", ""),
                "confidence": scores["confidence"],
                "confidence_label": scores["confidence_label"],
                "action_items": parsed.get("action_items", []),
                # Numeric scores
                "scores": scores,
                # Full LLM text (useful for detailed views / report export)
                "full_report": llm_output,
                "metadata": {
                    "agent": self.name,
                    "rule_recommendation": rule_recommendation,
                    "agents_used": [
                        k for k, v in {"research": research, "sentiment": sentiment, "risk": risk}.items()
                        if v and "error" not in v
                    ],
                },
            }

            self._log_execution(input_data, output)
            logger.info(
                f"Summary complete for {symbol}: {final_recommendation} "
                f"(sentiment={scores['sentiment_score']:.2f}, risk={scores['risk_score']:.2f})"
            )
            return output

        except Exception as e:
            logger.error(f"Error in SummaryAgent.execute: {e}")
            return {
                "symbol": input_data.get("symbol", ""),
                "error": str(e),
                "status": "failed",
            }

    def generate_short_summary(
        self,
        symbol: str,
        research: Dict[str, Any],
        sentiment: Dict[str, Any],
        risk: Dict[str, Any],
    ) -> str:
        """
        Generate a brief 2-sentence summary suitable for dashboard cards.

        Args:
            symbol: Stock ticker.
            research: ResearchAgent output.
            sentiment: SentimentAgent output.
            risk: RiskAgent output.

        Returns:
            Two-sentence summary string.
        """
        try:
            prompt = self.short_summary_prompt.format(
                symbol=symbol,
                research=research.get("findings", "No research data")[:400],
                sentiment=(
                    f"Sentiment: {sentiment.get('overall_sentiment', 'N/A')}, "
                    f"Score: {sentiment.get('sentiment_score', 'N/A')}"
                ),
                risk=(
                    f"Risk Level: {risk.get('risk_level', 'N/A')}, "
                    f"Score: {risk.get('overall_risk_score', 'N/A')}"
                ),
            )
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"Error generating short summary: {e}")
            return f"Analysis for {symbol} could not be summarized at this time."

    def generate_report_markdown(self, synthesis: Dict[str, Any]) -> str:
        """
        Convert a synthesis dict (from execute()) into a Markdown report string.

        Args:
            synthesis: Output dict from execute().

        Returns:
            Formatted Markdown string.
        """
        symbol = synthesis.get("symbol", "N/A")
        date = synthesis.get("analysis_date", datetime.now().isoformat())[:10]
        scores = synthesis.get("scores", {})

        positives = "\n".join(
            f"- {p}" for p in synthesis.get("key_positives", ["No data"])
        )
        negatives = "\n".join(
            f"- {n}" for n in synthesis.get("key_negatives", ["No data"])
        )
        actions = "\n".join(
            f"{i+1}. {a}"
            for i, a in enumerate(synthesis.get("action_items", ["Monitor closely"]))
        )

        return f"""# Investment Research Report: {symbol}

**Date:** {date}  
**Period Covered:** {synthesis.get('period_days', 7)} days  
**Confidence:** {synthesis.get('confidence_label', 'N/A')} ({synthesis.get('confidence', 0):.0%})

---

## Executive Summary

{synthesis.get('executive_summary', 'Not available.')}

---

## Key Positives

{positives}

## Key Negatives / Risks

{negatives}

---

## Scores

| Metric | Score |
|---|---|
| Sentiment Score | {scores.get('sentiment_score', 0):.2f} |
| Risk Score | {scores.get('risk_score', 0):.2f} |
| Composite Score | {scores.get('composite_score', 0):.2f} |

---

## Investment Recommendation

**{synthesis.get('recommendation', 'N/A')}**

{synthesis.get('recommendation_rationale', '')}

---

## Action Items

{actions}

---

*Generated by Financial News Analyzer — SummaryAgent*
"""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _synthesize_with_llm(
        self,
        symbol: str,
        analysis_date: str,
        period_days: int,
        research: Dict[str, Any],
        sentiment: Dict[str, Any],
        risk: Dict[str, Any],
    ) -> str:
        """Call the LLM with the full synthesis prompt."""
        research_text = self._format_research(research)
        sentiment_text = self._format_sentiment(sentiment)
        risk_text = self._format_risk(risk)

        prompt = self.synthesis_prompt.format(
            symbol=symbol,
            analysis_date=analysis_date,
            period_days=period_days,
            research_findings=research_text,
            sentiment_analysis=sentiment_text,
            risk_assessment=risk_text,
        )

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            return f"Synthesis unavailable due to LLM error: {e}"

    def _parse_llm_output(self, text: str) -> Dict[str, Any]:
        """
        Parse the free-form LLM output into structured fields.

        This is a heuristic parser; works well when the LLM follows the
        prompt template. Falls back gracefully when sections are missing.
        """
        result: Dict[str, Any] = {
            "executive_summary": "",
            "key_positives": [],
            "key_negatives": [],
            "recommendation": None,
            "recommendation_rationale": "",
            "confidence_label": "MEDIUM",
            "action_items": [],
        }

        if not text:
            return result

        lines = text.splitlines()
        current_section = None

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            # Detect section headers
            if "executive summary" in lower:
                current_section = "executive_summary"
                continue
            elif "key positive" in lower:
                current_section = "key_positives"
                continue
            elif "key negative" in lower or "key risk" in lower:
                current_section = "key_negatives"
                continue
            elif "recommendation" in lower and "rationale" not in lower:
                current_section = "recommendation"
                continue
            elif "rationale" in lower:
                current_section = "rationale"
                continue
            elif "confidence" in lower:
                current_section = "confidence"
                continue
            elif "action item" in lower or "immediate action" in lower:
                current_section = "action_items"
                continue

            # Skip empty lines
            if not stripped:
                continue

            # Populate sections
            if current_section == "executive_summary":
                result["executive_summary"] += (" " if result["executive_summary"] else "") + stripped

            elif current_section == "key_positives" and stripped.startswith(("-", "•", "*")):
                result["key_positives"].append(stripped.lstrip("-•* "))

            elif current_section == "key_negatives" and stripped.startswith(("-", "•", "*")):
                result["key_negatives"].append(stripped.lstrip("-•* "))

            elif current_section == "recommendation":
                # Look for a recommendation keyword in the line
                for rec in ["STRONG BUY", "BUY", "HOLD", "SELL", "AVOID"]:
                    if rec in stripped.upper():
                        result["recommendation"] = rec
                        break
                result["recommendation_rationale"] += (" " if result["recommendation_rationale"] else "") + stripped

            elif current_section == "rationale":
                result["recommendation_rationale"] += (" " if result["recommendation_rationale"] else "") + stripped

            elif current_section == "confidence":
                for level in ["HIGH", "MEDIUM", "LOW"]:
                    if level in stripped.upper():
                        result["confidence_label"] = level
                        break

            elif current_section == "action_items":
                clean = stripped.lstrip("0123456789.-•* ")
                if clean:
                    result["action_items"].append(clean)

        return result

    def _compute_composite_scores(
        self,
        sentiment: Dict[str, Any],
        risk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Derive numeric scores and a composite investment score.

        Composite = 0.5 × sentiment_score + 0.5 × (1 − risk_score)
        Range: 0 (worst) → 1 (best)
        """
        sentiment_score = float(sentiment.get("sentiment_score", 0.5))
        risk_score = float(risk.get("overall_risk_score", 0.5))
        sentiment_confidence = float(sentiment.get("confidence", 0.5))

        composite = 0.5 * sentiment_score + 0.5 * (1.0 - risk_score)

        # Confidence: how consistent / high-quality are the inputs?
        if sentiment_confidence >= 0.75 and "error" not in sentiment and "error" not in risk:
            confidence_label = "HIGH"
            confidence = 0.85
        elif sentiment_confidence >= 0.5:
            confidence_label = "MEDIUM"
            confidence = 0.65
        else:
            confidence_label = "LOW"
            confidence = 0.40

        return {
            "sentiment_score": round(sentiment_score, 4),
            "risk_score": round(risk_score, 4),
            "composite_score": round(composite, 4),
            "confidence": confidence,
            "confidence_label": confidence_label,
        }

    def _derive_recommendation(self, sentiment_score: float, risk_score: float) -> str:
        """
        Map sentiment + risk scores to an investment recommendation.

        Decision matrix (priority order):
        - STRONG BUY : sentiment ≥ 0.75 AND risk ≤ 0.35
        - BUY        : sentiment ≥ 0.60 AND risk ≤ 0.50
        - HOLD       : sentiment ≥ 0.40 AND risk ≤ 0.65
        - SELL       : sentiment ≥ 0.25 AND risk ≤ 0.80
        - AVOID      : everything else
        """
        m = self.RECOMMENDATION_MATRIX
        if sentiment_score >= m["strong_buy"]["sentiment_min"] and risk_score <= m["strong_buy"]["risk_max"]:
            return "STRONG BUY"
        if sentiment_score >= m["buy"]["sentiment_min"] and risk_score <= m["buy"]["risk_max"]:
            return "BUY"
        if sentiment_score >= m["hold"]["sentiment_min"] and risk_score <= m["hold"]["risk_max"]:
            return "HOLD"
        if sentiment_score >= m["sell"]["sentiment_min"] and risk_score <= m["sell"]["risk_max"]:
            return "SELL"
        return "AVOID"

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_research(self, research: Dict[str, Any]) -> str:
        if not research:
            return "No research data available."
        if "error" in research:
            return f"Research agent error: {research['error']}"
        findings = research.get("findings", "No findings.")
        sources = ", ".join(research.get("sources_used", [])) or "N/A"
        return (
            f"Findings: {str(findings)[:1500]}\n"
            f"Sources used: {sources}\n"
            f"Period: {research.get('period_days', 'N/A')} days"
        )

    def _format_sentiment(self, sentiment: Dict[str, Any]) -> str:
        if not sentiment:
            return "No sentiment data available."
        if "error" in sentiment:
            return f"Sentiment agent error: {sentiment['error']}"
        return (
            f"Overall sentiment: {sentiment.get('overall_sentiment', 'N/A')}\n"
            f"Sentiment score: {sentiment.get('sentiment_score', 'N/A')}\n"
            f"Confidence: {sentiment.get('confidence', 'N/A')}\n"
            f"Articles analyzed: {sentiment.get('text_count', 'N/A')}\n"
            f"LLM analysis: {str(sentiment.get('llm_analysis', {}).get('analysis', 'N/A'))[:600]}"
        )

    def _format_risk(self, risk: Dict[str, Any]) -> str:
        if not risk:
            return "No risk data available."
        if "error" in risk:
            return f"Risk agent error: {risk['error']}"
        risks_summary = "; ".join(
            f"{r.get('category', '?')} ({r.get('severity', '?')})"
            for r in risk.get("identified_risks", [])
        ) or "None identified"
        recommendations = "; ".join(risk.get("recommendations", [])) or "None"
        return (
            f"Overall risk level: {risk.get('risk_level', 'N/A')}\n"
            f"Overall risk score: {risk.get('overall_risk_score', 'N/A')}\n"
            f"Identified risks: {risks_summary}\n"
            f"Recommendations: {recommendations}\n"
            f"Alerts: {len(risk.get('alerts', []))} alert(s) generated"
        )
