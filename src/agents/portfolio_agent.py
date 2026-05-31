"""
Portfolio Agent — aggregates individual stock analyses into portfolio-level insights.
"""
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from src.agents.base import AgentExecutionError, BaseAgent


_RECOMMENDATION_MATRIX = {
    "strong_buy": {"sentiment_min": 0.75, "risk_max": 0.35},
    "buy":        {"sentiment_min": 0.60, "risk_max": 0.50},
    "hold":       {"sentiment_min": 0.40, "risk_max": 0.65},
    "sell":       {"sentiment_min": 0.25, "risk_max": 0.80},
    "avoid":      {"sentiment_min": 0.00, "risk_max": 1.00},
}

CONCENTRATION_THRESHOLD = 0.25
CORRELATED_RISK_MIN_HOLDINGS = 2  # > this many holdings → correlated


class PortfolioAgent(BaseAgent):
    """
    Aggregates individual stock analyses into a portfolio-level investment report.

    Capabilities:
    - Weighted portfolio sentiment and risk scores
    - Concentration risk flag (any single holding > 25% weight)
    - Correlated risk identification (risk category shared across > 2 holdings)
    - Portfolio recommendation from the same rule-based matrix as SummaryAgent
    - LLM-written executive narrative with structured fallback on failure
    """

    def __init__(
        self,
        llm: Any,
        tools: Optional[List[Any]] = None,
        verbose: bool = False,
    ):
        super().__init__(
            name="PortfolioAgent",
            description="Aggregates individual analyses into portfolio-level insights",
            llm=llm,
            tools=tools or [],
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a portfolio-level analysis report.

        Args:
            input_data: Dict with:
                - ``holdings``: ``[{"symbol": "AAPL", "weight": 0.40}, ...]``
                  Weights should sum to 1.0 (a warning is logged if they don't).
                - ``individual_analyses``: ``{"AAPL": <analyze_stock result>, ...}``

        Returns:
            Dict with:
            - ``portfolio_sentiment_score``, ``portfolio_risk_score``,
              ``portfolio_composite_score``
            - ``portfolio_recommendation``, ``portfolio_confidence``
            - ``concentration_risks``: holdings whose weight exceeds 25 %
            - ``correlated_risks``: risk categories shared across > 2 holdings
            - ``holdings_table``: per-holding scores and recommendation
            - ``executive_summary``: LLM-written narrative
            - ``analysis_date``, ``metadata``
        """
        try:
            holdings: List[Dict[str, Any]] = input_data.get("holdings", [])
            individual_analyses: Dict[str, Any] = input_data.get("individual_analyses", {})

            if not holdings:
                raise ValueError("No holdings provided")

            total_weight = sum(float(h.get("weight", 0)) for h in holdings)
            if total_weight <= 0:
                raise ValueError("Holdings weights must sum to a positive number")
            if abs(total_weight - 1.0) > 0.05:
                logger.warning(
                    f"Portfolio weights sum to {total_weight:.3f}; expected 1.0"
                )

            # ── Per-holding table ─────────────────────────────────────────
            holdings_table = self._build_holdings_table(holdings, individual_analyses)

            # ── Weighted aggregate scores ─────────────────────────────────
            w_total = sum(h["weight"] for h in holdings_table)
            p_sentiment = (
                sum(h["weight"] * h["sentiment_score"] for h in holdings_table) / w_total
            )
            p_risk = (
                sum(h["weight"] * h["risk_score"] for h in holdings_table) / w_total
            )
            p_composite = 0.5 * p_sentiment + 0.5 * (1.0 - p_risk)

            # ── Recommendation & confidence ───────────────────────────────
            recommendation = self._derive_recommendation(p_sentiment, p_risk)
            confidence = self._derive_confidence(p_composite)

            # ── Concentration risk ────────────────────────────────────────
            concentration_risks = [
                {"symbol": h["symbol"], "weight": h["weight"]}
                for h in holdings_table
                if h["weight"] > CONCENTRATION_THRESHOLD
            ]

            # ── Correlated risks ──────────────────────────────────────────
            correlated_risks = self._find_correlated_risks(holdings, individual_analyses)

            # ── LLM narrative ─────────────────────────────────────────────
            executive_summary = self._generate_summary(
                holdings_table=holdings_table,
                p_sentiment=p_sentiment,
                p_risk=p_risk,
                recommendation=recommendation,
                concentration_risks=concentration_risks,
                correlated_risks=correlated_risks,
            )

            output = {
                "portfolio_sentiment_score": round(p_sentiment, 4),
                "portfolio_risk_score": round(p_risk, 4),
                "portfolio_composite_score": round(p_composite, 4),
                "portfolio_recommendation": recommendation,
                "portfolio_confidence": confidence,
                "concentration_risks": concentration_risks,
                "correlated_risks": correlated_risks,
                "holdings_table": holdings_table,
                "executive_summary": executive_summary,
                "analysis_date": datetime.now().isoformat(),
                "metadata": {"agent": self.name},
            }

            self._log_execution(input_data, output)
            logger.info(
                f"Portfolio analysis complete: {recommendation} "
                f"(sentiment={p_sentiment:.2f}, risk={p_risk:.2f}, "
                f"composite={p_composite:.2f})"
            )
            return output

        except Exception as e:
            logger.error(f"PortfolioAgent.execute error: {e}")
            raise AgentExecutionError(
                agent_name=self.name,
                original_error=e,
                input_data=input_data,
            ) from e

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_holdings_table(
        self,
        holdings: List[Dict[str, Any]],
        individual_analyses: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build a normalized per-holding table with scores from analyses."""
        table = []
        for h in holdings:
            symbol = h["symbol"].upper()
            weight = float(h.get("weight", 0))
            analysis = individual_analyses.get(symbol, {})
            scores = analysis.get("scores", {})
            table.append({
                "symbol": symbol,
                "weight": weight,
                "recommendation": analysis.get("recommendation", "HOLD"),
                "sentiment_score": float(scores.get("sentiment_score", 0.5)),
                "risk_score": float(scores.get("risk_score", 0.5)),
                "composite_score": float(scores.get("composite_score", 0.5)),
            })
        return table

    def _find_correlated_risks(
        self,
        holdings: List[Dict[str, Any]],
        individual_analyses: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Identify risk categories that appear across more than
        ``CORRELATED_RISK_MIN_HOLDINGS`` holdings.
        """
        category_symbols: Dict[str, List[str]] = defaultdict(list)
        for h in holdings:
            symbol = h["symbol"].upper()
            risk_data = individual_analyses.get(symbol, {}).get("_risk", {})
            seen: set = set()
            for risk in risk_data.get("identified_risks", []):
                cat = risk.get("category", "").lower().strip()
                if cat and cat not in seen:
                    category_symbols[cat].append(symbol)
                    seen.add(cat)
        return [
            {"category": cat, "symbols": syms}
            for cat, syms in sorted(category_symbols.items())
            if len(syms) > CORRELATED_RISK_MIN_HOLDINGS
        ]

    @staticmethod
    def _derive_recommendation(sentiment: float, risk: float) -> str:
        m = _RECOMMENDATION_MATRIX
        if sentiment >= m["strong_buy"]["sentiment_min"] and risk <= m["strong_buy"]["risk_max"]:
            return "STRONG BUY"
        if sentiment >= m["buy"]["sentiment_min"] and risk <= m["buy"]["risk_max"]:
            return "BUY"
        if sentiment >= m["hold"]["sentiment_min"] and risk <= m["hold"]["risk_max"]:
            return "HOLD"
        if sentiment >= m["sell"]["sentiment_min"] and risk <= m["sell"]["risk_max"]:
            return "SELL"
        return "AVOID"

    @staticmethod
    def _derive_confidence(composite: float) -> str:
        if composite >= 0.65:
            return "HIGH"
        if composite >= 0.45:
            return "MEDIUM"
        return "LOW"

    def _generate_summary(
        self,
        holdings_table: List[Dict[str, Any]],
        p_sentiment: float,
        p_risk: float,
        recommendation: str,
        concentration_risks: List[Dict[str, Any]],
        correlated_risks: List[Dict[str, Any]],
    ) -> str:
        """Call the LLM for a narrative summary; fall back to a structured string."""
        symbols = ", ".join(h["symbol"] for h in holdings_table)
        conc_note = (
            f"Concentration risk flagged for: "
            f"{', '.join(c['symbol'] for c in concentration_risks)}."
            if concentration_risks
            else "No single-holding concentration risk."
        )
        corr_note = (
            f"Correlated risks across holdings: "
            f"{', '.join(c['category'] for c in correlated_risks)}."
            if correlated_risks
            else "No significant cross-holding risk correlation."
        )

        prompt = (
            "You are a portfolio risk analyst. Summarize this portfolio in exactly "
            "3 sentences covering the aggregate risk-return profile and any concentration "
            "or correlation concerns.\n\n"
            f"Holdings: {symbols}\n"
            f"Weighted Sentiment: {p_sentiment:.2f}  Weighted Risk: {p_risk:.2f}\n"
            f"Portfolio Recommendation: {recommendation}\n"
            f"{conc_note}\n{corr_note}"
        )

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.warning(f"PortfolioAgent LLM narrative failed ({exc}); using fallback")
            return (
                f"Portfolio of {len(holdings_table)} holdings ({symbols}) has a weighted "
                f"sentiment score of {p_sentiment:.2f} and risk score of {p_risk:.2f}, "
                f"yielding a {recommendation} recommendation. "
                f"{conc_note} {corr_note}"
            )
