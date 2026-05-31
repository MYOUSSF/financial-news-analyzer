"""
Unit tests for PortfolioAgent.

Run with:
    pytest tests/test_portfolio_agent.py -v
"""
import os
import sys
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.portfolio_agent import (
    CONCENTRATION_THRESHOLD,
    CORRELATED_RISK_MIN_HOLDINGS,
    PortfolioAgent,
)
from src.agents.base import AgentExecutionError


# ===========================================================================
# Fixtures & helpers
# ===========================================================================

def _make_llm(text: str = "Portfolio looks balanced with moderate risk.") -> Mock:
    llm = Mock()
    response = Mock()
    response.content = text
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm
    return llm


def _make_agent(llm=None) -> PortfolioAgent:
    return PortfolioAgent(llm=llm or _make_llm())


def _analysis(
    symbol: str,
    sentiment: float = 0.6,
    risk: float = 0.4,
    recommendation: str = "BUY",
    risk_categories: list | None = None,
) -> Dict[str, Any]:
    """Build a minimal individual analysis result dict."""
    risk_categories = risk_categories or []
    composite = round(0.5 * sentiment + 0.5 * (1 - risk), 4)
    return {
        "symbol": symbol,
        "recommendation": recommendation,
        "scores": {
            "sentiment_score": sentiment,
            "risk_score": risk,
            "composite_score": composite,
        },
        "_risk": {
            "overall_risk_score": risk,
            "identified_risks": [
                {"category": cat, "severity": "MEDIUM", "likelihood": 0.5}
                for cat in risk_categories
            ],
        },
    }


def _input_data(
    holdings: list,
    individual_analyses: dict | None = None,
) -> Dict[str, Any]:
    if individual_analyses is None:
        individual_analyses = {
            h["symbol"].upper(): _analysis(h["symbol"]) for h in holdings
        }
    return {"holdings": holdings, "individual_analyses": individual_analyses}


# ===========================================================================
# Initialisation
# ===========================================================================

class TestPortfolioAgentInit:

    def test_name(self):
        assert _make_agent().name == "PortfolioAgent"

    def test_no_tools_by_default(self):
        assert _make_agent().tools == []

    def test_verbose_flag(self):
        agent = PortfolioAgent(llm=_make_llm(), verbose=True)
        assert agent.verbose is True

    def test_exported_from_package(self):
        from src.agents import PortfolioAgent as PA
        assert PA is PortfolioAgent


# ===========================================================================
# execute() — happy path
# ===========================================================================

class TestPortfolioAgentExecute:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    @pytest.fixture
    def two_holdings(self):
        return [
            {"symbol": "AAPL", "weight": 0.60},
            {"symbol": "MSFT", "weight": 0.40},
        ]

    @pytest.fixture
    def two_analyses(self):
        return {
            "AAPL": _analysis("AAPL", sentiment=0.70, risk=0.30),
            "MSFT": _analysis("MSFT", sentiment=0.60, risk=0.40),
        }

    def test_returns_required_keys(self, agent, two_holdings, two_analyses):
        result = agent.execute({"holdings": two_holdings, "individual_analyses": two_analyses})
        for key in (
            "portfolio_sentiment_score",
            "portfolio_risk_score",
            "portfolio_composite_score",
            "portfolio_recommendation",
            "portfolio_confidence",
            "concentration_risks",
            "correlated_risks",
            "holdings_table",
            "executive_summary",
            "analysis_date",
            "metadata",
        ):
            assert key in result, f"Missing key: {key}"

    def test_metadata_agent_name(self, agent, two_holdings, two_analyses):
        result = agent.execute({"holdings": two_holdings, "individual_analyses": two_analyses})
        assert result["metadata"]["agent"] == "PortfolioAgent"

    def test_analysis_date_is_iso_string(self, agent, two_holdings, two_analyses):
        result = agent.execute({"holdings": two_holdings, "individual_analyses": two_analyses})
        # Should not raise
        from datetime import datetime
        datetime.fromisoformat(result["analysis_date"])

    # ── Weighted scores ───────────────────────────────────────────────

    def test_weighted_sentiment_score(self, agent, two_holdings, two_analyses):
        result = agent.execute({"holdings": two_holdings, "individual_analyses": two_analyses})
        expected = 0.60 * 0.70 + 0.40 * 0.60
        assert result["portfolio_sentiment_score"] == pytest.approx(expected, abs=1e-3)

    def test_weighted_risk_score(self, agent, two_holdings, two_analyses):
        result = agent.execute({"holdings": two_holdings, "individual_analyses": two_analyses})
        expected = 0.60 * 0.30 + 0.40 * 0.40
        assert result["portfolio_risk_score"] == pytest.approx(expected, abs=1e-3)

    def test_composite_score_formula(self, agent, two_holdings, two_analyses):
        result = agent.execute({"holdings": two_holdings, "individual_analyses": two_analyses})
        p_sent = result["portfolio_sentiment_score"]
        p_risk = result["portfolio_risk_score"]
        expected_composite = 0.5 * p_sent + 0.5 * (1 - p_risk)
        assert result["portfolio_composite_score"] == pytest.approx(expected_composite, abs=1e-4)

    def test_single_holding_scores_equal_underlying(self, agent):
        holdings = [{"symbol": "TSLA", "weight": 1.0}]
        analyses = {"TSLA": _analysis("TSLA", sentiment=0.72, risk=0.35)}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["portfolio_sentiment_score"] == pytest.approx(0.72)
        assert result["portfolio_risk_score"] == pytest.approx(0.35)

    # ── Holdings table ────────────────────────────────────────────────

    def test_holdings_table_length(self, agent, two_holdings, two_analyses):
        result = agent.execute({"holdings": two_holdings, "individual_analyses": two_analyses})
        assert len(result["holdings_table"]) == 2

    def test_holdings_table_keys(self, agent, two_holdings, two_analyses):
        result = agent.execute({"holdings": two_holdings, "individual_analyses": two_analyses})
        row = result["holdings_table"][0]
        for key in ("symbol", "weight", "recommendation", "sentiment_score", "risk_score", "composite_score"):
            assert key in row

    def test_holdings_table_weight_preserved(self, agent, two_holdings, two_analyses):
        result = agent.execute({"holdings": two_holdings, "individual_analyses": two_analyses})
        weights = {h["symbol"]: h["weight"] for h in result["holdings_table"]}
        assert weights["AAPL"] == pytest.approx(0.60)
        assert weights["MSFT"] == pytest.approx(0.40)

    def test_symbol_upcased_in_table(self, agent):
        holdings = [{"symbol": "aapl", "weight": 1.0}]
        analyses = {"AAPL": _analysis("AAPL")}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["holdings_table"][0]["symbol"] == "AAPL"

    # ── Recommendation ────────────────────────────────────────────────

    def test_strong_buy_recommendation(self, agent):
        holdings = [{"symbol": "AAPL", "weight": 1.0}]
        analyses = {"AAPL": _analysis("AAPL", sentiment=0.80, risk=0.20)}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["portfolio_recommendation"] == "STRONG BUY"

    def test_avoid_recommendation(self, agent):
        holdings = [{"symbol": "AAPL", "weight": 1.0}]
        analyses = {"AAPL": _analysis("AAPL", sentiment=0.15, risk=0.90)}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["portfolio_recommendation"] == "AVOID"

    def test_hold_recommendation(self, agent):
        holdings = [{"symbol": "AAPL", "weight": 1.0}]
        analyses = {"AAPL": _analysis("AAPL", sentiment=0.50, risk=0.50)}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["portfolio_recommendation"] == "HOLD"

    # ── Confidence ────────────────────────────────────────────────────

    def test_high_confidence_for_strong_composite(self, agent):
        holdings = [{"symbol": "AAPL", "weight": 1.0}]
        analyses = {"AAPL": _analysis("AAPL", sentiment=0.80, risk=0.20)}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["portfolio_confidence"] == "HIGH"

    def test_low_confidence_for_weak_composite(self, agent):
        holdings = [{"symbol": "AAPL", "weight": 1.0}]
        analyses = {"AAPL": _analysis("AAPL", sentiment=0.15, risk=0.90)}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["portfolio_confidence"] == "LOW"


# ===========================================================================
# Concentration risk
# ===========================================================================

class TestConcentrationRisk:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    def test_no_concentration_risk_under_threshold(self, agent):
        holdings = [
            {"symbol": "AAPL", "weight": 0.25},
            {"symbol": "MSFT", "weight": 0.25},
            {"symbol": "GOOGL", "weight": 0.25},
            {"symbol": "AMZN", "weight": 0.25},
        ]
        analyses = {s["symbol"]: _analysis(s["symbol"]) for s in holdings}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["concentration_risks"] == []

    def test_concentration_risk_above_threshold(self, agent):
        # AAPL at 0.80 exceeds 0.25; MSFT at 0.20 does not
        holdings = [
            {"symbol": "AAPL", "weight": 0.80},
            {"symbol": "MSFT", "weight": 0.20},
        ]
        analyses = {s["symbol"]: _analysis(s["symbol"]) for s in holdings}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        flagged = [c["symbol"] for c in result["concentration_risks"]]
        assert "AAPL" in flagged
        assert "MSFT" not in flagged

    def test_concentration_risk_at_exactly_threshold_not_flagged(self, agent):
        # CONCENTRATION_THRESHOLD = 0.25; exactly 0.25 should NOT be flagged (> not >=)
        holdings = [
            {"symbol": "AAPL", "weight": 0.25},
            {"symbol": "MSFT", "weight": 0.75},
        ]
        analyses = {s["symbol"]: _analysis(s["symbol"]) for s in holdings}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        flagged = [c["symbol"] for c in result["concentration_risks"]]
        assert "AAPL" not in flagged

    def test_all_holdings_can_be_flagged(self, agent):
        holdings = [
            {"symbol": "AAPL", "weight": 0.50},
            {"symbol": "MSFT", "weight": 0.50},
        ]
        analyses = {s["symbol"]: _analysis(s["symbol"]) for s in holdings}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert len(result["concentration_risks"]) == 2

    def test_concentration_risk_contains_weight(self, agent):
        holdings = [
            {"symbol": "AAPL", "weight": 0.70},
            {"symbol": "MSFT", "weight": 0.30},
        ]
        analyses = {s["symbol"]: _analysis(s["symbol"]) for s in holdings}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        flagged_aapl = next(c for c in result["concentration_risks"] if c["symbol"] == "AAPL")
        assert flagged_aapl["weight"] == pytest.approx(0.70)


# ===========================================================================
# Correlated risks
# ===========================================================================

class TestCorrelatedRisks:

    @pytest.fixture
    def agent(self):
        return _make_agent()

    def test_no_correlated_risks_with_two_holdings(self, agent):
        # Only 2 holdings — even if same category, < threshold (> 2)
        holdings = [
            {"symbol": "AAPL", "weight": 0.50},
            {"symbol": "MSFT", "weight": 0.50},
        ]
        analyses = {
            "AAPL": _analysis("AAPL", risk_categories=["regulatory"]),
            "MSFT": _analysis("MSFT", risk_categories=["regulatory"]),
        }
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["correlated_risks"] == []

    def test_correlated_risk_detected_across_three_holdings(self, agent):
        holdings = [
            {"symbol": "AAPL", "weight": 0.34},
            {"symbol": "MSFT", "weight": 0.33},
            {"symbol": "GOOGL", "weight": 0.33},
        ]
        analyses = {
            "AAPL":  _analysis("AAPL",  risk_categories=["regulatory"]),
            "MSFT":  _analysis("MSFT",  risk_categories=["regulatory"]),
            "GOOGL": _analysis("GOOGL", risk_categories=["regulatory"]),
        }
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        cats = [c["category"] for c in result["correlated_risks"]]
        assert "regulatory" in cats

    def test_correlated_risk_symbols_list(self, agent):
        holdings = [
            {"symbol": "AAPL",  "weight": 0.34},
            {"symbol": "MSFT",  "weight": 0.33},
            {"symbol": "GOOGL", "weight": 0.33},
        ]
        analyses = {
            "AAPL":  _analysis("AAPL",  risk_categories=["volatility"]),
            "MSFT":  _analysis("MSFT",  risk_categories=["volatility"]),
            "GOOGL": _analysis("GOOGL", risk_categories=["volatility"]),
        }
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        vol_risk = next(c for c in result["correlated_risks"] if c["category"] == "volatility")
        assert set(vol_risk["symbols"]) == {"AAPL", "MSFT", "GOOGL"}

    def test_uncorrelated_categories_not_flagged(self, agent):
        holdings = [
            {"symbol": "AAPL",  "weight": 0.34},
            {"symbol": "MSFT",  "weight": 0.33},
            {"symbol": "GOOGL", "weight": 0.33},
        ]
        analyses = {
            "AAPL":  _analysis("AAPL",  risk_categories=["regulatory"]),
            "MSFT":  _analysis("MSFT",  risk_categories=["volatility"]),
            "GOOGL": _analysis("GOOGL", risk_categories=["financial"]),
        }
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["correlated_risks"] == []

    def test_category_counted_once_per_holding(self, agent):
        """A category listed twice in one holding must only be counted once for that holding."""
        holdings = [
            {"symbol": "AAPL",  "weight": 0.34},
            {"symbol": "MSFT",  "weight": 0.33},
            {"symbol": "GOOGL", "weight": 0.33},
        ]
        analyses = {
            "AAPL":  _analysis("AAPL",  risk_categories=["regulatory"]),
            "MSFT":  _analysis("MSFT",  risk_categories=["regulatory", "regulatory"]),
            "GOOGL": _analysis("GOOGL", risk_categories=["regulatory"]),
        }
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        reg = next(c for c in result["correlated_risks"] if c["category"] == "regulatory")
        # Each symbol should appear exactly once
        assert sorted(reg["symbols"]) == sorted(["AAPL", "MSFT", "GOOGL"])


# ===========================================================================
# Error handling
# ===========================================================================

class TestPortfolioAgentErrors:

    def test_raises_agent_execution_error_on_empty_holdings(self):
        agent = _make_agent()
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute({"holdings": [], "individual_analyses": {}})
        assert "PortfolioAgent" in str(exc_info.value)

    def test_raises_on_zero_total_weight(self):
        agent = _make_agent()
        holdings = [{"symbol": "AAPL", "weight": 0.0}]
        with pytest.raises(AgentExecutionError):
            agent.execute({"holdings": holdings, "individual_analyses": {}})

    def test_missing_analysis_falls_back_to_defaults(self):
        """If an analysis is absent, default scores (0.5) are used without crashing."""
        agent = _make_agent()
        holdings = [{"symbol": "AAPL", "weight": 1.0}]
        # individual_analyses is empty — AAPL not present
        result = agent.execute({"holdings": holdings, "individual_analyses": {}})
        assert result["portfolio_sentiment_score"] == pytest.approx(0.5)
        assert result["portfolio_risk_score"] == pytest.approx(0.5)

    def test_llm_failure_uses_fallback_summary(self):
        failing_llm = Mock()
        failing_llm.invoke.side_effect = RuntimeError("LLM timeout")
        agent = PortfolioAgent(llm=failing_llm)
        holdings = [{"symbol": "AAPL", "weight": 1.0}]
        analyses = {"AAPL": _analysis("AAPL")}
        # Should not raise — fallback summary is generated
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert isinstance(result["executive_summary"], str)
        assert len(result["executive_summary"]) > 0

    def test_weights_warning_does_not_raise(self, capfd):
        agent = _make_agent()
        holdings = [
            {"symbol": "AAPL", "weight": 0.60},
            {"symbol": "MSFT", "weight": 0.20},  # intentionally sums to 0.80
        ]
        analyses = {s["symbol"]: _analysis(s["symbol"]) for s in holdings}
        # Should complete without raising even though weights don't sum to 1
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert "portfolio_recommendation" in result


# ===========================================================================
# LLM integration
# ===========================================================================

class TestPortfolioAgentLLM:

    def test_llm_invoked_once_for_summary(self):
        llm = _make_llm("A balanced portfolio with moderate risk.")
        agent = PortfolioAgent(llm=llm)
        holdings = [{"symbol": "AAPL", "weight": 1.0}]
        analyses = {"AAPL": _analysis("AAPL")}
        agent.execute({"holdings": holdings, "individual_analyses": analyses})
        llm.invoke.assert_called_once()

    def test_llm_response_used_as_executive_summary(self):
        expected_text = "Custom LLM portfolio narrative."
        agent = _make_agent(_make_llm(expected_text))
        holdings = [{"symbol": "AAPL", "weight": 1.0}]
        analyses = {"AAPL": _analysis("AAPL")}
        result = agent.execute({"holdings": holdings, "individual_analyses": analyses})
        assert result["executive_summary"] == expected_text


# ===========================================================================
# FinancialAnalysisChain.analyze_portfolio integration
# ===========================================================================

class TestAnalyzePortfolioChain:

    @pytest.fixture
    def chain(self):
        from src.chains.analysis_chain import FinancialAnalysisChain
        mock_llm = _make_llm()
        with patch.object(FinancialAnalysisChain, "_build_tools", return_value=[]), \
             patch.object(FinancialAnalysisChain, "_init_agents"), \
             patch.object(FinancialAnalysisChain, "_init_vector_store", return_value=None):
            c = FinancialAnalysisChain(llm=mock_llm)
        return c

    def test_analyze_portfolio_calls_analyze_stock_per_holding(self, chain):
        holdings = [
            {"symbol": "AAPL", "weight": 0.60},
            {"symbol": "MSFT", "weight": 0.40},
        ]
        mock_stock_result = {
            "symbol": "AAPL",
            "recommendation": "BUY",
            "scores": {"sentiment_score": 0.7, "risk_score": 0.3, "composite_score": 0.7},
            "_risk": {"identified_risks": [], "overall_risk_score": 0.3},
        }
        with patch.object(chain, "analyze_stock", return_value=mock_stock_result) as mock_analyze:
            result = chain.analyze_portfolio(holdings=holdings, days_back=7)
        assert mock_analyze.call_count == 2
        assert "portfolio_recommendation" in result

    def test_analyze_portfolio_handles_partial_failure(self, chain):
        """If one holding fails, the portfolio result is still produced."""
        holdings = [
            {"symbol": "AAPL", "weight": 0.60},
            {"symbol": "FAIL", "weight": 0.40},
        ]

        def _side_effect(symbol, **kwargs):
            if symbol == "FAIL":
                raise RuntimeError("network error")
            return {
                "symbol": symbol,
                "recommendation": "BUY",
                "scores": {"sentiment_score": 0.7, "risk_score": 0.3, "composite_score": 0.7},
                "_risk": {"identified_risks": [], "overall_risk_score": 0.3},
            }

        with patch.object(chain, "analyze_stock", side_effect=_side_effect):
            result = chain.analyze_portfolio(holdings=holdings, days_back=7)

        assert "portfolio_recommendation" in result


# ===========================================================================
# Session-scoped setup
# ===========================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    yield
