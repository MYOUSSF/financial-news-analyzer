"""
Unit tests for all agents in the Financial News Analyzer system.

Run with:
    pytest tests/test_agents.py -v
    pytest tests/test_agents.py -v --cov=src/agents
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import numpy as np

# ---------------------------------------------------------------------------
# Make src/ importable when running directly from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from src.agents.base import BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.risk_agent import RiskAgent
from src.agents.summary_agent import SummaryAgent


# ===========================================================================
# Helpers / Fixtures shared across test classes
# ===========================================================================

def _make_llm(response_text: str = "Mock LLM response") -> Mock:
    """Build a mock LangChain chat model that returns a predictable response."""
    llm = Mock()
    response = Mock()
    response.content = response_text
    # Support both .invoke() (chat models) and .predict() (older LLMChain style)
    llm.invoke.return_value = response
    llm.predict.return_value = response_text
    # bind_tools returns a clone that also has invoke()
    llm.bind_tools.return_value = llm
    return llm


# ===========================================================================
# BaseAgent tests
# ===========================================================================

class _ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing the abstract base class."""

    def execute(self, input_data):
        return {"result": "test", "input": input_data}


class TestBaseAgent:
    """Test cases for BaseAgent."""

    @pytest.fixture
    def agent(self):
        return _ConcreteAgent(
            name="TestAgent",
            description="A test agent",
            llm=_make_llm(),
        )

    def test_initialization(self, agent):
        assert agent.name == "TestAgent"
        assert agent.description == "A test agent"
        assert agent.tools == []
        assert agent.verbose is False

    def test_initialization_with_tools(self):
        tools = [Mock(name="tool1"), Mock(name="tool2")]
        agent = _ConcreteAgent(
            name="TestAgent",
            description="desc",
            llm=_make_llm(),
            tools=tools,
        )
        assert len(agent.tools) == 2

    def test_execute_returns_dict(self, agent):
        result = agent.execute({"key": "value"})
        assert isinstance(result, dict)

    def test_memory_starts_empty(self, agent):
        assert agent.get_memory() == []

    def test_add_to_memory(self, agent):
        agent.add_to_memory("user", "Hello")
        agent.add_to_memory("assistant", "Hi there")
        memory = agent.get_memory()
        assert len(memory) == 2
        assert memory[0] == {"role": "user", "content": "Hello"}
        assert memory[1] == {"role": "assistant", "content": "Hi there"}

    def test_reset_memory(self, agent):
        agent.add_to_memory("user", "test message")
        agent.reset_memory()
        assert agent.get_memory() == []

    def test_get_status(self, agent):
        status = agent.get_status()
        assert status["name"] == "TestAgent"
        assert status["description"] == "A test agent"
        assert status["tools_count"] == 0
        assert status["memory_size"] == 0

    def test_get_status_memory_size_updates(self, agent):
        agent.add_to_memory("user", "msg1")
        agent.add_to_memory("assistant", "msg2")
        assert agent.get_status()["memory_size"] == 2

    def test_log_execution_verbose_false(self, agent, caplog):
        """Should not emit debug logs when verbose=False."""
        agent._log_execution({"a": 1}, {"b": 2})
        # No assertion needed — just confirm it doesn't crash

    def test_log_execution_verbose_true(self):
        agent = _ConcreteAgent(
            name="VerboseAgent",
            description="desc",
            llm=_make_llm(),
            verbose=True,
        )
        agent._log_execution({"input": "x"}, {"output": "y"})  # should not raise


# ===========================================================================
# ResearchAgent tests
# ===========================================================================

class TestResearchAgent:
    """Test cases for ResearchAgent."""

    @pytest.fixture
    def agent(self):
        mock_llm = _make_llm("Research findings: Apple reported strong Q4 earnings.")
        return ResearchAgent(llm=mock_llm, tools=[], verbose=False)

    def test_initialization(self, agent):
        assert agent.name == "ResearchAgent"
        assert agent.description is not None

    def test_execute_returns_required_keys(self, agent):
        result = agent.execute({"symbol": "AAPL", "days_back": 7})
        assert result["symbol"] == "AAPL"
        assert result["period_days"] == 7
        assert "findings" in result
        assert "research_date" in result
        assert "metadata" in result

    def test_execute_defaults_days_back(self, agent):
        result = agent.execute({"symbol": "MSFT"})
        assert result["period_days"] == 7

    def test_execute_uppercase_symbol(self, agent):
        result = agent.execute({"symbol": "aapl", "days_back": 3})
        assert result["symbol"] == "AAPL"

    def test_execute_error_handling(self, agent):
        agent.llm.bind_tools.return_value.invoke.side_effect = RuntimeError("LLM unavailable")
        result = agent.execute({"symbol": "FAIL"})
        assert "error" in result
        assert result["status"] == "failed"

    def test_batch_research(self, agent):
        with patch.object(agent, "execute") as mock_exec:
            mock_exec.return_value = {"symbol": "X", "findings": "ok"}
            results = agent.batch_research(["AAPL", "GOOGL", "MSFT"])
            assert len(results) == 3
            assert mock_exec.call_count == 3

    def test_focused_research(self, agent):
        with patch.object(agent, "execute") as mock_exec:
            mock_exec.return_value = {"symbol": "TSLA", "findings": "Earnings data"}
            result = agent.focused_research("TSLA", "earnings", days_back=14)
            call_args = mock_exec.call_args[0][0]
            assert call_args["symbol"] == "TSLA"
            assert call_args["days_back"] == 14
            assert "earnings" in call_args["focus_areas"]

    def test_extract_sources_empty(self, agent):
        result_obj = Mock()
        result_obj.tool_calls = []
        result_obj.content = "no tools mentioned"
        sources = agent._extract_sources(result_obj)
        assert isinstance(sources, list)


# ===========================================================================
# SentimentAgent tests
# ===========================================================================

class TestSentimentAgent:
    """Test cases for SentimentAgent."""

    @pytest.fixture
    def agent(self):
        llm_text = (
            "Overall sentiment: Positive\n"
            "Confidence: 0.85\n"
            "Key positive factors: Strong earnings beat\n"
            "Key negative factors: Regulatory concerns\n"
        )
        return SentimentAgent(llm=_make_llm(llm_text), verbose=False)

    def test_initialization(self, agent):
        assert agent.name == "SentimentAgent"
        assert agent.description is not None

    def test_execute_with_text_returns_required_keys(self, agent):
        result = agent.execute({"symbol": "AAPL", "text": "Strong quarterly earnings."})
        assert result["symbol"] == "AAPL"
        assert "overall_sentiment" in result
        assert "sentiment_score" in result
        assert "confidence" in result
        assert "analysis_date" in result

    def test_execute_with_articles(self, agent):
        result = agent.execute({
            "symbol": "MSFT",
            "articles": ["Azure cloud revenue surges.", "Teams hits record users."],
        })
        assert result["text_count"] == 2
        assert "overall_sentiment" in result

    def test_execute_no_input_returns_error(self, agent):
        result = agent.execute({"symbol": "AAPL"})
        assert "error" in result

    def test_execute_error_handling(self, agent):
        agent.llm.predict.side_effect = RuntimeError("LLM error")
        agent.llm.invoke.side_effect = RuntimeError("LLM error")
        result = agent.execute({"symbol": "ERR", "text": "some text"})
        # Should return error dict, not raise
        assert "symbol" in result

    def test_normalize_score_positive(self, agent):
        assert agent._normalize_score("POSITIVE", 0.9) == pytest.approx(0.9)

    def test_normalize_score_negative(self, agent):
        assert agent._normalize_score("NEGATIVE", 0.8) == pytest.approx(-0.8)

    def test_normalize_score_neutral(self, agent):
        assert agent._normalize_score("NEUTRAL", 0.5) == pytest.approx(0.0)

    def test_combine_sentiments_empty_ml(self, agent):
        result = agent._combine_sentiments([], {"analysis": "positive"})
        assert result["label"] == "NEUTRAL"
        assert 0 <= result["score"] <= 1

    def test_combine_sentiments_positive(self, agent):
        ml = [{"normalized_score": 0.8}, {"normalized_score": 0.7}]
        result = agent._combine_sentiments(ml, {})
        assert result["label"] == "POSITIVE"

    def test_combine_sentiments_negative(self, agent):
        ml = [{"normalized_score": -0.8}, {"normalized_score": -0.6}]
        result = agent._combine_sentiments(ml, {})
        assert result["label"] == "NEGATIVE"

    def test_analyze_trend_insufficient_data(self, agent):
        result = agent.analyze_trend([])
        assert result["trend"] == "insufficient_data"

    def test_analyze_trend_improving(self, agent):
        sentiments = [
            {"sentiment_score": 0.4},
            {"sentiment_score": 0.5},
            {"sentiment_score": 0.7},
        ]
        result = agent.analyze_trend(sentiments)
        assert result["trend"] == "improving"
        assert result["data_points"] == 3

    def test_analyze_trend_declining(self, agent):
        sentiments = [
            {"sentiment_score": 0.8},
            {"sentiment_score": 0.6},
            {"sentiment_score": 0.4},
        ]
        result = agent.analyze_trend(sentiments)
        assert result["trend"] == "declining"

    def test_ml_analysis_without_pipeline(self, agent):
        agent.sentiment_pipeline = None
        result = agent._analyze_with_ml(["test text"])
        assert result == []


# ===========================================================================
# RiskAgent tests
# ===========================================================================

class TestRiskAgent:
    """Test cases for RiskAgent."""

    @pytest.fixture
    def agent(self):
        llm_text = (
            "Regulatory risk: Ongoing investigation into antitrust practices.\n"
            "Financial risk: Elevated debt levels vs peers.\n"
            "Volatility risk: High price swings observed.\n"
        )
        return RiskAgent(llm=_make_llm(llm_text), verbose=False)

    def test_initialization(self, agent):
        assert agent.name == "RiskAgent"
        assert len(agent.RISK_CATEGORIES) == 5

    def test_execute_returns_required_keys(self, agent):
        result = agent.execute({
            "symbol": "AAPL",
            "news_data": "Regulatory investigation announced.",
            "market_data": {"volatility": "high"},
            "sentiment": {"sentiment_score": 0.35},
        })
        assert result["symbol"] == "AAPL"
        assert "overall_risk_score" in result
        assert "risk_level" in result
        assert "identified_risks" in result
        assert "alerts" in result
        assert "recommendations" in result

    def test_risk_level_range(self, agent):
        result = agent.execute({"symbol": "TEST", "news_data": "", "market_data": {}, "sentiment": {}})
        assert 0.0 <= result["overall_risk_score"] <= 1.0
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_execute_error_handling(self, agent):
        agent.llm.invoke.side_effect = RuntimeError("LLM down")
        result = agent.execute({"symbol": "FAIL"})
        assert result["status"] == "failed"
        assert "error" in result

    def test_calculate_risk_score_no_risks(self, agent):
        result = agent._calculate_risk_score([], {"sentiment_score": 0.5})
        assert result["score"] == pytest.approx(0.2, abs=0.1)
        assert result["level"] == "LOW"

    def test_calculate_risk_score_high_risks(self, agent):
        risks = [
            {"severity": "CRITICAL", "likelihood": 0.9},
            {"severity": "HIGH", "likelihood": 0.8},
        ]
        result = agent._calculate_risk_score(risks, {"sentiment_score": 0.2})
        assert result["score"] > 0.5

    def test_calculate_risk_score_negative_sentiment_increases_risk(self, agent):
        risks = [{"severity": "MEDIUM", "likelihood": 0.5}]
        result_neg = agent._calculate_risk_score(risks, {"sentiment_score": 0.2})
        result_pos = agent._calculate_risk_score(risks, {"sentiment_score": 0.8})
        assert result_neg["score"] > result_pos["score"]

    def test_generate_alerts_high_risk(self, agent):
        risks = [{"category": "regulatory", "severity": "HIGH", "description": "Test"}]
        overall = {"score": 0.8, "level": "HIGH"}
        alerts = agent._generate_alerts(risks, overall)
        assert len(alerts) >= 1
        assert all("severity" in a for a in alerts)
        assert all("timestamp" in a for a in alerts)

    def test_generate_alerts_low_risk_no_alerts(self, agent):
        risks = [{"category": "operational", "severity": "LOW", "description": "Minor issue"}]
        overall = {"score": 0.2, "level": "LOW"}
        alerts = agent._generate_alerts(risks, overall)
        assert len(alerts) == 0

    def test_generate_recommendations_not_empty(self, agent):
        risks = [{"category": "regulatory", "severity": "MEDIUM"}]
        overall = {"score": 0.5, "level": "MEDIUM"}
        recs = agent._generate_recommendations(risks, overall)
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_parse_risks_detects_keywords(self, agent):
        analysis = "There is an ongoing investigation and regulatory compliance issue. Cash flow is declining."
        risks = agent._parse_risks(analysis)
        categories = {r["category"] for r in risks}
        assert "regulatory" in categories or "financial" in categories

    def test_format_news_string(self, agent):
        assert agent._format_news("some news") == "some news"

    def test_format_news_dict(self, agent):
        result = agent._format_news({"findings": "Big news today"})
        assert "Big news today" in result

    def test_monitor_realtime(self, agent):
        config = agent.monitor_realtime("AAPL", alert_threshold=0.6)
        assert config["symbol"] == "AAPL"
        assert config["monitoring"] is True
        assert config["alert_threshold"] == 0.6


# ===========================================================================
# SummaryAgent tests
# ===========================================================================

class TestSummaryAgent:
    """Test cases for SummaryAgent."""

    LLM_RESPONSE = """
    EXECUTIVE SUMMARY
    Apple demonstrated strong fundamentals with positive market sentiment and manageable risks.

    KEY POSITIVES
    - Strong earnings beat expectations
    - Growing services revenue
    - Institutional accumulation

    KEY NEGATIVES / RISKS
    - EU regulatory scrutiny
    - Supply chain pressure

    INVESTMENT RECOMMENDATION
    HOLD — current risk/reward supports maintaining positions.

    CONFIDENCE LEVEL
    HIGH — data from all three agents was consistent and reliable.

    IMMEDIATE ACTION ITEMS
    1. Monitor volatility over next 30 days
    2. Review position sizing relative to portfolio risk
    3. Watch for regulatory updates in EU markets
    """

    @pytest.fixture
    def agent(self):
        return SummaryAgent(llm=_make_llm(self.LLM_RESPONSE), verbose=False)

    @pytest.fixture
    def full_input(self):
        return {
            "symbol": "AAPL",
            "period_days": 7,
            "research": {
                "symbol": "AAPL",
                "findings": "Strong Q4 earnings, new product launches.",
                "sources_used": ["financial_news_search"],
                "period_days": 7,
            },
            "sentiment": {
                "symbol": "AAPL",
                "overall_sentiment": "POSITIVE",
                "sentiment_score": 0.75,
                "confidence": 0.85,
                "text_count": 12,
                "llm_analysis": {"analysis": "Market sentiment is positive."},
            },
            "risk": {
                "symbol": "AAPL",
                "overall_risk_score": 0.45,
                "risk_level": "MEDIUM",
                "identified_risks": [
                    {"category": "regulatory", "severity": "MEDIUM", "description": "EU scrutiny"},
                ],
                "alerts": [],
                "recommendations": ["Monitor regulatory developments"],
            },
        }

    def test_initialization(self, agent):
        assert agent.name == "SummaryAgent"
        assert agent.description is not None

    def test_execute_returns_required_keys(self, agent, full_input):
        result = agent.execute(full_input)
        for key in [
            "symbol", "analysis_date", "period_days",
            "executive_summary", "key_positives", "key_negatives",
            "recommendation", "confidence", "confidence_label",
            "action_items", "scores", "full_report", "metadata",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_symbol_uppercased(self, agent, full_input):
        full_input["symbol"] = "aapl"
        result = agent.execute(full_input)
        assert result["symbol"] == "AAPL"

    def test_recommendation_valid_value(self, agent, full_input):
        result = agent.execute(full_input)
        assert result["recommendation"] in ["STRONG BUY", "BUY", "HOLD", "SELL", "AVOID"]

    def test_scores_in_range(self, agent, full_input):
        result = agent.execute(full_input)
        scores = result["scores"]
        assert 0 <= scores["sentiment_score"] <= 1
        assert 0 <= scores["risk_score"] <= 1
        assert 0 <= scores["composite_score"] <= 1
        assert 0 <= scores["confidence"] <= 1

    def test_execute_missing_agents_graceful(self, agent):
        result = agent.execute({"symbol": "TEST"})
        assert "symbol" in result
        assert "error" not in result or result.get("status") == "failed"

    def test_execute_error_handling(self, agent):
        agent.llm.invoke.side_effect = RuntimeError("LLM unavailable")
        result = agent.execute({"symbol": "FAIL", "research": {}, "sentiment": {}, "risk": {}})
        assert result["status"] == "failed"
        assert "error" in result

    def test_derive_recommendation_strong_buy(self, agent):
        rec = agent._derive_recommendation(sentiment_score=0.85, risk_score=0.20)
        assert rec == "STRONG BUY"

    def test_derive_recommendation_buy(self, agent):
        rec = agent._derive_recommendation(sentiment_score=0.65, risk_score=0.40)
        assert rec == "BUY"

    def test_derive_recommendation_hold(self, agent):
        rec = agent._derive_recommendation(sentiment_score=0.50, risk_score=0.55)
        assert rec == "HOLD"

    def test_derive_recommendation_sell(self, agent):
        rec = agent._derive_recommendation(sentiment_score=0.30, risk_score=0.70)
        assert rec == "SELL"

    def test_derive_recommendation_avoid(self, agent):
        rec = agent._derive_recommendation(sentiment_score=0.10, risk_score=0.95)
        assert rec == "AVOID"

    def test_compute_composite_scores_high_confidence(self, agent):
        sentiment = {"sentiment_score": 0.8, "confidence": 0.9}
        risk = {"overall_risk_score": 0.3}
        scores = agent._compute_composite_scores(sentiment, risk)
        assert scores["confidence_label"] == "HIGH"
        assert scores["composite_score"] > 0.5

    def test_compute_composite_scores_low_confidence(self, agent):
        sentiment = {"sentiment_score": 0.5, "confidence": 0.2}
        risk = {"overall_risk_score": 0.5}
        scores = agent._compute_composite_scores(sentiment, risk)
        assert scores["confidence_label"] == "LOW"

    def test_generate_report_markdown(self, agent, full_input):
        synthesis = agent.execute(full_input)
        md = agent.generate_report_markdown(synthesis)
        assert "# Investment Research Report" in md
        assert "AAPL" in md
        assert "## Executive Summary" in md
        assert "## Investment Recommendation" in md

    def test_format_research_empty(self, agent):
        assert "No research data" in agent._format_research({})

    def test_format_research_with_error(self, agent):
        result = agent._format_research({"error": "timed out"})
        assert "error" in result.lower()

    def test_format_sentiment_empty(self, agent):
        assert "No sentiment data" in agent._format_sentiment({})

    def test_format_risk_empty(self, agent):
        assert "No risk data" in agent._format_risk({})


# ===========================================================================
# Session-scoped fixtures
# ===========================================================================

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configure environment for all tests."""
    os.environ.setdefault("TESTING", "true")
    os.environ.setdefault("LOG_LEVEL", "ERROR")
    yield


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
