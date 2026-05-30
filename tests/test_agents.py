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


from src.agents.base import AgentExecutionError, BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.risk_agent import RiskAgent
from src.agents.summary_agent import SummaryAgent, SynthesisOutput


# ===========================================================================
# Helpers / Fixtures shared across test classes
# ===========================================================================

def _make_llm(response_text: str = "Mock LLM response") -> Mock:
    """Build a mock LangChain chat model that returns a predictable response."""
    llm = Mock()
    response = Mock()
    response.content = response_text
    # Explicitly set tool_calls to an empty list so isinstance checks pass correctly
    response.tool_calls = []
    # Support both .invoke() (chat models) and .predict() (older LLMChain style)
    llm.invoke.return_value = response
    llm.predict.return_value = response_text
    # bind_tools returns a clone that also has invoke()
    llm.bind_tools.return_value = llm
    return llm


def _make_tool(name: str, run_result: str = "tool result") -> Mock:
    """Build a mock LangChain tool with a predictable _run() return value."""
    tool = Mock()
    tool.name = name
    tool._run.return_value = run_result
    return tool


def _make_synthesis_output(**overrides) -> SynthesisOutput:
    """Build a SynthesisOutput instance with sensible defaults for tests."""
    defaults = {
        "executive_summary": "Apple demonstrated strong fundamentals with positive market sentiment.",
        "key_positives": ["Strong earnings beat expectations", "Growing services revenue"],
        "key_negatives": ["EU regulatory scrutiny", "Supply chain pressure"],
        "recommendation": "HOLD",
        "recommendation_rationale": "Current risk/reward supports maintaining positions.",
        "confidence_label": "HIGH",
        "action_items": ["Monitor volatility over next 30 days", "Review position sizing"],
    }
    defaults.update(overrides)
    return SynthesisOutput(**defaults)


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
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute({"symbol": "FAIL"})
        assert exc_info.value.agent_name == "ResearchAgent"
        assert isinstance(exc_info.value.original_error, RuntimeError)

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

    def test_execute_tool_loop_executes_tools(self):
        """Tool calls returned by the LLM are executed and results fed back."""
        mock_tool = _make_tool("financial_news_search", "AAPL Q4 earnings beat expectations.")

        first_response = Mock()
        first_response.content = ""
        first_response.tool_calls = [
            {"name": "financial_news_search", "args": {"query": "AAPL news"}, "id": "call_1"}
        ]

        second_response = Mock()
        second_response.content = "Research complete based on tool output."
        second_response.tool_calls = []

        mock_llm = _make_llm()
        mock_llm.invoke.side_effect = [first_response, second_response]

        agent = ResearchAgent(llm=mock_llm, tools=[mock_tool], verbose=False)
        result = agent.execute({"symbol": "AAPL", "days_back": 7})

        mock_tool._run.assert_called_once()
        assert mock_llm.invoke.call_count == 2
        assert "financial_news_search" in result["sources_used"]
        assert result["symbol"] == "AAPL"
        assert result["findings"] == "Research complete based on tool output."

    def test_execute_tool_loop_max_iterations(self):
        """Loop stops after max_iterations even if the LLM keeps returning tool_calls."""
        mock_tool = _make_tool("stock_data", "price: 150")

        always_tool_response = Mock()
        always_tool_response.content = "Still calling tools..."
        always_tool_response.tool_calls = [
            {"name": "stock_data", "args": {"query": "AAPL"}, "id": "call_x"}
        ]

        mock_llm = _make_llm()
        mock_llm.invoke.return_value = always_tool_response

        agent = ResearchAgent(llm=mock_llm, tools=[mock_tool], verbose=False)
        result = agent.execute({"symbol": "AAPL", "days_back": 7})

        assert mock_llm.invoke.call_count <= agent.max_iterations
        assert "symbol" in result

    def test_execute_sources_collected_across_iterations(self):
        """Sources from multiple tool calls across iterations are all preserved."""
        tool_a = _make_tool("news_search", "news result")
        tool_b = _make_tool("stock_data", "stock result")

        first_response = Mock()
        first_response.content = ""
        first_response.tool_calls = [
            {"name": "news_search", "args": {"query": "AAPL"}, "id": "c1"}
        ]

        second_response = Mock()
        second_response.content = ""
        second_response.tool_calls = [
            {"name": "stock_data", "args": {"query": "AAPL"}, "id": "c2"}
        ]

        third_response = Mock()
        third_response.content = "Final analysis."
        third_response.tool_calls = []

        mock_llm = _make_llm()
        mock_llm.invoke.side_effect = [first_response, second_response, third_response]

        agent = ResearchAgent(llm=mock_llm, tools=[tool_a, tool_b], verbose=False)
        result = agent.execute({"symbol": "AAPL", "days_back": 7})

        assert "news_search" in result["sources_used"]
        assert "stock_data" in result["sources_used"]
        assert mock_llm.invoke.call_count == 3

    def test_execute_unknown_tool_handled_gracefully(self):
        """If the LLM requests an unknown tool, the loop continues without crashing."""
        first_response = Mock()
        first_response.content = ""
        first_response.tool_calls = [
            {"name": "nonexistent_tool", "args": {"query": "X"}, "id": "c1"}
        ]

        second_response = Mock()
        second_response.content = "Done."
        second_response.tool_calls = []

        mock_llm = _make_llm()
        mock_llm.invoke.side_effect = [first_response, second_response]

        agent = ResearchAgent(llm=mock_llm, tools=[], verbose=False)
        result = agent.execute({"symbol": "TEST", "days_back": 7})

        assert result["symbol"] == "TEST"
        assert "error" not in result


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
        with patch.object(agent, "_combine_sentiments", side_effect=RuntimeError("combine error")):
            with pytest.raises(AgentExecutionError) as exc_info:
                agent.execute({"symbol": "ERR", "text": "some text"})
        assert exc_info.value.agent_name == "SentimentAgent"
        assert isinstance(exc_info.value.original_error, RuntimeError)

    def test_normalize_score_positive(self, agent):
        assert agent._normalize_score("POSITIVE", 0.9) == pytest.approx(0.9)

    def test_normalize_score_negative(self, agent):
        assert agent._normalize_score("NEGATIVE", 0.8) == pytest.approx(-0.8)

    def test_normalize_score_neutral(self, agent):
        assert agent._normalize_score("NEUTRAL", 0.5) == pytest.approx(0.0)

    # FinBERT returns lowercase labels — verify case-insensitive handling
    def test_normalize_score_finbert_positive(self, agent):
        assert agent._normalize_score("positive", 0.9) == pytest.approx(0.9)

    def test_normalize_score_finbert_negative(self, agent):
        assert agent._normalize_score("negative", 0.8) == pytest.approx(-0.8)

    def test_normalize_score_finbert_neutral(self, agent):
        assert agent._normalize_score("neutral", 0.6) == pytest.approx(0.0)

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
        with patch("src.agents.sentiment_agent.get_sentiment_pipeline", return_value=None):
            result = agent._analyze_with_ml(["test text"])
        assert result == []

    def test_execute_deduplicates_identical_articles(self, agent):
        articles = [
            "Apple Q4 earnings beat expectations.",
            "Apple Q4 earnings beat expectations.",  # exact duplicate
        ]
        result = agent.execute({"symbol": "AAPL", "articles": articles})
        assert result["text_count"] == 1
        assert result["dedup_count"] == 1

    def test_execute_deduplicates_by_url_when_metadata_provided(self, agent):
        articles = ["Article body A.", "Article body B (different text, same URL)."]
        metadata = [{"url": "https://example.com/story"}, {"url": "https://example.com/story"}]
        result = agent.execute({"symbol": "AAPL", "articles": articles, "metadata": metadata})
        assert result["text_count"] == 1
        assert result["dedup_count"] == 1

    def test_execute_no_duplicates_preserves_all(self, agent):
        articles = ["First unique article.", "Second unique article."]
        result = agent.execute({"symbol": "AAPL", "articles": articles})
        assert result["text_count"] == 2
        assert result["dedup_count"] == 0


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
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute({"symbol": "FAIL"})
        assert exc_info.value.agent_name == "RiskAgent"
        assert isinstance(exc_info.value.original_error, RuntimeError)

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
        # Symbol appears right next to the keywords → risks are directly scoped
        analysis = "AAPL faces an ongoing investigation and compliance issue. Cash flow is declining."
        risks = agent._parse_risks(analysis, "AAPL")
        categories = {r["category"] for r in risks}
        assert "regulatory" in categories or "financial" in categories
        # All detected risks should be directly attributed to AAPL
        for r in risks:
            assert "scoped" in r

    def test_parse_risks_direct_mention_is_scoped(self, agent):
        analysis = "AAPL is under investigation by the SEC for accounting irregularities."
        risks = agent._parse_risks(analysis, "AAPL")
        regulatory = [r for r in risks if r["category"] == "regulatory"]
        assert len(regulatory) == 1
        assert regulatory[0]["scoped"] is True
        assert regulatory[0]["likelihood"] == 0.6

    def test_parse_risks_competitor_mention_has_reduced_likelihood(self, agent):
        # "investigation" is about a named competitor; AAPL is mentioned far away
        # (>150 chars from the keyword) so the risk should be indirect/unscoped.
        analysis = (
            "Apple (AAPL) reported strong Q4 results with record iPhone sales and "
            "robust services growth, showing resilience in a challenging macro environment. "
            "Meanwhile, competitor MegaCorp is under investigation by the FTC for antitrust "
            "violations related to its acquisition practices."
        )
        risks = agent._parse_risks(analysis, "AAPL")
        regulatory = [r for r in risks if r["category"] == "regulatory"]
        assert len(regulatory) == 1
        assert regulatory[0]["scoped"] is False
        assert regulatory[0]["likelihood"] == 0.3

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

    @pytest.fixture
    def agent(self):
        llm = _make_llm()
        llm.with_structured_output.return_value.invoke.return_value = _make_synthesis_output()
        return SummaryAgent(llm=llm, verbose=False)

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
        agent.llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("LLM unavailable")
        with pytest.raises(AgentExecutionError) as exc_info:
            agent.execute({"symbol": "FAIL", "research": {}, "sentiment": {}, "risk": {}})
        assert exc_info.value.agent_name == "SummaryAgent"
        assert isinstance(exc_info.value.original_error, RuntimeError)

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
# FinancialAnalysisChain tests
# ===========================================================================

class TestFinancialAnalysisChain:
    """Tests for FinancialAnalysisChain error propagation and fail_fast behaviour.

    analyze_stock() delegates to analyze_stock_async() internally, so all mocks
    must be AsyncMock on aexecute() rather than plain Mock on execute().
    """

    @pytest.fixture
    def chain(self):
        """Return a chain whose agents are all replaced by controllable AsyncMocks."""
        from src.chains.analysis_chain import FinancialAnalysisChain
        from unittest.mock import AsyncMock

        mock_llm = _make_llm()
        with patch.object(FinancialAnalysisChain, "_build_tools", return_value=[]), \
             patch.object(FinancialAnalysisChain, "_init_agents"), \
             patch.object(FinancialAnalysisChain, "_init_vector_store", return_value=None):
            c = FinancialAnalysisChain(llm=mock_llm)

        c.research_agent = Mock()
        c.sentiment_agent = Mock()
        c.risk_agent = Mock()
        c.summary_agent = Mock()
        c.vector_store = None

        # Default success responses (AsyncMock — used by analyze_stock_async)
        c.research_agent.aexecute = AsyncMock(return_value={
            "symbol": "AAPL", "findings": "Strong Q4.", "sources_used": [], "period_days": 7,
        })
        c.sentiment_agent.aexecute = AsyncMock(return_value={
            "symbol": "AAPL", "overall_sentiment": "POSITIVE", "sentiment_score": 0.7,
            "confidence": 0.8, "text_count": 1, "llm_analysis": {"analysis": "good"},
        })
        c.risk_agent.aexecute = AsyncMock(return_value={
            "symbol": "AAPL", "overall_risk_score": 0.3, "risk_level": "LOW",
            "identified_risks": [], "alerts": [], "recommendations": [],
        })
        c.summary_agent.aexecute = AsyncMock(return_value={
            "symbol": "AAPL", "recommendation": "BUY", "executive_summary": "Looks good.",
            "key_positives": [], "key_negatives": [], "action_items": [],
            "scores": {"sentiment_score": 0.7, "risk_score": 0.3,
                       "composite_score": 0.7, "confidence": 0.8, "confidence_label": "HIGH"},
            "confidence": 0.8, "confidence_label": "HIGH",
            "analysis_date": "2024-01-01T00:00:00", "period_days": 7,
            "full_report": "", "metadata": {},
        })
        return c

    def _make_agent_error(self, agent_name: str) -> AgentExecutionError:
        return AgentExecutionError(
            agent_name=agent_name,
            original_error=RuntimeError("network error"),
            input_data={"symbol": "AAPL"},
        )

    # -- fail_fast=False (default) ------------------------------------------

    def test_research_failure_continues_pipeline(self, chain):
        """Research failure with fail_fast=False: downstream agents still run."""
        chain.research_agent.aexecute.side_effect = self._make_agent_error("ResearchAgent")
        result = chain.analyze_stock("AAPL")
        chain.sentiment_agent.aexecute.assert_called_once()
        chain.risk_agent.aexecute.assert_called_once()
        chain.summary_agent.aexecute.assert_called_once()
        assert result["recommendation"] == "BUY"

    def test_sentiment_failure_continues_pipeline(self, chain):
        """Sentiment failure with fail_fast=False: risk and summary still run."""
        chain.sentiment_agent.aexecute.side_effect = self._make_agent_error("SentimentAgent")
        result = chain.analyze_stock("AAPL")
        chain.risk_agent.aexecute.assert_called_once()
        chain.summary_agent.aexecute.assert_called_once()
        assert result["recommendation"] == "BUY"

    def test_risk_failure_continues_pipeline(self, chain):
        """Risk failure with fail_fast=False: summary still runs."""
        chain.risk_agent.aexecute.side_effect = self._make_agent_error("RiskAgent")
        result = chain.analyze_stock("AAPL")
        chain.summary_agent.aexecute.assert_called_once()
        assert result["recommendation"] == "BUY"

    def test_summary_failure_always_returns_error(self, chain):
        """Summary failure is always surfaced regardless of fail_fast."""
        chain.summary_agent.aexecute.side_effect = self._make_agent_error("SummaryAgent")
        result = chain.analyze_stock("AAPL", fail_fast=False)
        assert result["status"] == "failed"
        assert result["failed_stage"] == "summary"

    # -- fail_fast=True -------------------------------------------------------

    def test_research_failure_halts_on_fail_fast(self, chain):
        """Research failure with fail_fast=True: pipeline halts, returns error dict."""
        chain.research_agent.aexecute.side_effect = self._make_agent_error("ResearchAgent")
        result = chain.analyze_stock("AAPL", fail_fast=True)
        assert result["status"] == "failed"
        assert result["symbol"] == "AAPL"
        assert result["failed_stage"] == "research"
        assert "error" in result
        # Research fails before the parallel stage — sentiment and risk never start
        chain.sentiment_agent.aexecute.assert_not_called()
        chain.risk_agent.aexecute.assert_not_called()
        chain.summary_agent.aexecute.assert_not_called()

    def test_sentiment_failure_halts_on_fail_fast(self, chain):
        """Sentiment failure with fail_fast=True: pipeline returns error before summary."""
        chain.sentiment_agent.aexecute.side_effect = self._make_agent_error("SentimentAgent")
        result = chain.analyze_stock("AAPL", fail_fast=True)
        assert result["status"] == "failed"
        assert result["failed_stage"] == "sentiment"
        # Summary is never invoked when fail_fast=True and sentiment fails
        chain.summary_agent.aexecute.assert_not_called()
        # Risk runs in parallel with sentiment, so it may have been called

    def test_error_response_contains_agent_message(self, chain):
        """The error string in the top-level response names the failing agent."""
        chain.research_agent.aexecute.side_effect = self._make_agent_error("ResearchAgent")
        result = chain.analyze_stock("AAPL", fail_fast=True)
        assert "ResearchAgent" in result["error"]

    # -- AgentExecutionError contract -----------------------------------------

    def test_agent_execution_error_str(self):
        """AgentExecutionError str() includes agent name and original error type."""
        exc = AgentExecutionError(
            agent_name="ResearchAgent",
            original_error=ValueError("bad input"),
            input_data={"symbol": "X"},
        )
        assert "ResearchAgent" in str(exc)
        assert "ValueError" in str(exc)
        assert exc.agent_name == "ResearchAgent"
        assert isinstance(exc.original_error, ValueError)


# ===========================================================================
# FinancialAnalysisChain — async parallel execution tests
# ===========================================================================

class TestFinancialAnalysisChainAsync:
    """Verify that analyze_stock_async() dispatches Sentiment + Risk in parallel."""

    @pytest.fixture
    def async_chain(self):
        """Chain with AsyncMock aexecute() on every agent."""
        from src.chains.analysis_chain import FinancialAnalysisChain
        from unittest.mock import AsyncMock

        mock_llm = _make_llm()
        with patch.object(FinancialAnalysisChain, "_build_tools", return_value=[]), \
             patch.object(FinancialAnalysisChain, "_init_agents"), \
             patch.object(FinancialAnalysisChain, "_init_vector_store", return_value=None):
            c = FinancialAnalysisChain(llm=mock_llm)

        c.research_agent = Mock()
        c.sentiment_agent = Mock()
        c.risk_agent = Mock()
        c.summary_agent = Mock()
        c.vector_store = None

        c.research_agent.aexecute = AsyncMock(return_value={
            "symbol": "AAPL", "findings": "Strong Q4.", "sources_used": [], "period_days": 7,
        })
        c.sentiment_agent.aexecute = AsyncMock(return_value={
            "symbol": "AAPL", "overall_sentiment": "POSITIVE", "sentiment_score": 0.7,
            "confidence": 0.8, "text_count": 1, "llm_analysis": {"analysis": "good"},
        })
        c.risk_agent.aexecute = AsyncMock(return_value={
            "symbol": "AAPL", "overall_risk_score": 0.3, "risk_level": "LOW",
            "identified_risks": [], "alerts": [], "recommendations": [],
        })
        c.summary_agent.aexecute = AsyncMock(return_value={
            "symbol": "AAPL", "recommendation": "BUY", "executive_summary": "Looks good.",
            "key_positives": [], "key_negatives": [], "action_items": [],
            "scores": {
                "sentiment_score": 0.7, "risk_score": 0.3,
                "composite_score": 0.7, "confidence": 0.8, "confidence_label": "HIGH",
            },
            "confidence": 0.8, "confidence_label": "HIGH",
            "analysis_date": "2024-01-01T00:00:00", "period_days": 7,
            "full_report": "", "metadata": {},
        })
        return c

    async def test_sentiment_and_risk_dispatched_via_gather(self, async_chain):
        """asyncio.gather is called with exactly 2 coroutines for Sentiment + Risk."""
        import asyncio as _asyncio

        real_gather = _asyncio.gather
        gather_arg_counts: list = []

        async def spy_gather(*coros, **kw):
            gather_arg_counts.append(len(coros))
            return await real_gather(*coros, **kw)

        with patch("asyncio.gather", side_effect=spy_gather):
            result = await async_chain.analyze_stock_async("AAPL")

        # gather must have been called exactly once with the 2 parallel coroutines
        assert gather_arg_counts == [2], (
            "Expected asyncio.gather called once with 2 coroutines "
            f"(sentiment + risk); got {gather_arg_counts}"
        )
        assert result["recommendation"] == "BUY"

    async def test_research_completes_before_parallel_stages(self, async_chain):
        """ResearchAgent.aexecute finishes before SentimentAgent and RiskAgent start."""
        import asyncio as _asyncio
        call_log: list = []

        orig_research = async_chain.research_agent.aexecute
        orig_sentiment = async_chain.sentiment_agent.aexecute

        async def log_research(data):
            call_log.append("research_start")
            result = await orig_research(data)
            call_log.append("research_end")
            return result

        async def log_sentiment(data):
            call_log.append("sentiment_start")
            return await orig_sentiment(data)

        async_chain.research_agent.aexecute = log_research
        async_chain.sentiment_agent.aexecute = log_sentiment

        await async_chain.analyze_stock_async("AAPL")

        assert "research_end" in call_log
        assert "sentiment_start" in call_log
        assert call_log.index("research_end") < call_log.index("sentiment_start")

    async def test_elapsed_seconds_present_and_non_negative(self, async_chain):
        """Result dict includes _elapsed_seconds as a non-negative float."""
        result = await async_chain.analyze_stock_async("AAPL")
        assert "_elapsed_seconds" in result
        assert isinstance(result["_elapsed_seconds"], float)
        assert result["_elapsed_seconds"] >= 0.0

    def test_sync_wrapper_delegates_to_async(self, async_chain):
        """analyze_stock() calls analyze_stock_async() and returns its result."""
        from unittest.mock import AsyncMock

        expected = {
            "symbol": "AAPL", "recommendation": "HOLD", "_elapsed_seconds": 0.5,
        }
        with patch.object(
            async_chain, "analyze_stock_async", new_callable=AsyncMock, return_value=expected
        ) as mock_async:
            result = async_chain.analyze_stock("AAPL", days_back=3)

        mock_async.assert_called_once_with(
            symbol="AAPL",
            days_back=3,
            focus_areas="earnings, product launches, regulatory issues, market sentiment",
            include_sentiment=True,
            include_risk=True,
            context_from_vector_store=True,
            fail_fast=False,
        )
        assert result["recommendation"] == "HOLD"


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
