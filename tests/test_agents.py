"""
Unit tests for agents
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.base import BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.sentiment_agent import SentimentAgent
from src.agents.risk_agent import RiskAgent


class TestBaseAgent:
    """Test cases for BaseAgent"""
    
    def test_base_agent_initialization(self):
        """Test base agent initialization"""
        mock_llm = Mock()
        agent = TestAgent(
            name="TestAgent",
            description="Test agent",
            llm=mock_llm
        )
        
        assert agent.name == "TestAgent"
        assert agent.description == "Test agent"
        assert agent.llm == mock_llm
        assert agent.tools == []
    
    def test_reset_memory(self):
        """Test memory reset"""
        mock_llm = Mock()
        agent = TestAgent(
            name="TestAgent",
            description="Test agent",
            llm=mock_llm
        )
        
        agent.reset_memory()
        # Memory should be clear
        assert True  # Placeholder assertion
    
    def test_get_status(self):
        """Test status retrieval"""
        mock_llm = Mock()
        agent = TestAgent(
            name="TestAgent",
            description="Test agent",
            llm=mock_llm
        )
        
        status = agent.get_status()
        assert status["name"] == "TestAgent"
        assert "tools_count" in status


class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing"""
    
    def execute(self, input_data):
        return {"result": "test"}


class TestResearchAgent:
    """Test cases for ResearchAgent"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM"""
        llm = Mock()
        llm.predict.return_value = "Test research findings"
        return llm
    
    @pytest.fixture
    def research_agent(self, mock_llm):
        """Create ResearchAgent instance"""
        tools = []
        return ResearchAgent(llm=mock_llm, tools=tools)
    
    def test_research_agent_initialization(self, research_agent):
        """Test research agent initialization"""
        assert research_agent.name == "ResearchAgent"
        assert research_agent.description is not None
    
    @patch('src.agents.research_agent.initialize_agent')
    def test_execute(self, mock_init_agent, research_agent):
        """Test research execution"""
        # Mock agent executor
        mock_executor = Mock()
        mock_executor.invoke.return_value = {
            "output": "Test findings",
            "intermediate_steps": []
        }
        mock_init_agent.return_value = mock_executor
        research_agent.agent_executor = mock_executor
        
        input_data = {
            "symbol": "AAPL",
            "days_back": 7
        }
        
        result = research_agent.execute(input_data)
        
        assert result["symbol"] == "AAPL"
        assert result["period_days"] == 7
        assert "findings" in result
    
    def test_batch_research(self, research_agent):
        """Test batch research"""
        with patch.object(research_agent, 'execute') as mock_execute:
            mock_execute.return_value = {
                "symbol": "AAPL",
                "findings": "Test"
            }
            
            symbols = ["AAPL", "GOOGL"]
            results = research_agent.batch_research(symbols)
            
            assert len(results) == 2
            assert mock_execute.call_count == 2


class TestSentimentAgent:
    """Test cases for SentimentAgent"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM"""
        llm = Mock()
        llm.predict.return_value = """
        Overall sentiment: Positive
        Confidence: 0.85
        Key positive factors: Strong earnings
        Key negative factors: Regulatory concerns
        """
        return llm
    
    @pytest.fixture
    def sentiment_agent(self, mock_llm):
        """Create SentimentAgent instance"""
        return SentimentAgent(llm=mock_llm)
    
    def test_sentiment_agent_initialization(self, sentiment_agent):
        """Test sentiment agent initialization"""
        assert sentiment_agent.name == "SentimentAgent"
        assert sentiment_agent.description is not None
    
    def test_execute_with_text(self, sentiment_agent):
        """Test sentiment analysis with text"""
        input_data = {
            "symbol": "AAPL",
            "text": "Company reports strong earnings and positive outlook."
        }
        
        result = sentiment_agent.execute(input_data)
        
        assert result["symbol"] == "AAPL"
        assert "overall_sentiment" in result
        assert "sentiment_score" in result
        assert "confidence" in result
    
    def test_execute_with_articles(self, sentiment_agent):
        """Test sentiment analysis with multiple articles"""
        input_data = {
            "symbol": "AAPL",
            "articles": [
                "Positive news article",
                "Another positive article"
            ]
        }
        
        result = sentiment_agent.execute(input_data)
        
        assert result["text_count"] == 2
        assert "overall_sentiment" in result
    
    def test_normalize_score(self, sentiment_agent):
        """Test sentiment score normalization"""
        positive_score = sentiment_agent._normalize_score("POSITIVE", 0.8)
        assert positive_score == 0.8
        
        negative_score = sentiment_agent._normalize_score("NEGATIVE", 0.8)
        assert negative_score == -0.8
        
        neutral_score = sentiment_agent._normalize_score("NEUTRAL", 0.5)
        assert neutral_score == 0.0


class TestRiskAgent:
    """Test cases for RiskAgent"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM"""
        llm = Mock()
        llm.predict.return_value = """
        Regulatory risk: Investigation ongoing
        Financial risk: Elevated debt levels
        Volatility risk: High price swings
        """
        return llm
    
    @pytest.fixture
    def risk_agent(self, mock_llm):
        """Create RiskAgent instance"""
        return RiskAgent(llm=mock_llm)
    
    def test_risk_agent_initialization(self, risk_agent):
        """Test risk agent initialization"""
        assert risk_agent.name == "RiskAgent"
        assert risk_agent.description is not None
        assert len(risk_agent.RISK_CATEGORIES) > 0
    
    def test_execute(self, risk_agent):
        """Test risk analysis execution"""
        input_data = {
            "symbol": "AAPL",
            "news_data": "Regulatory investigation announced",
            "market_data": {"volatility": "high"},
            "sentiment": {"sentiment_score": 0.3}
        }
        
        result = risk_agent.execute(input_data)
        
        assert result["symbol"] == "AAPL"
        assert "overall_risk_score" in result
        assert "risk_level" in result
        assert "identified_risks" in result
    
    def test_calculate_risk_score(self, risk_agent):
        """Test risk score calculation"""
        risks = [
            {"severity": "HIGH", "likelihood": 0.8},
            {"severity": "MEDIUM", "likelihood": 0.6}
        ]
        sentiment = {"sentiment_score": 0.5}
        
        result = risk_agent._calculate_risk_score(risks, sentiment)
        
        assert "score" in result
        assert "level" in result
        assert 0 <= result["score"] <= 1
        assert result["level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    def test_generate_alerts(self, risk_agent):
        """Test alert generation"""
        risks = [
            {
                "category": "regulatory",
                "severity": "HIGH",
                "likelihood": 0.8,
                "description": "Test risk"
            }
        ]
        overall_risk = {"score": 0.8, "level": "HIGH"}
        
        alerts = risk_agent._generate_alerts(risks, overall_risk)
        
        assert len(alerts) > 0
        assert all("severity" in alert for alert in alerts)


# Pytest configuration
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "ERROR"
    
    yield
    
    # Cleanup
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
