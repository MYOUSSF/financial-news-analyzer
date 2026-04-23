"""
News Tool - Fetches financial news from NewsAPI and other sources.
"""
import os
from typing import Any, List, Optional
from datetime import datetime, timedelta
from loguru import logger

from langchain_core.tools import BaseTool
from pydantic import Field

try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None


class NewsTool(BaseTool):
    """Tool for fetching financial news from NewsAPI."""
    
    name: str = "financial_news_search"
    description: str = """
    Useful for searching financial news articles about companies, stocks, or market events.
    Input should be a company name, stock symbol, or financial topic.
    Returns recent news articles with titles, descriptions, and sources.
    """
    
    api_key: Optional[str] = Field(default=None)
    max_results: int = Field(default=10)
    client: Optional[Any] = Field(default=None, exclude=True)
    
    def model_post_init(self, __context):
        """Initialize the NewsAPI client after model initialization."""
        api_key = self.api_key or os.getenv("NEWSAPI_KEY")
        
        if not api_key:
            logger.warning("NewsAPI key not found. Tool will return mock data.")
            self.client = None
        elif NewsApiClient:
            self.client = NewsApiClient(api_key=api_key)
        else:
            logger.warning("newsapi-python not installed. Install with: pip install newsapi-python")
            self.client = None
    
    def _run(self, query: str) -> str:
        """
        Fetch news articles for a given query.
        
        Args:
            query: Search query (company name, symbol, or topic)
        
        Returns:
            Formatted string with news articles
        """
        try:
            if not self.client:
                return self._get_mock_news(query)
            
            # Calculate date range (last 7 days)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)
            
            # Search for news
            response = self.client.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=self.max_results
            )
            
            if response['status'] != 'ok':
                return f"Error fetching news: {response.get('message', 'Unknown error')}"
            
            articles = response.get('articles', [])
            
            if not articles:
                return f"No recent news found for: {query}"
            
            # Format articles
            return self._format_articles(articles, query)
            
        except Exception as e:
            logger.error(f"Error in NewsTool: {str(e)}")
            return f"Error fetching news: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of _run."""
        return self._run(query)
    
    def _format_articles(self, articles: List[dict], query: str) -> str:
        """
        Format articles into a readable string.
        
        Args:
            articles: List of article dictionaries
            query: Original search query
        
        Returns:
            Formatted string
        """
        output = [f"Recent news for '{query}' ({len(articles)} articles):\n"]
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', 'No title')
            description = article.get('description', 'No description')
            source = article.get('source', {}).get('name', 'Unknown')
            published = article.get('publishedAt', '')
            url = article.get('url', '')
            
            output.append(f"\n{i}. {title}")
            output.append(f"   Source: {source}")
            output.append(f"   Published: {published}")
            output.append(f"   Summary: {description}")
            output.append(f"   URL: {url}")
        
        return "\n".join(output)
    
    def _get_mock_news(self, query: str) -> str:
        """
        Return mock news data for testing.
        
        Args:
            query: Search query
        
        Returns:
            Mock news data
        """
        return f"""
Recent news for '{query}' (3 articles):

1. {query} Reports Strong Q4 Earnings
   Source: Financial Times
   Published: 2024-03-30T10:00:00Z
   Summary: Company beats analyst expectations with 15% revenue growth year-over-year.
   URL: https://example.com/article1

2. Analysts Upgrade {query} Stock Rating
   Source: Bloomberg
   Published: 2024-03-29T14:30:00Z
   Summary: Major investment banks raise price targets following positive earnings call.
   URL: https://example.com/article2

3. {query} Announces New Product Launch
   Source: Reuters
   Published: 2024-03-28T09:15:00Z
   Summary: Company unveils innovative product line expected to drive future growth.
   URL: https://example.com/article3
"""


class FinancialNewsTool(BaseTool):
    """Extended news tool with financial-specific features."""
    
    name: str = "financial_news_analysis"
    description: str = """
    Advanced financial news analysis tool. Searches and analyzes news with 
    financial context including earnings, M&A, regulatory issues, and market sentiment.
    Input should be a stock symbol or company name.
    """
    
    news_tool: NewsTool = Field(default_factory=NewsTool)
    
    def _run(self, query: str) -> str:
        """
        Fetch and analyze financial news.
        
        Args:
            query: Stock symbol or company name
        
        Returns:
            Analyzed news with financial context
        """
        try:
            # Get raw news
            news = self.news_tool._run(query)
            
            # Add financial context analysis
            analysis = self._add_financial_context(news, query)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in FinancialNewsTool: {str(e)}")
            return f"Error analyzing financial news: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of _run."""
        return self._run(query)
    
    def _add_financial_context(self, news: str, query: str) -> str:
        """
        Add financial context to news.
        
        Args:
            news: Raw news data
            query: Search query
        
        Returns:
            News with financial context
        """
        # Analyze news for key financial events
        events = {
            "earnings": ["earnings", "revenue", "profit", "EPS"],
            "mergers": ["acquisition", "merger", "buyout", "takeover"],
            "regulatory": ["lawsuit", "investigation", "SEC", "fine", "regulation"],
            "leadership": ["CEO", "CFO", "resignation", "appointed", "executive"],
            "products": ["launch", "product", "release", "unveil"]
        }
        
        detected_events = []
        news_lower = news.lower()
        
        for event_type, keywords in events.items():
            if any(keyword.lower() in news_lower for keyword in keywords):
                detected_events.append(event_type)
        
        context = f"\n\n=== Financial Context Analysis for {query} ===\n"
        context += f"Detected Event Types: {', '.join(detected_events) if detected_events else 'None'}\n"
        
        if "earnings" in detected_events:
            context += "📊 Earnings-related news detected - Monitor for guidance and analyst reactions\n"
        if "mergers" in detected_events:
            context += "🤝 M&A activity detected - Assess strategic implications\n"
        if "regulatory" in detected_events:
            context += "⚠️  Regulatory issues detected - Monitor for financial impact\n"
        if "leadership" in detected_events:
            context += "👔 Leadership changes detected - Assess continuity and strategy\n"
        if "products" in detected_events:
            context += "🚀 Product news detected - Evaluate market impact potential\n"
        
        return news + context


def create_news_tools() -> List[BaseTool]:
    """
    Create and return all news-related tools.
    
    Returns:
        List of news tools
    """
    return [
        NewsTool(),
        FinancialNewsTool()
    ]
