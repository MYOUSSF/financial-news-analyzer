"""
Research Agent - Retrieves and analyzes financial news from multiple sources.
"""
from typing import Any, Dict, List
from datetime import datetime, timedelta
from loguru import logger

from langchain.agents import AgentType, initialize_agent
from langchain.prompts import PromptTemplate

from .base import BaseAgent


class ResearchAgent(BaseAgent):
    """
    Agent responsible for researching financial news and market events.
    
    This agent queries multiple data sources to gather relevant information
    about stocks, sectors, and market events.
    """
    
    def __init__(self, llm: Any, tools: List[Any], verbose: bool = False):
        """
        Initialize the Research Agent.
        
        Args:
            llm: Language model to use
            tools: List of tools (news_tool, stock_tool, etc.)
            verbose: Enable verbose logging
        """
        super().__init__(
            name="ResearchAgent",
            description="Retrieves and analyzes financial news and market data",
            llm=llm,
            tools=tools,
            verbose=verbose
        )
        
        # Initialize the agent executor
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        # Research prompt template
        self.research_prompt = PromptTemplate(
            input_variables=["symbol", "days_back", "focus_areas"],
            template="""
            You are a financial research analyst. Your task is to gather comprehensive 
            information about {symbol} from the past {days_back} days.
            
            Focus on:
            {focus_areas}
            
            Use the available tools to:
            1. Get recent news articles about the company
            2. Retrieve stock price data and trading volume
            3. Identify major events or announcements
            4. Gather information about market sentiment
            
            Provide a structured summary of your findings, including:
            - Key news events
            - Stock performance metrics
            - Notable market reactions
            - Any red flags or concerns
            
            Be thorough but concise.
            """
        )
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research on a given symbol or topic.
        
        Args:
            input_data: Dictionary containing:
                - symbol: Stock symbol to research
                - days_back: Number of days to look back (default: 7)
                - focus_areas: Specific areas to focus on (optional)
        
        Returns:
            Dictionary containing research findings
        """
        try:
            symbol = input_data.get("symbol", "").upper()
            days_back = input_data.get("days_back", 7)
            focus_areas = input_data.get("focus_areas", 
                "earnings, product launches, regulatory issues, market sentiment")
            
            logger.info(f"Starting research for {symbol} ({days_back} days)")
            
            # Format the research query
            query = self.research_prompt.format(
                symbol=symbol,
                days_back=days_back,
                focus_areas=focus_areas
            )
            
            # Execute the agent
            result = self.agent_executor.invoke({"input": query})
            
            # Parse and structure the output
            output = {
                "symbol": symbol,
                "research_date": datetime.now().isoformat(),
                "period_days": days_back,
                "findings": result.get("output", ""),
                "sources_used": self._extract_sources(result),
                "metadata": {
                    "agent": self.name,
                    "execution_time": result.get("execution_time", 0)
                }
            }
            
            self._log_execution(input_data, output)
            
            logger.info(f"Completed research for {symbol}")
            return output
            
        except Exception as e:
            logger.error(f"Error in ResearchAgent execution: {str(e)}")
            return {
                "symbol": input_data.get("symbol", ""),
                "error": str(e),
                "status": "failed"
            }
    
    def batch_research(self, symbols: List[str], days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Perform research on multiple symbols.
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back
        
        Returns:
            List of research results for each symbol
        """
        results = []
        for symbol in symbols:
            result = self.execute({
                "symbol": symbol,
                "days_back": days_back
            })
            results.append(result)
        
        logger.info(f"Completed batch research for {len(symbols)} symbols")
        return results
    
    def _extract_sources(self, result: Dict[str, Any]) -> List[str]:
        """Extract data sources used during research."""
        sources = []
        
        # Check intermediate steps for tool usage
        if "intermediate_steps" in result:
            for step in result["intermediate_steps"]:
                if len(step) > 0:
                    tool_name = getattr(step[0], "tool", "unknown")
                    if tool_name not in sources:
                        sources.append(tool_name)
        
        return sources
    
    def focused_research(
        self, 
        symbol: str, 
        topic: str, 
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Conduct focused research on a specific topic for a symbol.
        
        Args:
            symbol: Stock symbol
            topic: Specific topic to research (e.g., "earnings", "acquisitions")
            days_back: Number of days to look back
        
        Returns:
            Research findings focused on the specified topic
        """
        return self.execute({
            "symbol": symbol,
            "days_back": days_back,
            "focus_areas": topic
        })
