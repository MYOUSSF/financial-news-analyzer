"""
Research Agent - Retrieves and analyzes financial news from multiple sources.
"""
import json
import os
from typing import Any, Dict, List
from datetime import datetime
from loguru import logger

import yaml
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import PromptTemplate

from .base import AgentExecutionError, BaseAgent


def _load_max_iterations() -> int:
    """Read max_iterations from agents_config.yaml; fall back to 5."""
    try:
        config_path = os.path.join(
            os.path.dirname(__file__), "../../config/agents_config.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("research_agent", {}).get("max_iterations", 5)
    except Exception:
        return 5


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
            verbose=verbose,
        )

        self.max_iterations = _load_max_iterations()

        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

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
            """,
        )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute research on a given symbol or topic.

        Runs an agentic tool execution loop: invokes the LLM, executes any
        tool calls it requests, feeds results back, and repeats until the LLM
        produces a final answer or max_iterations is reached.

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
            focus_areas = input_data.get(
                "focus_areas",
                "earnings, product launches, regulatory issues, market sentiment",
            )

            logger.info(f"Starting research for {symbol} ({days_back} days)")

            query = self.research_prompt.format(
                symbol=symbol,
                days_back=days_back,
                focus_areas=focus_areas,
            )

            messages: List[Any] = [HumanMessage(content=query)]
            all_sources: List[str] = []
            final_result = None

            for iteration in range(self.max_iterations):
                result = self.llm_with_tools.invoke(messages)
                messages.append(result)
                final_result = result

                # Only treat a real non-empty list as active tool calls.
                # Mock objects and None are both handled safely by this check.
                raw_tool_calls = getattr(result, "tool_calls", None)
                tool_calls: List[Dict[str, Any]] = (
                    raw_tool_calls
                    if isinstance(raw_tool_calls, list) and raw_tool_calls
                    else []
                )

                if not tool_calls:
                    break

                logger.debug(
                    f"Iteration {iteration + 1}: executing {len(tool_calls)} tool call(s)"
                )

                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    args = tool_call.get("args", {})
                    call_id = tool_call.get("id", f"call_{iteration}_{tool_name}")

                    if tool_name and tool_name not in all_sources:
                        all_sources.append(tool_name)

                    matching_tool = next(
                        (t for t in self.tools if t.name == tool_name), None
                    )

                    if matching_tool:
                        # Normalise args to a single string expected by _run(query)
                        if isinstance(args, str):
                            tool_input = args
                        elif isinstance(args, dict) and len(args) == 1:
                            tool_input = next(iter(args.values()))
                        elif isinstance(args, dict):
                            tool_input = json.dumps(args)
                        else:
                            tool_input = str(args)

                        try:
                            tool_output = matching_tool._run(tool_input)
                            logger.debug(
                                f"Tool {tool_name} returned: {str(tool_output)[:120]}"
                            )
                        except Exception as exc:
                            logger.warning(f"Tool {tool_name} failed: {exc}")
                            tool_output = f"Error running tool: {exc}"
                    else:
                        logger.warning(f"Unknown tool requested: {tool_name!r}")
                        tool_output = f"Tool '{tool_name}' is not available."

                    messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=call_id)
                    )

            # Fall back to content-based source detection when no tools were called
            if not all_sources and final_result is not None:
                all_sources = self._extract_sources(final_result)

            output = {
                "symbol": symbol,
                "research_date": datetime.now().isoformat(),
                "period_days": days_back,
                "findings": (
                    final_result.content
                    if final_result and hasattr(final_result, "content")
                    else ""
                ),
                "sources_used": all_sources,
                "metadata": {"agent": self.name},
            }

            self._log_execution(input_data, output)
            logger.info(f"Completed research for {symbol}")
            return output

        except Exception as e:
            logger.error(f"Error in ResearchAgent execution: {e}")
            raise AgentExecutionError(
                agent_name=self.name,
                original_error=e,
                input_data=input_data,
            ) from e

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
            result = self.execute({"symbol": symbol, "days_back": days_back})
            results.append(result)

        logger.info(f"Completed batch research for {len(symbols)} symbols")
        return results

    def _extract_sources(self, result: Any) -> List[str]:
        """Extract data sources used during research."""
        sources = []

        raw_tool_calls = getattr(result, "tool_calls", None)
        if isinstance(raw_tool_calls, list) and raw_tool_calls:
            for tool_call in raw_tool_calls:
                tool_name = tool_call.get("name", "unknown")
                if tool_name not in sources:
                    sources.append(tool_name)
        elif hasattr(result, "content"):
            content = result.content.lower()
            for tool in self.tools:
                if tool.name.lower() in content:
                    sources.append(tool.name)

        return sources

    def focused_research(
        self,
        symbol: str,
        topic: str,
        days_back: int = 7,
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
        return self.execute(
            {"symbol": symbol, "days_back": days_back, "focus_areas": topic}
        )
