"""
Base Agent class for all specialized agents in the system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from loguru import logger


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the financial analysis system.

    Each agent should inherit from this class and implement the execute() method.
    Conversation history is maintained in a simple in-memory list so that
    agents can accumulate context across calls within a single session.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm: Any,
        tools: Optional[List[Any]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the base agent.

        Args:
            name: Agent name.
            description: Agent description / role.
            llm: LangChain-compatible language model.
            tools: List of LangChain tools available to this agent.
            verbose: Enable debug-level execution logging.
        """
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools or []
        self.verbose = verbose

        # Simple in-memory conversation history: list of {"role": ..., "content": ...}
        self._memory: List[Dict[str, str]] = []

        logger.info(f"Initialized {self.name}")

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary function.

        Args:
            input_data: Dictionary of input parameters (agent-specific).

        Returns:
            Dictionary containing the agent's output.
        """

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def add_to_memory(self, role: str, content: str) -> None:
        """
        Append a message to the agent's conversation history.

        Args:
            role: "user", "assistant", or "system".
            content: Message content.
        """
        self._memory.append({"role": role, "content": content})

    def reset_memory(self) -> None:
        """Clear the agent's conversation history."""
        self._memory.clear()
        logger.info(f"Reset memory for {self.name}")

    def get_memory(self) -> List[Dict[str, str]]:
        """Return a copy of the current conversation history."""
        return list(self._memory)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """
        Return the current status of the agent.

        Returns:
            Dict with name, description, tool count, and memory size.
        """
        return {
            "name": self.name,
            "description": self.description,
            "tools_count": len(self.tools),
            "memory_size": len(self._memory),
        }

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------

    def _log_execution(
        self, input_data: Dict[str, Any], output: Dict[str, Any]
    ) -> None:
        """Log agent execution details when verbose mode is enabled."""
        if self.verbose:
            logger.debug(f"{self.name} — Input : {input_data}")
            logger.debug(f"{self.name} — Output: {output}")
