"""
Base Agent class for all specialized agents in the system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from loguru import logger
# from langchain_core.memory import ConversationBufferMemory  # Not available in this version


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the financial analysis system.
    
    Each agent should inherit from this class and implement the execute method.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        llm: Any,
        tools: Optional[list] = None,
        verbose: bool = False
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name
            description: Agent description
            llm: Language model to use
            tools: List of tools available to the agent
            verbose: Whether to enable verbose logging
        """
        self.name = name
        self.description = description
        self.llm = llm
        self.tools = tools or []
        self.verbose = verbose
        
        # Initialize memory for conversation history (commented out due to import issues)
        # self.memory = ConversationBufferMemory(
        #     memory_key="chat_history",
        #     return_messages=True
        # )
        
        logger.info(f"Initialized {self.name}")
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary function.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            Dictionary containing the agent's output
        """
        pass
    
    def _log_execution(self, input_data: Dict[str, Any], output: Dict[str, Any]):
        """Log agent execution for debugging and monitoring."""
        if self.verbose:
            logger.debug(f"{self.name} - Input: {input_data}")
            logger.debug(f"{self.name} - Output: {output}")
    
    def reset_memory(self):
        """Reset the agent's conversation memory."""
        self.memory.clear()
        logger.info(f"Reset memory for {self.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            "name": self.name,
            "description": self.description,
            "tools_count": len(self.tools),
            "memory_size": len(self.memory.buffer) if hasattr(self.memory, 'buffer') else 0
        }
