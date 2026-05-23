from abc import ABC, abstractmethod
from core.models import AgentVote, Market


class BaseAgent(ABC):
    name: str = "BaseAgent"

    @abstractmethod
    def analyze(
        self, ticker: str, klines: list[dict], market: Market, **kwargs
    ) -> AgentVote:
        """Analyze market data and return a vote."""
        ...
