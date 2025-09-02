"""Base strategy class for environment generation."""

from abc import ABC, abstractmethod
from typing import List
from ..models.analysis import Analysis
from ..models.resolution import Resolution, Strategy, StrategyType


class BaseStrategy(ABC):
    """Base class for resolution strategies."""
    
    def __init__(self, strategy_type: StrategyType, priority: int = 0):
        self.strategy_type = strategy_type
        self.priority = priority
    
    @abstractmethod
    def can_handle(self, analysis: Analysis) -> bool:
        """Check if this strategy can handle the given analysis."""
        pass
    
    @abstractmethod
    def generate_solution(self, analysis: Analysis) -> Resolution:
        """Generate solution for the given analysis."""
        pass
    
    def _create_strategy_config(self, **kwargs) -> Strategy:
        """Create strategy configuration."""
        return Strategy(
            type=self.strategy_type,
            priority=self.priority,
            requirements=kwargs
        )
