"""Resolution Agent - Solution generation and validation."""

from typing import List, Optional
from ..models.analysis import Analysis
from ..models.resolution import Resolution, Strategy, StrategyType, GeneratedFile
from ..strategies import DockerStrategy, CondaStrategy, VenvStrategy


class ResolutionAgent:
    """Agent for generating solutions to compatibility issues."""
    
    def __init__(self):
        self.strategies = [
            DockerStrategy(),
            CondaStrategy(), 
            VenvStrategy()
        ]
    
    def resolve(self, analysis: Analysis, preferred_strategy: Optional[str] = None) -> Resolution:
        """Generate resolution for compatibility issues."""
        
        # Select strategy based on preference and capability
        strategy = self._select_strategy(analysis, preferred_strategy)
        
        if not strategy:
            raise ValueError("No suitable strategy found for this repository")
        
        # Generate solution using selected strategy
        return strategy.generate_solution(analysis)
    
    def _select_strategy(self, analysis: Analysis, preferred: Optional[str] = None) -> Optional[object]:
        """Select best strategy for the given analysis."""
        
        # Filter strategies that can handle this analysis
        capable_strategies = [s for s in self.strategies if s.can_handle(analysis)]
        
        if not capable_strategies:
            return None
        
        # If preferred strategy specified and available, use it
        if preferred:
            for strategy in capable_strategies:
                if strategy.strategy_type.value == preferred:
                    return strategy
        
        # Otherwise, select by priority (Docker > Conda > Venv for ML repos)
        return max(capable_strategies, key=lambda s: s.priority)
    
    def validate_solution(self, resolution: Resolution) -> bool:
        """Validate generated solution."""
        # TODO: Implement container-based validation
        # TODO: Build and test the generated environment
        return False
