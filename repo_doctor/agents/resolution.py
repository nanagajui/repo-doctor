"""Resolution Agent - Solution generation and validation."""

from typing import List, Optional
from ..models.analysis import Analysis
from ..models.resolution import Resolution, Strategy, StrategyType, GeneratedFile, ValidationResult
from ..strategies import DockerStrategy, CondaStrategy, VenvStrategy
from ..validators import ContainerValidator
from ..knowledge import KnowledgeBase, FileSystemStorage
from pathlib import Path
import os


class ResolutionAgent:
    """Agent for generating solutions to compatibility issues."""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.strategies = [
            DockerStrategy(),
            CondaStrategy(), 
            VenvStrategy()
        ]
        self.validator = ContainerValidator()
        
        # Initialize knowledge base
        if knowledge_base_path:
            kb_path = Path(knowledge_base_path)
        else:
            kb_path = Path.home() / ".repo-doctor" / "knowledge"
        
        self.knowledge_base = KnowledgeBase(kb_path)
    
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
    
    def validate_solution(self, resolution: Resolution, analysis: Analysis, 
                         timeout: int = 300) -> ValidationResult:
        """Validate generated solution using container testing."""
        if resolution.strategy.type == StrategyType.DOCKER:
            result = self.validator.validate_resolution(resolution, analysis, timeout)
            
            # Record outcome in knowledge base
            self.knowledge_base.record_outcome(analysis, resolution, result)
            
            return result
        else:
            # For non-Docker strategies, return success for now
            # TODO: Implement validation for Conda and Venv strategies
            return ValidationResult(
                status="success",
                duration=0.0,
                logs=["Validation skipped for non-Docker strategy"]
            )
    
    def get_similar_solutions(self, analysis: Analysis, limit: int = 3) -> List[dict]:
        """Get similar solutions from knowledge base."""
        return self.knowledge_base.get_similar_analyses(analysis, limit)
    
    def get_success_patterns(self, strategy_type: Optional[str] = None) -> dict:
        """Get successful resolution patterns."""
        return self.knowledge_base.get_success_patterns(strategy_type)
    
    def cleanup_validation_artifacts(self) -> int:
        """Clean up validation test containers and images."""
        return self.validator.cleanup_test_containers()
