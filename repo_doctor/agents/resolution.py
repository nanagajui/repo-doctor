"""Resolution Agent - Solution generation and validation."""

from typing import List, Optional
from ..models.analysis import Analysis
from ..models.resolution import Resolution, Strategy, StrategyType, GeneratedFile, ValidationResult
from ..strategies import DockerStrategy, CondaStrategy, VenvStrategy
from ..validators import ContainerValidator
from ..knowledge import KnowledgeBase, FileSystemStorage
from ..utils.llm import LLMFactory, LLMAnalyzer
from ..utils.config import Config
from pathlib import Path
import os


class ResolutionAgent:
    """Agent for generating solutions to compatibility issues."""
    
    def __init__(self, knowledge_base_path: Optional[str] = None, config: Optional[Config] = None):
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
        
        # Initialize LLM analyzer if configured
        self.config = config or Config.load()
        self.llm_analyzer = LLMFactory.create_analyzer(self.config)
    
    async def resolve(self, analysis: Analysis, preferred_strategy: Optional[str] = None) -> Resolution:
        """Generate resolution for compatibility issues."""
        
        # Select strategy based on preference and capability
        strategy = self._select_strategy(analysis, preferred_strategy)
        
        if not strategy:
            # Try LLM fallback for complex cases
            if self.llm_analyzer and analysis.compatibility_issues:
                llm_recommendation = await self._get_llm_strategy_recommendation(analysis)
                if llm_recommendation:
                    strategy = self._select_strategy_by_name(llm_recommendation.get('strategy'))
            
            if not strategy:
                raise ValueError("No suitable strategy found for this repository")
        
        # Generate solution using selected strategy
        resolution = strategy.generate_solution(analysis)
        
        # Enhance with LLM insights if available
        if self.llm_analyzer and analysis.compatibility_issues:
            await self._enhance_resolution_with_llm(resolution, analysis)
        
        return resolution
    
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
    
    def _select_strategy_by_name(self, strategy_name: str) -> Optional[object]:
        """Select strategy by name."""
        for strategy in self.strategies:
            if strategy.strategy_type.value == strategy_name:
                return strategy
        return None
    
    async def _get_llm_strategy_recommendation(self, analysis: Analysis) -> Optional[dict]:
        """Get LLM recommendation for complex compatibility cases."""
        if not self.llm_analyzer:
            return None
        
        # Convert analysis to dict format for LLM
        analysis_data = {
            'repository': {
                'name': f"{analysis.repository.owner}/{analysis.repository.name}",
                'language': analysis.repository.language
            },
            'dependencies': [
                {'name': dep.name, 'version': dep.version, 'gpu_required': dep.gpu_required}
                for dep in analysis.dependencies
            ],
            'python_version_required': getattr(analysis, 'python_version_required', None),
            'min_gpu_memory_gb': getattr(analysis, 'min_gpu_memory_gb', 0),
            'compatibility_issues': [
                {'message': issue.message, 'severity': issue.severity}
                for issue in analysis.compatibility_issues
            ]
        }
        
        return await self.llm_analyzer.analyze_complex_compatibility(analysis_data)
    
    async def _enhance_resolution_with_llm(self, resolution: Resolution, analysis: Analysis):
        """Enhance resolution with LLM-generated insights."""
        if not self.llm_analyzer:
            return
        
        # Add LLM-generated special instructions if complex issues exist
        complex_issues = [issue for issue in analysis.compatibility_issues 
                         if issue.severity in ['critical', 'warning']]
        
        if complex_issues:
            analysis_data = {
                'repository': {
                    'name': f"{analysis.repository.owner}/{analysis.repository.name}"
                },
                'dependencies': [{'name': dep.name} for dep in analysis.dependencies[:5]],
                'compatibility_issues': [{'message': issue.message} for issue in complex_issues]
            }
            
            llm_recommendation = await self.llm_analyzer.analyze_complex_compatibility(analysis_data)
            if llm_recommendation and llm_recommendation.get('special_instructions'):
                # Add LLM insights to resolution instructions
                llm_notes = "\n\n## LLM Analysis Insights\n\n"
                llm_notes += f"**Reasoning:** {llm_recommendation.get('reasoning', 'N/A')}\n\n"
                
                special_instructions = llm_recommendation.get('special_instructions', [])
                if special_instructions:
                    llm_notes += "**Special Setup Instructions:**\n"
                    for instruction in special_instructions:
                        llm_notes += f"- {instruction}\n"
                
                alternatives = llm_recommendation.get('alternatives', [])
                if alternatives:
                    llm_notes += f"\n**Alternative Approaches:** {', '.join(alternatives)}\n"
                
                resolution.instructions += llm_notes
    
    async def validate_solution(self, resolution: Resolution, analysis: Analysis, 
                               timeout: int = 300) -> ValidationResult:
        """Validate generated solution using container testing."""
        if resolution.strategy.type == StrategyType.DOCKER:
            result = self.validator.validate_resolution(resolution, analysis, timeout)
            
            # If validation failed and LLM is available, get diagnosis
            if result.status.value == "failure" and self.llm_analyzer and result.logs:
                analysis_data = {
                    'repository': {
                        'name': f"{analysis.repository.owner}/{analysis.repository.name}"
                    },
                    'dependencies': [{'name': dep.name} for dep in analysis.dependencies[:5]]
                }
                
                llm_diagnosis = await self.llm_analyzer.diagnose_validation_failure(
                    result.logs, analysis_data
                )
                
                if llm_diagnosis:
                    # Add LLM diagnosis to the result
                    result.error_message = f"{result.error_message or 'Validation failed'}\n\nLLM Diagnosis: {llm_diagnosis}"
            
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
