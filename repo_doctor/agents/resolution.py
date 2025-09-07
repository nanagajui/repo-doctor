"""Resolution Agent - Solution generation and validation."""

import os
import asyncio
import time
from pathlib import Path
from typing import List, Optional

from ..knowledge.base import KnowledgeBase
from ..utils.logging_config import get_logger, log_performance
from ..models.analysis import Analysis
from ..models.resolution import (
    GeneratedFile,
    Resolution,
    Strategy,
    StrategyType,
    ValidationResult,
    ValidationStatus,
)
from ..strategies import CondaStrategy, DockerStrategy, MicromambaStrategy, VenvStrategy
from ..utils.config import Config
from ..utils.llm import LLMAnalyzer, LLMFactory
from ..validators import ContainerValidator
from .contracts import (
    AgentContractValidator, 
    AgentDataFlow, 
    AgentErrorHandler, 
    AgentPerformanceMonitor
)


class ResolutionAgent:
    """Agent for generating solutions to compatibility issues."""

    def __init__(
        self, config: Optional[Config] = None, knowledge_base_path: Optional[str] = None
    ):
        self.logger = get_logger(__name__)
        self.config = config or Config()
        self.strategies = [DockerStrategy(), MicromambaStrategy(), CondaStrategy(), VenvStrategy()]
        self.validator = ContainerValidator()

        # Initialize knowledge base
        if knowledge_base_path:
            kb_path = Path(knowledge_base_path)
        else:
            kb_path = Path(self.config.knowledge_base.location).expanduser() / "knowledge"

        self.knowledge_base = KnowledgeBase(kb_path)

        # Initialize LLM analyzer if configured
        self.config = config or Config.load()
        self.llm_analyzer = LLMFactory.create_analyzer_sync(self.config)
        
        # Initialize contract validation
        self.performance_monitor = AgentPerformanceMonitor()

    async def resolve(
        self, analysis: Analysis, strategy: Optional[str] = None
    ) -> Resolution:
        """Generate resolution with contract validation.

        This method is now async to support both sync and async usage patterns.
        """
        return await self._resolve_async(analysis, strategy)

    def resolve_sync(
        self, analysis: Analysis, strategy: Optional[str] = None
    ) -> Resolution:
        """Generate resolution (sync) with contract validation.

        This is a fully synchronous implementation used by CLI wrappers.
        LLM-enhanced steps are skipped in the sync path to avoid event loop usage.
        """
        start_time = time.time()

        try:
            # Validate input analysis
            AgentContractValidator.validate_analysis(analysis)

            # Select strategy
            selected = self._select_strategy(analysis, strategy)
            if not selected:
                # Try LLM fallback only if analyzer exists and there are issues (best-effort, sync path skips awaits)
                raise ValueError("No suitable strategy found for this repository")

            # Generate solution using selected strategy
            try:
                resolution = selected.generate_solution(analysis)
            except Exception as e:
                raise ValueError(
                    f"Failed to generate solution with {selected.strategy_type.value} strategy: {str(e)}"
                )

            # Validate the resolution against contracts
            AgentContractValidator.validate_resolution(resolution)

            # Performance logging
            duration = time.time() - start_time
            if not self.performance_monitor.check_resolution_performance(duration):
                self.logger.warning(
                    f"Resolution agent took {duration:.2f}s (target: {self.performance_monitor.performance_targets['resolution_agent']}s)"
                )
            log_performance(
                "resolution_generation", duration, agent="ResolutionAgent", strategy=selected.strategy_type.value
            )

            return resolution

        except Exception as e:
            AgentErrorHandler.handle_resolution_error(e, analysis, "resolution_generation")

    async def _resolve_async(
        self, analysis: Analysis, preferred_strategy: Optional[str] = None
    ) -> Resolution:
        """Async implementation of resolution generation with contract validation."""
        start_time = time.time()
        
        try:
            # Validate input analysis
            AgentContractValidator.validate_analysis(analysis)
            
            # Select strategy based on preference and capability
            strategy = self._select_strategy(analysis, preferred_strategy)

            if not strategy:
                # Try LLM fallback for complex cases
                try:
                    if self.llm_analyzer and analysis.compatibility_issues:
                        llm_recommendation = await self._get_llm_strategy_recommendation(
                            analysis
                        )
                        if llm_recommendation:
                            strategy = self._select_strategy_by_name(
                                llm_recommendation.get("strategy")
                            )
                except Exception:
                    # LLM fallback failed, continue without it
                    pass

                if not strategy:
                    raise ValueError("No suitable strategy found for this repository")

            # Generate solution using selected strategy
            try:
                resolution = strategy.generate_solution(analysis)
            except Exception as e:
                raise ValueError(f"Failed to generate solution with {strategy.strategy_type.value} strategy: {str(e)}")

            # Enhance with LLM insights if available
            try:
                if self.llm_analyzer and analysis.compatibility_issues:
                    await self._enhance_resolution_with_llm(resolution, analysis)
            except Exception:
                # LLM enhancement failed, continue without it
                pass

            # Validate the resolution against contracts
            AgentContractValidator.validate_resolution(resolution)
            
            # Check performance
            duration = time.time() - start_time
            if not self.performance_monitor.check_resolution_performance(duration):
                self.logger.warning(
                    f"Resolution agent took {duration:.2f}s (target: {self.performance_monitor.performance_targets['resolution_agent']}s)"
                )
            
            # Log performance metrics
            log_performance("resolution_generation", duration, agent="ResolutionAgent", strategy=strategy.strategy_type.value)

            return resolution
            
        except Exception as e:
            # Handle errors with proper error handling
            AgentErrorHandler.handle_resolution_error(e, analysis, "resolution_generation")

    def _select_strategy(
        self, analysis: Analysis, preferred: Optional[str] = None
    ) -> Optional[object]:
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

    async def _get_llm_strategy_recommendation(
        self, analysis: Analysis
    ) -> Optional[dict]:
        """Get LLM recommendation for complex compatibility cases."""
        if not self.llm_analyzer:
            return None

        # Convert analysis to dict format for LLM
        analysis_data = {
            "repository": {
                "name": f"{analysis.repository.owner}/{analysis.repository.name}",
                "language": analysis.repository.language,
            },
            "dependencies": [
                {
                    "name": dep.name,
                    "version": dep.version,
                    "gpu_required": dep.gpu_required,
                }
                for dep in analysis.dependencies
            ],
            "python_version_required": getattr(
                analysis, "python_version_required", None
            ),
            "min_gpu_memory_gb": getattr(analysis, "min_gpu_memory_gb", 0),
            "compatibility_issues": [
                {"message": issue.message, "severity": issue.severity}
                for issue in analysis.compatibility_issues
            ],
        }

        return await self.llm_analyzer.analyze_complex_compatibility(analysis_data)

    async def _enhance_resolution_with_llm(
        self, resolution: Resolution, analysis: Analysis
    ):
        """Enhance resolution with LLM-generated insights."""
        if not self.llm_analyzer:
            return

        # Add LLM-generated special instructions if complex issues exist
        complex_issues = [
            issue
            for issue in analysis.compatibility_issues
            if issue.severity in ["critical", "warning"]
        ]

        if complex_issues:
            analysis_data = {
                "repository": {
                    "name": f"{analysis.repository.owner}/{analysis.repository.name}"
                },
                "dependencies": [
                    {"name": dep.name} for dep in analysis.dependencies[:5]
                ],
                "compatibility_issues": [
                    {"message": issue.message} for issue in complex_issues
                ],
            }

            llm_recommendation = await self.llm_analyzer.analyze_complex_compatibility(
                analysis_data
            )
            if llm_recommendation and llm_recommendation.get("special_instructions"):
                # Add LLM insights to resolution instructions
                llm_notes = "\n\n## LLM Analysis Insights\n\n"
                llm_notes += (
                    f"**Reasoning:** {llm_recommendation.get('reasoning', 'N/A')}\n\n"
                )

                special_instructions = llm_recommendation.get(
                    "special_instructions", []
                )
                if special_instructions:
                    llm_notes += "**Special Setup Instructions:**\n"
                    for instruction in special_instructions:
                        llm_notes += f"- {instruction}\n"

                alternatives = llm_recommendation.get("alternatives", [])
                if alternatives:
                    llm_notes += (
                        f"\n**Alternative Approaches:** {', '.join(alternatives)}\n"
                    )

                resolution.instructions += llm_notes

    async def validate_solution(
        self, resolution: Resolution, analysis: Analysis, timeout: int = 300
    ) -> ValidationResult:
        """Validate generated solution using container testing."""
        if resolution.strategy.type == StrategyType.DOCKER:
            result = self.validator.validate_resolution(resolution, analysis, timeout)

            # If validation failed and LLM is available, get diagnosis
            if result.status == ValidationStatus.FAILED and self.llm_analyzer and result.logs:
                analysis_data = {
                    "repository": {
                        "name": f"{analysis.repository.owner}/{analysis.repository.name}"
                    },
                    "dependencies": [
                        {"name": dep.name} for dep in analysis.dependencies[:5]
                    ],
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
                status=ValidationStatus.SUCCESS,
                duration=0.0,
                logs=["Validation skipped for non-Docker strategy"],
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
