"""Enhanced Resolution Agent with ML capabilities."""

import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..agents.resolution import ResolutionAgent
from ..models.analysis import Analysis
from ..models.resolution import Resolution, ValidationResult
from ..models.system import SystemProfile
from ..utils.config import Config
from .ml_knowledge_base import MLKnowledgeBase
from .adaptive_learning import AdaptiveLearningSystem
from .strategy_predictor import StrategySuccessPredictor
from .pattern_discovery import PatternDiscoveryEngine


class EnhancedResolutionAgent(ResolutionAgent):
    """Resolution agent with ML-enhanced learning capabilities."""

    def __init__(self, knowledge_base_path: Optional[str] = None, config: Optional[Config] = None):
        """Initialize enhanced resolution agent."""
        super().__init__(config=config, knowledge_base_path=knowledge_base_path)
        
        # Initialize ML components
        self.ml_kb = MLKnowledgeBase(Path(knowledge_base_path) if knowledge_base_path else Path.home() / ".repo-doctor" / "knowledge")
        self.learning_system = AdaptiveLearningSystem(self.ml_kb)
        self.pattern_engine = PatternDiscoveryEngine(self.ml_kb)
        self.strategy_predictor = StrategySuccessPredictor(storage=self.ml_kb.ml_storage)
        
        # ML learning state
        self.ml_enabled = True
        self.learning_confidence = 0.0

    async def resolve(self, analysis: Analysis, preferred_strategy: Optional[str] = None) -> Resolution:
        """Enhanced resolution with ML-powered strategy selection."""
        start_time = time.time()
        
        try:
            # Validate input analysis
            from ..agents.contracts import AgentContractValidator
            AgentContractValidator.validate_analysis(analysis)
            
            # Get system profile for ML recommendations
            system_profile = self._get_system_profile()
            
            # Get ML-enhanced recommendations
            ml_recommendations = self.learning_system.get_adaptive_recommendations(
                analysis.model_dump(), system_profile.model_dump()
            )
            
            # Select best strategy based on ML predictions
            if not preferred_strategy and self.ml_enabled:
                best_strategy = self._select_ml_optimized_strategy(analysis, system_profile, ml_recommendations)
            elif preferred_strategy:
                best_strategy = self._validate_strategy_with_ml(preferred_strategy, analysis, system_profile, ml_recommendations)
            else:
                # Fallback to traditional strategy selection
                best_strategy = self._select_strategy(analysis, preferred_strategy)
            
            if not best_strategy:
                # Try LLM fallback for complex cases
                try:
                    if self.llm_analyzer and analysis.compatibility_issues:
                        llm_recommendation = await self._get_llm_strategy_recommendation(analysis)
                        if llm_recommendation:
                            best_strategy = self._select_strategy_by_name(
                                llm_recommendation.get("strategy")
                            )
                except Exception:
                    pass
                
                if not best_strategy:
                    raise ValueError("No suitable strategy found for this repository")

            # Generate solution with ML insights
            resolution = await self._generate_ml_enhanced_solution(
                analysis, best_strategy, ml_recommendations
            )
            
            # Add learning-based insights
            if self.ml_enabled:
                resolution.insights = ml_recommendations.get("pattern_insights", [])
                resolution.confidence_score = ml_recommendations.get("learning_confidence", 0.5)
            
            # Validate the resolution
            from ..agents.contracts import AgentContractValidator
            AgentContractValidator.validate_resolution(resolution)
            
            # Check performance
            duration = time.time() - start_time
            if not self.performance_monitor.check_resolution_performance(duration):
                print(f"Warning: Resolution agent took {duration:.2f}s (target: {self.performance_monitor.performance_targets['resolution_agent']}s)")

            return resolution
            
        except Exception as e:
            from ..agents.contracts import AgentErrorHandler
            AgentErrorHandler.handle_resolution_error(e, analysis, "resolution_generation")

    async def validate_solution(self, resolution: Resolution, analysis: Analysis, timeout: int = 300) -> ValidationResult:
        """Enhanced validation with ML learning."""
        # Perform standard validation
        result = await super().validate_solution(resolution, analysis, timeout)
        
        # Record outcome for learning
        if self.ml_enabled:
            system_profile = self._get_system_profile()
            self.ml_kb.record_ml_analysis(analysis, resolution, result, system_profile)
            
            # Record feedback for continuous learning
            self.learning_system.record_feedback(
                analysis.model_dump(), 
                resolution.model_dump(), 
                result.model_dump()
            )
        
        return result

    def _get_system_profile(self) -> SystemProfile:
        """Get current system profile for ML recommendations."""
        from ..agents.profile import ProfileAgent
        profile_agent = ProfileAgent()
        return profile_agent.profile()

    def _select_ml_optimized_strategy(self, analysis: Analysis, system_profile: SystemProfile, 
                                    ml_recommendations: Dict[str, Any]) -> Optional[object]:
        """Select strategy using ML predictions."""
        if not self.ml_enabled or not ml_recommendations.get("recommendations"):
            return self._select_strategy(analysis, None)
        
        # Get ML recommendations
        recommendations = ml_recommendations["recommendations"]
        
        # Find the best strategy from ML recommendations
        for rec in recommendations:
            strategy_name = rec.get("strategy")
            if strategy_name:
                strategy = self._select_strategy_by_name(strategy_name)
                if strategy and strategy.can_handle(analysis):
                    self.learning_confidence = rec.get("success_probability", 0.5)
                    return strategy
        
        # Fallback to traditional selection
        return self._select_strategy(analysis, None)

    def _validate_strategy_with_ml(self, preferred_strategy: str, analysis: Analysis, 
                                 system_profile: SystemProfile, ml_recommendations: Dict[str, Any]) -> Optional[object]:
        """Validate preferred strategy with ML predictions."""
        # Check if preferred strategy is in ML recommendations
        if ml_recommendations.get("recommendations"):
            for rec in ml_recommendations["recommendations"]:
                if rec.get("strategy") == preferred_strategy:
                    strategy = self._select_strategy_by_name(preferred_strategy)
                    if strategy and strategy.can_handle(analysis):
                        self.learning_confidence = rec.get("success_probability", 0.5)
                        return strategy
        
        # Fallback to traditional selection
        return self._select_strategy(analysis, preferred_strategy)

    async def _generate_ml_enhanced_solution(self, analysis: Analysis, strategy: object, 
                                           ml_recommendations: Dict[str, Any]) -> Resolution:
        """Generate solution with ML-enhanced insights."""
        # Generate base solution
        resolution = strategy.generate_solution(analysis)
        
        # Enhance with ML insights
        if self.ml_enabled and ml_recommendations:
            # Add ML confidence to resolution
            resolution.confidence_score = ml_recommendations.get("learning_confidence", 0.5)
            
            # Add pattern insights to instructions
            pattern_insights = ml_recommendations.get("pattern_insights", [])
            if pattern_insights:
                ml_notes = "\n\n## ML Learning Insights\n\n"
                for insight in pattern_insights[:3]:  # Top 3 insights
                    ml_notes += f"**{insight.get('type', 'Insight')}:** {insight.get('message', 'N/A')}\n"
                    if insight.get('confidence'):
                        ml_notes += f"*Confidence: {insight['confidence']}*\n"
                    ml_notes += "\n"
                
                resolution.instructions += ml_notes
            
            # Add conflict predictions
            conflict_prediction = ml_recommendations.get("conflict_prediction")
            if conflict_prediction and conflict_prediction.get("conflict_probability", 0) > 0.3:
                conflict_notes = "\n\n## Dependency Conflict Prediction\n\n"
                conflict_notes += f"**Conflict Probability:** {conflict_prediction['conflict_probability']:.1%}\n"
                
                conflict_types = conflict_prediction.get("conflict_types", [])
                if conflict_types:
                    conflict_notes += f"**Potential Conflicts:** {', '.join(conflict_types)}\n"
                
                recommended_actions = conflict_prediction.get("recommended_actions", [])
                if recommended_actions:
                    conflict_notes += "**Recommended Actions:**\n"
                    for action in recommended_actions:
                        conflict_notes += f"- {action}\n"
                
                resolution.instructions += conflict_notes
        
        # Enhance with LLM if available
        try:
            if self.llm_analyzer and analysis.compatibility_issues:
                await self._enhance_resolution_with_llm(resolution, analysis)
        except Exception:
            pass
        
        return resolution

    def get_ml_insights(self, analysis: Analysis) -> List[Dict[str, Any]]:
        """Get ML-based insights for analysis."""
        if not self.ml_enabled:
            return []
        
        try:
            return self.pattern_engine.generate_insights(analysis.model_dump())
        except Exception as e:
            print(f"Error getting ML insights: {e}")
            return []

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning system metrics."""
        if not self.ml_enabled:
            return {"learning_enabled": False}
        
        try:
            return self.learning_system.get_learning_metrics()
        except Exception as e:
            return {"learning_enabled": True, "error": str(e)}

    def get_strategy_recommendations(self, analysis: Analysis) -> List[Dict[str, Any]]:
        """Get ML-enhanced strategy recommendations."""
        if not self.ml_enabled:
            return []
        
        try:
            system_profile = self._get_system_profile()
            repo_features = self.ml_kb.feature_extractor.extract_repository_features(analysis)
            system_features = self.ml_kb.feature_extractor.extract_system_features(system_profile)
            
            return self.strategy_predictor.get_strategy_recommendations(repo_features, system_features)
        except Exception as e:
            print(f"Error getting strategy recommendations: {e}")
            return []

    def discover_patterns(self, min_support: float = 0.1) -> List[Dict[str, Any]]:
        """Discover new patterns from recent data."""
        if not self.ml_enabled:
            return []
        
        try:
            patterns = self.learning_system.discover_new_patterns(min_support)
            return [
                {
                    "pattern_id": pattern.pattern_id,
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "support": pattern.support
                }
                for pattern in patterns
            ]
        except Exception as e:
            print(f"Error discovering patterns: {e}")
            return []

    def retrain_models(self) -> Dict[str, Any]:
        """Retrain ML models with latest data."""
        if not self.ml_enabled:
            return {"error": "Learning disabled"}
        
        try:
            return self.learning_system.retrain_models()
        except Exception as e:
            return {"error": f"Retraining failed: {str(e)}"}

    def export_learning_report(self) -> Dict[str, Any]:
        """Export comprehensive learning report."""
        if not self.ml_enabled:
            return {"learning_enabled": False}
        
        try:
            return self.learning_system.export_learning_report()
        except Exception as e:
            return {"learning_enabled": True, "error": str(e)}

    def enable_ml_learning(self):
        """Enable ML learning capabilities."""
        self.ml_enabled = True
        self.learning_system.enable_learning()

    def disable_ml_learning(self):
        """Disable ML learning capabilities."""
        self.ml_enabled = False
        self.learning_system.disable_learning()

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status."""
        if not self.ml_enabled:
            return {"learning_enabled": False}
        
        try:
            return self.learning_system.get_learning_status()
        except Exception as e:
            return {"learning_enabled": True, "error": str(e)}
