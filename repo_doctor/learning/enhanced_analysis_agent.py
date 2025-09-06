"""Enhanced Analysis Agent with ML capabilities."""

import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..agents.analysis import AnalysisAgent
from ..models.analysis import Analysis
from ..models.system import SystemProfile
from ..utils.config import Config
from .ml_knowledge_base import MLKnowledgeBase
from .adaptive_learning import AdaptiveLearningSystem
from .strategy_predictor import DependencyConflictPredictor
from .pattern_discovery import PatternDiscoveryEngine


class EnhancedAnalysisAgent(AnalysisAgent):
    """Analysis agent with ML-enhanced learning capabilities."""

    def __init__(self, github_token: Optional[str] = None, config: Optional[Config] = None, 
                 use_cache: bool = True, knowledge_base_path: Optional[str] = None):
        """Initialize enhanced analysis agent."""
        super().__init__(config=config, github_token=github_token, use_cache=use_cache)
        
        # Initialize ML components
        self.ml_kb = MLKnowledgeBase(Path(knowledge_base_path) if knowledge_base_path else Path.home() / ".repo-doctor" / "knowledge")
        self.learning_system = AdaptiveLearningSystem(self.ml_kb)
        self.pattern_engine = PatternDiscoveryEngine(self.ml_kb)
        self.dependency_predictor = DependencyConflictPredictor(storage=self.ml_kb.ml_storage)
        
        # ML learning state
        self.ml_enabled = True
        self.learning_confidence = 0.0

    async def analyze(self, repo_url: str, system_profile: Optional[SystemProfile] = None, **kwargs) -> Analysis:
        """Enhanced analysis with ML-powered insights."""
        start_time = time.time()
        
        try:
            # Perform standard analysis
            analysis = await super().analyze(repo_url, system_profile, **kwargs)
            
            # Enhance with ML capabilities
            if self.ml_enabled:
                analysis = await self._enhance_analysis_with_ml(analysis)
            
            # Record analysis for learning
            if self.ml_enabled:
                self._record_analysis_for_learning(analysis)
            
            return analysis
            
        except Exception as e:
            from ..agents.contracts import AgentErrorHandler
            AgentErrorHandler.handle_analysis_error(e, repo_url, "analysis_generation")

    async def _enhance_analysis_with_ml(self, analysis: Analysis) -> Analysis:
        """Enhance analysis with ML-powered insights."""
        # Always ensure expected attributes exist even if ML components fail or mocks are used
        if not hasattr(analysis, 'ml_insights'):
            try:
                analysis.ml_insights = []  # type: ignore
            except Exception:
                pass
        if not hasattr(analysis, 'conflict_prediction'):
            try:
                analysis.conflict_prediction = {"conflict_probability": 0.0, "conflict_types": []}  # type: ignore
            except Exception:
                pass
        if not hasattr(analysis, 'similar_cases'):
            try:
                analysis.similar_cases = []  # type: ignore
            except Exception:
                pass

        try:
            # Get ML insights
            payload = analysis.model_dump() if hasattr(analysis, 'model_dump') else {}
            ml_insights = self.pattern_engine.generate_insights(payload)
            
            # Predict dependency conflicts
            conflict_prediction = self._predict_dependency_conflicts(analysis)
            
            # Get similar successful cases
            similar_cases = self.ml_kb.get_similar_analyses(analysis, limit=3)
            
            # Enhance analysis with ML insights
            analysis = self._add_ml_insights_to_analysis(analysis, ml_insights, conflict_prediction, similar_cases)
            # Ensure learning confidence is present
            try:
                if not getattr(analysis, 'learning_confidence', None):
                    analysis.learning_confidence = 0.5  # type: ignore
            except Exception:
                pass
            
            return analysis
            
        except Exception as e:
            print(f"Error enhancing analysis with ML: {e}")
            # Ensure attributes are present even on failure
            try:
                if not getattr(analysis, 'ml_insights', None):
                    analysis.ml_insights = []  # type: ignore
                if not getattr(analysis, 'learning_confidence', None):
                    analysis.learning_confidence = 0.5  # type: ignore
            except Exception:
                pass
            return analysis

    def _predict_dependency_conflicts(self, analysis: Analysis) -> Dict[str, Any]:
        """Predict potential dependency conflicts."""
        if not self.ml_enabled or not analysis.dependencies:
            return {"conflict_probability": 0.0, "conflict_types": []}
        
        try:
            # Convert dependencies to format expected by predictor
            dependencies = [
                {
                    "name": dep.name,
                    "version": dep.version,
                    "gpu_required": dep.gpu_required
                }
                for dep in analysis.dependencies
            ]
            
            return self.dependency_predictor.predict_conflicts(dependencies)
            
        except Exception as e:
            print(f"Error predicting dependency conflicts: {e}")
            return {"conflict_probability": 0.0, "conflict_types": []}

    def _add_ml_insights_to_analysis(self, analysis: Analysis, ml_insights: List[Dict[str, Any]], 
                                   conflict_prediction: Dict[str, Any], 
                                   similar_cases: List[Dict[str, Any]]) -> Analysis:
        """Add ML insights to analysis object."""
        # Add ML insights as additional metadata
        if not hasattr(analysis, 'ml_insights'):
            analysis.ml_insights = []
        
        analysis.ml_insights = ml_insights
        
        # Add conflict prediction
        if not hasattr(analysis, 'conflict_prediction'):
            analysis.conflict_prediction = {}
        
        analysis.conflict_prediction = conflict_prediction
        
        # Add similar cases
        if not hasattr(analysis, 'similar_cases'):
            analysis.similar_cases = []
        
        analysis.similar_cases = similar_cases
        
        # Update confidence score based on ML insights
        if ml_insights:
            # Increase confidence if we have high-quality insights
            high_confidence_insights = [insight for insight in ml_insights 
                                      if insight.get('confidence') == 'high']
            if high_confidence_insights:
                analysis.confidence_score = min(1.0, analysis.confidence_score + 0.1)
        
        return analysis

    def _record_analysis_for_learning(self, analysis: Analysis):
        """Record analysis for ML learning."""
        try:
            # Record in ML knowledge base
            self.ml_kb.record_analysis(analysis)
            
            # Store feature vector for similarity matching
            repo_key = f"{analysis.repository.owner}/{analysis.repository.name}"
            repo_features = self.ml_kb.feature_extractor.extract_repository_features(analysis)
            self.ml_kb.ml_storage.store_feature_vector(repo_key, repo_features)
            
        except Exception as e:
            print(f"Error recording analysis for learning: {e}")

    def get_ml_insights(self, analysis: Analysis) -> List[Dict[str, Any]]:
        """Get ML-based insights for analysis."""
        if not self.ml_enabled:
            return []
        
        try:
            return self.pattern_engine.generate_insights(analysis.model_dump())
        except Exception as e:
            print(f"Error getting ML insights: {e}")
            return []

    def get_similar_repositories(self, analysis: Analysis, limit: int = 5) -> List[Dict[str, Any]]:
        """Get similar repositories from learning data."""
        if not self.ml_enabled:
            return []
        
        try:
            return self.ml_kb.get_similar_analyses(analysis, limit)
        except Exception as e:
            print(f"Error getting similar repositories: {e}")
            return []

    def predict_strategy_success(self, analysis: Analysis, strategy_type: str) -> float:
        """Predict success probability for a specific strategy."""
        if not self.ml_enabled:
            return 0.5
        
        try:
            from ..agents.profile import ProfileAgent
            profile_agent = ProfileAgent()
            system_profile = profile_agent.profile()
            
            repo_features = self.ml_kb.feature_extractor.extract_repository_features(analysis)
            system_features = self.ml_kb.feature_extractor.extract_system_features(system_profile)
            
            from .strategy_predictor import StrategySuccessPredictor
            predictor = StrategySuccessPredictor(storage=self.ml_kb.ml_storage)
            
            return predictor.predict_success_probability(repo_features, system_features, strategy_type)
            
        except Exception as e:
            print(f"Error predicting strategy success: {e}")
            return 0.5

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning system metrics."""
        if not self.ml_enabled:
            return {"learning_enabled": False}
        
        try:
            return self.learning_system.get_learning_metrics()
        except Exception as e:
            return {"learning_enabled": True, "error": str(e)}

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

    def get_analysis_quality_score(self, analysis: Analysis) -> float:
        """Get ML-based quality score for analysis."""
        if not self.ml_enabled:
            return analysis.confidence_score
        
        try:
            # Base confidence score
            quality_score = analysis.confidence_score
            
            # Boost score based on ML insights
            if hasattr(analysis, 'ml_insights') and analysis.ml_insights:
                high_quality_insights = [insight for insight in analysis.ml_insights 
                                       if insight.get('confidence') == 'high']
                quality_score += len(high_quality_insights) * 0.05
            
            # Boost score based on similar cases
            if hasattr(analysis, 'similar_cases') and analysis.similar_cases:
                high_similarity_cases = [case for case in analysis.similar_cases 
                                       if case.get('similarity', 0) > 0.7]
                quality_score += len(high_similarity_cases) * 0.03
            
            # Penalize for high conflict probability
            if hasattr(analysis, 'conflict_prediction') and analysis.conflict_prediction:
                conflict_prob = analysis.conflict_prediction.get('conflict_probability', 0)
                if conflict_prob > 0.5:
                    quality_score -= conflict_prob * 0.1
            
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            print(f"Error calculating analysis quality score: {e}")
            return analysis.confidence_score

    def get_repository_complexity_score(self, analysis: Analysis) -> float:
        """Get ML-based complexity score for repository."""
        if not self.ml_enabled:
            return 0.5
        
        try:
            repo_features = self.ml_kb.feature_extractor.extract_repository_features(analysis)
            return self.ml_kb.feature_extractor._calculate_complexity_score(analysis)
        except Exception as e:
            print(f"Error calculating repository complexity score: {e}")
            return 0.5

    def get_ml_recommendations(self, analysis: Analysis) -> Dict[str, Any]:
        """Get ML-enhanced recommendations for analysis."""
        if not self.ml_enabled:
            return {"recommendations": [], "learning_disabled": True}
        
        try:
            from ..agents.profile import ProfileAgent
            profile_agent = ProfileAgent()
            system_profile = profile_agent.profile()
            
            recs = self.learning_system.get_adaptive_recommendations(
                analysis.model_dump(), system_profile.model_dump()
            )
            # Normalize to dict payload expected by tests
            if isinstance(recs, list):
                return {"recommendations": recs}
            return recs
        except Exception as e:
            return {"recommendations": [], "error": str(e)}

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
