"""Adaptive learning system for continuous improvement."""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .ml_data_storage import MLDataStorage
from .ml_knowledge_base import MLKnowledgeBase
from .strategy_predictor import StrategySuccessPredictor, DependencyConflictPredictor
from .pattern_discovery import PatternDiscoveryEngine, DiscoveredPattern


@dataclass
class LearningMetrics:
    """Learning system performance metrics."""
    total_analyses: int
    success_rate: float
    model_accuracy: float
    pattern_count: int
    learning_velocity: float
    insight_quality: float
    last_update: float


class AdaptiveLearningSystem:
    """Continuous learning and recommendation improvement system."""

    def __init__(self, knowledge_base: MLKnowledgeBase):
        """Initialize adaptive learning system."""
        self.kb = knowledge_base
        self.ml_storage = knowledge_base.ml_storage
        
        # Initialize ML models
        self.strategy_predictor = StrategySuccessPredictor(storage=self.ml_storage)
        self.dependency_predictor = DependencyConflictPredictor(storage=self.ml_storage)
        self.pattern_engine = PatternDiscoveryEngine(knowledge_base)
        
        # Learning state
        self.learning_enabled = True
        self.last_retraining = 0
        self.retraining_interval = 24 * 3600  # 24 hours
        self.min_training_samples = 10
        
        # Performance tracking
        self.performance_history = []
        self.recommendation_feedback = []

    def get_adaptive_recommendations(self, analysis: Dict[str, Any], 
                                   system_profile: Dict[str, Any]):
        """Get ML-enhanced adaptive recommendations.

        Returns a list of recommendation dicts for compatibility with tests.
        Additional metadata (e.g., learning_confidence, pattern_insights) is
        attached on the instance for advanced callers.
        """
        if not self.learning_enabled:
            return {"recommendations": [], "learning_disabled": True}
        
        try:
            # Check if models need retraining
            self._check_retraining_need()
            
            # Get base recommendations from knowledge base
            base_recommendations = self.kb.get_ml_recommendations(analysis, system_profile)
            
            # Enhance with ML predictions
            ml_enhanced = self._enhance_with_ml_predictions(analysis, system_profile, base_recommendations)
            
            # Add pattern-based insights and learning confidence
            pattern_insights = self.pattern_engine.generate_insights(analysis)
            ml_enhanced["pattern_insights"] = pattern_insights
            ml_enhanced["learning_confidence"] = self._calculate_learning_confidence()

            # Store the last full recommendation payload for advanced consumers
            self.last_recommendation_payload = ml_enhanced

            # Return only the recommendations list for compatibility with tests
            return ml_enhanced.get("recommendations", [])
            
        except Exception as e:
            print(f"Error getting adaptive recommendations: {e}")
            # On error, return an empty list for tests; also store error payload
            self.last_recommendation_payload = {"recommendations": [], "error": str(e)}
            return []

    def record_feedback(self, analysis: Dict[str, Any], resolution: Dict[str, Any], 
                       outcome: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None):
        """Record feedback for continuous learning."""
        if not self.learning_enabled:
            return
        
        try:
            # Record in knowledge base
            self.kb.record_ml_analysis(analysis, resolution, outcome, analysis.get("system_profile", {}))
            
            # Record feedback for learning
            feedback_record = {
                "timestamp": time.time(),
                "analysis": analysis,
                "resolution": resolution,
                "outcome": outcome,
                "user_feedback": feedback or {},
                "learning_features": self._extract_learning_features(analysis, resolution, outcome)
            }
            
            self.recommendation_feedback.append(feedback_record)
            
            # Trigger learning updates
            self._update_learning_models()
            
        except Exception as e:
            print(f"Error recording feedback: {e}")

    def get_learning_metrics(self) -> LearningMetrics:
        """Get comprehensive learning metrics."""
        try:
            # Get basic metrics
            total_analyses = len(self.ml_storage.get_training_data())
            success_rate = self._calculate_success_rate()
            model_accuracy = self._calculate_model_accuracy()
            pattern_count = len(self.pattern_engine.discovered_patterns)
            learning_velocity = self._calculate_learning_velocity()
            insight_quality = self._calculate_insight_quality()
            
            return LearningMetrics(
                total_analyses=total_analyses,
                success_rate=success_rate,
                model_accuracy=model_accuracy,
                pattern_count=pattern_count,
                learning_velocity=learning_velocity,
                insight_quality=insight_quality,
                last_update=self.last_retraining
            )
            
        except Exception as e:
            print(f"Error calculating learning metrics: {e}")
            return LearningMetrics(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

    def discover_new_patterns(self, min_support: float = 0.1) -> List[DiscoveredPattern]:
        """Discover new patterns from recent data."""
        if not self.learning_enabled:
            return []
        
        try:
            # Discover patterns
            patterns = self.pattern_engine.discover_patterns(min_support)
            
            # Update learning state
            self.last_retraining = time.time()
            
            return patterns
            
        except Exception as e:
            print(f"Error discovering patterns: {e}")
            return []

    def retrain_models(self) -> Dict[str, Any]:
        """Retrain ML models with latest data."""
        if not self.learning_enabled:
            return {"error": "Learning disabled"}
        
        try:
            results = {}
            
            # Retrain strategy predictor
            strategy_result = self.strategy_predictor.train()
            results["strategy_predictor"] = strategy_result
            
            # Retrain dependency predictor
            dependency_result = self.dependency_predictor.train()
            results["dependency_predictor"] = dependency_result
            
            # Update retraining timestamp
            self.last_retraining = time.time()
            
            return results
            
        except Exception as e:
            return {"error": f"Retraining failed: {str(e)}"}

    def export_learning_report(self) -> Dict[str, Any]:
        """Export comprehensive learning report."""
        try:
            metrics = self.get_learning_metrics()
            pattern_summary = self.pattern_engine.get_pattern_summary()
            
            return {
                "learning_metrics": {
                    "total_analyses": metrics.total_analyses,
                    "success_rate": metrics.success_rate,
                    "model_accuracy": metrics.model_accuracy,
                    "pattern_count": metrics.pattern_count,
                    "learning_velocity": metrics.learning_velocity,
                    "insight_quality": metrics.insight_quality
                },
                "pattern_summary": pattern_summary,
                "model_info": {
                    "strategy_predictor": self.strategy_predictor.get_model_info(),
                    "dependency_predictor": self.dependency_predictor.get_model_info()
                },
                "learning_state": {
                    "enabled": self.learning_enabled,
                    "last_retraining": self.last_retraining,
                    "retraining_interval": self.retraining_interval
                }
            }
            
        except Exception as e:
            return {"error": f"Error generating report: {str(e)}"}

    def _check_retraining_need(self):
        """Check if models need retraining."""
        if not self.learning_enabled:
            return
        
        current_time = time.time()
        time_since_retraining = current_time - self.last_retraining
        
        # Check if enough time has passed
        if time_since_retraining < self.retraining_interval:
            return
        
        # Check if we have enough new data
        training_data = self.ml_storage.get_training_data()
        if len(training_data) < self.min_training_samples:
            return
        
        # Trigger retraining
        self.retrain_models()

    def _enhance_with_ml_predictions(self, analysis: Dict[str, Any], 
                                   system_profile: Dict[str, Any], 
                                   base_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance recommendations with ML predictions."""
        enhanced = base_recommendations.copy()
        
        try:
            # Get ML strategy recommendations
            ml_recommendations = self.strategy_predictor.get_strategy_recommendations(
                analysis, system_profile
            )
            
            # Merge with base recommendations
            if "recommendations" in enhanced:
                # Combine and deduplicate recommendations
                combined_recommendations = self._merge_recommendations(
                    enhanced["recommendations"], ml_recommendations
                )
                enhanced["recommendations"] = combined_recommendations
            else:
                enhanced["recommendations"] = ml_recommendations
            
            # Add ML confidence
            enhanced["ml_confidence"] = self._calculate_ml_confidence(ml_recommendations)
            
            # Add dependency conflict predictions
            dependencies = analysis.get("dependencies", [])
            if dependencies:
                conflict_prediction = self.dependency_predictor.predict_conflicts(dependencies)
                enhanced["conflict_prediction"] = conflict_prediction
            
        except Exception as e:
            print(f"Error enhancing with ML predictions: {e}")
        
        return enhanced

    def _merge_recommendations(self, base_recommendations: List[Dict[str, Any]], 
                             ml_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge base and ML recommendations."""
        # Create a map of strategies to recommendations
        strategy_map = {}
        
        # Add base recommendations
        for rec in base_recommendations:
            strategy = rec.get("strategy", "unknown")
            strategy_map[strategy] = rec
        
        # Add/update with ML recommendations
        for rec in ml_recommendations:
            strategy = rec.get("strategy", "unknown")
            if strategy in strategy_map:
                # Merge information
                base_rec = strategy_map[strategy]
                base_rec.update(rec)
                base_rec["source"] = "ml_enhanced"
            else:
                rec["source"] = "ml_only"
                strategy_map[strategy] = rec
        
        return list(strategy_map.values())

    def _calculate_ml_confidence(self, ml_recommendations: List[Dict[str, Any]]) -> float:
        """Calculate overall ML confidence."""
        if not ml_recommendations:
            return 0.0
        
        # Average probability of top recommendations
        top_probabilities = [rec.get("success_probability", 0.0) for rec in ml_recommendations[:3]]
        return np.mean(top_probabilities) if top_probabilities else 0.0

    def _extract_learning_features(self, analysis: Dict[str, Any], 
                                 resolution: Dict[str, Any], 
                                 outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for learning from feedback."""
        return {
            "success": outcome.get("success", False),
            "duration": outcome.get("duration", 0),
            "strategy_used": resolution.get("strategy_type", "unknown"),
            "analysis_confidence": analysis.get("confidence_score", 0.0),
            "dependencies_count": len(analysis.get("dependencies", [])),
            "gpu_required": analysis.get("gpu_required", False),
            "ml_dependencies": analysis.get("ml_dependencies", 0)
        }

    def _update_learning_models(self):
        """Update learning models with new feedback."""
        if not self.learning_enabled:
            return
        
        # Check if we have enough new feedback
        if len(self.recommendation_feedback) < 5:
            return
        
        # Trigger pattern discovery
        self.discover_new_patterns()
        
        # Clear old feedback to prevent memory buildup
        self.recommendation_feedback = self.recommendation_feedback[-50:]  # Keep last 50

    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate."""
        training_data = self.ml_storage.get_training_data()
        if not training_data:
            return 0.0
        
        successful = sum(1 for record in training_data 
                        if record.get("outcome", {}).get("success", False))
        
        return successful / len(training_data)

    def _calculate_model_accuracy(self) -> float:
        """Calculate average model accuracy."""
        accuracies = []
        
        # Strategy predictor accuracy
        strategy_info = self.strategy_predictor.get_model_info()
        if strategy_info.get("trained"):
            accuracies.append(strategy_info.get("metadata", {}).get("test_accuracy", 0.0))
        
        # Dependency predictor accuracy (if available)
        # This would be calculated from dependency predictor metrics
        
        return np.mean(accuracies) if accuracies else 0.0

    def _calculate_learning_velocity(self) -> float:
        """Calculate learning velocity (improvement over time)."""
        if len(self.performance_history) < 2:
            return 0.0
        
        # Calculate improvement in success rate over time
        recent_success_rate = self.performance_history[-1].get("success_rate", 0.0)
        older_success_rate = self.performance_history[0].get("success_rate", 0.0)
        
        return max(0.0, recent_success_rate - older_success_rate)

    def _calculate_insight_quality(self) -> float:
        """Calculate quality of generated insights."""
        # Simplified quality calculation
        # In practice, this would be based on user feedback and insight effectiveness
        
        pattern_count = len(self.pattern_engine.discovered_patterns)
        if pattern_count == 0:
            return 0.0
        
        # Quality based on pattern confidence and support
        avg_confidence = np.mean([
            pattern.confidence for pattern in self.pattern_engine.discovered_patterns.values()
        ])
        
        return min(1.0, avg_confidence)

    def _calculate_learning_confidence(self) -> float:
        """Calculate overall learning system confidence."""
        try:
            # Combine multiple confidence factors
            model_accuracy = self._calculate_model_accuracy()
            success_rate = self._calculate_success_rate()
            insight_quality = self._calculate_insight_quality()
            
            # Weighted average
            confidence = (
                model_accuracy * 0.4 +
                success_rate * 0.4 +
                insight_quality * 0.2
            )
            
            return min(1.0, confidence)
            
        except Exception as e:
            print(f"Error calculating learning confidence: {e}")
            return 0.0

    def enable_learning(self):
        """Enable learning system."""
        self.learning_enabled = True

    def disable_learning(self):
        """Disable learning system."""
        self.learning_enabled = False

    def reset_learning(self):
        """Reset learning system state."""
        self.last_retraining = 0
        self.performance_history = []
        self.recommendation_feedback = []
        
        # Clear discovered patterns
        self.pattern_engine.discovered_patterns = {}
        
        # Retrain models from scratch
        if self.learning_enabled:
            self.retrain_models()

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status."""
        metrics = self.get_learning_metrics()
        
        return {
            "enabled": self.learning_enabled,
            "total_analyses": metrics.total_analyses,
            "success_rate": metrics.success_rate,
            "model_accuracy": metrics.model_accuracy,
            "pattern_count": metrics.pattern_count,
            "last_retraining": metrics.last_update,
            "next_retraining": self.last_retraining + self.retraining_interval,
            "feedback_count": len(self.recommendation_feedback)
        }
