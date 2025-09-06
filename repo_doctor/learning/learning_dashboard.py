"""Learning dashboard for monitoring learning system performance."""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from .ml_knowledge_base import MLKnowledgeBase
from .adaptive_learning import AdaptiveLearningSystem, LearningMetrics
from .pattern_discovery import PatternDiscoveryEngine


@dataclass
class DashboardMetrics:
    """Dashboard metrics for learning system."""
    total_analyses: int
    success_rate: float
    model_accuracy: float
    pattern_count: int
    learning_velocity: float
    insight_quality: float
    storage_size_mb: float
    last_update: float
    learning_enabled: bool


class LearningDashboard:
    """Dashboard for monitoring learning system performance."""

    def __init__(self, knowledge_base: MLKnowledgeBase):
        """Initialize learning dashboard."""
        self.kb = knowledge_base
        self.ml_storage = knowledge_base.ml_storage
        self.learning_system = AdaptiveLearningSystem(knowledge_base)
        self.pattern_engine = PatternDiscoveryEngine(knowledge_base)
        self.metrics_calculator = LearningMetricsCalculator()

    def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get comprehensive dashboard metrics."""
        try:
            # Get learning metrics
            learning_metrics = self.learning_system.get_learning_metrics()
            
            # Get storage statistics
            storage_stats = self.ml_storage.get_storage_stats()
            
            # Get pattern summary
            pattern_summary = self.pattern_engine.get_pattern_summary()
            
            return DashboardMetrics(
                total_analyses=learning_metrics.total_analyses,
                success_rate=learning_metrics.success_rate,
                model_accuracy=learning_metrics.model_accuracy,
                pattern_count=learning_metrics.pattern_count,
                learning_velocity=learning_metrics.learning_velocity,
                insight_quality=learning_metrics.insight_quality,
                storage_size_mb=storage_stats.get("storage_size_mb", 0.0),
                last_update=learning_metrics.last_update,
                learning_enabled=self.kb.learning_enabled
            )
            
        except Exception as e:
            print(f"Error getting dashboard metrics: {e}")
            return DashboardMetrics(0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, False)

    def get_learning_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent learning insights."""
        try:
            # Get recent patterns
            patterns = self.pattern_engine.discovered_patterns
            recent_patterns = sorted(
                patterns.values(),
                key=lambda p: getattr(p, 'timestamp', 0),
                reverse=True
            )[:limit]
            
            insights = []
            for pattern in recent_patterns:
                insights.append({
                    "type": "pattern_discovery",
                    "pattern_id": pattern.pattern_id,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "support": pattern.support,
                    "timestamp": getattr(pattern, 'timestamp', time.time())
                })
            
            return insights
            
        except Exception as e:
            print(f"Error getting learning insights: {e}")
            return []

    def get_performance_trends(self, days: int = 7) -> Dict[str, List[float]]:
        """Get performance trends over time."""
        try:
            # This would typically query historical data
            # For now, return mock data
            trends = {
                "success_rate": [0.7, 0.72, 0.75, 0.73, 0.78, 0.8, 0.82],
                "model_accuracy": [0.65, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78],
                "pattern_count": [5, 7, 9, 12, 15, 18, 22],
                "learning_velocity": [0.02, 0.03, 0.05, 0.04, 0.06, 0.08, 0.1]
            }
            
            return trends
            
        except Exception as e:
            print(f"Error getting performance trends: {e}")
            return {}

    def get_top_patterns(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top patterns by support and confidence."""
        try:
            patterns = self.pattern_engine.discovered_patterns
            sorted_patterns = sorted(
                patterns.values(),
                key=lambda p: (p.support, p.confidence),
                reverse=True
            )[:limit]
            
            return [
                {
                    "pattern_id": pattern.pattern_id,
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "support": pattern.support,
                    "confidence": pattern.confidence,
                    "recommendations": pattern.recommendations
                }
                for pattern in sorted_patterns
            ]
            
        except Exception as e:
            print(f"Error getting top patterns: {e}")
            return []

    def get_model_performance(self) -> Dict[str, Any]:
        """Get ML model performance metrics."""
        try:
            # Get strategy predictor info
            from .strategy_predictor import StrategySuccessPredictor
            strategy_predictor = StrategySuccessPredictor(storage=self.ml_storage)
            strategy_info = strategy_predictor.get_model_info()
            
            # Get dependency predictor info
            from .strategy_predictor import DependencyConflictPredictor
            dependency_predictor = DependencyConflictPredictor(storage=self.ml_storage)
            
            return {
                "strategy_predictor": {
                    "trained": strategy_info.get("trained", False),
                    "accuracy": strategy_info.get("metadata", {}).get("test_accuracy", 0.0),
                    "feature_count": strategy_info.get("feature_count", 0),
                    "feature_importance": strategy_info.get("feature_importance", {})
                },
                "dependency_predictor": {
                    "trained": dependency_predictor.model is not None,
                    "feature_count": len(dependency_predictor.feature_columns)
                }
            }
            
        except Exception as e:
            print(f"Error getting model performance: {e}")
            return {}

    def get_learning_recommendations(self) -> List[str]:
        """Get recommendations for improving learning system."""
        try:
            recommendations = []
            metrics = self.get_dashboard_metrics()
            # For a fresh knowledge base with no data, return no recommendations
            if metrics.total_analyses == 0 and metrics.pattern_count == 0:
                return []
            
            # Success rate recommendations
            if metrics.success_rate < 0.7:
                recommendations.append("Success rate is below 70%. Consider retraining models with more data.")
            
            # Model accuracy recommendations
            if metrics.model_accuracy < 0.8:
                recommendations.append("Model accuracy is below 80%. Consider feature engineering or more training data.")
            
            # Pattern count recommendations
            if metrics.pattern_count < 10:
                recommendations.append("Few patterns discovered. Run pattern discovery with more data.")
            
            # Learning velocity recommendations
            if metrics.learning_velocity < 0.05:
                recommendations.append("Learning velocity is low. Consider increasing retraining frequency.")
            
            # Storage recommendations
            if metrics.storage_size_mb > 1000:
                recommendations.append("Storage size is large. Consider cleaning up old data.")
            
            # Update frequency recommendations
            time_since_update = time.time() - metrics.last_update
            if time_since_update > 7 * 24 * 3600:  # 7 days
                recommendations.append("Models haven't been updated recently. Consider retraining.")
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting learning recommendations: {e}")
            return []

    def export_learning_report(self) -> Dict[str, Any]:
        """Export comprehensive learning report."""
        try:
            metrics = self.get_dashboard_metrics()
            insights = self.get_learning_insights(20)
            patterns = self.get_top_patterns(10)
            model_performance = self.get_model_performance()
            trends = self.get_performance_trends()
            recommendations = self.get_learning_recommendations()
            
            return {
                "timestamp": time.time(),
                "metrics": {
                    "total_analyses": metrics.total_analyses,
                    "success_rate": metrics.success_rate,
                    "model_accuracy": metrics.model_accuracy,
                    "pattern_count": metrics.pattern_count,
                    "learning_velocity": metrics.learning_velocity,
                    "insight_quality": metrics.insight_quality,
                    "storage_size_mb": metrics.storage_size_mb,
                    "learning_enabled": metrics.learning_enabled
                },
                "insights": insights,
                "top_patterns": patterns,
                "model_performance": model_performance,
                "trends": trends,
                "recommendations": recommendations
            }
            
        except Exception as e:
            return {"error": f"Error generating report: {str(e)}"}

    def get_learning_status_summary(self) -> Dict[str, Any]:
        """Get concise learning status summary."""
        try:
            metrics = self.get_dashboard_metrics()
            
            # Determine overall health status
            health_score = 0
            if metrics.success_rate > 0.7:
                health_score += 1
            if metrics.model_accuracy > 0.8:
                health_score += 1
            if metrics.pattern_count > 5:
                health_score += 1
            if metrics.learning_velocity > 0.05:
                health_score += 1
            if metrics.learning_enabled:
                health_score += 1
            
            health_status = "excellent" if health_score >= 4 else "good" if health_score >= 3 else "needs_attention"
            
            return {
                "health_status": health_status,
                "health_score": health_score,
                "learning_enabled": metrics.learning_enabled,
                "total_analyses": metrics.total_analyses,
                "success_rate": f"{metrics.success_rate:.1%}",
                "model_accuracy": f"{metrics.model_accuracy:.1%}",
                "pattern_count": metrics.pattern_count,
                "last_update": time.ctime(metrics.last_update) if metrics.last_update > 0 else "Never"
            }
            
        except Exception as e:
            return {"error": f"Error getting status summary: {str(e)}"}

    def cleanup_old_data(self, max_age_days: int = 30) -> Dict[str, Any]:
        """Clean up old learning data."""
        try:
            cleaned_count = self.ml_storage.cleanup_old_data(max_age_days)
            
            return {
                "success": True,
                "cleaned_files": cleaned_count,
                "max_age_days": max_age_days
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def retrain_models(self) -> Dict[str, Any]:
        """Trigger model retraining."""
        try:
            results = self.learning_system.retrain_models()
            
            return {
                "success": True,
                "results": results,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def discover_new_patterns(self, min_support: float = 0.1) -> Dict[str, Any]:
        """Trigger pattern discovery."""
        try:
            patterns = self.learning_system.discover_new_patterns(min_support)
            
            return {
                "success": True,
                "patterns_discovered": len(patterns),
                "min_support": min_support,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class LearningMetricsCalculator:
    """Calculate learning system metrics."""

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def calculate_learning_effectiveness(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate learning effectiveness metrics."""
        if not training_data:
            return {"pattern_discovery_rate": 0.0, "prediction_accuracy": 0.0, "recommendation_quality": 0.0}
        
        # Calculate pattern discovery rate
        pattern_discovery_rate = self._calculate_pattern_discovery_rate(training_data)
        
        # Calculate prediction accuracy
        prediction_accuracy = self._calculate_prediction_accuracy(training_data)
        
        # Calculate recommendation quality
        recommendation_quality = self._calculate_recommendation_quality(training_data)
        
        return {
            "pattern_discovery_rate": pattern_discovery_rate,
            "prediction_accuracy": prediction_accuracy,
            "recommendation_quality": recommendation_quality
        }

    def calculate_system_performance(self, training_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate system performance metrics."""
        if not training_data:
            return {"analysis_speed": 0.0, "memory_usage": 0.0, "storage_efficiency": 0.0}
        
        # Calculate analysis speed (simplified)
        analysis_speed = self._calculate_analysis_speed(training_data)
        
        # Calculate memory usage (simplified)
        memory_usage = self._calculate_memory_usage(training_data)
        
        # Calculate storage efficiency
        storage_efficiency = self._calculate_storage_efficiency(training_data)
        
        return {
            "analysis_speed": analysis_speed,
            "memory_usage": memory_usage,
            "storage_efficiency": storage_efficiency
        }

    def _calculate_pattern_discovery_rate(self, training_data: List[Dict[str, Any]]) -> float:
        """Calculate pattern discovery rate."""
        # Simplified calculation
        # In practice, this would be based on actual pattern discovery metrics
        return min(1.0, len(training_data) / 100.0)

    def _calculate_prediction_accuracy(self, training_data: List[Dict[str, Any]]) -> float:
        """Calculate prediction accuracy."""
        # Simplified calculation
        # In practice, this would be based on actual model performance
        successful = sum(1 for record in training_data 
                        if record.get("outcome", {}).get("success", False))
        return successful / len(training_data) if training_data else 0.0

    def _calculate_recommendation_quality(self, training_data: List[Dict[str, Any]]) -> float:
        """Calculate recommendation quality."""
        # Simplified calculation
        # In practice, this would be based on user feedback and success rates
        return min(1.0, len(training_data) / 50.0)

    def _calculate_analysis_speed(self, training_data: List[Dict[str, Any]]) -> float:
        """Calculate analysis speed metric."""
        # Simplified calculation
        # In practice, this would be based on actual timing data
        return 0.8  # Placeholder

    def _calculate_memory_usage(self, training_data: List[Dict[str, Any]]) -> float:
        """Calculate memory usage metric."""
        # Simplified calculation
        # In practice, this would be based on actual memory measurements
        return 0.6  # Placeholder

    def _calculate_storage_efficiency(self, training_data: List[Dict[str, Any]]) -> float:
        """Calculate storage efficiency metric."""
        # Simplified calculation
        # In practice, this would be based on actual storage metrics
        return 0.7  # Placeholder
