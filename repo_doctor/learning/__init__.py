"""Learning system for Repo Doctor - ML-based intelligence enhancement."""

from .feature_extractor import FeatureExtractor
from .ml_knowledge_base import MLKnowledgeBase
from .ml_data_storage import MLDataStorage
from .data_quality_validator import DataQualityValidator
from .strategy_predictor import StrategySuccessPredictor, DependencyConflictPredictor
from .pattern_discovery import PatternDiscoveryEngine, DiscoveredPattern
from .adaptive_learning import AdaptiveLearningSystem, LearningMetrics
from .enhanced_resolution_agent import EnhancedResolutionAgent
from .enhanced_analysis_agent import EnhancedAnalysisAgent
from .learning_dashboard import LearningDashboard, DashboardMetrics

__all__ = [
    "FeatureExtractor",
    "MLKnowledgeBase", 
    "MLDataStorage",
    "DataQualityValidator",
    "StrategySuccessPredictor",
    "DependencyConflictPredictor",
    "PatternDiscoveryEngine",
    "DiscoveredPattern",
    "AdaptiveLearningSystem",
    "LearningMetrics",
    "EnhancedResolutionAgent",
    "EnhancedAnalysisAgent",
    "LearningDashboard",
    "DashboardMetrics",
]
