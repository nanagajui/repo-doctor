"""Agent system for repository analysis and resolution."""

from .analysis import AnalysisAgent
from .profile import ProfileAgent
from .resolution import ResolutionAgent

# Enhanced agents with ML capabilities
try:
    from ..learning import EnhancedAnalysisAgent, EnhancedResolutionAgent
    __all__ = ["ProfileAgent", "AnalysisAgent", "ResolutionAgent", "EnhancedAnalysisAgent", "EnhancedResolutionAgent"]
except ImportError:
    # Fallback if learning system not available
    __all__ = ["ProfileAgent", "AnalysisAgent", "ResolutionAgent"]
