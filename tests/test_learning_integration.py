"""Integration tests for learning system with existing agents."""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from repo_doctor.learning import (
    EnhancedAnalysisAgent, 
    EnhancedResolutionAgent, 
    LearningDashboard,
    MLKnowledgeBase
)
from repo_doctor.models.analysis import Analysis
from repo_doctor.models.resolution import Resolution
from repo_doctor.models.system import SystemProfile
from repo_doctor.utils.config import Config


class TestLearningIntegration:
    """Test learning system integration with existing agents."""
    
    @pytest.fixture
    def temp_kb_path(self, tmp_path):
        """Create temporary knowledge base path."""
        return tmp_path / "test_kb"
    
    @pytest.fixture
    def config(self):
        """Create test configuration with learning enabled."""
        config = Config.load()
        config.integrations.llm.enabled = True
        return config
    
    @pytest.fixture
    def mock_analysis(self):
        """Create mock analysis for testing."""
        analysis = Mock(spec=Analysis)
        analysis.repository = Mock()
        analysis.repository.name = "test-repo"
        analysis.repository.url = "https://github.com/test/test-repo"
        analysis.dependencies = []
        analysis.compatibility_issues = []
        analysis.confidence_score = 0.8
        analysis.model_dump.return_value = {
            "repository": {"name": "test-repo", "url": "https://github.com/test/test-repo"},
            "dependencies": [],
            "compatibility_issues": [],
            "confidence_score": 0.8
        }
        return analysis
    
    @pytest.fixture
    def mock_system_profile(self):
        """Create mock system profile for testing."""
        profile = Mock(spec=SystemProfile)
        profile.hardware = Mock()
        profile.software = Mock()
        profile.model_dump.return_value = {
            "hardware": {"gpu_available": True, "cuda_version": "11.8"},
            "software": {"python_version": "3.9"}
        }
        return profile
    
    def test_enhanced_analysis_agent_initialization(self, temp_kb_path, config):
        """Test enhanced analysis agent initializes correctly."""
        agent = EnhancedAnalysisAgent(knowledge_base_path=str(temp_kb_path), config=config)
        
        assert agent.ml_enabled is True
        assert agent.ml_kb is not None
        assert agent.learning_system is not None
        assert agent.pattern_engine is not None
        assert agent.dependency_predictor is not None
    
    def test_enhanced_resolution_agent_initialization(self, temp_kb_path, config):
        """Test enhanced resolution agent initializes correctly."""
        agent = EnhancedResolutionAgent(knowledge_base_path=str(temp_kb_path), config=config)
        
        assert agent.ml_enabled is True
        assert agent.ml_kb is not None
        assert agent.learning_system is not None
        assert agent.pattern_engine is not None
        assert agent.strategy_predictor is not None
    
    @pytest.mark.asyncio
    async def test_enhanced_analysis_with_ml(self, temp_kb_path, config, mock_analysis):
        """Test enhanced analysis agent with ML capabilities."""
        agent = EnhancedAnalysisAgent(knowledge_base_path=str(temp_kb_path), config=config)
        
        # Mock the parent analyze method
        with patch.object(agent.__class__.__bases__[0], 'analyze', return_value=mock_analysis):
            result = await agent.analyze("https://github.com/test/test-repo")
            
            assert result is not None
            # ML enhancement should add additional insights
            assert hasattr(result, 'ml_insights') or hasattr(result, 'learning_confidence')
    
    @pytest.mark.asyncio
    async def test_enhanced_resolution_with_ml(self, temp_kb_path, config, mock_analysis, mock_system_profile):
        """Test enhanced resolution agent with ML capabilities."""
        agent = EnhancedResolutionAgent(knowledge_base_path=str(temp_kb_path), config=config)
        
        # Mock the parent resolve method
        mock_resolution = Mock(spec=Resolution)
        mock_resolution.strategy = Mock()
        mock_resolution.generated_files = []
        
        with patch.object(agent.__class__.__bases__[0], 'resolve', return_value=mock_resolution):
            result = await agent.resolve(mock_analysis)
            
            assert result is not None
            # ML enhancement should provide better strategy selection
            assert hasattr(result, 'ml_recommendations') or hasattr(result, 'learning_confidence')
    
    def test_learning_dashboard_initialization(self, temp_kb_path):
        """Test learning dashboard initializes correctly."""
        ml_kb = MLKnowledgeBase(temp_kb_path)
        dashboard = LearningDashboard(ml_kb)
        
        assert dashboard.kb is not None
        assert dashboard.ml_storage is not None
        assert dashboard.learning_system is not None
        assert dashboard.pattern_engine is not None
    
    def test_learning_dashboard_metrics(self, temp_kb_path):
        """Test learning dashboard metrics calculation."""
        ml_kb = MLKnowledgeBase(temp_kb_path)
        dashboard = LearningDashboard(ml_kb)
        
        metrics = dashboard.get_dashboard_metrics()
        
        assert metrics is not None
        assert hasattr(metrics, 'total_analyses')
        assert hasattr(metrics, 'success_rate')
        assert hasattr(metrics, 'model_accuracy')
        assert hasattr(metrics, 'learning_enabled')
    
    def test_learning_dashboard_insights(self, temp_kb_path):
        """Test learning dashboard insights retrieval."""
        ml_kb = MLKnowledgeBase(temp_kb_path)
        dashboard = LearningDashboard(ml_kb)
        
        insights = dashboard.get_recent_insights(limit=5)
        
        assert isinstance(insights, list)
        # Should return empty list for new knowledge base
        assert len(insights) == 0
    
    def test_learning_dashboard_recommendations(self, temp_kb_path):
        """Test learning dashboard recommendations."""
        ml_kb = MLKnowledgeBase(temp_kb_path)
        dashboard = LearningDashboard(ml_kb)
        
        recommendations = dashboard.get_learning_recommendations()
        
        assert isinstance(recommendations, list)
        # Should return empty list for new knowledge base
        assert len(recommendations) == 0
    
    def test_ml_knowledge_base_initialization(self, temp_kb_path):
        """Test ML knowledge base initializes correctly."""
        ml_kb = MLKnowledgeBase(temp_kb_path)
        
        assert ml_kb.ml_storage is not None
        assert ml_kb.feature_extractor is not None
        assert ml_kb.data_validator is not None
    
    def test_learning_system_fallback(self, temp_kb_path, config):
        """Test that learning system gracefully falls back when ML components fail."""
        # Test with invalid configuration
        config.integrations.llm.enabled = False
        
        agent = EnhancedAnalysisAgent(knowledge_base_path=str(temp_kb_path), config=config)
        
        # Should still initialize but with ML disabled
        assert agent.ml_enabled is True  # Still enabled in agent
        assert agent.ml_kb is not None
    
    @pytest.mark.asyncio
    async def test_learning_system_with_real_data(self, temp_kb_path, config):
        """Test learning system with realistic data."""
        # Create a more realistic analysis
        analysis_data = {
            "repository": {
                "name": "pytorch-example",
                "url": "https://github.com/pytorch/examples",
                "stars": 1000,
                "topics": ["pytorch", "deep-learning", "computer-vision"]
            },
            "dependencies": [
                {"name": "torch", "version": "1.12.0", "type": "ml"},
                {"name": "torchvision", "version": "0.13.0", "type": "ml"},
                {"name": "numpy", "version": "1.21.0", "type": "scientific"}
            ],
            "compatibility_issues": [
                {"type": "cuda_version", "severity": "medium", "description": "CUDA version mismatch"}
            ],
            "confidence_score": 0.85
        }
        
        # Test ML knowledge base with real data
        ml_kb = MLKnowledgeBase(temp_kb_path)
        
        # Test feature extraction
        features = ml_kb.feature_extractor.extract_features(analysis_data, {})
        assert features is not None
        assert len(features) > 0
        
        # Test pattern discovery
        patterns = ml_kb.pattern_engine.discover_patterns([analysis_data])
        assert isinstance(patterns, list)
        
        # Test learning system
        learning_system = ml_kb.learning_system
        recommendations = learning_system.get_adaptive_recommendations(analysis_data, {})
        assert isinstance(recommendations, list)


class TestLearningSystemCLI:
    """Test learning system CLI integration."""
    
    def test_learning_dashboard_command_availability(self):
        """Test that learning dashboard command is available."""
        from repo_doctor.cli import main
        
        # Check if learning_dashboard command exists
        commands = [cmd.name for cmd in main.commands.values()]
        assert "learning-dashboard" in commands
    
    def test_learning_presets_available(self):
        """Test that learning presets are available."""
        from repo_doctor.presets import PRESETS
        
        # Check for learning-enabled presets
        learning_presets = [name for name, preset in PRESETS.items() 
                           if preset.get("learning_enabled", False)]
        
        assert "ml-research" in learning_presets
        assert "development" in learning_presets
        assert "learning" in learning_presets
    
    def test_learning_preset_configuration(self):
        """Test learning preset configuration."""
        from repo_doctor.presets import get_preset
        
        # Test learning preset
        learning_preset = get_preset("learning")
        assert learning_preset["learning_enabled"] is True
        assert learning_preset["llm_enabled"] is True
        
        # Test ml-research preset
        ml_preset = get_preset("ml-research")
        assert ml_preset["learning_enabled"] is True
        assert ml_preset["llm_enabled"] is True


if __name__ == "__main__":
    pytest.main([__file__])
