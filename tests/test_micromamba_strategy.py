"""Tests for MicromambaStrategy."""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from repo_doctor.strategies.micromamba import MicromambaStrategy
from repo_doctor.models.analysis import Analysis, DependencyInfo, RepositoryInfo
from repo_doctor.models.resolution import StrategyType
from repo_doctor.models.system import SystemProfile
from repo_doctor.models.analysis import RepositoryInfo, DependencyInfo


class TestMicromambaStrategy:
    """Test cases for MicromambaStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = MicromambaStrategy()
        
        # Mock repository
        self.mock_repo = RepositoryInfo(
            name="test-ml-project",
            url="https://github.com/test/ml-project",
            owner="test",
            description="Test ML project"
        )
        
        # Mock analysis with ML dependencies
        self.mock_analysis = Mock(spec=Analysis)
        self.mock_analysis.repository = self.mock_repo
        self.mock_analysis.python_version_required = "3.9"
        self.mock_analysis.get_python_dependencies.return_value = [
            DependencyInfo(name="torch", version="1.9.0", type="python", source="requirements.txt"),
            DependencyInfo(name="numpy", version="1.21.0", type="python", source="requirements.txt"),
            DependencyInfo(name="transformers", version="4.12.0", type="python", source="requirements.txt"),
            DependencyInfo(name="pandas", version="1.3.0", type="python", source="requirements.txt")
        ]
        self.mock_analysis.is_gpu_required.return_value = True

    def test_strategy_type(self):
        """Test strategy type is MICROMAMBA."""
        assert self.strategy.strategy_type == StrategyType.MICROMAMBA

    def test_priority_higher_than_conda(self):
        """Test micromamba has higher priority than conda (9 vs 8)."""
        assert self.strategy.priority == 9

    def test_can_handle_ml_dependencies(self):
        """Test strategy can handle ML dependencies."""
        assert self.strategy.can_handle(self.mock_analysis) is True

    def test_cannot_handle_non_ml_dependencies(self):
        """Test strategy rejects non-ML dependencies."""
        # Mock analysis without ML dependencies
        self.mock_analysis.get_python_dependencies.return_value = [
            DependencyInfo(name="requests", version="2.25.1", type="python", source="requirements.txt"),
            DependencyInfo(name="click", version="8.0.1", type="python", source="requirements.txt")
        ]
        
        assert self.strategy.can_handle(self.mock_analysis) is False

    def test_generate_solution_structure(self):
        """Test solution generation creates proper structure."""
        resolution = self.strategy.generate_solution(self.mock_analysis)
        
        # Check basic structure
        assert resolution.strategy.type == StrategyType.MICROMAMBA
        assert resolution.strategy.can_handle_gpu is True
        assert resolution.strategy.estimated_setup_time == 120
        
        # Check generated files
        file_paths = [f.path for f in resolution.generated_files]
        assert "environment.yml" in file_paths
        assert "setup_micromamba.sh" in file_paths
        
        # Check setup commands
        assert "chmod +x setup_micromamba.sh" in resolution.setup_commands
        assert "./setup_micromamba.sh" in resolution.setup_commands

    def test_environment_file_generation(self):
        """Test environment.yml generation."""
        env_content, req_content = self.strategy._generate_environment_files(self.mock_analysis)
        
        # Check environment.yml structure
        assert "name: test_ml_project" in env_content
        assert "channels:" in env_content
        assert "conda-forge" in env_content
        assert "pytorch" in env_content
        assert "nvidia" in env_content
        assert "dependencies:" in env_content
        assert "python=3.9" in env_content
        assert "numpy=1.21.0" in env_content
        assert "pandas=1.3.0" in env_content
        assert "cudatoolkit" in env_content  # GPU support
        
        # Check requirements.txt for pip-only packages
        assert req_content is not None
        assert "transformers==4.12.0" in req_content
        assert "torch==1.9.0" in req_content  # PyTorch goes to pip in this case

    def test_package_classification(self):
        """Test intelligent package classification between conda and pip."""
        # Test with packages that should go to conda
        conda_deps = [
            DependencyInfo(name="numpy", version="1.21.0", type="python", source="requirements.txt"),
            DependencyInfo(name="scipy", version="1.7.0", type="python", source="requirements.txt"),
            DependencyInfo(name="pandas", version="1.3.0", type="python", source="requirements.txt"),
            DependencyInfo(name="matplotlib", version="3.4.0", type="python", source="requirements.txt")
        ]
        
        self.mock_analysis.get_python_dependencies.return_value = conda_deps
        env_content, req_content = self.strategy._generate_environment_files(self.mock_analysis)
        
        # All should be in conda environment
        for dep in conda_deps:
            assert f"{dep.name}={dep.version}" in env_content
        
        # No pip requirements should be generated
        assert req_content == ""

    def test_setup_script_generation(self):
        """Test setup script generation."""
        script = self.strategy._generate_setup_script(self.mock_analysis, has_pip_deps=True)
        
        # Check script structure
        assert "#!/bin/bash" in script
        assert "micromamba create -f environment.yml" in script
        assert "micromamba run -p ./env pip install -r requirements.txt" in script
        assert "micromamba run -p ./env pip install -e ." in script
        assert "test_ml_project" in script

    def test_setup_script_without_pip_deps(self):
        """Test setup script when no pip dependencies exist."""
        script = self.strategy._generate_setup_script(self.mock_analysis, has_pip_deps=False)
        
        # Should not include pip install commands
        assert "pip install -r requirements.txt" not in script
        assert "micromamba create -f environment.yml" in script

    def test_instructions_generation(self):
        """Test user instructions generation."""
        instructions = self.strategy._generate_instructions(self.mock_analysis, has_pip_deps=True)
        
        # Check key instruction elements
        assert "Micromamba Environment Setup" in instructions
        assert "test-ml-project" in instructions
        assert "micromamba create -f environment.yml" in instructions
        assert "micromamba run -p ./env" in instructions
        assert "micromamba activate ./env" in instructions
        assert "Performance Benefits" in instructions
        assert "2-3x faster" in instructions
        assert "✅ CUDA toolkit included" in instructions  # GPU support

    def test_instructions_without_gpu(self):
        """Test instructions when GPU is not required."""
        self.mock_analysis.is_gpu_required.return_value = False
        instructions = self.strategy._generate_instructions(self.mock_analysis, has_pip_deps=False)
        
        assert "ℹ️ CPU-only environment configured" in instructions

    def test_estimated_size_smaller_than_conda(self):
        """Test estimated size is smaller than conda."""
        with patch.object(self.strategy.config.advanced, 'default_conda_size_mb', 2000):
            resolution = self.strategy.generate_solution(self.mock_analysis)
            
            # Should be 80% of conda size
            assert resolution.estimated_size_mb == 1600

    def test_gpu_support_integration(self):
        """Test GPU support is properly integrated."""
        self.mock_analysis.is_gpu_required.return_value = True
        
        env_content, _ = self.strategy._generate_environment_files(self.mock_analysis)
        
        # Check GPU packages are included
        assert "cudatoolkit" in env_content
        assert "cudnn" in env_content
        assert "nvidia" in env_content  # Channel

    def test_repository_name_sanitization(self):
        """Test repository name is properly sanitized for environment name."""
        self.mock_repo.name = "My-Complex_Project.Name"
        
        env_content, _ = self.strategy._generate_environment_files(self.mock_analysis)
        
        # Should be sanitized to valid environment name
        assert "name: my_complex_project.name" in env_content

    @patch('repo_doctor.strategies.micromamba.MicromambaStrategy._create_strategy_config')
    def test_strategy_config_creation(self, mock_create_config):
        """Test strategy configuration is created with correct parameters."""
        mock_create_config.return_value = Mock()
        
        self.strategy.generate_solution(self.mock_analysis)
        
        mock_create_config.assert_called_once_with(
            can_handle_gpu=True,
            estimated_setup_time=120
        )

    def test_requirements_file_conditional_generation(self):
        """Test requirements.txt is only generated when needed."""
        # Test with only conda packages
        conda_only_deps = [
            DependencyInfo(name="numpy", version="1.21.0", type="python", source="requirements.txt"),
            DependencyInfo(name="pandas", version="1.3.0", type="python", source="requirements.txt")
        ]
        
        self.mock_analysis.get_python_dependencies.return_value = conda_only_deps
        resolution = self.strategy.generate_solution(self.mock_analysis)
        
        file_paths = [f.path for f in resolution.generated_files]
        
        # Should only have environment.yml and setup script
        assert "environment.yml" in file_paths
        assert "setup_micromamba.sh" in file_paths
        assert "requirements.txt" not in file_paths

    def test_micromamba_auto_install_in_script(self):
        """Test setup script includes micromamba auto-installation."""
        script = self.strategy._generate_setup_script(self.mock_analysis, has_pip_deps=False)
        
        # Check auto-install logic
        assert "micromamba not found" in script
        assert "curl -Ls https://micro.mamba.pm/install.sh" in script
        assert "export PATH" in script

    def test_error_handling_in_script(self):
        """Test setup script includes proper error handling."""
        script = self.strategy._generate_setup_script(self.mock_analysis, has_pip_deps=False)
        
        # Check error handling
        assert "set -e" in script
        assert "exit 1" in script


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.advanced.default_conda_size_mb = 2000
    config.advanced.default_python_version = "3.9"
    return config


class TestMicromambaStrategyIntegration:
    """Integration tests for MicromambaStrategy."""

    def test_strategy_import(self):
        """Test strategy can be imported from strategies module."""
        from repo_doctor.strategies import MicromambaStrategy
        
        strategy = MicromambaStrategy()
        assert strategy.strategy_type == StrategyType.MICROMAMBA

    def test_strategy_in_resolution_agent(self):
        """Test strategy is available in ResolutionAgent."""
        from repo_doctor.agents.resolution import ResolutionAgent
        
        agent = ResolutionAgent()
        strategy_types = [s.strategy_type for s in agent.strategies]
        
        assert StrategyType.MICROMAMBA in strategy_types

    def test_micromamba_priority_ordering(self):
        """Test micromamba has correct priority in strategy list."""
        from repo_doctor.agents.resolution import ResolutionAgent
        
        agent = ResolutionAgent()
        
        # Find micromamba strategy
        micromamba_strategy = None
        conda_strategy = None
        
        for strategy in agent.strategies:
            if strategy.strategy_type == StrategyType.MICROMAMBA:
                micromamba_strategy = strategy
            elif strategy.strategy_type == StrategyType.CONDA:
                conda_strategy = strategy
        
        assert micromamba_strategy is not None
        assert conda_strategy is not None
        assert micromamba_strategy.priority > conda_strategy.priority
