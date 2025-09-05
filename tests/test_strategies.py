"""Test suite for environment generation strategies."""

from unittest.mock import Mock, patch

import pytest

from repo_doctor.models.analysis import Analysis, DependencyInfo, RepositoryInfo, DependencyType
from repo_doctor.strategies import CondaStrategy, DockerStrategy, VenvStrategy


@pytest.fixture
def mock_analysis():
    """Create a mock analysis for testing."""
    return Analysis(
        repository=RepositoryInfo(
            url="https://github.com/test/repo",
            name="repo",
            owner="test",
            branch="main",
            has_dockerfile=False,
            has_docker_compose=False,
            has_conda_env=False,
            has_requirements=True,
            has_setup_py=False,
            has_pyproject=False,
            readme_content="# Test Repo\nA machine learning project",
        ),
        dependencies=[
            DependencyInfo(
                name="torch",
                version=">=2.0.0",
                type=DependencyType.PYTHON,
                source="requirements.txt",
                gpu_required=True,
            ),
            DependencyInfo(
                name="numpy",
                version=">=1.24.0",
                type=DependencyType.PYTHON,
                source="requirements.txt",
                gpu_required=False,
            ),
            DependencyInfo(
                name="transformers",
                version=">=4.30.0",
                type=DependencyType.PYTHON,
                source="requirements.txt",
                gpu_required=False,
            ),
        ],
        python_version="3.10",
        gpu_required=True,
        compatibility_issues=[],
        confidence_score=0.9,
        analyzed_files=["requirements.txt"],
    )


class TestDockerStrategy:
    """Tests for Docker environment strategy."""

    def test_can_handle(self, mock_analysis):
        """Test Docker strategy can handle analysis."""
        strategy = DockerStrategy()

        # Docker strategy should always be able to handle
        assert strategy.can_handle(mock_analysis) is True

    def test_generate_dockerfile(self, mock_analysis):
        """Test Dockerfile generation."""
        strategy = DockerStrategy()
        resolution = strategy.generate_solution(mock_analysis)
        files = resolution.generated_files

        # Should generate multiple files
        assert len(files) > 0

        # Should have a Dockerfile
        dockerfile = next((f for f in files if f.path == "Dockerfile"), None)
        assert dockerfile is not None

        # Dockerfile should contain expected content
        content = dockerfile.content
        assert "FROM" in content
        assert "torch" in content
        assert "numpy" in content

    def test_generate_with_gpu_support(self, mock_analysis):
        """Test Dockerfile generation with GPU support."""
        strategy = DockerStrategy()
        resolution = strategy.generate_solution(mock_analysis)
        files = resolution.generated_files

        dockerfile = next((f for f in files if f.path == "Dockerfile"), None)
        assert dockerfile is not None

        # Should use CUDA base image for GPU
        content = dockerfile.content
        assert "cuda" in content.lower() or "nvidia" in content.lower()

    def test_generate_docker_compose(self, mock_analysis):
        """Test docker-compose.yml generation."""
        strategy = DockerStrategy()
        resolution = strategy.generate_solution(mock_analysis)
        files = resolution.generated_files

        # Should have docker-compose.yml
        compose = next((f for f in files if f.path == "docker-compose.yml"), None)
        assert compose is not None

        # Should have GPU runtime configuration if GPU required
        if mock_analysis.is_gpu_required():
            assert (
                "runtime: nvidia" in compose.content or "gpus: all" in compose.content
            )

    def test_generate_setup_script(self, mock_analysis):
        """Test setup.sh script generation."""
        strategy = DockerStrategy()
        resolution = strategy.generate_solution(mock_analysis)
        files = resolution.generated_files

        # Should have setup script
        setup = next((f for f in files if f.path == "setup.sh"), None)
        assert setup is not None

        # Should have Docker build command
        assert "docker build" in setup.content or "docker-compose" in setup.content


class TestCondaStrategy:
    """Tests for Conda environment strategy."""

    def test_can_handle(self, mock_analysis):
        """Test Conda strategy can handle analysis."""
        strategy = CondaStrategy()

        # Conda strategy should always be able to handle
        assert strategy.can_handle(mock_analysis) is True

    def test_generate_environment_yml(self, mock_analysis):
        """Test environment.yml generation."""
        strategy = CondaStrategy()
        resolution = strategy.generate_solution(mock_analysis)
        files = resolution.generated_files

        # Should have environment.yml
        env_file = next((f for f in files if f.path == "environment.yml"), None)
        assert env_file is not None

        # Should contain dependencies
        content = env_file.content
        assert "name:" in content
        assert "dependencies:" in content
        assert "python=3.10" in content or "python==3.10" in content
        assert "torch" in content
        assert "numpy" in content

    def test_generate_with_cuda_dependencies(self, mock_analysis):
        """Test Conda environment with CUDA dependencies."""
        strategy = CondaStrategy()
        resolution = strategy.generate_solution(mock_analysis)
        files = resolution.generated_files

        env_file = next((f for f in files if f.path == "environment.yml"), None)
        content = env_file.content

        # Should include CUDA-specific channels or packages
        if mock_analysis.is_gpu_required():
            assert "pytorch" in content or "conda-forge" in content
            assert "cudatoolkit" in content or "cuda" in content.lower()

    def test_generate_setup_instructions(self, mock_analysis):
        """Test setup instructions generation."""
        strategy = CondaStrategy()
        resolution = strategy.generate_solution(mock_analysis)
        files = resolution.generated_files

        # Should have setup script
        setup_script = next((f for f in files if "setup_conda.sh" in f.path), None)
        assert setup_script is not None

        # Should contain conda commands
        assert (
            "conda env create" in setup_script.content
            or "conda create" in setup_script.content
        )


class TestVenvStrategy:
    """Tests for virtual environment strategy."""

    def test_can_handle(self, mock_analysis):
        """Test Venv strategy can handle analysis."""
        strategy = VenvStrategy()

        # Create a non-GPU analysis for venv strategy
        non_gpu_analysis = Analysis(
            repository=mock_analysis.repository,
            dependencies=[
                DependencyInfo(
                    name="numpy",
                    version=">=1.24.0",
                    type=DependencyType.PYTHON,
                    source="requirements.txt",
                    gpu_required=False,
                ),
                DependencyInfo(
                    name="pandas",
                    version=">=2.0.0",
                    type=DependencyType.PYTHON,
                    source="requirements.txt",
                    gpu_required=False,
                ),
            ],
            python_version_required="3.10",
            cuda_version_required=None,
            min_memory_gb=0.0,
            min_gpu_memory_gb=0.0,
            compatibility_issues=[],
            analysis_time=0.0,
            confidence_score=0.8,
        )

        # Should be able to handle non-GPU analysis
        assert strategy.can_handle(non_gpu_analysis) is True

    def test_generate_requirements_txt(self, mock_analysis):
        """Test requirements.txt generation."""
        strategy = VenvStrategy()
        resolution = strategy.generate_solution(mock_analysis)
        files = resolution.generated_files

        # Should have requirements.txt
        reqs = next((f for f in files if f.path == "requirements.txt"), None)
        assert reqs is not None

        # Should contain all dependencies
        content = reqs.content
        assert "torch" in content
        assert "numpy" in content
        assert "transformers" in content

    def test_generate_setup_script(self, mock_analysis):
        """Test virtual environment setup script."""
        strategy = VenvStrategy()
        resolution = strategy.generate_solution(mock_analysis)
        files = resolution.generated_files

        # Should have setup script
        setup = next((f for f in files if f.path == "setup_venv.sh"), None)
        assert setup is not None

        # Should contain venv creation and activation
        content = setup.content
        assert "python" in content
        assert "venv" in content
        assert "pip install" in content
        assert "requirements.txt" in content

    def test_generate_with_gpu_warning(self, mock_analysis):
        """Test GPU warning in venv strategy."""
        strategy = VenvStrategy()
        resolution = strategy.generate_solution(mock_analysis)
        files = resolution.generated_files

        # If GPU required, should have warning in setup instructions
        if mock_analysis.is_gpu_required():
            instructions = next((f for f in files if "SETUP" in f.path), None)
            if instructions:
                assert (
                    "gpu" in instructions.content.lower()
                    or "cuda" in instructions.content.lower()
                )


class TestStrategyIntegration:
    """Integration tests for strategy selection and generation."""

    def test_strategy_selection_priority(self, mock_analysis):
        """Test strategy selection priority."""
        # Test strategy priorities
        docker = DockerStrategy()
        conda = CondaStrategy()
        venv = VenvStrategy()

        # Docker should have highest priority
        assert docker.priority > conda.priority
        assert conda.priority > venv.priority

        # Venv should have lowest priority
        assert venv.priority < conda.priority

    def test_all_strategies_generate_files(self, mock_analysis):
        """Test that all strategies generate some files."""
        strategies = [DockerStrategy(), CondaStrategy(), VenvStrategy()]

        for strategy in strategies:
            resolution = strategy.generate_solution(mock_analysis)
            assert len(resolution.generated_files) > 0
            assert all(f.path and f.content for f in resolution.generated_files)
