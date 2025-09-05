"""Integration tests for the complete repo-doctor workflow."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from repo_doctor.agents import AnalysisAgent, ProfileAgent, ResolutionAgent
from repo_doctor.models.analysis import (
    Analysis,
    DependencyInfo,
    DependencyType,
    RepositoryInfo,
)
from repo_doctor.models.resolution import (
    GeneratedFile,
    Resolution,
    Strategy,
    StrategyType,
)
from repo_doctor.utils.config import Config


class TestIntegration:
    """Integration tests for the full workflow."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Config.load()
        config.integrations.llm.enabled = False
        return config

    @pytest.fixture
    def mock_repo_data(self):
        """Create mock repository data."""
        return {
            "name": "test-repo",
            "owner": {"login": "test-owner"},
            "description": "A test repository",
            "stargazers_count": 100,
            "language": "Python",
            "default_branch": "main",
            "topics": ["machine-learning", "ai"],
        }

    @pytest.fixture
    def mock_file_contents(self):
        """Create mock file contents."""
        return {
            "requirements.txt": "torch>=2.0.0\nnumpy>=1.24.0\npandas>=2.0.0\n",
            "setup.py": """
from setuptools import setup

setup(
    name="test-package",
    install_requires=[
        "transformers>=4.30.0",
        "scikit-learn>=1.0.0"
    ],
    python_requires=">=3.8"
)
""",
            "README.md": """
# Test Repository

A machine learning project that requires:
- Python 3.8+
- CUDA 11.8 for GPU support
- At least 8GB RAM

## Installation
pip install -r requirements.txt
""",
        }

    def test_profile_agent(self):
        """Test ProfileAgent functionality."""
        agent = ProfileAgent()
        profile = agent.profile()

        assert profile is not None
        assert profile.platform in ["linux", "darwin", "windows"]
        assert profile.hardware.cpu_cores > 0
        assert profile.hardware.memory_gb > 0
        assert profile.software.python_version is not None

    @pytest.mark.asyncio
    async def test_analysis_agent(
        self, mock_config, mock_repo_data, mock_file_contents
    ):
        """Test AnalysisAgent functionality."""
        agent = AnalysisAgent(config=mock_config)

        # Mock GitHub API calls
        with patch.object(agent.github, "get_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.name = mock_repo_data["name"]
            mock_repo.owner.login = mock_repo_data["owner"]["login"]
            mock_repo.description = mock_repo_data["description"]
            mock_repo.stargazers_count = mock_repo_data["stargazers_count"]
            mock_repo.language = mock_repo_data["language"]
            mock_repo.default_branch = mock_repo_data["default_branch"]
            mock_repo.get_topics.return_value = mock_repo_data["topics"]

            # Mock file operations
            def get_contents(path):
                if path in mock_file_contents:
                    mock_content = Mock()
                    mock_content.decoded_content = mock_file_contents[path].encode()
                    return mock_content
                raise Exception("File not found")

            mock_repo.get_contents = get_contents
            mock_get_repo.return_value = mock_repo

            # Perform analysis
            analysis = await agent.analyze("https://github.com/test-owner/test-repo")

            assert analysis is not None
            assert analysis.repository.name == "test-repo"
            assert analysis.repository.owner == "test-owner"
            assert len(analysis.dependencies) > 0
            assert analysis.python_version_required is not None

    def test_resolution_agent(self, mock_config):
        """Test ResolutionAgent functionality."""
        # Create mock analysis
        analysis = Analysis(
            repository=RepositoryInfo(
                url="https://github.com/test/repo",
                name="test-repo",
                owner="test-owner",
                has_dockerfile=False,
                has_conda_env=False,
                has_requirements=True,
                has_setup_py=True,
                has_pyproject_toml=False,
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
            ],
            python_version_required="3.8",
            cuda_version_required="11.8",
            min_memory_gb=8.0,
            min_gpu_memory_gb=4.0,
            compatibility_issues=[],
            confidence_score=0.9,
        )

        agent = ResolutionAgent()

        # Test Docker strategy
        resolution = agent.resolve(analysis, strategy="docker")
        assert resolution is not None
        assert resolution.strategy.type == StrategyType.DOCKER
        assert len(resolution.generated_files) > 0

        # Check for Dockerfile
        dockerfile = resolution.get_file_by_name("Dockerfile")
        assert dockerfile is not None
        assert "FROM" in dockerfile.content

        # Check for docker-compose.yml
        compose = resolution.get_file_by_name("docker-compose.yml")
        if compose:
            assert "version" in compose.content or "services" in compose.content

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_config, mock_repo_data, mock_file_contents):
        """Test the complete pipeline from profiling to resolution."""
        # Step 1: Profile
        profile_agent = ProfileAgent()
        profile = profile_agent.profile()
        assert profile is not None

        # Step 2: Analysis (with mocked GitHub)
        analysis_agent = AnalysisAgent(config=mock_config)

        with patch.object(analysis_agent.github, "get_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.name = mock_repo_data["name"]
            mock_repo.owner.login = mock_repo_data["owner"]["login"]
            mock_repo.description = mock_repo_data["description"]
            mock_repo.stargazers_count = mock_repo_data["stargazers_count"]
            mock_repo.language = mock_repo_data["language"]
            mock_repo.default_branch = mock_repo_data["default_branch"]
            mock_repo.get_topics.return_value = mock_repo_data["topics"]

            def get_contents(path):
                if path in mock_file_contents:
                    mock_content = Mock()
                    mock_content.decoded_content = mock_file_contents[path].encode()
                    return mock_content
                raise Exception("File not found")

            mock_repo.get_contents = get_contents
            mock_get_repo.return_value = mock_repo

            analysis = await analysis_agent.analyze(
                "https://github.com/test-owner/test-repo"
            )
            assert analysis is not None

        # Step 3: Resolution
        resolution_agent = ResolutionAgent()
        resolution = resolution_agent.resolve(analysis, strategy="docker")

        assert resolution is not None
        assert len(resolution.generated_files) > 0
        assert resolution.instructions != ""

        # Verify we have actionable output
        has_dockerfile = any(
            f.path.endswith("Dockerfile") for f in resolution.generated_files
        )
        has_setup = any("setup" in f.path.lower() for f in resolution.generated_files)
        assert has_dockerfile or has_setup


class TestCLIIntegration:
    """Test CLI command integration."""

    @pytest.mark.asyncio
    async def test_check_command(self, tmp_path):
        """Test the check command workflow."""
        from repo_doctor.cli import _check_async

        # Mock the agents
        with patch("repo_doctor.cli.ProfileAgent") as MockProfile:
            with patch("repo_doctor.cli.AnalysisAgent") as MockAnalysis:
                with patch("repo_doctor.cli.ResolutionAgent") as MockResolution:
                    # Setup mocks
                    mock_profile = Mock()
                    mock_profile.platform = "linux"
                    mock_profile.hardware.cpu_cores = 8
                    mock_profile.hardware.memory_gb = 16.0
                    mock_profile.hardware.gpus = []
                    MockProfile.return_value.profile.return_value = mock_profile

                    mock_analysis = Mock()
                    mock_analysis.repository.name = "test-repo"
                    mock_analysis.dependencies = []
                    mock_analysis.python_version_required = "3.8"
                    mock_analysis.is_gpu_required.return_value = False
                    mock_analysis.get_critical_issues.return_value = []
                    MockAnalysis.return_value.analyze.return_value = mock_analysis

                    mock_resolution = Mock()
                    mock_resolution.strategy.type = StrategyType.DOCKER
                    mock_resolution.generated_files = [
                        GeneratedFile(
                            path="Dockerfile",
                            content="FROM python:3.8",
                            description="Docker container",
                            executable=False,
                        )
                    ]
                    mock_resolution.instructions = "Run docker build"
                    MockResolution.return_value.resolve.return_value = mock_resolution

                    # Run the command
                    output_dir = str(tmp_path / "output")
                    await _check_async(
                        repo_url="https://github.com/test/repo",
                        strategy="docker",
                        validate=False,
                        gpu_mode="flexible",
                        output=output_dir,
                        enable_llm=False,
                    )

                    # Verify the agents were called
                    assert MockProfile.return_value.profile.called
                    assert MockAnalysis.return_value.analyze.called
                    assert MockResolution.return_value.resolve.called
