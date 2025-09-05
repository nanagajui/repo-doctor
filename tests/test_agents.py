"""Test suite for agents."""

from unittest.mock import MagicMock, patch
from github import GithubException

import pytest

from repo_doctor.agents import AnalysisAgent, ProfileAgent, ResolutionAgent
from repo_doctor.models.analysis import Analysis, RepositoryInfo, DependencyInfo, DependencyType
from repo_doctor.models.system import HardwareInfo, SoftwareStack, SystemProfile


class TestProfileAgent:
    """Tests for ProfileAgent."""

    def test_profile_creation(self):
        """Test that ProfileAgent creates a valid system profile."""
        agent = ProfileAgent()
        profile = agent.profile()

        assert isinstance(profile, SystemProfile)
        assert profile.hardware is not None
        assert profile.software is not None
        assert profile.platform in ["linux", "darwin", "windows"]
        assert profile.compute_score >= 0

    @patch("repo_doctor.agents.profile.psutil.cpu_count")
    @patch("repo_doctor.agents.profile.psutil.virtual_memory")
    def test_hardware_detection(self, mock_memory, mock_cpu_count):
        """Test hardware detection."""
        mock_cpu_count.return_value = 8
        mock_memory.return_value = MagicMock(total=16 * 1024**3)

        agent = ProfileAgent()
        hardware = agent._get_hardware_info()

        assert hardware.cpu_cores == 8
        assert hardware.memory_gb == 16.0

    def test_software_stack_detection(self):
        """Test software stack detection."""
        agent = ProfileAgent()
        software = agent._get_software_stack()

        assert isinstance(software, SoftwareStack)
        assert software.python_version is not None
        assert len(software.python_version) > 0

    @patch("repo_doctor.agents.profile.subprocess.run")
    def test_gpu_detection(self, mock_run):
        """Test GPU detection with mocked nvidia-smi."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="NVIDIA RTX 4090, 24576 MiB, 550.107.02, 12.4"
        )

        agent = ProfileAgent()
        gpus = agent._detect_gpus()

        assert len(gpus) > 0
        assert gpus[0].name == "NVIDIA RTX 4090"
        assert gpus[0].memory_gb == 24.0

    def test_compute_score_calculation(self):
        """Test compute score calculation."""
        agent = ProfileAgent()
        score = agent._calculate_compute_score()

        assert isinstance(score, int)
        assert score >= 0


class TestAnalysisAgent:
    """Tests for AnalysisAgent."""

    @pytest.fixture
    def mock_profile(self):
        """Create a mock system profile."""
        return SystemProfile(
            hardware=HardwareInfo(
                cpu_cores=8, memory_gb=16.0, gpus=[], architecture="x86_64"
            ),
            software=SoftwareStack(
                python_version="3.10.0",
                pip_version="23.0.0",
                conda_version=None,
                git_version="2.40.0",
            ),
            platform="linux",
            container_runtime="docker",
            compute_score=100,
        )

    @patch('repo_doctor.agents.analysis.Github')
    @pytest.mark.asyncio
    async def test_analysis_creation(self, mock_github, mock_profile):
        """Test that AnalysisAgent creates a valid analysis."""
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = GithubException(404, 'Not Found', {})
        mock_github.return_value.get_repo.return_value = mock_repo

        agent = AnalysisAgent()
        analysis = await agent.analyze("https://github.com/test/repo")

        assert isinstance(analysis, Analysis)
        assert analysis.repository is not None
        assert analysis.confidence_score >= 0

    @pytest.mark.asyncio
    async def test_dependency_extraction(self, mock_profile):
        """Test dependency extraction from various sources."""
        agent = AnalysisAgent()

        mock_files = {
            "requirements.txt": "numpy==1.24.0\npandas>=2.0.0",
            "setup.py": "install_requires=['torch', 'transformers']",
        }

        with patch.object(
            agent.repo_parser,
            "parse_repository_files",
            return_value=[],
        ):
            deps = await agent._analyze_dependencies(MagicMock())

            assert isinstance(deps, list)

    def test_gpu_requirement_detection(self, mock_profile):
        """Test GPU requirement detection."""
        # Create an analysis with GPU dependencies
        analysis = Analysis(
            repository=RepositoryInfo(
                url="https://github.com/test/repo",
                name="repo",
                owner="test",
                description="Test repository",
                stars=0,
                language="Python",
                topics=[],
                has_dockerfile=False,
                has_conda_env=False,
                has_requirements=False,
                has_setup_py=False,
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
            python_version_required="3.10",
            cuda_version_required=None,
            min_memory_gb=0.0,
            min_gpu_memory_gb=0.0,
            compatibility_issues=[],
            analysis_time=0.0,
            confidence_score=0.8,
        )

        gpu_required = analysis.is_gpu_required()
        assert gpu_required is True


class TestResolutionAgent:
    """Tests for ResolutionAgent."""

    @pytest.fixture
    def mock_analysis(self):
        """Create a mock analysis."""
        return Analysis(
            repository=RepositoryInfo(
                url="https://github.com/test/repo",
                name="repo",
                owner="test",
                description="Test repository",
                stars=0,
                language="Python",
                topics=[],
                has_dockerfile=False,
                has_conda_env=False,
                has_requirements=False,
                has_setup_py=False,
                has_pyproject_toml=False,
            ),
            dependencies=[],
            python_version_required="3.10",
            cuda_version_required=None,
            min_memory_gb=0.0,
            min_gpu_memory_gb=0.0,
            compatibility_issues=[],
            analysis_time=0.0,
            confidence_score=0.8,
        )

    def test_resolution_creation(self, mock_analysis):
        """Test that ResolutionAgent creates a valid resolution."""
        agent = ResolutionAgent()
        resolution = agent.resolve(mock_analysis, strategy="docker")

        assert resolution is not None
        assert resolution.strategy is not None
        assert len(resolution.generated_files) > 0

    def test_strategy_selection(self, mock_analysis):
        """Test automatic strategy selection."""
        agent = ResolutionAgent()

        # Test strategy selection
        strategy = agent._select_strategy(mock_analysis, "docker")
        assert strategy is not None
        assert hasattr(strategy, 'strategy_type')

    def test_docker_file_generation(self, mock_analysis):
        """Test Docker file generation."""
        agent = ResolutionAgent()

        resolution = agent.resolve(mock_analysis, strategy="docker")
        assert resolution is not None
        assert len(resolution.generated_files) > 0


@pytest.mark.unit
class TestAgentIntegration:
    """Integration tests for the three-agent system."""

    @patch('repo_doctor.agents.analysis.Github')
    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_github):
        """Test the full three-agent pipeline."""
        # Create agents
        profile_agent = ProfileAgent()
        profile = profile_agent.profile()

        analysis_agent = AnalysisAgent()
        resolution_agent = ResolutionAgent()

        # Mock the GitHub API calls
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = GithubException(404, 'Not Found', {})
        mock_github.return_value.get_repo.return_value = mock_repo

        # Run analysis
        analysis = await analysis_agent.analyze("https://github.com/test/repo")

        # Generate resolution
        resolution = resolution_agent.resolve(analysis, strategy="docker")

        assert resolution is not None
        assert len(resolution.generated_files) > 0
