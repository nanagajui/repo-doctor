"""Test agent contracts and data flow."""

import pytest
import time
from unittest.mock import Mock, patch

from repo_doctor.agents.contracts import (
    AgentContractValidator,
    AgentDataFlow,
    AgentErrorHandler,
    AgentPerformanceMonitor,
)
from repo_doctor.agents.profile import ProfileAgent
from repo_doctor.agents.analysis import AnalysisAgent
from repo_doctor.agents.resolution import ResolutionAgent
from repo_doctor.models.analysis import Analysis, RepositoryInfo, DependencyInfo, DependencyType, CompatibilityIssue
from repo_doctor.models.resolution import Resolution, Strategy, StrategyType, ValidationResult, ValidationStatus
from repo_doctor.models.system import SystemProfile, HardwareInfo, SoftwareStack, GPUInfo


class TestAgentContractValidator:
    """Test agent contract validation."""

    def test_validate_system_profile_success(self):
        """Test successful system profile validation."""
        profile = SystemProfile(
            hardware=HardwareInfo(
                cpu_cores=4,
                memory_gb=8.0,
                gpus=[GPUInfo(name="RTX 3080", memory_gb=10.0)],
                architecture="x86_64"
            ),
            software=SoftwareStack(
                python_version="3.9.0",
                pip_version="21.0.0",
                conda_version="4.10.0",
                docker_version="20.10.0",
                git_version="2.30.0",
                cuda_version="11.8"
            ),
            platform="linux",
            container_runtime="docker",
            compute_score=85.0
        )
        
        assert AgentContractValidator.validate_system_profile(profile) is True

    def test_validate_system_profile_failure(self):
        """Test system profile validation failure."""
        profile = SystemProfile(
            hardware=HardwareInfo(
                cpu_cores=0,  # Invalid
                memory_gb=8.0,
                gpus=[],
                architecture="x86_64"
            ),
            software=SoftwareStack(
                python_version="unknown",  # Invalid
                pip_version="21.0.0",
                conda_version="4.10.0",
                docker_version="20.10.0",
                git_version="2.30.0",
                cuda_version="11.8"
            ),
            platform="linux",
            container_runtime="docker",
            compute_score=150.0  # Invalid
        )
        
        with pytest.raises(ValueError, match="SystemProfile validation failed"):
            AgentContractValidator.validate_system_profile(profile)

    def test_validate_analysis_success(self):
        """Test successful analysis validation."""
        analysis = Analysis(
            repository=RepositoryInfo(
                url="https://github.com/test/repo",
                name="repo",
                owner="test",
                description="Test repository",
                stars=100,
                language="Python",
                topics=["ml", "ai"]
            ),
            dependencies=[
                DependencyInfo(
                    name="torch",
                    version="1.12.0",
                    type=DependencyType.PYTHON,
                    source="requirements.txt",
                    gpu_required=True
                )
            ],
            python_version_required="3.8",
            cuda_version_required="11.6",
            min_memory_gb=4.0,
            min_gpu_memory_gb=6.0,
            compatibility_issues=[
                CompatibilityIssue(
                    type="version_conflict",
                    severity="warning",
                    message="Version conflict detected",
                    component="torch"
                )
            ],
            analysis_time=2.5,
            confidence_score=0.85
        )
        
        assert AgentContractValidator.validate_analysis(analysis) is True

    def test_validate_analysis_failure(self):
        """Test analysis validation failure."""
        analysis = Analysis(
            repository=RepositoryInfo(
                url="",  # Invalid
                name="",  # Invalid
                owner="",  # Invalid
            ),
            dependencies=[
                DependencyInfo(
                    name="",  # Invalid
                    version="1.12.0",
                    type=DependencyType.PYTHON,
                    source="",  # Invalid
                    gpu_required=True
                )
            ],
            compatibility_issues=[
                CompatibilityIssue(
                    type="version_conflict",
                    severity="invalid",  # Invalid
                    message="",  # Invalid
                    component=""  # Invalid
                )
            ],
            analysis_time=-1.0,  # Invalid
            confidence_score=1.5  # Invalid
        )
        
        with pytest.raises(ValueError, match="Analysis validation failed"):
            AgentContractValidator.validate_analysis(analysis)

    def test_validate_resolution_success(self):
        """Test successful resolution validation."""
        resolution = Resolution(
            strategy=Strategy(
                type=StrategyType.DOCKER,
                priority=1,
                requirements={"gpu": True},
                can_handle_gpu=True,
                estimated_setup_time=300
            ),
            generated_files=[],
            setup_commands=["docker build -t test ."],
            instructions="Build and run the Docker container",
            estimated_size_mb=1024
        )
        
        assert AgentContractValidator.validate_resolution(resolution) is True

    def test_validate_resolution_failure(self):
        """Test resolution validation failure."""
        resolution = Resolution(
            strategy=Strategy(
                type=StrategyType.DOCKER,
                priority=1,
                requirements={"gpu": True},
                can_handle_gpu=True,
                estimated_setup_time=300
            ),
            generated_files=[],
            setup_commands=[""],  # Invalid empty command
            instructions="",  # Invalid empty instructions
            estimated_size_mb=-100  # Invalid negative size
        )
        
        with pytest.raises(ValueError, match="Resolution validation failed"):
            AgentContractValidator.validate_resolution(resolution)


class TestAgentDataFlow:
    """Test agent data flow utilities."""

    def test_profile_to_analysis_context(self):
        """Test profile to analysis context conversion."""
        profile = SystemProfile(
            hardware=HardwareInfo(
                cpu_cores=4,
                memory_gb=8.0,
                gpus=[GPUInfo(name="RTX 3080", memory_gb=10.0)],
                architecture="x86_64"
            ),
            software=SoftwareStack(
                python_version="3.9.0",
                pip_version="21.0.0",
                conda_version="4.10.0",
                docker_version="20.10.0",
                git_version="2.30.0",
                cuda_version="11.8"
            ),
            platform="linux",
            container_runtime="docker",
            compute_score=85.0
        )
        
        context = AgentDataFlow.profile_to_analysis_context(profile)
        
        assert context["system_capabilities"]["has_gpu"] is True
        assert context["system_capabilities"]["has_cuda"] is True
        assert context["system_capabilities"]["can_run_containers"] is True
        assert context["system_capabilities"]["compute_score"] == 85.0
        assert context["system_capabilities"]["gpu_memory_gb"] == 10.0
        assert context["system_capabilities"]["cuda_version"] == "11.8"
        assert context["hardware"]["cpu_cores"] == 4
        assert context["hardware"]["memory_gb"] == 8.0
        assert context["software"]["python_version"] == "3.9.0"

    def test_analysis_to_resolution_context(self):
        """Test analysis to resolution context conversion."""
        analysis = Analysis(
            repository=RepositoryInfo(
                url="https://github.com/test/repo",
                name="repo",
                owner="test",
                language="Python",
                has_dockerfile=True,
                has_conda_env=False
            ),
            dependencies=[
                DependencyInfo(
                    name="torch",
                    version="1.12.0",
                    type=DependencyType.PYTHON,
                    source="requirements.txt",
                    gpu_required=True
                )
            ],
            python_version_required="3.8",
            cuda_version_required="11.6",
            min_memory_gb=4.0,
            min_gpu_memory_gb=6.0,
            compatibility_issues=[
                CompatibilityIssue(
                    type="version_conflict",
                    severity="warning",
                    message="Version conflict detected",
                    component="torch"
                )
            ],
            analysis_time=2.5,
            confidence_score=0.85
        )
        
        context = AgentDataFlow.analysis_to_resolution_context(analysis)
        
        assert context["repository"]["name"] == "repo"
        assert context["repository"]["owner"] == "test"
        assert context["repository"]["has_dockerfile"] is True
        assert context["requirements"]["python_version"] == "3.8"
        assert context["requirements"]["cuda_version"] == "11.6"
        assert context["requirements"]["gpu_required"] is True
        assert len(context["dependencies"]) == 1
        assert context["dependencies"][0]["name"] == "torch"
        assert len(context["compatibility_issues"]) == 1
        assert context["confidence_score"] == 0.85

    def test_resolution_to_knowledge_context(self):
        """Test resolution to knowledge context conversion."""
        analysis = Analysis(
            repository=RepositoryInfo(
                url="https://github.com/test/repo",
                name="repo",
                owner="test"
            ),
            dependencies=[],
            compatibility_issues=[],
            analysis_time=2.5,
            confidence_score=0.85
        )
        
        resolution = Resolution(
            strategy=Strategy(
                type=StrategyType.DOCKER,
                priority=1,
                requirements={"gpu": True},
                can_handle_gpu=True,
                estimated_setup_time=300
            ),
            generated_files=[],
            setup_commands=["docker build -t test ."],
            instructions="Build and run the Docker container",
            estimated_size_mb=1024,
            validation_result=ValidationResult(
                status=ValidationStatus.SUCCESS,
                duration=45.0,
                logs=["Build successful"]
            )
        )
        
        context = AgentDataFlow.resolution_to_knowledge_context(resolution, analysis)
        
        assert context["repository_key"] == "test/repo"
        assert context["strategy_used"] == "docker"
        assert context["success"] is True
        assert context["files_generated"] == 0
        assert context["setup_commands"] == 1
        assert context["estimated_size_mb"] == 1024
        assert context["analysis_confidence"] == 0.85
        assert context["compatibility_issues_count"] == 0


class TestAgentErrorHandler:
    """Test agent error handling."""

    def test_handle_profile_error(self):
        """Test profile error handling with fallback."""
        error = Exception("GPU detection failed")
        
        profile = AgentErrorHandler.handle_profile_error(error, "gpu_detection")
        
        assert profile.hardware.cpu_cores == 1
        assert profile.hardware.memory_gb == 4.0
        assert profile.hardware.architecture == "unknown"
        assert profile.software.python_version == "unknown"
        assert profile.compute_score == 0.0

    def test_handle_analysis_error(self):
        """Test analysis error handling with fallback."""
        error = Exception("GitHub API failed")
        repo_url = "https://github.com/test/repo"
        
        analysis = AgentErrorHandler.handle_analysis_error(error, repo_url, "github_api")
        
        assert analysis.repository.url == repo_url
        assert analysis.repository.name == "repo"
        assert analysis.repository.owner == "test"
        assert len(analysis.compatibility_issues) == 1
        assert analysis.compatibility_issues[0].type == "analysis_error"
        assert analysis.compatibility_issues[0].severity == "warning"
        assert analysis.confidence_score == 0.0

    def test_handle_resolution_error(self):
        """Test resolution error handling."""
        error = Exception("Strategy selection failed")
        analysis = Analysis(
            repository=RepositoryInfo(
                url="https://github.com/test/repo",
                name="repo",
                owner="test"
            ),
            dependencies=[],
            compatibility_issues=[],
            analysis_time=2.5,
            confidence_score=0.85
        )
        
        with pytest.raises(ValueError, match="Failed to generate resolution for test/repo"):
            AgentErrorHandler.handle_resolution_error(error, analysis, "strategy_selection")


class TestAgentPerformanceMonitor:
    """Test agent performance monitoring."""

    def test_performance_targets(self):
        """Test performance target definitions."""
        monitor = AgentPerformanceMonitor()
        
        assert monitor.performance_targets["profile_agent"] == 2.0
        assert monitor.performance_targets["analysis_agent"] == 10.0
        assert monitor.performance_targets["resolution_agent"] == 5.0

    def test_check_profile_performance(self):
        """Test profile agent performance checking."""
        monitor = AgentPerformanceMonitor()
        
        assert monitor.check_profile_performance(1.5) is True
        assert monitor.check_profile_performance(2.5) is False

    def test_check_analysis_performance(self):
        """Test analysis agent performance checking."""
        monitor = AgentPerformanceMonitor()
        
        assert monitor.check_analysis_performance(8.0) is True
        assert monitor.check_analysis_performance(12.0) is False

    def test_check_resolution_performance(self):
        """Test resolution agent performance checking."""
        monitor = AgentPerformanceMonitor()
        
        assert monitor.check_resolution_performance(3.0) is True
        assert monitor.check_resolution_performance(6.0) is False

    def test_get_performance_report(self):
        """Test performance report generation."""
        monitor = AgentPerformanceMonitor()
        
        report = monitor.get_performance_report("profile_agent", 1.5)
        
        assert report["agent"] == "profile_agent"
        assert report["duration"] == 1.5
        assert report["target"] == 2.0
        assert report["meets_target"] is True
        assert report["performance_ratio"] == 0.75


class TestAgentIntegration:
    """Test agent integration and data flow."""

    @pytest.fixture
    def mock_system_profile(self):
        """Create a mock system profile for testing."""
        return SystemProfile(
            hardware=HardwareInfo(
                cpu_cores=4,
                memory_gb=8.0,
                gpus=[GPUInfo(name="RTX 3080", memory_gb=10.0)],
                architecture="x86_64"
            ),
            software=SoftwareStack(
                python_version="3.9.0",
                pip_version="21.0.0",
                conda_version="4.10.0",
                docker_version="20.10.0",
                git_version="2.30.0",
                cuda_version="11.8"
            ),
            platform="linux",
            container_runtime="docker",
            compute_score=85.0
        )

    @pytest.fixture
    def mock_analysis(self):
        """Create a mock analysis for testing."""
        return Analysis(
            repository=RepositoryInfo(
                url="https://github.com/test/repo",
                name="repo",
                owner="test",
                description="Test repository",
                stars=100,
                language="Python",
                topics=["ml", "ai"]
            ),
            dependencies=[
                DependencyInfo(
                    name="torch",
                    version="1.12.0",
                    type=DependencyType.PYTHON,
                    source="requirements.txt",
                    gpu_required=True
                )
            ],
            python_version_required="3.8",
            cuda_version_required="11.6",
            min_memory_gb=4.0,
            min_gpu_memory_gb=6.0,
            compatibility_issues=[],
            analysis_time=2.5,
            confidence_score=0.85
        )

    def test_profile_agent_contract_compliance(self, mock_system_profile):
        """Test that ProfileAgent follows contracts."""
        with patch('repo_doctor.agents.profile.psutil') as mock_psutil:
            mock_psutil.cpu_count.return_value = 4
            mock_psutil.virtual_memory.return_value.total = 8 * 1024**3
            
            agent = ProfileAgent()
            profile = agent.profile()
            
            # Should be valid according to contracts
            AgentContractValidator.validate_system_profile(profile)

    def test_analysis_agent_contract_compliance(self, mock_analysis):
        """Test that AnalysisAgent follows contracts."""
        with patch('repo_doctor.agents.analysis.GitHubHelper') as mock_github:
            mock_github.return_value.parse_repo_url.return_value = {
                "owner": "test", "name": "repo"
            }
            mock_github.return_value.get_repo_info.return_value = {
                "name": "repo", "description": "Test", "stars": 100,
                "language": "Python", "topics": ["ml"]
            }
            
            agent = AnalysisAgent()
            
            # Mock the async methods
            with patch.object(agent, '_analyze_dependencies', return_value=[]):
                with patch.object(agent, '_check_dockerfiles', return_value={}):
                    with patch.object(agent, '_scan_documentation', return_value={}):
                        with patch.object(agent, '_check_ci_configs', return_value={}):
                            import asyncio
                            analysis = asyncio.run(agent.analyze("https://github.com/test/repo"))
                            
                            # Should be valid according to contracts
                            AgentContractValidator.validate_analysis(analysis)

    def test_resolution_agent_contract_compliance(self, mock_analysis):
        """Test that ResolutionAgent follows contracts."""
        with patch('repo_doctor.agents.resolution.DockerStrategy') as mock_strategy:
            mock_strategy.return_value.can_handle.return_value = True
            mock_strategy.return_value.priority = 1
            mock_strategy.return_value.strategy_type = StrategyType.DOCKER
            mock_strategy.return_value.generate_solution.return_value = Resolution(
                strategy=Strategy(
                    type=StrategyType.DOCKER,
                    priority=1,
                    requirements={"gpu": True},
                    can_handle_gpu=True,
                    estimated_setup_time=300
                ),
                generated_files=[],
                setup_commands=["docker build -t test ."],
                instructions="Build and run the Docker container",
                estimated_size_mb=1024
            )
            
            agent = ResolutionAgent()
            
            resolution = agent.resolve(mock_analysis, "docker")
            
            # Should be valid according to contracts
            AgentContractValidator.validate_resolution(resolution)

    def test_agent_data_flow_integration(self, mock_system_profile, mock_analysis):
        """Test complete data flow between agents."""
        # Profile to Analysis context
        profile_context = AgentDataFlow.profile_to_analysis_context(mock_system_profile)
        assert profile_context["system_capabilities"]["has_gpu"] is True
        
        # Analysis to Resolution context
        analysis_context = AgentDataFlow.analysis_to_resolution_context(mock_analysis)
        assert analysis_context["repository"]["name"] == "repo"
        assert analysis_context["requirements"]["gpu_required"] is True
        
        # Resolution to Knowledge context
        resolution = Resolution(
            strategy=Strategy(
                type=StrategyType.DOCKER,
                priority=1,
                requirements={"gpu": True},
                can_handle_gpu=True,
                estimated_setup_time=300
            ),
            generated_files=[],
            setup_commands=["docker build -t test ."],
            instructions="Build and run the Docker container",
            estimated_size_mb=1024
        )
        
        knowledge_context = AgentDataFlow.resolution_to_knowledge_context(resolution, mock_analysis)
        assert knowledge_context["repository_key"] == "test/repo"
        assert knowledge_context["strategy_used"] == "docker"
        assert knowledge_context["analysis_confidence"] == 0.85
