"""Test suite for data models."""

from datetime import datetime

import pytest

from repo_doctor.models.analysis import (
    Analysis,
    CompatibilityIssue,
    DependencyInfo,
    RepositoryInfo,
)
from repo_doctor.models.resolution import GeneratedFile, Resolution, ValidationResult
from repo_doctor.models.system import (
    GPUInfo,
    HardwareInfo,
    SoftwareStack,
    SystemProfile,
)


class TestSystemModels:
    """Tests for system-related models."""

    def test_gpu_info_creation(self):
        """Test GPUInfo model creation."""
        gpu = GPUInfo(
            name="NVIDIA RTX 4090",
            memory_gb=24.0,
            driver_version="550.107.02",
            cuda_version="12.4",
            compute_capability="8.9",
        )

        assert gpu.name == "NVIDIA RTX 4090"
        assert gpu.memory_gb == 24.0
        assert gpu.driver_version == "550.107.02"
        assert gpu.cuda_version == "12.4"
        assert gpu.compute_capability == "8.9"

    def test_hardware_info_creation(self):
        """Test HardwareInfo model creation."""
        hardware = HardwareInfo(
            cpu_cores=16, memory_gb=32.0, gpus=[], architecture="x86_64"
        )

        assert hardware.cpu_cores == 16
        assert hardware.memory_gb == 32.0
        assert hardware.architecture == "x86_64"
        assert len(hardware.gpus) == 0

    def test_software_stack_creation(self):
        """Test SoftwareStack model creation."""
        software = SoftwareStack(
            python_version="3.10.12",
            pip_version="23.2.1",
            conda_version="23.7.4",
            git_version="2.40.1",
        )

        assert software.python_version == "3.10.12"
        assert software.pip_version == "23.2.1"
        assert software.conda_version == "23.7.4"
        assert software.git_version == "2.40.1"

    def test_system_profile_creation(self):
        """Test SystemProfile model creation."""
        profile = SystemProfile(
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

        assert profile.platform == "linux"
        assert profile.container_runtime == "docker"
        assert profile.compute_score == 100
        assert profile.hardware.cpu_cores == 8


class TestAnalysisModels:
    """Tests for analysis-related models."""

    def test_dependency_info_creation(self):
        """Test DependencyInfo model creation."""
        from repo_doctor.models.analysis import DependencyType

        dep = DependencyInfo(
            name="torch",
            version=">=2.0.0",
            type=DependencyType.PYTHON,
            source="requirements.txt",
            optional=False,
            gpu_required=True,
        )

        assert dep.name == "torch"
        assert dep.version == ">=2.0.0"
        assert dep.type == DependencyType.PYTHON
        assert dep.source == "requirements.txt"
        assert dep.optional is False
        assert dep.gpu_required is True

    def test_repository_info_creation(self):
        """Test RepositoryInfo model creation."""
        repo = RepositoryInfo(
            url="https://github.com/test/repo",
            name="repo",
            owner="test",
            description="Test repository",
            stars=100,
            language="Python",
            topics=["ml", "ai"],
            has_dockerfile=True,
            has_conda_env=True,
            has_requirements=True,
            has_setup_py=False,
            has_pyproject_toml=False,
        )

        assert repo.url == "https://github.com/test/repo"
        assert repo.name == "repo"
        assert repo.owner == "test"
        assert repo.description == "Test repository"
        assert repo.stars == 100
        assert repo.has_dockerfile is True
        assert repo.has_conda_env is True

    def test_compatibility_issue_creation(self):
        """Test CompatibilityIssue model creation."""
        issue = CompatibilityIssue(
            type="version_conflict",
            severity="warning",
            message="CUDA version mismatch",
            component="cuda",
            suggested_fix="Update CUDA to version 12.0",
        )

        assert issue.type == "version_conflict"
        assert issue.severity == "warning"
        assert issue.message == "CUDA version mismatch"
        assert issue.component == "cuda"
        assert issue.suggested_fix == "Update CUDA to version 12.0"

    def test_analysis_creation(self):
        """Test Analysis model creation."""
        from repo_doctor.models.analysis import DependencyType

        analysis = Analysis(
            repository=RepositoryInfo(
                url="https://github.com/test/repo",
                name="repo",
                owner="test",
                has_dockerfile=False,
                has_conda_env=False,
                has_requirements=True,
                has_setup_py=False,
                has_pyproject_toml=False,
            ),
            dependencies=[
                DependencyInfo(
                    name="numpy",
                    version=">=1.24.0",
                    type=DependencyType.PYTHON,
                    source="requirements.txt",
                    optional=False,
                    gpu_required=False,
                )
            ],
            python_version_required="3.10",
            cuda_version_required=None,
            min_memory_gb=8.0,
            min_gpu_memory_gb=0.0,
            compatibility_issues=[],
            analysis_time=2.5,
            confidence_score=0.85,
        )

        assert analysis.python_version_required == "3.10"
        assert analysis.is_gpu_required() is False
        assert analysis.confidence_score == 0.85
        assert len(analysis.dependencies) == 1
        assert analysis.min_memory_gb == 8.0


class TestResolutionModels:
    """Tests for resolution-related models."""

    def test_generated_file_creation(self):
        """Test GeneratedFile model creation."""
        file = GeneratedFile(
            path="Dockerfile",
            content="FROM python:3.10\nRUN pip install numpy",
            description="Docker container configuration",
            executable=False,
        )

        assert file.path == "Dockerfile"
        assert "FROM python:3.10" in file.content
        assert file.description == "Docker container configuration"
        assert file.executable is False

    def test_validation_result_creation(self):
        """Test ValidationResult model creation."""
        from repo_doctor.models.resolution import ValidationStatus

        result = ValidationResult(
            status=ValidationStatus.SUCCESS,
            duration=45.2,
            logs=["Building container...", "Running tests...", "All tests passed"],
            error_message=None,
            container_id="abc123",
        )

        assert result.status == ValidationStatus.SUCCESS
        assert result.duration == 45.2
        assert len(result.logs) == 3
        assert result.error_message is None
        assert result.container_id == "abc123"

    def test_resolution_creation(self):
        """Test Resolution model creation."""
        from repo_doctor.models.resolution import Strategy, StrategyType

        strategy = Strategy(
            type=StrategyType.DOCKER,
            priority=1,
            requirements={"docker": ">=20.0"},
            can_handle_gpu=True,
            estimated_setup_time=120,
        )

        resolution = Resolution(
            strategy=strategy,
            generated_files=[
                GeneratedFile(
                    path="Dockerfile",
                    content="FROM python:3.10",
                    description="Container config",
                    executable=False,
                )
            ],
            setup_commands=["docker build -t app .", "docker run app"],
            validation_result=None,
            instructions="Build and run the Docker container",
            estimated_size_mb=500,
        )

        assert resolution.strategy.type == StrategyType.DOCKER
        assert len(resolution.generated_files) == 1
        assert len(resolution.setup_commands) == 2
        assert resolution.instructions != ""
        assert resolution.estimated_size_mb == 500

    def test_resolution_with_validation(self):
        """Test Resolution model with validation result."""
        from repo_doctor.models.resolution import (
            Strategy,
            StrategyType,
            ValidationStatus,
        )

        validation = ValidationResult(
            status=ValidationStatus.SUCCESS,
            duration=30.0,
            logs=["Build successful"],
            error_message=None,
            container_id="xyz789",
        )

        strategy = Strategy(type=StrategyType.DOCKER, priority=1, can_handle_gpu=False)

        resolution = Resolution(
            strategy=strategy,
            generated_files=[],
            setup_commands=[],
            validation_result=validation,
            instructions="Environment validated successfully",
        )

        assert resolution.validation_result is not None
        assert resolution.validation_result.status == ValidationStatus.SUCCESS
        assert resolution.validation_result.duration == 30.0
        assert resolution.is_validated() is True
