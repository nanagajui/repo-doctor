"""Tests for feature extraction module."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Any, Dict, List

from repo_doctor.learning.feature_extractor import FeatureExtractor
from repo_doctor.models.analysis import Analysis, DependencyInfo, DependencyType
from repo_doctor.models.resolution import Resolution, StrategyType
from repo_doctor.models.system import SystemProfile, HardwareInfo, SoftwareStack, GPUInfo


class TestFeatureExtractor:
    """Test FeatureExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()

    def test_init(self):
        """Test FeatureExtractor initialization."""
        assert isinstance(self.extractor.ml_dependencies, set)
        assert isinstance(self.extractor.gpu_dependencies, set)
        assert "torch" in self.extractor.ml_dependencies
        assert "tensorflow" in self.extractor.gpu_dependencies

    def test_extract_repository_features_basic(self):
        """Test basic repository feature extraction."""
        # Create mock analysis
        analysis = Mock(spec=Analysis)
        analysis.dependencies = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test", gpu_required=True),
            DependencyInfo(name="numpy", version="1.20.0", type=DependencyType.PYTHON, source="test", gpu_required=False),
        ]
        analysis.compatibility_issues = []
        analysis.get_critical_issues = Mock(return_value=[])
        analysis.get_warning_issues = Mock(return_value=[])
        analysis.confidence_score = 0.85
        
        # Mock repository
        repo = Mock()
        repo.size = 1000
        repo.language = "Python"
        repo.has_dockerfile = True
        repo.has_conda_env = False
        repo.star_count = 100
        repo.fork_count = 20
        repo.topics = ["machine-learning", "pytorch"]
        repo.name = "test-repo"
        repo.description = "A test ML repository"
        repo.has_tests = True
        repo.has_ci = True
        analysis.repository = repo
        
        features = self.extractor.extract_repository_features(analysis)
        
        assert features["repo_size"] == 1000
        assert features["language"] == "Python"
        assert features["has_dockerfile"] is True
        assert features["has_conda_env"] is False
        assert features["star_count"] == 100
        assert features["fork_count"] == 20
        assert features["topics_count"] == 2
        assert features["total_dependencies"] == 2
        assert features["gpu_dependencies"] == 1
        assert features["ml_dependencies"] == 1
        assert features["is_ml_repo"] is False  # Needs >= 2 ML deps
        assert features["is_research_repo"] is False

    def test_extract_repository_features_with_issues(self):
        """Test repository feature extraction with compatibility issues."""
        analysis = Mock(spec=Analysis)
        analysis.dependencies = []
        
        # Mock issues with string descriptions
        critical_issue = Mock()
        critical_issue.description = "Critical GPU compatibility issue"
        warning_issue = Mock()
        warning_issue.description = "Warning CUDA version issue"
        
        analysis.compatibility_issues = [critical_issue, warning_issue]
        analysis.get_critical_issues = Mock(return_value=[critical_issue])
        analysis.get_warning_issues = Mock(return_value=[warning_issue])
        analysis.repository = None
        
        features = self.extractor.extract_repository_features(analysis)
        
        assert features["critical_issues"] == 1
        assert features["warning_issues"] == 1
        assert features["gpu_issues"] == 2  # Both issues contain GPU/CUDA keywords
        assert features["cuda_version_conflicts"] == 1  # One issue contains CUDA
        assert features["python_version_conflicts"] == 0

    def test_extract_repository_features_mock_tolerant(self):
        """Test feature extraction handles mocks gracefully."""
        analysis = Mock(spec=Analysis)
        # Set up mock to avoid iteration errors
        analysis.dependencies = []
        analysis.compatibility_issues = []
        analysis.repository = None
        analysis.get_critical_issues = Mock(return_value=[])
        analysis.get_warning_issues = Mock(return_value=[])
        analysis.confidence_score = 0.5
        
        features = self.extractor.extract_repository_features(analysis)
        
        # Should not crash and return default values
        assert isinstance(features, dict)
        assert features["total_dependencies"] == 0
        assert features["critical_issues"] == 0

    def test_extract_system_features(self):
        """Test system feature extraction."""
        # Create mock system profile
        gpu1 = GPUInfo(name="NVIDIA RTX 3080", memory_gb=10.0)
        gpu2 = GPUInfo(name="AMD RX 6800", memory_gb=16.0)
        
        hardware = HardwareInfo(
            cpu_cores=8,
            memory_gb=32.0,
            gpus=[gpu1, gpu2],
            architecture="x86_64"
        )
        
        software = SoftwareStack(
            python_version="3.9.7",
            cuda_version="11.4"
        )
        
        profile = SystemProfile(
            hardware=hardware,
            software=software,
            container_runtime="docker",
            compute_score=85.5
        )
        
        features = self.extractor.extract_system_features(profile)
        
        assert features["cpu_cores"] == 8
        assert features["memory_gb"] == 32.0
        assert features["gpu_count"] == 2
        assert features["gpu_memory_total"] == 26.0
        assert features["gpu_memory_max"] == 16.0
        assert features["cuda_version"] == 11.4
        assert features["python_version"] == 3.9
        assert features["container_runtime"] == "docker"
        assert features["compute_score"] == 85.5
        assert features["has_nvidia_gpu"] is True
        assert features["has_amd_gpu"] is True

    def test_extract_system_features_no_gpus(self):
        """Test system feature extraction with no GPUs."""
        hardware = HardwareInfo(cpu_cores=4, memory_gb=16.0, gpus=[], architecture="x86_64")
        software = SoftwareStack(python_version="3.8.10", cuda_version="")
        profile = SystemProfile(hardware=hardware, software=software, 
                              container_runtime="podman", compute_score=45.0)
        
        features = self.extractor.extract_system_features(profile)
        
        assert features["gpu_count"] == 0
        assert features["gpu_memory_total"] == 0
        assert features["gpu_memory_max"] == 0
        assert features["cuda_version"] == 0.0
        assert features["has_nvidia_gpu"] is False
        assert features["has_amd_gpu"] is False

    def test_extract_resolution_features(self):
        """Test resolution feature extraction."""
        # Mock resolution
        resolution = Mock(spec=Resolution)
        resolution.strategy = Mock()
        resolution.strategy.type = StrategyType.DOCKER
        resolution.strategy.requirements = {
            "estimated_setup_time": 300,
            "requires_gpu": True,
            "requires_cuda": True
        }
        
        # Mock generated files
        dockerfile = Mock()
        dockerfile.name = "Dockerfile"
        dockerfile.content = "FROM nvidia/cuda:11.4-runtime-ubuntu20.04"
        
        compose_file = Mock()
        compose_file.name = "docker-compose.yml"
        compose_file.content = "services:\n  app:\n    deploy:\n      resources:\n        reservations:\n          devices:\n            - driver: nvidia"
        
        resolution.generated_files = [dockerfile, compose_file]
        resolution.setup_commands = ["docker build", "docker run"]
        resolution.estimated_size_mb = 2048
        
        features = self.extractor.extract_resolution_features(resolution)
        
        assert features["strategy_type"] == "docker"
        assert features["files_generated"] == 2
        assert features["setup_commands"] == 2
        assert features["estimated_size_mb"] == 2048
        assert features["estimated_setup_time"] == 300
        assert features["requires_gpu"] is True
        assert features["requires_cuda"] is True
        assert features["docker_base_image"] == "nvidia_cuda"
        assert features["has_gpu_support"] is True

    def test_extract_learning_features(self):
        """Test learning feature extraction."""
        analysis = Mock(spec=Analysis)
        analysis.confidence_score = 0.92
        analysis.dependencies = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test", gpu_required=True),
            DependencyInfo(name="transformers", version="4.12.0", type=DependencyType.PYTHON, source="test", gpu_required=False),
            DependencyInfo(name="datasets", version="1.15.0", type=DependencyType.PYTHON, source="test", gpu_required=False),
        ]
        analysis.compatibility_issues = []
        analysis.is_gpu_required.return_value = True
        
        resolution = Mock(spec=Resolution)
        
        # Mock outcome with success
        outcome = Mock()
        outcome.status = Mock()
        outcome.status.value = "SUCCESS"
        outcome.duration = 120
        outcome.error_message = ""
        
        features = self.extractor.extract_learning_features(analysis, resolution, outcome)
        
        assert features["success"] is True
        assert features["duration"] == 120
        assert features["error_type"] == "no_error"
        assert features["confidence_score"] == 0.92
        assert features["similarity_to_known"] == 0.8  # 3 ML deps, 1 GPU dep
        assert features["complexity_score"] > 0

    def test_count_gpu_dependencies(self):
        """Test GPU dependency counting."""
        deps = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test", gpu_required=True),
            DependencyInfo(name="numpy", version="1.20.0", type=DependencyType.PYTHON, source="test", gpu_required=False),
            DependencyInfo(name="cupy", version="9.0.0", type=DependencyType.PYTHON, source="test", gpu_required=True),
        ]
        
        count = self.extractor._count_gpu_dependencies(deps)
        assert count == 2

    def test_count_ml_dependencies(self):
        """Test ML dependency counting."""
        deps = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test"),
            DependencyInfo(name="tensorflow", version="2.6.0", type=DependencyType.PYTHON, source="test"),
            DependencyInfo(name="numpy", version="1.20.0", type=DependencyType.PYTHON, source="test"),
            DependencyInfo(name="transformers", version="4.12.0", type=DependencyType.PYTHON, source="test"),
        ]
        
        count = self.extractor._count_ml_dependencies(deps)
        assert count == 3  # torch, tensorflow, transformers

    def test_calculate_dependency_diversity(self):
        """Test dependency diversity calculation."""
        deps = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test"),  # ml
            DependencyInfo(name="cupy", version="9.0.0", type=DependencyType.PYTHON, source="test", gpu_required=True),  # gpu
            DependencyInfo(name="pytest", version="6.2.0", type=DependencyType.PYTHON, source="test"),  # test
            DependencyInfo(name="numpy", version="1.20.0", type=DependencyType.PYTHON, source="test"),  # other
        ]
        
        diversity = self.extractor._calculate_dependency_diversity(deps)
        assert diversity == 1.0  # 4 categories / 4 deps = 1.0

    def test_count_version_constraints(self):
        """Test version constraint counting."""
        deps = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test"),  # exact
            DependencyInfo(name="numpy", version=">=1.20.0", type=DependencyType.PYTHON, source="test"),  # constraint
            DependencyInfo(name="pandas", version="*", type=DependencyType.PYTHON, source="test"),  # wildcard
            DependencyInfo(name="scipy", version=None, type=DependencyType.PYTHON, source="test"),  # no version
        ]
        
        count = self.extractor._count_version_constraints(deps)
        assert count == 2  # torch and numpy have non-wildcard versions

    def test_count_pinned_versions(self):
        """Test pinned version counting."""
        deps = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test"),  # pinned
            DependencyInfo(name="numpy", version=">=1.20.0", type=DependencyType.PYTHON, source="test"),  # not pinned
            DependencyInfo(name="pandas", version="1.3.0", type=DependencyType.PYTHON, source="test"),  # pinned
        ]
        
        count = self.extractor._count_pinned_versions(deps)
        assert count == 2  # torch and pandas

    def test_extract_python_version(self):
        """Test Python version extraction."""
        assert self.extractor._extract_python_version("3.9.7") == 3.9
        assert self.extractor._extract_python_version("Python 3.8.10") == 3.8
        assert self.extractor._extract_python_version("invalid") == 0.0
        assert self.extractor._extract_python_version("") == 0.0

    def test_extract_cuda_version(self):
        """Test CUDA version extraction."""
        assert self.extractor._extract_cuda_version("11.4") == 11.4
        assert self.extractor._extract_cuda_version("CUDA 10.2") == 10.2
        assert self.extractor._extract_cuda_version("invalid") == 0.0
        assert self.extractor._extract_cuda_version("") == 0.0

    def test_is_ml_repository(self):
        """Test ML repository detection."""
        # ML repository (>= 2 ML deps)
        ml_deps = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test"),
            DependencyInfo(name="tensorflow", version="2.6.0", type=DependencyType.PYTHON, source="test"),
        ]
        assert self.extractor._is_ml_repository(ml_deps) is True
        
        # Not ML repository (< 2 ML deps)
        non_ml_deps = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test"),
            DependencyInfo(name="numpy", version="1.20.0", type=DependencyType.PYTHON, source="test"),
        ]
        assert self.extractor._is_ml_repository(non_ml_deps) is False

    def test_is_research_repository(self):
        """Test research repository detection."""
        # Research repository
        research_repo = Mock()
        research_repo.name = "research-project"
        research_repo.description = "Experimental study on deep learning"
        assert self.extractor._is_research_repository(research_repo) is True
        
        # Non-research repository
        regular_repo = Mock()
        regular_repo.name = "web-app"
        regular_repo.description = "A web application for users"
        assert self.extractor._is_research_repository(regular_repo) is False

    def test_categorize_error(self):
        """Test error categorization."""
        assert self.extractor._categorize_error("") == "no_error"
        assert self.extractor._categorize_error("CUDA out of memory") == "gpu_error"
        assert self.extractor._categorize_error("Permission denied") == "permission_error"
        assert self.extractor._categorize_error("Network connection failed") == "network_error"
        assert self.extractor._categorize_error("Out of memory") == "memory_error"
        assert self.extractor._categorize_error("Import error: module not found") == "dependency_error"
        assert self.extractor._categorize_error("Build failed") == "build_error"
        assert self.extractor._categorize_error("Unknown issue") == "unknown_error"

    def test_calculate_similarity_to_known(self):
        """Test similarity calculation."""
        analysis = Mock(spec=Analysis)
        
        # High similarity (3+ ML, 1+ GPU)
        analysis.dependencies = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test", gpu_required=True),
            DependencyInfo(name="tensorflow", version="2.6.0", type=DependencyType.PYTHON, source="test"),
            DependencyInfo(name="transformers", version="4.12.0", type=DependencyType.PYTHON, source="test"),
        ]
        assert self.extractor._calculate_similarity_to_known(analysis) == 0.8
        
        # Medium similarity (2 ML)
        analysis.dependencies = [
            DependencyInfo(name="torch", version="1.9.0", type=DependencyType.PYTHON, source="test"),
            DependencyInfo(name="tensorflow", version="2.6.0", type=DependencyType.PYTHON, source="test"),
        ]
        assert self.extractor._calculate_similarity_to_known(analysis) == 0.6

    def test_calculate_complexity_score(self):
        """Test complexity score calculation."""
        analysis = Mock(spec=Analysis)
        analysis.dependencies = [Mock() for _ in range(10)]  # 10 dependencies
        analysis.compatibility_issues = [Mock() for _ in range(5)]  # 5 issues
        analysis.is_gpu_required.return_value = True
        
        # Mock version constraints
        for i, dep in enumerate(analysis.dependencies):
            dep.version = "1.0.0" if i < 5 else None
        
        score = self.extractor._calculate_complexity_score(analysis)
        assert score > 0.3  # Should be moderate due to many deps and issues

    def test_extract_docker_base_image(self):
        """Test Docker base image extraction."""
        resolution = Mock(spec=Resolution)
        
        # NVIDIA CUDA image
        dockerfile = Mock()
        dockerfile.name = "Dockerfile"
        dockerfile.content = "FROM nvidia/cuda:11.4-runtime-ubuntu20.04"
        resolution.generated_files = [dockerfile]
        
        image_type = self.extractor._extract_docker_base_image(resolution)
        assert image_type == "nvidia_cuda"
        
        # Python image
        dockerfile.content = "FROM python:3.9-slim"
        image_type = self.extractor._extract_docker_base_image(resolution)
        assert image_type == "python"
        
        # Ubuntu image
        dockerfile.content = "FROM ubuntu:20.04"
        image_type = self.extractor._extract_docker_base_image(resolution)
        assert image_type == "ubuntu_debian"

    def test_has_gpu_support(self):
        """Test GPU support detection."""
        resolution = Mock(spec=Resolution)
        
        # With GPU support
        compose_file = Mock()
        compose_file.name = "docker-compose.yml"
        compose_file.content = "services:\n  app:\n    deploy:\n      resources:\n        reservations:\n          devices:\n            - driver: nvidia"
        resolution.generated_files = [compose_file]
        
        assert self.extractor._has_gpu_support(resolution) is True
        
        # Without GPU support
        compose_file.content = "services:\n  app:\n    image: python:3.9"
        assert self.extractor._has_gpu_support(resolution) is False

    def test_file_path_handling(self):
        """Test robust file path handling in Docker methods."""
        resolution = Mock(spec=Resolution)
        
        # Test file with path attribute instead of name
        dockerfile = Mock()
        dockerfile.path = "/app/Dockerfile"
        dockerfile.content = "FROM nvidia/cuda:11.4-runtime-ubuntu20.04"
        # Remove name attribute to test fallback
        if hasattr(dockerfile, 'name'):
            delattr(dockerfile, 'name')
        resolution.generated_files = [dockerfile]
        
        image_type = self.extractor._extract_docker_base_image(resolution)
        assert image_type == "nvidia_cuda"
