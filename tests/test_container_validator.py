"""Tests for container validation."""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import docker

from repo_doctor.validators.container import ContainerValidator
from repo_doctor.models.analysis import Analysis, DependencyInfo
from repo_doctor.models.resolution import Resolution, GeneratedFile, ValidationResult, ValidationStatus


class TestContainerValidator:
    """Test ContainerValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch('docker.from_env'):
            self.validator = ContainerValidator()

    @patch('docker.from_env')
    def test_init_with_docker_available(self, mock_docker):
        """Test ContainerValidator initialization with Docker available."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        with patch.object(ContainerValidator, '_check_gpu_support', return_value=True):
            validator = ContainerValidator()
            
        assert validator.docker_client == mock_client
        assert validator.gpu_available is True

    @patch('docker.from_env')
    def test_init_with_docker_unavailable(self, mock_docker):
        """Test ContainerValidator initialization with Docker unavailable."""
        mock_docker.side_effect = docker.errors.DockerException("Docker not available")
        
        validator = ContainerValidator()
        
        assert validator.docker_client is None
        assert validator.gpu_available is False

    def test_validate_docker_solution_no_client(self):
        """Test Docker solution validation when client not available."""
        self.validator.docker_client = None
        
        result = self.validator.validate_docker_solution("/path/to/dockerfile")
        
        assert result.status == ValidationStatus.FAILED
        assert "Docker client not available" in result.error_message

    @patch('docker.from_env')
    def test_validate_docker_solution_success(self, mock_docker):
        """Test successful Docker solution validation."""
        # Mock Docker client and operations
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock image build
        mock_image = Mock()
        mock_image.id = "test-image-id"
        build_logs = [
            {"stream": "Step 1/3 : FROM python:3.9"},
            {"stream": "Successfully built test-image-id"}
        ]
        mock_client.images.build.return_value = (mock_image, build_logs)
        
        # Mock container run
        mock_container = Mock()
        mock_container.id = "test-container-id"
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_client.containers.run.return_value = mock_container
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        
        result = validator.validate_docker_solution("/path/to/dockerfile")
        
        assert result.status == ValidationStatus.SUCCESS
        assert result.container_id == "test-container-id"
        assert "Container test successful" in result.logs

    @patch('docker.from_env')
    def test_validate_docker_solution_build_error(self, mock_docker):
        """Test Docker solution validation with build error."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock build error
        mock_client.images.build.side_effect = docker.errors.BuildError("Build failed", build_log=[])
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        
        result = validator.validate_docker_solution("/path/to/dockerfile")
        
        assert result.status == ValidationStatus.FAILED
        assert "Docker build failed" in result.error_message

    @patch('docker.from_env')
    def test_validate_docker_solution_container_error(self, mock_docker):
        """Test Docker solution validation with container error."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock successful build
        mock_image = Mock()
        mock_image.id = "test-image-id"
        mock_client.images.build.return_value = (mock_image, [])
        
        # Mock container error
        mock_client.containers.run.side_effect = docker.errors.ContainerError(
            container="test", exit_status=1, command="python --version", image="test", stderr="Error"
        )
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        
        result = validator.validate_docker_solution("/path/to/dockerfile")
        
        assert result.status == ValidationStatus.FAILED
        assert "Container execution failed" in result.error_message

    @patch('docker.from_env')
    def test_validate_docker_solution_container_exit_code(self, mock_docker):
        """Test Docker solution validation with non-zero exit code."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock successful build
        mock_image = Mock()
        mock_image.id = "test-image-id"
        mock_client.images.build.return_value = (mock_image, [])
        
        # Mock container with non-zero exit code
        mock_container = Mock()
        mock_container.id = "test-container-id"
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_client.containers.run.return_value = mock_container
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        
        result = validator.validate_docker_solution("/path/to/dockerfile")
        
        assert result.status == ValidationStatus.FAILED
        assert "Container failed with exit code 1" in result.error_message

    def test_validate_resolution_no_client(self):
        """Test resolution validation when client not available."""
        self.validator.docker_client = None
        
        resolution = Mock()
        analysis = Mock()
        
        result = self.validator.validate_resolution(resolution, analysis)
        
        assert result.status == ValidationStatus.FAILED
        assert "Docker client not available" in result.error_message

    @patch('tempfile.TemporaryDirectory')
    def test_validate_resolution_success(self, mock_temp_dir):
        """Test successful resolution validation."""
        # Mock temporary directory
        temp_path = Path("/tmp/test")
        mock_temp_dir.return_value.__enter__.return_value = str(temp_path)
        
        # Mock resolution with generated files
        dockerfile_content = "FROM python:3.9\nRUN pip install numpy"
        requirements_content = "numpy==1.21.0"
        
        generated_files = [
            GeneratedFile(path="Dockerfile", content=dockerfile_content, description="Docker configuration", executable=False),
            GeneratedFile(path="requirements.txt", content=requirements_content, description="Python dependencies", executable=False)
        ]
        
        resolution = Mock()
        resolution.generated_files = generated_files
        
        analysis = Mock()
        
        # Mock _validate_in_directory to return success
        with patch.object(self.validator, '_validate_in_directory') as mock_validate:
            mock_validate.return_value = ValidationResult(
                status=ValidationStatus.SUCCESS,
                duration=10.0,
                logs=["All tests passed"]
            )
            
            # Mock file operations
            with patch('pathlib.Path.mkdir'), \
                 patch('builtins.open', create=True), \
                 patch('pathlib.Path.chmod'):
                
                result = self.validator.validate_resolution(resolution, analysis)
        
        assert result.status == ValidationStatus.SUCCESS

    def test_validate_in_directory_no_dockerfile(self):
        """Test validation in directory without Dockerfile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            analysis = Mock()
            
            result = self.validator._validate_in_directory(temp_path, analysis, 300)
            
            assert result.status == ValidationStatus.FAILED
            assert "No Dockerfile found" in result.error_message

    @patch('docker.from_env')
    def test_validate_in_directory_success(self, mock_docker):
        """Test successful validation in directory."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock image build
        mock_image = Mock()
        mock_image.id = "test-image-id"
        build_logs = [{"stream": "Successfully built"}]
        mock_client.images.build.return_value = (mock_image, build_logs)
        
        # Mock container runs for tests
        mock_container = Mock()
        mock_container.id = "test-container-id"
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_client.containers.run.return_value = mock_container
        
        # Mock analysis
        analysis = Mock()
        analysis.is_gpu_required.return_value = False
        analysis.get_python_dependencies.return_value = [
            Mock(name="numpy"),
            Mock(name="torch")
        ]
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        validator.gpu_available = False
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create Dockerfile
            dockerfile_path = temp_path / "Dockerfile"
            dockerfile_path.write_text("FROM python:3.9")
            
            result = validator._validate_in_directory(temp_path, analysis, 300)
        
        assert result.status == ValidationStatus.SUCCESS
        assert "🎉 All validation tests passed!" in result.logs

    @patch('docker.from_env')
    def test_validate_in_directory_test_failure(self, mock_docker):
        """Test validation in directory with test failure."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock image build
        mock_image = Mock()
        mock_image.id = "test-image-id"
        mock_client.images.build.return_value = (mock_image, [])
        
        # Mock container run with failure
        mock_container = Mock()
        mock_container.id = "test-container-id"
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.return_value = b"Test failed"
        mock_client.containers.run.return_value = mock_container
        
        # Mock analysis
        analysis = Mock()
        analysis.is_gpu_required.return_value = False
        analysis.get_python_dependencies.return_value = []
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create Dockerfile
            dockerfile_path = temp_path / "Dockerfile"
            dockerfile_path.write_text("FROM python:3.9")
            
            result = validator._validate_in_directory(temp_path, analysis, 300)
        
        assert result.status == ValidationStatus.FAILED
        assert "failed with exit code 1" in result.error_message

    @patch('docker.from_env')
    def test_validate_in_directory_gpu_tests(self, mock_docker):
        """Test validation in directory with GPU tests."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock image build
        mock_image = Mock()
        mock_image.id = "test-image-id"
        mock_client.images.build.return_value = (mock_image, [])
        
        # Mock container runs
        mock_container = Mock()
        mock_container.id = "test-container-id"
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_client.containers.run.return_value = mock_container
        
        # Mock analysis with GPU requirements
        analysis = Mock()
        analysis.is_gpu_required.return_value = True
        analysis.get_python_dependencies.return_value = [
            Mock(name="torch"),
            Mock(name="tensorflow")
        ]
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        validator.gpu_available = True
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create Dockerfile
            dockerfile_path = temp_path / "Dockerfile"
            dockerfile_path.write_text("FROM nvidia/cuda:11.8-base-ubuntu20.04")
            
            result = validator._validate_in_directory(temp_path, analysis, 300)
        
        assert result.status == ValidationStatus.SUCCESS
        
        # Verify GPU-specific container runs were called
        calls = mock_client.containers.run.call_args_list
        gpu_calls = [call for call in calls if 'device_requests' in call[1]]
        assert len(gpu_calls) > 0

    @patch('docker.from_env')
    def test_check_gpu_support_success(self, mock_docker):
        """Test GPU support check success."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock successful GPU container run
        mock_container = Mock()
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_client.containers.run.return_value = mock_container
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        
        result = validator._check_gpu_support()
        assert result is True

    @patch('docker.from_env')
    def test_check_gpu_support_failure(self, mock_docker):
        """Test GPU support check failure."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock failed GPU container run
        mock_container = Mock()
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_client.containers.run.return_value = mock_container
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        
        result = validator._check_gpu_support()
        assert result is False

    @patch('docker.from_env')
    def test_check_gpu_support_exception(self, mock_docker):
        """Test GPU support check with exception."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock exception during GPU check
        mock_client.containers.run.side_effect = Exception("GPU not available")
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        
        result = validator._check_gpu_support()
        assert result is False

    def test_get_validation_tests_basic(self):
        """Test getting basic validation tests."""
        analysis = Mock()
        analysis.is_gpu_required.return_value = False
        analysis.get_python_dependencies.return_value = []
        
        tests = self.validator._get_validation_tests(analysis)
        
        assert "python_version" in tests
        assert "pip_functionality" in tests
        assert "basic_imports" in tests
        assert tests["python_version"] == "python --version"

    def test_get_validation_tests_with_dependencies(self):
        """Test getting validation tests with dependencies."""
        # Create mock dependencies with proper name attribute
        torch_dep = Mock()
        torch_dep.name = "torch"
        numpy_dep = Mock()
        numpy_dep.name = "numpy"
        tensorflow_dep = Mock()
        tensorflow_dep.name = "tensorflow"
        
        analysis = Mock()
        analysis.is_gpu_required.return_value = False
        analysis.get_python_dependencies.return_value = [
            torch_dep, numpy_dep, tensorflow_dep
        ]
        
        tests = self.validator._get_validation_tests(analysis)
        
        # Only torch, tensorflow, and jax get import tests
        assert "import_torch" in tests
        assert "import_tensorflow" in tests
        assert "python -c 'import torch" in tests["import_torch"]
        # numpy doesn't get an import test since it's not in the special list

    def test_get_validation_tests_with_gpu(self):
        """Test getting validation tests with GPU requirements."""
        # Create mock dependencies with proper name attribute
        torch_dep = Mock()
        torch_dep.name = "torch"
        tensorflow_dep = Mock()
        tensorflow_dep.name = "tensorflow-gpu"
        
        analysis = Mock()
        analysis.is_gpu_required.return_value = True
        analysis.get_python_dependencies.return_value = [
            torch_dep, tensorflow_dep
        ]
        
        tests = self.validator._get_validation_tests(analysis)
        
        assert "gpu_nvidia_smi" in tests
        assert "pytorch_gpu" in tests
        assert "tensorflow_gpu" in tests
        assert tests["gpu_nvidia_smi"] == "nvidia-smi"

    def test_cleanup_test_containers_no_client(self):
        """Test cleanup when no Docker client available."""
        self.validator.docker_client = None
        
        result = self.validator.cleanup_test_containers()
        assert result == 0

    @patch('docker.from_env')
    def test_cleanup_test_containers_success(self, mock_docker):
        """Test successful cleanup of test containers."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock containers with repo-doctor-test tags
        mock_container1 = Mock()
        mock_container1.image.tags = ["repo-doctor-test-123:latest"]
        mock_container2 = Mock()
        mock_container2.image.tags = ["other-image:latest"]
        
        mock_client.containers.list.return_value = [mock_container1, mock_container2]
        
        # Mock images with repo-doctor-test tags
        mock_image1 = Mock()
        mock_image1.id = "image-1"
        mock_image1.tags = ["repo-doctor-test-456:latest"]
        mock_image2 = Mock()
        mock_image2.id = "image-2"
        mock_image2.tags = ["other-image:latest"]
        
        mock_client.images.list.return_value = [mock_image1, mock_image2]
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        
        result = validator.cleanup_test_containers()
        
        # Should clean up 1 container + 1 image = 2 items
        assert result == 2
        mock_container1.remove.assert_called_once_with(force=True)
        mock_client.images.remove.assert_called_once_with("image-1", force=True)

    @patch('docker.from_env')
    def test_cleanup_test_containers_exception(self, mock_docker):
        """Test cleanup with exception handling."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock exception during cleanup
        mock_client.containers.list.side_effect = Exception("Cleanup failed")
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        
        result = validator.cleanup_test_containers()
        assert result == 0  # Should handle exception gracefully


class TestContainerValidatorIntegration:
    """Integration tests for ContainerValidator."""

    @patch('docker.from_env')
    def test_full_validation_workflow_mock(self, mock_docker):
        """Test full validation workflow with mocked Docker."""
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock successful operations
        mock_image = Mock()
        mock_image.id = "test-image-id"
        mock_client.images.build.return_value = (mock_image, [])
        
        mock_container = Mock()
        mock_container.id = "test-container-id"
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_client.containers.run.return_value = mock_container
        
        # Create test resolution
        dockerfile_content = """FROM python:3.9
RUN pip install numpy torch
CMD ["python", "--version"]"""
        
        generated_files = [
            GeneratedFile(path="Dockerfile", content=dockerfile_content, description="Docker configuration", executable=False),
            GeneratedFile(path="requirements.txt", content="numpy==1.21.0\ntorch==1.9.0", description="Python dependencies", executable=False)
        ]
        
        resolution = Mock()
        resolution.generated_files = generated_files
        
        # Create test analysis
        analysis = Mock()
        analysis.is_gpu_required.return_value = False
        analysis.get_python_dependencies.return_value = [
            Mock(name="numpy"),
            Mock(name="torch")
        ]
        
        validator = ContainerValidator()
        validator.docker_client = mock_client
        validator.gpu_available = False
        
        # Test validation
        result = validator.validate_resolution(resolution, analysis, timeout=60)
        
        assert result.status == ValidationStatus.SUCCESS
        assert result.duration > 0

    def test_error_handling_robustness(self):
        """Test that validator handles errors gracefully."""
        validator = ContainerValidator()
        
        # Test with no Docker client
        validator.docker_client = None
        
        result = validator.validate_docker_solution("/nonexistent/dockerfile")
        assert result.status == ValidationStatus.FAILED
        
        result = validator.validate_resolution(Mock(), Mock())
        assert result.status == ValidationStatus.FAILED
        
        cleanup_result = validator.cleanup_test_containers()
        assert cleanup_result == 0
