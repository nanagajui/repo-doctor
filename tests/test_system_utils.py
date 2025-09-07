"""Tests for system utilities."""

import json
import platform
import subprocess
from unittest.mock import Mock, patch, MagicMock
import pytest
import psutil

from repo_doctor.utils.system import SystemDetector


class TestSystemDetector:
    """Test SystemDetector class."""

    def test_get_python_version(self):
        """Test getting Python version."""
        version = SystemDetector.get_python_version()
        assert isinstance(version, str)
        assert len(version.split('.')) >= 2  # At least major.minor

    def test_get_platform_info(self):
        """Test getting platform information."""
        info = SystemDetector.get_platform_info()
        
        assert "system" in info
        assert "machine" in info
        assert "processor" in info
        assert "release" in info
        
        assert isinstance(info["system"], str)
        assert len(info["system"]) > 0

    def test_get_memory_info(self):
        """Test getting memory information."""
        info = SystemDetector.get_memory_info()
        
        assert "total_gb" in info
        assert "available_gb" in info
        assert "used_gb" in info
        assert "percent_used" in info
        
        assert info["total_gb"] > 0
        assert info["available_gb"] >= 0
        assert info["used_gb"] >= 0
        assert 0 <= info["percent_used"] <= 100

    def test_get_cpu_info(self):
        """Test getting CPU information."""
        info = SystemDetector.get_cpu_info()
        
        assert "physical_cores" in info
        assert "logical_cores" in info
        assert "max_frequency" in info
        assert "current_frequency" in info
        
        assert info["physical_cores"] > 0
        assert info["logical_cores"] >= info["physical_cores"]

    @patch('subprocess.run')
    def test_check_command_available_success(self, mock_run):
        """Test command availability check when command exists."""
        mock_run.return_value = Mock(returncode=0)
        
        result = SystemDetector.check_command_available("python")
        assert result is True
        
        mock_run.assert_called_once_with(
            ["python", "--version"], capture_output=True, timeout=5
        )

    @patch('subprocess.run')
    def test_check_command_available_not_found(self, mock_run):
        """Test command availability check when command not found."""
        mock_run.side_effect = FileNotFoundError()
        
        result = SystemDetector.check_command_available("nonexistent")
        assert result is False

    @patch('subprocess.run')
    def test_check_command_available_timeout(self, mock_run):
        """Test command availability check with timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 5)
        
        result = SystemDetector.check_command_available("slow_command")
        assert result is False

    @patch('subprocess.run')
    def test_check_command_available_error(self, mock_run):
        """Test command availability check with error."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "cmd")
        
        result = SystemDetector.check_command_available("error_command")
        assert result is False

    @patch('subprocess.run')
    def test_get_gpu_info_success(self, mock_run):
        """Test successful GPU info retrieval."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 3080, 10240, 2048, 470.57.02, 8.6\nNVIDIA GeForce GTX 1080, 8192, 1024, 470.57.02, 6.1"
        mock_run.return_value = mock_result
        
        gpus = SystemDetector.get_gpu_info()
        
        assert len(gpus) == 2
        assert gpus[0]["name"] == "NVIDIA GeForce RTX 3080"
        assert gpus[0]["memory_total_mb"] == 10240
        assert gpus[0]["memory_used_mb"] == 2048
        assert gpus[0]["driver_version"] == "470.57.02"
        assert gpus[0]["compute_capability"] == "8.6"
        
        assert gpus[1]["name"] == "NVIDIA GeForce GTX 1080"
        assert gpus[1]["memory_total_mb"] == 8192

    @patch('subprocess.run')
    def test_get_gpu_info_no_gpu(self, mock_run):
        """Test GPU info retrieval when no GPU available."""
        mock_run.side_effect = FileNotFoundError()
        
        gpus = SystemDetector.get_gpu_info()
        assert gpus == []

    @patch('subprocess.run')
    def test_get_gpu_info_timeout(self, mock_run):
        """Test GPU info retrieval with timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("nvidia-smi", 10)
        
        gpus = SystemDetector.get_gpu_info()
        assert gpus == []

    @patch('subprocess.run')
    def test_get_gpu_info_error(self, mock_run):
        """Test GPU info retrieval with error."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        gpus = SystemDetector.get_gpu_info()
        assert gpus == []

    @patch('subprocess.run')
    def test_get_gpu_info_malformed_output(self, mock_run):
        """Test GPU info retrieval with malformed output."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Invalid, Output\nToo, Few, Fields"
        mock_run.return_value = mock_result
        
        gpus = SystemDetector.get_gpu_info()
        assert gpus == []

    @patch('subprocess.run')
    def test_get_cuda_version_success(self, mock_run):
        """Test successful CUDA version retrieval."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0"""
        mock_run.return_value = mock_result
        
        version = SystemDetector.get_cuda_version()
        assert version == "11.5"

    @patch('subprocess.run')
    def test_get_cuda_version_not_found(self, mock_run):
        """Test CUDA version retrieval when CUDA not available."""
        mock_run.side_effect = FileNotFoundError()
        
        version = SystemDetector.get_cuda_version()
        assert version is None

    @patch('subprocess.run')
    def test_get_cuda_version_timeout(self, mock_run):
        """Test CUDA version retrieval with timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("nvcc", 5)
        
        version = SystemDetector.get_cuda_version()
        assert version is None

    @patch('subprocess.run')
    def test_get_cuda_version_no_version_info(self, mock_run):
        """Test CUDA version retrieval with no version info."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "No version information found"
        mock_run.return_value = mock_result
        
        version = SystemDetector.get_cuda_version()
        assert version is None

    @patch.object(SystemDetector, 'check_command_available')
    def test_get_docker_info_not_available(self, mock_check):
        """Test Docker info when Docker not available."""
        mock_check.return_value = False
        
        info = SystemDetector.get_docker_info()
        assert info is None

    @patch.object(SystemDetector, 'check_command_available')
    @patch.object(SystemDetector, '_check_docker_gpu_support')
    @patch('subprocess.run')
    def test_get_docker_info_success(self, mock_run, mock_gpu_support, mock_check):
        """Test successful Docker info retrieval."""
        mock_check.return_value = True
        mock_gpu_support.return_value = True
        
        # Mock version and info commands
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "Docker version 20.10.12, build e91ed57"
        
        info_result = Mock()
        info_result.returncode = 0
        
        mock_run.side_effect = [version_result, info_result]
        
        info = SystemDetector.get_docker_info()
        
        assert info["available"] is True
        assert "Docker version 20.10.12" in info["version"]
        assert info["daemon_running"] is True
        assert info["supports_gpu"] is True

    @patch.object(SystemDetector, 'check_command_available')
    @patch('subprocess.run')
    def test_get_docker_info_daemon_not_running(self, mock_run, mock_check):
        """Test Docker info when daemon not running."""
        mock_check.return_value = True
        
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "Docker version 20.10.12"
        
        info_result = Mock()
        info_result.returncode = 1  # Docker daemon not running
        
        mock_run.side_effect = [version_result, info_result]
        
        with patch.object(SystemDetector, '_check_docker_gpu_support', return_value=False):
            info = SystemDetector.get_docker_info()
        
        assert info["available"] is True
        assert info["daemon_running"] is False

    @patch.object(SystemDetector, 'check_command_available')
    @patch('subprocess.run')
    def test_get_docker_info_timeout(self, mock_run, mock_check):
        """Test Docker info with timeout."""
        mock_check.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired("docker", 5)
        
        info = SystemDetector.get_docker_info()
        assert info["available"] is False

    @patch('subprocess.run')
    def test_check_docker_gpu_support_success(self, mock_run):
        """Test Docker GPU support check success."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        result = SystemDetector._check_docker_gpu_support()
        assert result is True

    @patch('subprocess.run')
    def test_check_docker_gpu_support_failure(self, mock_run):
        """Test Docker GPU support check failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        result = SystemDetector._check_docker_gpu_support()
        assert result is False

    @patch('subprocess.run')
    def test_check_docker_gpu_support_timeout(self, mock_run):
        """Test Docker GPU support check timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("docker", 30)
        
        result = SystemDetector._check_docker_gpu_support()
        assert result is False

    @patch.object(SystemDetector, 'check_command_available')
    def test_get_conda_info_not_available(self, mock_check):
        """Test Conda info when Conda not available."""
        mock_check.return_value = False
        
        info = SystemDetector.get_conda_info()
        assert info is None

    @patch.object(SystemDetector, 'check_command_available')
    @patch('subprocess.run')
    def test_get_conda_info_success(self, mock_run, mock_check):
        """Test successful Conda info retrieval."""
        mock_check.return_value = True
        
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "conda 4.12.0"
        
        info_data = {
            "conda_version": "4.12.0",
            "python_version": "3.9.7",
            "platform": "linux-64",
            "envs_dirs": ["/home/user/miniconda3/envs"]
        }
        
        info_result = Mock()
        info_result.returncode = 0
        info_result.stdout = json.dumps(info_data)
        
        mock_run.side_effect = [version_result, info_result]
        
        info = SystemDetector.get_conda_info()
        
        assert info["available"] is True
        assert info["version"] == "conda 4.12.0"
        assert info["conda_version"] == "4.12.0"
        assert info["python_version"] == "3.9.7"
        assert info["platform"] == "linux-64"

    @patch.object(SystemDetector, 'check_command_available')
    @patch('subprocess.run')
    def test_get_conda_info_version_only(self, mock_run, mock_check):
        """Test Conda info with version only."""
        mock_check.return_value = True
        
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "conda 4.12.0"
        
        info_result = Mock()
        info_result.returncode = 1  # Info command fails
        
        mock_run.side_effect = [version_result, info_result]
        
        info = SystemDetector.get_conda_info()
        
        assert info["available"] is True
        assert info["version"] == "conda 4.12.0"
        assert "conda_version" not in info

    @patch.object(SystemDetector, 'check_command_available')
    @patch('subprocess.run')
    def test_get_conda_info_invalid_json(self, mock_run, mock_check):
        """Test Conda info with invalid JSON."""
        mock_check.return_value = True
        
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "conda 4.12.0"
        
        info_result = Mock()
        info_result.returncode = 0
        info_result.stdout = "Invalid JSON"
        
        mock_run.side_effect = [version_result, info_result]
        
        info = SystemDetector.get_conda_info()
        
        assert info["available"] is True
        assert info["version"] == "conda 4.12.0"
        assert "conda_version" not in info

    @patch.object(SystemDetector, 'check_command_available')
    @patch('subprocess.run')
    def test_get_conda_info_timeout(self, mock_run, mock_check):
        """Test Conda info with timeout."""
        mock_check.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired("conda", 5)
        
        info = SystemDetector.get_conda_info()
        assert info["available"] is False

    @patch.object(SystemDetector, 'check_command_available')
    def test_get_micromamba_info_not_available(self, mock_check):
        """Test Micromamba info when Micromamba not available."""
        mock_check.return_value = False
        
        info = SystemDetector.get_micromamba_info()
        assert info is None

    @patch.object(SystemDetector, 'check_command_available')
    @patch('subprocess.run')
    def test_get_micromamba_info_success(self, mock_run, mock_check):
        """Test successful Micromamba info retrieval."""
        mock_check.return_value = True
        
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "micromamba 0.24.0"
        
        info_result = Mock()
        info_result.returncode = 0
        info_result.stdout = """platform : linux-64
channels : conda-forge, bioconda
"""
        
        mock_run.side_effect = [version_result, info_result]
        
        info = SystemDetector.get_micromamba_info()
        
        assert info["available"] is True
        assert info["version"] == "micromamba 0.24.0"
        assert info["platform"] == "linux-64"
        assert info["channels"] == "conda-forge, bioconda"

    @patch.object(SystemDetector, 'check_command_available')
    @patch('subprocess.run')
    def test_get_micromamba_info_version_only(self, mock_run, mock_check):
        """Test Micromamba info with version only."""
        mock_check.return_value = True
        
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "micromamba 0.24.0"
        
        info_result = Mock()
        info_result.returncode = 1  # Info command fails
        
        mock_run.side_effect = [version_result, info_result]
        
        info = SystemDetector.get_micromamba_info()
        
        assert info["available"] is True
        assert info["version"] == "micromamba 0.24.0"
        assert "platform" not in info

    @patch.object(SystemDetector, 'check_command_available')
    @patch('subprocess.run')
    def test_get_micromamba_info_timeout(self, mock_run, mock_check):
        """Test Micromamba info with timeout."""
        mock_check.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired("micromamba", 5)
        
        info = SystemDetector.get_micromamba_info()
        assert info["available"] is False


class TestSystemDetectorIntegration:
    """Integration tests for SystemDetector."""

    def test_system_detection_workflow(self):
        """Test complete system detection workflow."""
        # Test basic system info (should always work)
        python_version = SystemDetector.get_python_version()
        assert isinstance(python_version, str)
        
        platform_info = SystemDetector.get_platform_info()
        assert "system" in platform_info
        
        memory_info = SystemDetector.get_memory_info()
        assert memory_info["total_gb"] > 0
        
        cpu_info = SystemDetector.get_cpu_info()
        assert cpu_info["physical_cores"] > 0
        
        # Test command availability (common commands)
        python_available = SystemDetector.check_command_available("python")
        # Python should be available since we're running tests with it
        assert python_available is True

    @patch('subprocess.run')
    def test_comprehensive_system_scan_mock(self, mock_run):
        """Test comprehensive system scan with mocked external commands."""
        # Mock all subprocess calls to return success
        def mock_subprocess(*args, **kwargs):
            cmd = args[0]
            result = Mock()
            result.returncode = 0
            
            if "nvidia-smi" in cmd:
                result.stdout = "NVIDIA GeForce RTX 3080, 10240, 2048, 470.57.02, 8.6"
            elif "nvcc" in cmd:
                result.stdout = "Cuda compilation tools, release 11.5, V11.5.119"
            elif "docker" in cmd and "--version" in cmd:
                result.stdout = "Docker version 20.10.12"
            elif "docker" in cmd and "info" in cmd:
                result.stdout = "Docker info output"
            elif "conda" in cmd and "--version" in cmd:
                result.stdout = "conda 4.12.0"
            elif "conda" in cmd and "info" in cmd:
                result.stdout = json.dumps({"conda_version": "4.12.0", "platform": "linux-64"})
            elif "micromamba" in cmd and "--version" in cmd:
                result.stdout = "micromamba 0.24.0"
            elif "micromamba" in cmd and "info" in cmd:
                result.stdout = "platform : linux-64\nchannels : conda-forge"
            else:
                result.stdout = "Command output"
            
            return result
        
        mock_run.side_effect = mock_subprocess
        
        # Test all detection methods
        with patch.object(SystemDetector, '_check_docker_gpu_support', return_value=True):
            gpu_info = SystemDetector.get_gpu_info()
            cuda_version = SystemDetector.get_cuda_version()
            docker_info = SystemDetector.get_docker_info()
            conda_info = SystemDetector.get_conda_info()
            micromamba_info = SystemDetector.get_micromamba_info()
        
        # Verify results
        assert len(gpu_info) == 1
        assert gpu_info[0]["name"] == "NVIDIA GeForce RTX 3080"
        
        assert cuda_version == "11.5"
        
        assert docker_info["available"] is True
        assert docker_info["daemon_running"] is True
        
        assert conda_info["available"] is True
        assert conda_info["conda_version"] == "4.12.0"
        
        assert micromamba_info["available"] is True
        assert micromamba_info["platform"] == "linux-64"

    def test_error_handling_robustness(self):
        """Test that all methods handle errors gracefully."""
        # These should not raise exceptions even if external commands fail
        try:
            SystemDetector.get_gpu_info()
            SystemDetector.get_cuda_version()
            SystemDetector.get_docker_info()
            SystemDetector.get_conda_info()
            SystemDetector.get_micromamba_info()
            SystemDetector.check_command_available("nonexistent_command_12345")
        except Exception as e:
            pytest.fail(f"System detection method raised unexpected exception: {e}")
