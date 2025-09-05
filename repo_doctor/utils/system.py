"""System detection utilities."""

import platform
import subprocess
from typing import Any, Dict, List, Optional

import psutil


class SystemDetector:
    """Utility class for system detection and capability assessment."""

    @staticmethod
    def get_python_version() -> str:
        """Get current Python version."""
        return platform.python_version()

    @staticmethod
    def get_platform_info() -> Dict[str, str]:
        """Get platform information."""
        return {
            "system": platform.system().lower(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "release": platform.release(),
        }

    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get memory information in GB."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent,
        }

    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Get CPU information."""
        return {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else None,
            "current_frequency": (
                psutil.cpu_freq().current if psutil.cpu_freq() else None
            ),
        }

    @staticmethod
    def check_command_available(command: str) -> bool:
        """Check if a command is available in PATH."""
        try:
            subprocess.run([command, "--version"], capture_output=True, timeout=5)
            return True
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            return False

    @staticmethod
    def get_gpu_info() -> List[Dict[str, Any]]:
        """Get GPU information using nvidia-smi."""
        gpus = []

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 4:
                            gpus.append(
                                {
                                    "name": parts[0],
                                    "memory_total_mb": int(parts[1]),
                                    "memory_used_mb": int(parts[2]),
                                    "driver_version": parts[3],
                                    "compute_capability": (
                                        parts[4] if len(parts) > 4 else None
                                    ),
                                }
                            )

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass

        return gpus

    @staticmethod
    def get_cuda_version() -> Optional[str]:
        """Get CUDA version from nvcc."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line.lower():
                        import re

                        version_match = re.search(r"release (\d+\.\d+)", line)
                        if version_match:
                            return version_match.group(1)
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass
        return None

    @staticmethod
    def get_docker_info() -> Optional[Dict[str, Any]]:
        """Get Docker information."""
        if not SystemDetector.check_command_available("docker"):
            return None

        try:
            # Get Docker version
            version_result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=5
            )

            # Check if Docker daemon is running
            info_result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, timeout=5
            )

            return {
                "available": True,
                "version": (
                    version_result.stdout.strip()
                    if version_result.returncode == 0
                    else None
                ),
                "daemon_running": info_result.returncode == 0,
                "supports_gpu": SystemDetector._check_docker_gpu_support(),
            }

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return {"available": False}

    @staticmethod
    def _check_docker_gpu_support() -> bool:
        """Check if Docker supports GPU (nvidia-docker)."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "all",
                    "nvidia/cuda:11.8-base-ubuntu20.04",
                    "nvidia-smi",
                ],
                capture_output=True,
                timeout=30,
            )
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            return False

    @staticmethod
    def get_conda_info() -> Optional[Dict[str, Any]]:
        """Get Conda information."""
        if not SystemDetector.check_command_available("conda"):
            return None

        try:
            version_result = subprocess.run(
                ["conda", "--version"], capture_output=True, text=True, timeout=5
            )

            info_result = subprocess.run(
                ["conda", "info", "--json"], capture_output=True, text=True, timeout=10
            )

            conda_info = {
                "available": True,
                "version": (
                    version_result.stdout.strip()
                    if version_result.returncode == 0
                    else "unknown"
                ),
            }

            if info_result.returncode == 0:
                import json

                try:
                    info_data = json.loads(info_result.stdout)
                    conda_info.update(
                        {
                            "conda_version": info_data.get("conda_version"),
                            "python_version": info_data.get("python_version"),
                            "platform": info_data.get("platform"),
                            "envs_dirs": info_data.get("envs_dirs", []),
                        }
                    )
                except json.JSONDecodeError:
                    pass

            return conda_info

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return {"available": False}

    @staticmethod
    def get_micromamba_info() -> Optional[Dict[str, Any]]:
        """Get Micromamba information."""
        if not SystemDetector.check_command_available("micromamba"):
            return None

        try:
            version_result = subprocess.run(
                ["micromamba", "--version"], capture_output=True, text=True, timeout=5
            )

            info_result = subprocess.run(
                ["micromamba", "info"], capture_output=True, text=True, timeout=10
            )

            micromamba_info = {
                "available": True,
                "version": (
                    version_result.stdout.strip()
                    if version_result.returncode == 0
                    else "unknown"
                ),
            }

            if info_result.returncode == 0:
                # Parse micromamba info output
                info_lines = info_result.stdout.strip().split('\n')
                for line in info_lines:
                    if 'platform' in line.lower():
                        micromamba_info["platform"] = line.split(':')[-1].strip()
                    elif 'channels' in line.lower():
                        micromamba_info["channels"] = line.split(':')[-1].strip()

            return micromamba_info

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return {"available": False}
