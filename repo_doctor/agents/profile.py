"""Profile Agent - System profiling and capability detection."""

import asyncio
import os
import platform
import shutil
import subprocess
import time
from typing import List, Optional, Dict, Any

import psutil

from ..models.system import GPUInfo, HardwareInfo, SoftwareStack, SystemProfile
from .contracts import AgentContractValidator, AgentErrorHandler, AgentPerformanceMonitor
from ..utils.logging_config import get_logger, log_performance


class ProfileAgent:
    """Agent for profiling system capabilities."""

    def __init__(self, config: Optional[Any] = None) -> None:
        from ..utils.config import Config
        self.config = config or Config.load()
        self.performance_monitor = AgentPerformanceMonitor(self.config)
        self.logger = get_logger(__name__)
        # Enable a fast profile mode for tests/CI to avoid long-running commands
        self._fast_profile = os.getenv("REPO_DOCTOR_FAST_PROFILE", "0") in {"1", "true", "True"}

    def profile(self) -> SystemProfile:
        """Generate complete system profile with dual sync/async compatibility.

        Returns an object that:
        - behaves like `SystemProfile` for synchronous usage (attribute access forwards), and
        - is awaitable; `await agent.profile()` resolves to the same `SystemProfile`.
        """
        start_time = time.time()

        try:
            # Run synchronously
            profile = self._profile_sync()

            # Validate the profile against contracts
            AgentContractValidator.validate_system_profile(profile)

            # Check performance
            duration = time.time() - start_time
            if not self.performance_monitor.check_profile_performance(duration):
                self.logger.warning(
                    f"Profile agent took {duration:.2f}s (target: {self.performance_monitor.performance_targets['profile_agent']}s)"
                )

            # Log performance metrics
            log_performance("system_profile", duration, agent="ProfileAgent")

            # Return an awaitable SystemProfile subclass for compatibility
            class _AwaitableSystemProfile(SystemProfile):
                def __await__(self):
                    async def _coro():
                        return self
                    return _coro().__await__()

            return _AwaitableSystemProfile(**profile.model_dump())  # type: ignore[return-value]
        except Exception as e:
            return AgentErrorHandler.handle_profile_error(e, "system_profiling")

    async def profile_async(self) -> SystemProfile:
        """Async variant of profile to integrate with async workflows."""
        start_time = time.time()

        try:
            profile = await asyncio.to_thread(self._profile_sync)

            AgentContractValidator.validate_system_profile(profile)

            duration = time.time() - start_time
            if not self.performance_monitor.check_profile_performance(duration):
                self.logger.warning(
                    f"Profile agent took {duration:.2f}s (target: {self.performance_monitor.performance_targets['profile_agent']}s)"
                )

            log_performance("system_profile", duration, agent="ProfileAgent")

            return profile
        except Exception as e:
            return AgentErrorHandler.handle_profile_error(e, "system_profiling")
    
    def _profile_sync(self) -> SystemProfile:
        """Synchronous profiling implementation."""
        return SystemProfile(
            hardware=self._get_hardware_info(),
            software=self._get_software_stack(),
            platform=platform.system().lower(),
            container_runtime=self._detect_container_runtime(),
            compute_score=self._calculate_compute_score(),
        )
    
    def profile_sync(self) -> SystemProfile:
        """Synchronous profile method for backward compatibility."""
        return self._profile_sync()

    def _get_hardware_info(self) -> HardwareInfo:
        """Get hardware information."""
        try:
            cpu_cores = psutil.cpu_count(logical=False) or 1
            memory_gb = psutil.virtual_memory().total / (1024**3)
            gpus = self._detect_gpus()
            architecture = platform.machine()
        except Exception:
            # Fallback values if psutil fails
            cpu_cores = 1
            memory_gb = 4.0
            gpus = []
            architecture = "unknown"
            
        return HardwareInfo(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpus=gpus,
            architecture=architecture,
        )

    def _get_software_stack(self) -> SoftwareStack:
        """Get software stack information."""
        try:
            python_version = platform.python_version()
        except Exception:
            python_version = "unknown"
        
        # In fast mode, avoid ALL external subprocess calls to prevent timeouts in CI/tests
        if self._fast_profile:
            return SoftwareStack(
                python_version=python_version,
                pip_version=None,
                conda_version=None,
                micromamba_version=None,
                docker_version=None,
                git_version=None,
                cuda_version=None,
            )

        # Use a conservative timeout for version checks derived from config
        cmd_timeout = self._effective_cmd_timeout()

        pip_version = self._run_command(["pip", "--version"], timeout=cmd_timeout)
        git_version = self._run_command(["git", "--version"], timeout=cmd_timeout)

        conda_version = self._run_command(["conda", "--version"], timeout=cmd_timeout)
        micromamba_version = self._run_command(["micromamba", "--version"], timeout=cmd_timeout)
        docker_version = self._run_command(["docker", "--version"], timeout=cmd_timeout)
        cuda_version = self._detect_cuda_version()

        return SoftwareStack(
            python_version=python_version,
            pip_version=pip_version,
            conda_version=conda_version,
            micromamba_version=micromamba_version,
            docker_version=docker_version,
            git_version=git_version,
            cuda_version=cuda_version,
        )

    def _detect_gpus(self) -> List[GPUInfo]:
        """Detect available GPUs."""
        gpus = []

        # Try nvidia-smi first
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=self.config.advanced.gpu_detection_timeout,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            # Parse memory value, handling both "24576 MiB" and "24576" formats
                            memory_str = parts[1]
                            if " MiB" in memory_str:
                                memory_mb = float(memory_str.replace(" MiB", ""))
                            elif " MB" in memory_str:
                                memory_mb = float(memory_str.replace(" MB", ""))
                            else:
                                memory_mb = float(memory_str)
                            
                            gpus.append(
                                GPUInfo(
                                    name=parts[0],
                                    memory_gb=memory_mb / 1024,
                                    driver_version=parts[2],
                                    cuda_version=self._detect_cuda_version(),
                                )
                            )
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass

        return gpus

    def _detect_cuda_version(self) -> Optional[str]:
        """Detect CUDA version."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=self.config.advanced.version_check_timeout
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line.lower():
                        # Extract version from line like "Cuda compilation tools, release 11.8, V11.8.89"
                        parts = line.split("release")
                        if len(parts) > 1:
                            version = parts[1].split(",")[0].strip()
                            return version
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass
        return None

    def _detect_container_runtime(self) -> Optional[str]:
        """Detect available container runtime."""
        # In fast profile mode, skip container runtime probing entirely
        if self._fast_profile:
            return None

        eff_timeout = self._effective_cmd_timeout()
        for runtime in ["docker", "podman"]:
            # Skip if binary not present
            if shutil.which(runtime) is None:
                continue
            try:
                result = subprocess.run(
                    [runtime, "--version"], capture_output=True, text=True, timeout=eff_timeout
                )
                if result.returncode == 0:
                    return runtime
            except (
                subprocess.TimeoutExpired,
                subprocess.CalledProcessError,
                FileNotFoundError,
            ):
                continue
        return None

    def _run_command(self, command: List[str], timeout: Optional[int] = None) -> Optional[str]:
        """Get version from command output with safety guards.

        - Respects configured timeouts
        - Skips invocation if binary not present
        - Extracts semantic version when possible
        """
        # Skip if the executable is not available to avoid spawn errors or hangs
        if not command or shutil.which(command[0]) is None:
            return None

        # Determine timeout: prefer provided, else config-derived
        eff_timeout = timeout if timeout is not None else self._effective_cmd_timeout()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=eff_timeout,
            )
            if result.returncode == 0:
                # Extract version number from output
                output = result.stdout.strip()
                # Simple version extraction - look for patterns like "1.2.3"
                import re

                version_match = re.search(r"(\d+\.\d+\.\d+)", output)
                if version_match:
                    return version_match.group(1)
                return output.split("\n")[0]  # Return first line if no version pattern
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass
        return None

    def _effective_cmd_timeout(self) -> int:
        """Compute a conservative timeout for version commands.

        Uses the lower of version_check_timeout and profile_agent_timeout (rounded),
        with a minimum of 1 second to avoid zero timeouts.
        """
        try:
            # profile_agent_timeout is a float (performance target), use it as an upper bound
            perf_cap = max(1, int(self.performance_monitor.performance_targets.get("profile_agent", 2)))
        except Exception:
            perf_cap = 2

        # Configured version check timeout
        cfg_timeout = getattr(self.config.advanced, "version_check_timeout", 5)

        return max(1, min(perf_cap, int(cfg_timeout)))

    def _calculate_compute_score(self) -> float:
        """Calculate overall compute capability score."""
        # Simple scoring algorithm - can be enhanced
        score = 0.0

        # CPU contribution (0-30 points)
        cpu_cores = psutil.cpu_count(logical=False) or 1
        score += min(cpu_cores * 2, 30)

        # Memory contribution (0-20 points)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        score += min(memory_gb, 20)

        # GPU contribution (0-50 points)
        gpus = self._detect_gpus()
        if gpus:
            for gpu in gpus:
                score += min(gpu.memory_gb * 2, 25)  # Up to 25 points per GPU

        return int(min(score, 100.0))  # Cap at 100 and return integer
