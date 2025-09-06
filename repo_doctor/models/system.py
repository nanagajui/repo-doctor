"""System profiling models."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class GPUInfo(BaseModel):
    """GPU information."""

    name: str
    memory_gb: float
    cuda_version: Optional[str] = None
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None


class HardwareInfo(BaseModel):
    """Hardware information."""

    cpu_cores: int
    memory_gb: float
    gpus: List[GPUInfo] = Field(default_factory=list)
    architecture: str  # x86_64, arm64, etc.


class SoftwareStack(BaseModel):
    """Software stack information."""

    python_version: str
    pip_version: Optional[str] = None
    conda_version: Optional[str] = None
    micromamba_version: Optional[str] = None
    docker_version: Optional[str] = None
    git_version: Optional[str] = None
    cuda_version: Optional[str] = None
    installed_packages: Dict[str, str] = Field(default_factory=dict)


class SystemProfile(BaseModel):
    """Complete system profile."""

    hardware: HardwareInfo
    software: SoftwareStack
    platform: str = Field(default="linux")  # linux, darwin, win32
    container_runtime: Optional[str] = None  # docker, podman, none
    compute_score: float = Field(
        default=0.0, description="Overall compute capability score"
    )

    def has_gpu(self) -> bool:
        """Check if system has GPU."""
        return len(self.hardware.gpus) > 0

    def has_cuda(self) -> bool:
        """Check if CUDA is available."""
        return self.software.cuda_version is not None

    def can_run_containers(self) -> bool:
        """Check if system can run containers."""
        return self.container_runtime is not None
