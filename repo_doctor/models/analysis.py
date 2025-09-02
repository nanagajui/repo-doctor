"""Repository analysis models."""

from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field
from enum import Enum


class DependencyType(str, Enum):
    """Types of dependencies."""
    PYTHON = "python"
    CONDA = "conda"
    SYSTEM = "system"
    GPU = "gpu"


class DependencyInfo(BaseModel):
    """Dependency information."""
    name: str
    version: Optional[str] = None
    type: DependencyType
    source: str  # requirements.txt, setup.py, etc.
    optional: bool = False
    gpu_required: bool = False


class RepositoryInfo(BaseModel):
    """Repository metadata."""
    url: str
    name: str
    owner: str
    description: Optional[str] = None
    stars: int = 0
    language: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    has_dockerfile: bool = False
    has_conda_env: bool = False
    has_requirements: bool = False
    has_setup_py: bool = False
    has_pyproject_toml: bool = False


class CompatibilityIssue(BaseModel):
    """Compatibility issue found during analysis."""
    type: str  # version_conflict, missing_dependency, gpu_mismatch, etc.
    severity: str  # critical, warning, info
    message: str
    component: str  # package name, system requirement, etc.
    suggested_fix: Optional[str] = None


class Analysis(BaseModel):
    """Complete repository analysis."""
    repository: RepositoryInfo
    dependencies: List[DependencyInfo] = Field(default_factory=list)
    python_version_required: Optional[str] = None
    cuda_version_required: Optional[str] = None
    min_memory_gb: float = 0.0
    min_gpu_memory_gb: float = 0.0
    compatibility_issues: List[CompatibilityIssue] = Field(default_factory=list)
    analysis_time: float = 0.0
    confidence_score: float = 0.0
    
    def get_critical_issues(self) -> List[CompatibilityIssue]:
        """Get critical compatibility issues."""
        return [issue for issue in self.compatibility_issues if issue.severity == "critical"]
    
    def is_gpu_required(self) -> bool:
        """Check if GPU is required."""
        return any(dep.gpu_required for dep in self.dependencies) or self.min_gpu_memory_gb > 0
    
    def get_python_dependencies(self) -> List[DependencyInfo]:
        """Get Python package dependencies."""
        return [dep for dep in self.dependencies if dep.type == DependencyType.PYTHON]
