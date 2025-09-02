"""Resolution and strategy models."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class StrategyType(str, Enum):
    """Types of resolution strategies."""
    DOCKER = "docker"
    CONDA = "conda"
    VENV = "venv"
    DEVCONTAINER = "devcontainer"


class ValidationStatus(str, Enum):
    """Validation status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class GeneratedFile(BaseModel):
    """Generated file information."""
    path: str
    content: str
    description: str
    executable: bool = False


class ValidationResult(BaseModel):
    """Result of solution validation."""
    status: ValidationStatus
    duration: float = 0.0
    logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    container_id: Optional[str] = None


class Strategy(BaseModel):
    """Resolution strategy configuration."""
    type: StrategyType
    priority: int = 0
    requirements: Dict[str, Any] = Field(default_factory=dict)
    can_handle_gpu: bool = False
    estimated_setup_time: int = 0  # seconds


class Resolution(BaseModel):
    """Complete resolution with generated artifacts."""
    strategy: Strategy
    generated_files: List[GeneratedFile] = Field(default_factory=list)
    setup_commands: List[str] = Field(default_factory=list)
    validation_result: Optional[ValidationResult] = None
    instructions: str = ""
    estimated_size_mb: int = 0
    
    def is_validated(self) -> bool:
        """Check if resolution has been validated."""
        return (self.validation_result is not None and 
                self.validation_result.status == ValidationStatus.SUCCESS)
    
    def get_file_by_name(self, filename: str) -> Optional[GeneratedFile]:
        """Get generated file by name."""
        for file in self.generated_files:
            if file.path.endswith(filename):
                return file
        return None
