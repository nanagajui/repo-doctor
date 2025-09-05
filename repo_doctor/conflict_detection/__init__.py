"""Dependency conflict detection system for ML/AI packages."""

from .detector import MLPackageConflictDetector, DependencyConflict, ConflictSeverity
from .cuda_matrix import CUDACompatibilityMatrix, CUDARequirement
from .pip_parser import PipErrorParser, PipErrorType, PipConflict

__all__ = [
    "MLPackageConflictDetector",
    "DependencyConflict",
    "ConflictSeverity",
    "CUDACompatibilityMatrix",
    "CUDARequirement",
    "PipErrorParser",
    "PipErrorType",
    "PipConflict",
]