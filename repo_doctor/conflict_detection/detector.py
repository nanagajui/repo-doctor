"""ML/AI package conflict detection system."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from packaging import version
from packaging.requirements import Requirement


class ConflictSeverity(Enum):
    """Severity levels for dependency conflicts."""
    
    CRITICAL = "critical"  # Will definitely break
    WARNING = "warning"    # May cause issues
    INFO = "info"         # Minor compatibility concern


@dataclass
class DependencyConflict:
    """Represents a dependency conflict between two packages."""
    
    package1: str
    package2: str
    version1: str
    version2: str
    severity: ConflictSeverity
    description: str
    suggested_resolution: str
    conflict_type: str = "version"  # version, cuda, system


class MLPackageConflictDetector:
    """Specialized detector for ML/AI package conflicts."""
    
    def __init__(self) -> None:
        """Initialize the conflict detector with known conflict patterns."""
        # Known problematic combinations based on real-world issues
        self.conflict_matrix = {
            # PyTorch ecosystem conflicts
            ('torch', 'torchvision'): {
                'patterns': [
                    # torch 2.0.0+ with torchvision < 0.15.0
                    (r'^2\.\d+\.\d+', r'^0\.1[0-4]\.\d+', ConflictSeverity.CRITICAL),
                    # torch 1.13.x with torchvision 0.15.x+
                    (r'^1\.13\.\d+', r'^0\.1[5-9]\.\d+', ConflictSeverity.CRITICAL),
                ],
                'resolution': 'Use torch 2.0+ with torchvision 0.15+ or torch 1.13.x with torchvision 0.14.x'
            },
            ('torch', 'torchaudio'): {
                'patterns': [
                    # torch 2.0+ with torchaudio < 2.0
                    (r'^2\.\d+\.\d+', r'^0\.\d+\.\d+', ConflictSeverity.CRITICAL),
                ],
                'resolution': 'Use matching major versions: torch 2.x with torchaudio 2.x'
            },
            ('transformers', 'torch'): {
                'patterns': [
                    # transformers 4.20+ requires torch 1.9+
                    (r'^4\.2[0-9]\.\d+', r'^1\.[0-8]\.\d+', ConflictSeverity.CRITICAL),
                    # transformers 4.30+ requires torch 1.10+
                    (r'^4\.[3-9][0-9]\.\d+', r'^1\.9\.\d+', ConflictSeverity.WARNING),
                ],
                'resolution': 'Update torch to 1.10+ for transformers 4.30+'
            },
            ('xformers', 'torch'): {
                'patterns': [
                    # xformers requires specific torch builds
                    (r'^0\.0\.2[0-9]', r'^2\.[0-3]\.\d+', ConflictSeverity.WARNING),
                ],
                'resolution': 'Check xformers compatibility matrix for your torch version'
            },
            ('tensorflow', 'torch'): {
                'patterns': [
                    # Generally compatible but can cause CUDA conflicts
                    (r'.*', r'.*', ConflictSeverity.WARNING),
                ],
                'resolution': 'Consider using separate environments for TensorFlow and PyTorch'
            },
            ('ultralytics', 'torch'): {
                'patterns': [
                    # ultralytics 8.3.40 doesn't support torch 2.4.0
                    (r'^8\.3\.40', r'^2\.4\.\d+', ConflictSeverity.CRITICAL),
                ],
                'resolution': 'Use ultralytics 8.3.41+ with torch 2.4.0+'
            },
            ('accelerate', 'transformers'): {
                'patterns': [
                    # accelerate 0.20+ requires transformers 4.28+
                    (r'^0\.2[0-9]\.\d+', r'^4\.[0-2][0-7]\.\d+', ConflictSeverity.WARNING),
                ],
                'resolution': 'Update transformers to 4.28+ for accelerate 0.20+'
            },
            ('diffusers', 'transformers'): {
                'patterns': [
                    # diffusers 0.20+ requires transformers 4.25+
                    (r'^0\.2[0-9]\.\d+', r'^4\.2[0-4]\.\d+', ConflictSeverity.WARNING),
                ],
                'resolution': 'Update transformers to 4.25+ for diffusers 0.20+'
            },
            ('datasets', 'pandas'): {
                'patterns': [
                    # datasets 2.14+ has issues with pandas 2.0+
                    (r'^2\.1[4-9]\.\d+', r'^2\.\d+\.\d+', ConflictSeverity.WARNING),
                ],
                'resolution': 'Use pandas 1.5.x with datasets 2.14+ or wait for updates'
            },
            ('scipy', 'numpy'): {
                'patterns': [
                    # scipy 1.11+ requires numpy 1.21+
                    (r'^1\.1[1-9]\.\d+', r'^1\.[0-2]0\.\d+', ConflictSeverity.CRITICAL),
                ],
                'resolution': 'Update numpy to 1.21+ for scipy 1.11+'
            },
        }
        
        # CUDA version compatibility for major ML frameworks
        self.cuda_compatibility = {
            'torch': {
                '2.0.0': ['11.7', '11.8'],
                '2.0.1': ['11.7', '11.8'],
                '2.1.0': ['11.8', '12.1'],
                '2.1.1': ['11.8', '12.1'],
                '2.1.2': ['11.8', '12.1'],
                '2.2.0': ['11.8', '12.1'],
                '2.2.1': ['11.8', '12.1'],
                '2.2.2': ['11.8', '12.1'],
                '2.3.0': ['11.8', '12.1', '12.4'],
                '2.3.1': ['11.8', '12.1', '12.4'],
                '2.4.0': ['11.8', '12.1', '12.4'],
                '2.4.1': ['11.8', '12.1', '12.4'],
                '2.5.0': ['11.8', '12.1', '12.4', '12.6'],
            },
            'tensorflow': {
                '2.13.0': ['11.8'],
                '2.14.0': ['11.8'],
                '2.15.0': ['11.8', '12.2'],
                '2.16.0': ['12.2', '12.3'],
                '2.17.0': ['12.3', '12.5'],
            },
            'jax': {
                '0.4.20': ['11.8', '12.2'],
                '0.4.23': ['11.8', '12.2', '12.3'],
                '0.4.25': ['12.2', '12.3'],
            }
        }
    
    def detect_conflicts(
        self, 
        dependencies: Dict[str, str], 
        cuda_version: Optional[str] = None
    ) -> List[DependencyConflict]:
        """
        Detect dependency conflicts in the given dependency set.
        
        Args:
            dependencies: Dictionary of package names to version specifications
            cuda_version: Optional CUDA version for compatibility checking
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Check pairwise conflicts
        for (pkg1, pkg2), conflict_info in self.conflict_matrix.items():
            if pkg1 in dependencies and pkg2 in dependencies:
                v1, v2 = dependencies[pkg1], dependencies[pkg2]
                
                # Remove version specifiers for comparison
                v1_clean = self._clean_version(v1)
                v2_clean = self._clean_version(v2)
                
                for pattern1, pattern2, severity in conflict_info['patterns']:
                    if (re.match(pattern1, v1_clean) and 
                        re.match(pattern2, v2_clean)):
                        conflicts.append(DependencyConflict(
                            package1=pkg1,
                            package2=pkg2,
                            version1=v1,
                            version2=v2,
                            severity=severity,
                            description=f"Version conflict between {pkg1} {v1} and {pkg2} {v2}",
                            suggested_resolution=conflict_info['resolution'],
                            conflict_type="version"
                        ))
        
        # Check CUDA compatibility if provided
        if cuda_version:
            cuda_conflicts = self._check_cuda_compatibility(dependencies, cuda_version)
            conflicts.extend(cuda_conflicts)
        
        # Check for known problematic version ranges
        range_conflicts = self._check_version_ranges(dependencies)
        conflicts.extend(range_conflicts)
        
        return conflicts
    
    def _clean_version(self, version_spec: str) -> str:
        """Clean version specification to extract actual version number."""
        # Remove common version specifiers
        version_clean = re.sub(r'^[><=~^!]+', '', version_spec)
        version_clean = re.sub(r'[,<>=~^!].*$', '', version_clean)
        return version_clean.strip()
    
    def _check_cuda_compatibility(
        self, 
        dependencies: Dict[str, str], 
        cuda_version: str
    ) -> List[DependencyConflict]:
        """Check CUDA compatibility for ML packages."""
        conflicts = []
        cuda_major_minor = '.'.join(cuda_version.split('.')[:2])
        
        for package, compat_matrix in self.cuda_compatibility.items():
            if package in dependencies:
                pkg_version = self._clean_version(dependencies[package])
                
                # Find closest version in compatibility matrix
                closest_version = self._find_closest_version(
                    pkg_version, 
                    list(compat_matrix.keys())
                )
                
                if closest_version and cuda_major_minor not in compat_matrix[closest_version]:
                    supported_cuda = ', '.join(compat_matrix[closest_version])
                    conflicts.append(DependencyConflict(
                        package1=package,
                        package2="CUDA",
                        version1=dependencies[package],
                        version2=cuda_version,
                        severity=ConflictSeverity.CRITICAL,
                        description=f"{package} {pkg_version} is not compatible with CUDA {cuda_version}",
                        suggested_resolution=f"Use CUDA {supported_cuda} or adjust {package} version",
                        conflict_type="cuda"
                    ))
        
        return conflicts
    
    def _find_closest_version(
        self, 
        target_version: str, 
        available_versions: List[str]
    ) -> Optional[str]:
        """Find the closest available version to the target."""
        try:
            target = version.parse(target_version)
            closest = None
            min_diff = float('inf')
            
            for av in available_versions:
                try:
                    av_parsed = version.parse(av)
                    # Simple distance metric
                    diff = abs(float(str(target).replace('.', '')) - 
                              float(str(av_parsed).replace('.', '')))
                    if diff < min_diff:
                        min_diff = diff
                        closest = av
                except:
                    continue
            
            return closest
        except:
            # If parsing fails, return the first available version
            return available_versions[0] if available_versions else None
    
    def _check_version_ranges(self, dependencies: Dict[str, str]) -> List[DependencyConflict]:
        """Check for known problematic version ranges."""
        conflicts = []
        
        # Known problematic packages with specific version issues
        problematic_versions = {
            'numpy': {
                '1.24.0': (ConflictSeverity.WARNING, 
                          "numpy 1.24.0 has breaking API changes",
                          "Consider using numpy<1.24 or >=1.24.1"),
            },
            'pandas': {
                '2.0.0': (ConflictSeverity.WARNING,
                         "pandas 2.0.0 has significant breaking changes",
                         "Review migration guide if upgrading from 1.x"),
            },
            'scikit-learn': {
                '1.3.0': (ConflictSeverity.INFO,
                         "scikit-learn 1.3.0 deprecates several features",
                         "Check deprecation warnings in your code"),
            },
        }
        
        for package, version_spec in dependencies.items():
            if package in problematic_versions:
                clean_version = self._clean_version(version_spec)
                for prob_version, (severity, desc, resolution) in problematic_versions[package].items():
                    if clean_version == prob_version:
                        conflicts.append(DependencyConflict(
                            package1=package,
                            package2="",
                            version1=version_spec,
                            version2="",
                            severity=severity,
                            description=desc,
                            suggested_resolution=resolution,
                            conflict_type="version"
                        ))
        
        return conflicts
    
    def prioritize_conflicts(
        self, 
        conflicts: List[DependencyConflict]
    ) -> List[DependencyConflict]:
        """
        Prioritize conflicts by severity and impact.
        
        Args:
            conflicts: List of detected conflicts
            
        Returns:
            Sorted list of conflicts by priority
        """
        # Define priority order
        severity_order = {
            ConflictSeverity.CRITICAL: 0,
            ConflictSeverity.WARNING: 1,
            ConflictSeverity.INFO: 2
        }
        
        # Sort by severity, then by conflict type (cuda conflicts are high priority)
        return sorted(
            conflicts,
            key=lambda c: (
                severity_order[c.severity],
                0 if c.conflict_type == "cuda" else 1,
                c.package1
            )
        )