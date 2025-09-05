"""CUDA compatibility matrix for ML frameworks."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CUDARequirement:
    """Represents CUDA requirements for a package version."""
    
    min_cuda: str
    max_cuda: Optional[str]
    recommended_cuda: List[str]
    compute_capability: Optional[str] = None


class CUDACompatibilityMatrix:
    """Manages CUDA compatibility information for ML packages."""
    
    def __init__(self) -> None:
        """Initialize with comprehensive CUDA compatibility data."""
        # Comprehensive CUDA compatibility matrix
        self.compatibility_matrix = {
            'torch': {
                # PyTorch 2.x series
                '2.5.0': CUDARequirement('11.8', '12.6', ['12.1', '12.4']),
                '2.4.1': CUDARequirement('11.8', '12.4', ['12.1', '12.4']),
                '2.4.0': CUDARequirement('11.8', '12.4', ['12.1', '12.4']),
                '2.3.1': CUDARequirement('11.8', '12.4', ['11.8', '12.1']),
                '2.3.0': CUDARequirement('11.8', '12.4', ['11.8', '12.1']),
                '2.2.2': CUDARequirement('11.8', '12.1', ['11.8', '12.1']),
                '2.2.1': CUDARequirement('11.8', '12.1', ['11.8', '12.1']),
                '2.2.0': CUDARequirement('11.8', '12.1', ['11.8', '12.1']),
                '2.1.2': CUDARequirement('11.8', '12.1', ['11.8', '12.1']),
                '2.1.1': CUDARequirement('11.8', '12.1', ['11.8', '12.1']),
                '2.1.0': CUDARequirement('11.8', '12.1', ['11.8', '12.1']),
                '2.0.1': CUDARequirement('11.7', '11.8', ['11.7', '11.8']),
                '2.0.0': CUDARequirement('11.7', '11.8', ['11.7', '11.8']),
                # PyTorch 1.x series
                '1.13.1': CUDARequirement('11.6', '11.7', ['11.6', '11.7']),
                '1.13.0': CUDARequirement('11.6', '11.7', ['11.6', '11.7']),
                '1.12.1': CUDARequirement('10.2', '11.6', ['11.3', '11.6']),
                '1.12.0': CUDARequirement('10.2', '11.6', ['11.3', '11.6']),
            },
            'tensorflow': {
                # TensorFlow 2.x series
                '2.17.0': CUDARequirement('12.3', '12.5', ['12.3']),
                '2.16.2': CUDARequirement('12.2', '12.3', ['12.3']),
                '2.16.1': CUDARequirement('12.2', '12.3', ['12.3']),
                '2.16.0': CUDARequirement('12.2', '12.3', ['12.3']),
                '2.15.1': CUDARequirement('11.8', '12.2', ['12.2']),
                '2.15.0': CUDARequirement('11.8', '12.2', ['12.2']),
                '2.14.1': CUDARequirement('11.8', '11.8', ['11.8']),
                '2.14.0': CUDARequirement('11.8', '11.8', ['11.8']),
                '2.13.1': CUDARequirement('11.8', '11.8', ['11.8']),
                '2.13.0': CUDARequirement('11.8', '11.8', ['11.8']),
                '2.12.0': CUDARequirement('11.8', '11.8', ['11.8']),
                '2.11.0': CUDARequirement('11.2', '11.2', ['11.2']),
            },
            'jax': {
                # JAX versions
                '0.4.25': CUDARequirement('12.2', '12.3', ['12.3']),
                '0.4.24': CUDARequirement('12.2', '12.3', ['12.3']),
                '0.4.23': CUDARequirement('11.8', '12.3', ['12.2']),
                '0.4.20': CUDARequirement('11.8', '12.2', ['11.8', '12.2']),
                '0.4.14': CUDARequirement('11.8', '12.0', ['11.8']),
                '0.4.13': CUDARequirement('11.8', '11.8', ['11.8']),
            },
            'cupy': {
                # CuPy versions (direct CUDA bindings)
                '13.0.0': CUDARequirement('11.2', '12.3', ['12.x']),
                '12.3.0': CUDARequirement('11.2', '12.2', ['11.8', '12.x']),
                '12.0.0': CUDARequirement('11.2', '12.0', ['11.8']),
                '11.6.0': CUDARequirement('10.2', '11.8', ['11.2', '11.8']),
            },
            'mxnet': {
                # Apache MXNet
                '1.9.1': CUDARequirement('10.2', '11.2', ['11.2']),
                '1.9.0': CUDARequirement('10.2', '11.2', ['11.2']),
                '1.8.0': CUDARequirement('10.1', '11.0', ['10.2']),
            },
        }
        
        # CUDA version to compute capability mapping
        self.cuda_compute_capability = {
            '10.0': '3.5',
            '10.1': '3.5',
            '10.2': '3.5',
            '11.0': '3.5',
            '11.1': '3.5',
            '11.2': '3.5',
            '11.3': '3.5',
            '11.4': '3.5',
            '11.5': '3.5',
            '11.6': '3.5',
            '11.7': '3.5',
            '11.8': '3.5',
            '12.0': '5.0',
            '12.1': '5.0',
            '12.2': '5.0',
            '12.3': '5.0',
            '12.4': '5.0',
            '12.5': '5.0',
            '12.6': '5.0',
        }
        
        # Known CUDA runtime conflicts
        self.runtime_conflicts = {
            ('11.x', '12.x'): "CUDA 11.x and 12.x runtime libraries are incompatible",
            ('10.x', '11.x'): "CUDA 10.x and 11.x have ABI incompatibilities",
        }
    
    def check_compatibility(
        self, 
        package: str, 
        package_version: str, 
        cuda_version: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a package version is compatible with CUDA version.
        
        Args:
            package: Package name (e.g., 'torch')
            package_version: Package version (e.g., '2.0.0')
            cuda_version: CUDA version (e.g., '11.8')
            
        Returns:
            Tuple of (is_compatible, error_message)
        """
        if package not in self.compatibility_matrix:
            return True, None  # Unknown package, assume compatible
        
        package_matrix = self.compatibility_matrix[package]
        
        # Find exact or closest version match
        if package_version in package_matrix:
            req = package_matrix[package_version]
        else:
            # Find closest version
            closest = self._find_closest_version(
                package_version, 
                list(package_matrix.keys())
            )
            if not closest:
                return True, None  # No version info, assume compatible
            req = package_matrix[closest]
        
        # Check CUDA compatibility
        cuda_major_minor = '.'.join(cuda_version.split('.')[:2])
        
        # Check if CUDA version is in range
        if req.min_cuda and self._compare_versions(cuda_major_minor, req.min_cuda) < 0:
            return False, f"{package} {package_version} requires CUDA >= {req.min_cuda}, but got {cuda_version}"
        
        if req.max_cuda and self._compare_versions(cuda_major_minor, req.max_cuda) > 0:
            return False, f"{package} {package_version} requires CUDA <= {req.max_cuda}, but got {cuda_version}"
        
        # Check if it's a recommended version
        if cuda_major_minor not in req.recommended_cuda:
            rec_str = ', '.join(req.recommended_cuda)
            msg = f"{package} {package_version} works best with CUDA {rec_str}, using {cuda_version} may cause issues"
            return True, msg  # Compatible but not recommended
        
        return True, None
    
    def get_recommended_cuda(
        self, 
        packages: Dict[str, str]
    ) -> List[str]:
        """
        Get recommended CUDA versions for a set of packages.
        
        Args:
            packages: Dictionary of package names to versions
            
        Returns:
            List of recommended CUDA versions that work with all packages
        """
        all_recommended = None
        
        for package, version in packages.items():
            if package not in self.compatibility_matrix:
                continue
            
            package_matrix = self.compatibility_matrix[package]
            
            if version in package_matrix:
                req = package_matrix[version]
            else:
                closest = self._find_closest_version(version, list(package_matrix.keys()))
                if not closest:
                    continue
                req = package_matrix[closest]
            
            if all_recommended is None:
                all_recommended = set(req.recommended_cuda)
            else:
                all_recommended &= set(req.recommended_cuda)
        
        if all_recommended:
            return sorted(list(all_recommended))
        return []
    
    def check_multi_cuda_conflict(
        self, 
        packages: Dict[str, str]
    ) -> List[Tuple[str, str, str]]:
        """
        Check for packages requiring incompatible CUDA versions.
        
        Args:
            packages: Dictionary of package names to versions
            
        Returns:
            List of (package1, package2, conflict_description) tuples
        """
        conflicts = []
        package_cuda_reqs = {}
        
        # Collect CUDA requirements for each package
        for package, version in packages.items():
            if package not in self.compatibility_matrix:
                continue
            
            package_matrix = self.compatibility_matrix[package]
            
            if version in package_matrix:
                req = package_matrix[version]
            else:
                closest = self._find_closest_version(version, list(package_matrix.keys()))
                if not closest:
                    continue
                req = package_matrix[closest]
            
            package_cuda_reqs[package] = req
        
        # Check for conflicts between packages
        packages_list = list(package_cuda_reqs.keys())
        for i in range(len(packages_list)):
            for j in range(i + 1, len(packages_list)):
                pkg1, pkg2 = packages_list[i], packages_list[j]
                req1, req2 = package_cuda_reqs[pkg1], package_cuda_reqs[pkg2]
                
                # Check if there's any overlap in recommended versions
                overlap = set(req1.recommended_cuda) & set(req2.recommended_cuda)
                if not overlap:
                    conflicts.append((
                        f"{pkg1} {packages[pkg1]}",
                        f"{pkg2} {packages[pkg2]}",
                        f"No common CUDA version: {pkg1} needs {req1.recommended_cuda}, "
                        f"{pkg2} needs {req2.recommended_cuda}"
                    ))
        
        return conflicts
    
    def _find_closest_version(
        self, 
        target: str, 
        versions: List[str]
    ) -> Optional[str]:
        """Find the closest version from available versions."""
        try:
            # Parse target version
            target_parts = [int(x) for x in target.split('.')]
            
            best_match = None
            min_diff = float('inf')
            
            for v in versions:
                v_parts = [int(x) for x in v.split('.')]
                
                # Calculate difference
                diff = 0
                for i in range(max(len(target_parts), len(v_parts))):
                    t = target_parts[i] if i < len(target_parts) else 0
                    vp = v_parts[i] if i < len(v_parts) else 0
                    diff += abs(t - vp) * (100 ** (2 - i))  # Weight by position
                
                if diff < min_diff:
                    min_diff = diff
                    best_match = v
            
            return best_match
        except:
            return None
    
    def _compare_versions(self, v1: str, v2: str) -> int:
        """
        Compare two version strings.
        
        Returns:
            -1 if v1 < v2, 0 if equal, 1 if v1 > v2
        """
        try:
            v1_parts = [int(x) for x in v1.split('.')]
            v2_parts = [int(x) for x in v2.split('.')]
            
            for i in range(max(len(v1_parts), len(v2_parts))):
                p1 = v1_parts[i] if i < len(v1_parts) else 0
                p2 = v2_parts[i] if i < len(v2_parts) else 0
                
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1
            
            return 0
        except:
            return 0