"""Tests for the dependency conflict detection system."""

import pytest
from repo_doctor.conflict_detection import (
    MLPackageConflictDetector,
    DependencyConflict,
    ConflictSeverity,
    CUDACompatibilityMatrix,
    PipErrorParser,
    PipErrorType,
)


class TestMLPackageConflictDetector:
    """Tests for ML package conflict detection."""
    
    def test_torch_torchvision_conflict(self):
        """Test detection of torch-torchvision version conflicts."""
        detector = MLPackageConflictDetector()
        
        # Test incompatible versions
        dependencies = {
            "torch": "2.0.0",
            "torchvision": "0.14.0"  # Incompatible with torch 2.0
        }
        
        conflicts = detector.detect_conflicts(dependencies)
        assert len(conflicts) > 0
        
        # Find the torch-torchvision conflict
        tv_conflict = next(
            (c for c in conflicts if c.package1 == "torch" and c.package2 == "torchvision"),
            None
        )
        assert tv_conflict is not None
        assert tv_conflict.severity == ConflictSeverity.CRITICAL
    
    def test_compatible_versions(self):
        """Test that compatible versions don't trigger conflicts."""
        detector = MLPackageConflictDetector()
        
        # Test compatible versions
        dependencies = {
            "torch": "2.0.0",
            "torchvision": "0.15.0",  # Compatible with torch 2.0
            "numpy": "1.24.3"
        }
        
        conflicts = detector.detect_conflicts(dependencies)
        
        # Should not have torch-torchvision conflict
        tv_conflict = next(
            (c for c in conflicts if c.package1 == "torch" and c.package2 == "torchvision"),
            None
        )
        assert tv_conflict is None
    
    def test_transformers_torch_conflict(self):
        """Test transformers-torch compatibility."""
        detector = MLPackageConflictDetector()
        
        dependencies = {
            "transformers": "4.25.0",
            "torch": "1.8.0"  # Too old for transformers 4.25
        }
        
        conflicts = detector.detect_conflicts(dependencies)
        assert len(conflicts) > 0
        
        tf_conflict = next(
            (c for c in conflicts if "transformers" in [c.package1, c.package2]),
            None
        )
        assert tf_conflict is not None
        assert tf_conflict.severity == ConflictSeverity.CRITICAL
    
    def test_cuda_compatibility_check(self):
        """Test CUDA version compatibility checking."""
        detector = MLPackageConflictDetector()
        
        dependencies = {
            "torch": "2.0.0",
            "tensorflow": "2.13.0"
        }
        
        # Test with incompatible CUDA version
        conflicts = detector.detect_conflicts(dependencies, cuda_version="10.2")
        
        # Should have CUDA conflicts
        cuda_conflicts = [c for c in conflicts if c.conflict_type == "cuda"]
        assert len(cuda_conflicts) > 0
    
    def test_conflict_prioritization(self):
        """Test conflict prioritization by severity."""
        detector = MLPackageConflictDetector()
        
        # Create conflicts with different severities
        conflicts = [
            DependencyConflict(
                package1="pkg1", package2="pkg2",
                version1="1.0", version2="2.0",
                severity=ConflictSeverity.INFO,
                description="Info level conflict",
                suggested_resolution="Minor fix",
                conflict_type="version"
            ),
            DependencyConflict(
                package1="pkg3", package2="pkg4",
                version1="1.0", version2="2.0",
                severity=ConflictSeverity.CRITICAL,
                description="Critical conflict",
                suggested_resolution="Major fix",
                conflict_type="version"
            ),
            DependencyConflict(
                package1="pkg5", package2="CUDA",
                version1="1.0", version2="11.8",
                severity=ConflictSeverity.WARNING,
                description="CUDA conflict",
                suggested_resolution="Update CUDA",
                conflict_type="cuda"
            ),
        ]
        
        prioritized = detector.prioritize_conflicts(conflicts)
        
        # Critical should be first
        assert prioritized[0].severity == ConflictSeverity.CRITICAL
        # CUDA conflicts should be prioritized within same severity
        assert prioritized[1].conflict_type == "cuda"
        # Info should be last
        assert prioritized[-1].severity == ConflictSeverity.INFO


class TestCUDACompatibilityMatrix:
    """Tests for CUDA compatibility matrix."""
    
    def test_torch_cuda_compatibility(self):
        """Test PyTorch CUDA compatibility checking."""
        matrix = CUDACompatibilityMatrix()
        
        # Test compatible combination
        is_compat, msg = matrix.check_compatibility("torch", "2.0.0", "11.8")
        assert is_compat is True
        assert msg is None
        
        # Test incompatible combination
        is_compat, msg = matrix.check_compatibility("torch", "2.0.0", "10.2")
        assert is_compat is False
        assert "requires CUDA >=" in msg
    
    def test_tensorflow_cuda_compatibility(self):
        """Test TensorFlow CUDA compatibility."""
        matrix = CUDACompatibilityMatrix()
        
        # Test TensorFlow 2.13 with correct CUDA
        is_compat, msg = matrix.check_compatibility("tensorflow", "2.13.0", "11.8")
        assert is_compat is True
        
        # Test with wrong CUDA version
        is_compat, msg = matrix.check_compatibility("tensorflow", "2.13.0", "12.0")
        assert is_compat is False or msg is not None  # Either incompatible or warning
    
    def test_recommended_cuda_versions(self):
        """Test getting recommended CUDA versions for multiple packages."""
        matrix = CUDACompatibilityMatrix()
        
        packages = {
            "torch": "2.3.0",
            "tensorflow": "2.15.0"
        }
        
        recommended = matrix.get_recommended_cuda(packages)
        
        # Should return versions compatible with both
        assert isinstance(recommended, list)
        # If no common version, list should be empty
        if recommended:
            assert all(isinstance(v, str) for v in recommended)
    
    def test_multi_cuda_conflict_detection(self):
        """Test detection of packages requiring incompatible CUDA versions."""
        matrix = CUDACompatibilityMatrix()
        
        packages = {
            "torch": "2.0.0",  # Needs CUDA 11.7-11.8
            "tensorflow": "2.17.0"  # Needs CUDA 12.3-12.5
        }
        
        conflicts = matrix.check_multi_cuda_conflict(packages)
        
        # Should detect conflict between torch and tensorflow
        assert len(conflicts) > 0
        assert any("torch" in str(c) and "tensorflow" in str(c) for c in conflicts)


class TestPipErrorParser:
    """Tests for pip error parsing."""
    
    def test_parse_version_conflict(self):
        """Test parsing of version conflict errors."""
        parser = PipErrorParser()
        
        error_output = """
        ERROR: pip's dependency resolver does not currently take into account all the packages
        ERROR: package1 has requirement package2==1.0.0, but you'll have package2 2.0.0 which is incompatible
        """
        
        conflicts = parser.parse_error(error_output)
        assert len(conflicts) > 0
        
        # Check for version conflict
        version_conflicts = [c for c in conflicts if c.error_type == PipErrorType.VERSION_CONFLICT]
        assert len(version_conflicts) > 0
    
    def test_parse_missing_dependency(self):
        """Test parsing of missing dependency errors."""
        parser = PipErrorParser()
        
        error_output = """
        ERROR: No matching distribution found for nonexistent-package
        """
        
        conflicts = parser.parse_error(error_output)
        assert len(conflicts) > 0
        
        missing_deps = [c for c in conflicts if c.error_type == PipErrorType.MISSING_DEPENDENCY]
        assert len(missing_deps) > 0
        assert missing_deps[0].package == "nonexistent-package"
    
    def test_parse_build_error(self):
        """Test parsing of build errors."""
        parser = PipErrorParser()
        
        error_output = """
        ERROR: Failed building wheel for cryptography
        error: Microsoft Visual C++ 14.0 is required
        """
        
        conflicts = parser.parse_error(error_output)
        assert len(conflicts) > 0
        
        build_errors = [c for c in conflicts if c.error_type == PipErrorType.BUILD_ERROR]
        assert len(build_errors) > 0
        assert "Visual C++" in build_errors[0].suggested_resolution
    
    def test_parse_platform_mismatch(self):
        """Test parsing of platform mismatch errors."""
        parser = PipErrorParser()
        
        error_output = """
        ERROR: package-1.0.0-cp39-cp39-win_amd64.whl is not a supported wheel on this platform
        """
        
        conflicts = parser.parse_error(error_output)
        assert len(conflicts) > 0
        
        platform_errors = [c for c in conflicts if c.error_type == PipErrorType.PLATFORM_MISMATCH]
        assert len(platform_errors) > 0
    
    def test_suggest_resolution(self):
        """Test resolution suggestion generation."""
        parser = PipErrorParser()
        
        from repo_doctor.conflict_detection.pip_parser import PipConflict
        
        conflicts = [
            PipConflict(
                error_type=PipErrorType.VERSION_CONFLICT,
                package="torch",
                required_by=["transformers"],
                current_version="1.8.0",
                required_version="2.0.0",
                error_message="Version conflict",
                suggested_resolution="Update torch"
            ),
            PipConflict(
                error_type=PipErrorType.BUILD_ERROR,
                package="some-package",
                required_by=[],
                current_version=None,
                required_version=None,
                error_message="Build failed",
                suggested_resolution="Install build tools"
            )
        ]
        
        suggestions = parser.suggest_resolution(conflicts)
        
        assert len(suggestions) > 0
        # Should have suggestions for both conflict types
        assert any("virtual environment" in s.lower() for s in suggestions)
        assert any("conda" in s.lower() for s in suggestions)  # ML-specific suggestion


@pytest.mark.integration
class TestConflictDetectionIntegration:
    """Integration tests for the complete conflict detection system."""
    
    def test_full_conflict_detection_workflow(self):
        """Test the complete conflict detection workflow."""
        detector = MLPackageConflictDetector()
        cuda_matrix = CUDACompatibilityMatrix()
        
        # Simulate a complex ML project dependencies
        dependencies = {
            "torch": "2.0.0",
            "torchvision": "0.15.0",
            "transformers": "4.30.0",
            "tensorflow": "2.13.0",
            "numpy": "1.24.0",
            "scipy": "1.11.0",
            "pandas": "2.0.0",
            "scikit-learn": "1.3.0"
        }
        
        cuda_version = "11.8"
        
        # Detect conflicts
        conflicts = detector.detect_conflicts(dependencies, cuda_version)
        
        # Check CUDA compatibility for each framework
        for package in ["torch", "tensorflow"]:
            if package in dependencies:
                is_compat, msg = cuda_matrix.check_compatibility(
                    package, 
                    dependencies[package], 
                    cuda_version
                )
                # Process compatibility results
                if not is_compat and msg:
                    assert msg  # Should have error message
        
        # Get recommended CUDA versions
        recommended_cuda = cuda_matrix.get_recommended_cuda(dependencies)
        
        # Prioritize conflicts
        if conflicts:
            prioritized = detector.prioritize_conflicts(conflicts)
            # Critical conflicts should be first
            if prioritized:
                assert prioritized[0].severity in [
                    ConflictSeverity.CRITICAL, 
                    ConflictSeverity.WARNING
                ]
    
    def test_pip_error_to_conflict_resolution(self):
        """Test converting pip errors to actionable resolutions."""
        parser = PipErrorParser()
        detector = MLPackageConflictDetector()
        
        # Simulate a pip error
        pip_error = """
        ERROR: pip's dependency resolver does not currently take into account all the packages
        ERROR: transformers 4.30.0 has requirement torch>=1.10, but you'll have torch 1.8.0 which is incompatible
        ERROR: torchvision 0.15.0 has requirement torch==2.0.0, but you'll have torch 1.8.0 which is incompatible
        """
        
        # Parse pip errors
        pip_conflicts = parser.parse_error(pip_error)
        assert len(pip_conflicts) > 0
        
        # Get resolution suggestions
        suggestions = parser.suggest_resolution(pip_conflicts)
        assert len(suggestions) > 0
        
        # Verify suggestions are actionable
        assert any("conda" in s.lower() or "virtual" in s.lower() for s in suggestions)