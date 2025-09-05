"""Parser for pip installation errors and conflict resolution."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PipErrorType(Enum):
    """Types of pip installation errors."""
    
    VERSION_CONFLICT = "version_conflict"
    MISSING_DEPENDENCY = "missing_dependency"
    INCOMPATIBLE_VERSION = "incompatible_version"
    PLATFORM_MISMATCH = "platform_mismatch"
    NETWORK_ERROR = "network_error"
    BUILD_ERROR = "build_error"
    UNKNOWN = "unknown"


@dataclass
class PipConflict:
    """Represents a pip installation conflict."""
    
    error_type: PipErrorType
    package: str
    required_by: List[str]
    current_version: Optional[str]
    required_version: Optional[str]
    error_message: str
    suggested_resolution: str


class PipErrorParser:
    """Parse and analyze pip installation errors."""
    
    def __init__(self) -> None:
        """Initialize the pip error parser with common patterns."""
        # Common pip error patterns
        self.error_patterns = {
            # Version conflict patterns
            r"ERROR: .* has requirement (.+)==([0-9.]+), but you'll have (.+) ([0-9.]+) which is incompatible": 
                self._parse_version_conflict,
            r"ERROR: Cannot install .* because these package versions have conflicting dependencies":
                self._parse_conflicting_dependencies,
            r"ERROR: pip's dependency resolver does not currently take into account all the packages":
                self._parse_resolver_conflict,
            # Missing dependency patterns
            r"ERROR: No matching distribution found for (.+)":
                self._parse_missing_distribution,
            r"ERROR: Could not find a version that satisfies the requirement (.+)":
                self._parse_unsatisfied_requirement,
            # Build error patterns
            r"ERROR: Failed building wheel for (.+)":
                self._parse_build_error,
            r"error: Microsoft Visual C\+\+ .+ is required":
                self._parse_compiler_error,
            # Platform mismatch patterns
            r"ERROR: .+ is not a supported wheel on this platform":
                self._parse_platform_mismatch,
        }
        
        # Resolution strategies for common conflicts
        self.resolution_strategies = {
            PipErrorType.VERSION_CONFLICT: [
                "Try installing packages in a specific order",
                "Use pip install --force-reinstall to override conflicts",
                "Create a fresh virtual environment",
                "Manually specify compatible versions",
            ],
            PipErrorType.MISSING_DEPENDENCY: [
                "Check if the package name is correct",
                "Try upgrading pip: pip install --upgrade pip",
                "Check if you need to use a different index: --index-url or --extra-index-url",
                "The package might not be available for your Python version",
            ],
            PipErrorType.BUILD_ERROR: [
                "Install build dependencies (e.g., gcc, python-dev)",
                "Try installing a pre-built wheel instead",
                "Check if you have the required system libraries",
                "Consider using conda instead for complex packages",
            ],
            PipErrorType.PLATFORM_MISMATCH: [
                "Check your platform and Python version",
                "Look for platform-specific wheels",
                "Build from source if wheels aren't available",
                "Use a Docker container with the correct platform",
            ],
        }
    
    def parse_error(self, error_output: str) -> List[PipConflict]:
        """
        Parse pip error output and extract conflicts.
        
        Args:
            error_output: The error output from pip install
            
        Returns:
            List of identified conflicts
        """
        conflicts = []
        lines = error_output.split('\n')
        
        # Try to match against known patterns
        for line in lines:
            for pattern, parser_func in self.error_patterns.items():
                match = re.search(pattern, line)
                if match:
                    conflict = parser_func(match, error_output)
                    if conflict:
                        conflicts.append(conflict)
        
        # If no specific patterns matched, try generic parsing
        if not conflicts:
            conflicts.extend(self._parse_generic_errors(error_output))
        
        return conflicts
    
    def _parse_version_conflict(self, match: re.Match, full_output: str) -> PipConflict:
        """Parse version conflict errors."""
        required_package = match.group(1)
        required_version = match.group(2)
        installed_package = match.group(3)
        installed_version = match.group(4)
        
        # Extract which package requires this
        required_by = self._extract_required_by(full_output, required_package)
        
        return PipConflict(
            error_type=PipErrorType.VERSION_CONFLICT,
            package=required_package,
            required_by=required_by,
            current_version=installed_version,
            required_version=required_version,
            error_message=match.group(0),
            suggested_resolution=f"Downgrade {installed_package} to {required_version} or find compatible versions"
        )
    
    def _parse_conflicting_dependencies(self, match: re.Match, full_output: str) -> PipConflict:
        """Parse conflicting dependencies error."""
        # Extract the conflict details from subsequent lines
        conflicts_info = self._extract_conflict_details(full_output)
        
        if conflicts_info:
            package, details = conflicts_info[0]
            return PipConflict(
                error_type=PipErrorType.VERSION_CONFLICT,
                package=package,
                required_by=details.get('required_by', []),
                current_version=details.get('current'),
                required_version=details.get('required'),
                error_message=match.group(0),
                suggested_resolution="Resolve version conflicts by finding mutually compatible versions"
            )
        
        return None
    
    def _parse_resolver_conflict(self, match: re.Match, full_output: str) -> PipConflict:
        """Parse dependency resolver conflicts."""
        # This is a generic resolver issue
        return PipConflict(
            error_type=PipErrorType.VERSION_CONFLICT,
            package="multiple",
            required_by=[],
            current_version=None,
            required_version=None,
            error_message=match.group(0),
            suggested_resolution="Use pip install --use-deprecated=legacy-resolver or resolve conflicts manually"
        )
    
    def _parse_missing_distribution(self, match: re.Match, full_output: str) -> PipConflict:
        """Parse missing distribution errors."""
        package = match.group(1)
        
        return PipConflict(
            error_type=PipErrorType.MISSING_DEPENDENCY,
            package=package,
            required_by=self._extract_required_by(full_output, package),
            current_version=None,
            required_version=None,
            error_message=match.group(0),
            suggested_resolution=f"Check if {package} exists on PyPI or use correct package name"
        )
    
    def _parse_unsatisfied_requirement(self, match: re.Match, full_output: str) -> PipConflict:
        """Parse unsatisfied requirement errors."""
        requirement = match.group(1)
        
        # Extract package name from requirement
        package = requirement.split('==')[0].split('>=')[0].split('<=')[0].strip()
        
        return PipConflict(
            error_type=PipErrorType.INCOMPATIBLE_VERSION,
            package=package,
            required_by=[],
            current_version=None,
            required_version=requirement,
            error_message=match.group(0),
            suggested_resolution=f"No compatible version of {package} found for requirement {requirement}"
        )
    
    def _parse_build_error(self, match: re.Match, full_output: str) -> PipConflict:
        """Parse build/compilation errors."""
        package = match.group(1)
        
        # Check for specific build error types
        resolution = "Install build dependencies"
        if "Microsoft Visual C++" in full_output:
            resolution = "Install Microsoft Visual C++ Build Tools"
        elif "gcc" in full_output or "g++" in full_output:
            resolution = "Install gcc/g++ compiler"
        elif "Rust" in full_output:
            resolution = "Install Rust compiler"
        
        return PipConflict(
            error_type=PipErrorType.BUILD_ERROR,
            package=package,
            required_by=[],
            current_version=None,
            required_version=None,
            error_message=match.group(0),
            suggested_resolution=resolution
        )
    
    def _parse_compiler_error(self, match: re.Match, full_output: str) -> PipConflict:
        """Parse compiler requirement errors."""
        return PipConflict(
            error_type=PipErrorType.BUILD_ERROR,
            package="unknown",
            required_by=[],
            current_version=None,
            required_version=None,
            error_message=match.group(0),
            suggested_resolution="Install Microsoft Visual C++ Build Tools from https://visualstudio.microsoft.com/downloads/"
        )
    
    def _parse_platform_mismatch(self, match: re.Match, full_output: str) -> PipConflict:
        """Parse platform mismatch errors."""
        # Extract package name from the error
        package = self._extract_package_from_wheel_error(full_output)
        
        return PipConflict(
            error_type=PipErrorType.PLATFORM_MISMATCH,
            package=package or "unknown",
            required_by=[],
            current_version=None,
            required_version=None,
            error_message=match.group(0),
            suggested_resolution="Download correct wheel for your platform or build from source"
        )
    
    def _parse_generic_errors(self, error_output: str) -> List[PipConflict]:
        """Parse generic or unrecognized errors."""
        conflicts = []
        
        # Look for common error indicators
        if "ResolutionImpossible" in error_output:
            conflicts.append(PipConflict(
                error_type=PipErrorType.VERSION_CONFLICT,
                package="multiple",
                required_by=[],
                current_version=None,
                required_version=None,
                error_message="Resolution impossible due to conflicting dependencies",
                suggested_resolution="Review all package versions and find compatible combinations"
            ))
        
        if "Network" in error_output or "Connection" in error_output:
            conflicts.append(PipConflict(
                error_type=PipErrorType.NETWORK_ERROR,
                package="unknown",
                required_by=[],
                current_version=None,
                required_version=None,
                error_message="Network connectivity issue",
                suggested_resolution="Check internet connection and proxy settings"
            ))
        
        return conflicts
    
    def _extract_required_by(self, full_output: str, package: str) -> List[str]:
        """Extract which packages require the conflicting package."""
        required_by = []
        
        # Look for patterns like "required by X"
        patterns = [
            rf"{package}.*required by ([a-zA-Z0-9\-_]+)",
            rf"([a-zA-Z0-9\-_]+) requires {package}",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, full_output, re.IGNORECASE)
            required_by.extend(matches)
        
        return list(set(required_by))
    
    def _extract_conflict_details(self, full_output: str) -> List[Tuple[str, Dict]]:
        """Extract detailed conflict information from pip output."""
        conflicts = []
        
        # Parse the structured conflict output from modern pip
        conflict_block_pattern = r"The conflict is caused by:(.*?)(?=\n\n|\Z)"
        match = re.search(conflict_block_pattern, full_output, re.DOTALL)
        
        if match:
            block = match.group(1)
            lines = block.strip().split('\n')
            
            for line in lines:
                # Parse lines like "package 1.0.0 depends on other-package>=2.0"
                dep_pattern = r"([a-zA-Z0-9\-_]+) ([0-9.]+) (?:depends on|requires) ([a-zA-Z0-9\-_]+)([><=!]+[0-9.]+)?"
                dep_match = re.search(dep_pattern, line)
                
                if dep_match:
                    package = dep_match.group(3)
                    conflicts.append((package, {
                        'required_by': [dep_match.group(1)],
                        'required': dep_match.group(4) if dep_match.group(4) else None,
                        'current': None
                    }))
        
        return conflicts
    
    def _extract_package_from_wheel_error(self, full_output: str) -> Optional[str]:
        """Extract package name from wheel-related errors."""
        # Look for wheel filename patterns
        wheel_pattern = r"([a-zA-Z0-9\-_]+)-[0-9.]+-.*\.whl"
        match = re.search(wheel_pattern, full_output)
        
        if match:
            return match.group(1)
        
        return None
    
    def suggest_resolution(self, conflicts: List[PipConflict]) -> List[str]:
        """
        Suggest resolution strategies for identified conflicts.
        
        Args:
            conflicts: List of pip conflicts
            
        Returns:
            List of suggested resolution strategies
        """
        suggestions = []
        
        # Group conflicts by type
        conflict_types = set(c.error_type for c in conflicts)
        
        for conflict_type in conflict_types:
            if conflict_type in self.resolution_strategies:
                suggestions.extend(self.resolution_strategies[conflict_type])
        
        # Add specific suggestions based on packages involved
        packages = set(c.package for c in conflicts if c.package != "unknown")
        
        if "torch" in packages or "tensorflow" in packages:
            suggestions.append("Consider using conda for ML packages with complex dependencies")
        
        if any(c.error_type == PipErrorType.VERSION_CONFLICT for c in conflicts):
            suggestions.append("Create a requirements.txt with exact versions that work together")
        
        return list(set(suggestions))  # Remove duplicates