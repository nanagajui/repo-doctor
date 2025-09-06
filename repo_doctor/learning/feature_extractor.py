"""Feature extraction for ML-based learning system."""

import re
from typing import Any, Dict, List, Set
from collections import Counter

from ..models.analysis import Analysis, DependencyInfo
from ..models.resolution import Resolution
from ..models.system import SystemProfile


class FeatureExtractor:
    """Extract ML features from analysis and resolution data."""

    def __init__(self):
        """Initialize feature extractor."""
        self.ml_dependencies = {
            "torch", "tensorflow", "keras", "jax", "mxnet", "paddlepaddle",
            "transformers", "huggingface_hub", "datasets", "accelerate",
            "diffusers", "xformers", "bitsandbytes", "optimum", "peft",
            "trl", "sentencepiece", "tokenizers", "protobuf", "onnx",
            "onnxruntime", "triton", "flash_attn", "deepspeed", "fairscale"
        }
        
        self.gpu_dependencies = {
            "torch", "tensorflow", "jax", "cupy", "numba", "cudf",
            "rapids", "xgboost", "lightgbm", "catboost", "optuna"
        }

    def extract_repository_features(self, analysis: Analysis) -> Dict[str, Any]:
        """Extract features from repository characteristics."""
        # Safely access repository and analysis fields, tolerating mocks
        repo = getattr(analysis, 'repository', None)
        deps = getattr(analysis, 'dependencies', []) or []
        issues = getattr(analysis, 'compatibility_issues', []) or []

        def safe_len(obj) -> int:
            try:
                return len(obj)  # type: ignore
            except Exception:
                return 0

        def safe_attr(o, name, default=None):
            return getattr(o, name, default) if o is not None else default

        # Methods may not exist on mocks; handle gracefully
        try:
            critical_issues = analysis.get_critical_issues()
        except Exception:
            critical_issues = []
        try:
            warning_issues = analysis.get_warning_issues()
        except Exception:
            warning_issues = []

        topics = safe_attr(repo, 'topics', []) or []

        return {
            # Repository metadata
            "repo_size": safe_attr(repo, 'size', 0),
            "language": safe_attr(repo, 'language', None),
            "has_dockerfile": bool(safe_attr(repo, 'has_dockerfile', False)),
            "has_conda_env": bool(safe_attr(repo, 'has_conda_env', False)),
            "star_count": safe_attr(repo, 'star_count', safe_attr(repo, 'stars', 0)),
            "fork_count": safe_attr(repo, 'fork_count', 0),
            "topics_count": safe_len(topics),
            
            # Dependency complexity
            "total_dependencies": safe_len(deps),
            "gpu_dependencies": self._count_gpu_dependencies(deps),
            "ml_dependencies": self._count_ml_dependencies(deps),
            "dependency_diversity": self._calculate_dependency_diversity(deps),
            "version_constraints": self._count_version_constraints(deps),
            "pinned_versions": self._count_pinned_versions(deps),
            
            # Compatibility issues
            "critical_issues": safe_len(critical_issues),
            "warning_issues": safe_len(warning_issues),
            "gpu_issues": self._count_gpu_issues(issues),
            "cuda_version_conflicts": self._count_cuda_conflicts(issues),
            "python_version_conflicts": self._count_python_conflicts(issues),
            
            # System requirements
            "python_version_required": self._extract_python_version(safe_attr(analysis, 'python_version_required', '')),
            "cuda_version_required": self._extract_cuda_version(safe_attr(analysis, 'cuda_version_required', '')),
            "min_memory_gb": safe_attr(analysis, 'min_memory_gb', 0.0),
            "min_gpu_memory_gb": safe_attr(analysis, 'min_gpu_memory_gb', 0.0),
            
            # Repository characteristics
            "is_ml_repo": self._is_ml_repository(deps),
            "is_research_repo": self._is_research_repository(repo) if repo is not None else False,
            "has_tests": bool(safe_attr(repo, 'has_tests', False)),
            "has_ci": bool(safe_attr(repo, 'has_ci', False)),
        }

    def extract_system_features(self, profile: SystemProfile) -> Dict[str, Any]:
        """Extract features from system profile."""
        return {
            "cpu_cores": profile.hardware.cpu_cores,
            "memory_gb": profile.hardware.memory_gb,
            "gpu_count": len(profile.hardware.gpus),
            "gpu_memory_total": sum(gpu.memory_gb for gpu in profile.hardware.gpus),
            "gpu_memory_max": max((gpu.memory_gb for gpu in profile.hardware.gpus), default=0),
            "cuda_version": self._extract_cuda_version(profile.software.cuda_version),
            "python_version": self._extract_python_version(profile.software.python_version),
            "container_runtime": profile.container_runtime,
            "compute_score": profile.compute_score,
            "has_nvidia_gpu": any(gpu.vendor == "nvidia" for gpu in profile.hardware.gpus),
            "has_amd_gpu": any(gpu.vendor == "amd" for gpu in profile.hardware.gpus),
        }

    def extract_resolution_features(self, resolution: Resolution) -> Dict[str, Any]:
        """Extract features from resolution strategy."""
        return {
            "strategy_type": resolution.strategy.type.value,
            "files_generated": len(resolution.generated_files),
            "setup_commands": len(resolution.setup_commands),
            "estimated_size_mb": resolution.estimated_size_mb,
            "estimated_setup_time": resolution.strategy.requirements.get("estimated_setup_time", 0),
            "requires_gpu": resolution.strategy.requirements.get("requires_gpu", False),
            "requires_cuda": resolution.strategy.requirements.get("requires_cuda", False),
            "docker_base_image": self._extract_docker_base_image(resolution),
            "has_gpu_support": self._has_gpu_support(resolution),
        }

    def extract_learning_features(self, analysis: Analysis, resolution: Resolution, 
                                outcome: Any) -> Dict[str, Any]:
        """Extract features for learning and pattern recognition."""
        return {
            "success": getattr(outcome, 'status', {}).get('value') == 'success' if hasattr(outcome, 'status') else False,
            "duration": getattr(outcome, 'duration', 0) if hasattr(outcome, 'duration') else 0,
            "error_type": self._categorize_error(getattr(outcome, 'error_message', '')),
            "confidence_score": analysis.confidence_score,
            "similarity_to_known": self._calculate_similarity_to_known(analysis),
            "complexity_score": self._calculate_complexity_score(analysis),
        }

    def _count_gpu_dependencies(self, dependencies: List[DependencyInfo]) -> int:
        """Count GPU-required dependencies."""
        return sum(1 for dep in dependencies if dep.gpu_required)

    def _count_ml_dependencies(self, dependencies: List[DependencyInfo]) -> int:
        """Count ML framework dependencies."""
        return sum(1 for dep in dependencies if dep.name.lower() in self.ml_dependencies)

    def _calculate_dependency_diversity(self, dependencies: List[DependencyInfo]) -> float:
        """Calculate diversity of dependency types."""
        if not dependencies:
            return 0.0
        
        categories = []
        for dep in dependencies:
            if dep.name.lower() in self.ml_dependencies:
                categories.append("ml")
            elif dep.gpu_required:
                categories.append("gpu")
            elif "test" in dep.name.lower() or "pytest" in dep.name.lower():
                categories.append("test")
            elif "dev" in dep.name.lower() or "dev" in dep.name.lower():
                categories.append("dev")
            else:
                categories.append("other")
        
        unique_categories = len(set(categories))
        total_deps = len(dependencies)
        return unique_categories / total_deps if total_deps > 0 else 0.0

    def _count_version_constraints(self, dependencies: List[DependencyInfo]) -> int:
        """Count dependencies with version constraints."""
        return sum(1 for dep in dependencies if dep.version and dep.version != "*")

    def _count_pinned_versions(self, dependencies: List[DependencyInfo]) -> int:
        """Count dependencies with pinned versions (exact version)."""
        pinned_count = 0
        for dep in dependencies:
            if dep.version and not any(op in dep.version for op in [">", "<", "~", "!=", ">=", "<="]):
                pinned_count += 1
        return pinned_count

    def _count_gpu_issues(self, compatibility_issues: List[Any]) -> int:
        """Count GPU-related compatibility issues."""
        gpu_keywords = ["gpu", "cuda", "nvidia", "amd", "rocm", "opencl"]
        count = 0
        for issue in compatibility_issues:
            if hasattr(issue, 'description'):
                issue_text = issue.description.lower()
                if any(keyword in issue_text for keyword in gpu_keywords):
                    count += 1
        return count

    def _count_cuda_conflicts(self, compatibility_issues: List[Any]) -> int:
        """Count CUDA version conflicts."""
        cuda_keywords = ["cuda", "nvidia"]
        count = 0
        for issue in compatibility_issues:
            if hasattr(issue, 'description'):
                issue_text = issue.description.lower()
                if any(keyword in issue_text for keyword in cuda_keywords) and "version" in issue_text:
                    count += 1
        return count

    def _count_python_conflicts(self, compatibility_issues: List[Any]) -> int:
        """Count Python version conflicts."""
        python_keywords = ["python", "py"]
        count = 0
        for issue in compatibility_issues:
            if hasattr(issue, 'description'):
                issue_text = issue.description.lower()
                if any(keyword in issue_text for keyword in python_keywords) and "version" in issue_text:
                    count += 1
        return count

    def _extract_python_version(self, version: str) -> float:
        """Extract Python version as float for ML features."""
        if not version:
            return 0.0
        
        # Extract major.minor version
        match = re.search(r'(\d+)\.(\d+)', version)
        if match:
            major, minor = match.groups()
            return float(f"{major}.{minor}")
        return 0.0

    def _extract_cuda_version(self, version: str) -> float:
        """Extract CUDA version as float for ML features."""
        if not version:
            return 0.0
        
        # Extract major.minor version
        match = re.search(r'(\d+)\.(\d+)', version)
        if match:
            major, minor = match.groups()
            return float(f"{major}.{minor}")
        return 0.0

    def _is_ml_repository(self, dependencies: List[DependencyInfo]) -> bool:
        """Check if repository is ML-focused."""
        ml_dep_count = self._count_ml_dependencies(dependencies)
        return ml_dep_count >= 2  # At least 2 ML dependencies

    def _is_research_repository(self, repository: Any) -> bool:
        """Check if repository appears to be research-focused."""
        research_keywords = ["research", "paper", "experiment", "thesis", "study"]
        repo_text = f"{repository.name} {repository.description or ''}".lower()
        return any(keyword in repo_text for keyword in research_keywords)

    def _extract_docker_base_image(self, resolution: Resolution) -> str:
        """Extract Docker base image type."""
        for file in resolution.generated_files:
            if file.name == "Dockerfile" and hasattr(file, 'content'):
                content = file.content
                if "nvidia/cuda" in content:
                    return "nvidia_cuda"
                elif "python" in content:
                    return "python"
                elif "ubuntu" in content or "debian" in content:
                    return "ubuntu_debian"
        return "unknown"

    def _has_gpu_support(self, resolution: Resolution) -> bool:
        """Check if resolution has GPU support."""
        for file in resolution.generated_files:
            if file.name == "docker-compose.yml" and hasattr(file, 'content'):
                content = file.content
                if "gpus" in content or "nvidia" in content:
                    return True
        return False

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message for learning."""
        if not error_message:
            return "no_error"
        
        error_lower = error_message.lower()
        
        if "cuda" in error_lower or "gpu" in error_lower:
            return "gpu_error"
        elif "permission" in error_lower or "denied" in error_lower:
            return "permission_error"
        elif "network" in error_lower or "connection" in error_lower:
            return "network_error"
        elif "memory" in error_lower or "oom" in error_lower:
            return "memory_error"
        elif "dependency" in error_lower or "import" in error_lower:
            return "dependency_error"
        elif "build" in error_lower or "compile" in error_lower:
            return "build_error"
        else:
            return "unknown_error"

    def _calculate_similarity_to_known(self, analysis: Analysis) -> float:
        """Calculate similarity to known successful patterns."""
        # This would be enhanced with actual pattern matching
        # For now, return a simple heuristic
        ml_dep_count = self._count_ml_dependencies(analysis.dependencies)
        gpu_dep_count = self._count_gpu_dependencies(analysis.dependencies)
        
        # Higher similarity for common ML patterns
        if ml_dep_count >= 3 and gpu_dep_count >= 1:
            return 0.8
        elif ml_dep_count >= 2:
            return 0.6
        elif ml_dep_count >= 1:
            return 0.4
        else:
            return 0.2

    def _calculate_complexity_score(self, analysis: Analysis) -> float:
        """Calculate repository complexity score."""
        score = 0.0
        
        # Dependency complexity
        score += min(len(analysis.dependencies) / 50.0, 1.0) * 0.3
        
        # Issue complexity
        score += min(len(analysis.compatibility_issues) / 20.0, 1.0) * 0.3
        
        # GPU complexity
        if analysis.is_gpu_required():
            score += 0.2
        
        # Version constraint complexity
        constrained_deps = self._count_version_constraints(analysis.dependencies)
        score += min(constrained_deps / 20.0, 1.0) * 0.2
        
        return min(score, 1.0)
