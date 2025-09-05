"""File system storage implementation for knowledge base."""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.analysis import Analysis
from ..models.resolution import Resolution, ValidationResult


class FileSystemStorage:
    """File system based storage for knowledge base."""

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize directory structure
        self._init_directory_structure()

    def _init_directory_structure(self):
        """Initialize the knowledge base directory structure."""
        directories = ["repos", "compatibility", "patterns", "cache"]

        for directory in directories:
            (self.base_path / directory).mkdir(exist_ok=True)

        # Initialize compatibility matrices
        self._init_compatibility_matrices()

    def _init_compatibility_matrices(self):
        """Initialize compatibility matrix files."""
        compatibility_dir = self.base_path / "compatibility"

        # CUDA compatibility matrix
        cuda_matrix_file = compatibility_dir / "cuda_matrix.json"
        if not cuda_matrix_file.exists():
            cuda_matrix = {
                "pytorch": {
                    "1.13.0": ["11.6", "11.7"],
                    "1.12.0": ["11.3", "11.6"],
                    "2.0.0": ["11.7", "11.8"],
                    "2.1.0": ["11.8", "12.1"],
                },
                "tensorflow": {
                    "2.10.0": ["11.2"],
                    "2.11.0": ["11.2", "11.8"],
                    "2.12.0": ["11.8"],
                    "2.13.0": ["11.8"],
                },
            }
            with open(cuda_matrix_file, "w") as f:
                json.dump(cuda_matrix, f, indent=2)

        # Python compatibility matrix
        python_matrix_file = compatibility_dir / "python_matrix.json"
        if not python_matrix_file.exists():
            python_matrix = {
                "pytorch": {
                    "1.13.0": ["3.7", "3.8", "3.9", "3.10"],
                    "2.0.0": ["3.8", "3.9", "3.10", "3.11"],
                    "2.1.0": ["3.8", "3.9", "3.10", "3.11"],
                },
                "tensorflow": {
                    "2.10.0": ["3.7", "3.8", "3.9", "3.10"],
                    "2.11.0": ["3.7", "3.8", "3.9", "3.10", "3.11"],
                    "2.12.0": ["3.8", "3.9", "3.10", "3.11"],
                },
            }
            with open(python_matrix_file, "w") as f:
                json.dump(python_matrix, f, indent=2)

    def store_analysis(
        self, repo_key: str, analysis: Analysis, commit_hash: str
    ) -> bool:
        """Store analysis results."""
        try:
            repo_dir = self.base_path / "repos" / repo_key
            repo_dir.mkdir(parents=True, exist_ok=True)

            analysis_dir = repo_dir / "analyses"
            analysis_dir.mkdir(exist_ok=True)

            analysis_file = analysis_dir / f"{commit_hash}.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis.model_dump(), f, indent=2)

            return True
        except Exception:
            return False

    def store_solution(
        self, repo_key: str, solution: Resolution, outcome: ValidationResult
    ) -> bool:
        """Store solution and outcome."""
        try:
            repo_dir = self.base_path / "repos" / repo_key
            repo_dir.mkdir(parents=True, exist_ok=True)

            # Determine outcome directory
            outcome_dir = (
                "successful" if outcome.status.value == "success" else "failed"
            )
            solution_dir = repo_dir / "solutions" / outcome_dir
            solution_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            import hashlib

            solution_id = hashlib.md5(
                f"{solution.strategy.type.value}_{repo_key}".encode()
            ).hexdigest()[:8]

            solution_file = solution_dir / f"{solution_id}.json"
            with open(solution_file, "w") as f:
                json.dump(
                    {
                        "solution": solution.model_dump(),
                        "outcome": outcome.model_dump(),
                    },
                    f,
                    indent=2,
                )

            return True
        except Exception:
            return False

    def get_compatibility_matrix(self, matrix_type: str) -> Dict[str, Any]:
        """Get compatibility matrix by type."""
        matrix_file = self.base_path / "compatibility" / f"{matrix_type}_matrix.json"

        if not matrix_file.exists():
            return {}

        try:
            with open(matrix_file) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def update_compatibility_matrix(
        self, matrix_type: str, data: Dict[str, Any]
    ) -> bool:
        """Update compatibility matrix."""
        try:
            matrix_file = (
                self.base_path / "compatibility" / f"{matrix_type}_matrix.json"
            )
            with open(matrix_file, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False

    def get_cached_analysis(
        self, repo_key: str, cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached analysis result."""
        cache_file = (
            self.base_path / "cache" / f"{repo_key.replace('/', '_')}_{cache_key}.json"
        )

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                cached_data = json.load(f)

            # Check if cache is still valid (7 days)
            import time

            if time.time() - cached_data.get("timestamp", 0) > 604800:  # 7 days
                cache_file.unlink()  # Remove expired cache
                return None

            return cached_data.get("data")
        except (json.JSONDecodeError, KeyError):
            return None

    def cache_analysis(
        self, repo_key: str, cache_key: str, data: Dict[str, Any]
    ) -> bool:
        """Cache analysis result."""
        try:
            cache_file = (
                self.base_path
                / "cache"
                / f"{repo_key.replace('/', '_')}_{cache_key}.json"
            )

            import time

            cached_data = {"timestamp": time.time(), "data": data}

            with open(cache_file, "w") as f:
                json.dump(cached_data, f, indent=2)

            return True
        except Exception:
            return False

    def cleanup_cache(self, max_age_days: int = 7) -> int:
        """Clean up expired cache files."""
        cache_dir = self.base_path / "cache"
        if not cache_dir.exists():
            return 0

        import time

        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        cleaned_count = 0

        for cache_file in cache_dir.glob("*.json"):
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    cleaned_count += 1
            except OSError:
                continue

        return cleaned_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_repos": 0,
            "total_analyses": 0,
            "successful_solutions": 0,
            "failed_solutions": 0,
            "cache_files": 0,
            "storage_size_mb": 0,
        }

        try:
            # Count repositories
            repos_dir = self.base_path / "repos"
            if repos_dir.exists():
                stats["total_repos"] = len(list(repos_dir.iterdir()))

                # Count analyses and solutions
                for repo_dir in repos_dir.iterdir():
                    if repo_dir.is_dir():
                        analyses_dir = repo_dir / "analyses"
                        if analyses_dir.exists():
                            stats["total_analyses"] += len(
                                list(analyses_dir.glob("*.json"))
                            )

                        solutions_dir = repo_dir / "solutions"
                        if solutions_dir.exists():
                            successful_dir = solutions_dir / "successful"
                            failed_dir = solutions_dir / "failed"

                            if successful_dir.exists():
                                stats["successful_solutions"] += len(
                                    list(successful_dir.glob("*.json"))
                                )
                            if failed_dir.exists():
                                stats["failed_solutions"] += len(
                                    list(failed_dir.glob("*.json"))
                                )

            # Count cache files
            cache_dir = self.base_path / "cache"
            if cache_dir.exists():
                stats["cache_files"] = len(list(cache_dir.glob("*.json")))

            # Calculate storage size
            total_size = sum(
                f.stat().st_size for f in self.base_path.rglob("*") if f.is_file()
            )
            stats["storage_size_mb"] = round(total_size / (1024 * 1024), 2)

        except Exception:
            pass

        return stats
