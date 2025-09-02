"""Knowledge base for storing and retrieving analysis patterns."""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from ..models.analysis import Analysis
from ..models.resolution import Resolution, ValidationResult


class KnowledgeBase:
    """Knowledge base for learning from repository analyses."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        (self.storage_path / "repos").mkdir(exist_ok=True)
        (self.storage_path / "compatibility").mkdir(exist_ok=True)
        (self.storage_path / "patterns").mkdir(exist_ok=True)
    
    def record_analysis(self, analysis: Analysis, commit_hash: Optional[str] = None) -> str:
        """Record analysis results."""
        repo_key = f"{analysis.repository.owner}/{analysis.repository.name}"
        repo_dir = self.storage_path / "repos" / repo_key
        repo_dir.mkdir(parents=True, exist_ok=True)
        
        # Use commit hash or generate from analysis
        if not commit_hash:
            analysis_str = json.dumps(analysis.model_dump(), sort_keys=True)
            commit_hash = hashlib.md5(analysis_str.encode()).hexdigest()[:12]
        
        # Save analysis
        analysis_file = repo_dir / "analyses" / f"{commit_hash}.json"
        analysis_file.parent.mkdir(exist_ok=True)
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis.model_dump(), f, indent=2)
        
        return commit_hash
    
    def record_outcome(self, analysis: Analysis, solution: Resolution, outcome: ValidationResult):
        """Record solution outcome for learning."""
        repo_key = f"{analysis.repository.owner}/{analysis.repository.name}"
        repo_dir = self.storage_path / "repos" / repo_key
        
        # Determine outcome directory
        outcome_dir = "successful" if outcome.status.value == "success" else "failed"
        solution_dir = repo_dir / "solutions" / outcome_dir
        solution_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate solution ID
        solution_id = hashlib.md5(
            f"{solution.strategy.type.value}_{analysis.repository.url}".encode()
        ).hexdigest()[:8]
        
        # Save solution and outcome
        solution_file = solution_dir / f"{solution_id}.json"
        with open(solution_file, 'w') as f:
            json.dump({
                "solution": solution.model_dump(),
                "outcome": outcome.model_dump(),
                "timestamp": outcome.duration
            }, f, indent=2)
        
        # Update patterns
        self._update_patterns(analysis, solution, outcome)
    
    def get_similar_analyses(self, analysis: Analysis, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar analyses from knowledge base."""
        similar = []
        
        # Simple similarity based on repository topics and dependencies
        target_topics = set(analysis.repository.topics)
        target_deps = {dep.name for dep in analysis.dependencies}
        
        repos_dir = self.storage_path / "repos"
        if not repos_dir.exists():
            return similar
        
        for repo_dir in repos_dir.iterdir():
            if not repo_dir.is_dir():
                continue
                
            analyses_dir = repo_dir / "analyses"
            if not analyses_dir.exists():
                continue
            
            for analysis_file in analyses_dir.glob("*.json"):
                try:
                    with open(analysis_file) as f:
                        stored_analysis = json.load(f)
                    
                    # Calculate similarity score
                    stored_topics = set(stored_analysis.get("repository", {}).get("topics", []))
                    stored_deps = {dep["name"] for dep in stored_analysis.get("dependencies", [])}
                    
                    topic_similarity = len(target_topics & stored_topics) / max(len(target_topics | stored_topics), 1)
                    dep_similarity = len(target_deps & stored_deps) / max(len(target_deps | stored_deps), 1)
                    
                    similarity_score = (topic_similarity + dep_similarity) / 2
                    
                    if similarity_score > 0.1:  # Minimum similarity threshold
                        similar.append({
                            "analysis": stored_analysis,
                            "similarity": similarity_score,
                            "file_path": str(analysis_file)
                        })
                
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Sort by similarity and return top results
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:limit]
    
    def get_success_patterns(self, strategy_type: Optional[str] = None) -> Dict[str, Any]:
        """Get successful resolution patterns."""
        patterns_file = self.storage_path / "patterns" / "proven_fixes.json"
        
        if not patterns_file.exists():
            return {}
        
        try:
            with open(patterns_file) as f:
                patterns = json.load(f)
            
            if strategy_type:
                return patterns.get(strategy_type, {})
            return patterns
        
        except json.JSONDecodeError:
            return {}
    
    def get_failure_patterns(self) -> Dict[str, Any]:
        """Get common failure patterns."""
        patterns_file = self.storage_path / "patterns" / "common_failures.json"
        
        if not patterns_file.exists():
            return {}
        
        try:
            with open(patterns_file) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    
    def _update_patterns(self, analysis: Analysis, solution: Resolution, outcome: ValidationResult):
        """Update learned patterns based on outcome."""
        if outcome.status.value == "success":
            self._update_success_patterns(solution)
        else:
            self._analyze_failure(analysis, solution, outcome)
    
    def _update_success_patterns(self, solution: Resolution):
        """Update successful patterns."""
        patterns_file = self.storage_path / "patterns" / "proven_fixes.json"
        
        # Load existing patterns
        patterns = {}
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    patterns = json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Update pattern for this strategy type
        strategy_key = solution.strategy.type.value
        if strategy_key not in patterns:
            patterns[strategy_key] = {"count": 0, "avg_setup_time": 0}
        
        patterns[strategy_key]["count"] += 1
        # Update average setup time (simple moving average)
        current_avg = patterns[strategy_key]["avg_setup_time"]
        new_time = solution.strategy.requirements.get("estimated_setup_time", 0)
        patterns[strategy_key]["avg_setup_time"] = (current_avg + new_time) / 2
        
        # Save updated patterns
        with open(patterns_file, 'w') as f:
            json.dump(patterns, f, indent=2)
    
    def _analyze_failure(self, analysis: Analysis, solution: Resolution, outcome: ValidationResult):
        """Analyze failure and update failure patterns."""
        failures_file = self.storage_path / "patterns" / "common_failures.json"
        
        # Load existing failures
        failures = {}
        if failures_file.exists():
            try:
                with open(failures_file) as f:
                    failures = json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Extract failure type from error message
        error_msg = outcome.error_message or "unknown_error"
        failure_key = self._categorize_failure(error_msg)
        
        if failure_key not in failures:
            failures[failure_key] = {"count": 0, "examples": []}
        
        failures[failure_key]["count"] += 1
        failures[failure_key]["examples"].append({
            "repo": f"{analysis.repository.owner}/{analysis.repository.name}",
            "strategy": solution.strategy.type.value,
            "error": error_msg[:200]  # Truncate long errors
        })
        
        # Keep only recent examples (last 10)
        failures[failure_key]["examples"] = failures[failure_key]["examples"][-10:]
        
        # Save updated failures
        with open(failures_file, 'w') as f:
            json.dump(failures, f, indent=2)
    
    def _categorize_failure(self, error_msg: str) -> str:
        """Categorize failure based on error message."""
        error_lower = error_msg.lower()
        
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
        else:
            return "unknown_error"
