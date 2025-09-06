"""ML-enhanced knowledge base for Repo Doctor."""

from typing import Dict, List, Optional, Any, Tuple
import json
import time
from pathlib import Path

from ..knowledge.base import KnowledgeBase
from .ml_data_storage import MLDataStorage
from .feature_extractor import FeatureExtractor
from .data_quality_validator import DataQualityValidator
from .strategy_predictor import StrategySuccessPredictor, DependencyConflictPredictor
from .pattern_discovery import PatternDiscoveryEngine
from ..models.analysis import Analysis
from ..models.resolution import Resolution, ValidationResult
from ..models.system import SystemProfile


class MLKnowledgeBase(KnowledgeBase):
    """Enhanced knowledge base with machine learning capabilities."""
    
    def __init__(self, storage_path: str):
        """Initialize ML-enhanced knowledge base."""
        super().__init__(storage_path)
        
        # Initialize ML components
        self.feature_extractor = FeatureExtractor()
        self.ml_storage = MLDataStorage(Path(storage_path) / "ml_data")
        self.data_validator = DataQualityValidator()
        
        # Learning state
        self.learning_enabled = True
        self.pattern_cache = {}
        self.last_pattern_update = 0
        # Components expected by tests and other modules
        self.pattern_engine = PatternDiscoveryEngine(self)
        from .adaptive_learning import AdaptiveLearningSystem
        self.learning_system = AdaptiveLearningSystem(self)

    def record_ml_analysis(self, analysis: Analysis, resolution: Resolution, 
                          outcome: ValidationResult, system_profile: SystemProfile) -> str:
        """Record analysis with ML-optimized features and learning."""
        # Record in traditional knowledge base
        commit_hash = self.record_analysis(analysis)
        self.record_outcome(analysis, resolution, outcome)
        
        if not self.learning_enabled:
            return commit_hash
        
        try:
            # Extract comprehensive features for ML
            repo_features = self.feature_extractor.extract_repository_features(analysis)
            system_features = self.feature_extractor.extract_system_features(system_profile)
            resolution_features = self.feature_extractor.extract_resolution_features(resolution)
            learning_features = self.feature_extractor.extract_learning_features(analysis, resolution, outcome)
            
            # Create ML training record
            ml_record = {
                "timestamp": time.time(),
                "commit_hash": commit_hash,
                "repository_features": repo_features,
                "system_features": system_features,
                "resolution_features": resolution_features,
                "learning_features": learning_features,
                "outcome": {
                    "success": outcome.status.value == "success",
                    "duration": outcome.duration,
                    "error_type": self._categorize_error(outcome.error_message),
                },
                "metadata": {
                    "repo_key": f"{analysis.repository.owner}/{analysis.repository.name}",
                    "confidence_score": analysis.confidence_score,
                    "analysis_time": analysis.analysis_time,
                }
            }
            
            # Validate data quality
            quality_issues = self.data_validator.validate_training_record(ml_record)
            if quality_issues:
                print(f"Data quality issues found: {len(quality_issues)}")
                # Clean the record
                ml_record = self.data_validator.clean_training_record(ml_record)
            
            # Store in ML-optimized format
            self.ml_storage.store_training_record(ml_record)
            
            # Store feature vectors for similarity matching
            repo_key = f"{analysis.repository.owner}/{analysis.repository.name}"
            self.ml_storage.store_feature_vector(repo_key, repo_features)
            
            # Update learning patterns
            self._update_learning_patterns(analysis, resolution, outcome)
            
        except Exception as e:
            print(f"Error recording ML analysis: {e}")
            # Continue with traditional recording even if ML fails
        
        return commit_hash

    def get_ml_recommendations(self, analysis: Analysis, system_profile: SystemProfile) -> Dict[str, Any]:
        """Get ML-enhanced recommendations for analysis."""
        if not self.learning_enabled:
            return {"recommendations": [], "confidence": 0.0, "learning_disabled": True}
        
        try:
            # Extract features for current analysis
            repo_features = self.feature_extractor.extract_repository_features(analysis)
            system_features = self.feature_extractor.extract_system_features(system_profile)
            
            # Find similar successful cases
            similar_cases = self._find_similar_successful_cases(repo_features, system_features)
            
            # Generate recommendations based on similar cases
            recommendations = self._generate_ml_recommendations(similar_cases, repo_features, system_features)
            
            return {
                "recommendations": recommendations,
                "confidence": self._calculate_recommendation_confidence(similar_cases),
                "similar_cases_count": len(similar_cases),
                "learning_enabled": True
            }
            
        except Exception as e:
            print(f"Error generating ML recommendations: {e}")
            return {"recommendations": [], "confidence": 0.0, "error": str(e)}

    def get_learning_insights(self, analysis: Analysis) -> List[Dict[str, Any]]:
        """Get learning-based insights for analysis."""
        if not self.learning_enabled:
            return []
        
        try:
            repo_key = f"{analysis.repository.owner}/{analysis.repository.name}"
            
            # Get feature vector for current analysis
            current_features = self.feature_extractor.extract_repository_features(analysis)
            
            # Find similar cases
            similar_cases = self._find_similar_cases(current_features)
            
            # Generate insights
            insights = []
            for case in similar_cases[:3]:  # Top 3 similar cases
                insight = self._generate_insight_from_case(analysis, case)
                if insight:
                    insights.append(insight)
            
            return insights
            
        except Exception as e:
            print(f"Error generating learning insights: {e}")
            return []

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning system performance metrics."""
        if not self.learning_enabled:
            return {"learning_enabled": False}
        
        try:
            # Get storage statistics
            storage_stats = self.ml_storage.get_storage_stats()
            
            # Get pattern statistics
            patterns = self.ml_storage.get_patterns()
            
            # Calculate learning metrics
            total_records = storage_stats.get("training_records", 0)
            successful_patterns = len(patterns.get("successful", {}))
            failed_patterns = len(patterns.get("failed", {}))
            
            success_rate = 0.0
            if total_records > 0:
                # This would be calculated from actual success/failure data
                success_rate = 0.7  # Placeholder - would be calculated from training data
            
            return {
                "learning_enabled": True,
                "total_training_records": total_records,
                "successful_patterns": successful_patterns,
                "failed_patterns": failed_patterns,
                "success_rate": success_rate,
                "storage_size_mb": storage_stats.get("storage_size_mb", 0),
                "feature_vectors": storage_stats.get("feature_vectors", 0),
                "last_update": self.last_pattern_update
            }
            
        except Exception as e:
            print(f"Error getting learning metrics: {e}")
            return {"learning_enabled": False, "error": str(e)}

    def export_learning_data(self, format: str = "csv") -> str:
        """Export learning data for analysis."""
        if not self.learning_enabled:
            return ""
        
        try:
            if format.lower() == "csv":
                return self.ml_storage.export_training_data_csv()
            else:
                print(f"Unsupported export format: {format}")
                return ""
                
        except Exception as e:
            print(f"Error exporting learning data: {e}")
            return ""

    def cleanup_learning_data(self, max_age_days: int = 30) -> int:
        """Clean up old learning data."""
        if not self.learning_enabled:
            return 0
        
        try:
            return self.ml_storage.cleanup_old_data(max_age_days)
        except Exception as e:
            print(f"Error cleaning up learning data: {e}")
            return 0

    def _update_learning_patterns(self, analysis: Analysis, resolution: Resolution, outcome: ValidationResult):
        """Update learning patterns based on new analysis."""
        try:
            # Get current patterns
            patterns = self.ml_storage.get_patterns()
            
            # Update patterns based on outcome
            if outcome.status.value == "success":
                self._update_success_patterns(patterns, analysis, resolution)
            else:
                self._update_failure_patterns(patterns, analysis, resolution, outcome)
            
            # Store updated patterns
            self.ml_storage.store_patterns(patterns)
            self.last_pattern_update = time.time()
            
        except Exception as e:
            print(f"Error updating learning patterns: {e}")

    def _update_success_patterns(self, patterns: Dict[str, Any], analysis: Analysis, resolution: Resolution):
        """Update successful resolution patterns."""
        if "successful" not in patterns:
            patterns["successful"] = {}
        
        strategy_type = resolution.strategy.type.value
        if strategy_type not in patterns["successful"]:
            patterns["successful"][strategy_type] = {
                "count": 0,
                "avg_setup_time": 0,
                "common_dependencies": {},
                "gpu_success_rate": 0.0
            }
        
        # Update counts
        patterns["successful"][strategy_type]["count"] += 1
        
        # Update average setup time
        current_avg = patterns["successful"][strategy_type]["avg_setup_time"]
        new_time = resolution.strategy.requirements.get("estimated_setup_time", 0)
        count = patterns["successful"][strategy_type]["count"]
        patterns["successful"][strategy_type]["avg_setup_time"] = (
            (current_avg * (count - 1) + new_time) / count
        )
        
        # Update common dependencies
        for dep in analysis.dependencies:
            dep_name = dep.name.lower()
            if dep_name not in patterns["successful"][strategy_type]["common_dependencies"]:
                patterns["successful"][strategy_type]["common_dependencies"][dep_name] = 0
            patterns["successful"][strategy_type]["common_dependencies"][dep_name] += 1
        
        # Update GPU success rate
        if analysis.is_gpu_required():
            gpu_successes = patterns["successful"][strategy_type].get("gpu_successes", 0) + 1
            patterns["successful"][strategy_type]["gpu_successes"] = gpu_successes
            patterns["successful"][strategy_type]["gpu_success_rate"] = (
                gpu_successes / patterns["successful"][strategy_type]["count"]
            )

    def _update_failure_patterns(self, patterns: Dict[str, Any], analysis: Analysis, 
                                resolution: Resolution, outcome: ValidationResult):
        """Update failure patterns."""
        if "failed" not in patterns:
            patterns["failed"] = {}
        
        strategy_type = resolution.strategy.type.value
        error_type = self._categorize_error(outcome.error_message)
        
        if strategy_type not in patterns["failed"]:
            patterns["failed"][strategy_type] = {}
        
        if error_type not in patterns["failed"][strategy_type]:
            patterns["failed"][strategy_type][error_type] = {
                "count": 0,
                "common_causes": [],
                "suggested_fixes": []
            }
        
        patterns["failed"][strategy_type][error_type]["count"] += 1
        
        # Add common causes and suggested fixes
        cause = f"Repo: {analysis.repository.name}, Dependencies: {len(analysis.dependencies)}"
        if cause not in patterns["failed"][strategy_type][error_type]["common_causes"]:
            patterns["failed"][strategy_type][error_type]["common_causes"].append(cause)
        
        fix = self._suggest_fix_for_error(error_type)
        if fix and fix not in patterns["failed"][strategy_type][error_type]["suggested_fixes"]:
            patterns["failed"][strategy_type][error_type]["suggested_fixes"].append(fix)

    def _find_similar_successful_cases(self, repo_features: Dict[str, Any], 
                                     system_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar successful cases from learning data."""
        similar_cases = []
        
        try:
            # Get all training records
            training_data = self.ml_storage.get_training_data()
            
            for record in training_data:
                if not record.get("outcome", {}).get("success", False):
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(repo_features, record.get("repository_features", {}))
                
                if similarity > 0.3:  # Minimum similarity threshold
                    similar_cases.append({
                        "record": record,
                        "similarity": similarity,
                        "strategy": record.get("resolution_features", {}).get("strategy_type", "unknown")
                    })
            
            # Sort by similarity
            similar_cases.sort(key=lambda x: x["similarity"], reverse=True)
            
        except Exception as e:
            print(f"Error finding similar cases: {e}")
        
        return similar_cases[:10]  # Return top 10 similar cases

    def _find_similar_cases(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar cases based on feature similarity."""
        similar_cases = []
        
        try:
            # Get all feature vectors
            training_data = self.ml_storage.get_training_data()
            
            for record in training_data:
                record_features = record.get("repository_features", {})
                similarity = self._calculate_similarity(features, record_features)
                
                if similarity > 0.2:  # Lower threshold for general similarity
                    similar_cases.append({
                        "record": record,
                        "similarity": similarity
                    })
            
            # Sort by similarity
            similar_cases.sort(key=lambda x: x["similarity"], reverse=True)
            
        except Exception as e:
            print(f"Error finding similar cases: {e}")
        
        return similar_cases[:5]  # Return top 5 similar cases

    def _calculate_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between two feature vectors."""
        if not features1 or not features2:
            return 0.0
        
        # Define weights for different feature types
        weights = {
            "ml_dependencies": 0.3,
            "gpu_dependencies": 0.2,
            "total_dependencies": 0.1,
            "is_ml_repo": 0.2,
            "python_version_required": 0.1,
            "cuda_version_required": 0.1
        }
        
        similarity = 0.0
        total_weight = 0.0
        
        for feature, weight in weights.items():
            if feature in features1 and feature in features2:
                val1 = features1[feature]
                val2 = features2[feature]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numeric similarity
                    if val1 == 0 and val2 == 0:
                        feature_sim = 1.0
                    else:
                        feature_sim = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2), 1)
                elif isinstance(val1, bool) and isinstance(val2, bool):
                    # Boolean similarity
                    feature_sim = 1.0 if val1 == val2 else 0.0
                else:
                    # String similarity (simple)
                    feature_sim = 1.0 if str(val1) == str(val2) else 0.0
                
                similarity += feature_sim * weight
                total_weight += weight
        
        return similarity / total_weight if total_weight > 0 else 0.0

    def _generate_ml_recommendations(self, similar_cases: List[Dict[str, Any]], 
                                   repo_features: Dict[str, Any], 
                                   system_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ML-based recommendations."""
        recommendations = []
        
        if not similar_cases:
            return recommendations
        
        # Group by strategy type
        strategy_success = {}
        for case in similar_cases:
            strategy = case["strategy"]
            if strategy not in strategy_success:
                strategy_success[strategy] = {"count": 0, "total_similarity": 0.0}
            
            strategy_success[strategy]["count"] += 1
            strategy_success[strategy]["total_similarity"] += case["similarity"]
        
        # Generate recommendations
        for strategy, stats in strategy_success.items():
            avg_similarity = stats["total_similarity"] / stats["count"]
            success_probability = min(avg_similarity, 0.95)  # Cap at 95%
            
            recommendations.append({
                "strategy": strategy,
                "success_probability": success_probability,
                "confidence": "high" if success_probability > 0.7 else "medium" if success_probability > 0.4 else "low",
                "similar_cases": stats["count"],
                "reasoning": f"Based on {stats['count']} similar successful cases"
            })
        
        # Sort by success probability
        recommendations.sort(key=lambda x: x["success_probability"], reverse=True)
        
        return recommendations

    def _generate_insight_from_case(self, analysis: Analysis, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate insight from a similar case."""
        try:
            record = case["record"]
            similarity = case["similarity"]
            
            # Extract relevant information
            strategy = record.get("resolution_features", {}).get("strategy_type", "unknown")
            success = record.get("outcome", {}).get("success", False)
            
            if not success:
                return None
            
            # Generate insight
            insight = {
                "type": "similar_successful_case",
                "similarity": similarity,
                "strategy": strategy,
                "message": f"Similar repository successfully used {strategy} strategy",
                "confidence": "high" if similarity > 0.7 else "medium"
            }
            
            return insight
            
        except Exception as e:
            print(f"Error generating insight: {e}")
            return None

    def _calculate_recommendation_confidence(self, similar_cases: List[Dict[str, Any]]) -> float:
        """Calculate confidence in recommendations based on similar cases."""
        if not similar_cases:
            return 0.0
        
        # Higher confidence with more similar cases and higher similarity
        avg_similarity = sum(case["similarity"] for case in similar_cases) / len(similar_cases)
        case_count_factor = min(len(similar_cases) / 5.0, 1.0)  # Normalize to max 5 cases
        
        confidence = (avg_similarity * 0.7 + case_count_factor * 0.3)
        return min(confidence, 1.0)

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

    def _suggest_fix_for_error(self, error_type: str) -> str:
        """Suggest fix for error type."""
        fixes = {
            "gpu_error": "Check CUDA version compatibility and GPU availability",
            "permission_error": "Verify file permissions and user access",
            "network_error": "Check internet connection and proxy settings",
            "memory_error": "Increase available memory or reduce resource usage",
            "dependency_error": "Update or reinstall conflicting dependencies",
            "build_error": "Check build tools and compiler versions",
            "unknown_error": "Review error logs for specific issues"
        }
        
        return fixes.get(error_type, "Review error details for specific solution")
