"""Pattern discovery engine for learning system."""

import time
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
from dataclasses import dataclass

from .ml_data_storage import MLDataStorage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ml_knowledge_base import MLKnowledgeBase


@dataclass
class DiscoveredPattern:
    """Represents a discovered pattern."""
    pattern_type: str
    pattern_id: str
    description: str
    confidence: float
    support: int
    examples: List[Dict[str, Any]]
    conditions: Dict[str, Any]
    recommendations: List[str]


class PatternDiscoveryEngine:
    """Discover patterns and insights from analysis data."""

    def __init__(self, knowledge_base: "MLKnowledgeBase"):
        """Initialize pattern discovery engine."""
        self.kb = knowledge_base
        self.ml_storage = knowledge_base.ml_storage
        self.pattern_miner = PatternMiner()
        self.insight_generator = InsightGenerator()
        self.discovered_patterns = {}

    def discover_patterns(self, min_support: float = 0.1, min_confidence: float = 0.7) -> List[DiscoveredPattern]:
        """Discover frequent patterns in successful resolutions."""
        try:
            # Get successful resolution data
            successful_data = self._get_successful_resolutions()
            
            if len(successful_data) < 5:
                return []  # Need minimum data for pattern discovery
            
            patterns = []
            
            # Discover different types of patterns
            patterns.extend(self._discover_strategy_patterns(successful_data, min_support, min_confidence))
            patterns.extend(self._discover_dependency_patterns(successful_data, min_support, min_confidence))
            patterns.extend(self._discover_system_patterns(successful_data, min_support, min_confidence))
            patterns.extend(self._discover_error_patterns(min_support, min_confidence))
            
            # Store discovered patterns
            self._store_patterns(patterns)
            
            return patterns
            
        except Exception as e:
            print(f"Error discovering patterns: {e}")
            return []

    def generate_insights(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights for current analysis based on learned patterns."""
        try:
            insights = []
            
            # Find similar successful cases
            similar_cases = self._find_similar_successful_cases(analysis)
            
            # Generate insights from patterns
            for case in similar_cases[:3]:  # Top 3 similar cases
                insight = self.insight_generator.generate_insight(analysis, case)
                if insight:
                    insights.append(insight)
            
            # Generate insights from discovered patterns
            pattern_insights = self._generate_pattern_insights(analysis)
            insights.extend(pattern_insights)
            
            return insights
            
        except Exception as e:
            print(f"Error generating insights: {e}")
            return []

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of discovered patterns."""
        return {
            "total_patterns": len(self.discovered_patterns),
            "pattern_types": self._get_pattern_type_counts(),
            "most_common_patterns": self._get_most_common_patterns(),
            "pattern_confidence_stats": self._get_pattern_confidence_stats()
        }

    def _get_successful_resolutions(self) -> List[Dict[str, Any]]:
        """Get successful resolution data from storage."""
        training_data = self.ml_storage.get_training_data()
        return [record for record in training_data 
                if record.get("outcome", {}).get("success", False)]

    def _discover_strategy_patterns(self, successful_data: List[Dict[str, Any]], 
                                  min_support: float, min_confidence: float) -> List[DiscoveredPattern]:
        """Discover patterns related to strategy success."""
        patterns = []
        
        # Group by strategy type
        strategy_groups = defaultdict(list)
        for record in successful_data:
            strategy = record.get("resolution_features", {}).get("strategy_type", "unknown")
            strategy_groups[strategy].append(record)
        
        # Analyze each strategy
        for strategy, records in strategy_groups.items():
            if len(records) < 2:  # Need at least 2 examples
                continue
            
            # Find common characteristics
            common_features = self._find_common_features(records)
            
            if common_features:
                pattern = DiscoveredPattern(
                    pattern_type="strategy_success",
                    pattern_id=f"strategy_{strategy}_success",
                    description=f"Successful {strategy} strategy patterns",
                    confidence=self._calculate_pattern_confidence(records, common_features),
                    support=len(records),
                    examples=records[:5],  # Top 5 examples
                    conditions=common_features,
                    recommendations=[f"Consider {strategy} strategy for similar cases"]
                )
                patterns.append(pattern)
        
        return patterns

    def _discover_dependency_patterns(self, successful_data: List[Dict[str, Any]], 
                                    min_support: float, min_confidence: float) -> List[DiscoveredPattern]:
        """Discover patterns related to dependency management."""
        patterns = []
        
        # Group by dependency characteristics
        dependency_groups = defaultdict(list)
        
        for record in successful_data:
            repo_features = record.get("repository_features", {})
            ml_deps = repo_features.get("ml_dependencies", 0)
            gpu_deps = repo_features.get("gpu_dependencies", 0)
            total_deps = repo_features.get("total_dependencies", 0)
            
            # Categorize by dependency complexity
            if ml_deps >= 3 and gpu_deps >= 1:
                dependency_groups["complex_ml_gpu"].append(record)
            elif ml_deps >= 2:
                dependency_groups["ml_heavy"].append(record)
            elif gpu_deps >= 1:
                dependency_groups["gpu_required"].append(record)
            elif total_deps >= 20:
                dependency_groups["many_dependencies"].append(record)
            else:
                dependency_groups["simple"].append(record)
        
        # Analyze each dependency group
        for group_type, records in dependency_groups.items():
            if len(records) < 2:
                continue
            
            # Find common successful strategies
            strategy_counts = Counter()
            for record in records:
                strategy = record.get("resolution_features", {}).get("strategy_type", "unknown")
                strategy_counts[strategy] += 1
            
            # Find most successful strategy for this group
            most_common_strategy = strategy_counts.most_common(1)[0]
            strategy, count = most_common_strategy
            
            if count >= len(records) * min_confidence:
                pattern = DiscoveredPattern(
                    pattern_type="dependency_strategy",
                    pattern_id=f"deps_{group_type}_{strategy}",
                    description=f"{strategy.title()} strategy works well for {group_type.replace('_', ' ')} repositories",
                    confidence=count / len(records),
                    support=len(records),
                    examples=records[:3],
                    conditions={"dependency_type": group_type, "strategy": strategy},
                    recommendations=[f"Use {strategy} strategy for {group_type.replace('_', ' ')} repositories"]
                )
                patterns.append(pattern)
        
        return patterns

    def _discover_system_patterns(self, successful_data: List[Dict[str, Any]], 
                                min_support: float, min_confidence: float) -> List[DiscoveredPattern]:
        """Discover patterns related to system requirements."""
        patterns = []
        
        # Group by system characteristics
        system_groups = defaultdict(list)
        
        for record in successful_data:
            system_features = record.get("system_features", {})
            gpu_count = system_features.get("gpu_count", 0)
            memory_gb = system_features.get("memory_gb", 0)
            cpu_cores = system_features.get("cpu_cores", 0)
            
            # Categorize by system capabilities
            if gpu_count >= 1 and memory_gb >= 16:
                system_groups["high_end_gpu"].append(record)
            elif gpu_count >= 1:
                system_groups["gpu_available"].append(record)
            elif memory_gb >= 16 and cpu_cores >= 8:
                system_groups["high_end_cpu"].append(record)
            else:
                system_groups["standard"].append(record)
        
        # Analyze each system group
        for group_type, records in system_groups.items():
            if len(records) < 2:
                continue
            
            # Find common successful strategies
            strategy_counts = Counter()
            for record in records:
                strategy = record.get("resolution_features", {}).get("strategy_type", "unknown")
                strategy_counts[strategy] += 1
            
            most_common_strategy = strategy_counts.most_common(1)[0]
            strategy, count = most_common_strategy
            
            if count >= len(records) * min_confidence:
                pattern = DiscoveredPattern(
                    pattern_type="system_strategy",
                    pattern_id=f"sys_{group_type}_{strategy}",
                    description=f"{strategy.title()} strategy works well on {group_type.replace('_', ' ')} systems",
                    confidence=count / len(records),
                    support=len(records),
                    examples=records[:3],
                    conditions={"system_type": group_type, "strategy": strategy},
                    recommendations=[f"Use {strategy} strategy on {group_type.replace('_', ' ')} systems"]
                )
                patterns.append(pattern)
        
        return patterns

    def _discover_error_patterns(self, min_support: float, min_confidence: float) -> List[DiscoveredPattern]:
        """Discover patterns in failed resolutions."""
        patterns = []
        
        # Get failed resolution data
        training_data = self.ml_storage.get_training_data()
        failed_data = [record for record in training_data 
                      if not record.get("outcome", {}).get("success", False)]
        
        if len(failed_data) < 3:
            return patterns
        
        # Group by error types
        error_groups = defaultdict(list)
        
        for record in failed_data:
            error_type = record.get("outcome", {}).get("error_type", "unknown")
            error_groups[error_type].append(record)
        
        # Analyze each error group
        for error_type, records in error_groups.items():
            if len(records) < 2:
                continue
            
            # Find common characteristics of failed cases
            common_features = self._find_common_features(records)
            
            if common_features:
                pattern = DiscoveredPattern(
                    pattern_type="error_pattern",
                    pattern_id=f"error_{error_type}",
                    description=f"Common characteristics of {error_type} failures",
                    confidence=self._calculate_pattern_confidence(records, common_features),
                    support=len(records),
                    examples=records[:3],
                    conditions=common_features,
                    recommendations=self._generate_error_recommendations(error_type)
                )
                patterns.append(pattern)
        
        return patterns

    def _find_common_features(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common features across a set of records."""
        if not records:
            return {}
        
        common_features = {}
        
        # Analyze repository features
        repo_features = [r.get("repository_features", {}) for r in records]
        if repo_features:
            common_features.update(self._find_common_repo_features(repo_features))
        
        # Analyze system features
        system_features = [r.get("system_features", {}) for r in records]
        if system_features:
            common_features.update(self._find_common_system_features(system_features))
        
        # Analyze resolution features
        resolution_features = [r.get("resolution_features", {}) for r in records]
        if resolution_features:
            common_features.update(self._find_common_resolution_features(resolution_features))
        
        return common_features

    def _find_common_repo_features(self, repo_features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common repository features."""
        common = {}
        
        # Count categorical features
        ml_deps_counts = [rf.get("ml_dependencies", 0) for rf in repo_features_list]
        gpu_deps_counts = [rf.get("gpu_dependencies", 0) for rf in repo_features_list]
        total_deps_counts = [rf.get("total_dependencies", 0) for rf in repo_features_list]
        
        # Find common ranges
        if ml_deps_counts:
            common["ml_dependencies_range"] = f"{min(ml_deps_counts)}-{max(ml_deps_counts)}"
        if gpu_deps_counts:
            common["gpu_dependencies_range"] = f"{min(gpu_deps_counts)}-{max(gpu_deps_counts)}"
        if total_deps_counts:
            common["total_dependencies_range"] = f"{min(total_deps_counts)}-{max(total_deps_counts)}"
        
        # Find common boolean features
        is_ml_repo_count = sum(1 for rf in repo_features_list if rf.get("is_ml_repo", False))
        if is_ml_repo_count >= len(repo_features_list) * 0.8:
            common["is_ml_repo"] = True
        
        has_gpu_required_count = sum(1 for rf in repo_features_list if rf.get("gpu_dependencies", 0) > 0)
        if has_gpu_required_count >= len(repo_features_list) * 0.8:
            common["requires_gpu"] = True
        
        return common

    def _find_common_system_features(self, system_features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common system features."""
        common = {}
        
        # Count system characteristics
        gpu_counts = [sf.get("gpu_count", 0) for sf in system_features_list]
        memory_counts = [sf.get("memory_gb", 0) for sf in system_features_list]
        cpu_counts = [sf.get("cpu_cores", 0) for sf in system_features_list]
        
        # Find common ranges
        if gpu_counts:
            common["gpu_count_range"] = f"{min(gpu_counts)}-{max(gpu_counts)}"
        if memory_counts:
            common["memory_gb_range"] = f"{min(memory_counts)}-{max(memory_counts)}"
        if cpu_counts:
            common["cpu_cores_range"] = f"{min(cpu_counts)}-{max(cpu_counts)}"
        
        # Find common boolean features
        has_nvidia_count = sum(1 for sf in system_features_list if sf.get("has_nvidia_gpu", False))
        if has_nvidia_count >= len(system_features_list) * 0.8:
            common["has_nvidia_gpu"] = True
        
        return common

    def _find_common_resolution_features(self, resolution_features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find common resolution features."""
        common = {}
        
        # Count strategy types
        strategies = [rf.get("strategy_type", "unknown") for rf in resolution_features_list]
        strategy_counts = Counter(strategies)
        
        if strategy_counts:
            most_common_strategy = strategy_counts.most_common(1)[0]
            if most_common_strategy[1] >= len(resolution_features_list) * 0.8:
                common["strategy_type"] = most_common_strategy[0]
        
        # Find common boolean features
        has_gpu_support_count = sum(1 for rf in resolution_features_list if rf.get("has_gpu_support", False))
        if has_gpu_support_count >= len(resolution_features_list) * 0.8:
            common["has_gpu_support"] = True
        
        return common

    def _calculate_pattern_confidence(self, records: List[Dict[str, Any]], 
                                    common_features: Dict[str, Any]) -> float:
        """Calculate confidence in a discovered pattern."""
        if not records or not common_features:
            return 0.0
        
        # Simple confidence calculation based on feature consistency
        total_features = len(common_features)
        if total_features == 0:
            return 0.0
        
        # Count how many records match the common features
        matching_records = 0
        for record in records:
            matches = 0
            for feature, value in common_features.items():
                if self._feature_matches(record, feature, value):
                    matches += 1
            
            if matches >= total_features * 0.8:  # 80% feature match
                matching_records += 1
        
        return matching_records / len(records)

    def _feature_matches(self, record: Dict[str, Any], feature: str, expected_value: Any) -> bool:
        """Check if a record matches a feature condition."""
        # Navigate through nested structure to find feature
        value = self._get_nested_value(record, feature)
        
        if value is None:
            return False
        
        if isinstance(expected_value, bool):
            return value == expected_value
        elif isinstance(expected_value, str) and "range" in feature:
            # Handle range features
            return True  # Simplified - would need more complex range matching
        else:
            return str(value) == str(expected_value)

    def _get_nested_value(self, record: Dict[str, Any], feature_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = feature_path.split(".")
        value = record
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value

    def _generate_error_recommendations(self, error_type: str) -> List[str]:
        """Generate recommendations for error types."""
        recommendations = {
            "gpu_error": [
                "Check CUDA version compatibility",
                "Verify GPU availability and drivers",
                "Consider CPU-only alternatives"
            ],
            "dependency_error": [
                "Update conflicting dependencies",
                "Use virtual environment isolation",
                "Check version constraints"
            ],
            "permission_error": [
                "Check file permissions",
                "Run with appropriate user privileges",
                "Verify directory access rights"
            ],
            "memory_error": [
                "Increase available memory",
                "Reduce batch size or model complexity",
                "Use memory-efficient alternatives"
            ],
            "network_error": [
                "Check internet connection",
                "Configure proxy settings if needed",
                "Retry with different network"
            ]
        }
        
        return recommendations.get(error_type, ["Review error logs for specific issues"])

    def _find_similar_successful_cases(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar successful cases for insight generation."""
        # This would use the ML knowledge base to find similar cases
        # For now, return empty list
        return []

    def _generate_pattern_insights(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights based on discovered patterns."""
        insights = []
        
        # Check if analysis matches any discovered patterns
        for pattern_id, pattern in self.discovered_patterns.items():
            if self._analysis_matches_pattern(analysis, pattern):
                insight = {
                    "type": "pattern_match",
                    "pattern_id": pattern_id,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "recommendations": pattern.recommendations
                }
                insights.append(insight)
        
        return insights

    def _analysis_matches_pattern(self, analysis: Dict[str, Any], pattern: DiscoveredPattern) -> bool:
        """Check if analysis matches a discovered pattern."""
        # Simplified pattern matching
        # In practice, this would be more sophisticated
        return pattern.confidence > 0.7

    def _store_patterns(self, patterns: List[DiscoveredPattern]):
        """Store discovered patterns."""
        for pattern in patterns:
            self.discovered_patterns[pattern.pattern_id] = pattern
        
        # Store in ML storage
        pattern_data = {
            pattern_id: {
                "type": pattern.pattern_type,
                "description": pattern.description,
                "confidence": pattern.confidence,
                "support": pattern.support,
                "conditions": pattern.conditions,
                "recommendations": pattern.recommendations
            }
            for pattern_id, pattern in self.discovered_patterns.items()
        }
        
        self.ml_storage.store_patterns(pattern_data)

    def _get_pattern_type_counts(self) -> Dict[str, int]:
        """Get counts of different pattern types."""
        type_counts = Counter()
        for pattern in self.discovered_patterns.values():
            type_counts[pattern.pattern_type] += 1
        return dict(type_counts)

    def _get_most_common_patterns(self) -> List[Dict[str, Any]]:
        """Get most common patterns by support."""
        sorted_patterns = sorted(
            self.discovered_patterns.values(),
            key=lambda p: p.support,
            reverse=True
        )
        
        return [
            {
                "pattern_id": pattern.pattern_id,
                "description": pattern.description,
                "support": pattern.support,
                "confidence": pattern.confidence
            }
            for pattern in sorted_patterns[:5]
        ]

    def _get_pattern_confidence_stats(self) -> Dict[str, float]:
        """Get confidence statistics for patterns."""
        confidences = [pattern.confidence for pattern in self.discovered_patterns.values()]
        
        if not confidences:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": np.mean(confidences),
            "min": np.min(confidences),
            "max": np.max(confidences)
        }


class PatternMiner:
    """Mine patterns from data using association rule learning."""

    def __init__(self):
        """Initialize pattern miner."""
        pass

    def mine_association_rules(self, data: List[Dict[str, Any]], 
                             min_support: float) -> List[Dict[str, Any]]:
        """Mine association rules from data."""
        # Simplified association rule mining
        # In practice, would use libraries like mlxtend or apyori
        rules = []
        
        # Find frequent itemsets
        frequent_itemsets = self._find_frequent_itemsets(data, min_support)
        
        # Generate association rules
        for itemset in frequent_itemsets:
            if len(itemset) > 1:
                rule = self._generate_rule(itemset, data)
                if rule:
                    rules.append(rule)
        
        return rules

    def _find_frequent_itemsets(self, data: List[Dict[str, Any]], 
                              min_support: float) -> List[set]:
        """Find frequent itemsets in data."""
        # Simplified implementation
        # Count item frequencies
        item_counts = Counter()
        
        for record in data:
            items = self._extract_items(record)
            for item in items:
                item_counts[item] += 1
        
        # Filter by minimum support
        min_count = int(len(data) * min_support)
        frequent_items = {item for item, count in item_counts.items() if count >= min_count}
        
        return [frozenset([item]) for item in frequent_items]

    def _extract_items(self, record: Dict[str, Any]) -> List[str]:
        """Extract items from a record for association rule mining."""
        items = []
        
        # Extract strategy type
        strategy = record.get("resolution_features", {}).get("strategy_type", "unknown")
        items.append(f"strategy:{strategy}")
        
        # Extract repository characteristics
        repo_features = record.get("repository_features", {})
        if repo_features.get("is_ml_repo", False):
            items.append("repo_type:ml")
        if repo_features.get("gpu_dependencies", 0) > 0:
            items.append("repo:gpu_required")
        
        # Extract system characteristics
        system_features = record.get("system_features", {})
        if system_features.get("gpu_count", 0) > 0:
            items.append("system:has_gpu")
        if system_features.get("memory_gb", 0) >= 16:
            items.append("system:high_memory")
        
        return items

    def _generate_rule(self, itemset: frozenset, data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate association rule from itemset."""
        # Simplified rule generation
        if len(itemset) < 2:
            return None
        
        items = list(itemset)
        antecedent = items[:-1]
        consequent = items[-1]
        
        return {
            "antecedent": antecedent,
            "consequent": consequent,
            "support": 0.5,  # Simplified
            "confidence": 0.7  # Simplified
        }


class InsightGenerator:
    """Generate insights from patterns and similar cases."""

    def __init__(self):
        """Initialize insight generator."""
        pass

    def generate_insight(self, analysis: Dict[str, Any], 
                        similar_case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate insight from similar case."""
        try:
            # Extract relevant information
            case_strategy = similar_case.get("strategy", "unknown")
            case_success = similar_case.get("similarity", 0.0)
            
            if case_success < 0.3:  # Low similarity
                return None
            
            # Generate insight
            insight = {
                "type": "similar_case_insight",
                "similarity": case_success,
                "strategy": case_strategy,
                "message": f"Similar repository successfully used {case_strategy} strategy",
                "confidence": "high" if case_success > 0.7 else "medium"
            }
            
            return insight
            
        except Exception as e:
            print(f"Error generating insight: {e}")
            return None
