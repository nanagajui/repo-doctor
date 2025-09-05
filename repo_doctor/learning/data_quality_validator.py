"""Data quality validation for ML training data."""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class DataQualityIssue:
    """Represents a data quality issue."""
    issue_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    field: Optional[str] = None
    suggested_fix: Optional[str] = None


class DataQualityValidator:
    """Validate and clean ML training data."""

    def __init__(self):
        """Initialize data quality validator."""
        self.quality_rules = self._initialize_quality_rules()
        self.cleaning_rules = self._initialize_cleaning_rules()

    def validate_training_record(self, record: Dict[str, Any]) -> List[DataQualityIssue]:
        """Validate a single training record for quality issues."""
        issues = []
        
        # Check required fields
        issues.extend(self._check_required_fields(record))
        
        # Check data types
        issues.extend(self._check_data_types(record))
        
        # Check value ranges
        issues.extend(self._check_value_ranges(record))
        
        # Check consistency
        issues.extend(self._check_consistency(record))
        
        # Check completeness
        issues.extend(self._check_completeness(record))
        
        return issues

    def clean_training_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a training record by applying cleaning rules."""
        cleaned_record = record.copy()
        
        # Apply cleaning rules
        for rule in self.cleaning_rules:
            cleaned_record = rule(cleaned_record)
        
        return cleaned_record

    def validate_feature_vector(self, features: Dict[str, Any]) -> List[DataQualityIssue]:
        """Validate a feature vector for quality issues."""
        issues = []
        
        # Check for missing values
        issues.extend(self._check_missing_values(features))
        
        # Check for outliers
        issues.extend(self._check_outliers(features))
        
        # Check for invalid values
        issues.extend(self._check_invalid_values(features))
        
        return issues

    def clean_feature_vector(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a feature vector."""
        cleaned_features = {}
        
        for key, value in features.items():
            if value is None:
                # Replace None with appropriate default
                cleaned_features[key] = self._get_default_value(key)
            elif isinstance(value, (int, float)):
                # Handle numeric values
                if np.isnan(value) or np.isinf(value):
                    cleaned_features[key] = self._get_default_value(key)
                else:
                    cleaned_features[key] = value
            elif isinstance(value, str):
                # Clean string values
                cleaned_features[key] = self._clean_string_value(value)
            else:
                cleaned_features[key] = value
        
        return cleaned_features

    def get_data_quality_report(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive data quality report."""
        report = {
            "total_records": len(records),
            "quality_issues": [],
            "field_statistics": {},
            "recommendations": []
        }
        
        if not records:
            return report
        
        # Analyze each record
        all_issues = []
        for i, record in enumerate(records):
            issues = self.validate_training_record(record)
            for issue in issues:
                issue.record_index = i
                all_issues.append(issue)
        
        # Categorize issues
        issue_counts = {}
        for issue in all_issues:
            key = f"{issue.issue_type}_{issue.severity}"
            issue_counts[key] = issue_counts.get(key, 0) + 1
        
        report["quality_issues"] = issue_counts
        
        # Analyze field statistics
        report["field_statistics"] = self._analyze_field_statistics(records)
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(all_issues, records)
        
        return report

    def _initialize_quality_rules(self) -> List[callable]:
        """Initialize data quality validation rules."""
        return [
            self._check_required_fields,
            self._check_data_types,
            self._check_value_ranges,
            self._check_consistency,
            self._check_completeness
        ]

    def _initialize_cleaning_rules(self) -> List[callable]:
        """Initialize data cleaning rules."""
        return [
            self._clean_numeric_values,
            self._clean_string_values,
            self._clean_missing_values,
            self._normalize_versions
        ]

    def _check_required_fields(self, record: Dict[str, Any]) -> List[DataQualityIssue]:
        """Check for required fields."""
        issues = []
        required_fields = ["timestamp", "repository_features", "system_features", "resolution_features"]
        
        for field in required_fields:
            if field not in record:
                issues.append(DataQualityIssue(
                    issue_type="missing_required_field",
                    severity="critical",
                    description=f"Required field '{field}' is missing",
                    field=field,
                    suggested_fix=f"Add the missing '{field}' field"
                ))
        
        return issues

    def _check_data_types(self, record: Dict[str, Any]) -> List[DataQualityIssue]:
        """Check data types of fields."""
        issues = []
        
        # Check timestamp
        if "timestamp" in record and not isinstance(record["timestamp"], (int, float)):
            issues.append(DataQualityIssue(
                issue_type="invalid_data_type",
                severity="high",
                description="Timestamp should be numeric",
                field="timestamp",
                suggested_fix="Convert timestamp to numeric value"
            ))
        
        # Check feature dictionaries
        for feature_type in ["repository_features", "system_features", "resolution_features"]:
            if feature_type in record and not isinstance(record[feature_type], dict):
                issues.append(DataQualityIssue(
                    issue_type="invalid_data_type",
                    severity="high",
                    description=f"{feature_type} should be a dictionary",
                    field=feature_type,
                    suggested_fix=f"Convert {feature_type} to dictionary format"
                ))
        
        return issues

    def _check_value_ranges(self, record: Dict[str, Any]) -> List[DataQualityIssue]:
        """Check value ranges for numeric fields."""
        issues = []
        
        # Check repository features
        if "repository_features" in record:
            repo_features = record["repository_features"]
            
            # Check star_count
            if "star_count" in repo_features:
                star_count = repo_features["star_count"]
                if isinstance(star_count, (int, float)) and star_count < 0:
                    issues.append(DataQualityIssue(
                        issue_type="invalid_value_range",
                        severity="medium",
                        description="Star count cannot be negative",
                        field="repository_features.star_count",
                        suggested_fix="Set star_count to 0 or positive value"
                    ))
            
            # Check dependency counts
            for count_field in ["total_dependencies", "gpu_dependencies", "ml_dependencies"]:
                if count_field in repo_features:
                    count = repo_features[count_field]
                    if isinstance(count, (int, float)) and count < 0:
                        issues.append(DataQualityIssue(
                            issue_type="invalid_value_range",
                            severity="medium",
                            description=f"{count_field} cannot be negative",
                            field=f"repository_features.{count_field}",
                            suggested_fix=f"Set {count_field} to 0 or positive value"
                        ))
        
        return issues

    def _check_consistency(self, record: Dict[str, Any]) -> List[DataQualityIssue]:
        """Check data consistency."""
        issues = []
        
        if "repository_features" in record and "resolution_features" in record:
            repo_features = record["repository_features"]
            resolution_features = record["resolution_features"]
            
            # Check GPU consistency
            repo_gpu_required = repo_features.get("gpu_dependencies", 0) > 0
            resolution_gpu_support = resolution_features.get("has_gpu_support", False)
            
            if repo_gpu_required and not resolution_gpu_support:
                issues.append(DataQualityIssue(
                    issue_type="inconsistent_data",
                    severity="medium",
                    description="Repository requires GPU but resolution doesn't support it",
                    field="resolution_features.has_gpu_support",
                    suggested_fix="Verify GPU support configuration"
                ))
        
        return issues

    def _check_completeness(self, record: Dict[str, Any]) -> List[DataQualityIssue]:
        """Check data completeness."""
        issues = []
        
        # Check if feature dictionaries are empty
        for feature_type in ["repository_features", "system_features", "resolution_features"]:
            if feature_type in record:
                features = record[feature_type]
                if isinstance(features, dict) and len(features) == 0:
                    issues.append(DataQualityIssue(
                        issue_type="incomplete_data",
                        severity="high",
                        description=f"{feature_type} is empty",
                        field=feature_type,
                        suggested_fix=f"Populate {feature_type} with relevant data"
                    ))
        
        return issues

    def _check_missing_values(self, features: Dict[str, Any]) -> List[DataQualityIssue]:
        """Check for missing values in feature vector."""
        issues = []
        
        for key, value in features.items():
            if value is None:
                issues.append(DataQualityIssue(
                    issue_type="missing_value",
                    severity="medium",
                    description=f"Feature '{key}' has missing value",
                    field=key,
                    suggested_fix=f"Provide default value for '{key}'"
                ))
        
        return issues

    def _check_outliers(self, features: Dict[str, Any]) -> List[DataQualityIssue]:
        """Check for outliers in numeric features."""
        issues = []
        
        # Define reasonable ranges for common features
        ranges = {
            "star_count": (0, 1000000),
            "total_dependencies": (0, 1000),
            "cpu_cores": (1, 128),
            "memory_gb": (1, 1000),
            "gpu_memory_total": (0, 1000)
        }
        
        for key, value in features.items():
            if key in ranges and isinstance(value, (int, float)):
                min_val, max_val = ranges[key]
                if value < min_val or value > max_val:
                    issues.append(DataQualityIssue(
                        issue_type="outlier_value",
                        severity="low",
                        description=f"Feature '{key}' value {value} is outside expected range [{min_val}, {max_val}]",
                        field=key,
                        suggested_fix=f"Verify if value {value} is correct for '{key}'"
                    ))
        
        return issues

    def _check_invalid_values(self, features: Dict[str, Any]) -> List[DataQualityIssue]:
        """Check for invalid values."""
        issues = []
        
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    issues.append(DataQualityIssue(
                        issue_type="invalid_numeric_value",
                        severity="high",
                        description=f"Feature '{key}' has invalid numeric value: {value}",
                        field=key,
                        suggested_fix=f"Replace invalid value with default for '{key}'"
                    ))
        
        return issues

    def _clean_numeric_values(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean numeric values in record."""
        cleaned = record.copy()
        
        def clean_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            cleaned_dict = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    cleaned_dict[k] = clean_nested_dict(v)
                elif isinstance(v, (int, float)):
                    if np.isnan(v) or np.isinf(v):
                        cleaned_dict[k] = 0
                    else:
                        cleaned_dict[k] = v
                else:
                    cleaned_dict[k] = v
            return cleaned_dict
        
        for feature_type in ["repository_features", "system_features", "resolution_features"]:
            if feature_type in cleaned and isinstance(cleaned[feature_type], dict):
                cleaned[feature_type] = clean_nested_dict(cleaned[feature_type])
        
        return cleaned

    def _clean_string_values(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean string values in record."""
        cleaned = record.copy()
        
        def clean_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            cleaned_dict = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    cleaned_dict[k] = clean_nested_dict(v)
                elif isinstance(v, str):
                    cleaned_dict[k] = self._clean_string_value(v)
                else:
                    cleaned_dict[k] = v
            return cleaned_dict
        
        for feature_type in ["repository_features", "system_features", "resolution_features"]:
            if feature_type in cleaned and isinstance(cleaned[feature_type], dict):
                cleaned[feature_type] = clean_nested_dict(cleaned[feature_type])
        
        return cleaned

    def _clean_missing_values(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean missing values in record."""
        cleaned = record.copy()
        
        # Set default timestamp if missing
        if "timestamp" not in cleaned or cleaned["timestamp"] is None:
            import time
            cleaned["timestamp"] = time.time()
        
        return cleaned

    def _normalize_versions(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize version strings."""
        cleaned = record.copy()
        
        def normalize_version_string(version_str: str) -> str:
            if not version_str or not isinstance(version_str, str):
                return version_str
            
            # Remove common version prefixes
            version_str = re.sub(r'^[vV]', '', version_str)
            # Normalize separators
            version_str = re.sub(r'[._-]+', '.', version_str)
            return version_str
        
        def clean_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            cleaned_dict = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    cleaned_dict[k] = clean_nested_dict(v)
                elif isinstance(v, str) and "version" in k.lower():
                    cleaned_dict[k] = normalize_version_string(v)
                else:
                    cleaned_dict[k] = v
            return cleaned_dict
        
        for feature_type in ["repository_features", "system_features", "resolution_features"]:
            if feature_type in cleaned and isinstance(cleaned[feature_type], dict):
                cleaned[feature_type] = clean_nested_dict(cleaned[feature_type])
        
        return cleaned

    def _clean_string_value(self, value: str) -> str:
        """Clean a single string value."""
        if not value:
            return ""
        
        # Remove extra whitespace
        value = value.strip()
        
        # Remove non-printable characters
        value = ''.join(char for char in value if char.isprintable())
        
        return value

    def _get_default_value(self, field_name: str) -> Any:
        """Get default value for a field."""
        defaults = {
            "star_count": 0,
            "fork_count": 0,
            "total_dependencies": 0,
            "gpu_dependencies": 0,
            "ml_dependencies": 0,
            "cpu_cores": 1,
            "memory_gb": 4,
            "gpu_count": 0,
            "gpu_memory_total": 0,
            "success": False,
            "duration": 0,
            "confidence_score": 0.5,
            "complexity_score": 0.0
        }
        
        return defaults.get(field_name, 0)

    def _analyze_field_statistics(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze field statistics across records."""
        stats = {}
        
        if not records:
            return stats
        
        # Collect all field names
        all_fields = set()
        for record in records:
            all_fields.update(self._get_all_field_names(record))
        
        # Analyze each field
        for field in all_fields:
            values = []
            for record in records:
                value = self._get_nested_value(record, field)
                if value is not None:
                    values.append(value)
            
            if values:
                stats[field] = {
                    "count": len(values),
                    "missing": len(records) - len(values),
                    "unique": len(set(str(v) for v in values))
                }
                
                # Add numeric statistics if applicable
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    stats[field].update({
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "mean": np.mean(numeric_values),
                        "std": np.std(numeric_values)
                    })
        
        return stats

    def _generate_recommendations(self, issues: List[DataQualityIssue], records: List[Dict[str, Any]]) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        if not issues:
            return ["Data quality is good - no issues found"]
        
        # Count issues by type
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        # Generate recommendations based on issue types
        if "missing_required_field" in issue_counts:
            recommendations.append(f"Fix {issue_counts['missing_required_field']} missing required fields")
        
        if "invalid_data_type" in issue_counts:
            recommendations.append(f"Fix {issue_counts['invalid_data_type']} data type issues")
        
        if "missing_value" in issue_counts:
            recommendations.append(f"Handle {issue_counts['missing_value']} missing values")
        
        if "outlier_value" in issue_counts:
            recommendations.append(f"Review {issue_counts['outlier_value']} outlier values")
        
        # Add general recommendations
        if len(issues) > len(records) * 0.1:  # More than 10% of records have issues
            recommendations.append("Consider implementing data validation pipeline")
        
        return recommendations

    def _get_all_field_names(self, record: Dict[str, Any], prefix: str = "") -> List[str]:
        """Get all field names from a nested dictionary."""
        fields = []
        for key, value in record.items():
            field_name = f"{prefix}.{key}" if prefix else key
            fields.append(field_name)
            
            if isinstance(value, dict):
                fields.extend(self._get_all_field_names(value, field_name))
        
        return fields

    def _get_nested_value(self, record: Dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = field_path.split(".")
        value = record
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
