"""ML models for strategy success prediction."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from .ml_data_storage import MLDataStorage


class StrategySuccessPredictor:
    """ML model to predict strategy success probability."""

    def __init__(self, model_path: Optional[Path] = None, storage: Optional[MLDataStorage] = None):
        """Initialize strategy success predictor."""
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_importance = {}
        self.model_path = model_path or Path("models/strategy_predictor.pkl")
        self.storage = storage
        self.feature_columns = []
        self.model_metadata = {}
        
        # Load existing model if available
        self.load_model()

    def train(self, training_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Train the strategy success prediction model."""
        if training_data is None and self.storage:
            training_data = self.storage.get_training_data()
        
        if not training_data:
            return {"error": "No training data available"}
        
        try:
            # Prepare features and targets
            X, y, feature_names = self._prepare_training_data(training_data)
            
            if len(X) < 10:  # Need minimum data for training
                return {"error": "Insufficient training data", "samples": len(X)}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
            
            # Extract feature importance
            self._extract_feature_importance(feature_names)
            
            # Generate detailed evaluation
            y_pred = self.model.predict(X_test_scaled)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)
            confusion_mat = confusion_matrix(y_test, y_pred)
            
            # Store metadata
            self.model_metadata = {
                "training_samples": len(X),
                "feature_count": len(feature_names),
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "classification_report": classification_rep,
                "confusion_matrix": confusion_mat.tolist()
            }
            
            # Save model
            self._save_model()
            
            return {
                "success": True,
                "training_samples": len(X),
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "feature_importance": self.feature_importance
            }
            
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}

    def predict_success_probability(self, repo_features: Dict[str, Any], 
                                  system_features: Dict[str, Any], 
                                  strategy_type: str) -> float:
        """Predict success probability for a specific strategy."""
        if not self.model or not self.scaler:
            return 0.5  # Default probability if no model
        
        try:
            # Prepare features
            features = self._combine_features(repo_features, system_features, strategy_type)
            
            # Ensure we have the right number of features
            if len(features) != len(self.feature_columns):
                # Pad or truncate features to match training data
                features = self._align_features(features)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict probability
            proba = self.model.predict_proba(features_scaled)[0]
            
            # Return probability of success (class 1)
            return float(proba[1]) if len(proba) > 1 else 0.5
            
        except Exception as e:
            print(f"Error predicting success probability: {e}")
            return 0.5

    def get_strategy_recommendations(self, repo_features: Dict[str, Any], 
                                   system_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get ranked strategy recommendations with success probabilities."""
        strategies = ["docker", "conda", "venv"]
        recommendations = []
        
        for strategy in strategies:
            prob = self.predict_success_probability(repo_features, system_features, strategy)
            confidence = self._calculate_confidence(prob)
            
            recommendations.append({
                "strategy": strategy,
                "success_probability": prob,
                "confidence": confidence,
                "reasoning": self._generate_reasoning(strategy, prob, repo_features, system_features)
            })
        
        # Sort by success probability
        recommendations.sort(key=lambda x: x["success_probability"], reverse=True)
        
        return recommendations

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if not self.model:
            return {"trained": False}
        
        return {
            "trained": True,
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_columns),
            "metadata": self.model_metadata,
            "feature_importance": self.feature_importance
        }

    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for ML model."""
        features_list = []
        targets = []
        feature_names = []
        
        for record in training_data:
            try:
                # Extract features
                repo_features = record.get("repository_features", {})
                system_features = record.get("system_features", {})
                resolution_features = record.get("resolution_features", {})
                
                # Get strategy type
                strategy_type = resolution_features.get("strategy_type", "unknown")
                
                # Combine features
                combined_features = self._combine_features(repo_features, system_features, strategy_type)
                
                # Get target (success/failure)
                success = record.get("outcome", {}).get("success", False)
                target = 1 if success else 0
                
                features_list.append(combined_features)
                targets.append(target)
                
            except Exception as e:
                print(f"Error processing training record: {e}")
                continue
        
        if not features_list:
            return np.array([]), np.array([]), []
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(targets)
        
        # Store feature names for later use
        if features_list:
            self.feature_columns = list(features_list[0].keys())
            feature_names = self.feature_columns
        
        return X, y, feature_names

    def _combine_features(self, repo_features: Dict[str, Any], 
                         system_features: Dict[str, Any], 
                         strategy_type: str) -> Dict[str, Any]:
        """Combine repository, system, and strategy features."""
        combined = {}
        
        # Add repository features
        for key, value in repo_features.items():
            combined[f"repo_{key}"] = self._normalize_value(value)
        
        # Add system features
        for key, value in system_features.items():
            combined[f"sys_{key}"] = self._normalize_value(value)
        
        # Add strategy type as one-hot encoding
        strategies = ["docker", "conda", "venv"]
        for strategy in strategies:
            combined[f"strategy_{strategy}"] = 1.0 if strategy == strategy_type else 0.0
        
        # Add interaction features
        combined["gpu_repo_sys"] = (
            repo_features.get("gpu_dependencies", 0) * 
            system_features.get("gpu_count", 0)
        )
        
        combined["ml_deps_complexity"] = (
            repo_features.get("ml_dependencies", 0) * 
            repo_features.get("total_dependencies", 0)
        )
        
        return combined

    def _normalize_value(self, value: Any) -> float:
        """Normalize a value to float for ML features."""
        if value is None:
            return 0.0
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # Try to convert string to float
            try:
                return float(value)
            except ValueError:
                # Use hash for categorical strings
                return float(hash(value) % 1000) / 1000.0
        else:
            return 0.0

    def _align_features(self, features: Dict[str, Any]) -> List[float]:
        """Align features with training data columns."""
        aligned = []
        
        for col in self.feature_columns:
            value = features.get(col, 0.0)
            aligned.append(self._normalize_value(value))
        
        return aligned

    def _extract_feature_importance(self, feature_names: List[str]):
        """Extract and store feature importance."""
        if not self.model or not hasattr(self.model, 'feature_importances_'):
            return
        
        importances = self.model.feature_importances_
        
        # Create feature importance dictionary
        for i, importance in enumerate(importances):
            if i < len(feature_names):
                self.feature_importance[feature_names[i]] = float(importance)
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

    def _calculate_confidence(self, probability: float) -> str:
        """Calculate confidence level based on probability."""
        if probability >= 0.8:
            return "high"
        elif probability >= 0.6:
            return "medium"
        elif probability >= 0.4:
            return "low"
        else:
            return "very_low"

    def _generate_reasoning(self, strategy: str, probability: float, 
                          repo_features: Dict[str, Any], 
                          system_features: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for recommendation."""
        reasons = []
        
        # GPU-related reasoning
        if repo_features.get("gpu_dependencies", 0) > 0:
            if strategy == "docker" and system_features.get("gpu_count", 0) > 0:
                reasons.append("Docker supports GPU requirements")
            elif strategy == "conda" and system_features.get("gpu_count", 0) > 0:
                reasons.append("Conda can handle GPU dependencies")
            else:
                reasons.append("GPU dependencies may be challenging")
        
        # ML dependencies reasoning
        if repo_features.get("ml_dependencies", 0) > 2:
            if strategy == "docker":
                reasons.append("Docker provides isolated ML environment")
            elif strategy == "conda":
                reasons.append("Conda handles complex ML dependencies well")
        
        # System resources reasoning
        if repo_features.get("total_dependencies", 0) > 20:
            if strategy == "docker":
                reasons.append("Docker containerizes complex dependencies")
            elif strategy == "conda":
                reasons.append("Conda manages many dependencies effectively")
        
        # Default reasoning based on probability
        if not reasons:
            if probability > 0.7:
                reasons.append(f"{strategy.title()} strategy shows high success probability")
            elif probability > 0.4:
                reasons.append(f"{strategy.title()} strategy shows moderate success probability")
            else:
                reasons.append(f"{strategy.title()} strategy shows low success probability")
        
        return "; ".join(reasons)

    def _save_model(self):
        """Save trained model and metadata."""
        if not self.model or not self.scaler:
            return
        
        try:
            # Create model directory
            model_dir = self.model_path.parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "feature_importance": self.feature_importance,
                "metadata": self.model_metadata
            }
            
            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)
            
            print(f"Model saved to {self.model_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Load trained model and metadata."""
        if not self.model_path.exists():
            return
        
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get("model")
            self.scaler = model_data.get("scaler")
            self.feature_columns = model_data.get("feature_columns", [])
            self.feature_importance = model_data.get("feature_importance", {})
            self.model_metadata = model_data.get("metadata", {})
            
            print(f"Model loaded from {self.model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")


class DependencyConflictPredictor:
    """ML model to predict dependency conflicts."""

    def __init__(self, model_path: Optional[Path] = None, storage: Optional[MLDataStorage] = None):
        """Initialize dependency conflict predictor."""
        self.model = None
        self.scaler = None
        self.conflict_patterns = {}
        self.model_path = model_path or Path("models/dependency_conflict_predictor.pkl")
        self.storage = storage
        self.feature_columns = []
        
        # Load existing model if available
        self.load_model()

    def train(self, training_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Train dependency conflict prediction model."""
        if training_data is None and self.storage:
            training_data = self.storage.get_training_data()
        
        if not training_data:
            return {"error": "No training data available"}
        
        try:
            # Prepare features and targets
            X, y, feature_names = self._prepare_dependency_features(training_data)
            
            if len(X) < 10:
                return {"error": "Insufficient training data", "samples": len(X)}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            # Extract conflict patterns
            self._extract_conflict_patterns(training_data)
            
            # Save model
            self._save_model()
            
            return {
                "success": True,
                "training_samples": len(X),
                "train_accuracy": train_score,
                "test_accuracy": test_score
            }
            
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}

    def predict_conflicts(self, dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict potential dependency conflicts."""
        if not self.model or not self.scaler:
            return {"conflict_probability": 0.0, "conflict_types": []}
        
        try:
            # Prepare dependency features
            features = self._prepare_dependency_features_single(dependencies)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict conflicts
            conflict_prob = self.model.predict_proba(features_scaled)[0][1]
            conflict_types = self._identify_conflict_types(dependencies)
            
            return {
                "conflict_probability": float(conflict_prob),
                "conflict_types": conflict_types,
                "recommended_actions": self._get_conflict_resolutions(conflict_types)
            }
            
        except Exception as e:
            print(f"Error predicting conflicts: {e}")
            return {"conflict_probability": 0.0, "conflict_types": []}

    def _prepare_dependency_features(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare dependency features for training."""
        features_list = []
        targets = []
        feature_names = []
        
        for record in training_data:
            try:
                repo_features = record.get("repository_features", {})
                dependencies = record.get("dependencies", [])
                
                # Extract dependency features
                dep_features = self._extract_dependency_features(dependencies)
                
                # Add repository context
                dep_features.update({
                    "total_deps": repo_features.get("total_dependencies", 0),
                    "gpu_deps": repo_features.get("gpu_dependencies", 0),
                    "ml_deps": repo_features.get("ml_dependencies", 0)
                })
                
                # Get target (conflict/no conflict)
                has_conflicts = any(
                    issue.get("type") == "dependency_conflict" 
                    for issue in record.get("compatibility_issues", [])
                )
                target = 1 if has_conflicts else 0
                
                features_list.append(dep_features)
                targets.append(target)
                
            except Exception as e:
                print(f"Error processing dependency features: {e}")
                continue
        
        if not features_list:
            return np.array([]), np.array([]), []
        
        # Convert to numpy arrays
        X = np.array([list(f.values()) for f in features_list])
        y = np.array(targets)
        
        # Store feature names
        if features_list:
            self.feature_columns = list(features_list[0].keys())
            feature_names = self.feature_columns
        
        return X, y, feature_names

    def _extract_dependency_features(self, dependencies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract features from dependency list."""
        if not dependencies:
            return {}
        
        # Count different types of dependencies
        ml_deps = 0
        gpu_deps = 0
        version_constraints = 0
        pinned_versions = 0
        
        for dep in dependencies:
            if dep.get("name", "").lower() in ["torch", "tensorflow", "jax", "keras"]:
                ml_deps += 1
            if dep.get("gpu_required", False):
                gpu_deps += 1
            if dep.get("version") and dep["version"] != "*":
                version_constraints += 1
            if dep.get("version") and not any(op in dep["version"] for op in [">", "<", "~", "!=", ">=", "<="]):
                pinned_versions += 1
        
        return {
            "dependency_count": len(dependencies),
            "ml_dependencies": ml_deps,
            "gpu_dependencies": gpu_deps,
            "version_constraints": version_constraints,
            "pinned_versions": pinned_versions,
            "constraint_ratio": version_constraints / len(dependencies) if dependencies else 0
        }

    def _prepare_dependency_features_single(self, dependencies: List[Dict[str, Any]]) -> List[float]:
        """Prepare features for a single dependency list."""
        features = self._extract_dependency_features(dependencies)
        
        # Align with training features
        aligned = []
        for col in self.feature_columns:
            value = features.get(col, 0.0)
            aligned.append(self._normalize_value(value))
        
        return aligned

    def _normalize_value(self, value: Any) -> float:
        """Normalize a value to float."""
        if value is None:
            return 0.0
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return 0.0

    def _identify_conflict_types(self, dependencies: List[Dict[str, Any]]) -> List[str]:
        """Identify potential conflict types."""
        conflict_types = []
        
        # Check for ML framework conflicts
        ml_frameworks = set()
        for dep in dependencies:
            name = dep.get("name", "").lower()
            if name in ["torch", "tensorflow", "jax", "mxnet"]:
                ml_frameworks.add(name)
        
        if len(ml_frameworks) > 1:
            conflict_types.append("ml_framework_conflict")
        
        # Check for CUDA version conflicts
        cuda_versions = set()
        for dep in dependencies:
            if "cuda" in dep.get("name", "").lower():
                version = dep.get("version", "")
                if version:
                    cuda_versions.add(version)
        
        if len(cuda_versions) > 1:
            conflict_types.append("cuda_version_conflict")
        
        return conflict_types

    def _get_conflict_resolutions(self, conflict_types: List[str]) -> List[str]:
        """Get recommended actions for conflict types."""
        resolutions = []
        
        for conflict_type in conflict_types:
            if conflict_type == "ml_framework_conflict":
                resolutions.append("Use separate environments for different ML frameworks")
            elif conflict_type == "cuda_version_conflict":
                resolutions.append("Standardize CUDA version across all dependencies")
            else:
                resolutions.append("Review dependency versions for compatibility")
        
        return resolutions

    def _extract_conflict_patterns(self, training_data: List[Dict[str, Any]]):
        """Extract common conflict patterns from training data."""
        # This would analyze training data to find common conflict patterns
        # For now, return empty patterns
        self.conflict_patterns = {}

    def _save_model(self):
        """Save trained model."""
        if not self.model or not self.scaler:
            return
        
        try:
            model_dir = self.model_path.parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "conflict_patterns": self.conflict_patterns
            }
            
            with open(self.model_path, "wb") as f:
                pickle.dump(model_data, f)
            
        except Exception as e:
            print(f"Error saving dependency conflict model: {e}")

    def load_model(self):
        """Load trained model."""
        if not self.model_path.exists():
            return
        
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get("model")
            self.scaler = model_data.get("scaler")
            self.feature_columns = model_data.get("feature_columns", [])
            self.conflict_patterns = model_data.get("conflict_patterns", {})
            
        except Exception as e:
            print(f"Error loading dependency conflict model: {e}")
