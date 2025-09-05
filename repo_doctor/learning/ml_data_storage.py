"""ML-optimized data storage for learning system."""

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..models.analysis import Analysis
from ..models.resolution import Resolution, ValidationResult
from ..models.system import SystemProfile


class MLDataStorage:
    """ML-optimized data storage for training and inference."""

    def __init__(self, storage_path: Path):
        """Initialize ML data storage."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create ML-specific directories
        (self.storage_path / "training_data").mkdir(exist_ok=True)
        (self.storage_path / "models").mkdir(exist_ok=True)
        (self.storage_path / "features").mkdir(exist_ok=True)
        (self.storage_path / "patterns").mkdir(exist_ok=True)
        
        # Initialize data structures
        self.training_records = []
        self.feature_cache = {}

    def store_training_record(self, record: Dict[str, Any]) -> bool:
        """Store a training record for ML model training."""
        try:
            # Add timestamp if not present
            if "timestamp" not in record:
                record["timestamp"] = time.time()
            
            # Store in memory for batch processing
            self.training_records.append(record)
            
            # Save to disk periodically (every 10 records)
            if len(self.training_records) >= 10:
                self._flush_training_records()
            
            return True
        except Exception as e:
            print(f"Error storing training record: {e}")
            return False

    def store_feature_vector(self, repo_key: str, features: Dict[str, Any]) -> bool:
        """Store feature vector for a repository."""
        try:
            feature_file = self.storage_path / "features" / f"{repo_key.replace('/', '_')}.json"
            
            feature_data = {
                "repo_key": repo_key,
                "features": features,
                "timestamp": time.time(),
                "feature_count": len(features)
            }
            
            with open(feature_file, "w") as f:
                json.dump(feature_data, f, indent=2)
            
            # Cache in memory
            self.feature_cache[repo_key] = features
            
            return True
        except Exception as e:
            print(f"Error storing feature vector: {e}")
            return False

    def get_training_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get training data for ML model training."""
        # Load from disk if not in memory
        if not self.training_records:
            self._load_training_records()
        
        if limit:
            return self.training_records[-limit:]
        return self.training_records

    def get_feature_vector(self, repo_key: str) -> Optional[Dict[str, Any]]:
        """Get feature vector for a repository."""
        # Check cache first
        if repo_key in self.feature_cache:
            return self.feature_cache[repo_key]
        
        # Load from disk
        feature_file = self.storage_path / "features" / f"{repo_key.replace('/', '_')}.json"
        if feature_file.exists():
            try:
                with open(feature_file) as f:
                    data = json.load(f)
                    features = data.get("features", {})
                    self.feature_cache[repo_key] = features
                    return features
            except (json.JSONDecodeError, KeyError):
                pass
        
        return None

    def store_model(self, model_name: str, model: Any, metadata: Dict[str, Any] = None) -> bool:
        """Store trained ML model."""
        try:
            model_file = self.storage_path / "models" / f"{model_name}.pkl"
            metadata_file = self.storage_path / "models" / f"{model_name}_metadata.json"
            
            # Save model
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            
            # Save metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "model_name": model_name,
                "timestamp": time.time(),
                "model_type": type(model).__name__
            })
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error storing model: {e}")
            return False

    def load_model(self, model_name: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Load trained ML model and metadata."""
        try:
            model_file = self.storage_path / "models" / f"{model_name}.pkl"
            metadata_file = self.storage_path / "models" / f"{model_name}_metadata.json"
            
            if not model_file.exists():
                return None, None
            
            # Load model
            with open(model_file, "rb") as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
            
            return model, metadata
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

    def store_patterns(self, patterns: Dict[str, Any]) -> bool:
        """Store discovered patterns."""
        try:
            patterns_file = self.storage_path / "patterns" / "discovered_patterns.json"
            
            pattern_data = {
                "patterns": patterns,
                "timestamp": time.time(),
                "pattern_count": len(patterns)
            }
            
            with open(patterns_file, "w") as f:
                json.dump(pattern_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error storing patterns: {e}")
            return False

    def get_patterns(self) -> Dict[str, Any]:
        """Get discovered patterns."""
        patterns_file = self.storage_path / "patterns" / "discovered_patterns.json"
        
        if not patterns_file.exists():
            return {}
        
        try:
            with open(patterns_file) as f:
                data = json.load(f)
                return data.get("patterns", {})
        except (json.JSONDecodeError, KeyError):
            return {}

    def export_training_data_csv(self, filename: Optional[str] = None) -> str:
        """Export training data as CSV for analysis."""
        if not filename:
            timestamp = int(time.time())
            filename = f"training_data_{timestamp}.csv"
        
        csv_file = self.storage_path / filename
        
        try:
            # Load all training records
            self._load_training_records()
            
            if not self.training_records:
                return str(csv_file)
            
            # Flatten nested dictionaries for CSV
            flattened_records = []
            for record in self.training_records:
                flattened = self._flatten_dict(record)
                flattened_records.append(flattened)
            
            # Create DataFrame and save
            df = pd.DataFrame(flattened_records)
            df.to_csv(csv_file, index=False)
            
            return str(csv_file)
        except Exception as e:
            print(f"Error exporting training data: {e}")
            return str(csv_file)

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "training_records": len(self.training_records),
            "feature_vectors": len(self.feature_cache),
            "models": 0,
            "patterns": 0,
            "storage_size_mb": 0
        }
        
        try:
            # Count models
            models_dir = self.storage_path / "models"
            if models_dir.exists():
                stats["models"] = len(list(models_dir.glob("*.pkl")))
            
            # Count patterns
            patterns_file = self.storage_path / "patterns" / "discovered_patterns.json"
            if patterns_file.exists():
                with open(patterns_file) as f:
                    data = json.load(f)
                    stats["patterns"] = data.get("pattern_count", 0)
            
            # Calculate storage size
            total_size = sum(
                f.stat().st_size for f in self.storage_path.rglob("*") if f.is_file()
            )
            stats["storage_size_mb"] = round(total_size / (1024 * 1024), 2)
            
        except Exception:
            pass
        
        return stats

    def cleanup_old_data(self, max_age_days: int = 30) -> int:
        """Clean up old training data and feature vectors."""
        cleaned_count = 0
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        
        try:
            # Clean up old feature vectors
            features_dir = self.storage_path / "features"
            if features_dir.exists():
                for feature_file in features_dir.glob("*.json"):
                    if feature_file.stat().st_mtime < cutoff_time:
                        feature_file.unlink()
                        cleaned_count += 1
            
            # Clean up old training data
            training_dir = self.storage_path / "training_data"
            if training_dir.exists():
                for training_file in training_dir.glob("*.json"):
                    if training_file.stat().st_mtime < cutoff_time:
                        training_file.unlink()
                        cleaned_count += 1
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        return cleaned_count

    def _flush_training_records(self):
        """Flush training records to disk."""
        if not self.training_records:
            return
        
        try:
            timestamp = int(time.time())
            training_file = self.storage_path / "training_data" / f"batch_{timestamp}.json"
            
            with open(training_file, "w") as f:
                json.dump(self.training_records, f, indent=2)
            
            # Clear memory
            self.training_records = []
        except Exception as e:
            print(f"Error flushing training records: {e}")

    def _load_training_records(self):
        """Load training records from disk."""
        training_dir = self.storage_path / "training_data"
        if not training_dir.exists():
            return
        
        self.training_records = []
        
        for training_file in training_dir.glob("*.json"):
            try:
                with open(training_file) as f:
                    batch_records = json.load(f)
                    if isinstance(batch_records, list):
                        self.training_records.extend(batch_records)
            except (json.JSONDecodeError, KeyError):
                continue

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
