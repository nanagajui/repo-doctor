"""Configuration management for repo-doctor."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class DefaultConfig(BaseModel):
    """Default configuration settings."""
    strategy: str = "auto"
    validation: bool = True
    gpu_mode: str = "flexible"


class KnowledgeBaseConfig(BaseModel):
    """Knowledge base configuration."""
    location: str = "~/.repo-doctor/kb/"
    sync: bool = False


class AdvancedConfig(BaseModel):
    """Advanced configuration settings."""
    parallel_analysis: bool = True
    cache_ttl: int = 604800  # 7 days
    container_timeout: int = 300  # 5 minutes


class IntegrationsConfig(BaseModel):
    """External integrations configuration."""
    openai_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    github_token: Optional[str] = None
    use_llm_fallback: bool = False


class Config(BaseModel):
    """Complete configuration for repo-doctor."""
    defaults: DefaultConfig = Field(default_factory=DefaultConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from file or create default."""
        if config_path is None:
            config_path = Path.home() / ".repo-doctor" / "config.yaml"
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                
                # Resolve environment variables
                config_data = cls._resolve_env_vars(config_data)
                return cls(**config_data)
            
            except (yaml.YAMLError, ValueError) as e:
                print(f"Warning: Error loading config from {config_path}: {e}")
                print("Using default configuration.")
        
        # Create default config and save it
        config = cls()
        config.save(config_path)
        return config
    
    def save(self, config_path: Optional[Path] = None):
        """Save configuration to file."""
        if config_path is None:
            config_path = Path.home() / ".repo-doctor" / "config.yaml"
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)
    
    @staticmethod
    def _resolve_env_vars(data: Any) -> Any:
        """Recursively resolve environment variables in configuration."""
        if isinstance(data, dict):
            return {k: Config._resolve_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Config._resolve_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.getenv(env_var)
        else:
            return data
    
    def get_knowledge_base_path(self) -> Path:
        """Get resolved knowledge base path."""
        path_str = self.knowledge_base.location
        if path_str.startswith("~/"):
            return Path.home() / path_str[2:]
        return Path(path_str)
