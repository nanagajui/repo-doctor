"""Configuration management for repo-doctor."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from ..utils.logging_config import get_logger


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
    
    # Performance monitoring thresholds (seconds)
    profile_agent_timeout: float = 2.0
    analysis_agent_timeout: float = 10.0
    resolution_agent_timeout: float = 5.0
    
    # Command execution timeouts (seconds)
    gpu_detection_timeout: int = 10
    version_check_timeout: int = 5
    
    # Strategy defaults
    default_conda_size_mb: int = 1024
    default_docker_size_mb: int = 2048
    default_venv_size_mb: int = 512
    default_python_version: str = "3.10"
    
    # Cache configuration
    cache_enabled: bool = True


class LLMConfig(BaseModel):
    """LLM integration configuration with smart discovery."""

    enabled: bool = True
    base_url: Optional[str] = None  # None means use smart discovery
    api_key: Optional[str] = None
    model: str = "openai/gpt-oss-20b"
    timeout: int = 30
    max_tokens: int = 512
    temperature: float = 0.1
    use_smart_discovery: bool = True  # Enable smart discovery by default


class IntegrationsConfig(BaseModel):
    """External integrations configuration."""

    openai_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    github_token: Optional[str] = None
    use_llm_fallback: bool = False
    llm: LLMConfig = Field(default_factory=LLMConfig)


class Config(BaseModel):
    """Complete configuration for repo-doctor."""

    defaults: DefaultConfig = Field(default_factory=DefaultConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from file or create default."""
        logger = get_logger(__name__)
        
        if config_path is None:
            config_path = Path.home() / ".repo-doctor" / "config.yaml"

        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = yaml.safe_load(f) or {}

                # Handle simplified configuration format
                config_data = cls._normalize_config(config_data)
                
                # Resolve environment variables
                config_data = cls._resolve_env_vars(config_data)
                return cls(**config_data)

            except (yaml.YAMLError, ValueError) as e:
                logger.warning(f"Error loading config from {config_path}: {e}")
                logger.warning("Using default configuration.")

        # Create default config and save it
        config = cls()
        config.save(config_path)
        return config

    def save(self, config_path: Optional[Path] = None):
        """Save configuration to file."""
        if config_path is None:
            config_path = Path.home() / ".repo-doctor" / "config.yaml"

        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)

    @staticmethod
    def _normalize_config(data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize simplified config format to full format."""
        normalized = {}
        
        # Handle top-level simple keys
        if "strategy" in data:
            normalized.setdefault("defaults", {})["strategy"] = data.pop("strategy")
        
        # Handle preset application
        if "preset" in data:
            from ..presets import get_preset
            preset = get_preset(data.pop("preset"))
            normalized.setdefault("defaults", {})["strategy"] = preset["strategy"]
            normalized.setdefault("defaults", {})["gpu_mode"] = preset["gpu_mode"]
            normalized.setdefault("defaults", {})["validation"] = preset["validation"]
        
        # Handle simplified LLM config
        if "llm" in data and isinstance(data["llm"], dict):
            normalized.setdefault("integrations", {})["llm"] = data.pop("llm")
        
        # Handle advanced section
        if "advanced" in data:
            normalized["advanced"] = data.pop("advanced")
        
        # Handle performance section (maps to advanced)
        if "performance" in data:
            perf = data.pop("performance")
            normalized.setdefault("advanced", {}).update({
                "parallel_analysis": perf.get("parallel_agents", True),
                "cache_ttl": perf.get("cache_ttl", 3600),
            })
        
        # Merge any remaining top-level keys into defaults
        for key, value in data.items():
            if key not in ["defaults", "knowledge_base", "integrations"]:
                normalized.setdefault("defaults", {})[key] = value
            else:
                normalized[key] = value
        
        return normalized
    
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
