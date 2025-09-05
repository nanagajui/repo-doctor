"""Preset profiles for common use cases - STREAM B Simplification."""

from typing import Dict, Any


# Preset configurations for different use cases
PRESETS: Dict[str, Dict[str, Any]] = {
    "ml-research": {
        "name": "ML Research",
        "description": "Optimized for ML research and experimentation",
        "strategy": "conda",
        "gpu_mode": "flexible",
        "validation": False,
        "cache_enabled": True,
        "llm_enabled": True,
        "learning_enabled": True,
        "notes": "Fast iteration with Conda environments, flexible GPU requirements, ML learning enabled"
    },
    "production": {
        "name": "Production Deployment",
        "description": "Optimized for production deployments",
        "strategy": "docker",
        "gpu_mode": "strict",
        "validation": True,
        "cache_enabled": True,
        "llm_enabled": False,
        "notes": "Docker containers with strict validation and GPU requirements"
    },
    "development": {
        "name": "Local Development",
        "description": "Optimized for local development and testing",
        "strategy": "venv",
        "gpu_mode": "cpu_fallback",
        "validation": False,
        "cache_enabled": True,
        "llm_enabled": True,
        "learning_enabled": True,
        "notes": "Virtual environments with CPU fallback for quick testing, ML learning enabled"
    },
    "ci-cd": {
        "name": "CI/CD Pipeline",
        "description": "Optimized for continuous integration/deployment",
        "strategy": "docker",
        "gpu_mode": "flexible",
        "validation": True,
        "cache_enabled": False,  # Fresh analysis each time
        "llm_enabled": False,
        "notes": "Reproducible Docker builds with validation"
    },
    "quick": {
        "name": "Quick Analysis",
        "description": "Fastest possible analysis for quick checks",
        "strategy": "auto",
        "gpu_mode": "cpu_fallback",
        "validation": False,
        "cache_enabled": True,
        "llm_enabled": False,
        "learning_enabled": False,
        "notes": "Maximum speed with caching and no validation"
    },
    "learning": {
        "name": "Learning Mode",
        "description": "Optimized for ML learning and pattern discovery",
        "strategy": "auto",
        "gpu_mode": "flexible",
        "validation": True,
        "cache_enabled": True,
        "llm_enabled": True,
        "learning_enabled": True,
        "notes": "Full ML learning capabilities with pattern discovery and adaptive recommendations"
    }
}


def get_preset(name: str) -> Dict[str, Any]:
    """
    Get a preset configuration by name.
    
    Args:
        name: Preset name
        
    Returns:
        Preset configuration dictionary
        
    Raises:
        ValueError: If preset name not found
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Preset '{name}' not found. Available presets: {available}")
    
    return PRESETS[name].copy()


def list_presets() -> Dict[str, str]:
    """
    List all available presets with descriptions.
    
    Returns:
        Dictionary of preset names to descriptions
    """
    return {
        name: config["description"]
        for name, config in PRESETS.items()
    }


def apply_preset_to_config(config: Any, preset_name: str) -> None:
    """
    Apply a preset to a configuration object.
    
    Args:
        config: Configuration object to modify
        preset_name: Name of preset to apply
    """
    preset = get_preset(preset_name)
    
    # Apply preset values
    if hasattr(config, 'defaults'):
        config.defaults.strategy = preset["strategy"]
        config.defaults.gpu_mode = preset["gpu_mode"]
        config.defaults.validation = preset["validation"]
    
    if hasattr(config, 'advanced'):
        config.advanced.cache_enabled = preset.get("cache_enabled", True)
    
    if hasattr(config, 'integrations') and hasattr(config.integrations, 'llm'):
        config.integrations.llm.enabled = preset.get("llm_enabled", False)
    
    # Apply learning system settings if available
    if hasattr(config, 'learning'):
        config.learning.enabled = preset.get("learning_enabled", False)