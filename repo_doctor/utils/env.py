"""Environment variable loading utilities with .env file support."""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from .logging_config import get_logger


class EnvLoader:
    """Utility class for loading environment variables with .env file support."""
    
    _loaded = False
    
    @classmethod
    def load_env_files(cls, search_paths: Optional[list] = None) -> bool:
        """
        Load environment variables from .env files.
        
        Args:
            search_paths: List of paths to search for .env files.
                         If None, searches common locations.
        
        Returns:
            True if any .env file was loaded, False otherwise.
        """
        if cls._loaded or not DOTENV_AVAILABLE:
            return cls._loaded
        
        if search_paths is None:
            # Default search paths
            search_paths = [
                Path.cwd() / ".env",  # Current directory
                Path.cwd() / "repo-doctor" / ".env",  # Project subdirectory
                Path.home() / ".repo-doctor" / ".env",  # User config directory
            ]
        
        loaded_any = False
        logger = get_logger(__name__)
        
        for env_path in search_paths:
            env_file = Path(env_path)
            if env_file.exists() and env_file.is_file():
                try:
                    load_dotenv(env_file, override=False)  # Don't override existing env vars
                    loaded_any = True
                    logger.info(f"Loaded environment variables from: {env_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {env_file}: {e}")
        
        cls._loaded = loaded_any
        return loaded_any
    
    @classmethod
    def get_token(cls, token_name: str, fallback_names: Optional[list] = None) -> Optional[str]:
        """
        Get token from environment variables, loading .env files if needed.
        
        Args:
            token_name: Primary token environment variable name
            fallback_names: Alternative names to check
        
        Returns:
            Token value if found, None otherwise
        """
        # Ensure .env files are loaded
        cls.load_env_files()
        
        # Check primary token name
        token = os.getenv(token_name)
        if token:
            return token.strip()
        
        # Check fallback names
        if fallback_names:
            for fallback_name in fallback_names:
                token = os.getenv(fallback_name)
                if token:
                    return token.strip()
        
        return None
    
    @classmethod
    def get_github_token(cls) -> Optional[str]:
        """Get GitHub token from environment, checking multiple possible names."""
        return cls.get_token("GITHUB_TOKEN", ["GH_TOKEN", "GITHUB_ACCESS_TOKEN"])
    
    @classmethod
    def get_hf_token(cls) -> Optional[str]:
        """Get Hugging Face token from environment, checking multiple possible names."""
        return cls.get_token("HF_TOKEN", ["HUGGINGFACE_TOKEN", "HUGGING_FACE_TOKEN"])
    
    @classmethod
    def get_openai_token(cls) -> Optional[str]:
        """Get OpenAI token from environment."""
        return cls.get_token("OPENAI_API_KEY", ["OPENAI_TOKEN"])


def load_environment() -> dict:
    """
    Load and return environment configuration.
    
    Returns:
        Dictionary with available tokens and configuration
    """
    EnvLoader.load_env_files()
    
    return {
        "github_token": EnvLoader.get_github_token(),
        "hf_token": EnvLoader.get_hf_token(), 
        "openai_token": EnvLoader.get_openai_token(),
        "dotenv_available": DOTENV_AVAILABLE,
        "env_loaded": EnvLoader._loaded,
    }


# Auto-load on import
EnvLoader.load_env_files()