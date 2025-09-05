"""Smart LLM server discovery and configuration."""

import asyncio
import os
import platform
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import aiohttp
import json


class LLMDiscovery:
    """Intelligent LLM server discovery and configuration."""
    
    def __init__(self):
        self.is_wsl = self._detect_wsl()
        self.is_windows = platform.system().lower() == "windows"
        self.is_linux = platform.system().lower() == "linux"
        
    def _detect_wsl(self) -> bool:
        """Detect if running in WSL (Windows Subsystem for Linux)."""
        try:
            # Check for WSL-specific indicators
            if os.path.exists("/proc/version"):
                with open("/proc/version", "r") as f:
                    version_info = f.read().lower()
                    return "microsoft" in version_info or "wsl" in version_info
            
            # Check for WSL environment variables
            return "WSL_DISTRO_NAME" in os.environ or "WSLENV" in os.environ
        except Exception:
            return False
    
    def _get_wsl_host_ip(self) -> Optional[str]:
        """Get the Windows host IP from WSL."""
        try:
            # Try to get the Windows host IP from WSL
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Extract IP from output like "default via 172.29.96.1 dev eth0"
                for line in result.stdout.split('\n'):
                    if 'default via' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            return parts[2]
        except Exception:
            pass
        
        # Fallback: try common WSL host IPs
        common_wsl_ips = [
            "172.29.96.1",  # Your current server
            "172.20.240.1",
            "172.17.0.1",
            "192.168.1.1"
        ]
        
        return common_wsl_ips[0]  # Return your known working IP as default
    
    def get_candidate_urls(self) -> List[str]:
        """Get list of candidate LLM server URLs to try."""
        candidates = []
        
        # 1. Check environment variables first
        if "LLM_BASE_URL" in os.environ:
            candidates.append(os.environ["LLM_BASE_URL"])
        
        # 2. Check for custom config file
        config_file = Path.home() / ".repo-doctor" / "llm_config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    if "base_url" in config:
                        candidates.append(config["base_url"])
            except Exception:
                pass
        
        # 3. WSL-specific URLs
        if self.is_wsl:
            wsl_host_ip = self._get_wsl_host_ip()
            if wsl_host_ip:
                candidates.extend([
                    f"http://{wsl_host_ip}:1234/v1",
                    f"http://{wsl_host_ip}:8080/v1",
                    f"http://{wsl_host_ip}:11434/v1",  # Ollama default
                ])
        
        # 4. Localhost URLs (works in most environments)
        candidates.extend([
            "http://localhost:1234/v1",
            "http://127.0.0.1:1234/v1",
            "http://localhost:8080/v1",
            "http://127.0.0.1:8080/v1",
            "http://localhost:11434/v1",  # Ollama default
            "http://127.0.0.1:11434/v1",
        ])
        
        # 5. Docker/container URLs
        candidates.extend([
            "http://host.docker.internal:1234/v1",
            "http://host.docker.internal:8080/v1",
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for url in candidates:
            if url not in seen:
                seen.add(url)
                unique_candidates.append(url)
        
        return unique_candidates
    
    async def discover_llm_server(self, timeout: int = 3) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Discover the best available LLM server.
        
        Returns:
            Tuple of (base_url, server_info) if found, None otherwise
        """
        candidates = self.get_candidate_urls()
        
        # Test each candidate URL
        for url in candidates:
            try:
                server_info = await self._test_llm_server(url, timeout)
                if server_info:
                    return url, server_info
            except Exception:
                continue
        
        return None
    
    async def _test_llm_server(self, base_url: str, timeout: int) -> Optional[Dict[str, Any]]:
        """Test if an LLM server is available at the given URL."""
        try:
            async with aiohttp.ClientSession() as session:
                # Test /models endpoint
                async with session.get(
                    f"{base_url}/models",
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        models_data = await response.json()
                        
                        # Extract server information
                        server_info = {
                            "base_url": base_url,
                            "status": "available",
                            "models": models_data.get("data", []),
                            "model_count": len(models_data.get("data", [])),
                            "server_type": self._detect_server_type(models_data),
                            "environment": self._get_environment_info()
                        }
                        
                        return server_info
        except Exception:
            pass
        
        return None
    
    def _detect_server_type(self, models_data: Dict[str, Any]) -> str:
        """Detect the type of LLM server based on response."""
        try:
            # Check for LM Studio indicators
            if "object" in models_data and models_data.get("object") == "list":
                return "lm_studio"
            
            # Check for Ollama indicators
            models = models_data.get("data", [])
            if models and any("ollama" in str(model).lower() for model in models):
                return "ollama"
            
            # Check for vLLM indicators
            if any("vllm" in str(model).lower() for model in models):
                return "vllm"
            
            return "unknown"
        except Exception:
            return "unknown"
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment."""
        return {
            "platform": platform.system(),
            "is_wsl": self.is_wsl,
            "is_windows": self.is_windows,
            "is_linux": self.is_linux,
            "python_version": platform.python_version(),
        }
    
    def _get_default_model(self) -> str:
        """Get default model from configuration."""
        try:
            from .config import Config
            config = Config.load()
            return config.integrations.llm.model
        except Exception:
            # Fallback if config loading fails
            return "openai/gpt-oss-20b"
    
    def get_smart_config(self, discovered_url: Optional[str] = None) -> Dict[str, Any]:
        """Get smart configuration based on discovered server."""
        # Get default model from config
        default_model = self._get_default_model()
        
        if discovered_url:
            return {
                "enabled": True,
                "base_url": discovered_url,
                "api_key": os.getenv("LLM_API_KEY"),
                "model": default_model,
                "timeout": 30,
                "max_tokens": 512,
                "temperature": 0.1,
                "discovery_method": "auto_detected"
            }
        else:
            return {
                "enabled": False,
                "base_url": "http://localhost:1234/v1",
                "api_key": os.getenv("LLM_API_KEY"),
                "model": default_model,
                "timeout": 30,
                "max_tokens": 512,
                "temperature": 0.1,
                "discovery_method": "fallback"
            }
    
    def save_discovered_config(self, discovered_url: str, server_info: Dict[str, Any]):
        """Save discovered configuration for future use."""
        config_dir = Path.home() / ".repo-doctor"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "llm_config.json"
        config_data = {
            "base_url": discovered_url,
            "server_info": server_info,
            "discovered_at": asyncio.get_event_loop().time(),
            "environment": self._get_environment_info()
        }
        
        try:
            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
        except Exception:
            pass  # Fail silently if can't save config


class SmartLLMConfig:
    """Smart LLM configuration manager."""
    
    def __init__(self):
        self.discovery = LLMDiscovery()
        self._cached_config = None
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes
    
    async def get_config(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get smart LLM configuration."""
        current_time = asyncio.get_event_loop().time()
        
        # Use cached config if still valid
        if (not force_refresh and 
            self._cached_config and 
            (current_time - self._cache_timestamp) < self._cache_ttl):
            return self._cached_config
        
        # Discover LLM server
        discovery_result = await self.discovery.discover_llm_server()
        
        if discovery_result:
            discovered_url, server_info = discovery_result
            config = self.discovery.get_smart_config(discovered_url)
            config["server_info"] = server_info
            
            # Save discovered config
            self.discovery.save_discovered_config(discovered_url, server_info)
        else:
            config = self.discovery.get_smart_config()
        
        # Cache the result
        self._cached_config = config
        self._cache_timestamp = current_time
        
        return config
    
    def get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration when discovery fails."""
        return self.discovery.get_smart_config()


# Global instance for easy access
smart_llm_config = SmartLLMConfig()
