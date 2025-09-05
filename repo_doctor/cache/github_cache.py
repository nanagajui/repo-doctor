"""GitHub API caching layer for performance optimization."""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta

from ..utils.logging_config import get_logger


class CacheEntry:
    """Represents a cached API response."""
    
    def __init__(self, data: Any, timestamp: float, ttl: int) -> None:
        self.data = data
        self.timestamp = timestamp
        self.ttl = ttl
        self.hits = 0
    
    def is_valid(self) -> bool:
        """Check if cache entry is still valid."""
        return (time.time() - self.timestamp) < self.ttl
    
    def get(self) -> Any:
        """Get cached data and increment hit counter."""
        self.hits += 1
        return self.data


class GitHubCache:
    """
    Caching layer for GitHub API responses to reduce latency and API calls.
    
    Features:
    - TTL-based expiration
    - Memory and disk persistence
    - Cache statistics
    - Automatic cleanup of expired entries
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, default_ttl: int = 3600) -> None:
        """
        Initialize GitHub cache.
        
        Args:
            cache_dir: Directory for persistent cache storage
            default_ttl: Default time-to-live in seconds (1 hour)
        """
        self.logger = get_logger(__name__)
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "api_calls_saved": 0
        }
        
        # Set up persistent cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".repo-doctor" / "cache" / "github"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load persistent cache on initialization
        self._load_persistent_cache()
    
    def _generate_cache_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate a unique cache key for the request."""
        key_parts = [url]
        if params:
            # Sort params for consistent key generation
            sorted_params = sorted(params.items())
            key_parts.append(str(sorted_params))
        
        key_string = "|".join(key_parts)
        # Use hash for shorter keys
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, url: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        Get cached response for a GitHub API request.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            Cached data if available and valid, None otherwise
        """
        cache_key = self._generate_cache_key(url, params)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if entry.is_valid():
                self.stats["hits"] += 1
                self.stats["api_calls_saved"] += 1
                return entry.get()
            else:
                # Remove expired entry
                del self.memory_cache[cache_key]
                self.stats["evictions"] += 1
        
        # Check persistent cache
        cached_data = self._load_from_disk(cache_key)
        if cached_data:
            self.stats["hits"] += 1
            self.stats["api_calls_saved"] += 1
            # Populate memory cache for faster access
            self.memory_cache[cache_key] = CacheEntry(
                cached_data["data"],
                cached_data["timestamp"],
                cached_data["ttl"]
            )
            return cached_data["data"]
        
        self.stats["misses"] += 1
        return None
    
    def set(self, url: str, data: Any, params: Optional[Dict] = None, 
            ttl: Optional[int] = None) -> None:
        """
        Cache a GitHub API response.
        
        Args:
            url: API endpoint URL
            data: Response data to cache
            params: Query parameters
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        cache_key = self._generate_cache_key(url, params)
        ttl = ttl or self.default_ttl
        timestamp = time.time()
        
        # Store in memory cache
        self.memory_cache[cache_key] = CacheEntry(data, timestamp, ttl)
        
        # Persist to disk
        self._save_to_disk(cache_key, data, timestamp, ttl)
    
    def invalidate(self, url: str, params: Optional[Dict] = None) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            True if entry was found and invalidated, False otherwise
        """
        cache_key = self._generate_cache_key(url, params)
        
        # Remove from memory cache
        removed = False
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
            removed = True
        
        # Remove from disk cache
        cache_file = self.cache_dir / f"{cache_key}.cache"
        if cache_file.exists():
            cache_file.unlink()
            removed = True
        
        if removed:
            self.stats["evictions"] += 1
        
        return removed
    
    def clear(self) -> None:
        """Clear all cache entries."""
        # Clear memory cache
        self.memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.cache"):
            cache_file.unlink()
        
        # Reset stats except total API calls saved
        api_calls_saved = self.stats["api_calls_saved"]
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "api_calls_saved": api_calls_saved
        }
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        removed = 0
        
        # Clean memory cache
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if not entry.is_valid()
        ]
        for key in expired_keys:
            del self.memory_cache[key]
            removed += 1
        
        # Clean disk cache
        current_time = time.time()
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                
                if (current_time - cached_data["timestamp"]) >= cached_data["ttl"]:
                    cache_file.unlink()
                    removed += 1
            except Exception:
                # Remove corrupted cache files
                cache_file.unlink()
                removed += 1
        
        self.stats["evictions"] += removed
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "evictions": self.stats["evictions"],
            "api_calls_saved": self.stats["api_calls_saved"],
            "memory_entries": len(self.memory_cache),
            "disk_entries": len(list(self.cache_dir.glob("*.cache")))
        }
    
    def _save_to_disk(self, cache_key: str, data: Any, timestamp: float, ttl: int) -> None:
        """Save cache entry to disk."""
        cache_file = self.cache_dir / f"{cache_key}.cache"
        cached_data = {
            "data": data,
            "timestamp": timestamp,
            "ttl": ttl
        }
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            # Log error but don't fail - disk cache is optional
            self.logger.warning(f"Failed to save to disk cache: {e}")
    
    def _load_from_disk(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cache entry from disk."""
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
            
            # Check if still valid
            if (time.time() - cached_data["timestamp"]) < cached_data["ttl"]:
                return cached_data
            else:
                # Remove expired file
                cache_file.unlink()
                self.stats["evictions"] += 1
                return None
        except Exception:
            # Remove corrupted cache file
            cache_file.unlink()
            return None
    
    def _load_persistent_cache(self) -> None:
        """Load valid entries from disk cache into memory on startup."""
        loaded = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_key = cache_file.stem
                cached_data = self._load_from_disk(cache_key)
                if cached_data:
                    self.memory_cache[cache_key] = CacheEntry(
                        cached_data["data"],
                        cached_data["timestamp"],
                        cached_data["ttl"]
                    )
                    loaded += 1
            except Exception:
                # Skip corrupted files
                continue
        
        if loaded > 0:
            self.logger.info(f"Loaded {loaded} cache entries from disk")


class CachedGitHubHelper:
    """
    Wrapper for GitHubHelper that adds caching capabilities.
    
    This class wraps the existing GitHubHelper to add transparent caching
    without modifying the original implementation.
    """
    
    def __init__(self, github_helper: Any, cache: Optional[GitHubCache] = None) -> None:
        """
        Initialize cached GitHub helper.
        
        Args:
            github_helper: Original GitHubHelper instance
            cache: GitHubCache instance (creates new if not provided)
        """
        self.helper = github_helper
        self.cache = cache or GitHubCache()
    
    async def get_repository_info(self, repo_url: str) -> Any:
        """Get repository info with caching."""
        # Check cache first
        cached = self.cache.get(repo_url)
        if cached:
            return cached
        
        # Fetch from API
        result = await self.helper.get_repository_info(repo_url)
        
        # Cache the result
        self.cache.set(repo_url, result)
        
        return result
    
    async def get_file_content(self, repo_url: str, file_path: str) -> Any:
        """Get file content with caching."""
        cache_key = f"{repo_url}:{file_path}"
        
        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Fetch from API
        result = await self.helper.get_file_content(repo_url, file_path)
        
        # Cache with shorter TTL for file contents (they change more often)
        self.cache.set(cache_key, result, ttl=1800)  # 30 minutes
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()