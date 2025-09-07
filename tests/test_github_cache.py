"""Tests for GitHub cache functionality."""

import json
import pickle
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from repo_doctor.cache.github_cache import CacheEntry, GitHubCache, CachedGitHubHelper


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_cache_entry_creation(self):
        """Test CacheEntry creation."""
        data = {"test": "data"}
        timestamp = time.time()
        ttl = 3600
        
        entry = CacheEntry(data, timestamp, ttl)
        
        assert entry.data == data
        assert entry.timestamp == timestamp
        assert entry.ttl == ttl
        assert entry.hits == 0

    def test_cache_entry_is_valid_fresh(self):
        """Test cache entry validity when fresh."""
        data = {"test": "data"}
        timestamp = time.time()
        ttl = 3600
        
        entry = CacheEntry(data, timestamp, ttl)
        
        assert entry.is_valid() is True

    def test_cache_entry_is_valid_expired(self):
        """Test cache entry validity when expired."""
        data = {"test": "data"}
        timestamp = time.time() - 7200  # 2 hours ago
        ttl = 3600  # 1 hour TTL
        
        entry = CacheEntry(data, timestamp, ttl)
        
        assert entry.is_valid() is False

    def test_cache_entry_get_increments_hits(self):
        """Test that get() increments hit counter."""
        data = {"test": "data"}
        timestamp = time.time()
        ttl = 3600
        
        entry = CacheEntry(data, timestamp, ttl)
        
        result = entry.get()
        assert result == data
        assert entry.hits == 1
        
        # Call again
        result = entry.get()
        assert result == data
        assert entry.hits == 2


class TestGitHubCache:
    """Test GitHubCache class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = GitHubCache(cache_dir=Path(self.temp_dir), default_ttl=3600)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_initialization(self):
        """Test GitHubCache initialization."""
        assert self.cache.default_ttl == 3600
        assert self.cache.cache_dir == Path(self.temp_dir)
        assert len(self.cache.memory_cache) == 0
        assert self.cache.stats["hits"] == 0
        assert self.cache.stats["misses"] == 0

    def test_cache_initialization_default_dir(self):
        """Test GitHubCache initialization with default directory."""
        with patch('pathlib.Path.home') as mock_home:
            mock_home.return_value = Path("/tmp/test_home")
            with patch('pathlib.Path.mkdir'):
                cache = GitHubCache()
                expected_dir = Path("/tmp/test_home") / ".repo-doctor" / "cache" / "github"
                assert cache.cache_dir == expected_dir

    def test_generate_cache_key_url_only(self):
        """Test cache key generation with URL only."""
        url = "https://api.github.com/repos/owner/repo"
        key = self.cache._generate_cache_key(url)
        
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

    def test_generate_cache_key_with_params(self):
        """Test cache key generation with parameters."""
        url = "https://api.github.com/repos/owner/repo"
        params = {"page": 1, "per_page": 100}
        
        key1 = self.cache._generate_cache_key(url, params)
        key2 = self.cache._generate_cache_key(url, {"per_page": 100, "page": 1})
        
        # Should be same regardless of param order
        assert key1 == key2
        
        # Should be different from URL-only key
        key3 = self.cache._generate_cache_key(url)
        assert key1 != key3

    def test_set_and_get_memory_cache(self):
        """Test setting and getting from memory cache."""
        url = "https://api.github.com/repos/owner/repo"
        data = {"name": "repo", "stars": 100}
        
        # Set data
        self.cache.set(url, data)
        
        # Get data
        result = self.cache.get(url)
        
        assert result == data
        assert self.cache.stats["hits"] == 1
        assert self.cache.stats["misses"] == 0
        assert self.cache.stats["api_calls_saved"] == 1

    def test_get_cache_miss(self):
        """Test cache miss."""
        url = "https://api.github.com/repos/nonexistent/repo"
        
        result = self.cache.get(url)
        
        assert result is None
        assert self.cache.stats["hits"] == 0
        assert self.cache.stats["misses"] == 1

    def test_set_and_get_with_params(self):
        """Test setting and getting with parameters."""
        url = "https://api.github.com/repos/owner/repo/issues"
        params = {"state": "open", "page": 1}
        data = [{"id": 1, "title": "Issue 1"}]
        
        # Set data
        self.cache.set(url, data, params)
        
        # Get data
        result = self.cache.get(url, params)
        
        assert result == data
        assert self.cache.stats["hits"] == 1

    def test_set_with_custom_ttl(self):
        """Test setting data with custom TTL."""
        url = "https://api.github.com/repos/owner/repo"
        data = {"name": "repo"}
        custom_ttl = 1800
        
        self.cache.set(url, data, ttl=custom_ttl)
        
        cache_key = self.cache._generate_cache_key(url)
        entry = self.cache.memory_cache[cache_key]
        
        assert entry.ttl == custom_ttl

    def test_expired_entry_removal(self):
        """Test automatic removal of expired entries."""
        url = "https://api.github.com/repos/owner/repo"
        data = {"name": "repo"}
        
        # Set data with very short TTL
        self.cache.set(url, data, ttl=1)
        
        # Verify it's cached
        result = self.cache.get(url)
        assert result == data
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be removed automatically
        result = self.cache.get(url)
        assert result is None
        # Note: evictions count includes both memory and disk cleanup
        assert self.cache.stats["evictions"] >= 1

    def test_invalidate_existing_entry(self):
        """Test invalidating an existing cache entry."""
        url = "https://api.github.com/repos/owner/repo"
        data = {"name": "repo"}
        
        # Set data
        self.cache.set(url, data)
        
        # Verify it's cached
        assert self.cache.get(url) == data
        
        # Invalidate
        result = self.cache.invalidate(url)
        
        assert result is True
        assert self.cache.get(url) is None
        assert self.cache.stats["evictions"] == 1

    def test_invalidate_nonexistent_entry(self):
        """Test invalidating a non-existent cache entry."""
        url = "https://api.github.com/repos/nonexistent/repo"
        
        result = self.cache.invalidate(url)
        
        assert result is False
        assert self.cache.stats["evictions"] == 0

    def test_clear_cache(self):
        """Test clearing all cache entries."""
        # Add multiple entries
        urls = [
            "https://api.github.com/repos/owner/repo1",
            "https://api.github.com/repos/owner/repo2",
            "https://api.github.com/repos/owner/repo3"
        ]
        
        for url in urls:
            self.cache.set(url, {"name": url.split("/")[-1]})
        
        # Verify entries exist
        assert len(self.cache.memory_cache) == 3
        
        # Clear cache
        self.cache.clear()
        
        # Verify cache is empty
        assert len(self.cache.memory_cache) == 0
        assert self.cache.stats["hits"] == 0
        assert self.cache.stats["misses"] == 0
        assert self.cache.stats["evictions"] == 0

    def test_cleanup_expired_memory_cache(self):
        """Test cleanup of expired entries from memory cache."""
        # Add fresh entry
        self.cache.set("https://api.github.com/repos/owner/fresh", {"name": "fresh"})
        
        # Add expired entry manually
        cache_key = self.cache._generate_cache_key("https://api.github.com/repos/owner/expired")
        expired_entry = CacheEntry({"name": "expired"}, time.time() - 7200, 3600)
        self.cache.memory_cache[cache_key] = expired_entry
        
        # Cleanup
        removed = self.cache.cleanup_expired()
        
        assert removed >= 1  # May include disk cleanup too
        assert len(self.cache.memory_cache) == 1
        # Check that fresh entry is still there
        fresh_key = self.cache._generate_cache_key("https://api.github.com/repos/owner/fresh")
        assert fresh_key in self.cache.memory_cache

    def test_cleanup_expired_disk_cache(self):
        """Test cleanup of expired entries from disk cache."""
        # Create expired cache file manually
        cache_key = "expired_key"
        cache_file = self.cache.cache_dir / f"{cache_key}.cache"
        
        expired_data = {
            "data": {"name": "expired"},
            "timestamp": time.time() - 7200,  # 2 hours ago
            "ttl": 3600  # 1 hour TTL
        }
        
        with open(cache_file, "wb") as f:
            pickle.dump(expired_data, f)
        
        # Cleanup
        removed = self.cache.cleanup_expired()
        
        assert removed == 1
        assert not cache_file.exists()

    def test_cleanup_corrupted_cache_files(self):
        """Test cleanup of corrupted cache files."""
        # Create corrupted cache file
        cache_file = self.cache.cache_dir / "corrupted.cache"
        cache_file.write_text("invalid pickle data")
        
        # Cleanup
        removed = self.cache.cleanup_expired()
        
        assert removed == 1
        assert not cache_file.exists()

    def test_get_stats_empty_cache(self):
        """Test getting statistics from empty cache."""
        stats = self.cache.get_stats()
        
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == "0.0%"
        assert stats["evictions"] == 0
        assert stats["api_calls_saved"] == 0
        assert stats["memory_entries"] == 0
        assert stats["disk_entries"] == 0

    def test_get_stats_with_data(self):
        """Test getting statistics with cache data."""
        # Add some cache activity
        self.cache.set("https://api.github.com/repos/owner/repo1", {"name": "repo1"})
        self.cache.set("https://api.github.com/repos/owner/repo2", {"name": "repo2"})
        
        # Generate hits and misses
        self.cache.get("https://api.github.com/repos/owner/repo1")  # Hit
        self.cache.get("https://api.github.com/repos/owner/repo1")  # Hit
        self.cache.get("https://api.github.com/repos/owner/nonexistent")  # Miss
        
        stats = self.cache.get_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == "66.7%"
        assert stats["api_calls_saved"] == 2
        assert stats["memory_entries"] == 2

    def test_save_to_disk_success(self):
        """Test successful saving to disk."""
        cache_key = "test_key"
        data = {"test": "data"}
        timestamp = time.time()
        ttl = 3600
        
        self.cache._save_to_disk(cache_key, data, timestamp, ttl)
        
        cache_file = self.cache.cache_dir / f"{cache_key}.cache"
        assert cache_file.exists()
        
        # Verify content
        with open(cache_file, "rb") as f:
            saved_data = pickle.load(f)
        
        assert saved_data["data"] == data
        assert saved_data["timestamp"] == timestamp
        assert saved_data["ttl"] == ttl

    def test_save_to_disk_error_handling(self):
        """Test error handling when saving to disk fails."""
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch.object(self.cache.logger, 'warning') as mock_warning:
                self.cache._save_to_disk("test_key", {"data": "test"}, time.time(), 3600)
                mock_warning.assert_called_once()

    def test_load_from_disk_success(self):
        """Test successful loading from disk."""
        cache_key = "test_key"
        data = {"test": "data"}
        timestamp = time.time()
        ttl = 3600
        
        # Save data first
        self.cache._save_to_disk(cache_key, data, timestamp, ttl)
        
        # Load data
        loaded_data = self.cache._load_from_disk(cache_key)
        
        assert loaded_data is not None
        assert loaded_data["data"] == data
        assert loaded_data["timestamp"] == timestamp
        assert loaded_data["ttl"] == ttl

    def test_load_from_disk_nonexistent_file(self):
        """Test loading from non-existent disk file."""
        result = self.cache._load_from_disk("nonexistent_key")
        assert result is None

    def test_load_from_disk_expired_file(self):
        """Test loading expired file from disk."""
        cache_key = "expired_key"
        cache_file = self.cache.cache_dir / f"{cache_key}.cache"
        
        # Create expired data
        expired_data = {
            "data": {"name": "expired"},
            "timestamp": time.time() - 7200,  # 2 hours ago
            "ttl": 3600  # 1 hour TTL
        }
        
        with open(cache_file, "wb") as f:
            pickle.dump(expired_data, f)
        
        # Try to load
        result = self.cache._load_from_disk(cache_key)
        
        assert result is None
        assert not cache_file.exists()  # Should be removed
        assert self.cache.stats["evictions"] == 1

    def test_load_from_disk_corrupted_file(self):
        """Test loading corrupted file from disk."""
        cache_key = "corrupted_key"
        cache_file = self.cache.cache_dir / f"{cache_key}.cache"
        
        # Create corrupted file
        cache_file.write_text("invalid pickle data")
        
        # Try to load
        result = self.cache._load_from_disk(cache_key)
        
        assert result is None
        assert not cache_file.exists()  # Should be removed

    def test_load_persistent_cache_on_init(self):
        """Test loading persistent cache on initialization."""
        # Create cache files manually
        cache_data = [
            {"key": "key1", "data": {"name": "repo1"}},
            {"key": "key2", "data": {"name": "repo2"}}
        ]
        
        for item in cache_data:
            cache_file = self.cache.cache_dir / f"{item['key']}.cache"
            cached_data = {
                "data": item["data"],
                "timestamp": time.time(),
                "ttl": 3600
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)
        
        # Create new cache instance (should load existing files)
        with patch.object(self.cache.logger, 'info') as mock_info:
            new_cache = GitHubCache(cache_dir=Path(self.temp_dir))
            mock_info.assert_called_once()
            assert "Loaded 2 cache entries" in mock_info.call_args[0][0]
        
        assert len(new_cache.memory_cache) == 2

    def test_disk_cache_integration(self):
        """Test integration between memory and disk cache."""
        url = "https://api.github.com/repos/owner/repo"
        data = {"name": "repo", "stars": 100}
        
        # Set data (should save to both memory and disk)
        self.cache.set(url, data)
        
        # Clear memory cache
        self.cache.memory_cache.clear()
        
        # Get data (should load from disk to memory)
        result = self.cache.get(url)
        
        assert result == data
        assert len(self.cache.memory_cache) == 1  # Should be loaded back to memory


class TestCachedGitHubHelper:
    """Test CachedGitHubHelper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_helper = Mock()
        self.mock_cache = Mock()
        self.cached_helper = CachedGitHubHelper(self.mock_helper, self.mock_cache)

    @pytest.mark.asyncio
    async def test_get_repository_info_cache_hit(self):
        """Test get_repository_info with cache hit."""
        repo_url = "https://github.com/owner/repo"
        cached_data = {"name": "repo", "stars": 100}
        
        self.mock_cache.get.return_value = cached_data
        
        result = await self.cached_helper.get_repository_info(repo_url)
        
        assert result == cached_data
        self.mock_cache.get.assert_called_once_with(repo_url)
        self.mock_helper.get_repository_info.assert_not_called()
        self.mock_cache.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_repository_info_cache_miss(self):
        """Test get_repository_info with cache miss."""
        repo_url = "https://github.com/owner/repo"
        api_data = {"name": "repo", "stars": 100}
        
        self.mock_cache.get.return_value = None
        # Make the helper method async
        async def mock_get_repo_info(url):
            return api_data
        self.mock_helper.get_repository_info = mock_get_repo_info
        
        result = await self.cached_helper.get_repository_info(repo_url)
        
        assert result == api_data
        self.mock_cache.get.assert_called_once_with(repo_url)
        self.mock_cache.set.assert_called_once_with(repo_url, api_data)

    @pytest.mark.asyncio
    async def test_get_file_content_cache_hit(self):
        """Test get_file_content with cache hit."""
        repo_url = "https://github.com/owner/repo"
        file_path = "README.md"
        cached_content = "# Test Repository"
        
        self.mock_cache.get.return_value = cached_content
        
        result = await self.cached_helper.get_file_content(repo_url, file_path)
        
        assert result == cached_content
        expected_cache_key = f"{repo_url}:{file_path}"
        self.mock_cache.get.assert_called_once_with(expected_cache_key)
        self.mock_helper.get_file_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_file_content_cache_miss(self):
        """Test get_file_content with cache miss."""
        repo_url = "https://github.com/owner/repo"
        file_path = "README.md"
        file_content = "# Test Repository"
        
        self.mock_cache.get.return_value = None
        # Make the helper method async
        async def mock_get_file_content(url, path):
            return file_content
        self.mock_helper.get_file_content = mock_get_file_content
        
        result = await self.cached_helper.get_file_content(repo_url, file_path)
        
        assert result == file_content
        expected_cache_key = f"{repo_url}:{file_path}"
        self.mock_cache.get.assert_called_once_with(expected_cache_key)
        self.mock_cache.set.assert_called_once_with(expected_cache_key, file_content, ttl=1800)

    def test_parse_repo_url_passthrough(self):
        """Test parse_repo_url passes through to helper."""
        url = "https://github.com/owner/repo"
        expected_result = {"owner": "owner", "name": "repo", "full_name": "owner/repo"}
        
        self.mock_helper.parse_repo_url.return_value = expected_result
        
        result = self.cached_helper.parse_repo_url(url)
        
        assert result == expected_result
        self.mock_helper.parse_repo_url.assert_called_once_with(url)

    def test_get_cache_stats_passthrough(self):
        """Test get_cache_stats passes through to cache."""
        expected_stats = {
            "hits": 10,
            "misses": 5,
            "hit_rate": "66.7%",
            "api_calls_saved": 10
        }
        
        self.mock_cache.get_stats.return_value = expected_stats
        
        result = self.cached_helper.get_cache_stats()
        
        assert result == expected_stats
        self.mock_cache.get_stats.assert_called_once()

    def test_cached_helper_with_default_cache(self):
        """Test CachedGitHubHelper with default cache."""
        helper = CachedGitHubHelper(self.mock_helper)
        
        assert helper.helper == self.mock_helper
        assert isinstance(helper.cache, GitHubCache)


class TestGitHubCacheIntegration:
    """Integration tests for GitHub cache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = GitHubCache(cache_dir=Path(self.temp_dir), default_ttl=3600)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_cache_workflow(self):
        """Test complete cache workflow."""
        # Test data
        repos = [
            {"url": "https://api.github.com/repos/owner/repo1", "data": {"name": "repo1", "stars": 100}},
            {"url": "https://api.github.com/repos/owner/repo2", "data": {"name": "repo2", "stars": 200}},
            {"url": "https://api.github.com/repos/owner/repo3", "data": {"name": "repo3", "stars": 300}}
        ]
        
        # Set multiple entries
        for repo in repos:
            self.cache.set(repo["url"], repo["data"])
        
        # Verify all entries are cached
        for repo in repos:
            result = self.cache.get(repo["url"])
            assert result == repo["data"]
        
        # Check statistics
        stats = self.cache.get_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 0
        assert stats["memory_entries"] == 3
        assert stats["disk_entries"] == 3
        
        # Invalidate one entry
        self.cache.invalidate(repos[1]["url"])
        
        # Verify invalidation
        assert self.cache.get(repos[1]["url"]) is None
        assert self.cache.get(repos[0]["url"]) == repos[0]["data"]
        
        # Clear all cache
        self.cache.clear()
        
        # Verify cache is empty
        for repo in repos:
            assert self.cache.get(repo["url"]) is None
        
        stats = self.cache.get_stats()
        assert stats["memory_entries"] == 0
        assert stats["disk_entries"] == 0

    def test_cache_persistence_across_instances(self):
        """Test cache persistence across different instances."""
        url = "https://api.github.com/repos/owner/repo"
        data = {"name": "repo", "stars": 100}
        
        # Set data in first instance
        self.cache.set(url, data)
        
        # Create new instance with same cache directory
        new_cache = GitHubCache(cache_dir=Path(self.temp_dir))
        
        # Should load data from disk
        result = new_cache.get(url)
        assert result == data

    def test_error_handling_robustness(self):
        """Test that cache handles errors gracefully."""
        # Test cache operations with disk errors
        self.cache.set("test_url", {"data": "test"})
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Should not raise exception
            result = self.cache.get("test_url")
            # Should still work from memory cache
            assert result == {"data": "test"}
        
        # Test disk save error handling
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch.object(self.cache.logger, 'warning') as mock_warning:
                # Should not raise exception, just log warning
                self.cache.set("test_url_2", {"data": "test2"})
                mock_warning.assert_called_once()
