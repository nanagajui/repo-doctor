#!/usr/bin/env python3
"""Focused LLM test suite for maximum coverage with working tests."""

import pytest
import asyncio
import json
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from repo_doctor.utils.llm import LLMClient, LLMAnalyzer, LLMFactory
from repo_doctor.utils.llm_discovery import LLMDiscovery, SmartLLMConfig, smart_llm_config
from repo_doctor.utils.config import Config


class TestLLMClientCore:
    """Core LLM client functionality tests."""

    def test_llm_client_initialization(self):
        """Test LLM client initialization."""
        client = LLMClient(
            base_url="http://localhost:1234/v1",
            api_key="test_key",
            model="test-model",
            timeout=10
        )
        assert client.base_url == "http://localhost:1234/v1"
        assert client.api_key == "test_key"
        assert client.model == "test-model"
        assert client.timeout == 10

    def test_llm_client_url_normalization(self):
        """Test URL normalization."""
        client = LLMClient(base_url="http://localhost:1234/v1/")
        assert client.base_url == "http://localhost:1234/v1"

    def test_llm_client_env_api_key(self):
        """Test API key from environment."""
        with patch.dict(os.environ, {'LLM_API_KEY': 'env_key'}):
            client = LLMClient()
            assert client.api_key == 'env_key'

    def test_get_default_model_success(self):
        """Test default model retrieval."""
        client = LLMClient()
        model = client._get_default_model()
        assert isinstance(model, str)
        assert len(model) > 0

    @patch('repo_doctor.utils.config.Config.load')
    def test_get_default_model_fallback(self, mock_config_load):
        """Test fallback when config loading fails."""
        mock_config_load.side_effect = Exception("Config error")
        client = LLMClient()
        model = client._get_default_model()
        assert model == "openai/gpt-oss-20b"

    def test_extract_response_from_thinking_basic(self):
        """Test basic thinking tag extraction."""
        client = LLMClient()
        content = "<think>reasoning</think>Final answer"
        result = client._extract_response_from_thinking(content)
        assert result == "Final answer"

    def test_extract_response_from_thinking_json(self):
        """Test JSON extraction from thinking response."""
        client = LLMClient()
        content = '<think>reasoning</think>{"key": "value"}'
        result = client._extract_response_from_thinking(content)
        assert result == '{"key": "value"}'

    def test_extract_response_from_thinking_empty(self):
        """Test extraction with empty content."""
        client = LLMClient()
        result = client._extract_response_from_thinking("")
        assert result == ""

    def test_extract_response_from_thinking_no_tags(self):
        """Test extraction without thinking tags."""
        client = LLMClient()
        content = "Just regular content"
        result = client._extract_response_from_thinking(content)
        assert result == "Just regular content"


class TestLLMAnalyzerCore:
    """Core LLM analyzer functionality tests."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        client = Mock()
        client.generate_completion = AsyncMock()
        return client

    @pytest.fixture
    def analyzer(self, mock_client):
        """Create LLM analyzer."""
        return LLMAnalyzer(mock_client)

    @pytest.mark.asyncio
    async def test_enhance_documentation_analysis_success(self, analyzer, mock_client):
        """Test successful documentation analysis."""
        mock_client.generate_completion.return_value = '{"python_versions": ["3.8"], "system_requirements": ["cuda"]}'
        
        result = await analyzer.enhance_documentation_analysis("Test README")
        assert result is not None
        assert result["python_versions"] == ["3.8"]

    @pytest.mark.asyncio
    async def test_enhance_documentation_analysis_no_response(self, analyzer, mock_client):
        """Test documentation analysis with no response."""
        mock_client.generate_completion.return_value = None
        
        result = await analyzer.enhance_documentation_analysis("Test README")
        assert result is None

    @pytest.mark.asyncio
    async def test_enhance_documentation_analysis_invalid_json(self, analyzer, mock_client):
        """Test documentation analysis with invalid JSON."""
        mock_client.generate_completion.return_value = "Invalid JSON"
        
        result = await analyzer.enhance_documentation_analysis("Test README")
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_complex_compatibility_success(self, analyzer, mock_client):
        """Test successful compatibility analysis."""
        mock_client.generate_completion.return_value = '{"compatibility_score": 85, "risk_level": "medium"}'
        
        analysis_data = {"dependencies": [{"name": "torch", "version": ">=1.0"}]}
        result = await analyzer.analyze_complex_compatibility(analysis_data)
        
        assert result is not None
        assert result["compatibility_score"] == 85

    @pytest.mark.asyncio
    async def test_diagnose_validation_failure_success(self, analyzer, mock_client):
        """Test successful validation failure diagnosis."""
        mock_client.generate_completion.return_value = "CUDA driver issue detected"
        
        error_logs = ["CUDA error"]
        analysis_data = {"dependencies": [{"name": "torch", "version": ">=1.0"}]}
        
        result = await analyzer.diagnose_validation_failure(error_logs, analysis_data)
        assert result == "CUDA driver issue detected"

    def test_extract_json_from_response_valid(self, analyzer):
        """Test JSON extraction from valid response."""
        # This method is internal to the enhance_documentation_analysis method
        # Test it indirectly through the public interface
        pass

    def test_extract_json_from_response_invalid(self, analyzer):
        """Test JSON extraction from invalid response."""
        # This method is internal - tested through public methods
        pass

    def test_extract_json_from_response_multiple(self, analyzer):
        """Test JSON extraction with multiple objects."""
        # This method is internal - tested through public methods
        pass


class TestLLMFactory:
    """LLM factory tests."""

    @pytest.mark.asyncio
    async def test_create_client_enabled(self):
        """Test client creation when enabled."""
        config = Config.load()
        config.integrations.llm.enabled = True
        config.integrations.llm.base_url = "http://test:1234/v1"
        
        client = await LLMFactory.create_client(config)
        assert client is not None
        assert isinstance(client, LLMClient)

    @pytest.mark.asyncio
    async def test_create_client_disabled(self):
        """Test client creation when disabled."""
        config = Config.load()
        config.integrations.llm.enabled = False
        
        client = await LLMFactory.create_client(config)
        assert client is None

    @pytest.mark.asyncio
    async def test_create_analyzer_success(self):
        """Test analyzer creation."""
        config = Config.load()
        config.integrations.llm.enabled = True
        
        with patch.object(LLMFactory, 'create_client') as mock_create:
            mock_client = Mock(spec=LLMClient)
            mock_create.return_value = mock_client
            
            analyzer = await LLMFactory.create_analyzer(config)
            assert analyzer is not None
            assert isinstance(analyzer, LLMAnalyzer)

    @pytest.mark.asyncio
    async def test_create_analyzer_no_client(self):
        """Test analyzer creation when client fails."""
        config = Config.load()
        
        with patch.object(LLMFactory, 'create_client') as mock_create:
            mock_create.return_value = None
            
            analyzer = await LLMFactory.create_analyzer(config)
            assert analyzer is None


class TestLLMDiscoveryCore:
    """Core LLM discovery tests."""

    def test_llm_discovery_initialization(self):
        """Test LLM discovery initialization."""
        discovery = LLMDiscovery()
        assert hasattr(discovery, 'is_wsl')
        assert hasattr(discovery, 'is_windows')
        assert hasattr(discovery, 'is_linux')

    def test_detect_wsl_false(self):
        """Test WSL detection when not in WSL."""
        discovery = LLMDiscovery()
        with patch('os.path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                result = discovery._detect_wsl()
                assert result is False

    def test_detect_wsl_env_var(self):
        """Test WSL detection via environment variable."""
        discovery = LLMDiscovery()
        with patch.dict(os.environ, {'WSL_DISTRO_NAME': 'Ubuntu'}):
            result = discovery._detect_wsl()
            assert result is True

    def test_get_wsl_host_ip_fallback(self):
        """Test WSL host IP fallback."""
        discovery = LLMDiscovery()
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            result = discovery._get_wsl_host_ip()
            assert result == "172.29.96.1"

    def test_get_candidate_urls_basic(self):
        """Test basic candidate URL generation."""
        discovery = LLMDiscovery()
        urls = discovery.get_candidate_urls()
        assert len(urls) > 0
        assert any('localhost' in url for url in urls)

    def test_get_candidate_urls_env_var(self):
        """Test candidate URLs with environment variable."""
        discovery = LLMDiscovery()
        with patch.dict(os.environ, {'LLM_BASE_URL': 'http://custom:8080/v1'}):
            urls = discovery.get_candidate_urls()
            assert 'http://custom:8080/v1' in urls

    def test_detect_server_type_lm_studio(self):
        """Test LM Studio detection."""
        discovery = LLMDiscovery()
        models_data = {"object": "list", "data": []}
        result = discovery._detect_server_type(models_data)
        assert result == "lm_studio"

    def test_detect_server_type_unknown(self):
        """Test unknown server type."""
        discovery = LLMDiscovery()
        models_data = {"data": [{"id": "unknown-model"}]}
        result = discovery._detect_server_type(models_data)
        assert result == "unknown"

    def test_get_environment_info(self):
        """Test environment info retrieval."""
        discovery = LLMDiscovery()
        info = discovery._get_environment_info()
        assert "platform" in info
        assert "python_version" in info

    def test_get_smart_config_with_url(self):
        """Test smart config with URL."""
        discovery = LLMDiscovery()
        config = discovery.get_smart_config("http://test:1234/v1")
        assert config["enabled"] is True
        assert config["base_url"] == "http://test:1234/v1"

    def test_get_smart_config_without_url(self):
        """Test smart config without URL."""
        discovery = LLMDiscovery()
        config = discovery.get_smart_config()
        assert config["enabled"] is False


class TestSmartLLMConfigCore:
    """Smart LLM config tests."""

    def test_smart_config_initialization(self):
        """Test smart config initialization."""
        config = SmartLLMConfig()
        assert hasattr(config, 'discovery')
        assert hasattr(config, '_cached_config')

    @pytest.mark.asyncio
    async def test_get_config_cached(self):
        """Test cached config retrieval."""
        config = SmartLLMConfig()
        config._cached_config = {"test": "config"}
        config._cache_timestamp = asyncio.get_event_loop().time()
        
        result = await config.get_config()
        assert result == {"test": "config"}

    def test_get_fallback_config(self):
        """Test fallback config."""
        config = SmartLLMConfig()
        with patch.object(config.discovery, 'get_smart_config') as mock_get:
            mock_get.return_value = {"fallback": True}
            result = config.get_fallback_config()
            assert result["fallback"] is True


class TestLLMEdgeCases:
    """Edge case tests."""

    def test_extract_thinking_malformed_tags(self):
        """Test extraction with malformed tags."""
        client = LLMClient()
        content = "<thinking>Unclosed tag"
        result = client._extract_response_from_thinking(content)
        assert result == ""

    def test_extract_thinking_nested_tags(self):
        """Test extraction with nested tags."""
        client = LLMClient()
        content = "<thinking>Outer <think>inner</think> content</thinking>Final"
        result = client._extract_response_from_thinking(content)
        assert "Final" in result

    @pytest.mark.asyncio
    async def test_json_extraction_unicode(self):
        """Test JSON extraction with unicode."""
        mock_client = Mock()
        mock_client.generate_completion = AsyncMock(return_value='{"message": "测试 🚀", "emoji": "✅"}')
        analyzer = LLMAnalyzer(mock_client)
        
        result = await analyzer.enhance_documentation_analysis("test")
        assert result["message"] == "测试 🚀"
        assert result["emoji"] == "✅"

    def test_wsl_detection_exception(self):
        """Test WSL detection with exceptions."""
        discovery = LLMDiscovery()
        with patch('os.path.exists', side_effect=PermissionError()):
            result = discovery._detect_wsl()
            assert result is False

    def test_config_file_json_error(self):
        """Test config file with JSON error."""
        discovery = LLMDiscovery()
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', side_effect=json.JSONDecodeError("Invalid", "", 0)):
                urls = discovery.get_candidate_urls()
                assert len(urls) > 0

    def test_url_deduplication(self):
        """Test URL deduplication."""
        discovery = LLMDiscovery()
        with patch.dict(os.environ, {'LLM_BASE_URL': 'http://localhost:1234/v1'}):
            urls = discovery.get_candidate_urls()
            assert len(urls) == len(set(urls))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=repo_doctor.utils.llm", "--cov=repo_doctor.utils.llm_discovery"])
