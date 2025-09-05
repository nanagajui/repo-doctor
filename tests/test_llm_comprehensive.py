#!/usr/bin/env python3
"""Comprehensive test suite for LLM integration with maximum coverage."""

import pytest
import asyncio
import json
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import aiohttp
from aioresponses import aioresponses

from repo_doctor.utils.llm import LLMClient, LLMAnalyzer, LLMFactory
from repo_doctor.utils.llm_discovery import LLMDiscovery, SmartLLMConfig, smart_llm_config
from repo_doctor.utils.config import Config


class TestLLMClient:
    """Comprehensive tests for LLMClient class."""

    @pytest.fixture
    def llm_client(self):
        """Create LLM client for testing."""
        return LLMClient(
            base_url="http://localhost:1234/v1",
            api_key="test_key",
            model="test-model",
            timeout=10
        )

    @pytest.fixture
    def llm_client_no_params(self):
        """Create LLM client with minimal parameters."""
        return LLMClient()

    def test_llm_client_initialization(self, llm_client):
        """Test LLM client initialization."""
        assert llm_client.base_url == "http://localhost:1234/v1"
        assert llm_client.api_key == "test_key"
        assert llm_client.model == "test-model"
        assert llm_client.timeout == 10
        assert not llm_client.available
        assert not llm_client._availability_checked

    def test_llm_client_default_initialization(self, llm_client_no_params):
        """Test LLM client with default parameters."""
        assert llm_client_no_params.base_url is None
        assert llm_client_no_params.timeout == 30
        assert llm_client_no_params.use_smart_discovery

    def test_get_default_model_success(self):
        """Test successful model retrieval from config."""
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

    @pytest.mark.asyncio
    async def test_check_availability_cached(self, llm_client):
        """Test availability check with cached result."""
        llm_client._availability_checked = True
        llm_client.available = True
        
        result = await llm_client._check_availability()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_availability_smart_discovery_success(self):
        """Test availability check with smart discovery."""
        from repo_doctor.utils.llm_discovery import smart_llm_config
        client = LLMClient(use_smart_discovery=True)
        
        with patch.object(smart_llm_config, 'get_config', return_value={"enabled": True, "base_url": "http://test:1234/v1"}):
            with aioresponses() as m:
                m.get("http://test:1234/v1/models", payload={"data": []}, status=200)
                
                result = await client._check_availability()
                assert result is True
                assert client.available

    @pytest.mark.asyncio
    async def test_check_availability_smart_discovery_disabled(self):
        """Test availability check with smart discovery disabled."""
        from repo_doctor.utils.llm_discovery import smart_llm_config
        client = LLMClient(use_smart_discovery=True)
        
        with patch.object(smart_llm_config, 'get_config', return_value={"enabled": False}):
            result = await client._check_availability()
            assert result is False

    @pytest.mark.asyncio
    async def test_check_availability_direct_url_success(self, llm_client):
        """Test availability check with direct URL."""
        with aioresponses() as m:
            m.get("http://localhost:1234/v1/models", payload={"data": []}, status=200)
            
            result = await llm_client._check_availability()
            assert result is True
            assert llm_client.available

    @pytest.mark.asyncio
    async def test_check_availability_direct_url_failure(self, llm_client):
        """Test availability check failure with direct URL."""
        with aioresponses() as m:
            m.get("http://localhost:1234/v1/models", status=500)
            
            result = await llm_client._check_availability()
            assert result is False
            assert not llm_client.available

    @pytest.mark.asyncio
    async def test_check_availability_network_error(self, llm_client):
        """Test availability check with network error."""
        with aioresponses() as m:
            m.get("http://localhost:1234/v1/models", exception=aiohttp.ClientError())
            
            result = await llm_client._check_availability()
            assert result is False

    @pytest.mark.asyncio
    async def test_generate_completion_success(self, llm_client):
        """Test successful completion generation."""
        llm_client.available = True
        
        with aioresponses() as m:
            response_data = {
                "choices": [{"message": {"content": "Test response"}}]
            }
            m.post("http://localhost:1234/v1/chat/completions", payload=response_data, status=200)
            
            result = await llm_client.generate_completion("Test prompt")
            assert result == "Test response"

    @pytest.mark.asyncio
    async def test_generate_completion_unavailable(self, llm_client):
        """Test completion generation when service unavailable."""
        llm_client.available = False
        
        result = await llm_client.generate_completion("Test prompt")
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_completion_network_error(self, llm_client):
        """Test completion generation with network error."""
        llm_client.available = True
        
        with aioresponses() as m:
            m.post("http://localhost:1234/v1/chat/completions", exception=aiohttp.ClientError())
            
            result = await llm_client.generate_completion("Test prompt")
            assert result is None

    @pytest.mark.asyncio
    async def test_generate_completion_invalid_response(self, llm_client):
        """Test completion generation with invalid response."""
        llm_client.available = True
        
        with aioresponses() as m:
            m.post("http://localhost:1234/v1/chat/completions", payload={"invalid": "response"}, status=200)
            
            result = await llm_client.generate_completion("Test prompt")
            assert result is None

    def test_extract_response_from_thinking_no_content(self, llm_client):
        """Test extraction with no content."""
        result = llm_client._extract_response_from_thinking("")
        assert result == ""

    def test_extract_response_from_thinking_with_think_tags(self, llm_client):
        """Test extraction with think tags."""
        content = "<think>reasoning</think>Final answer"
        result = llm_client._extract_response_from_thinking(content)
        assert result == "Final answer"

    def test_extract_response_from_thinking_with_thinking_tags(self, llm_client):
        """Test extraction with thinking tags."""
        content = "<thinking>reasoning</thinking>Final answer"
        result = llm_client._extract_response_from_thinking(content)
        assert result == "Final answer"

    def test_extract_response_from_thinking_with_json(self, llm_client):
        """Test extraction with JSON content."""
        content = '<think>reasoning</think>{"key": "value"}'
        result = llm_client._extract_response_from_thinking(content)
        assert result == '{"key": "value"}'

    def test_extract_response_from_thinking_invalid_json(self, llm_client):
        """Test extraction with invalid JSON."""
        content = '<think>reasoning</think>{"invalid": json}'
        result = llm_client._extract_response_from_thinking(content)
        assert result == '{"invalid": json}'

    def test_extract_response_from_thinking_multiple_json(self, llm_client):
        """Test extraction with multiple JSON objects."""
        content = '<think>reasoning</think>{"first": "json"}{"second": "json"}'
        result = llm_client._extract_response_from_thinking(content)
        assert result == '{"second": "json"}'  # Should return the last valid JSON


class TestLLMAnalyzer:
    """Comprehensive tests for LLMAnalyzer class."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create mock LLM client."""
        client = Mock(spec=LLMClient)
        client.generate_completion = AsyncMock()
        client.available = True
        return client

    @pytest.fixture
    def llm_analyzer(self, mock_llm_client):
        """Create LLM analyzer with mock client."""
        return LLMAnalyzer(mock_llm_client)

    @pytest.mark.asyncio
    async def test_enhance_documentation_analysis_success(self, llm_analyzer, mock_llm_client):
        """Test successful documentation analysis."""
        mock_llm_client.generate_completion.return_value = '{"python_versions": ["3.8"], "system_requirements": ["cuda"]}'
        
        result = await llm_analyzer.enhance_documentation_analysis("Test README content")
        
        assert result is not None
        assert result["python_versions"] == ["3.8"]
        assert result["system_requirements"] == ["cuda"]

    @pytest.mark.asyncio
    async def test_enhance_documentation_analysis_no_response(self, llm_analyzer, mock_llm_client):
        """Test documentation analysis with no LLM response."""
        mock_llm_client.generate_completion.return_value = None
        
        result = await llm_analyzer.enhance_documentation_analysis("Test README content")
        assert result is None

    @pytest.mark.asyncio
    async def test_enhance_documentation_analysis_invalid_json(self, llm_analyzer, mock_llm_client):
        """Test documentation analysis with invalid JSON response."""
        mock_llm_client.generate_completion.return_value = "Invalid JSON response"
        
        result = await llm_analyzer.enhance_documentation_analysis("Test README content")
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_complex_compatibility_success(self, llm_analyzer, mock_llm_client):
        """Test successful compatibility analysis."""
        mock_llm_client.generate_completion.return_value = '{"compatibility_score": 85, "risk_level": "medium"}'
        
        analysis_data = {"dependencies": [{"name": "torch", "version": ">=1.0"}]}
        result = await llm_analyzer.analyze_complex_compatibility(analysis_data)
        
        assert result is not None
        assert result["compatibility_score"] == 85
        assert result["risk_level"] == "medium"

    @pytest.mark.asyncio
    async def test_analyze_complex_compatibility_no_response(self, llm_analyzer, mock_llm_client):
        """Test compatibility analysis with no response."""
        mock_llm_client.generate_completion.return_value = None
        
        result = await llm_analyzer.analyze_complex_compatibility({})
        assert result is None

    @pytest.mark.asyncio
    async def test_diagnose_validation_failure_success(self, llm_analyzer, mock_llm_client):
        """Test successful validation failure diagnosis."""
        mock_llm_client.generate_completion.return_value = "CUDA driver issue detected"
        
        error_logs = ["CUDA error: no kernel image available"]
        analysis_data = {"dependencies": [{"name": "torch", "version": ">=1.0"}]}
        
        result = await llm_analyzer.diagnose_validation_failure(error_logs, analysis_data)
        assert result == "CUDA driver issue detected"

    @pytest.mark.asyncio
    async def test_diagnose_validation_failure_no_response(self, llm_analyzer, mock_llm_client):
        """Test validation failure diagnosis with no response."""
        mock_llm_client.generate_completion.return_value = None
        
        result = await llm_analyzer.diagnose_validation_failure([], {})
        assert result is None

    def test_extract_json_from_response_valid(self, llm_analyzer):
        """Test JSON extraction from valid response."""
        response = 'Some text {"key": "value"} more text'
        result = llm_analyzer._extract_json_from_response(response)
        assert result == {"key": "value"}

    def test_extract_json_from_response_invalid(self, llm_analyzer):
        """Test JSON extraction from invalid response."""
        response = "No JSON here"
        result = llm_analyzer._extract_json_from_response(response)
        assert result is None

    def test_extract_json_from_response_multiple_objects(self, llm_analyzer):
        """Test JSON extraction with multiple objects."""
        response = '{"first": 1} {"second": 2}'
        result = llm_analyzer._extract_json_from_response(response)
        assert result == {"second": 2}  # Should return the last valid JSON


class TestLLMFactory:
    """Tests for LLMFactory class."""

    @pytest.mark.asyncio
    async def test_create_client_enabled(self):
        """Test client creation when LLM is enabled."""
        config = Config.load()
        config.integrations.llm.enabled = True
        config.integrations.llm.base_url = "http://test:1234/v1"
        
        client = await LLMFactory.create_client(config)
        assert client is not None
        assert isinstance(client, LLMClient)

    @pytest.mark.asyncio
    async def test_create_client_disabled(self):
        """Test client creation when LLM is disabled."""
        config = Config.load()
        config.integrations.llm.enabled = False
        
        client = await LLMFactory.create_client(config)
        assert client is None

    @pytest.mark.asyncio
    async def test_create_analyzer_success(self):
        """Test analyzer creation with valid client."""
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
        """Test analyzer creation when client creation fails."""
        config = Config.load()
        
        with patch.object(LLMFactory, 'create_client') as mock_create:
            mock_create.return_value = None
            
            analyzer = await LLMFactory.create_analyzer(config)
            assert analyzer is None


class TestLLMDiscovery:
    """Comprehensive tests for LLMDiscovery class."""

    @pytest.fixture
    def llm_discovery(self):
        """Create LLM discovery instance."""
        return LLMDiscovery()

    def test_detect_wsl_true(self, llm_discovery):
        """Test WSL detection when running in WSL."""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="microsoft wsl")):
                result = llm_discovery._detect_wsl()
                assert result is True

    def test_detect_wsl_false(self, llm_discovery):
        """Test WSL detection when not in WSL."""
        with patch('os.path.exists', return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                result = llm_discovery._detect_wsl()
                assert result is False

    def test_detect_wsl_env_var(self, llm_discovery):
        """Test WSL detection via environment variable."""
        with patch.dict(os.environ, {'WSL_DISTRO_NAME': 'Ubuntu'}):
            result = llm_discovery._detect_wsl()
            assert result is True

    def test_get_wsl_host_ip_success(self, llm_discovery):
        """Test successful WSL host IP retrieval."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "default via 172.29.96.1 dev eth0"
            
            result = llm_discovery._get_wsl_host_ip()
            assert result == "172.29.96.1"

    def test_get_wsl_host_ip_failure(self, llm_discovery):
        """Test WSL host IP retrieval failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            
            result = llm_discovery._get_wsl_host_ip()
            assert result == "172.29.96.1"  # Should return fallback

    def test_get_candidate_urls_env_var(self, llm_discovery):
        """Test candidate URL generation with environment variable."""
        with patch.dict(os.environ, {'LLM_BASE_URL': 'http://custom:8080/v1'}):
            urls = llm_discovery.get_candidate_urls()
            assert 'http://custom:8080/v1' in urls

    def test_get_candidate_urls_config_file(self, llm_discovery):
        """Test candidate URL generation with config file."""
        config_data = {"base_url": "http://config:9090/v1"}
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
                urls = llm_discovery.get_candidate_urls()
                assert 'http://config:9090/v1' in urls

    def test_get_candidate_urls_wsl(self, llm_discovery):
        """Test candidate URL generation for WSL."""
        llm_discovery.is_wsl = True
        
        with patch.object(llm_discovery, '_get_wsl_host_ip', return_value='172.29.96.1'):
            urls = llm_discovery.get_candidate_urls()
            assert 'http://172.29.96.1:1234/v1' in urls

    @pytest.mark.asyncio
    async def test_discover_llm_server_success(self, llm_discovery):
        """Test successful LLM server discovery."""
        with patch.object(llm_discovery, 'get_candidate_urls', return_value=['http://test:1234/v1']):
            with patch.object(llm_discovery, '_test_llm_server', return_value={'status': 'available'}):
                result = await llm_discovery.discover_llm_server()
                assert result is not None
                assert result[0] == 'http://test:1234/v1'

    @pytest.mark.asyncio
    async def test_discover_llm_server_failure(self, llm_discovery):
        """Test LLM server discovery failure."""
        with patch.object(llm_discovery, 'get_candidate_urls', return_value=['http://test:1234/v1']):
            with patch.object(llm_discovery, '_test_llm_server', return_value=None):
                result = await llm_discovery.discover_llm_server()
                assert result is None

    @pytest.mark.asyncio
    async def test_test_llm_server_success(self, llm_discovery):
        """Test successful LLM server testing."""
        with aioresponses() as m:
            models_data = {"data": [{"id": "test-model"}]}
            m.get("http://test:1234/v1/models", payload=models_data, status=200)
            
            result = await llm_discovery._test_llm_server("http://test:1234/v1", 5)
            assert result is not None
            assert result["status"] == "available"
            assert result["model_count"] == 1

    @pytest.mark.asyncio
    async def test_test_llm_server_failure(self, llm_discovery):
        """Test LLM server testing failure."""
        with aioresponses() as m:
            m.get("http://test:1234/v1/models", status=500)
            
            result = await llm_discovery._test_llm_server("http://test:1234/v1", 5)
            assert result is None

    def test_detect_server_type_lm_studio(self, llm_discovery):
        """Test LM Studio server type detection."""
        models_data = {"object": "list", "data": []}
        result = llm_discovery._detect_server_type(models_data)
        assert result == "lm_studio"

    def test_detect_server_type_ollama(self, llm_discovery):
        """Test Ollama server type detection."""
        models_data = {"data": [{"id": "ollama/model"}]}
        result = llm_discovery._detect_server_type(models_data)
        assert result == "ollama"

    def test_detect_server_type_unknown(self, llm_discovery):
        """Test unknown server type detection."""
        models_data = {"data": [{"id": "unknown-model"}]}
        result = llm_discovery._detect_server_type(models_data)
        assert result == "unknown"

    def test_get_environment_info(self, llm_discovery):
        """Test environment info retrieval."""
        info = llm_discovery._get_environment_info()
        assert "platform" in info
        assert "python_version" in info
        assert isinstance(info["is_wsl"], bool)

    def test_get_smart_config_with_url(self, llm_discovery):
        """Test smart config generation with discovered URL."""
        config = llm_discovery.get_smart_config("http://discovered:1234/v1")
        assert config["enabled"] is True
        assert config["base_url"] == "http://discovered:1234/v1"
        assert config["discovery_method"] == "auto_detected"

    def test_get_smart_config_without_url(self, llm_discovery):
        """Test smart config generation without discovered URL."""
        config = llm_discovery.get_smart_config()
        assert config["enabled"] is False
        assert config["discovery_method"] == "fallback"

    def test_save_discovered_config(self, llm_discovery):
        """Test saving discovered configuration."""
        with patch('pathlib.Path.mkdir'):
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('json.dump') as mock_dump:
                    llm_discovery.save_discovered_config("http://test:1234/v1", {"status": "available"})
                    mock_dump.assert_called_once()


class TestSmartLLMConfig:
    """Tests for SmartLLMConfig class."""

    @pytest.fixture
    def smart_config(self):
        """Create SmartLLMConfig instance."""
        return SmartLLMConfig()

    @pytest.mark.asyncio
    async def test_get_config_cached(self, smart_config):
        """Test config retrieval with cached result."""
        smart_config._cached_config = {"test": "config"}
        smart_config._cache_timestamp = asyncio.get_event_loop().time()
        
        result = await smart_config.get_config()
        assert result == {"test": "config"}

    @pytest.mark.asyncio
    async def test_get_config_discovery_success(self, smart_config):
        """Test config retrieval with successful discovery."""
        with patch.object(smart_config.discovery, 'discover_llm_server') as mock_discover:
            mock_discover.return_value = ("http://test:1234/v1", {"status": "available"})
            
            with patch.object(smart_config.discovery, 'get_smart_config') as mock_get_config:
                mock_get_config.return_value = {"enabled": True}
                
                result = await smart_config.get_config()
                assert result["enabled"] is True
                assert "server_info" in result

    @pytest.mark.asyncio
    async def test_get_config_discovery_failure(self, smart_config):
        """Test config retrieval with discovery failure."""
        with patch.object(smart_config.discovery, 'discover_llm_server') as mock_discover:
            mock_discover.return_value = None
            
            with patch.object(smart_config.discovery, 'get_smart_config') as mock_get_config:
                mock_get_config.return_value = {"enabled": False}
                
                result = await smart_config.get_config()
                assert result["enabled"] is False

    def test_get_fallback_config(self, smart_config):
        """Test fallback config retrieval."""
        with patch.object(smart_config.discovery, 'get_smart_config') as mock_get_config:
            mock_get_config.return_value = {"fallback": True}
            
            result = smart_config.get_fallback_config()
            assert result["fallback"] is True


# Helper function for mock_open
def mock_open(read_data=""):
    """Create a mock for open() function."""
    from unittest.mock import mock_open as _mock_open
    return _mock_open(read_data=read_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=repo_doctor.utils.llm", "--cov=repo_doctor.utils.llm_discovery"])
