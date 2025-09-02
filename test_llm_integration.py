#!/usr/bin/env python3
"""Test script for LLM integration in Repo Doctor."""

import asyncio
import os
import sys
from pathlib import Path

# Add repo-doctor to path
sys.path.insert(0, str(Path(__file__).parent))

from repo_doctor.utils.config import Config
from repo_doctor.utils.llm import LLMFactory, LLMClient, LLMAnalyzer
from repo_doctor.agents.analysis import AnalysisAgent
from repo_doctor.agents.resolution import ResolutionAgent


async def test_llm_client():
    """Test basic LLM client functionality."""
    print("🧪 Testing LLM Client...")
    
    # Test multiple possible URLs
    test_urls = [
        "http://172.29.96.1:1234/v1",
        "http://127.0.0.1:1234/v1",
        "http://localhost:1234/v1",
        "http://127.0.0.1:8080/v1",
        "http://localhost:8080/v1"
    ]
    
    available = False
    working_url = None
    
    for url in test_urls:
        print(f"   Testing {url}...")
        client = LLMClient(
            base_url=url,
            model="qwen/qwen3-4b-thinking-2507",
            timeout=5
        )
        
        # Check availability
        if await client._check_availability():
            available = True
            working_url = url
            print(f"   ✅ LLM Server Available at: {url}")
            
            # Test simple completion
            response = await client.generate_completion(
                "What is Python? Answer in one sentence.",
                max_tokens=50
            )
            print(f"   Test Response: {response[:100] if response else 'None'}...")
            break
        else:
            print(f"   ❌ No server at {url}")
    
    if not available:
        print("   ❌ No LLM server found at any tested URL")
        print("   💡 To start an LLM server:")
        print("      - LM Studio: Load qwen/qwen3-4b-thinking-2507 and start server")
        print("      - Ollama: ollama serve && ollama run qwen:3-4b-thinking")
        print("      - Or any OpenAI-compatible API server")
    
    return available


async def test_llm_analyzer():
    """Test LLM analyzer functionality."""
    print("\n🧪 Testing LLM Analyzer...")
    
    config = Config.load()
    config.integrations.llm.enabled = True
    config.integrations.llm.base_url = "http://172.29.96.1:1234/v1"
    config.integrations.llm.model = "qwen/qwen3-4b-thinking-2507"
    
    analyzer = LLMFactory.create_analyzer(config)
    
    if not analyzer:
        print("   ❌ LLM Analyzer not available")
        return False
    
    # Test documentation analysis
    readme_content = """
    # Test ML Project
    
    This project requires Python 3.9+ and CUDA 11.8 for GPU acceleration.
    
    ## Requirements
    - NVIDIA GPU with 8GB+ VRAM
    - Python 3.9 or higher
    - CUDA 11.8 toolkit
    
    ## Installation
    ```bash
    pip install torch torchvision
    ```
    """
    
    doc_analysis = await analyzer.enhance_documentation_analysis(readme_content)
    print(f"   Documentation Analysis: {'✅' if doc_analysis else '❌'}")
    
    if doc_analysis:
        print(f"   - Python Versions: {doc_analysis.get('python_versions', [])}")
        print(f"   - GPU Requirements: {doc_analysis.get('gpu_requirements', 'None')}")
        print(f"   - Complexity: {doc_analysis.get('installation_complexity', 'Unknown')}")
    
    return doc_analysis is not None


async def test_integration_workflow():
    """Test full integration workflow."""
    print("\n🧪 Testing Integration Workflow...")
    
    # Setup configuration
    config = Config.load()
    config.integrations.llm.enabled = True
    config.integrations.llm.base_url = "http://172.29.96.1:1234/v1"
    config.integrations.llm.model = "qwen/qwen3-4b-thinking-2507"
    
    # Test with a simple repository (using a mock scenario)
    print("   Testing with mock repository analysis...")
    
    # This would normally be done with a real repository
    # For testing, we'll just verify the agents can be created with LLM config
    try:
        analysis_agent = AnalysisAgent(config=config)
        resolution_agent = ResolutionAgent(config=config)
        
        print("   ✅ Agents created successfully with LLM configuration")
        print(f"   - Analysis Agent LLM: {'✅' if analysis_agent.llm_analyzer else '❌'}")
        print(f"   - Resolution Agent LLM: {'✅' if resolution_agent.llm_analyzer else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\n🧪 Testing Configuration System...")
    
    # Test loading default config
    config = Config.load()
    print(f"   Default Config Loaded: ✅")
    print(f"   - LLM Enabled: {config.integrations.llm.enabled}")
    print(f"   - LLM Model: {config.integrations.llm.model}")
    print(f"   - LLM URL: {config.integrations.llm.base_url}")
    
    # Test LLM factory
    client = LLMFactory.create_client(config)
    analyzer = LLMFactory.create_analyzer(config)
    
    print(f"   LLM Client Created: {'✅' if client else '❌'}")
    print(f"   LLM Analyzer Created: {'✅' if analyzer else '❌'}")
    
    return True


async def main():
    """Run all tests."""
    print("🚀 Repo Doctor LLM Integration Test Suite")
    print("=" * 50)
    
    # Test configuration
    config_ok = test_configuration()
    
    # Test LLM client
    client_ok = await test_llm_client()
    
    # Test LLM analyzer (only if client works)
    analyzer_ok = False
    if client_ok:
        analyzer_ok = await test_llm_analyzer()
    
    # Test integration workflow
    integration_ok = await test_integration_workflow()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Configuration: {'✅' if config_ok else '❌'}")
    print(f"   LLM Client: {'✅' if client_ok else '❌'}")
    print(f"   LLM Analyzer: {'✅' if analyzer_ok else '❌'}")
    print(f"   Integration: {'✅' if integration_ok else '❌'}")
    
    if client_ok and analyzer_ok and integration_ok:
        print("\n🎉 All tests passed! LLM integration is ready.")
        print("\n💡 Usage Examples:")
        print("   # Enable LLM for analysis")
        print("   repo-doctor check https://github.com/user/repo --enable-llm")
        print("   # Use custom LLM server")
        print("   repo-doctor check https://github.com/user/repo --enable-llm --llm-url http://localhost:8080/v1")
        print("   # Use different model")
        print("   repo-doctor check https://github.com/user/repo --enable-llm --llm-model custom-model")
    else:
        print("\n⚠️  Some tests failed. Check LLM server configuration.")
        print("   Make sure qwen/qwen3-4b-thinking-2507 is running on http://localhost:1234/v1")


if __name__ == "__main__":
    asyncio.run(main())
