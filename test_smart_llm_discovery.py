#!/usr/bin/env python3
"""Test script for smart LLM discovery system."""

import asyncio
import sys
from pathlib import Path

# Add repo-doctor to path
sys.path.insert(0, str(Path(__file__).parent))

from repo_doctor.utils.llm_discovery import LLMDiscovery, smart_llm_config
from repo_doctor.utils.llm import LLMClient, LLMAnalyzer, LLMFactory
from repo_doctor.utils.config import Config


async def test_llm_discovery():
    """Test the LLM discovery system."""
    print("🔍 Testing Smart LLM Discovery System")
    print("=" * 50)
    
    # Initialize discovery
    discovery = LLMDiscovery()
    
    # Show environment info
    print(f"Platform: {platform.system()}")
    print(f"WSL Detected: {discovery.is_wsl}")
    print(f"Windows: {discovery.is_windows}")
    print(f"Linux: {discovery.is_linux}")
    print()
    
    # Show candidate URLs
    candidates = discovery.get_candidate_urls()
    print(f"🔍 Testing {len(candidates)} candidate URLs:")
    for i, url in enumerate(candidates, 1):
        print(f"  {i}. {url}")
    print()
    
    # Test discovery
    print("🚀 Discovering LLM server...")
    discovery_result = await discovery.discover_llm_server(timeout=3)
    
    if discovery_result:
        discovered_url, server_info = discovery_result
        print(f"✅ LLM Server Found!")
        print(f"   URL: {discovered_url}")
        print(f"   Status: {server_info['status']}")
        print(f"   Server Type: {server_info['server_type']}")
        print(f"   Models Available: {server_info['model_count']}")
        print(f"   Environment: {server_info['environment']['platform']}")
        
        if server_info['models']:
            print(f"   Model Names: {[m.get('id', 'unknown') for m in server_info['models'][:3]]}")
    else:
        print("❌ No LLM server found")
        print("💡 Make sure an LLM server is running on one of the candidate URLs")
    
    return discovery_result


async def test_smart_config():
    """Test smart configuration system."""
    print("\n⚙️  Testing Smart Configuration")
    print("=" * 50)
    
    # Get smart config
    config = await smart_llm_config.get_config()
    
    print(f"Smart Config:")
    print(f"  Enabled: {config.get('enabled', False)}")
    print(f"  Base URL: {config.get('base_url', 'None')}")
    print(f"  Model: {config.get('model', 'None')}")
    print(f"  Discovery Method: {config.get('discovery_method', 'None')}")
    print(f"  Timeout: {config.get('timeout', 'None')}")
    
    if 'server_info' in config:
        server_info = config['server_info']
        print(f"  Server Type: {server_info.get('server_type', 'Unknown')}")
        print(f"  Model Count: {server_info.get('model_count', 0)}")
    
    return config


async def test_llm_client():
    """Test LLM client with smart discovery."""
    print("\n🤖 Testing LLM Client with Smart Discovery")
    print("=" * 50)
    
    # Test with smart discovery enabled
    print("Testing with smart discovery enabled...")
    client = LLMClient(use_smart_discovery=True)
    
    # Check availability
    available = await client._check_availability()
    print(f"LLM Available: {available}")
    
    if available:
        print(f"Using URL: {client.base_url}")
        print(f"Using Model: {client.model}")
        
        # Test a simple completion
        print("\nTesting completion...")
        try:
            response = await client.generate_completion(
                "What is Python? Answer in one sentence.",
                max_tokens=50
            )
            if response:
                print(f"✅ Response: {response[:100]}...")
            else:
                print("❌ No response received")
        except Exception as e:
            print(f"❌ Completion failed: {e}")
    else:
        print("❌ LLM not available - skipping completion test")
    
    return available


async def test_llm_analyzer():
    """Test LLM analyzer with smart discovery."""
    print("\n🧠 Testing LLM Analyzer with Smart Discovery")
    print("=" * 50)
    
    # Load config
    config = Config.load()
    config.integrations.llm.enabled = True
    
    # Create analyzer with smart discovery
    analyzer = await LLMFactory.create_analyzer(config, use_smart_discovery=True)
    
    if analyzer:
        print("✅ LLM Analyzer created successfully")
        
        # Test documentation analysis
        print("\nTesting documentation analysis...")
        sample_readme = """
        # My ML Project
        
        This project requires Python 3.9+ and CUDA 11.8.
        Install with: pip install torch torchvision
        
        GPU memory: 8GB minimum required.
        """
        
        try:
            analysis = await analyzer.enhance_documentation_analysis(sample_readme)
            if analysis:
                print("✅ Documentation analysis successful:")
                print(f"   Python versions: {analysis.get('python_versions', [])}")
                print(f"   GPU requirements: {analysis.get('gpu_requirements', 'None')}")
                print(f"   System requirements: {analysis.get('system_requirements', [])}")
            else:
                print("❌ No analysis returned")
        except Exception as e:
            print(f"❌ Documentation analysis failed: {e}")
    else:
        print("❌ Could not create LLM analyzer")


async def test_config_override():
    """Test configuration override."""
    print("\n🔧 Testing Configuration Override")
    print("=" * 50)
    
    # Test with specific URL override
    print("Testing with specific URL override...")
    client = LLMClient(
        base_url="http://172.29.96.1:1234/v1",
        use_smart_discovery=False  # Disable smart discovery
    )
    
    available = await client._check_availability()
    print(f"LLM Available at 172.29.96.1:1234: {available}")
    
    if available:
        print(f"✅ Successfully connected to {client.base_url}")
    else:
        print(f"❌ Could not connect to {client.base_url}")


async def main():
    """Run all tests."""
    print("🚀 Starting Smart LLM Discovery Tests\n")
    
    try:
        # Test discovery
        discovery_result = await test_llm_discovery()
        
        # Test smart config
        config = await test_smart_config()
        
        # Test LLM client
        client_available = await test_llm_client()
        
        # Test LLM analyzer
        await test_llm_analyzer()
        
        # Test config override
        await test_config_override()
        
        print("\n" + "=" * 50)
        print("🎉 Smart LLM Discovery Tests Completed!")
        
        if discovery_result:
            print("✅ LLM server discovered and working")
        else:
            print("⚠️  No LLM server found - check your setup")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import platform
    success = asyncio.run(main())
    exit(0 if success else 1)
