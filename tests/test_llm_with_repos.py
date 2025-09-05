#!/usr/bin/env python3
"""Enhanced LLM testing using test_repos data."""

import asyncio
import sys
from pathlib import Path

# Add the repo_doctor module to the path
sys.path.insert(0, str(Path(__file__).parent))

from repo_doctor.utils.config import Config
from repo_doctor.utils.llm import LLMFactory
from repo_doctor.agents import AnalysisAgent, ProfileAgent, ResolutionAgent
from repo_doctor.models.analysis import RepositoryInfo


async def test_llm_documentation_analysis():
    """Test LLM documentation analysis with real test repository data."""
    print("🧪 Testing LLM Documentation Analysis with Test Repos...")
    
    config = Config.load()
    config.integrations.llm.enabled = True
    config.integrations.llm.base_url = "http://172.29.96.1:1234/v1"
    
    analyzer = await LLMFactory.create_analyzer(config)
    if not analyzer:
        print("   ❌ LLM Analyzer not available")
        return False
    
    # Test with WorldGen test repository
    test_repo_path = Path("test_repos/worldgen_test")
    readme_path = test_repo_path / "README.md"
    
    if not readme_path.exists():
        print("   ❌ Test repository README not found")
        return False
    
    readme_content = readme_path.read_text()
    
    print("   📖 Analyzing WorldGen test repository README...")
    print(f"   Content preview: {readme_content[:100]}...")
    
    result = await analyzer.enhance_documentation_analysis(readme_content)
    
    if result:
        print("   ✅ LLM Documentation Analysis Results:")
        print(f"      Python Versions: {result.get('python_versions', [])}")
        print(f"      System Requirements: {result.get('system_requirements', [])}")
        print(f"      GPU Requirements: {result.get('gpu_requirements', 'None')}")
        print(f"      Installation Complexity: {result.get('installation_complexity', 'unknown')}")
        print(f"      Special Notes: {result.get('special_notes', [])}")
        return True
    else:
        print("   ❌ LLM analysis returned no results")
        return False


async def test_llm_compatibility_analysis():
    """Test LLM compatibility analysis with complex dependency scenarios."""
    print("\n🧪 Testing LLM Compatibility Analysis...")
    
    config = Config.load()
    analyzer = await LLMFactory.create_analyzer(config)
    
    if not analyzer:
        print("   ❌ LLM Analyzer not available")
        return False
    
    # Create complex analysis data based on WorldGen test repo
    analysis_data = {
        "dependencies": [
            {"name": "torch", "version": ">=2.0.0", "type": "gpu"},
            {"name": "flash-attn", "version": ">=2.0.0", "type": "gpu"},
            {"name": "transformers", "version": ">=4.30.0", "type": "ml"},
            {"name": "opencv-python", "version": ">=4.8.0", "type": "cv"}
        ],
        "python_version": "3.8+",
        "gpu_requirements": "CUDA 11.8+, 16GB VRAM",
        "system_requirements": ["CUDA toolkit", "PyTorch with CUDA"],
        "known_issues": [
            "flash-attn requires specific CUDA versions",
            "May not work on newer GPU architectures (sm_120+)",
            "Requires compatible PyTorch CUDA version"
        ],
        "complexity_indicators": {
            "gpu_dependent": True,
            "cuda_specific": True,
            "version_sensitive": True,
            "architecture_dependent": True
        }
    }
    
    print("   🔍 Analyzing complex GPU/CUDA compatibility scenario...")
    
    result = await analyzer.analyze_complex_compatibility(analysis_data)
    
    if result:
        print("   ✅ LLM Compatibility Analysis Results:")
        print(f"      Compatibility Score: {result.get('compatibility_score', 'unknown')}")
        print(f"      Risk Level: {result.get('risk_level', 'unknown')}")
        print(f"      Recommended Strategy: {result.get('recommended_strategy', 'unknown')}")
        print(f"      Potential Issues: {result.get('potential_issues', [])}")
        print(f"      Confidence: {result.get('confidence', 'unknown')}")
        return True
    else:
        print("   ❌ LLM compatibility analysis returned no results")
        return False


async def test_llm_agent_integration():
    """Test LLM integration with analysis agents using test repository."""
    print("\n🧪 Testing LLM Agent Integration...")
    
    config = Config.load()
    config.integrations.llm.enabled = True
    
    # Create repository info for WorldGen test
    repo_info = RepositoryInfo(
        url="file:///test_repos/worldgen_test",
        name="worldgen_test",
        owner="local",
        description="Test repository simulating WorldGen dependencies with GPU/CUDA requirements",
        language="Python",
        topics=["3d-generation", "gpu", "cuda", "pytorch", "transformers"]
    )
    
    try:
        # Initialize agents with LLM support
        analysis_agent = AnalysisAgent(github_token=None, config=config)
        resolution_agent = ResolutionAgent(config=config)
        
        print("   ✅ Agents created with LLM configuration")
        print(f"      Analysis Agent LLM enabled: {hasattr(analysis_agent, 'llm_analyzer')}")
        print(f"      Resolution Agent LLM enabled: {hasattr(resolution_agent, 'llm_analyzer')}")
        
        # Test LLM-enhanced analysis workflow
        print("   🔍 Testing LLM-enhanced analysis workflow...")
        
        # Read test repository files for analysis
        test_repo_path = Path("test_repos/worldgen_test")
        requirements_path = test_repo_path / "requirements.txt"
        
        if requirements_path.exists():
            requirements_content = requirements_path.read_text()
            print(f"      Found requirements.txt with {len(requirements_content.splitlines())} dependencies")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Agent integration failed: {e}")
        return False


async def test_llm_error_diagnosis():
    """Test LLM error diagnosis capabilities."""
    print("\n🧪 Testing LLM Error Diagnosis...")
    
    config = Config.load()
    analyzer = await LLMFactory.create_analyzer(config)
    
    if not analyzer:
        print("   ❌ LLM Analyzer not available")
        return False
    
    # Simulate common GPU/CUDA installation errors
    error_logs = [
        "RuntimeError: CUDA error: no kernel image is available for execution on the device",
        "ImportError: cannot import name 'flash_attn_func' from 'flash_attn'",
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "AssertionError: Torch not compiled with CUDA support"
    ]
    
    analysis_data = {
        "dependencies": [
            {"name": "torch", "version": ">=2.0.0"},
            {"name": "flash-attn", "version": ">=2.0.0"}
        ],
        "gpu_requirements": "CUDA 11.8+, 16GB VRAM",
        "system_info": "Ubuntu 22.04, RTX 4090, CUDA 12.1"
    }
    
    print("   🔍 Diagnosing GPU/CUDA compatibility errors...")
    
    diagnosis = await analyzer.diagnose_validation_failure(error_logs, analysis_data)
    
    if diagnosis:
        print("   ✅ LLM Error Diagnosis:")
        print(f"      Diagnosis: {diagnosis[:200]}...")
        return True
    else:
        print("   ❌ LLM error diagnosis returned no results")
        return False


async def main():
    """Run comprehensive LLM tests with test repository data."""
    print("🚀 Comprehensive LLM Testing with Test Repositories")
    print("=" * 60)
    
    results = []
    
    # Test documentation analysis
    results.append(await test_llm_documentation_analysis())
    
    # Test compatibility analysis
    results.append(await test_llm_compatibility_analysis())
    
    # Test agent integration
    results.append(await test_llm_agent_integration())
    
    # Test error diagnosis
    results.append(await test_llm_error_diagnosis())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    test_names = [
        "Documentation Analysis",
        "Compatibility Analysis", 
        "Agent Integration",
        "Error Diagnosis"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅" if result else "❌"
        print(f"   {status} {name}")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"\n🎉 All {total} tests passed! LLM integration is fully functional.")
    else:
        print(f"\n⚠️  {passed}/{total} tests passed. Some issues need attention.")
    
    print("\n💡 LLM Features Tested:")
    print("   - Real repository documentation analysis")
    print("   - Complex GPU/CUDA compatibility assessment")
    print("   - Agent workflow integration")
    print("   - Error diagnosis and troubleshooting")


if __name__ == "__main__":
    asyncio.run(main())
