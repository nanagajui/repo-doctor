#!/usr/bin/env python3
"""Integration test for the complete agent system."""

import asyncio
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from github import Github, GithubException

from repo_doctor.agents.profile import ProfileAgent
from repo_doctor.agents.analysis import AnalysisAgent
from repo_doctor.agents.resolution import ResolutionAgent
from repo_doctor.agents.contracts import (
    AgentContractValidator,
    AgentDataFlow,
    AgentPerformanceMonitor
)
from repo_doctor.knowledge import KnowledgeBase


@patch('repo_doctor.agents.analysis.Github')
async def test_complete_agent_workflow(mock_github):
    """Test the complete workflow from profile to resolution."""
    print("🚀 Starting Repo Doctor Agent Integration Test")
    print("=" * 60)
    
    # Initialize performance monitor
    monitor = AgentPerformanceMonitor()
    
    # Step 1: Profile Agent
    print("\n📊 Step 1: System Profiling")
    print("-" * 30)
    
    start_time = time.time()
    profile_agent = ProfileAgent()
    profile = profile_agent.profile()
    profile_duration = time.time() - start_time
    
    print(f"✅ Profile generated in {profile_duration:.2f}s")
    print(f"   CPU Cores: {profile.hardware.cpu_cores}")
    print(f"   Memory: {profile.hardware.memory_gb:.1f} GB")
    print(f"   GPUs: {len(profile.hardware.gpus)}")
    print(f"   CUDA: {profile.software.cuda_version or 'Not available'}")
    print(f"   Compute Score: {profile.compute_score}")
    
    # Validate profile
    try:
        AgentContractValidator.validate_system_profile(profile)
        print("✅ Profile validation passed")
    except Exception as e:
        print(f"❌ Profile validation failed: {e}")
        return False
    
    # Check performance
    if monitor.check_profile_performance(profile_duration):
        print("✅ Profile performance target met")
    else:
        print(f"⚠️  Profile performance exceeded target ({profile_duration:.2f}s > 2.0s)")
    
    mock_repo = MagicMock()
    mock_repo.get_contents.side_effect = GithubException(404, 'Not Found', {})
    mock_github.return_value.get_repo.return_value = mock_repo

    # Step 2: Analysis Agent
    print("\n🔍 Step 2: Repository Analysis")
    print("-" * 30)
    
    # Test with a well-known ML repository
    test_repo = "https://github.com/huggingface/transformers"
    
    start_time = time.time()
    analysis_agent = AnalysisAgent()
    analysis = await analysis_agent.analyze(test_repo, profile)
    analysis_duration = time.time() - start_time
    
    print(f"✅ Analysis completed in {analysis_duration:.2f}s")
    print(f"   Repository: {analysis.repository.owner}/{analysis.repository.name}")
    print(f"   Dependencies: {len(analysis.dependencies)}")
    print(f"   Compatibility Issues: {len(analysis.compatibility_issues)}")
    print(f"   Critical Issues: {len(analysis.get_critical_issues())}")
    print(f"   GPU Required: {analysis.is_gpu_required()}")
    print(f"   Confidence Score: {analysis.confidence_score:.2f}")
    
    # Validate analysis
    try:
        AgentContractValidator.validate_analysis(analysis)
        print("✅ Analysis validation passed")
    except Exception as e:
        print(f"❌ Analysis validation failed: {e}")
        return False
    
    # Check performance
    if monitor.check_analysis_performance(analysis_duration):
        print("✅ Analysis performance target met")
    else:
        print(f"⚠️  Analysis performance exceeded target ({analysis_duration:.2f}s > 10.0s)")
    
    # Step 3: Data Flow Testing
    print("\n🔄 Step 3: Agent Data Flow")
    print("-" * 30)
    
    # Test profile to analysis context
    profile_context = AgentDataFlow.profile_to_analysis_context(profile)
    print(f"✅ Profile context generated")
    print(f"   System has GPU: {profile_context['system_capabilities']['has_gpu']}")
    print(f"   System has CUDA: {profile_context['system_capabilities']['has_cuda']}")
    print(f"   Can run containers: {profile_context['system_capabilities']['can_run_containers']}")
    
    # Test analysis to resolution context
    analysis_context = AgentDataFlow.analysis_to_resolution_context(analysis)
    print(f"✅ Analysis context generated")
    print(f"   Repository: {analysis_context['repository']['name']}")
    print(f"   Python version required: {analysis_context['requirements']['python_version']}")
    print(f"   GPU required: {analysis_context['requirements']['gpu_required']}")
    
    # Step 4: Resolution Agent
    print("\n🛠️  Step 4: Solution Generation")
    print("-" * 30)
    
    start_time = time.time()
    resolution_agent = ResolutionAgent()
    resolution = resolution_agent.resolve(analysis, "docker")
    resolution_duration = time.time() - start_time
    
    print(f"✅ Resolution generated in {resolution_duration:.2f}s")
    print(f"   Strategy: {resolution.strategy.type}")
    print(f"   Generated Files: {len(resolution.generated_files)}")
    print(f"   Setup Commands: {len(resolution.setup_commands)}")
    print(f"   Estimated Size: {resolution.estimated_size_mb} MB")
    
    # Validate resolution
    try:
        AgentContractValidator.validate_resolution(resolution)
        print("✅ Resolution validation passed")
    except Exception as e:
        print(f"❌ Resolution validation failed: {e}")
        return False
    
    # Check performance
    if monitor.check_resolution_performance(resolution_duration):
        print("✅ Resolution performance target met")
    else:
        print(f"⚠️  Resolution performance exceeded target ({resolution_duration:.2f}s > 5.0s)")
    
    # Step 5: Knowledge Base Integration
    print("\n🧠 Step 5: Knowledge Base Integration")
    print("-" * 30)
    
    # Initialize knowledge base
    kb_path = Path.home() / ".repo-doctor" / "knowledge" / "test"
    kb = KnowledgeBase(kb_path)
    
    # Record analysis
    commit_hash = kb.record_analysis(analysis)
    print(f"✅ Analysis recorded with hash: {commit_hash}")
    
    # Get similar analyses
    similar = kb.get_similar_analyses(analysis, limit=3)
    print(f"✅ Found {len(similar)} similar analyses")
    
    # Get success patterns
    patterns = kb.get_success_patterns("docker")
    print(f"✅ Retrieved success patterns for Docker strategy")
    
    # Test knowledge context
    knowledge_context = kb.get_knowledge_context(analysis, resolution)
    print(f"✅ Knowledge context generated")
    print(f"   Repository key: {knowledge_context['repository_key']}")
    print(f"   Strategy used: {knowledge_context['strategy_used']}")
    print(f"   Analysis confidence: {knowledge_context['analysis_confidence']:.2f}")
    
    # Step 6: Performance Summary
    print("\n📈 Step 6: Performance Summary")
    print("-" * 30)
    
    total_time = profile_duration + analysis_duration + resolution_duration
    
    print(f"Profile Agent:     {profile_duration:.2f}s (target: 2.0s)")
    print(f"Analysis Agent:    {analysis_duration:.2f}s (target: 10.0s)")
    print(f"Resolution Agent:  {resolution_duration:.2f}s (target: 5.0s)")
    print(f"Total Time:        {total_time:.2f}s")
    
    # Performance reports
    profile_report = monitor.get_performance_report("profile_agent", profile_duration)
    analysis_report = monitor.get_performance_report("analysis_agent", analysis_duration)
    resolution_report = monitor.get_performance_report("resolution_agent", resolution_duration)
    
    print(f"\nPerformance Ratios:")
    print(f"Profile:     {profile_report['performance_ratio']:.2f}x target")
    print(f"Analysis:    {analysis_report['performance_ratio']:.2f}x target")
    print(f"Resolution:  {resolution_report['performance_ratio']:.2f}x target")
    
    # Overall assessment
    all_targets_met = (
        profile_report['meets_target'] and
        analysis_report['meets_target'] and
        resolution_report['meets_target']
    )
    
    if all_targets_met:
        print("\n🎉 All performance targets met!")
    else:
        print("\n⚠️  Some performance targets exceeded")
    
    print("\n" + "=" * 60)
    print("✅ Integration test completed successfully!")
    print("All agents are working correctly with proper contracts and data flow.")
    
    return True


async def main():
    """Main test function."""
    try:
        success = await test_complete_agent_workflow()
        if success:
            print("\n🎯 All tests passed! The agent system is ready for use.")
            return 0
        else:
            print("\n❌ Some tests failed. Please check the output above.")
            return 1
    except Exception as e:
        print(f"\n💥 Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
