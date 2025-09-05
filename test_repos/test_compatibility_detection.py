#!/usr/bin/env python3
"""Test script to test compatibility detection with WorldGen-like dependencies."""

import asyncio
import sys
from pathlib import Path

# Add the repo_doctor module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from repo_doctor.agents import AnalysisAgent, ProfileAgent, ResolutionAgent
from repo_doctor.models.analysis import (
    Analysis, RepositoryInfo, DependencyInfo, DependencyType, CompatibilityIssue
)
from repo_doctor.utils.config import Config

async def test_worldgen_compatibility():
    """Test compatibility detection for WorldGen-like repository."""
    
    print("🔍 Testing WorldGen compatibility detection...")
    
    # Create a mock analysis that simulates WorldGen
    repo_info = RepositoryInfo(
        url="https://github.com/ZiYang-xie/WorldGen.git",
        name="WorldGen",
        owner="ZiYang-xie",
        description="Generate Any 3D Scene in Seconds",
        language="Python",
        topics=["3d-generation", "gpu", "cuda", "flash-attention"]
    )
    
    # Create dependencies that would cause compatibility issues
    dependencies = [
        DependencyInfo(
            name="torch",
            version=">=2.0.0",
            type=DependencyType.PYTHON,
            source="requirements.txt",
            gpu_required=True,
        ),
        DependencyInfo(
            name="flash-attn",
            version=">=2.0.0",
            type=DependencyType.PYTHON,
            source="requirements.txt",
            gpu_required=True,
        ),
        DependencyInfo(
            name="transformers",
            version=">=4.30.0",
            type=DependencyType.PYTHON,
            source="requirements.txt",
            gpu_required=False,
        ),
    ]
    
    # Create compatibility issues that should be detected
    compatibility_issues = [
        CompatibilityIssue(
            type="cuda_incompatibility",
            severity="critical",
            message="flash-attn requires CUDA 11.8+ but may not support sm_120 architecture (Blackwell GPUs)",
            component="flash-attn",
            suggested_fix="Consider using alternative attention implementations or wait for flash-attn update"
        ),
        CompatibilityIssue(
            type="gpu_requirement",
            severity="warning",
            message="Repository requires GPU but CUDA compatibility with newer architectures unclear",
            component="gpu_dependencies",
            suggested_fix="Verify CUDA compatibility with your GPU architecture"
        ),
    ]
    
    # Create analysis
    analysis = Analysis(
        repository=repo_info,
        dependencies=dependencies,
        python_version_required="3.8",
        cuda_version_required="11.8",
        min_memory_gb=16.0,
        min_gpu_memory_gb=16.0,
        compatibility_issues=compatibility_issues,
        analysis_time=5.0,
        confidence_score=0.85,
    )
    
    print(f"\n📊 Analysis Results:")
    print(f"Repository: {analysis.repository.owner}/{analysis.repository.name}")
    print(f"Dependencies: {len(analysis.dependencies)}")
    print(f"Python Required: {analysis.python_version_required}")
    print(f"CUDA Required: {analysis.cuda_version_required}")
    print(f"Min GPU Memory: {analysis.min_gpu_memory_gb}GB")
    print(f"GPU Required: {analysis.is_gpu_required()}")
    
    print(f"\n⚠️ Compatibility Issues ({len(analysis.compatibility_issues)}):")
    for issue in analysis.compatibility_issues:
        severity_emoji = {
            "critical": "🔴",
            "warning": "🟡", 
            "info": "🔵"
        }.get(issue.severity, "⚪")
        
        print(f"{severity_emoji} {issue.severity.upper()}: {issue.message}")
        if issue.suggested_fix:
            print(f"   💡 Fix: {issue.suggested_fix}")
    
    # Test resolution generation
    print(f"\n💡 Testing resolution generation...")
    config = Config.load()
    resolution_agent = ResolutionAgent(config=config)
    
    try:
        resolution = await resolution_agent.resolve(analysis, preferred_strategy="docker")
        print(f"✅ Generated {len(resolution.generated_files)} files using {resolution.strategy.type.value} strategy")
        
        # Show generated files
        for file in resolution.generated_files:
            print(f"   📄 {file.path} - {file.description}")
            
    except Exception as e:
        print(f"❌ Resolution failed: {str(e)}")
    
    print(f"\n✅ WorldGen compatibility test completed!")

if __name__ == "__main__":
    asyncio.run(test_worldgen_compatibility())
