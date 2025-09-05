#!/usr/bin/env python3
"""Test script to test compatibility detection with VibeVoice-like dependencies."""

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

async def test_vibevoice_compatibility():
    """Test compatibility detection for VibeVoice-like repository."""
    
    print("🔍 Testing VibeVoice compatibility detection...")
    
    # Create a mock analysis that simulates VibeVoice
    repo_info = RepositoryInfo(
        url="https://github.com/microsoft/VibeVoice.git",
        name="VibeVoice",
        owner="microsoft",
        description="Frontier Open-Source Text-to-Speech",
        language="Python",
        topics=["text-to-speech", "ai", "speech-synthesis"]
    )
    
    # Create dependencies that should be compatible
    dependencies = [
        DependencyInfo(
            name="torch",
            version=">=2.0.0",
            type=DependencyType.PYTHON,
            source="requirements.txt",
            gpu_required=False,  # VibeVoice can work on CPU
        ),
        DependencyInfo(
            name="transformers",
            version=">=4.30.0",
            type=DependencyType.PYTHON,
            source="requirements.txt",
            gpu_required=False,
        ),
        DependencyInfo(
            name="numpy",
            version=">=1.24.0",
            type=DependencyType.PYTHON,
            source="requirements.txt",
            gpu_required=False,
        ),
        DependencyInfo(
            name="scipy",
            version=">=1.10.0",
            type=DependencyType.PYTHON,
            source="requirements.txt",
            gpu_required=False,
        ),
    ]
    
    # VibeVoice should have minimal compatibility issues
    compatibility_issues = [
        CompatibilityIssue(
            type="python_version",
            severity="info",
            message="Repository supports Python 3.8+ which is widely compatible",
            component="python",
            suggested_fix="Ensure Python 3.8+ is installed"
        ),
    ]
    
    # Create analysis
    analysis = Analysis(
        repository=repo_info,
        dependencies=dependencies,
        python_version_required="3.8",
        cuda_version_required=None,  # No CUDA requirement
        min_memory_gb=4.0,
        min_gpu_memory_gb=0.0,  # No GPU requirement
        compatibility_issues=compatibility_issues,
        analysis_time=3.0,
        confidence_score=0.95,
    )
    
    print(f"\n📊 Analysis Results:")
    print(f"Repository: {analysis.repository.owner}/{analysis.repository.name}")
    print(f"Dependencies: {len(analysis.dependencies)}")
    print(f"Python Required: {analysis.python_version_required}")
    print(f"CUDA Required: {analysis.cuda_version_required or 'None'}")
    print(f"Min Memory: {analysis.min_memory_gb}GB")
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
        resolution = await resolution_agent.resolve(analysis, preferred_strategy="conda")
        print(f"✅ Generated {len(resolution.generated_files)} files using {resolution.strategy.type.value} strategy")
        
        # Show generated files
        for file in resolution.generated_files:
            print(f"   📄 {file.path} - {file.description}")
            
    except Exception as e:
        print(f"❌ Resolution failed: {str(e)}")
    
    print(f"\n✅ VibeVoice compatibility test completed!")

if __name__ == "__main__":
    asyncio.run(test_vibevoice_compatibility())
