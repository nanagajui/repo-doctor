#!/usr/bin/env python3
"""Test script to analyze local repositories without GitHub API calls."""

import asyncio
import sys
import os
from pathlib import Path

# Add the repo_doctor module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from repo_doctor.agents import AnalysisAgent, ProfileAgent, ResolutionAgent
from repo_doctor.models.analysis import RepositoryInfo
from repo_doctor.utils.config import Config

async def test_local_analysis():
    """Test analysis of a local repository."""
    
    # Create a mock repository info for local testing
    repo_info = RepositoryInfo(
        url="file:///test_repos/worldgen_test",
        name="worldgen_test",
        owner="local",
        description="Test repository simulating WorldGen dependencies",
        language="Python",
        topics=["3d-generation", "gpu", "cuda"]
    )
    
    print("🔍 Testing local repository analysis...")
    print(f"Repository: {repo_info.owner}/{repo_info.name}")
    
    # Initialize agents
    config = Config.load()
    analysis_agent = AnalysisAgent(github_token=None, config=config)
    
    # Test dependency analysis by reading local files
    repo_path = Path(__file__).parent / "worldgen_test"
    
    # Read requirements.txt
    requirements_file = repo_path / "requirements.txt"
    if requirements_file.exists():
        print(f"\n📦 Found requirements.txt:")
        with open(requirements_file) as f:
            print(f.read())
    
    # Read setup.py
    setup_file = repo_path / "setup.py"
    if setup_file.exists():
        print(f"\n⚙️ Found setup.py:")
        with open(setup_file) as f:
            print(f.read())
    
    # Read README.md
    readme_file = repo_path / "README.md"
    if readme_file.exists():
        print(f"\n📖 Found README.md:")
        with open(readme_file) as f:
            print(f.read())
    
    print("\n✅ Local repository analysis test completed!")

if __name__ == "__main__":
    asyncio.run(test_local_analysis())
