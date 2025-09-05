"""Integration tests with actual ML repositories - STREAM B Testing Enhancement."""

import asyncio
import pytest
from pathlib import Path
from typing import List, Dict, Any
import time

from repo_doctor.agents import ProfileAgent, AnalysisAgent, ResolutionAgent
from repo_doctor.models.analysis import Analysis
from repo_doctor.models.resolution import Resolution, StrategyType
from repo_doctor.utils.config import Config
from repo_doctor.knowledge import KnowledgeBase
from repo_doctor.cache import GitHubCache


class TestRealRepositories:
    """Integration tests with actual ML repositories."""
    
    # Top ML repositories for testing
    REPOSITORIES = [
        "huggingface/transformers",
        "pytorch/pytorch",
        "tensorflow/tensorflow",
        "openai/whisper",
        "CompVis/stable-diffusion"
    ]
    
    # Smaller repos for faster testing
    FAST_TEST_REPOS = [
        "huggingface/datasets",
        "openai/gym",
    ]
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config.load()
        # Disable LLM for testing to ensure consistent results
        config.integrations.llm.enabled = False
        return config
    
    @pytest.fixture
    def cache(self):
        """Create test cache instance."""
        cache_dir = Path("/tmp/test_repo_doctor_cache")
        return GitHubCache(cache_dir)
    
    @pytest.fixture
    def knowledge_base(self):
        """Create test knowledge base."""
        kb_path = Path("/tmp/test_repo_doctor_kb")
        return KnowledgeBase(kb_path)
    
    @pytest.mark.integration
    @pytest.mark.parametrize("repo_url", FAST_TEST_REPOS)
    async def test_end_to_end_analysis(self, repo_url: str, config, cache):
        """Test complete analysis workflow on real repos."""
        # Profile system
        profile_agent = ProfileAgent()
        system_profile = profile_agent.profile()
        
        assert system_profile is not None
        assert system_profile.hardware is not None
        assert system_profile.software is not None
        
        # Analyze repository with caching
        analysis_agent = AnalysisAgent(config=config, use_cache=True)
        
        start_time = time.time()
        analysis = await analysis_agent.analyze(f"https://github.com/{repo_url}", system_profile)
        analysis_time = time.time() - start_time
        
        # Assertions
        assert analysis is not None
        assert analysis.repository is not None
        assert analysis.repository.name in repo_url
        assert len(analysis.dependencies) > 0
        
        # Performance check - should be under 10 seconds with cache
        assert analysis_time < 10, f"Analysis took {analysis_time:.2f}s, expected <10s"
        
        # Check confidence score
        assert 0 <= analysis.confidence_score <= 1
        
        # Verify cache is working
        if hasattr(analysis_agent, 'cache'):
            cache_stats = analysis_agent.cache.get_stats()
            # On second run, should have cache hits
            analysis2 = await analysis_agent.analyze(f"https://github.com/{repo_url}", system_profile)
            cache_stats2 = analysis_agent.cache.get_stats()
            assert cache_stats2['hits'] > cache_stats['hits'], "Cache should have hits on second run"
    
    @pytest.mark.integration
    @pytest.mark.parametrize("repo_url", FAST_TEST_REPOS)
    async def test_environment_generation(self, repo_url: str, config):
        """Test that generated environments are valid."""
        # Quick analysis
        profile_agent = ProfileAgent()
        system_profile = profile_agent.profile()
        
        analysis_agent = AnalysisAgent(config=config, use_cache=True)
        analysis = await analysis_agent.analyze(f"https://github.com/{repo_url}", system_profile)
        
        # Generate resolution
        resolution_agent = ResolutionAgent(config=config)
        resolution = await resolution_agent.resolve(analysis)
        
        # Assertions
        assert resolution is not None
        assert resolution.strategy is not None
        assert resolution.strategy.type in [StrategyType.DOCKER, StrategyType.CONDA, StrategyType.VENV]
        assert len(resolution.generated_files) > 0
        
        # Check that files have content
        for file in resolution.generated_files:
            assert file.path is not None
            assert file.content is not None
            assert len(file.content) > 0
            
        # Check for essential files based on strategy
        file_paths = [f.path for f in resolution.generated_files]
        if resolution.strategy.type == StrategyType.DOCKER:
            assert "Dockerfile" in file_paths
        elif resolution.strategy.type == StrategyType.CONDA:
            assert "environment.yml" in file_paths
        elif resolution.strategy.type == StrategyType.VENV:
            assert "requirements.txt" in file_paths or "setup.sh" in file_paths
    
    @pytest.mark.integration
    async def test_parallel_agent_execution(self, config):
        """Test parallel execution of agents."""
        repo_url = f"https://github.com/{self.FAST_TEST_REPOS[0]}"
        
        # Test parallel execution as implemented in CLI
        async def profile_system():
            """Profile system in parallel."""
            profile_agent = ProfileAgent()
            return profile_agent.profile()
        
        async def prepare_analysis():
            """Prepare analysis agent in parallel."""
            return AnalysisAgent(config=config, use_cache=True)
        
        # Run both tasks in parallel and measure time
        start_time = time.time()
        profile_future = asyncio.create_task(profile_system())
        analysis_prep_future = asyncio.create_task(prepare_analysis())
        
        system_profile, analysis_agent = await asyncio.gather(
            profile_future, 
            analysis_prep_future
        )
        parallel_time = time.time() - start_time
        
        # Run sequentially for comparison
        start_time = time.time()
        profile_agent = ProfileAgent()
        system_profile_seq = profile_agent.profile()
        analysis_agent_seq = AnalysisAgent(config=config, use_cache=True)
        sequential_time = time.time() - start_time
        
        # Parallel should be faster (or at least not slower)
        assert parallel_time <= sequential_time * 1.1, "Parallel execution should be efficient"
        
        # Both approaches should produce valid results
        assert system_profile is not None
        assert analysis_agent is not None
    
    @pytest.mark.integration
    async def test_fast_path_for_known_repos(self, knowledge_base, config):
        """Test fast path optimization for recently analyzed repositories."""
        repo_url = f"https://github.com/{self.FAST_TEST_REPOS[0]}"
        
        # First analysis
        profile_agent = ProfileAgent()
        system_profile = profile_agent.profile()
        
        analysis_agent = AnalysisAgent(config=config, use_cache=True)
        analysis1 = await analysis_agent.analyze(repo_url, system_profile)
        
        # Record in knowledge base
        resolution_agent = ResolutionAgent(config=config)
        resolution1 = await resolution_agent.resolve(analysis1)
        
        # Save to knowledge base
        knowledge_base.record_analysis(analysis1)
        
        # Test fast path methods
        assert knowledge_base.has_recent_analysis(repo_url) == True
        
        recent_analysis = knowledge_base.get_recent_analysis(repo_url)
        assert recent_analysis is not None
        
        # Test with old date - should return False
        assert knowledge_base.has_recent_analysis(repo_url, max_age_days=0) == False
    
    @pytest.mark.integration
    async def test_cache_performance(self, cache, config):
        """Test cache performance improvements."""
        repo_url = f"https://github.com/{self.FAST_TEST_REPOS[0]}"
        
        # Clear cache to start fresh
        cache.clear()
        
        # First analysis without cache hits
        analysis_agent = AnalysisAgent(config=config, use_cache=True)
        
        start_time = time.time()
        analysis1 = await analysis_agent.analyze(repo_url)
        first_run_time = time.time() - start_time
        
        # Get cache stats after first run
        if hasattr(analysis_agent, 'cache'):
            stats1 = analysis_agent.cache.get_stats()
            initial_misses = stats1['misses']
            
            # Second analysis should use cache
            start_time = time.time()
            analysis2 = await analysis_agent.analyze(repo_url)
            second_run_time = time.time() - start_time
            
            stats2 = analysis_agent.cache.get_stats()
            
            # Assertions
            assert stats2['hits'] > 0, "Should have cache hits on second run"
            assert stats2['api_calls_saved'] > 0, "Should save API calls"
            assert second_run_time < first_run_time, "Cached run should be faster"
            
            # Cache hit rate should be good
            hit_rate = stats2['hits'] / (stats2['hits'] + stats2['misses']) * 100
            assert hit_rate > 30, f"Cache hit rate {hit_rate:.1f}% should be >30%"


class PerformanceBenchmark:
    """Performance benchmarks for repository analysis."""
    
    @pytest.mark.benchmark
    async def benchmark_analysis_speed(self):
        """Measure analysis speed across different repository sizes."""
        repos = {
            "small": "openai/gym",  # Small repo
            "medium": "huggingface/datasets",  # Medium repo
            "large": "huggingface/transformers"  # Large repo
        }
        
        config = Config.load()
        config.integrations.llm.enabled = False
        
        results = {}
        for size, repo in repos.items():
            profile_agent = ProfileAgent()
            system_profile = profile_agent.profile()
            
            analysis_agent = AnalysisAgent(config=config, use_cache=True)
            
            start_time = time.time()
            analysis = await analysis_agent.analyze(f"https://github.com/{repo}", system_profile)
            duration = time.time() - start_time
            
            results[size] = {
                "repo": repo,
                "duration": duration,
                "dependencies": len(analysis.dependencies),
                "issues": len(analysis.compatibility_issues)
            }
        
        # Print results
        print("\nAnalysis Speed Benchmark:")
        for size, data in results.items():
            print(f"  {size}: {data['duration']:.2f}s - {data['dependencies']} deps")
        
        # All should complete within reasonable time
        assert all(r['duration'] < 15 for r in results.values()), "All analyses should complete <15s"
    
    @pytest.mark.benchmark
    async def benchmark_cache_effectiveness(self):
        """Benchmark cache effectiveness across multiple runs."""
        repo = "openai/gym"
        runs = 5
        
        config = Config.load()
        config.integrations.llm.enabled = False
        
        cache = GitHubCache(Path("/tmp/benchmark_cache"))
        cache.clear()
        
        times = []
        for i in range(runs):
            analysis_agent = AnalysisAgent(config=config, use_cache=True)
            
            start_time = time.time()
            await analysis_agent.analyze(f"https://github.com/{repo}")
            duration = time.time() - start_time
            times.append(duration)
            
            if hasattr(analysis_agent, 'cache'):
                stats = analysis_agent.cache.get_stats()
                print(f"Run {i+1}: {duration:.2f}s - Cache hits: {stats['hits']}")
        
        # First run should be slowest
        assert times[0] == max(times), "First run should be slowest"
        
        # Subsequent runs should be faster
        avg_cached = sum(times[1:]) / len(times[1:])
        assert avg_cached < times[0] * 0.7, "Cached runs should be >30% faster"