# Performance Features Documentation

This document details the performance optimizations and features in Repo Doctor.

## Overview

Repo Doctor is designed for sub-10-second analysis with intelligent caching, parallel processing, and ML-powered optimizations.

## Caching System

### GitHub API Caching

**Purpose**: Reduce API calls and improve response times

**Features**:
- Memory + disk persistence
- TTL-based expiration (1 hour default)
- Smart invalidation
- Cache statistics and monitoring

**Implementation**:
```python
# Automatic caching in analysis agent
analysis_agent = AnalysisAgent(use_cache=True)

# Cache statistics
stats = analysis_agent.cache.get_stats()
print(f"Hit rate: {stats['hit_rate']}")
print(f"API calls saved: {stats['api_calls_saved']}")
```

**Configuration**:
```yaml
# In ~/.repo-doctor/config.yaml
advanced:
  cache_enabled: true
  cache_ttl: 3600  # 1 hour
  cache_dir: ~/.repo-doctor/cache/
```

### Knowledge Base Caching

**Purpose**: Fast path for recently analyzed repositories

**Features**:
- Recent analysis detection (7 days default)
- Cached resolution retrieval
- Pattern-based similarity matching

**Usage**:
```python
# Check for recent analysis
if knowledge_base.has_recent_analysis(repo_url):
    analysis = knowledge_base.get_recent_analysis(repo_url)
    resolution = knowledge_base.get_cached_resolution(repo_url)
```

## Parallel Processing

### Agent Parallelization

**Purpose**: Run independent operations in parallel

**Implementation**:
```python
# Profile and analysis prep run in parallel
async def profile_system():
    return ProfileAgent().profile()

async def prepare_analysis():
    return AnalysisAgent(config=config, use_cache=True)

# Parallel execution
profile_future = asyncio.create_task(profile_system())
analysis_prep_future = asyncio.create_task(prepare_analysis())

system_profile, analysis_agent = await asyncio.gather(
    profile_future, analysis_prep_future
)
```

**Performance Impact**:
- 30-50% reduction in total analysis time
- Better resource utilization
- Improved user experience

### Async Operations

**Purpose**: Non-blocking I/O operations

**Features**:
- Async GitHub API calls
- Parallel dependency analysis
- Concurrent file processing

## Learning System Performance

### ML-Powered Optimizations

**Purpose**: Continuously improve analysis speed and accuracy

**Features**:
- Pattern-based fast paths
- Strategy success prediction
- Adaptive recommendations
- Learning from successful resolutions

**Performance Benefits**:
- Faster analysis for similar repositories
- Better strategy selection
- Reduced trial-and-error

### Feature Extraction

**Purpose**: Efficient ML feature processing

**Features**:
- Cached feature vectors
- Batch processing
- Optimized data structures

**Implementation**:
```python
# Efficient feature extraction
features = feature_extractor.extract_features(analysis_data, system_profile)
cached_features = ml_storage.get_cached_features(repo_url)
```

## Performance Monitoring

### Real-Time Metrics

**Purpose**: Monitor and optimize performance

**Metrics Tracked**:
- Analysis duration
- Cache hit rates
- Memory usage
- API call counts
- Learning effectiveness

**Access**:
```bash
# View performance metrics
repo-doctor learning-dashboard

# Check system health
repo-doctor health

# Monitor cache performance
repo-doctor learning-dashboard | grep -i cache
```

### Benchmarking

**Purpose**: Measure and compare performance

**Benchmarks**:
- Analysis speed by repository size
- Cache effectiveness
- Memory usage patterns
- Learning system accuracy

**Usage**:
```python
# Run performance benchmarks
pytest tests/test_real_repositories.py::PerformanceBenchmark -v
```

## Optimization Strategies

### Repository Size Optimization

**Small Repos** (< 10 dependencies):
- Use `--preset quick`
- Enable caching
- Skip validation

**Medium Repos** (10-50 dependencies):
- Use `--preset development`
- Enable learning
- Use parallel processing

**Large Repos** (> 50 dependencies):
- Use `--preset learning`
- Enable full caching
- Use ML optimizations

### Memory Optimization

**Low Memory Systems**:
- Use `--preset quick`
- Disable learning system
- Clear cache regularly

**High Memory Systems**:
- Use `--preset learning`
- Enable full caching
- Use ML features

### Network Optimization

**Slow Networks**:
- Enable aggressive caching
- Use `--preset quick`
- Disable LLM features

**Fast Networks**:
- Use `--preset learning`
- Enable all features
- Use parallel processing

## Performance Tuning

### Configuration Tuning

**Cache Settings**:
```yaml
advanced:
  cache_enabled: true
  cache_ttl: 3600  # Adjust based on usage
  cache_size_limit: 1000  # Max cache entries
  cache_cleanup_interval: 3600  # Cleanup frequency
```

**Learning Settings**:
```yaml
learning:
  enabled: true
  pattern_discovery: true
  adaptive_recommendations: true
  learning_threshold: 0.7  # Confidence threshold
```

**Performance Settings**:
```yaml
performance:
  parallel_agents: true
  max_concurrent_requests: 10
  request_timeout: 30
  retry_attempts: 3
```

### Environment Variables

**Performance Tuning**:
```bash
# Cache settings
export REPO_DOCTOR_CACHE_TTL=3600
export REPO_DOCTOR_CACHE_SIZE=1000

# Learning settings
export REPO_DOCTOR_LEARNING_ENABLED=true
export REPO_DOCTOR_LEARNING_THRESHOLD=0.7

# Performance settings
export REPO_DOCTOR_PARALLEL_AGENTS=true
export REPO_DOCTOR_MAX_CONCURRENT=10
```

## Performance Best Practices

### 1. Use Appropriate Presets

- **Quick**: Fastest analysis, minimal features
- **Development**: Balanced performance and features
- **Learning**: Full ML features, best long-term performance
- **Production**: Reliable, validated results

### 2. Enable Caching

- Always use caching for repeated analyses
- Monitor cache hit rates
- Clear cache when needed

### 3. Monitor Performance

- Use `repo-doctor health` regularly
- Check `repo-doctor learning-dashboard`
- Monitor system resources

### 4. Optimize for Use Case

- **Research**: Use learning preset for pattern discovery
- **Production**: Use production preset for reliability
- **Development**: Use development preset for balance

### 5. Handle Large Repositories

- Use learning system for better strategy selection
- Enable full caching
- Consider breaking into smaller components

## Troubleshooting Performance

### Slow Analysis

1. Check cache status: `repo-doctor learning-dashboard`
2. Use quick preset: `--preset quick`
3. Check system health: `repo-doctor health`
4. Monitor memory usage

### High Memory Usage

1. Use quick preset: `--preset quick`
2. Disable learning: `--preset production`
3. Clear cache: `rm -rf ~/.repo-doctor/cache/`
4. Check knowledge base size

### Cache Issues

1. Clear corrupted cache: `rm -rf ~/.repo-doctor/cache/`
2. Check disk space: `df -h ~/.repo-doctor/`
3. Verify cache permissions
4. Monitor cache statistics

## Performance Metrics

### Expected Performance

**Hardware Requirements**:
- CPU: 2+ cores recommended
- RAM: 4GB minimum, 8GB recommended
- Disk: 1GB for cache and knowledge base
- Network: Stable internet connection

**Performance Targets**:
- Small repos: 2-5 seconds
- Medium repos: 5-8 seconds
- Large repos: 8-12 seconds
- Cached analysis: 1-3 seconds

**Cache Performance**:
- Hit rate: 70%+ after warm-up
- API calls saved: 50%+ reduction
- Memory usage: < 500MB typical

### Monitoring Commands

```bash
# System health
repo-doctor health

# Learning dashboard
repo-doctor learning-dashboard

# Cache statistics
repo-doctor learning-dashboard | grep -i cache

# Performance benchmarks
pytest tests/test_real_repositories.py::PerformanceBenchmark -v
```

## Future Performance Improvements

### Planned Optimizations

1. **Distributed Caching**: Redis-based cache for multiple instances
2. **Incremental Analysis**: Only analyze changed dependencies
3. **Predictive Caching**: Pre-cache likely repositories
4. **GPU Acceleration**: CUDA-accelerated ML operations
5. **Streaming Analysis**: Real-time analysis of large repositories

### Research Areas

1. **ML Model Optimization**: Smaller, faster models
2. **Cache Algorithms**: Better eviction strategies
3. **Parallel Processing**: More granular parallelism
4. **Memory Management**: Better memory usage patterns
5. **Network Optimization**: Better API call batching
