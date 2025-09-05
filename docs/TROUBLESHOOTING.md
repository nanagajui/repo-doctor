# Repo Doctor Troubleshooting Guide

This guide helps you resolve common issues with Repo Doctor.

## Common Issues

### 1. Analysis Takes Too Long

**Symptoms:**
- Analysis takes more than 10 seconds
- Slow GitHub API responses
- High memory usage

**Solutions:**
```bash
# Enable caching for faster subsequent runs
repo-doctor check <repo-url> --preset quick

# Check cache status
repo-doctor learning-dashboard

# Clear cache if corrupted
rm -rf ~/.repo-doctor/cache/

# Use parallel processing (enabled by default)
repo-doctor check <repo-url> --preset development
```

**Performance Tips:**
- Use `--preset quick` for fastest analysis
- Enable caching with `--preset development`
- Check system health with `repo-doctor health`

### 2. Learning System Not Working

**Symptoms:**
- No learning insights in dashboard
- "Learning system not available" error
- ML features not enabled

**Solutions:**
```bash
# Check if learning system is available
repo-doctor learning-dashboard

# Enable learning with preset
repo-doctor check <repo-url> --preset learning

# Check system health
repo-doctor health

# Verify knowledge base
ls -la ~/.repo-doctor/knowledge/
```

**Configuration:**
```yaml
# In ~/.repo-doctor/config.yaml
learning:
  enabled: true
  pattern_discovery: true
  adaptive_recommendations: true
```

### 3. GitHub API Rate Limiting

**Symptoms:**
- "API rate limit exceeded" errors
- Slow analysis with many repositories
- 403 Forbidden errors

**Solutions:**
```bash
# Check GitHub token
repo-doctor tokens

# Set GitHub token
export GITHUB_TOKEN=your_token_here

# Use caching to reduce API calls
repo-doctor check <repo-url> --preset development

# Check rate limit status
repo-doctor health
```

**Rate Limit Management:**
- Caching reduces API calls by 70%+
- Use `--preset quick` for minimal API usage
- Set `GITHUB_TOKEN` for higher limits

### 4. Docker/Container Issues

**Symptoms:**
- Docker strategy fails
- Container validation errors
- GPU detection issues

**Solutions:**
```bash
# Check Docker installation
docker --version
docker run hello-world

# Check GPU support
nvidia-smi

# Use alternative strategy
repo-doctor check <repo-url> --strategy conda

# Check system profile
repo-doctor health
```

**Docker Troubleshooting:**
- Ensure Docker is running: `systemctl start docker`
- Check GPU support: `docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi`
- Use `--gpu-mode flexible` for compatibility

### 5. LLM Integration Issues

**Symptoms:**
- "LLM not available" errors
- Slow LLM responses
- Connection timeouts

**Solutions:**
```bash
# Check LLM server status
curl http://localhost:1234/v1/models

# Disable LLM for faster analysis
repo-doctor check <repo-url> --disable-llm

# Use different LLM server
repo-doctor check <repo-url> --llm-url http://other-server:1234/v1

# Check LLM configuration
repo-doctor health
```

**LLM Configuration:**
```yaml
# In ~/.repo-doctor/config.yaml
integrations:
  llm:
    enabled: true
    base_url: http://localhost:1234/v1
    model: qwen/qwen3-4b-thinking-2507
    timeout: 30
```

### 6. Cache Issues

**Symptoms:**
- Stale analysis results
- Cache not working
- Disk space issues

**Solutions:**
```bash
# Clear cache
rm -rf ~/.repo-doctor/cache/

# Check cache status
repo-doctor learning-dashboard

# Disable cache temporarily
repo-doctor check <repo-url> --preset production

# Check disk space
df -h ~/.repo-doctor/
```

**Cache Management:**
- Cache TTL: 1 hour (configurable)
- Auto-cleanup of expired entries
- Memory + disk persistence

### 7. Memory Issues

**Symptoms:**
- High memory usage
- Out of memory errors
- Slow performance

**Solutions:**
```bash
# Use quick preset for minimal memory
repo-doctor check <repo-url> --preset quick

# Disable learning system
repo-doctor check <repo-url> --preset production

# Check memory usage
repo-doctor health

# Clear knowledge base if too large
rm -rf ~/.repo-doctor/knowledge/
```

**Memory Optimization:**
- Use `--preset quick` for minimal memory
- Disable learning with `--preset production`
- Monitor with `repo-doctor learning-dashboard`

### 8. Configuration Issues

**Symptoms:**
- Invalid configuration errors
- Settings not applied
- Preset not working

**Solutions:**
```bash
# Check current configuration
repo-doctor health

# Reset to defaults
rm ~/.repo-doctor/config.yaml

# Use specific preset
repo-doctor check <repo-url> --preset ml-research

# Validate configuration
python -c "from repo_doctor.utils.config import Config; Config.load()"
```

**Configuration Validation:**
```bash
# Test configuration loading
python -c "
from repo_doctor.utils.config import Config
config = Config.load()
print('Configuration loaded successfully')
print(f'Strategy: {config.defaults.strategy}')
print(f'LLM enabled: {config.integrations.llm.enabled}')
"
```

## Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Enable debug logging
repo-doctor --log-level DEBUG check <repo-url>

# Log to file
repo-doctor --log-file debug.log --log-level DEBUG check <repo-url>

# Check logs
tail -f debug.log
```

## Performance Monitoring

Monitor system performance and identify bottlenecks:

```bash
# Check system health
repo-doctor health

# View learning dashboard
repo-doctor learning-dashboard

# Monitor cache performance
repo-doctor learning-dashboard | grep -i cache

# Check agent performance
repo-doctor health | grep -i agent
```

## Getting Help

1. **Check system health**: `repo-doctor health`
2. **View learning dashboard**: `repo-doctor learning-dashboard`
3. **Enable debug logging**: `repo-doctor --log-level DEBUG`
4. **Check configuration**: `repo-doctor tokens`
5. **Review logs**: Check `~/.repo-doctor/logs/` directory

## Common Error Messages

### "Learning system not available"
- Install required ML dependencies
- Check knowledge base permissions
- Use `--preset production` to disable learning

### "API rate limit exceeded"
- Set `GITHUB_TOKEN` environment variable
- Use caching with `--preset development`
- Wait for rate limit reset

### "Docker not available"
- Install and start Docker
- Use `--strategy conda` as alternative
- Check Docker permissions

### "LLM connection failed"
- Check LLM server status
- Use `--disable-llm` to skip LLM
- Verify LLM configuration

### "Cache corrupted"
- Clear cache: `rm -rf ~/.repo-doctor/cache/`
- Restart analysis
- Check disk space

## Performance Benchmarks

Expected performance on modern hardware:

- **Small repos** (< 10 dependencies): 2-5 seconds
- **Medium repos** (10-50 dependencies): 5-8 seconds  
- **Large repos** (> 50 dependencies): 8-12 seconds
- **Cached analysis**: 1-3 seconds
- **Learning mode**: +2-3 seconds overhead

Use `repo-doctor learning-dashboard` to monitor actual performance.
