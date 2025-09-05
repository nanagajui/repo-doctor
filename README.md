Repo Doctor

A sophisticated CLI tool that diagnoses and resolves GitHub repository compatibility issues with local development environments, specifically targeting ML/AI repositories with complex GPU dependencies.

## Features

- **⚡ Sub-10-second analysis** with intelligent caching and parallel processing
- **🤖 ML-Powered Learning System** that continuously improves recommendations
- **📊 Learning Dashboard** for monitoring AI performance and insights
- **🎯 Preset Profiles** for different use cases (ML research, production, development)
- **🔄 GitHub API Caching** with persistent storage and smart invalidation
- **⚡ Parallel Agent Execution** for faster analysis workflows
- **🧠 Enhanced Agents** with ML capabilities and pattern discovery
- **📈 Performance Monitoring** with real-time metrics and optimization
- **🔧 Automated environment generation** (Docker, Conda, Micromamba, venv)
- **🎮 GPU-aware** compatibility checking for ML workloads
- **🏗️ Three-agent architecture** for comprehensive analysis
- **🤖 LLM Integration** with qwen/qwen3-4b-thinking-2507 for enhanced analysis
- **🔍 Advanced conflict detection** for ML package compatibility
- **💡 AI-powered error diagnosis** and resolution suggestions

## Architecture

### Three-Agent System

1. **Profile Agent**: Captures comprehensive system state (GPU, CUDA, hardware)
2. **Analysis Agent**: Deep repository analysis with multi-source validation
3. **Resolution Agent**: Generates working solutions with multiple strategies

### Project Structure

```
repo_doctor/
├── agents/          # Three-agent system
│   ├── profile.py   # System profiling
│   ├── analysis.py  # Repository analysis
│   └── resolution.py # Solution generation
├── models/          # Pydantic data models
├── strategies/      # Environment generation strategies
│   ├── docker.py    # Docker containers
│   ├── conda.py     # Conda environments
│   ├── micromamba.py # Micromamba environments (2-3x faster)
│   └── venv.py      # Virtual environments
├── knowledge/       # Learning system
├── validators/      # Solution validation
└── utils/          # Configuration and utilities
```

## Installation

```bash
# Clone and install
git clone <repo-url>
cd repo-doctor
pip install -e .

# Or install with agent features
pip install -e ".[agents]"
```

## Quick Start

```bash
# Basic analysis with intelligent caching
repo-doctor check https://github.com/huggingface/diffusers

# Use preset profiles for different use cases
repo-doctor check https://github.com/huggingface/transformers --preset ml-research
repo-doctor check https://github.com/pytorch/pytorch --preset production
repo-doctor check https://github.com/openai/whisper --preset learning

# Advanced usage with validation and learning
repo-doctor check https://github.com/CompVis/stable-diffusion \
    --strategy docker \
    --validate \
    --gpu-mode strict \
    --preset learning

# Custom output directory
repo-doctor check https://github.com/owner/repo --output ./my-env/

# Enable LLM assistance for complex repositories
repo-doctor check https://github.com/complex/ml-project --enable-llm --validate

# View learning system dashboard
repo-doctor learning-dashboard

# Check system health and performance
repo-doctor health
```

## Commands

- `check <repo_url>` - Analyze repository and generate environment
- `learning-dashboard` - View ML learning system performance and insights
- `health` - Check system health and performance metrics
- `tokens` - Verify API token configuration

### Available Options for `check`:
- `--preset` - Use preset configuration (ml-research, production, development, learning, quick)
- `--strategy` - Choose environment strategy (docker|conda|venv|auto)
- `--validate/--no-validate` - Enable/disable solution validation
- `--gpu-mode` - GPU compatibility mode (strict|flexible|cpu_fallback)
- `--output` - Output directory for generated files (default: outputs/{owner}-{repo})
- `--enable-llm/--disable-llm` - Enable/disable LLM assistance
- `--llm-url` - Custom LLM server URL
- `--llm-model` - Custom LLM model name
- `--quick` - Quick mode (skip validation for faster results)
- `--advanced` - Show advanced options

## Preset Profiles

Repo Doctor includes intelligent preset profiles for different use cases:

- **`ml-research`** - ML research and experimentation with learning enabled
- **`production`** - Production deployments with strict validation
- **`development`** - Local development with CPU fallback
- **`learning`** - Full ML learning capabilities with pattern discovery
- **`quick`** - Fastest analysis with caching and no validation
- **`ci-cd`** - CI/CD pipelines with reproducible builds

## Learning System

The ML-powered learning system continuously improves recommendations:

- **Pattern Discovery** - Automatically identifies common ML repository patterns
- **Strategy Prediction** - Predicts success probability for different strategies
- **Adaptive Learning** - Learns from successful resolutions and failures
- **Insight Generation** - Provides actionable insights and recommendations
- **Performance Monitoring** - Tracks learning effectiveness and model accuracy

View learning metrics with: `repo-doctor learning-dashboard`

## Performance Features

- **GitHub API Caching** - Reduces API calls with intelligent caching
- **Parallel Processing** - Agents run in parallel for faster analysis
- **Fast Path Optimization** - Quick results for recently analyzed repositories
- **Memory + Disk Persistence** - Cache survives restarts
- **Performance Monitoring** - Real-time metrics and optimization suggestions

## Configuration

Configuration is stored in `~/.repo-doctor/config.yaml`:

```yaml
defaults:
  strategy: auto  # docker|conda|venv|auto
  validation: true
  gpu_mode: flexible  # strict|flexible|cpu_fallback

knowledge_base:
  location: ~/.repo-doctor/kb/
  
integrations:
  github_token: ${GITHUB_TOKEN}  # Optional for private repos
  llm:
    enabled: false
    base_url: http://localhost:1234/v1
    model: qwen/qwen3-4b-thinking-2507

# Learning system configuration
learning:
  enabled: true
  pattern_discovery: true
  adaptive_recommendations: true

# Performance settings
advanced:
  cache_enabled: true
  cache_ttl: 3600  # 1 hour
  parallel_agents: true
```

## LLM Enhancement Roadmap

Repo Doctor includes a comprehensive plan for expanding LLM integration across all aspects of the application:

### Current LLM Features ✅
- **Documentation Analysis**: Natural language understanding of README files
- **Error Diagnosis**: AI-powered validation failure analysis
- **Strategy Recommendations**: LLM fallback for complex compatibility cases
- **Configuration**: Full CLI and config system integration

### Planned Enhancements 🚀
- **Script Generation**: Custom setup, test, and deployment scripts
- **Advanced Analysis**: Code pattern recognition and ML workflow analysis
- **Tool Integration**: LLM-controlled GitHub API and Docker operations
- **Multi-Model Support**: Specialized models for different tasks
- **Knowledge Enhancement**: Dynamic learning from successful resolutions

See `docs/LLM_ENHANCEMENT_PLAN.md` for the complete roadmap with implementation timeline and technical details.

## Development Status

**✅ PRODUCTION READY**: All phases complete with comprehensive functionality

**Completed Features**:
- ✅ Three-agent architecture (Profile, Analysis, Resolution)
- ✅ Enhanced agents with ML capabilities and pattern discovery
- ✅ Multi-strategy environment generation (Docker, Conda, Venv)
- ✅ ML-powered learning system with adaptive recommendations
- ✅ GitHub API caching with persistent storage
- ✅ Parallel agent execution for performance optimization
- ✅ Preset profiles for different use cases
- ✅ Learning dashboard for monitoring AI performance
- ✅ Container validation with GPU support
- ✅ LLM integration for enhanced analysis
- ✅ Rich CLI interface with progress indicators
- ✅ Comprehensive testing and validation
- ✅ Performance monitoring and optimization

## Contributing

This project follows the three-agent architecture pattern. When contributing:

1. **Profile Agent**: Enhance system detection capabilities
2. **Analysis Agent**: Improve repository parsing and dependency extraction
3. **Resolution Agent**: Add new environment generation strategies

## License

MIT License
