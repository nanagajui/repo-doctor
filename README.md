# Repo Doctor

A sophisticated CLI tool that diagnoses and resolves GitHub repository compatibility issues with local development environments, specifically targeting ML/AI repositories with complex GPU dependencies.

## Features

- **10-second compatibility verdict** for any public GitHub repo
- **Automated environment generation** (Docker, Conda, venv)
- **GPU-aware** compatibility checking for ML workloads
- **Learning system** that improves with each analysis
- **Three-agent architecture** for comprehensive analysis

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
# Check repository compatibility
repo-doctor check https://github.com/huggingface/diffusers

# Advanced usage with validation
repo-doctor check https://github.com/CompVis/stable-diffusion \
    --strategy docker \
    --validate \
    --gpu-mode strict

# Learn from repository patterns
repo-doctor learn https://github.com/pytorch/pytorch --from-ci

# View learned patterns
repo-doctor patterns --show-failures
```

## Commands

- `check <repo_url>` - Analyze repository and generate environment
- `learn <repo_url>` - Learn patterns from repository
- `patterns` - Show learned compatibility patterns
- `cache` - Manage knowledge base cache

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
```

## Development Status

**Phase 1 Complete**: Core architecture, CLI interface, and foundational components

**Next Steps**:
- Implement repository analysis logic
- Add container validation
- Enhance learning system
- Add comprehensive testing

## Contributing

This project follows the three-agent architecture pattern. When contributing:

1. **Profile Agent**: Enhance system detection capabilities
2. **Analysis Agent**: Improve repository parsing and dependency extraction
3. **Resolution Agent**: Add new environment generation strategies

## License

MIT License
