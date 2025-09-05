# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup & Installation
```bash
# Set up complete development environment (recommended)
./setup_dev.sh

# Manual install for development
pip install -e ".[agents,dev]"
```

### Code Quality
```bash
# Format code (MUST run before committing)
make format

# Lint code
make lint

# Type check
make type-check

# Run all checks
make check-all
```

### Testing
```bash
# Run all tests
make test

# Run tests with coverage report
make test-cov

# Run only fast tests
make test-fast
```

## Architecture Overview

### Three-Agent System
The core functionality is built around three specialized agents that work together:

1. **ProfileAgent** (`agents/profile.py`): Captures system capabilities - hardware (CPU, GPU, memory), software stack (Python, CUDA, Docker), and compute scores
2. **AnalysisAgent** (`agents/analysis.py`): Analyzes GitHub repositories - extracts dependencies, identifies ML frameworks, detects GPU requirements
3. **ResolutionAgent** (`agents/resolution.py`): Generates working solutions using multiple strategies (Docker, Conda, venv)

### Strategy Pattern for Environment Generation
- **Base Strategy** (`strategies/base.py`): Abstract base class defining the interface
- **Docker Strategy** (`strategies/docker.py`): Generates Dockerfiles with GPU support
- **Conda Strategy** (`strategies/conda.py`): Creates conda environment.yml files
- **Venv Strategy** (`strategies/venv.py`): Generates virtual environment setup scripts

### Data Models (Pydantic)
- **SystemProfile** (`models/system.py`): Hardware, software, GPU capabilities
- **Analysis** (`models/analysis.py`): Repository analysis results, dependencies, compatibility
- **Resolution** (`models/resolution.py`): Generated solutions and validation results

### Advanced Conflict Detection (`conflict_detection/`)
- **ML Package Detector** (`detector.py`): Specialized conflict detection for ML/AI packages
- **CUDA Matrix** (`cuda_matrix.py`): Comprehensive CUDA compatibility for PyTorch, TensorFlow, JAX
- **Pip Parser** (`pip_parser.py`): Parse pip installation errors and suggest resolutions

### Key Utilities
- **LLM Integration** (`utils/llm.py`): Optional LLM assistance for complex dependency resolution
- **GitHub Client** (`utils/github.py`): Repository fetching and analysis
- **Parser Utilities** (`utils/parsers.py`): Requirements.txt, pyproject.toml, environment.yml parsing
- **Config Management** (`utils/config.py`): User configuration handling

## Configuration

User configuration is stored in `~/.repo-doctor/config.yaml` with defaults:
- `strategy`: auto (docker|conda|venv|auto)
- `validation`: true
- `gpu_mode`: flexible (strict|flexible|cpu_fallback)
- `llm.enabled`: false (can enable for enhanced analysis)

## CLI Entry Point

The main CLI commands:

### Basic Usage
- `repo-doctor check <repo_url>` - Check repository with smart defaults and presets
- `repo-doctor check <repo_url> --preset production` - Use preset configuration
- `repo-doctor check-advanced <repo_url>` - Access all configuration options
- `repo-doctor presets` - Show available preset configurations

### Learning System
- `repo-doctor learning-dashboard` - Show learning system performance metrics
- `repo-doctor learn <repo_url>` - Learn patterns from repository analysis

### Configuration Options
- `--preset`: Use predefined configuration (ml-research, production, development, quick, learning)
- `--strategy`: Environment generation strategy (auto, docker, conda, venv)
- `--validate`: Enable solution validation  
- `--gpu-mode`: GPU compatibility strictness (strict, flexible, cpu_fallback)
- `--enable-llm`: Enable LLM assistance
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file`: Log to structured JSON file

## Important Patterns

1. **Always use async/await** for I/O operations (GitHub API, Docker operations)
2. **Rich console output** for user feedback - use Progress spinners and Tables
3. **Pydantic models** for all data structures - ensures validation and serialization
4. **Strategy pattern** for extensible environment generation approaches
5. **Knowledge base** (`knowledge/` module) learns from each analysis to improve future results
6. **Conflict detection** - Use `MLPackageConflictDetector` for ML package compatibility checking
7. **CUDA compatibility** - Check ML framework CUDA requirements with `CUDACompatibilityMatrix`
8. **Severity-based prioritization** - Always prioritize CRITICAL conflicts over WARNING/INFO