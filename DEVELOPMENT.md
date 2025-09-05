# Repo Doctor Development Guide

This guide will help you set up and work with the Repo Doctor codebase.

## Quick Start

### 1. Automated Setup
```bash
# Run the automated setup script
./setup_dev.sh
```

### 2. Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[agents,dev]"

# Create necessary directories
mkdir -p tests .repo-doctor knowledge_base
```

## Development Environment

### Virtual Environment
- **Location**: `venv/` (in project root)
- **Python Version**: 3.8+ (tested with 3.12)
- **Activation**: `source venv/bin/activate`

### Dependencies
- **Core**: All production dependencies from `pyproject.toml`
- **Development**: `black`, `isort`, `flake8`, `mypy`, `pytest`, `pytest-cov`
- **Agents**: `langchain`, `openai`, `chromadb` (optional LLM features)

## Development Commands

### Using Makefile
```bash
# Setup
make setup-dev          # Set up development environment
make install            # Install package in development mode
make install-dev        # Install with development dependencies

# Code Quality
make format             # Format code with black and isort
make lint               # Lint code with flake8
make type-check         # Type check with mypy
make check-all          # Run all code quality checks

# Testing
make test               # Run tests
make test-cov           # Run tests with coverage
make test-fast          # Run fast tests only

# Utilities
make clean              # Clean up temporary files
make dev                # Complete development workflow
make ci                 # CI workflow
```

### Direct Commands
```bash
# Code formatting
black .
isort .

# Linting
flake8 repo_doctor/

# Type checking
mypy repo_doctor/

# Testing
pytest
pytest --cov=repo_doctor --cov-report=html
```

## Project Structure

```
repo-doctor/
├── repo_doctor/           # Main package
│   ├── agents/           # Three-agent system
│   │   ├── profile.py    # System profiling
│   │   ├── analysis.py   # Repository analysis
│   │   └── resolution.py # Solution generation
│   ├── models/           # Pydantic data models
│   ├── strategies/       # Environment strategies
│   ├── knowledge/        # Learning system
│   ├── validators/       # Solution validation
│   ├── utils/           # Utilities and helpers
│   └── cli.py           # CLI interface
├── tests/               # Test suite
├── outputs/             # Generated output files
│   ├── README.md        # Output directory documentation
│   └── {repo-name}/     # Repository-specific outputs
├── venv/               # Virtual environment
├── .repo-doctor/       # Configuration and cache
├── knowledge_base/     # Knowledge base storage
├── pyproject.toml      # Project configuration
├── Makefile           # Development commands
├── setup_dev.sh       # Setup script
└── .gitignore         # Git ignore rules
```

## Configuration

### Default Configuration
Location: `~/.repo-doctor/config.yaml`

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
    timeout: 30
    max_tokens: 512
    temperature: 0.1
```

### Environment Variables
- `GITHUB_TOKEN`: GitHub API token for private repositories
- `REPO_DOCTOR_CONFIG`: Path to custom config file
- `REPO_DOCTOR_LLM_URL`: Override LLM server URL
- `REPO_DOCTOR_LLM_MODEL`: Override LLM model name

## Testing

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=repo_doctor --cov-report=html

# Fast tests only
pytest -m "not slow"

# Specific test file
pytest tests/test_agents.py

# Specific test
pytest tests/test_agents.py::test_profile_agent
```

### Test Structure
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows

### Test Markers
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests

## Code Quality

### Formatting
- **Black**: Code formatting (line length: 88)
- **isort**: Import sorting
- **Configuration**: `pyproject.toml`

### Linting
- **flake8**: Style and error checking
- **Configuration**: `.flake8`

### Type Checking
- **mypy**: Static type checking
- **Configuration**: `pyproject.toml`

### Pre-commit Hooks
```bash
# Install pre-commit (optional)
pip install pre-commit
pre-commit install
```

## Development Workflow

### 1. Feature Development
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes
# ... edit code ...

# Format and lint
make format
make lint

# Run tests
make test

# Commit changes
git add .
git commit -m "Add new feature"
```

### 2. Bug Fixes
```bash
# Create bugfix branch
git checkout -b bugfix/fix-issue

# Make changes
# ... fix code ...

# Test the fix
make test

# Commit fix
git add .
git commit -m "Fix issue description"
```

### 3. Code Review
- Ensure all tests pass
- Code is properly formatted
- No linting errors
- Type checking passes
- Documentation updated

## Architecture

### Three-Agent System
1. **Profile Agent**: System detection and profiling
2. **Analysis Agent**: Repository analysis and parsing
3. **Resolution Agent**: Solution generation and validation

### Key Components
- **Models**: Pydantic data models for type safety
- **Strategies**: Environment generation strategies (Docker, Conda, Venv)
- **Knowledge Base**: Learning system for pattern recognition
- **Validators**: Solution validation and testing
- **CLI**: Rich terminal interface with progress indicators

## Debugging

### Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Debug Mode
```bash
# Enable debug logging
export REPO_DOCTOR_DEBUG=1
repo-doctor check <repo_url>
```

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated
2. **Permission Errors**: Check Docker permissions for validation
3. **Network Issues**: Verify GitHub API access and LLM server connectivity
4. **Output Files**: Generated files are saved to `outputs/{owner}-{repo}/` by default

## Output Directory

### Structure
The `outputs/` directory contains all generated files from Repo Doctor analyses:

```
outputs/
├── README.md                    # Directory documentation
├── huggingface-transformers/     # Example: transformers repo output
│   ├── Dockerfile              # Generated Dockerfile
│   ├── docker-compose.yml      # Generated Docker Compose file
│   ├── setup.sh                # Generated setup script
│   └── SETUP_INSTRUCTIONS.md   # Generated setup instructions
└── owner-repo/                  # Other repository outputs
```

### Default Behavior
- **Without `--output`**: Files saved to `outputs/{owner}-{repo}/`
- **With `--output`**: Files saved to specified directory
- **Directory creation**: Automatically created if it doesn't exist

### File Types Generated
- **Dockerfile**: Container definition with GPU support
- **docker-compose.yml**: Multi-container orchestration
- **setup.sh**: Automated setup script
- **SETUP_INSTRUCTIONS.md**: Detailed setup guide
- **validation_results.json**: Validation results (if enabled)

### Cleanup
```bash
# Remove all generated outputs
rm -rf outputs/*/

# Remove specific repository output
rm -rf outputs/huggingface-transformers/
```

## Contributing

### Code Style
- Follow PEP 8 (enforced by flake8)
- Use type hints (enforced by mypy)
- Format with Black and isort
- Write docstrings for public functions

### Testing
- Write tests for new features
- Maintain test coverage above 80%
- Use appropriate test markers

### Documentation
- Update README.md for user-facing changes
- Update DEVELOPMENT.md for development changes
- Add docstrings for new functions/classes

## Performance

### Profiling
```bash
# Profile CLI performance
python -m cProfile -o profile.stats -m repo_doctor.cli check <repo_url>
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

### Optimization
- Use async operations where possible
- Cache expensive operations
- Optimize Docker operations
- Minimize network requests

## Troubleshooting

### Common Problems

1. **Virtual Environment Issues**
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install -e ".[agents,dev]"
   ```

2. **Docker Permission Issues**
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

3. **LLM Server Issues**
   ```bash
   # Test LLM server connectivity
   curl http://localhost:1234/v1/models
   ```

4. **GitHub API Issues**
   ```bash
   # Test GitHub API access
   curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
   ```

### Getting Help
- Check the logs for error messages
- Verify configuration settings
- Test individual components
- Check system requirements

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 2GB disk space
- Internet connection

### Recommended
- Python 3.10+
- 8GB RAM
- 10GB disk space
- GPU support (for ML repositories)
- Docker (for validation)

### Supported Platforms
- Linux (primary)
- WSL2 (Windows)
- macOS (limited GPU support)

## License

MIT License - see LICENSE file for details.
