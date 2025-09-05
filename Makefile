# Repo Doctor Development Makefile

.PHONY: help install install-dev test test-cov lint format type-check clean setup-dev

# Default target
help:
	@echo "Repo Doctor Development Commands"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  setup-dev     Set up development environment"
	@echo "  install       Install package in development mode"
	@echo "  install-dev   Install with development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  format        Format code with black and isort"
	@echo "  lint          Lint code with flake8"
	@echo "  type-check    Type check with mypy"
	@echo "  check-all     Run all code quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  test-fast     Run fast tests only"
	@echo ""
	@echo "Utilities:"
	@echo "  clean         Clean up temporary files"
	@echo "  clean-outputs Clean up generated outputs"
	@echo "  outputs-info  Show output directory information"
	@echo "  docs          Generate documentation"
	@echo ""

# Setup development environment
setup-dev:
	@echo "🚀 Setting up development environment..."
	./setup_dev.sh

# Install package in development mode
install:
	@echo "📦 Installing package in development mode..."
	pip install -e .

# Install with development dependencies
install-dev:
	@echo "📦 Installing with development dependencies..."
	pip install -e ".[agents,dev]"

# Format code
format:
	@echo "🎨 Formatting code..."
	black .
	isort .

# Lint code
lint:
	@echo "🔍 Linting code..."
	flake8 .

# Type check
type-check:
	@echo "🔍 Type checking..."
	mypy .

# Run all code quality checks
check-all: format lint type-check
	@echo "✅ All code quality checks passed!"

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest

# Run tests with coverage
test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest --cov=repo_doctor --cov-report=term-missing --cov-report=html

# Run fast tests only
test-fast:
	@echo "🧪 Running fast tests..."
	pytest -m "not slow"

# Clean up temporary files
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

# Clean up generated outputs
clean-outputs:
	@echo "🗑️ Cleaning up generated outputs..."
	rm -rf outputs/*/
	@echo "✅ Outputs cleaned!"

# Show output directory info
outputs-info:
	@echo "📁 Output Directory Information"
	@echo "=============================="
	@echo "Location: outputs/"
	@echo "Default behavior: Files saved to outputs/{owner}-{repo}/"
	@echo "Custom output: Use --output flag"
	@echo ""
	@echo "Generated files:"
	@echo "  • Dockerfile"
	@echo "  • docker-compose.yml"
	@echo "  • setup.sh"
	@echo "  • SETUP_INSTRUCTIONS.md"
	@echo ""
	@echo "Cleanup: make clean-outputs"

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "Documentation generation not yet implemented"

# Development workflow
dev: install-dev format lint type-check test
	@echo "✅ Development workflow complete!"

# CI workflow
ci: install-dev lint type-check test-cov
	@echo "✅ CI workflow complete!"
