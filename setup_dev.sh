#!/bin/bash

# Repo Doctor Development Environment Setup Script

set -e

echo "🚀 Setting up Repo Doctor development environment..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the repo-doctor root directory"
    exit 1
fi

# Check if Python 3.8+ is available
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ is required, but found Python $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip, setuptools, and wheel
echo "⬆️  Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install the project in development mode with all dependencies
echo "📥 Installing project dependencies..."
pip install -e ".[agents,dev]"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p tests
mkdir -p .repo-doctor
mkdir -p knowledge_base

# Create a basic test file if it doesn't exist
if [ ! -f "tests/__init__.py" ]; then
    echo "📝 Creating basic test structure..."
    touch tests/__init__.py
fi

# Create a basic configuration file
if [ ! -f ".repo-doctor/config.yaml" ]; then
    echo "⚙️  Creating default configuration..."
    mkdir -p .repo-doctor
    cat > .repo-doctor/config.yaml << EOF
defaults:
  strategy: auto  # docker|conda|venv|auto
  validation: true
  gpu_mode: flexible  # strict|flexible|cpu_fallback

knowledge_base:
  location: ~/.repo-doctor/kb/
  
integrations:
  github_token: \${GITHUB_TOKEN}  # Optional for private repos
  llm:
    enabled: false
    base_url: http://localhost:1234/v1
    model: qwen/qwen3-4b-thinking-2507
    timeout: 30
    max_tokens: 512
    temperature: 0.1
EOF
fi

# Test the installation
echo "🧪 Testing installation..."
if python -c "import repo_doctor; print('✅ Repo Doctor imported successfully')"; then
    echo "✅ Installation test passed"
else
    echo "❌ Installation test failed"
    exit 1
fi

# Test CLI availability
echo "🔧 Testing CLI availability..."
if repo-doctor --help > /dev/null 2>&1; then
    echo "✅ CLI test passed"
else
    echo "❌ CLI test failed"
    exit 1
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the virtual environment: source venv/bin/activate"
echo "   2. Run tests: pytest"
echo "   3. Format code: black . && isort ."
echo "   4. Lint code: flake8 ."
echo "   5. Type check: mypy ."
echo "   6. Test repo-doctor: repo-doctor check https://github.com/huggingface/transformers"
echo ""
echo "🔧 Development commands:"
echo "   - Format code: black . && isort ."
echo "   - Lint code: flake8 ."
echo "   - Type check: mypy ."
echo "   - Run tests: pytest"
echo "   - Run tests with coverage: pytest --cov=repo_doctor"
echo "   - Test CLI: repo-doctor --help"
echo ""
echo "📚 Configuration:"
echo "   - Config file: ~/.repo-doctor/config.yaml"
echo "   - Knowledge base: ~/.repo-doctor/kb/"
echo ""
echo "Happy coding! 🚀"
