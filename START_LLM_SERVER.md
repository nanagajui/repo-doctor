# Starting LLM Server for Repo Doctor

## Quick Start Options

### Option 1: LM Studio (Recommended)
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Search for and download `qwen/qwen3-4b-thinking-2507`
3. Load the model in LM Studio
4. Start the local server (default: http://localhost:1234/v1)
5. Test with: `curl http://localhost:1234/v1/models`

### Option 2: Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull and run the model (in another terminal)
ollama pull qwen:3-4b-thinking
ollama run qwen:3-4b-thinking
```

### Option 3: vLLM (Advanced)
```bash
# Install vLLM
pip install vllm

# Start server with qwen model
python -m vllm.entrypoints.openai.api_server \
  --model qwen/qwen3-4b-thinking-2507 \
  --port 1234
```

## Verify Server is Running

```bash
# Test connection
curl http://localhost:1234/v1/models

# Expected response should include model information
```

## Test with Repo Doctor

Once the server is running:

```bash
# Test LLM integration
cd /path/to/repo-doctor
python test_llm_integration.py

# Use with repo analysis
repo-doctor check https://github.com/user/repo --enable-llm
```

## Configuration

The default configuration expects:
- **URL**: http://localhost:1234/v1
- **Model**: qwen/qwen3-4b-thinking-2507
- **Port**: 1234

You can override these with CLI options:
```bash
repo-doctor check https://github.com/user/repo \
  --enable-llm \
  --llm-url http://localhost:8080/v1 \
  --llm-model custom-model
```
