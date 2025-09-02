# LLM Integration Guide

Repo Doctor now includes advanced LLM integration for enhanced analysis and error diagnosis using the qwen/qwen3-4b-thinking-2507 model.

## Features

### 🤖 LLM-Powered Documentation Analysis
- **Nuanced requirement extraction** from README files and documentation
- **Python version detection** from natural language descriptions
- **GPU requirement inference** beyond simple keyword matching
- **System dependency extraction** with context understanding

### 🔧 LLM Fallback Resolution
- **Complex compatibility case handling** when standard strategies fail
- **Strategy recommendations** for difficult repositories
- **Special instruction generation** for unique setup requirements
- **Alternative approach suggestions** when primary solutions don't work

### 🩺 LLM-Based Error Diagnosis
- **Validation failure analysis** with specific fix suggestions
- **Container build error interpretation** with actionable advice
- **Enhanced error messages** with AI-generated insights
- **Root cause identification** for complex dependency conflicts

## Configuration

### 1. Configuration File
Create `~/.repo-doctor/config.yaml`:

```yaml
integrations:
  llm:
    enabled: true
    base_url: http://localhost:1234/v1
    model: qwen/qwen3-4b-thinking-2507
    timeout: 30
    max_tokens: 512
    temperature: 0.1
```

### 2. Environment Variables
```bash
export REPO_DOCTOR_LLM_ENABLED=true
export REPO_DOCTOR_LLM_URL=http://localhost:1234/v1
export REPO_DOCTOR_LLM_API_KEY=your_api_key  # Optional
```

### 3. CLI Options
```bash
# Enable LLM for analysis
repo-doctor check https://github.com/user/repo --enable-llm

# Use custom LLM server
repo-doctor check https://github.com/user/repo --enable-llm --llm-url http://localhost:8080/v1

# Use different model
repo-doctor check https://github.com/user/repo --enable-llm --llm-model custom-model
```

## Setup Instructions

### 1. Install LLM Server
You need a compatible LLM server running the qwen/qwen3-4b-thinking-2507 model. Options include:

- **LM Studio**: Download and run qwen/qwen3-4b-thinking-2507
- **Ollama**: `ollama run qwen:3-4b-thinking`
- **vLLM**: Deploy with OpenAI-compatible API
- **Text Generation WebUI**: Load the model with API enabled

### 2. Verify Server
Test your LLM server is working:
```bash
curl http://localhost:1234/v1/models
```

### 3. Test Integration
Run the integration test:
```bash
python test_llm_integration.py
```

## Usage Examples

### Basic Usage with LLM
```bash
# Analyze with LLM assistance
repo-doctor check https://github.com/huggingface/transformers --enable-llm

# Output includes LLM insights:
# 🤖 LLM assistance enabled: qwen/qwen3-4b-thinking-2507
# 
# ## LLM Analysis Insights
# **Reasoning:** This repository requires GPU acceleration for optimal performance...
# **Special Setup Instructions:**
# - Ensure CUDA 11.8 compatibility for PyTorch
# - Consider using Docker for consistent environment
```

### Complex Repository Analysis
For repositories with unclear documentation or complex dependencies:
```bash
repo-doctor check https://github.com/complex/ml-project --enable-llm --validate
```

The LLM will:
1. **Analyze documentation** to extract hidden requirements
2. **Recommend strategies** when standard approaches fail
3. **Diagnose validation errors** with specific fixes
4. **Suggest alternatives** if the primary solution doesn't work

### Configuration Override
```bash
# Use different LLM server temporarily
repo-doctor check https://github.com/user/repo \
  --enable-llm \
  --llm-url http://192.168.1.100:8080/v1 \
  --llm-model llama2-7b-chat
```

## LLM Integration Points

### 1. Analysis Agent Enhancement
- **Documentation parsing** with natural language understanding
- **Requirement extraction** beyond regex patterns
- **Context-aware dependency detection**

### 2. Resolution Agent Fallback
- **Strategy selection** for edge cases
- **Custom instruction generation**
- **Alternative solution recommendations**

### 3. Validation Error Diagnosis
- **Build failure analysis** with specific fixes
- **Dependency conflict resolution**
- **Environment setup troubleshooting**

## Performance Impact

- **Minimal overhead** when LLM is disabled
- **Async processing** doesn't block main analysis
- **Fallback gracefully** if LLM server is unavailable
- **Caching** of LLM responses for similar repositories

## Troubleshooting

### LLM Server Not Available
```
❌ LLM Server Available: False
```
**Solution**: Ensure your LLM server is running on the configured URL.

### Model Not Found
```
LLM request failed: Model not found
```
**Solution**: Verify the model name matches what's loaded in your LLM server.

### Timeout Issues
```
LLM request failed: Timeout
```
**Solution**: Increase timeout in configuration or use a faster model.

### API Key Issues
```
LLM request failed: Unauthorized
```
**Solution**: Set the correct API key in configuration or environment variables.

## Advanced Configuration

### Custom Prompts
The LLM integration uses carefully crafted prompts for different tasks. You can extend the system by modifying the prompts in `repo_doctor/utils/llm.py`.

### Multiple Models
Configure different models for different tasks:
```yaml
integrations:
  llm:
    enabled: true
    base_url: http://localhost:1234/v1
    model: qwen/qwen3-4b-thinking-2507  # Default model
    # Add custom model configurations as needed
```

### Rate Limiting
The system includes built-in rate limiting and error handling to prevent overwhelming the LLM server.

## Security Considerations

- **Local deployment recommended** for sensitive repositories
- **API keys** should be stored securely in environment variables
- **Network access** only to configured LLM endpoints
- **No data persistence** in LLM integration layer
