# Token Setup Guide

This guide helps you configure API tokens for Repo Doctor to improve performance and access to repositories.

## Why Tokens Matter

API tokens provide several benefits:

- **Higher Rate Limits**: GitHub API increases from 60/hour to 5000/hour with authentication
- **Private Repository Access**: Access your private repositories for analysis
- **Better Reliability**: Reduced rate limiting and "Backoff" messages
- **Enhanced Features**: Full access to repository metadata and content

## GitHub Token Setup

### 1. Create a GitHub Personal Access Token

1. Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Give it a descriptive name: `repo-doctor-access`
4. Select scopes:
   - `repo` - Full repository access (for private repos)
   - `read:user` - Read user profile (for user info)
   - `read:org` - Read organization info (for organization repos)
5. Set expiration (recommend 90 days for security)
6. Click "Generate token"
7. **Copy the token immediately** - you won't see it again!

### 2. Set the Environment Variable

#### Linux/macOS (Bash/Zsh)
```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.profile
export GITHUB_TOKEN=ghp_your_token_here

# Reload your shell configuration
source ~/.bashrc  # or ~/.zshrc
```

#### Windows (PowerShell)
```powershell
# Set for current session
$env:GITHUB_TOKEN="ghp_your_token_here"

# Set permanently (requires restart)
[Environment]::SetEnvironmentVariable("GITHUB_TOKEN", "ghp_your_token_here", "User")
```

#### Windows (Command Prompt)
```cmd
# Set for current session
set GITHUB_TOKEN=ghp_your_token_here

# Set permanently
setx GITHUB_TOKEN "ghp_your_token_here"
```

### 3. Verify Token Setup

Check that your token is working:

```bash
# Check if token is set
repo-doctor tokens

# Check system health (includes rate limit info)
repo-doctor health

# Test with a repository analysis
repo-doctor check https://github.com/huggingface/transformers
```

## Hugging Face Token Setup (Optional)

For enhanced ML model analysis and downloads:

### 1. Create a Hugging Face Token

1. Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name: `repo-doctor-access`
4. Select role: `read` (sufficient for most use cases)
5. Click "Generate a token"
6. Copy the token

### 2. Set the Environment Variable

```bash
# Linux/macOS
export HF_TOKEN=hf_your_token_here

# Windows PowerShell
$env:HF_TOKEN="hf_your_token_here"

# Windows Command Prompt
set HF_TOKEN=hf_your_token_here
```

## Configuration File

Repo Doctor can also load tokens from configuration. Create `~/.repo-doctor/config.yaml`:

```yaml
defaults:
  strategy: auto
  validation: true
  gpu_mode: flexible

integrations:
  github_token: ${GITHUB_TOKEN}  # Reads from environment
  # Alternatively, you can set directly (not recommended for security)
  # github_token: ghp_your_token_here
  
  # For LLM features
  llm:
    enabled: false
    base_url: http://localhost:1234/v1
    model: qwen/qwen3-4b-thinking-2507
```

## Token Security Best Practices

### 1. Token Permissions
- Use minimal required scopes
- For public repos only: no scopes needed
- For private repos: `repo` scope
- Avoid `admin:org` or other high-privilege scopes

### 2. Token Management
- Set reasonable expiration dates (30-90 days)
- Regenerate tokens periodically
- Revoke unused tokens immediately
- Never commit tokens to version control

### 3. Environment Variables
- Use environment variables instead of hardcoded tokens
- Add tokens to your shell profile for persistence
- Consider using a secrets manager for production use

## Troubleshooting

### Common Issues

#### "No GitHub token found" warning
```bash
# Check if token is set
echo $GITHUB_TOKEN

# If empty, set the token
export GITHUB_TOKEN=your_token_here
```

#### "Rate limit exceeded" or "Backoff" messages
- Usually means no token is set or token is invalid
- Check token validity: `repo-doctor tokens`
- Verify token hasn't expired on GitHub

#### Token authentication failed
- Check token is copied correctly (no extra spaces)
- Verify token hasn't been revoked
- Ensure token has required scopes

#### Private repository access denied
- Token needs `repo` scope for private repositories
- Verify you have access to the repository
- Organization repositories may need additional permissions

### Testing Your Setup

1. **Basic connectivity**:
   ```bash
   repo-doctor health
   ```

2. **Rate limit check**:
   ```bash
   repo-doctor tokens --github
   ```

3. **Full analysis test**:
   ```bash
   repo-doctor check https://github.com/huggingface/transformers --output test-output
   ```

### Getting Help

If you encounter issues:

1. Check system health: `repo-doctor health`
2. Verify tokens: `repo-doctor tokens`
3. Check logs for specific error messages
4. Visit [GitHub token troubleshooting](https://docs.github.com/en/authentication/troubleshooting-ssh/error-permission-denied-publickey)

## Integration with CI/CD

For automated environments:

### GitHub Actions
```yaml
env:
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}  # Auto-provided
  HF_TOKEN: ${{ secrets.HF_TOKEN }}          # Add as repository secret
```

### Docker
```bash
docker run -e GITHUB_TOKEN=$GITHUB_TOKEN repo-doctor check <repo-url>
```

### Docker Compose
```yaml
services:
  repo-doctor:
    environment:
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - HF_TOKEN=${HF_TOKEN}
```

## Advanced Configuration

### Multiple Tokens
For organizations with multiple GitHub accounts:

```bash
# Different tokens for different contexts
export GITHUB_TOKEN_PERSONAL=ghp_personal_token
export GITHUB_TOKEN_WORK=ghp_work_token

# Use specific token
GITHUB_TOKEN=$GITHUB_TOKEN_WORK repo-doctor check <repo-url>
```

### Token Rotation
Create a script for regular token rotation:

```bash
#!/bin/bash
# rotate-tokens.sh
echo "Current GitHub rate limit:"
repo-doctor tokens --github

echo "Remember to:"
echo "1. Generate new token on GitHub"
echo "2. Update GITHUB_TOKEN environment variable"
echo "3. Test with: repo-doctor health"
```

This guide should help you set up tokens properly and avoid the rate limiting issues that cause "Backoff" messages.