"""LLM integration utilities for enhanced analysis."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from .llm_discovery import smart_llm_config


class LLMClient:
    """Client for local LLM integration (LM Studio, Ollama, etc.) with smart discovery."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
        use_smart_discovery: bool = True,
    ):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or self._get_default_model()
        self.timeout = timeout
        self.use_smart_discovery = use_smart_discovery
        self.available = False
        self._availability_checked = False
        self._smart_config = None

    def _get_default_model(self) -> str:
        """Get default model from configuration."""
        try:
            from .config import Config
            config = Config.load()
            return config.integrations.llm.model
        except Exception:
            # Fallback if config loading fails
            return "openai/gpt-oss-20b"

    async def _check_availability(self) -> bool:
        """Check if LLM service is available with smart discovery."""
        if self._availability_checked:
            return self.available
        
        # Use smart discovery if enabled and no specific URL provided
        if self.use_smart_discovery and not self.base_url:
            try:
                self._smart_config = await smart_llm_config.get_config()
                if self._smart_config.get("enabled", False):
                    self.base_url = self._smart_config["base_url"]
                    self.model = self._smart_config.get("model", self.model)
                    self.timeout = self._smart_config.get("timeout", self.timeout)
            except Exception:
                pass
        
        # Fallback to default if smart discovery failed
        if not self.base_url:
            self.base_url = "http://localhost:1234/v1"
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models", timeout=5
                ) as response:
                    self.available = response.status == 200
                    self._availability_checked = True
                    return self.available
        except Exception:
            self.available = False
            self._availability_checked = True
            return False

    async def generate_completion(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.1
    ) -> Optional[str]:
        """Generate completion using local LLM."""
        if not self.available:
            await self._check_availability()
            if not self.available:
                return None

        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            return self._extract_response_from_thinking(content)
                    # Mark as unavailable if we get an error response
                    self.available = False
                    return None

        except Exception as e:
            # Mark as unavailable on any exception
            self.available = False
            return None

    def _extract_response_from_thinking(self, content: str) -> str:
        """Extract actual response from thinking tags for openai/gpt-oss-20b and other models."""
        if not content:
            return content

        import re
        import json

        # First, try to extract content after thinking tags
        # Handle various thinking tag formats: <think>, <thinking>, etc.
        thinking_patterns = [
            r"</think>\s*(.*?)(?=<think>|$)",  # Content after </think>
            r"</thinking>\s*(.*?)(?=<thinking>|$)",  # Content after </thinking>
            r"</?think(?:ing)?>.*?</think(?:ing)?>\s*(.*)",  # Content after complete thinking blocks
        ]
        
        for pattern in thinking_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                # Get the last match (most complete)
                extracted = matches[-1].strip()
                if extracted:
                    content = extracted
                    break

        # If content still has thinking tags, remove them
        if re.search(r"<think(?:ing)?>", content, re.IGNORECASE):
            # Remove everything from opening think tag to end
            content = re.sub(r"<think(?:ing)?>.*", "", content, flags=re.DOTALL | re.IGNORECASE)
            content = content.strip()

        # Try to extract JSON objects from the cleaned content
        json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
        json_matches = list(re.finditer(json_pattern, content, re.DOTALL))
        
        if json_matches:
            # Validate and return the most complete JSON
            for match in reversed(json_matches):  # Start with the last match
                try:
                    json_str = match.group().strip()
                    json.loads(json_str)  # Validate JSON
                    return json_str
                except json.JSONDecodeError:
                    continue
        
        # If no valid JSON found, return cleaned content
        return content.strip()


class LLMAnalyzer:
    """LLM-powered analysis for complex cases."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def analyze_complex_compatibility(
        self, analysis_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyze complex compatibility issues using LLM."""
        if not self.llm.available:
            await self.llm._check_availability()
            if not self.llm.available:
                return None

        prompt = f"""
<thinking>
I need to analyze this repository's compatibility issues and recommend the best strategy.

Repository: {analysis_data.get('repository', {}).get('name', 'Unknown')}
Dependencies: {[dep['name'] for dep in analysis_data.get('dependencies', [])]}
Python Version: {analysis_data.get('python_version_required', 'Not specified')}
GPU Required: {analysis_data.get('min_gpu_memory_gb', 0) > 0}
Issues: {[issue['message'] for issue in analysis_data.get('compatibility_issues', [])]}

Let me think through the best strategy based on:
1. Complexity of dependencies
2. GPU requirements
3. Python version constraints
4. Compatibility issues present
</thinking>

You are a Python environment compatibility expert. Based on the repository analysis, provide a strategic recommendation.

**Repository Analysis:**
- Name: {analysis_data.get('repository', {}).get('name', 'Unknown')}
- Dependencies: {[dep['name'] for dep in analysis_data.get('dependencies', [])]}
- Python Version: {analysis_data.get('python_version_required', 'Not specified')}
- GPU Required: {analysis_data.get('min_gpu_memory_gb', 0) > 0}
- Issues: {[issue['message'] for issue in analysis_data.get('compatibility_issues', [])]}

**CRITICAL: Your response must be ONLY a valid JSON object with no additional text, explanations, or formatting.**

**Required JSON Schema:**
{{
  "strategy": "docker|conda|venv|micromamba",
  "reasoning": "Clear explanation of why this strategy is optimal",
  "special_instructions": ["specific setup step 1", "specific setup step 2"],
  "alternatives": ["alternative strategy 1", "alternative strategy 2"]
}}
"""

        response = await self.llm.generate_completion(prompt, max_tokens=800)
        if response:
            try:
                # Extract JSON from response using improved parsing
                import re
                import json
                
                # Try multiple JSON extraction methods
                json_candidates = []
                
                # Method 1: Look for complete JSON objects
                json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
                json_matches = re.findall(json_pattern, response, re.DOTALL)
                json_candidates.extend(json_matches)
                
                # Method 2: Extract content between specific markers if present
                if '```json' in response:
                    json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
                    if json_block_match:
                        json_candidates.append(json_block_match.group(1))
                
                # Method 3: Look for JSON after common prefixes
                prefixes = ['JSON:', 'Response:', 'Result:']
                for prefix in prefixes:
                    if prefix in response:
                        after_prefix = response.split(prefix, 1)[1].strip()
                        json_match = re.search(r"\{.*\}", after_prefix, re.DOTALL)
                        if json_match:
                            json_candidates.append(json_match.group())
                
                # Validate and return the first valid JSON
                for candidate in json_candidates:
                    try:
                        parsed = json.loads(candidate.strip())
                        return parsed
                    except json.JSONDecodeError:
                        continue
            except json.JSONDecodeError:
                pass

        return None

    async def enhance_documentation_analysis(
        self, readme_content: str
    ) -> Optional[Dict[str, Any]]:
        """Extract nuanced requirements from documentation using LLM."""
        if not readme_content:
            return None
            
        if not self.llm.available:
            await self.llm._check_availability()
            if not self.llm.available:
                return None

        prompt = f"""
<thinking>
I need to carefully analyze this README content to extract:
1. Python version requirements
2. System-level dependencies
3. GPU/CUDA requirements
4. Installation complexity assessment
5. Special setup considerations

Let me examine the content systematically...
</thinking>

Analyze the following README content and extract Python environment requirements with high precision.

**README Content:**
```
{readme_content[:2000]}
```

**CRITICAL INSTRUCTIONS:**
1. Your response must be ONLY a valid JSON object
2. No explanatory text, thinking tags, or additional formatting
3. Extract information accurately from the provided content
4. Use "unknown" or empty arrays if information is not clearly specified

**Required JSON Schema:**
{{
  "python_versions": ["version1", "version2"],
  "system_requirements": ["requirement1", "requirement2"],
  "gpu_requirements": "description of GPU/CUDA needs",
  "installation_complexity": "simple|moderate|complex",
  "special_notes": ["important note 1", "important note 2"]
}}

**Field Definitions:**
- python_versions: Specific Python versions mentioned (e.g., ["3.9", "3.10", "3.11"])
- system_requirements: OS-level dependencies (e.g., ["cuda", "nvidia-driver", "cmake"])
- gpu_requirements: GPU/CUDA version requirements as a string
- installation_complexity: Assessment based on number of steps and dependencies
- special_notes: Critical setup warnings or important considerations
"""

        response = await self.llm.generate_completion(prompt, max_tokens=600)
        if response:
            try:
                # Extract JSON from response using improved parsing
                import re
                import json
                
                # Try multiple JSON extraction methods
                json_candidates = []
                
                # Method 1: Look for complete JSON objects
                json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
                json_matches = re.findall(json_pattern, response, re.DOTALL)
                json_candidates.extend(json_matches)
                
                # Method 2: Extract content between specific markers if present
                if '```json' in response:
                    json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
                    if json_block_match:
                        json_candidates.append(json_block_match.group(1))
                
                # Method 3: Look for JSON after common prefixes
                prefixes = ['JSON:', 'Response:', 'Result:']
                for prefix in prefixes:
                    if prefix in response:
                        after_prefix = response.split(prefix, 1)[1].strip()
                        json_match = re.search(r"\{.*\}", after_prefix, re.DOTALL)
                        if json_match:
                            json_candidates.append(json_match.group())
                
                # Validate and return the first valid JSON
                for candidate in json_candidates:
                    try:
                        parsed = json.loads(candidate.strip())
                        return parsed
                    except json.JSONDecodeError:
                        continue
            except json.JSONDecodeError:
                pass

        return None

    async def diagnose_validation_failure(
        self, error_logs: List[str], analysis_data: Dict[str, Any]
    ) -> Optional[str]:
        """Diagnose validation failures and suggest fixes."""
        if not self.llm.available:
            await self.llm._check_availability()
            if not self.llm.available:
                return None

        logs_text = "\n".join(error_logs[-10:])  # Last 10 log lines

        prompt = f"""
<thinking>
I need to analyze this Docker container validation failure:

Repository: {analysis_data.get('repository', {}).get('name', 'Unknown')}
Key Dependencies: {', '.join([dep['name'] for dep in analysis_data.get('dependencies', [])[:5]])}

Error Logs:
{logs_text}

Let me identify:
1. The root cause of the failure
2. Whether it's a dependency issue, version conflict, or system requirement
3. The most direct fix
</thinking>

Diagnose this Docker container validation failure and provide a specific fix.

**Repository:** {analysis_data.get('repository', {}).get('name', 'Unknown')}
**Key Dependencies:** {', '.join([dep['name'] for dep in analysis_data.get('dependencies', [])[:5]])}

**Error Logs:**
```
{logs_text}
```

**CRITICAL: Provide ONLY a concise diagnosis and specific fix in 2-3 clear sentences. No thinking tags or extra commentary.**

Format your response as:
DIAGNOSIS: [Root cause of the failure]
FIX: [Specific actionable solution]
"""

        response = await self.llm.generate_completion(prompt, max_tokens=300)
        return response


class LLMFactory:
    """Factory for creating LLM clients from configuration with smart discovery."""

    @staticmethod
    async def create_client(config, use_smart_discovery: bool = True) -> Optional[LLMClient]:
        """Create LLM client from configuration with smart discovery."""
        if not config.integrations.llm.enabled:
            return None

        # If smart discovery is enabled, let the client handle discovery
        if use_smart_discovery:
            return LLMClient(
                base_url=config.integrations.llm.base_url,
                api_key=config.integrations.llm.api_key,
                model=config.integrations.llm.model,
                timeout=config.integrations.llm.timeout,
                use_smart_discovery=True,
            )
        else:
            return LLMClient(
                base_url=config.integrations.llm.base_url,
                api_key=config.integrations.llm.api_key,
                model=config.integrations.llm.model,
                timeout=config.integrations.llm.timeout,
                use_smart_discovery=False,
            )

    @staticmethod
    async def create_analyzer(config, use_smart_discovery: bool = True) -> Optional[LLMAnalyzer]:
        """Create LLM analyzer from configuration with smart discovery."""
        client = await LLMFactory.create_client(config, use_smart_discovery)
        if client is None:
            return None
        return LLMAnalyzer(client)
    
    @staticmethod
    def create_client_sync(config, use_smart_discovery: bool = True) -> Optional[LLMClient]:
        """Create LLM client synchronously (for backward compatibility)."""
        if not config.integrations.llm.enabled:
            return None

        return LLMClient(
            base_url=config.integrations.llm.base_url,
            api_key=config.integrations.llm.api_key,
            model=config.integrations.llm.model,
            timeout=config.integrations.llm.timeout,
            use_smart_discovery=use_smart_discovery,
        )

    @staticmethod
    def create_analyzer_sync(config, use_smart_discovery: bool = True) -> Optional[LLMAnalyzer]:
        """Create LLM analyzer synchronously (for backward compatibility)."""
        client = LLMFactory.create_client_sync(config, use_smart_discovery)
        if client is None:
            return None
        return LLMAnalyzer(client)
