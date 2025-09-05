"""LLM integration utilities for enhanced analysis."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


class LLMClient:
    """Client for local LLM integration (LM Studio, Ollama, etc.)."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3-4b-thinking-2507",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model
        self.timeout = timeout
        self.available = False
        self._availability_checked = False

    async def _check_availability(self) -> bool:
        """Check if LLM service is available."""
        if self._availability_checked:
            return self.available
            
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
        """Extract actual response from thinking tags for qwen/qwen3-4b-thinking-2507."""
        if not content:
            return content

        import re

        # For qwen thinking models, find JSON objects in the content
        # Look for complete JSON objects (handling nested braces)
        json_pattern = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
        json_matches = list(re.finditer(json_pattern, content, re.DOTALL))

        if json_matches:
            # Return the last (most complete) JSON match
            return json_matches[-1].group().strip()

        # If no JSON found but has thinking tags, try to find content after </think>
        if "</think>" in content:
            think_end_match = re.search(r"</think>\s*(.*)", content, re.DOTALL)
            if think_end_match:
                return think_end_match.group(1).strip()

        # If has unclosed <think> tag, remove everything from <think> onwards
        if "<think>" in content:
            cleaned_content = re.sub(r"<think>.*", "", content, flags=re.DOTALL)
            return cleaned_content.strip()

        # No thinking tags, return as-is
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
You are a Python environment compatibility expert. Analyze this repository data and suggest solutions:

Repository: {analysis_data.get('repository', {}).get('name', 'Unknown')}
Dependencies: {[dep['name'] for dep in analysis_data.get('dependencies', [])]}
Python Version: {analysis_data.get('python_version_required', 'Not specified')}
GPU Required: {analysis_data.get('min_gpu_memory_gb', 0) > 0}
Issues: {[issue['message'] for issue in analysis_data.get('compatibility_issues', [])]}

IMPORTANT: Respond ONLY with a valid JSON object. Do not include any explanatory text before or after the JSON.

Required JSON format:
{{"strategy": "docker|conda|venv", "reasoning": "explanation", "special_instructions": ["step1", "step2"], "alternatives": ["alt1", "alt2"]}}
"""

        response = await self.llm.generate_completion(prompt, max_tokens=800)
        if response:
            try:
                # Extract JSON from response
                import re

                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
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
Analyze this README content for Python environment requirements:

{readme_content[:2000]}

IMPORTANT: Respond ONLY with a valid JSON object. Do not include any explanatory text, thinking, or comments before or after the JSON.

Required JSON format:
{{"python_versions": ["3.9", "3.10"], "system_requirements": ["cuda", "nvidia-driver"], "gpu_requirements": "CUDA 11.8+ required", "installation_complexity": "moderate", "special_notes": ["note1", "note2"]}}

Extract:
- python_versions: list of supported Python versions (e.g., ["3.9", "3.10"])
- system_requirements: list of system-level dependencies
- gpu_requirements: GPU/CUDA requirements description
- installation_complexity: "simple", "moderate", or "complex"
- special_notes: important setup considerations
"""

        response = await self.llm.generate_completion(prompt, max_tokens=600)
        if response:
            try:
                import re

                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
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
A Docker container validation failed for this Python repository. Diagnose the issue and suggest a fix:

Repository: {analysis_data.get('repository', {}).get('name', 'Unknown')}
Dependencies: {', '.join([dep['name'] for dep in analysis_data.get('dependencies', [])[:5]])}
Error Logs:
{logs_text}

IMPORTANT: Respond with ONLY a concise diagnosis and specific fix suggestion in 2-3 sentences. Do not include any thinking process or extra commentary.
"""

        response = await self.llm.generate_completion(prompt, max_tokens=300)
        return response


class LLMFactory:
    """Factory for creating LLM clients from configuration."""

    @staticmethod
    def create_client(config) -> Optional[LLMClient]:
        """Create LLM client from configuration."""
        if not config.integrations.llm.enabled:
            return None

        return LLMClient(
            base_url=config.integrations.llm.base_url,
            api_key=config.integrations.llm.api_key,
            model=config.integrations.llm.model,
            timeout=config.integrations.llm.timeout,
        )

    @staticmethod
    def create_analyzer(config) -> Optional[LLMAnalyzer]:
        """Create LLM analyzer from configuration."""
        client = LLMFactory.create_client(config)
        if client is None:
            return None
        return LLMAnalyzer(client)
