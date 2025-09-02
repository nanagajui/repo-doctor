"""LLM integration utilities for enhanced analysis."""

import aiohttp
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import os


class LLMClient:
    """Client for local LLM integration (LM Studio, Ollama, etc.)."""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv('LLM_API_KEY')
        self.available = False
        self._check_availability()
    
    async def _check_availability(self) -> bool:
        """Check if LLM service is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/models", timeout=5) as response:
                    self.available = response.status == 200
                    return self.available
        except Exception:
            self.available = False
            return False
    
    async def generate_completion(self, prompt: str, max_tokens: int = 512, 
                                temperature: float = 0.1) -> Optional[str]:
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
                "model": "local-model",  # LM Studio uses this for loaded model
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    return None
        
        except Exception as e:
            print(f"LLM request failed: {e}")
            return None


class LLMAnalyzer:
    """LLM-powered analysis for complex cases."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    async def analyze_complex_compatibility(self, analysis_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze complex compatibility issues using LLM."""
        if not self.llm.available:
            return None
        
        prompt = f"""
You are a Python environment compatibility expert. Analyze this repository data and suggest solutions:

Repository: {analysis_data.get('repository', {}).get('name', 'Unknown')}
Dependencies: {[dep['name'] for dep in analysis_data.get('dependencies', [])]}
Python Version: {analysis_data.get('python_version_required', 'Not specified')}
GPU Required: {analysis_data.get('min_gpu_memory_gb', 0) > 0}
Issues: {[issue['message'] for issue in analysis_data.get('compatibility_issues', [])]}

Provide a JSON response with:
1. "strategy": recommended strategy (docker/conda/venv)
2. "reasoning": why this strategy is best
3. "special_instructions": any special setup steps needed
4. "alternatives": other viable options

Response format: {{"strategy": "...", "reasoning": "...", "special_instructions": [...], "alternatives": [...]}}
"""
        
        response = await self.llm.generate_completion(prompt, max_tokens=800)
        if response:
            try:
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    async def enhance_documentation_analysis(self, readme_content: str) -> Optional[Dict[str, Any]]:
        """Extract nuanced requirements from documentation using LLM."""
        if not self.llm.available or not readme_content:
            return None
        
        prompt = f"""
Analyze this README content for Python environment requirements:

{readme_content[:2000]}  # Truncate for token limits

Extract and return JSON with:
1. "python_versions": list of supported Python versions
2. "system_requirements": list of system-level dependencies
3. "gpu_requirements": GPU/CUDA requirements if any
4. "installation_complexity": "simple"/"moderate"/"complex"
5. "special_notes": any important setup considerations

Response format: {{"python_versions": [...], "system_requirements": [...], "gpu_requirements": "...", "installation_complexity": "...", "special_notes": [...]}}
"""
        
        response = await self.llm.generate_completion(prompt, max_tokens=600)
        if response:
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    async def diagnose_validation_failure(self, error_logs: List[str], 
                                        analysis_data: Dict[str, Any]) -> Optional[str]:
        """Diagnose validation failures and suggest fixes."""
        if not self.llm.available:
            return None
        
        logs_text = "\n".join(error_logs[-10:])  # Last 10 log lines
        
        prompt = f"""
A Docker container validation failed for this Python repository. Diagnose the issue and suggest a fix:

Repository: {analysis_data.get('repository', {}).get('name', 'Unknown')}
Dependencies: {[dep['name'] for dep in analysis_data.get('dependencies', [])[:5]}
Error Logs:
{logs_text}

Provide a concise diagnosis and specific fix suggestion in 2-3 sentences.
"""
        
        response = await self.llm.generate_completion(prompt, max_tokens=300)
        return response


class LLMConfig:
    """Configuration management for LLM integration."""
    
    @staticmethod
    def get_config() -> Dict[str, Any]:
        """Get LLM configuration from environment or defaults."""
        return {
            "enabled": os.getenv('REPO_DOCTOR_LLM_ENABLED', 'false').lower() == 'true',
            "base_url": os.getenv('REPO_DOCTOR_LLM_URL', 'http://localhost:1234/v1'),
            "api_key": os.getenv('REPO_DOCTOR_LLM_API_KEY'),
            "timeout": int(os.getenv('REPO_DOCTOR_LLM_TIMEOUT', '30')),
            "max_tokens": int(os.getenv('REPO_DOCTOR_LLM_MAX_TOKENS', '512'))
        }
    
    @staticmethod
    def is_enabled() -> bool:
        """Check if LLM integration is enabled."""
        return LLMConfig.get_config()["enabled"]
