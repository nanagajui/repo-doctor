"""GitHub API utilities with enhanced rate limiting and error handling."""

import logging
import os
import re
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from github import Github, GithubException, RateLimitExceededException
from requests.exceptions import RequestException

from .env import EnvLoader


def retry_on_rate_limit(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying GitHub API calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except RateLimitExceededException as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    # Calculate delay with jitter
                    delay = base_delay * (2 ** attempt) + (time.time() % 1)
                    reset_time = e.rate_limit.reset
                    
                    if reset_time:
                        wait_time = min(delay, reset_time.timestamp() - time.time())
                        if wait_time > 0:
                            self.logger.warning(
                                f"Rate limit exceeded. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}"
                            )
                            time.sleep(wait_time)
                    else:
                        time.sleep(delay)
                        
                except (RequestException, GithubException) as e:
                    if hasattr(e, 'status') and e.status == 403 and "rate limit" in str(e).lower():
                        last_exception = e
                        if attempt == max_retries:
                            break
                        
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"API error (likely rate limit): {e}. Retrying in {delay}s")
                        time.sleep(delay)
                    else:
                        # Non-rate-limit error, don't retry
                        raise e
                        
            # All retries exhausted
            self.logger.error(f"Max retries ({max_retries}) exceeded for GitHub API call")
            raise last_exception
            
        return wrapper
    return decorator


class GitHubHelper:
    """Enhanced GitHub API helper with rate limiting and robust error handling."""

    def __init__(self, token: Optional[str] = None):
        # Use EnvLoader to get token from .env file or environment
        self.token = token or EnvLoader.get_github_token()
        self.github = Github(self.token) if self.token else Github()
        self.logger = logging.getLogger(__name__)
        
        # Track rate limit status
        self._last_rate_limit_check = 0
        self._rate_limit_remaining = None
        self._rate_limit_reset = None
        
        # Warn about missing token
        if not self.token:
            self.logger.warning(
                "No GitHub token found. API rate limit is 60/hour instead of 5000/hour. "
                "Set GITHUB_TOKEN in environment or .env file for better performance."
            )

    def check_rate_limit(self) -> Dict[str, Any]:
        """Check current rate limit status."""
        try:
            rate_limit = self.github.get_rate_limit()
            core_limit = rate_limit.rate  # Use .rate instead of .core
            
            self._rate_limit_remaining = core_limit.remaining
            self._rate_limit_reset = core_limit.reset
            self._last_rate_limit_check = time.time()
            
            return {
                "remaining": core_limit.remaining,
                "limit": core_limit.limit,
                "reset": core_limit.reset,
                "reset_in_seconds": (core_limit.reset.timestamp() - time.time()),
                "used": core_limit.limit - core_limit.remaining,
                "percentage_used": ((core_limit.limit - core_limit.remaining) / core_limit.limit) * 100,
            }
        except Exception as e:
            self.logger.error(f"Failed to check rate limit: {e}")
            return {
                "remaining": 0,
                "limit": 60 if not self.token else 5000,
                "reset": None,
                "reset_in_seconds": 3600,
                "used": 0,
                "percentage_used": 0,
            }

    def get_rate_limit_status(self) -> str:
        """Get human-readable rate limit status."""
        status = self.check_rate_limit()
        
        if status["remaining"] < 100:
            urgency = "🔴 CRITICAL"
        elif status["remaining"] < 500:
            urgency = "🟡 WARNING"
        else:
            urgency = "🟢 OK"
            
        reset_time = ""
        if status["reset"]:
            reset_time = f" (resets at {status['reset'].strftime('%H:%M:%S')})"
            
        return (
            f"{urgency} GitHub API: {status['remaining']}/{status['limit']} remaining"
            f"{reset_time}"
        )

    def warn_if_rate_limit_low(self, threshold: int = 100):
        """Warn user if rate limit is getting low."""
        status = self.check_rate_limit()
        if status["remaining"] < threshold:
            self.logger.warning(
                f"GitHub API rate limit is low: {status['remaining']}/{status['limit']} remaining. "
                f"Resets in {status['reset_in_seconds']:.0f} seconds."
            )

    def parse_repo_url(self, url: str) -> Dict[str, str]:
        """Parse GitHub repository URL to extract owner and repo name."""
        # Handle various GitHub URL formats
        patterns = [
            r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
            r"github\.com/([^/]+)/([^/]+)/.*",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return {
                    "owner": match.group(1),
                    "name": match.group(2),
                    "full_name": f"{match.group(1)}/{match.group(2)}",
                }

        raise ValueError(f"Invalid GitHub URL format: {url}")

    @retry_on_rate_limit(max_retries=3, base_delay=2.0)
    def get_repo_info(self, owner: str, name: str) -> Optional[Dict[str, Any]]:
        """Get repository information from GitHub API."""
        try:
            repo = self.github.get_repo(f"{owner}/{name}")

            return {
                "name": repo.name,
                "full_name": repo.full_name,
                "description": repo.description,
                "language": repo.language,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "topics": repo.get_topics(),
                "default_branch": repo.default_branch,
                "created_at": repo.created_at.isoformat() if repo.created_at else None,
                "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                "size": repo.size,
                "open_issues": repo.open_issues_count,
                "has_issues": repo.has_issues,
                "has_wiki": repo.has_wiki,
                "has_pages": repo.has_pages,
                "archived": repo.archived,
                "disabled": repo.disabled,
                "private": repo.private,
            }

        except GithubException as e:
            if e.status == 404:
                return None  # Repository not found or private
            elif e.status == 403:
                self.logger.warning(f"Access denied to repository {owner}/{name} - may be private")
                return None
            raise e

    @retry_on_rate_limit(max_retries=2, base_delay=1.0)
    def get_file_content(
        self, owner: str, name: str, file_path: str, branch: str = "main"
    ) -> Optional[str]:
        """Get content of a specific file from repository."""
        try:
            repo = self.github.get_repo(f"{owner}/{name}")

            # Try main branch first, then master
            branches_to_try = [branch, "main", "master"]

            for branch_name in branches_to_try:
                try:
                    file_content = repo.get_contents(file_path, ref=branch_name)
                    if hasattr(file_content, "decoded_content"):
                        return file_content.decoded_content.decode("utf-8")
                except GithubException:
                    continue

            return None

        except GithubException:
            return None

    @retry_on_rate_limit(max_retries=2, base_delay=1.0)
    def list_files(
        self, owner: str, name: str, path: str = "", branch: str = "main"
    ) -> List[Dict[str, Any]]:
        """List files in a repository directory."""
        try:
            repo = self.github.get_repo(f"{owner}/{name}")

            # Try different branch names
            branches_to_try = [branch, "main", "master"]

            for branch_name in branches_to_try:
                try:
                    contents = repo.get_contents(path, ref=branch_name)
                    if not isinstance(contents, list):
                        contents = [contents]

                    return [
                        {
                            "name": content.name,
                            "path": content.path,
                            "type": content.type,
                            "size": content.size,
                            "download_url": content.download_url,
                        }
                        for content in contents
                    ]

                except GithubException:
                    continue

            return []

        except GithubException:
            return []

    def check_file_exists(
        self, owner: str, name: str, file_path: str, branch: str = "main"
    ) -> bool:
        """Check if a file exists in the repository."""
        return self.get_file_content(owner, name, file_path, branch) is not None

    def get_requirements_files(self, owner: str, name: str) -> Dict[str, Optional[str]]:
        """Get common requirements files from repository."""
        requirements_files = {
            "requirements.txt": None,
            "setup.py": None,
            "pyproject.toml": None,
            "environment.yml": None,
            "Pipfile": None,
            "poetry.lock": None,
        }

        for filename in requirements_files.keys():
            content = self.get_file_content(owner, name, filename)
            if content:
                requirements_files[filename] = content

        return requirements_files

    def get_docker_files(self, owner: str, name: str) -> Dict[str, Optional[str]]:
        """Get Docker-related files from repository."""
        docker_files = {
            "Dockerfile": None,
            "docker-compose.yml": None,
            "docker-compose.yaml": None,
            ".dockerignore": None,
        }

        for filename in docker_files.keys():
            content = self.get_file_content(owner, name, filename)
            if content:
                docker_files[filename] = content

        return docker_files

    def get_ci_configs(self, owner: str, name: str) -> Dict[str, Optional[str]]:
        """Get CI/CD configuration files."""
        ci_files = {
            ".github/workflows": None,
            ".gitlab-ci.yml": None,
            ".travis.yml": None,
            "azure-pipelines.yml": None,
            ".circleci/config.yml": None,
        }

        # For GitHub Actions, list workflow files
        workflows = self.list_files(owner, name, ".github/workflows")
        if workflows:
            workflow_contents = {}
            for workflow in workflows:
                if workflow["name"].endswith((".yml", ".yaml")):
                    content = self.get_file_content(owner, name, workflow["path"])
                    if content:
                        workflow_contents[workflow["name"]] = content
            ci_files[".github/workflows"] = workflow_contents

        # Check other CI files
        for filename in [
            ".gitlab-ci.yml",
            ".travis.yml",
            "azure-pipelines.yml",
            ".circleci/config.yml",
        ]:
            content = self.get_file_content(owner, name, filename)
            if content:
                ci_files[filename] = content

        return ci_files

    def get_readme_content(self, owner: str, name: str) -> Optional[str]:
        """Get README content from repository."""
        readme_files = ["README.md", "README.rst", "README.txt", "README"]

        for readme_file in readme_files:
            content = self.get_file_content(owner, name, readme_file)
            if content:
                return content

        return None
