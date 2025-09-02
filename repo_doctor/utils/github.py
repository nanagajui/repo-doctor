"""GitHub API utilities."""

import re
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
from github import Github, GithubException


class GitHubHelper:
    """Helper class for GitHub API operations."""
    
    def __init__(self, token: Optional[str] = None):
        self.github = Github(token) if token else Github()
    
    def parse_repo_url(self, url: str) -> Dict[str, str]:
        """Parse GitHub repository URL to extract owner and repo name."""
        # Handle various GitHub URL formats
        patterns = [
            r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$',
            r'github\.com/([^/]+)/([^/]+)/.*',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return {
                    "owner": match.group(1),
                    "name": match.group(2),
                    "full_name": f"{match.group(1)}/{match.group(2)}"
                }
        
        raise ValueError(f"Invalid GitHub URL format: {url}")
    
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
                "private": repo.private
            }
        
        except GithubException as e:
            if e.status == 404:
                return None  # Repository not found or private
            raise e
    
    def get_file_content(self, owner: str, name: str, file_path: str, 
                        branch: str = "main") -> Optional[str]:
        """Get content of a specific file from repository."""
        try:
            repo = self.github.get_repo(f"{owner}/{name}")
            
            # Try main branch first, then master
            branches_to_try = [branch, "main", "master"]
            
            for branch_name in branches_to_try:
                try:
                    file_content = repo.get_contents(file_path, ref=branch_name)
                    if hasattr(file_content, 'decoded_content'):
                        return file_content.decoded_content.decode('utf-8')
                except GithubException:
                    continue
            
            return None
        
        except GithubException:
            return None
    
    def list_files(self, owner: str, name: str, path: str = "", 
                   branch: str = "main") -> List[Dict[str, Any]]:
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
                    
                    return [{
                        "name": content.name,
                        "path": content.path,
                        "type": content.type,
                        "size": content.size,
                        "download_url": content.download_url
                    } for content in contents]
                
                except GithubException:
                    continue
            
            return []
        
        except GithubException:
            return []
    
    def check_file_exists(self, owner: str, name: str, file_path: str,
                         branch: str = "main") -> bool:
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
            "poetry.lock": None
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
            ".dockerignore": None
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
            ".circleci/config.yml": None
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
        for filename in [".gitlab-ci.yml", ".travis.yml", "azure-pipelines.yml", ".circleci/config.yml"]:
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
