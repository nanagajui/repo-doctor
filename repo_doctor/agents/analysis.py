"""Analysis Agent - Repository analysis and dependency detection."""

import asyncio
import aiohttp
import time
from typing import List, Optional
from github import Github
from ..models.analysis import Analysis, RepositoryInfo, DependencyInfo, CompatibilityIssue, DependencyType
from ..utils.github import GitHubHelper
from ..utils.parsers import RepositoryParser


class AnalysisAgent:
    """Agent for analyzing repositories."""
    
    def __init__(self, github_token: Optional[str] = None):
        self.github = Github(github_token) if github_token else Github()
        self.github_helper = GitHubHelper(github_token)
        self.repo_parser = RepositoryParser(self.github_helper)
    
    async def analyze(self, repo_url: str) -> Analysis:
        """Analyze repository for compatibility issues."""
        start_time = time.time()
        
        # Parse repository URL
        repo_info = self._parse_repo_url(repo_url)
        
        # Parallel analysis
        results = await asyncio.gather(
            self._analyze_dependencies(repo_info),
            self._check_dockerfiles(repo_info),
            self._scan_documentation(repo_info),
            self._check_ci_configs(repo_info),
            return_exceptions=True
        )
        
        analysis_time = time.time() - start_time
        return self._consolidate_findings(repo_info, results, analysis_time)
    
    def _parse_repo_url(self, repo_url: str) -> RepositoryInfo:
        """Parse GitHub repository URL."""
        try:
            repo_data = self.github_helper.parse_repo_url(repo_url)
            repo_info = self.github_helper.get_repo_info(repo_data["owner"], repo_data["name"])
            
            if repo_info:
                return RepositoryInfo(
                    url=repo_url,
                    name=repo_info["name"],
                    owner=repo_data["owner"],
                    description=repo_info.get("description"),
                    stars=repo_info.get("stars", 0),
                    language=repo_info.get("language"),
                    topics=repo_info.get("topics", [])
                )
            else:
                # Fallback for private repos or API issues
                return RepositoryInfo(
                    url=repo_url,
                    name=repo_data["name"],
                    owner=repo_data["owner"]
                )
        except Exception as e:
            raise ValueError(f"Failed to parse repository URL: {repo_url}. Error: {str(e)}")
    
    async def _analyze_dependencies(self, repo_info: RepositoryInfo) -> List[DependencyInfo]:
        """Analyze repository dependencies."""
        try:
            dependencies = await self.repo_parser.parse_repository_files(
                repo_info.owner, repo_info.name
            )
            return dependencies
        except Exception as e:
            return [DependencyInfo(
                name="analysis_error",
                version=None,
                type=DependencyType.SYSTEM,
                source="error",
                optional=True
            )]
    
    async def _check_dockerfiles(self, repo_info: RepositoryInfo) -> dict:
        """Check for Docker configurations."""
        docker_info = {
            "has_dockerfile": False,
            "has_compose": False,
            "base_images": []
        }
        
        try:
            # Check for Dockerfile
            dockerfile_content = await self._get_file_content(repo_info.owner, repo_info.name, "Dockerfile")
            if dockerfile_content:
                docker_info["has_dockerfile"] = True
                # Extract base images
                import re
                from_matches = re.findall(r'^FROM\s+([^\s]+)', dockerfile_content, re.MULTILINE | re.IGNORECASE)
                docker_info["base_images"] = from_matches
            
            # Check for docker-compose.yml
            compose_files = ["docker-compose.yml", "docker-compose.yaml"]
            for compose_file in compose_files:
                compose_content = await self._get_file_content(repo_info.owner, repo_info.name, compose_file)
                if compose_content:
                    docker_info["has_compose"] = True
                    break
        
        except Exception:
            pass
        
        return docker_info
    
    async def _scan_documentation(self, repo_info: RepositoryInfo) -> dict:
        """Scan README and documentation for requirements."""
        doc_info = {
            "python_version": None,
            "cuda_mentioned": False,
            "gpu_required": False,
            "installation_commands": [],
            "system_requirements": []
        }
        
        try:
            # Check README files
            readme_files = ["README.md", "README.rst", "README.txt", "readme.md"]
            readme_content = None
            
            for readme_file in readme_files:
                content = await self._get_file_content(repo_info.owner, repo_info.name, readme_file)
                if content:
                    readme_content = content.lower()
                    break
            
            if readme_content:
                # Extract Python version requirements
                import re
                python_patterns = [
                    r'python\s*([><=!~]+)\s*([0-9.]+)',
                    r'requires\s+python\s*([><=!~]+)\s*([0-9.]+)',
                    r'python\s+([0-9.]+)\s*or\s+higher',
                    r'python\s+([0-9.]+)\+'
                ]
                
                for pattern in python_patterns:
                    matches = re.findall(pattern, readme_content)
                    if matches:
                        if isinstance(matches[0], tuple):
                            doc_info["python_version"] = matches[0][-1]  # Get version number
                        else:
                            doc_info["python_version"] = matches[0]
                        break
                
                # Check for CUDA/GPU mentions
                cuda_keywords = ['cuda', 'gpu', 'nvidia', 'cudnn', 'tensorrt']
                doc_info["cuda_mentioned"] = any(keyword in readme_content for keyword in cuda_keywords)
                
                gpu_requirement_patterns = [
                    r'requires?\s+gpu',
                    r'gpu\s+required',
                    r'cuda\s+required',
                    r'nvidia\s+gpu'
                ]
                doc_info["gpu_required"] = any(re.search(pattern, readme_content) for pattern in gpu_requirement_patterns)
                
                # Extract installation commands
                install_patterns = [
                    r'pip install[^\n]+',
                    r'conda install[^\n]+',
                    r'apt-get install[^\n]+',
                    r'brew install[^\n]+'
                ]
                
                for pattern in install_patterns:
                    matches = re.findall(pattern, readme_content)
                    doc_info["installation_commands"].extend(matches)
                
                # Extract system requirements
                req_patterns = [
                    r'requirements?:([^\n]+)',
                    r'dependencies:([^\n]+)',
                    r'prerequisites:([^\n]+)'
                ]
                
                for pattern in req_patterns:
                    matches = re.findall(pattern, readme_content)
                    doc_info["system_requirements"].extend(matches)
        
        except Exception:
            pass
        
        return doc_info
    
    async def _check_ci_configs(self, repo_info: RepositoryInfo) -> dict:
        """Check CI/CD configurations."""
        ci_info = {
            "has_github_actions": False,
            "python_versions": [],
            "test_commands": [],
            "has_travis": False,
            "has_circleci": False
        }
        
        try:
            # Check GitHub Actions
            workflow_files = [
                ".github/workflows/test.yml",
                ".github/workflows/ci.yml", 
                ".github/workflows/main.yml",
                ".github/workflows/python-app.yml"
            ]
            
            for workflow_file in workflow_files:
                content = await self._get_file_content(repo_info.owner, repo_info.name, workflow_file)
                if content:
                    ci_info["has_github_actions"] = True
                    
                    # Extract Python versions
                    import re
                    python_version_patterns = [
                        r'python-version:\s*\[([^\]]+)\]',
                        r'python-version:\s*["\']([^"\'])+["\']',
                        r'python:\s*\[([^\]]+)\]'
                    ]
                    
                    for pattern in python_version_patterns:
                        matches = re.findall(pattern, content)
                        for match in matches:
                            versions = [v.strip().strip('"\'') for v in match.split(',')]
                            ci_info["python_versions"].extend(versions)
                    
                    # Extract test commands
                    test_patterns = [
                        r'run:\s*([^\n]*test[^\n]*)',
                        r'run:\s*([^\n]*pytest[^\n]*)',
                        r'run:\s*([^\n]*unittest[^\n]*)',
                        r'run:\s*([^\n]*tox[^\n]*)'
                    ]
                    
                    for pattern in test_patterns:
                        matches = re.findall(pattern, content)
                        ci_info["test_commands"].extend(matches)
                    
                    break
            
            # Check Travis CI
            travis_content = await self._get_file_content(repo_info.owner, repo_info.name, ".travis.yml")
            if travis_content:
                ci_info["has_travis"] = True
                
                # Extract Python versions from Travis
                import re
                python_matches = re.findall(r'python:\s*\n((?:\s*-\s*[^\n]+\n?)+)', travis_content)
                for match in python_matches:
                    versions = re.findall(r'-\s*([^\n]+)', match)
                    ci_info["python_versions"].extend([v.strip().strip('"\'') for v in versions])
            
            # Check CircleCI
            circleci_content = await self._get_file_content(repo_info.owner, repo_info.name, ".circleci/config.yml")
            if circleci_content:
                ci_info["has_circleci"] = True
            
            # Remove duplicates
            ci_info["python_versions"] = list(set(ci_info["python_versions"]))
            ci_info["test_commands"] = list(set(ci_info["test_commands"]))
        
        except Exception:
            pass
        
        return ci_info
    
    def _consolidate_findings(self, repo_info: RepositoryInfo, results: List, analysis_time: float) -> Analysis:
        """Consolidate analysis results."""
        dependencies = []
        compatibility_issues = []
        docker_info = {}
        doc_info = {}
        ci_info = {}
        
        # Process results from parallel analysis
        for i, result in enumerate(results):
            if isinstance(result, list):
                dependencies.extend(result)
            elif isinstance(result, dict):
                if i == 1:  # Docker info
                    docker_info = result
                elif i == 2:  # Documentation info
                    doc_info = result
                elif i == 3:  # CI info
                    ci_info = result
            elif isinstance(result, Exception):
                compatibility_issues.append(CompatibilityIssue(
                    type="analysis_error",
                    severity="warning",
                    message=f"Analysis error: {str(result)}",
                    component="analyzer"
                ))
        
        # Analyze compatibility issues
        compatibility_issues.extend(self._detect_compatibility_issues(dependencies, doc_info))
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(dependencies, docker_info, doc_info)
        
        return Analysis(
            repository=repo_info,
            dependencies=dependencies,
            python_version_required=doc_info.get("python_version"),
            cuda_version_required=self._extract_cuda_version(dependencies),
            min_gpu_memory_gb=self._estimate_gpu_memory(dependencies),
            compatibility_issues=compatibility_issues,
            analysis_time=analysis_time,
            confidence_score=confidence_score
        )
    
    def _detect_compatibility_issues(self, dependencies: List[DependencyInfo], doc_info: dict) -> List[CompatibilityIssue]:
        """Detect potential compatibility issues."""
        issues = []
        
        # Check for GPU requirements without CUDA
        gpu_deps = [dep for dep in dependencies if dep.gpu_required]
        if gpu_deps and not doc_info.get("cuda_mentioned", False):
            issues.append(CompatibilityIssue(
                type="gpu_requirement",
                severity="warning",
                message="Repository appears to require GPU but CUDA setup not clearly documented",
                component="gpu_dependencies",
                suggested_fix="Ensure CUDA toolkit is installed and configured"
            ))
        
        # Check for conflicting package versions
        package_counts = {}
        for dep in dependencies:
            if dep.name in package_counts:
                package_counts[dep.name] += 1
            else:
                package_counts[dep.name] = 1
        
        for package, count in package_counts.items():
            if count > 1:
                issues.append(CompatibilityIssue(
                    type="version_conflict",
                    severity="info",
                    message=f"Package '{package}' specified multiple times with potentially different versions",
                    component=package,
                    suggested_fix="Review dependency specifications for consistency"
                ))
        
        return issues
    
    def _extract_cuda_version(self, dependencies: List[DependencyInfo]) -> Optional[str]:
        """Extract CUDA version requirements from dependencies."""
        for dep in dependencies:
            if dep.name in ['torch', 'tensorflow']:
                # This would need more sophisticated version mapping
                return None  # TODO: Implement CUDA version mapping
        return None
    
    def _estimate_gpu_memory(self, dependencies: List[DependencyInfo]) -> float:
        """Estimate minimum GPU memory requirements."""
        gpu_deps = [dep for dep in dependencies if dep.gpu_required]
        if not gpu_deps:
            return 0.0
        
        # Basic estimation - can be enhanced with actual model requirements
        if any(dep.name in ['torch', 'tensorflow'] for dep in gpu_deps):
            return 4.0  # Assume 4GB minimum for ML workloads
        
        return 0.0
    
    def _calculate_confidence_score(self, dependencies: List[DependencyInfo], docker_info: dict, doc_info: dict) -> float:
        """Calculate confidence score for the analysis."""
        score = 0.0
        
        # Base score for finding dependencies
        if dependencies:
            score += 0.4
        
        # Bonus for finding requirements files
        req_sources = set(dep.source.split(':')[0] for dep in dependencies)
        if 'requirements.txt' in req_sources:
            score += 0.2
        if 'setup.py' in req_sources or 'pyproject.toml' in req_sources:
            score += 0.2
        
        # Bonus for Docker configuration
        if docker_info.get("has_dockerfile"):
            score += 0.1
        
        # Bonus for documentation analysis
        if doc_info.get("python_version"):
            score += 0.1
        
        return min(score, 1.0)
    
    async def _get_file_content(self, owner: str, name: str, filename: str) -> Optional[str]:
        """Get file content from GitHub repository."""
        try:
            repo = self.github.get_repo(f"{owner}/{name}")
            file_content = repo.get_contents(filename)
            return file_content.decoded_content.decode('utf-8')
        except Exception:
            return None
