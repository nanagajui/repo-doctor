"""Analysis Agent - Repository analysis and dependency detection."""

import asyncio
import time
from typing import List, Optional

import aiohttp
from github import Github, GithubException, RateLimitExceededException
from ..utils.logging_config import get_logger, log_performance

from ..conflict_detection import (
    MLPackageConflictDetector,
    ConflictSeverity,
    CUDACompatibilityMatrix,
)
from ..models.analysis import (
    Analysis,
    CompatibilityIssue,
    DependencyInfo,
    DependencyType,
    RepositoryInfo,
)
from ..models.system import SystemProfile
from ..utils.config import Config
from ..utils.env import EnvLoader
from ..utils.github import GitHubHelper
from ..utils.llm import LLMAnalyzer, LLMFactory
from ..utils.parsers import RepositoryParser
from .contracts import (
    AgentContractValidator, 
    AgentDataFlow, 
    AgentErrorHandler, 
    AgentPerformanceMonitor
)


class AnalysisAgent:
    """Agent for analyzing repositories."""

    def __init__(
        self, config: Optional[Config] = None, github_token: Optional[str] = None,
        use_cache: bool = True
    ):
        self.logger = get_logger(__name__)
        self.config = config or Config()
        # Use EnvLoader if no token provided
        token = github_token or self.config.integrations.github_token or EnvLoader.get_github_token()
        self.github = Github(token) if token else Github()
        self.github_helper = GitHubHelper(token)
        
        # Wrap with cache if enabled (STREAM B optimization)
        if use_cache:
            from ..cache import GitHubCache
            from ..cache.github_cache import CachedGitHubHelper
            self.cache = GitHubCache()
            self.github_helper = CachedGitHubHelper(self.github_helper, self.cache)
        else:
            self.cache = None
            
        self.repo_parser = RepositoryParser(self.github_helper)

        # Initialize LLM analyzer if configured
        self.config = config or Config.load()
        self.llm_analyzer = LLMFactory.create_analyzer_sync(self.config)
        
        # Initialize conflict detection system
        self.conflict_detector = MLPackageConflictDetector()
        self.cuda_matrix = CUDACompatibilityMatrix()
        
        # Initialize contract validation
        self.performance_monitor = AgentPerformanceMonitor()

    async def analyze(self, repo_url: str, system_profile: Optional[SystemProfile] = None) -> Analysis:
        """Analyze repository for compatibility issues with contract validation."""
        start_time = time.time()

        try:
            # Parse repository URL
            repo_info = self._parse_repo_url(repo_url)

            # Parallel analysis with enhanced error recovery
            analysis_tasks = [
                ("dependencies", self._analyze_dependencies(repo_info)),
                ("docker_files", self._check_dockerfiles(repo_info)),
                ("documentation", self._scan_documentation(repo_info)),
                ("ci_configs", self._check_ci_configs(repo_info)),
            ]
            
            results = await asyncio.gather(
                *[task[1] for task in analysis_tasks],
                return_exceptions=True,
            )
            
            # Process results and log any failures
            processed_results = []
            for i, (task_name, result) in enumerate(zip([task[0] for task in analysis_tasks], results)):
                if isinstance(result, Exception):
                    self.logger.warning(f"Task '{task_name}' failed: {result}")
                    # Provide fallback values based on task type
                    if task_name == "dependencies":
                        processed_results.append([])
                    else:
                        processed_results.append({})
                else:
                    processed_results.append(result)

            analysis_time = time.time() - start_time
            analysis = self._consolidate_findings(repo_info, processed_results, analysis_time)
            
            # Validate the analysis against contracts
            AgentContractValidator.validate_analysis(analysis)
            
            # Check performance
            if not self.performance_monitor.check_analysis_performance(analysis_time):
                self.logger.warning(
                    f"Analysis agent took {analysis_time:.2f}s (target: {self.performance_monitor.performance_targets['analysis_agent']}s)"
                )
            
            # Log performance metrics
            log_performance("repository_analysis", analysis_time, agent="AnalysisAgent", repo=repo_url)
            
            return analysis
            
        except Exception as e:
            # Enhanced error handling with specific error types
            if isinstance(e, (RateLimitExceededException, GithubException)):
                if hasattr(e, 'status') and e.status == 403:
                    # Likely rate limit or access issue
                    self.logger.error(f"GitHub API access issue for {repo_url}: {e}")
                else:
                    self.logger.error(f"GitHub API error for {repo_url}: {e}")
            else:
                self.logger.error(f"Unexpected error analyzing {repo_url}: {e}")
            
            # Handle errors with fallback analysis
            return AgentErrorHandler.handle_analysis_error(e, repo_url, "repository_analysis")

    def _parse_repo_url(self, repo_url: str) -> RepositoryInfo:
        """Parse GitHub repository URL."""
        try:
            repo_data = self.github_helper.parse_repo_url(repo_url)
            repo_info = self.github_helper.get_repo_info(
                repo_data["owner"], repo_data["name"]
            )

            if repo_info:
                return RepositoryInfo(
                    url=repo_url,
                    name=repo_info["name"],
                    owner=repo_data["owner"],
                    description=repo_info.get("description"),
                    stars=repo_info.get("stars", 0),
                    language=repo_info.get("language"),
                    topics=repo_info.get("topics", []),
                )
            else:
                # Fallback for private repos or API issues
                return RepositoryInfo(
                    url=repo_url, name=repo_data["name"], owner=repo_data["owner"]
                )
        except Exception as e:
            # Try to extract basic info from URL even if GitHub API fails
            try:
                import re
                match = re.match(r"https://github\.com/([^/]+)/([^/]+)", repo_url)
                if match:
                    owner, name = match.groups()
                    return RepositoryInfo(
                        url=repo_url, name=name, owner=owner
                    )
            except Exception:
                pass
            raise ValueError(
                f"Failed to parse repository URL: {repo_url}. Error: {str(e)}"
            )

    async def _analyze_dependencies(
        self, repo_info: RepositoryInfo
    ) -> List[DependencyInfo]:
        """Analyze repository dependencies."""
        try:
            dependencies = await self.repo_parser.parse_repository_files(
                repo_info.owner, repo_info.name
            )
            return dependencies
        except Exception as e:
            # Return empty list instead of error dependency to avoid confusion
            return []

    async def _check_dockerfiles(self, repo_info: RepositoryInfo) -> dict:
        """Check for Docker configurations."""
        docker_info = {"has_dockerfile": False, "has_compose": False, "base_images": []}

        try:
            # Check for Dockerfile
            dockerfile_content = await self._get_file_content(
                repo_info.owner, repo_info.name, "Dockerfile"
            )
            if dockerfile_content:
                docker_info["has_dockerfile"] = True
                # Extract base images
                import re

                from_matches = re.findall(
                    r"^FROM\s+([^\s]+)",
                    dockerfile_content,
                    re.MULTILINE | re.IGNORECASE,
                )
                docker_info["base_images"] = from_matches

            # Check for docker-compose.yml
            compose_files = ["docker-compose.yml", "docker-compose.yaml"]
            for compose_file in compose_files:
                compose_content = await self._get_file_content(
                    repo_info.owner, repo_info.name, compose_file
                )
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
            "system_requirements": [],
        }

        try:
            # Check README files
            readme_files = ["README.md", "README.rst", "README.txt", "readme.md"]
            readme_content = None
            raw_readme_content = None

            for readme_file in readme_files:
                content = await self._get_file_content(
                    repo_info.owner, repo_info.name, readme_file
                )
                if content:
                    raw_readme_content = content
                    readme_content = content.lower()
                    break

            if readme_content:
                # Extract Python version requirements
                import re

                python_patterns = [
                    r"python\s*([><=!~]+)\s*([0-9.]+)",
                    r"requires\s+python\s*([><=!~]+)\s*([0-9.]+)",
                    r"python\s+([0-9.]+)\s*or\s+higher",
                    r"python\s+([0-9.]+)\+",
                ]

                for pattern in python_patterns:
                    matches = re.findall(pattern, readme_content)
                    if matches:
                        if isinstance(matches[0], tuple):
                            doc_info["python_version"] = matches[0][
                                -1
                            ]  # Get version number
                        else:
                            doc_info["python_version"] = matches[0]
                        break

                # Check for CUDA/GPU mentions
                cuda_keywords = ["cuda", "gpu", "nvidia", "cudnn", "tensorrt"]
                doc_info["cuda_mentioned"] = any(
                    keyword in readme_content for keyword in cuda_keywords
                )

                gpu_requirement_patterns = [
                    r"requires?\s+gpu",
                    r"gpu\s+required",
                    r"cuda\s+required",
                    r"nvidia\s+gpu",
                ]
                doc_info["gpu_required"] = any(
                    re.search(pattern, readme_content)
                    for pattern in gpu_requirement_patterns
                )

                # Extract installation commands
                install_patterns = [
                    r"pip install[^\n]+",
                    r"conda install[^\n]+",
                    r"apt-get install[^\n]+",
                    r"brew install[^\n]+",
                ]

                for pattern in install_patterns:
                    matches = re.findall(pattern, readme_content)
                    doc_info["installation_commands"].extend(matches)

                # Extract system requirements
                req_patterns = [
                    r"requirements?:([^\n]+)",
                    r"dependencies:([^\n]+)",
                    r"prerequisites:([^\n]+)",
                ]

                for pattern in req_patterns:
                    matches = re.findall(pattern, readme_content)
                    doc_info["system_requirements"].extend(matches)

                # Enhance with LLM analysis if available
                if self.llm_analyzer and raw_readme_content:
                    llm_doc_analysis = (
                        await self.llm_analyzer.enhance_documentation_analysis(
                            raw_readme_content
                        )
                    )
                    if llm_doc_analysis:
                        # Merge LLM insights with regex-based analysis
                        if (
                            llm_doc_analysis.get("python_versions")
                            and not doc_info["python_version"]
                        ):
                            versions = llm_doc_analysis["python_versions"]
                            if versions:
                                doc_info["python_version"] = versions[
                                    0
                                ]  # Use first detected version

                        if (
                            llm_doc_analysis.get("gpu_requirements")
                            and not doc_info["gpu_required"]
                        ):
                            gpu_req = llm_doc_analysis["gpu_requirements"].lower()
                            doc_info["gpu_required"] = (
                                "required" in gpu_req or "cuda" in gpu_req
                            )

                        if llm_doc_analysis.get("system_requirements"):
                            doc_info["system_requirements"].extend(
                                llm_doc_analysis["system_requirements"]
                            )

                        # Store LLM analysis for later use
                        doc_info["llm_analysis"] = llm_doc_analysis

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
            "has_circleci": False,
        }

        try:
            # Check GitHub Actions
            workflow_files = [
                ".github/workflows/test.yml",
                ".github/workflows/ci.yml",
                ".github/workflows/main.yml",
                ".github/workflows/python-app.yml",
            ]

            for workflow_file in workflow_files:
                content = await self._get_file_content(
                    repo_info.owner, repo_info.name, workflow_file
                )
                if content:
                    ci_info["has_github_actions"] = True

                    # Extract Python versions
                    import re

                    python_version_patterns = [
                        r"python-version:\s*\[([^\]]+)\]",
                        r'python-version:\s*["\']([^"\'])+["\']',
                        r"python:\s*\[([^\]]+)\]",
                    ]

                    for pattern in python_version_patterns:
                        matches = re.findall(pattern, content)
                        for match in matches:
                            versions = [
                                v.strip().strip("\"'") for v in match.split(",")
                            ]
                            ci_info["python_versions"].extend(versions)

                    # Extract test commands
                    test_patterns = [
                        r"run:\s*([^\n]*test[^\n]*)",
                        r"run:\s*([^\n]*pytest[^\n]*)",
                        r"run:\s*([^\n]*unittest[^\n]*)",
                        r"run:\s*([^\n]*tox[^\n]*)",
                    ]

                    for pattern in test_patterns:
                        matches = re.findall(pattern, content)
                        ci_info["test_commands"].extend(matches)

                    break

            # Check Travis CI
            travis_content = await self._get_file_content(
                repo_info.owner, repo_info.name, ".travis.yml"
            )
            if travis_content:
                ci_info["has_travis"] = True

                # Extract Python versions from Travis
                import re

                python_matches = re.findall(
                    r"python:\s*\n((?:\s*-\s*[^\n]+\n?)+)", travis_content
                )
                for match in python_matches:
                    versions = re.findall(r"-\s*([^\n]+)", match)
                    ci_info["python_versions"].extend(
                        [v.strip().strip("\"'") for v in versions]
                    )

            # Check CircleCI
            circleci_content = await self._get_file_content(
                repo_info.owner, repo_info.name, ".circleci/config.yml"
            )
            if circleci_content:
                ci_info["has_circleci"] = True

            # Remove duplicates
            ci_info["python_versions"] = list(set(ci_info["python_versions"]))
            ci_info["test_commands"] = list(set(ci_info["test_commands"]))

        except Exception:
            pass

        return ci_info

    def _consolidate_findings(
        self, repo_info: RepositoryInfo, results: List, analysis_time: float
    ) -> Analysis:
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
                compatibility_issues.append(
                    CompatibilityIssue(
                        type="analysis_error",
                        severity="warning",
                        message=f"Analysis error: {str(result)}",
                        component="analyzer",
                    )
                )

        # Analyze compatibility issues
        compatibility_issues.extend(
            self._detect_compatibility_issues(dependencies, doc_info)
        )

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            dependencies, docker_info, doc_info
        )

        # Update repository info with detected files
        repo_info.has_dockerfile = docker_info.get("has_dockerfile", False)
        repo_info.has_conda_env = any("environment.yml" in dep.source for dep in dependencies)
        repo_info.has_requirements = any("requirements.txt" in dep.source for dep in dependencies)
        repo_info.has_setup_py = any("setup.py" in dep.source for dep in dependencies)
        repo_info.has_pyproject_toml = any("pyproject.toml" in dep.source for dep in dependencies)

        return Analysis(
            repository=repo_info,
            dependencies=dependencies,
            python_version_required=doc_info.get("python_version"),
            cuda_version_required=self._extract_cuda_version(dependencies),
            min_gpu_memory_gb=self._estimate_gpu_memory(dependencies),
            compatibility_issues=compatibility_issues,
            analysis_time=analysis_time,
            confidence_score=confidence_score,
        )

    def _detect_compatibility_issues(
        self, dependencies: List[DependencyInfo], doc_info: dict
    ) -> List[CompatibilityIssue]:
        """Detect potential compatibility issues using advanced conflict detection."""
        issues = []

        # Convert dependencies to format expected by conflict detector
        dep_dict = {}
        for dep in dependencies:
            if dep.type == DependencyType.PYTHON and dep.name and dep.version:
                dep_dict[dep.name] = dep.version

        # Use ML package conflict detector
        if dep_dict:
            # Detect version conflicts
            conflicts = self.conflict_detector.detect_conflicts(dep_dict)
            
            # Check CUDA compatibility if we have system CUDA version
            cuda_version = doc_info.get("cuda_version") or self._extract_cuda_version(dependencies)
            if cuda_version:
                cuda_conflicts = self.conflict_detector.detect_conflicts(dep_dict, cuda_version)
                conflicts.extend(cuda_conflicts)
            
            # Prioritize conflicts by severity
            if conflicts:
                conflicts = self.conflict_detector.prioritize_conflicts(conflicts)
            
            # Convert conflicts to CompatibilityIssues
            for conflict in conflicts:
                severity_map = {
                    ConflictSeverity.CRITICAL: "critical",
                    ConflictSeverity.WARNING: "warning",
                    ConflictSeverity.INFO: "info"
                }
                
                issues.append(
                    CompatibilityIssue(
                        type=conflict.conflict_type,
                        severity=severity_map[conflict.severity],
                        message=conflict.description,
                        component=f"{conflict.package1}-{conflict.package2}",
                        suggested_fix=conflict.suggested_resolution
                    )
                )

        # Check CUDA compatibility for each ML framework
        ml_frameworks = ["torch", "tensorflow", "jax", "mxnet"]
        for framework in ml_frameworks:
            if framework in dep_dict:
                # Get system CUDA version if available
                system_cuda = doc_info.get("cuda_version")
                if system_cuda:
                    is_compat, msg = self.cuda_matrix.check_compatibility(
                        framework,
                        dep_dict[framework],
                        system_cuda
                    )
                    if not is_compat and msg:
                        issues.append(
                            CompatibilityIssue(
                                type="cuda_incompatibility",
                                severity="critical",
                                message=msg,
                                component=framework,
                                suggested_fix=f"Install compatible CUDA version for {framework}"
                            )
                        )

        # Check for multi-CUDA conflicts
        cuda_conflicts = self.cuda_matrix.check_multi_cuda_conflict(dep_dict)
        for pkg1, pkg2, conflict_desc in cuda_conflicts:
            issues.append(
                CompatibilityIssue(
                    type="cuda_version_conflict",
                    severity="critical",
                    message=conflict_desc,
                    component=f"{pkg1}-{pkg2}",
                    suggested_fix="Use a single ML framework or ensure CUDA compatibility"
                )
            )

        # Get recommended CUDA versions
        recommended_cuda = self.cuda_matrix.get_recommended_cuda(dep_dict)
        if recommended_cuda and not doc_info.get("cuda_version"):
            issues.append(
                CompatibilityIssue(
                    type="cuda_recommendation",
                    severity="info",
                    message=f"Recommended CUDA versions: {', '.join(recommended_cuda)}",
                    component="cuda",
                    suggested_fix=f"Install CUDA {recommended_cuda[0]} for best compatibility"
                )
            )

        # Check for GPU requirements without CUDA (original check)
        gpu_deps = [dep for dep in dependencies if dep.gpu_required]
        if gpu_deps and not doc_info.get("cuda_mentioned", False):
            issues.append(
                CompatibilityIssue(
                    type="gpu_requirement",
                    severity="warning",
                    message="Repository appears to require GPU but CUDA setup not clearly documented",
                    component="gpu_dependencies",
                    suggested_fix="Ensure CUDA toolkit is installed and configured",
                )
            )

        # Check for duplicate packages (original check, enhanced)
        package_counts = {}
        for dep in dependencies:
            if dep.name in package_counts:
                package_counts[dep.name].append(dep.version)
            else:
                package_counts[dep.name] = [dep.version]

        for package, versions in package_counts.items():
            if len(versions) > 1 and len(set(versions)) > 1:
                issues.append(
                    CompatibilityIssue(
                        type="duplicate_dependency",
                        severity="warning",
                        message=f"Package '{package}' specified multiple times with different versions: {', '.join(set(versions))}",
                        component=package,
                        suggested_fix="Consolidate to a single compatible version",
                    )
                )

        return issues

    def _extract_cuda_version(
        self, dependencies: List[DependencyInfo]
    ) -> Optional[str]:
        """Extract CUDA version requirements from dependencies."""
        # Build dependency dict for CUDA matrix
        dep_dict = {}
        for dep in dependencies:
            if dep.type == DependencyType.PYTHON and dep.name and dep.version:
                dep_dict[dep.name] = dep.version
        
        # Get recommended CUDA versions based on dependencies
        if dep_dict:
            recommended = self.cuda_matrix.get_recommended_cuda(dep_dict)
            if recommended:
                return recommended[0]  # Return the first recommended version
        
        return None

    def _estimate_gpu_memory(self, dependencies: List[DependencyInfo]) -> float:
        """Estimate minimum GPU memory requirements."""
        gpu_deps = [dep for dep in dependencies if dep.gpu_required]
        if not gpu_deps:
            return 0.0

        # Basic estimation - can be enhanced with actual model requirements
        if any(dep.name in ["torch", "tensorflow"] for dep in gpu_deps):
            return 4.0  # Assume 4GB minimum for ML workloads

        return 0.0

    def _calculate_confidence_score(
        self, dependencies: List[DependencyInfo], docker_info: dict, doc_info: dict
    ) -> float:
        """Calculate confidence score for the analysis."""
        score = 0.0

        # Base score for finding dependencies
        if dependencies:
            score += 0.4

        # Bonus for finding requirements files
        req_sources = set(dep.source.split(":")[0] for dep in dependencies)
        if "requirements.txt" in req_sources:
            score += 0.2
        if "setup.py" in req_sources or "pyproject.toml" in req_sources:
            score += 0.2

        # Bonus for Docker configuration
        if docker_info.get("has_dockerfile"):
            score += 0.1

        # Bonus for documentation analysis
        if doc_info.get("python_version"):
            score += 0.1

        return min(score, 1.0)

    async def _get_file_content(
        self, owner: str, name: str, filename: str
    ) -> Optional[str]:
        """Get file content from GitHub repository."""
        try:
            repo = self.github.get_repo(f"{owner}/{name}")
            file_content = repo.get_contents(filename)
            return file_content.decoded_content.decode("utf-8")
        except Exception:
            return None
