"""Repository parsing utilities for dependency extraction."""

import ast
import asyncio
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiohttp

from ..models.analysis import DependencyInfo, DependencyType


class RequirementsParser:
    """Parser for requirements.txt files."""

    @staticmethod
    def parse_requirements_txt(content: str) -> List[DependencyInfo]:
        """Parse requirements.txt content."""
        dependencies = []

        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Handle -r includes (recursive requirements)
            if line.startswith("-r "):
                continue  # TODO: Handle recursive requirements

            # Parse package specification
            dep_info = RequirementsParser._parse_requirement_line(line)
            if dep_info:
                dep_info.source = f"requirements.txt:L{line_num}"
                dependencies.append(dep_info)

        return dependencies

    @staticmethod
    def _parse_requirement_line(line: str) -> Optional[DependencyInfo]:
        """Parse a single requirement line."""
        # Remove inline comments
        line = line.split("#")[0].strip()

        # Common patterns for package specifications
        patterns = [
            r"^([a-zA-Z0-9_-]+)\s*([><=!~]+)\s*([0-9.]+.*?)$",  # package>=1.0.0
            r"^([a-zA-Z0-9_-]+)\s*$",  # package (no version)
        ]

        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                name = match.group(1).lower()
                version = match.group(3) if len(match.groups()) > 2 else None

                # Detect GPU-related packages
                gpu_packages = {
                    "torch",
                    "tensorflow",
                    "tensorflow-gpu",
                    "cupy",
                    "numba",
                    "jax",
                    "jaxlib",
                    "paddle",
                    "mxnet",
                    "mindspore",
                }

                return DependencyInfo(
                    name=name,
                    version=version,
                    type=DependencyType.PYTHON,
                    source="requirements.txt",
                    gpu_required=name in gpu_packages,
                )

        return None


class SetupPyParser:
    """Parser for setup.py files."""

    @staticmethod
    def parse_setup_py(content: str) -> List[DependencyInfo]:
        """Parse setup.py content using AST."""
        dependencies = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and hasattr(node.func, "id"):
                    if node.func.id == "setup":
                        deps = SetupPyParser._extract_setup_dependencies(node)
                        dependencies.extend(deps)

        except SyntaxError:
            # If AST parsing fails, try regex fallback
            dependencies = SetupPyParser._parse_setup_py_regex(content)

        return dependencies

    @staticmethod
    def _extract_setup_dependencies(setup_node: ast.Call) -> List[DependencyInfo]:
        """Extract dependencies from setup() call."""
        dependencies = []

        for keyword in setup_node.keywords:
            if keyword.arg in ["install_requires", "requires"]:
                if isinstance(keyword.value, ast.List):
                    for item in keyword.value.elts:
                        if isinstance(item, ast.Str):
                            dep = RequirementsParser._parse_requirement_line(item.s)
                            if dep:
                                dep.source = "setup.py"
                                dependencies.append(dep)

        return dependencies

    @staticmethod
    def _parse_setup_py_regex(content: str) -> List[DependencyInfo]:
        """Fallback regex parsing for setup.py."""
        dependencies = []

        # Look for install_requires patterns
        patterns = [r"install_requires\s*=\s*\[(.*?)\]", r"requires\s*=\s*\[(.*?)\]"]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                # Extract quoted strings
                deps = re.findall(r'["\']([^"\']+)["\']', match)
                for dep_str in deps:
                    dep = RequirementsParser._parse_requirement_line(dep_str)
                    if dep:
                        dep.source = "setup.py"
                        dependencies.append(dep)

        return dependencies


class PyProjectParser:
    """Parser for pyproject.toml files."""

    @staticmethod
    def parse_pyproject_toml(content: str) -> List[DependencyInfo]:
        """Parse pyproject.toml content."""
        dependencies = []

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                # Fallback to regex parsing
                return PyProjectParser._parse_pyproject_regex(content)

        try:
            data = tomllib.loads(content)

            # Check project.dependencies
            if "project" in data and "dependencies" in data["project"]:
                for dep_str in data["project"]["dependencies"]:
                    dep = RequirementsParser._parse_requirement_line(dep_str)
                    if dep:
                        dep.source = "pyproject.toml"
                        dependencies.append(dep)

            # Check tool.poetry.dependencies
            if (
                "tool" in data
                and "poetry" in data["tool"]
                and "dependencies" in data["tool"]["poetry"]
            ):
                poetry_deps = data["tool"]["poetry"]["dependencies"]
                for name, version_spec in poetry_deps.items():
                    if name == "python":
                        continue

                    version = None
                    if isinstance(version_spec, str):
                        version = version_spec
                    elif isinstance(version_spec, dict) and "version" in version_spec:
                        version = version_spec["version"]

                    gpu_packages = {
                        "torch",
                        "tensorflow",
                        "tensorflow-gpu",
                        "cupy",
                        "numba",
                        "jax",
                        "jaxlib",
                        "paddle",
                        "mxnet",
                        "mindspore",
                    }

                    dependencies.append(
                        DependencyInfo(
                            name=name,
                            version=version,
                            type=DependencyType.PYTHON,
                            source="pyproject.toml",
                            gpu_required=name in gpu_packages,
                        )
                    )

        except Exception:
            # Fallback to regex parsing
            dependencies = PyProjectParser._parse_pyproject_regex(content)

        return dependencies

    @staticmethod
    def _parse_pyproject_regex(content: str) -> List[DependencyInfo]:
        """Fallback regex parsing for pyproject.toml."""
        dependencies = []

        # Simple regex patterns for common cases
        dep_patterns = [
            r"dependencies\s*=\s*\[(.*?)\]",
            r"\[tool\.poetry\.dependencies\](.*?)(?=\[|\Z)",
        ]

        for pattern in dep_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                deps = re.findall(r'["\']([^"\']+)["\']', match)
                for dep_str in deps:
                    if "=" in dep_str or ">" in dep_str or "<" in dep_str:
                        dep = RequirementsParser._parse_requirement_line(dep_str)
                        if dep:
                            dep.source = "pyproject.toml"
                            dependencies.append(dep)

        return dependencies


class ImportScanner:
    """Scanner for Python imports in source code."""

    @staticmethod
    def scan_imports_from_content(
        content: str, filename: str = "unknown"
    ) -> List[DependencyInfo]:
        """Scan Python imports from source code content."""
        dependencies = []

        try:
            tree = ast.parse(content)
            imports = ImportScanner._extract_imports(tree)

            for import_name in imports:
                # Map import names to package names (basic mapping)
                package_name = ImportScanner._map_import_to_package(import_name)

                if package_name and package_name not in ["builtins", "sys", "os"]:
                    gpu_packages = {
                        "torch",
                        "tensorflow",
                        "tf",
                        "cupy",
                        "numba",
                        "jax",
                        "jaxlib",
                        "paddle",
                        "mxnet",
                        "mindspore",
                    }

                    dependencies.append(
                        DependencyInfo(
                            name=package_name,
                            version=None,
                            type=DependencyType.PYTHON,
                            source=f"imports:{filename}",
                            gpu_required=package_name in gpu_packages,
                        )
                    )

        except SyntaxError:
            pass  # Skip files with syntax errors

        return dependencies

    @staticmethod
    def _extract_imports(tree: ast.AST) -> Set[str]:
        """Extract import names from AST."""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

        return imports

    @staticmethod
    def _map_import_to_package(import_name: str) -> str:
        """Map import name to package name."""
        # Common mappings from import name to package name
        import_to_package = {
            "cv2": "opencv-python",
            "PIL": "pillow",
            "sklearn": "scikit-learn",
            "tf": "tensorflow",
            "np": "numpy",
            "pd": "pandas",
            "plt": "matplotlib",
            "sns": "seaborn",
        }

        return import_to_package.get(import_name, import_name)


class RepositoryParser:
    """Main repository parser that coordinates all parsing strategies."""

    def __init__(self, github_backend):
        # github_backend can be:
        # - an object with attribute `.github` exposing a PyGithub client (e.g., AnalysisAgent, GitHubHelper)
        # - a PyGithub client itself exposing `.get_repo`
        # - or a wrapper with `.get_repo`
        self._backend = github_backend

    def _get_repo(self, owner: str, name: str):
        # Prefer `.github` attribute if present
        client = getattr(self._backend, "github", None)
        if client and hasattr(client, "get_repo"):
            return client.get_repo(f"{owner}/{name}")
        # Fallback: backend itself may be a client
        if hasattr(self._backend, "get_repo"):
            return self._backend.get_repo(f"{owner}/{name}")
        # Last resort: try nested `.github_helper.github`
        helper = getattr(self._backend, "github_helper", None)
        if helper and hasattr(helper, "github") and hasattr(helper.github, "get_repo"):
            return helper.github.get_repo(f"{owner}/{name}")
        raise RuntimeError("No GitHub client available in RepositoryParser backend")

    async def parse_repository_files(
        self, owner: str, name: str
    ) -> List[DependencyInfo]:
        """Parse repository files for dependencies."""
        all_dependencies = []

        # Parse different file types
        parsers = [
            ("requirements.txt", RequirementsParser.parse_requirements_txt),
            ("setup.py", SetupPyParser.parse_setup_py),
            ("pyproject.toml", PyProjectParser.parse_pyproject_toml),
        ]

        for filename, parser_func in parsers:
            content = await self._get_file_content(owner, name, filename)
            if content:
                dependencies = parser_func(content)
                all_dependencies.extend(dependencies)

        # Scan Python files for imports (limited to avoid rate limits)
        python_files = await self._get_python_files(owner, name, limit=10)
        for file_path, content in python_files:
            imports = ImportScanner.scan_imports_from_content(content, file_path)
            all_dependencies.extend(imports)

        # Deduplicate dependencies
        return self._deduplicate_dependencies(all_dependencies)

    async def _get_file_content(
        self, owner: str, name: str, filename: str
    ) -> Optional[str]:
        """Get file content from GitHub repository."""
        try:
            repo = self._get_repo(owner, name)
            file_content = repo.get_contents(filename)
            return file_content.decoded_content.decode("utf-8")
        except Exception:
            return None

    async def _get_python_files(
        self, owner: str, name: str, limit: int = 10
    ) -> List[tuple]:
        """Get Python file contents from repository."""
        python_files = []

        try:
            repo = self._get_repo(owner, name)
            contents = repo.get_contents("")

            count = 0
            for content_file in contents:
                if count >= limit:
                    break

                if (
                    content_file.name.endswith(".py") and content_file.size < 50000
                ):  # Limit file size
                    try:
                        file_content = repo.get_contents(content_file.path)
                        python_files.append(
                            (
                                content_file.path,
                                file_content.decoded_content.decode("utf-8"),
                            )
                        )
                        count += 1
                    except Exception:
                        continue

        except Exception:
            pass

        return python_files

    def _deduplicate_dependencies(
        self, dependencies: List[DependencyInfo]
    ) -> List[DependencyInfo]:
        """Remove duplicate dependencies, preferring more specific versions."""
        seen = {}

        for dep in dependencies:
            key = dep.name.lower()

            if key not in seen:
                seen[key] = dep
            else:
                # Prefer dependency with version information
                existing = seen[key]
                if dep.version and not existing.version:
                    seen[key] = dep
                elif dep.gpu_required and not existing.gpu_required:
                    seen[key] = dep

        return list(seen.values())
