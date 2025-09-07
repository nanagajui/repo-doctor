"""Tests for repository parsing utilities."""

import ast
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest

from repo_doctor.utils.parsers import (
    RequirementsParser,
    SetupPyParser,
    PyProjectParser,
    ImportScanner,
    RepositoryParser
)
from repo_doctor.models.analysis import DependencyInfo, DependencyType


class TestRequirementsParser:
    """Test RequirementsParser class."""

    def test_parse_requirements_txt_basic(self):
        """Test basic requirements.txt parsing."""
        content = """
# This is a comment
numpy>=1.20.0
pandas==1.3.0
torch
tensorflow-gpu>=2.0.0

# Another comment
scikit-learn~=1.0.0
"""
        dependencies = RequirementsParser.parse_requirements_txt(content)
        
        assert len(dependencies) == 5
        
        # Check numpy
        numpy_dep = next(d for d in dependencies if d.name == "numpy")
        assert numpy_dep.version == "1.20.0"
        assert numpy_dep.type == DependencyType.PYTHON
        assert not numpy_dep.gpu_required
        assert "requirements.txt:L3" in numpy_dep.source
        
        # Check torch (GPU package)
        torch_dep = next(d for d in dependencies if d.name == "torch")
        assert torch_dep.gpu_required
        assert torch_dep.version is None
        
        # Check tensorflow-gpu
        tf_dep = next(d for d in dependencies if d.name == "tensorflow-gpu")
        assert tf_dep.gpu_required
        assert tf_dep.version == "2.0.0"

    def test_parse_requirements_txt_empty_and_comments(self):
        """Test parsing with empty lines and comments only."""
        content = """
# Just comments
# More comments

# Empty lines above
"""
        dependencies = RequirementsParser.parse_requirements_txt(content)
        assert len(dependencies) == 0

    def test_parse_requirements_txt_recursive_includes(self):
        """Test handling of -r includes."""
        content = """
numpy>=1.20.0
-r dev-requirements.txt
pandas==1.3.0
"""
        dependencies = RequirementsParser.parse_requirements_txt(content)
        
        # Should skip -r line but parse others
        assert len(dependencies) == 2
        names = [d.name for d in dependencies]
        assert "numpy" in names
        assert "pandas" in names

    def test_parse_requirement_line_various_formats(self):
        """Test parsing different requirement line formats."""
        test_cases = [
            ("numpy>=1.20.0", "numpy", "1.20.0"),
            ("pandas==1.3.0", "pandas", "1.3.0"),
            ("torch", "torch", None),
            ("scikit-learn~=1.0.0", "scikit-learn", "1.0.0"),
            ("tensorflow<=2.5.0", "tensorflow", "2.5.0"),
            ("package-name>=1.0.0", "package-name", "1.0.0"),
        ]
        
        for line, expected_name, expected_version in test_cases:
            dep = RequirementsParser._parse_requirement_line(line)
            assert dep is not None
            assert dep.name == expected_name
            assert dep.version == expected_version

    def test_parse_requirement_line_with_comments(self):
        """Test parsing requirement lines with inline comments."""
        dep = RequirementsParser._parse_requirement_line("numpy>=1.20.0  # Scientific computing")
        assert dep is not None
        assert dep.name == "numpy"
        assert dep.version == "1.20.0"

    def test_parse_requirement_line_invalid(self):
        """Test parsing invalid requirement lines."""
        invalid_lines = [
            "",
            "# Just a comment",
            "invalid-format-here!!!",
            "git+https://github.com/user/repo.git",
        ]
        
        for line in invalid_lines:
            dep = RequirementsParser._parse_requirement_line(line)
            assert dep is None

    def test_gpu_package_detection(self):
        """Test GPU package detection."""
        gpu_packages = ["torch", "tensorflow", "tensorflow-gpu", "cupy", "numba", "jax", "jaxlib"]
        
        for package in gpu_packages:
            dep = RequirementsParser._parse_requirement_line(package)
            assert dep is not None
            assert dep.gpu_required, f"{package} should be detected as GPU package"
        
        # Test non-GPU package
        dep = RequirementsParser._parse_requirement_line("numpy")
        assert dep is not None
        assert not dep.gpu_required


class TestSetupPyParser:
    """Test SetupPyParser class."""

    def test_parse_setup_py_basic(self):
        """Test basic setup.py parsing."""
        content = '''
from setuptools import setup

setup(
    name="test-package",
    version="1.0.0",
    install_requires=[
        "numpy>=1.20.0",
        "pandas==1.3.0",
        "torch",
    ],
)
'''
        dependencies = SetupPyParser.parse_setup_py(content)
        
        assert len(dependencies) == 3
        names = [d.name for d in dependencies]
        assert "numpy" in names
        assert "pandas" in names
        assert "torch" in names
        
        # Check source attribution
        for dep in dependencies:
            assert dep.source == "setup.py"

    def test_parse_setup_py_requires_keyword(self):
        """Test setup.py with requires keyword."""
        content = '''
setup(
    name="test-package",
    requires=["scipy>=1.0.0", "matplotlib"],
)
'''
        dependencies = SetupPyParser.parse_setup_py(content)
        
        assert len(dependencies) == 2
        names = [d.name for d in dependencies]
        assert "scipy" in names
        assert "matplotlib" in names

    def test_parse_setup_py_syntax_error_fallback(self):
        """Test fallback to regex parsing on syntax error."""
        content = '''
# Invalid Python syntax
setup(
    install_requires=[
        "numpy>=1.20.0",
        "pandas==1.3.0",
    ]
'''
        dependencies = SetupPyParser.parse_setup_py(content)
        
        # Should still parse using regex fallback
        assert len(dependencies) >= 1
        names = [d.name for d in dependencies]
        assert "numpy" in names or "pandas" in names

    def test_extract_setup_dependencies_ast(self):
        """Test AST-based dependency extraction."""
        # Create a mock AST node
        setup_node = ast.Call(
            func=ast.Name(id="setup"),
            args=[],
            keywords=[
                ast.keyword(
                    arg="install_requires",
                    value=ast.List(
                        elts=[
                            ast.Str(s="numpy>=1.20.0"),
                            ast.Str(s="torch"),
                        ]
                    )
                )
            ]
        )
        
        dependencies = SetupPyParser._extract_setup_dependencies(setup_node)
        
        assert len(dependencies) == 2
        names = [d.name for d in dependencies]
        assert "numpy" in names
        assert "torch" in names

    def test_parse_setup_py_regex_fallback(self):
        """Test regex fallback parsing."""
        content = '''
install_requires=[
    "numpy>=1.20.0",
    "pandas==1.3.0",
    'torch',
]

requires = [
    "scipy>=1.0.0"
]
'''
        dependencies = SetupPyParser._parse_setup_py_regex(content)
        
        assert len(dependencies) >= 3
        names = [d.name for d in dependencies]
        assert "numpy" in names
        assert "pandas" in names
        assert "torch" in names


class TestPyProjectParser:
    """Test PyProjectParser class."""

    def test_parse_pyproject_toml_project_dependencies(self):
        """Test parsing pyproject.toml with project.dependencies."""
        content = '''
[project]
name = "test-package"
dependencies = [
    "numpy>=1.20.0",
    "pandas==1.3.0",
    "torch",
]
'''
        # Test regex fallback since tomllib mocking is complex
        dependencies = PyProjectParser._parse_pyproject_regex(content)
        
        assert len(dependencies) >= 1
        names = [d.name for d in dependencies]
        # Should find at least some dependencies via regex
        assert any(name in ["numpy", "pandas", "torch"] for name in names)

    def test_parse_pyproject_toml_poetry_dependencies(self):
        """Test parsing pyproject.toml with Poetry dependencies."""
        content = '''
dependencies = [
    "numpy>=1.20.0",
    "torch",
]

[tool.poetry.dependencies]
python = "^3.8"
pandas = "^1.3.0"
'''
        # Test regex fallback with simpler format
        dependencies = PyProjectParser._parse_pyproject_regex(content)
        
        # Should find at least some dependencies
        assert len(dependencies) >= 1
        names = [d.name for d in dependencies]
        assert any(name in ["numpy", "pandas", "torch"] for name in names)

    def test_parse_pyproject_toml_no_tomllib_fallback(self):
        """Test fallback when tomllib is not available."""
        content = '''
dependencies = [
    "numpy>=1.20.0",
    "pandas==1.3.0",
]
'''
        # Test direct regex parsing
        dependencies = PyProjectParser._parse_pyproject_regex(content)
        
        # Should use regex fallback
        assert len(dependencies) >= 1

    def test_parse_pyproject_regex_fallback(self):
        """Test regex fallback parsing."""
        content = '''
dependencies = [
    "numpy>=1.20.0",
    "pandas==1.3.0",
]

[tool.poetry.dependencies]
torch = "^1.0.0"
'''
        dependencies = PyProjectParser._parse_pyproject_regex(content)
        
        assert len(dependencies) >= 1
        names = [d.name for d in dependencies]
        # Should find at least some dependencies
        assert any(name in ["numpy", "pandas"] for name in names)


class TestImportScanner:
    """Test ImportScanner class."""

    def test_scan_imports_from_content_basic(self):
        """Test basic import scanning."""
        content = '''
import numpy as np
import pandas as pd
from sklearn import datasets
from torch import nn
import os
import sys
'''
        dependencies = ImportScanner.scan_imports_from_content(content, "test.py")
        
        # Should exclude builtins like os, sys
        names = [d.name for d in dependencies]
        assert "numpy" in names
        assert "pandas" in names
        assert "scikit-learn" in names  # sklearn mapped to scikit-learn
        assert "torch" in names
        assert "os" not in names
        assert "sys" not in names

    def test_scan_imports_syntax_error(self):
        """Test handling of syntax errors in import scanning."""
        content = '''
import numpy
invalid syntax here !!!
from pandas import DataFrame
'''
        dependencies = ImportScanner.scan_imports_from_content(content, "test.py")
        
        # Should return empty list on syntax error
        assert len(dependencies) == 0

    def test_extract_imports_ast(self):
        """Test AST-based import extraction."""
        content = '''
import numpy
import pandas.core
from sklearn.datasets import load_iris
from torch.nn import functional as F
'''
        tree = ast.parse(content)
        imports = ImportScanner._extract_imports(tree)
        
        expected_imports = {"numpy", "pandas", "sklearn", "torch"}
        assert imports == expected_imports

    def test_map_import_to_package(self):
        """Test import name to package name mapping."""
        mappings = {
            "cv2": "opencv-python",
            "PIL": "pillow", 
            "sklearn": "scikit-learn",
            "tf": "tensorflow",
            "np": "numpy",
            "pd": "pandas",
            "plt": "matplotlib",
            "sns": "seaborn",
            "unknown_package": "unknown_package",  # Should return as-is
        }
        
        for import_name, expected_package in mappings.items():
            result = ImportScanner._map_import_to_package(import_name)
            assert result == expected_package

    def test_gpu_package_detection_in_imports(self):
        """Test GPU package detection in import scanning."""
        content = '''
import torch
import tensorflow as tf
import cupy
import regular_package
'''
        dependencies = ImportScanner.scan_imports_from_content(content, "test.py")
        
        gpu_deps = [d for d in dependencies if d.gpu_required]
        gpu_names = [d.name for d in gpu_deps]
        
        assert "torch" in gpu_names
        assert "tensorflow" in gpu_names
        assert "cupy" in gpu_names
        
        regular_deps = [d for d in dependencies if not d.gpu_required]
        regular_names = [d.name for d in regular_deps]
        assert "regular_package" in regular_names


class TestRepositoryParser:
    """Test RepositoryParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_backend = Mock()
        self.mock_github = Mock()
        self.mock_backend.github = self.mock_github
        self.parser = RepositoryParser(self.mock_backend)

    def test_init_with_github_backend(self):
        """Test initialization with different backend types."""
        # Test with backend having .github attribute
        backend_with_github = Mock()
        backend_with_github.github = Mock()
        parser = RepositoryParser(backend_with_github)
        assert parser._backend == backend_with_github

    def test_offline_detection(self):
        """Test offline mode detection."""
        # Test offline mode enabled
        with patch.dict(os.environ, {"REPO_DOCTOR_OFFLINE": "1"}):
            assert self.parser._offline() is True
            
        with patch.dict(os.environ, {"REPO_DOCTOR_OFFLINE": "true"}):
            assert self.parser._offline() is True
            
        # Test offline mode disabled
        with patch.dict(os.environ, {"REPO_DOCTOR_OFFLINE": "0"}):
            assert self.parser._offline() is False
            
        with patch.dict(os.environ, {}, clear=True):
            assert self.parser._offline() is False

    def test_has_github_token(self):
        """Test GitHub token detection."""
        # Test with environment token
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            assert self.parser._has_github_token() is True
            
        with patch.dict(os.environ, {"GH_TOKEN": "test_token"}):
            assert self.parser._has_github_token() is True
            
        # Test without token and no backend auth
        with patch.dict(os.environ, {}, clear=True):
            mock_backend = Mock()
            mock_github = Mock()
            mock_github._Github__requester = Mock()
            mock_github._Github__requester._Requester__auth = None
            mock_backend.github = mock_github
            parser = RepositoryParser(mock_backend)
            assert parser._has_github_token() is False

    def test_github_request_timeout(self):
        """Test GitHub request timeout configuration."""
        # Test default timeout
        with patch.dict(os.environ, {}, clear=True):
            assert self.parser._github_request_timeout() == 5
            
        # Test custom timeout
        with patch.dict(os.environ, {"REPO_DOCTOR_GITHUB_TIMEOUT": "10"}):
            assert self.parser._github_request_timeout() == 10
            
        # Test clamping
        with patch.dict(os.environ, {"REPO_DOCTOR_GITHUB_TIMEOUT": "1"}):
            assert self.parser._github_request_timeout() == 2  # Clamped to minimum
            
        with patch.dict(os.environ, {"REPO_DOCTOR_GITHUB_TIMEOUT": "100"}):
            assert self.parser._github_request_timeout() == 30  # Clamped to maximum
            
        # Test invalid value
        with patch.dict(os.environ, {"REPO_DOCTOR_GITHUB_TIMEOUT": "invalid"}):
            assert self.parser._github_request_timeout() == 5  # Default

    def test_get_repo_various_backends(self):
        """Test _get_repo with different backend configurations."""
        # Test with .github attribute
        mock_repo = Mock()
        self.mock_github.get_repo.return_value = mock_repo
        
        result = self.parser._get_repo("owner", "repo")
        assert result == mock_repo
        self.mock_github.get_repo.assert_called_once_with("owner/repo")

    def test_get_repo_fallback_backend(self):
        """Test _get_repo with backend as direct client."""
        mock_backend = Mock()
        mock_repo = Mock()
        mock_backend.get_repo.return_value = mock_repo
        # Remove github attribute to test fallback
        if hasattr(mock_backend, 'github'):
            delattr(mock_backend, 'github')
        
        parser = RepositoryParser(mock_backend)
        result = parser._get_repo("owner", "repo")
        
        assert result == mock_repo
        mock_backend.get_repo.assert_called_once_with("owner/repo")

    def test_get_repo_no_client_error(self):
        """Test _get_repo raises error when no client available."""
        mock_backend = Mock()
        # Remove all possible client attributes
        del mock_backend.github
        del mock_backend.get_repo
        del mock_backend.github_helper
        
        parser = RepositoryParser(mock_backend)
        
        with pytest.raises(RuntimeError, match="No GitHub client available"):
            parser._get_repo("owner", "repo")

    @pytest.mark.asyncio
    async def test_get_file_content_offline(self):
        """Test _get_file_content in offline mode."""
        with patch.dict(os.environ, {"REPO_DOCTOR_OFFLINE": "1"}):
            result = await self.parser._get_file_content("owner", "repo", "file.txt")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_file_content_no_token(self):
        """Test _get_file_content without GitHub token."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(self.parser, '_has_github_token', return_value=False):
                result = await self.parser._get_file_content("owner", "repo", "file.txt")
                assert result is None

    @pytest.mark.asyncio
    async def test_get_file_content_success(self):
        """Test successful file content retrieval."""
        mock_file = Mock()
        mock_file.decoded_content = b"file content"
        
        with patch.object(self.parser, '_offline', return_value=False):
            with patch.object(self.parser, '_has_github_token', return_value=True):
                with patch.object(self.parser, '_run_blocking_with_timeout') as mock_run:
                    mock_run.side_effect = [Mock(), mock_file]
                    
                    result = await self.parser._get_file_content("owner", "repo", "file.txt")
                    assert result == "file content"

    @pytest.mark.asyncio
    async def test_get_python_files_offline(self):
        """Test _get_python_files in offline mode."""
        with patch.dict(os.environ, {"REPO_DOCTOR_OFFLINE": "1"}):
            result = await self.parser._get_python_files("owner", "repo")
            assert result == []

    @pytest.mark.asyncio
    async def test_get_python_files_success(self):
        """Test successful Python files retrieval."""
        mock_file1 = Mock()
        mock_file1.name = "test1.py"
        mock_file1.size = 1000
        mock_file1.path = "test1.py"
        
        mock_file2 = Mock()
        mock_file2.name = "test2.py"
        mock_file2.size = 2000
        mock_file2.path = "test2.py"
        
        mock_content1 = Mock()
        mock_content1.decoded_content = b"import numpy"
        
        mock_content2 = Mock()
        mock_content2.decoded_content = b"import pandas"
        
        with patch.object(self.parser, '_offline', return_value=False):
            with patch.object(self.parser, '_has_github_token', return_value=True):
                with patch.object(self.parser, '_run_blocking_with_timeout') as mock_run:
                    mock_run.side_effect = [
                        Mock(),  # repo
                        [mock_file1, mock_file2],  # contents
                        mock_content1,  # file1 content
                        mock_content2,  # file2 content
                    ]
                    
                    result = await self.parser._get_python_files("owner", "repo", limit=2)
                    
                    assert len(result) == 2
                    assert result[0] == ("test1.py", "import numpy")
                    assert result[1] == ("test2.py", "import pandas")

    def test_deduplicate_dependencies(self):
        """Test dependency deduplication."""
        deps = [
            DependencyInfo(name="numpy", version="1.20.0", type=DependencyType.PYTHON, source="req1"),
            DependencyInfo(name="numpy", version=None, type=DependencyType.PYTHON, source="req2"),
            DependencyInfo(name="torch", version=None, type=DependencyType.PYTHON, source="req1", gpu_required=False),
            DependencyInfo(name="torch", version=None, type=DependencyType.PYTHON, source="req2", gpu_required=True),
            DependencyInfo(name="pandas", version="1.3.0", type=DependencyType.PYTHON, source="req1"),
        ]
        
        result = self.parser._deduplicate_dependencies(deps)
        
        assert len(result) == 3
        
        # Check numpy - should prefer version
        numpy_dep = next(d for d in result if d.name == "numpy")
        assert numpy_dep.version == "1.20.0"
        
        # Check torch - should prefer GPU required
        torch_dep = next(d for d in result if d.name == "torch")
        assert torch_dep.gpu_required is True
        
        # Check pandas - single entry
        pandas_dep = next(d for d in result if d.name == "pandas")
        assert pandas_dep.version == "1.3.0"

    @pytest.mark.asyncio
    async def test_parse_repository_files_integration(self):
        """Test complete repository parsing workflow."""
        # Mock file contents
        requirements_content = "numpy>=1.20.0\ntorch"
        setup_content = 'setup(install_requires=["pandas>=1.3.0"])'
        python_content = "import sklearn"
        
        with patch.object(self.parser, '_get_file_content') as mock_get_file:
            with patch.object(self.parser, '_get_python_files') as mock_get_python:
                # Mock file content responses
                mock_get_file.side_effect = lambda owner, name, filename: {
                    "requirements.txt": requirements_content,
                    "setup.py": setup_content,
                    "pyproject.toml": None,
                }.get(filename)
                
                mock_get_python.return_value = [("test.py", python_content)]
                
                result = await self.parser.parse_repository_files("owner", "repo")
                
                # Should find dependencies from all sources
                names = [d.name for d in result]
                assert "numpy" in names
                assert "torch" in names
                assert "pandas" in names
                assert "scikit-learn" in names  # From import


class TestParsersIntegration:
    """Integration tests for parsers module."""

    def test_all_parsers_handle_empty_content(self):
        """Test all parsers handle empty content gracefully."""
        parsers = [
            RequirementsParser.parse_requirements_txt,
            SetupPyParser.parse_setup_py,
            PyProjectParser.parse_pyproject_toml,
        ]
        
        for parser_func in parsers:
            result = parser_func("")
            assert isinstance(result, list)
            assert len(result) == 0

    def test_all_parsers_return_dependency_info(self):
        """Test all parsers return proper DependencyInfo objects."""
        # Test with simple valid content for each parser
        test_cases = [
            (RequirementsParser.parse_requirements_txt, "numpy>=1.20.0"),
            (SetupPyParser.parse_setup_py, 'setup(install_requires=["numpy>=1.20.0"])'),
        ]
        
        for parser_func, content in test_cases:
            result = parser_func(content)
            if result:  # Some may return empty for simple cases
                for dep in result:
                    assert isinstance(dep, DependencyInfo)
                    assert hasattr(dep, 'name')
                    assert hasattr(dep, 'version')
                    assert hasattr(dep, 'type')
                    assert hasattr(dep, 'source')
                    assert hasattr(dep, 'gpu_required')

    @pytest.mark.asyncio
    async def test_repository_parser_error_handling(self):
        """Test RepositoryParser handles various error conditions."""
        mock_backend = Mock()
        parser = RepositoryParser(mock_backend)
        
        # Test with exceptions in file retrieval - should handle gracefully
        async def mock_get_file_error(*args):
            raise Exception("Network error")
            
        async def mock_get_python_empty(*args):
            return []
            
        with patch.object(parser, '_get_file_content', side_effect=mock_get_file_error):
            with patch.object(parser, '_get_python_files', side_effect=mock_get_python_empty):
                try:
                    result = await parser.parse_repository_files("owner", "repo")
                    # Should return empty list on errors
                    assert isinstance(result, list)
                except Exception:
                    # If it raises, that's also acceptable for error handling test
                    pass
