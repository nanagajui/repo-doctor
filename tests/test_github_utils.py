"""Tests for GitHub utilities."""

import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import pytest
from github import Github, GithubException, RateLimitExceededException
from requests.exceptions import RequestException

from repo_doctor.utils.github import GitHubHelper, retry_on_rate_limit


class TestGitHubHelper:
    """Test GitHubHelper class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.helper = GitHubHelper()

    @patch('repo_doctor.utils.github.EnvLoader.get_github_token')
    def test_init_with_token(self, mock_get_token):
        """Test GitHubHelper initialization with token."""
        mock_get_token.return_value = "test_token"
        helper = GitHubHelper()
        assert helper.token == "test_token"

    @patch('repo_doctor.utils.github.EnvLoader.get_github_token')
    def test_init_without_token(self, mock_get_token):
        """Test GitHubHelper initialization without token."""
        mock_get_token.return_value = None
        helper = GitHubHelper()
        assert helper.token is None

    def test_init_with_explicit_token(self):
        """Test GitHubHelper initialization with explicit token."""
        helper = GitHubHelper(token="explicit_token")
        assert helper.token == "explicit_token"

    def test_parse_repo_url_standard(self):
        """Test parsing standard GitHub URL."""
        url = "https://github.com/owner/repo"
        result = self.helper.parse_repo_url(url)
        
        assert result["owner"] == "owner"
        assert result["name"] == "repo"
        assert result["full_name"] == "owner/repo"

    def test_parse_repo_url_with_git_suffix(self):
        """Test parsing GitHub URL with .git suffix."""
        url = "https://github.com/owner/repo.git"
        result = self.helper.parse_repo_url(url)
        
        assert result["owner"] == "owner"
        assert result["name"] == "repo"
        assert result["full_name"] == "owner/repo"

    def test_parse_repo_url_with_trailing_slash(self):
        """Test parsing GitHub URL with trailing slash."""
        url = "https://github.com/owner/repo/"
        result = self.helper.parse_repo_url(url)
        
        assert result["owner"] == "owner"
        assert result["name"] == "repo"
        assert result["full_name"] == "owner/repo"

    def test_parse_repo_url_with_path(self):
        """Test parsing GitHub URL with additional path."""
        url = "https://github.com/owner/repo/tree/main"
        result = self.helper.parse_repo_url(url)
        
        assert result["owner"] == "owner"
        assert result["name"] == "repo"
        assert result["full_name"] == "owner/repo"

    def test_parse_repo_url_invalid(self):
        """Test parsing invalid GitHub URL."""
        with pytest.raises(ValueError, match="Invalid GitHub URL format"):
            self.helper.parse_repo_url("https://invalid.com/repo")

    @patch('repo_doctor.utils.github.Github')
    def test_check_rate_limit_success(self, mock_github_class):
        """Test successful rate limit check."""
        # Mock the rate limit response
        mock_rate_limit = Mock()
        mock_rate_limit.rate.remaining = 4500
        mock_rate_limit.rate.limit = 5000
        mock_rate_limit.rate.reset = datetime.now(timezone.utc)
        
        mock_github = Mock()
        mock_github.get_rate_limit.return_value = mock_rate_limit
        mock_github_class.return_value = mock_github
        
        helper = GitHubHelper(token="test_token")
        result = helper.check_rate_limit()
        
        assert result["remaining"] == 4500
        assert result["limit"] == 5000
        assert result["used"] == 500
        assert result["percentage_used"] == 10.0

    @patch('repo_doctor.utils.github.Github')
    def test_check_rate_limit_error(self, mock_github_class):
        """Test rate limit check with error."""
        mock_github = Mock()
        mock_github.get_rate_limit.side_effect = Exception("API Error")
        mock_github_class.return_value = mock_github
        
        helper = GitHubHelper(token="test_token")
        result = helper.check_rate_limit()
        
        assert result["remaining"] == 0
        assert result["limit"] == 5000  # With token
        assert result["reset"] is None

    def test_get_rate_limit_status_ok(self):
        """Test rate limit status when OK."""
        with patch.object(self.helper, 'check_rate_limit') as mock_check:
            mock_check.return_value = {
                "remaining": 1000,
                "limit": 5000,
                "reset": datetime.now(timezone.utc),
                "reset_in_seconds": 3600,
                "used": 4000,
                "percentage_used": 80.0
            }
            
            status = self.helper.get_rate_limit_status()
            assert "🟢 OK" in status
            assert "1000/5000" in status

    def test_get_rate_limit_status_warning(self):
        """Test rate limit status when warning."""
        with patch.object(self.helper, 'check_rate_limit') as mock_check:
            mock_check.return_value = {
                "remaining": 200,
                "limit": 5000,
                "reset": datetime.now(timezone.utc),
                "reset_in_seconds": 3600,
                "used": 4800,
                "percentage_used": 96.0
            }
            
            status = self.helper.get_rate_limit_status()
            assert "🟡 WARNING" in status
            assert "200/5000" in status

    def test_get_rate_limit_status_critical(self):
        """Test rate limit status when critical."""
        with patch.object(self.helper, 'check_rate_limit') as mock_check:
            mock_check.return_value = {
                "remaining": 50,
                "limit": 5000,
                "reset": datetime.now(timezone.utc),
                "reset_in_seconds": 3600,
                "used": 4950,
                "percentage_used": 99.0
            }
            
            status = self.helper.get_rate_limit_status()
            assert "🔴 CRITICAL" in status
            assert "50/5000" in status

    def test_warn_if_rate_limit_low(self):
        """Test warning when rate limit is low."""
        with patch.object(self.helper, 'check_rate_limit') as mock_check:
            with patch.object(self.helper.logger, 'warning') as mock_warning:
                mock_check.return_value = {
                    "remaining": 50,
                    "limit": 5000,
                    "reset_in_seconds": 3600
                }
                
                self.helper.warn_if_rate_limit_low(threshold=100)
                mock_warning.assert_called_once()

    @patch('repo_doctor.utils.github.Github')
    def test_get_repo_info_success(self, mock_github_class):
        """Test successful repository info retrieval."""
        # Mock repository object
        mock_repo = Mock()
        mock_repo.name = "test-repo"
        mock_repo.full_name = "owner/test-repo"
        mock_repo.description = "Test repository"
        mock_repo.language = "Python"
        mock_repo.stargazers_count = 100
        mock_repo.forks_count = 20
        mock_repo.get_topics.return_value = ["python", "testing"]
        mock_repo.default_branch = "main"
        mock_repo.created_at = datetime.now(timezone.utc)
        mock_repo.updated_at = datetime.now(timezone.utc)
        mock_repo.size = 1024
        mock_repo.open_issues_count = 5
        mock_repo.has_issues = True
        mock_repo.has_wiki = False
        mock_repo.has_pages = False
        mock_repo.archived = False
        mock_repo.disabled = False
        mock_repo.private = False
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        helper = GitHubHelper(token="test_token")
        result = helper.get_repo_info("owner", "test-repo")
        
        assert result["name"] == "test-repo"
        assert result["full_name"] == "owner/test-repo"
        assert result["description"] == "Test repository"
        assert result["language"] == "Python"
        assert result["stars"] == 100
        assert result["forks"] == 20
        assert result["topics"] == ["python", "testing"]

    @patch('repo_doctor.utils.github.Github')
    def test_get_repo_info_not_found(self, mock_github_class):
        """Test repository info retrieval when repo not found."""
        mock_github = Mock()
        mock_github.get_repo.side_effect = GithubException(404, "Not Found", {})
        mock_github_class.return_value = mock_github
        
        helper = GitHubHelper(token="test_token")
        result = helper.get_repo_info("owner", "nonexistent")
        
        assert result is None

    @patch('repo_doctor.utils.github.Github')
    def test_get_repo_info_access_denied(self, mock_github_class):
        """Test repository info retrieval when access denied."""
        mock_github = Mock()
        mock_github.get_repo.side_effect = GithubException(403, "Forbidden", {})
        mock_github_class.return_value = mock_github
        
        helper = GitHubHelper(token="test_token")
        result = helper.get_repo_info("owner", "private-repo")
        
        assert result is None

    @patch('repo_doctor.utils.github.Github')
    def test_get_file_content_success(self, mock_github_class):
        """Test successful file content retrieval."""
        mock_file_content = Mock()
        mock_file_content.decoded_content = b"print('Hello, World!')"
        
        mock_repo = Mock()
        mock_repo.get_contents.return_value = mock_file_content
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        helper = GitHubHelper(token="test_token")
        result = helper.get_file_content("owner", "repo", "main.py")
        
        assert result == "print('Hello, World!')"

    @patch('repo_doctor.utils.github.Github')
    def test_get_file_content_not_found(self, mock_github_class):
        """Test file content retrieval when file not found."""
        mock_repo = Mock()
        mock_repo.get_contents.side_effect = GithubException(404, "Not Found", {})
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        helper = GitHubHelper(token="test_token")
        result = helper.get_file_content("owner", "repo", "nonexistent.py")
        
        assert result is None

    @patch('repo_doctor.utils.github.Github')
    def test_get_file_content_branch_fallback(self, mock_github_class):
        """Test file content retrieval with branch fallback."""
        mock_file_content = Mock()
        mock_file_content.decoded_content = b"content"
        
        mock_repo = Mock()
        # First call (main branch) fails, second call (master branch) succeeds
        mock_repo.get_contents.side_effect = [
            GithubException(404, "Not Found", {}),
            mock_file_content
        ]
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        helper = GitHubHelper(token="test_token")
        result = helper.get_file_content("owner", "repo", "file.txt", branch="main")
        
        assert result == "content"

    @patch('repo_doctor.utils.github.Github')
    def test_list_files_success(self, mock_github_class):
        """Test successful file listing."""
        mock_file1 = Mock()
        mock_file1.name = "file1.py"
        mock_file1.path = "src/file1.py"
        mock_file1.type = "file"
        mock_file1.size = 1024
        mock_file1.download_url = "https://raw.githubusercontent.com/owner/repo/main/src/file1.py"
        
        mock_file2 = Mock()
        mock_file2.name = "file2.py"
        mock_file2.path = "src/file2.py"
        mock_file2.type = "file"
        mock_file2.size = 2048
        mock_file2.download_url = "https://raw.githubusercontent.com/owner/repo/main/src/file2.py"
        
        mock_repo = Mock()
        mock_repo.get_contents.return_value = [mock_file1, mock_file2]
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        helper = GitHubHelper(token="test_token")
        result = helper.list_files("owner", "repo", "src")
        
        assert len(result) == 2
        assert result[0]["name"] == "file1.py"
        assert result[0]["path"] == "src/file1.py"
        assert result[1]["name"] == "file2.py"
        assert result[1]["size"] == 2048

    @patch('repo_doctor.utils.github.Github')
    def test_list_files_single_file(self, mock_github_class):
        """Test file listing when single file returned."""
        mock_file = Mock()
        mock_file.name = "single.py"
        mock_file.path = "single.py"
        mock_file.type = "file"
        mock_file.size = 512
        mock_file.download_url = "https://raw.githubusercontent.com/owner/repo/main/single.py"
        
        mock_repo = Mock()
        mock_repo.get_contents.return_value = mock_file  # Single file, not list
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        mock_github_class.return_value = mock_github
        
        helper = GitHubHelper(token="test_token")
        result = helper.list_files("owner", "repo", "")
        
        assert len(result) == 1
        assert result[0]["name"] == "single.py"

    def test_check_file_exists_true(self):
        """Test file existence check when file exists."""
        with patch.object(self.helper, 'get_file_content') as mock_get_content:
            mock_get_content.return_value = "file content"
            
            result = self.helper.check_file_exists("owner", "repo", "existing.py")
            assert result is True

    def test_check_file_exists_false(self):
        """Test file existence check when file doesn't exist."""
        with patch.object(self.helper, 'get_file_content') as mock_get_content:
            mock_get_content.return_value = None
            
            result = self.helper.check_file_exists("owner", "repo", "nonexistent.py")
            assert result is False

    def test_get_requirements_files(self):
        """Test getting requirements files."""
        with patch.object(self.helper, 'get_file_content') as mock_get_content:
            def mock_content(owner, repo, filename):
                if filename == "requirements.txt":
                    return "numpy==1.21.0\npandas==1.3.0"
                elif filename == "setup.py":
                    return "from setuptools import setup"
                return None
            
            mock_get_content.side_effect = mock_content
            
            result = self.helper.get_requirements_files("owner", "repo")
            
            assert result["requirements.txt"] == "numpy==1.21.0\npandas==1.3.0"
            assert result["setup.py"] == "from setuptools import setup"
            assert result["pyproject.toml"] is None

    def test_get_docker_files(self):
        """Test getting Docker files."""
        with patch.object(self.helper, 'get_file_content') as mock_get_content:
            def mock_content(owner, repo, filename):
                if filename == "Dockerfile":
                    return "FROM python:3.9"
                elif filename == "docker-compose.yml":
                    return "version: '3.8'"
                return None
            
            mock_get_content.side_effect = mock_content
            
            result = self.helper.get_docker_files("owner", "repo")
            
            assert result["Dockerfile"] == "FROM python:3.9"
            assert result["docker-compose.yml"] == "version: '3.8'"
            assert result[".dockerignore"] is None

    def test_get_ci_configs(self):
        """Test getting CI configuration files."""
        with patch.object(self.helper, 'get_file_content') as mock_get_content:
            with patch.object(self.helper, 'list_files') as mock_list_files:
                # Mock workflow files
                mock_list_files.return_value = [
                    {"name": "ci.yml", "path": ".github/workflows/ci.yml"},
                    {"name": "deploy.yaml", "path": ".github/workflows/deploy.yaml"}
                ]
                
                def mock_content(owner, repo, filename):
                    if filename == ".github/workflows/ci.yml":
                        return "name: CI"
                    elif filename == ".github/workflows/deploy.yaml":
                        return "name: Deploy"
                    elif filename == ".travis.yml":
                        return "language: python"
                    return None
                
                mock_get_content.side_effect = mock_content
                
                result = self.helper.get_ci_configs("owner", "repo")
                
                assert isinstance(result[".github/workflows"], dict)
                assert result[".github/workflows"]["ci.yml"] == "name: CI"
                assert result[".github/workflows"]["deploy.yaml"] == "name: Deploy"
                assert result[".travis.yml"] == "language: python"

    def test_get_readme_content(self):
        """Test getting README content."""
        with patch.object(self.helper, 'get_file_content') as mock_get_content:
            def mock_content(owner, repo, filename):
                if filename == "README.md":
                    return "# Test Repository"
                return None
            
            mock_get_content.side_effect = mock_content
            
            result = self.helper.get_readme_content("owner", "repo")
            assert result == "# Test Repository"

    def test_get_readme_content_fallback(self):
        """Test getting README content with fallback to different formats."""
        with patch.object(self.helper, 'get_file_content') as mock_get_content:
            def mock_content(owner, repo, filename):
                if filename == "README.rst":
                    return "Test Repository\n==============="
                return None
            
            mock_get_content.side_effect = mock_content
            
            result = self.helper.get_readme_content("owner", "repo")
            assert result == "Test Repository\n==============="


class TestRetryDecorator:
    """Test retry_on_rate_limit decorator."""

    def test_retry_decorator_success(self):
        """Test retry decorator with successful call."""
        mock_self = Mock()
        mock_self.logger = Mock()
        
        @retry_on_rate_limit(max_retries=2, base_delay=0.1)
        def test_method(self):
            return "success"
        
        result = test_method(mock_self)
        assert result == "success"

    def test_retry_decorator_rate_limit_exceeded(self):
        """Test retry decorator with rate limit exceeded."""
        mock_self = Mock()
        mock_self.logger = Mock()
        
        call_count = 0
        
        @retry_on_rate_limit(max_retries=2, base_delay=0.1)
        def test_method(self):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Mock rate limit exception
                mock_rate_limit = Mock()
                mock_rate_limit.reset = datetime.now(timezone.utc)
                exc = RateLimitExceededException(429, "Rate limit exceeded", {})
                exc.rate_limit = mock_rate_limit
                raise exc
            return "success"
        
        result = test_method(mock_self)
        assert result == "success"
        assert call_count == 3

    def test_retry_decorator_max_retries_exceeded(self):
        """Test retry decorator when max retries exceeded."""
        mock_self = Mock()
        mock_self.logger = Mock()
        
        @retry_on_rate_limit(max_retries=1, base_delay=0.1)
        def test_method(self):
            mock_rate_limit = Mock()
            mock_rate_limit.reset = datetime.now(timezone.utc)
            exc = RateLimitExceededException(429, "Rate limit exceeded", {})
            exc.rate_limit = mock_rate_limit
            raise exc
        
        with pytest.raises(RateLimitExceededException):
            test_method(mock_self)

    def test_retry_decorator_non_rate_limit_error(self):
        """Test retry decorator with non-rate-limit error."""
        mock_self = Mock()
        mock_self.logger = Mock()
        
        @retry_on_rate_limit(max_retries=2, base_delay=0.1)
        def test_method(self):
            raise GithubException(500, "Server Error", {})
        
        with pytest.raises(GithubException):
            test_method(mock_self)

    def test_retry_decorator_request_exception_rate_limit(self):
        """Test retry decorator with RequestException containing rate limit."""
        mock_self = Mock()
        mock_self.logger = Mock()
        
        call_count = 0
        
        @retry_on_rate_limit(max_retries=1, base_delay=0.1)
        def test_method(self):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                exc = RequestException("rate limit exceeded")
                exc.status = 403
                raise exc
            return "success"
        
        result = test_method(mock_self)
        assert result == "success"
        assert call_count == 2


class TestGitHubHelperIntegration:
    """Integration tests for GitHubHelper."""

    def test_full_workflow_mock(self):
        """Test full workflow with mocked GitHub API."""
        with patch('repo_doctor.utils.github.Github') as mock_github_class:
            # Mock repository
            mock_repo = Mock()
            mock_repo.name = "test-repo"
            mock_repo.full_name = "owner/test-repo"
            mock_repo.description = "Test repository"
            mock_repo.language = "Python"
            mock_repo.stargazers_count = 100
            mock_repo.forks_count = 20
            mock_repo.get_topics.return_value = ["python"]
            mock_repo.default_branch = "main"
            mock_repo.created_at = datetime.now(timezone.utc)
            mock_repo.updated_at = datetime.now(timezone.utc)
            mock_repo.size = 1024
            mock_repo.open_issues_count = 5
            mock_repo.has_issues = True
            mock_repo.has_wiki = False
            mock_repo.has_pages = False
            mock_repo.archived = False
            mock_repo.disabled = False
            mock_repo.private = False
            
            # Mock file content
            mock_file_content = Mock()
            mock_file_content.decoded_content = b"numpy==1.21.0"
            mock_repo.get_contents.return_value = mock_file_content
            
            mock_github = Mock()
            mock_github.get_repo.return_value = mock_repo
            mock_github_class.return_value = mock_github
            
            helper = GitHubHelper(token="test_token")
            
            # Test URL parsing
            url_info = helper.parse_repo_url("https://github.com/owner/test-repo")
            assert url_info["owner"] == "owner"
            assert url_info["name"] == "test-repo"
            
            # Test repo info
            repo_info = helper.get_repo_info("owner", "test-repo")
            assert repo_info["name"] == "test-repo"
            assert repo_info["language"] == "Python"
            
            # Test file content
            content = helper.get_file_content("owner", "test-repo", "requirements.txt")
            assert content == "numpy==1.21.0"
            
            # Test file existence
            exists = helper.check_file_exists("owner", "test-repo", "requirements.txt")
            assert exists is True
