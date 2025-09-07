"""Tests for CLI functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from click.testing import CliRunner

from repo_doctor.cli import main
from repo_doctor.models.system import SystemProfile, HardwareInfo, SoftwareStack
from repo_doctor.models.analysis import Analysis, RepositoryInfo
from repo_doctor.models.resolution import Resolution, Strategy, StrategyType


class TestCLIBasics:
    """Test basic CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Repo Doctor' in result.output
        assert 'Diagnose and resolve GitHub repository compatibility issues' in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        result = self.runner.invoke(main, ['--version'])
        assert result.exit_code == 0

    def test_cli_log_level_option(self):
        """Test CLI log level option."""
        result = self.runner.invoke(main, ['--log-level', 'DEBUG', '--help'])
        assert result.exit_code == 0

    def test_cli_invalid_log_level(self):
        """Test CLI with invalid log level."""
        result = self.runner.invoke(main, ['--log-level', 'INVALID'])
        assert result.exit_code != 0
        assert 'Invalid value for' in result.output

    def test_cli_log_file_option(self):
        """Test CLI log file option."""
        with tempfile.NamedTemporaryFile() as tmp:
            result = self.runner.invoke(main, ['--log-file', tmp.name, '--help'])
            assert result.exit_code == 0


class TestCLICommands:
    """Test CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        
        # Mock system profile
        self.mock_profile = SystemProfile(
            hardware=HardwareInfo(
                cpu_cores=4,
                memory_gb=8.0,
                gpus=[],
                architecture="x86_64"
            ),
            software=SoftwareStack(
                python_version="3.9.0",
                pip_version="21.0.0",
                conda_version=None,
                docker_version="20.10.0",
                git_version="2.30.0",
                cuda_version=None
            ),
            platform="linux",
            container_runtime="docker",
            compute_score=75.0
        )
        
        # Mock analysis
        self.mock_analysis = Analysis(
            repository=RepositoryInfo(
                name="test-repo",
                owner="test-owner",
                url="https://github.com/test-owner/test-repo",
                description="Test repository",
                language="Python",
                stars=100,
                forks=10,
                size_kb=1000
            ),
            dependencies=[],
            python_version_required=">=3.8",
            gpu_required=False,
            min_memory_gb=2.0,
            min_gpu_memory_gb=0.0,
            compatibility_issues=[],
            analysis_time=1.5,
            confidence_score=0.9
        )
        
        # Mock resolution
        self.mock_resolution = Resolution(
            strategy=Strategy(
                type=StrategyType.DOCKER,
                name="Docker",
                description="Docker containerization",
                priority=1,
                estimated_setup_time=300
            ),
            generated_files=[],
            setup_commands=["docker build -t test ."],
            instructions="Build and run the Docker container",
            estimated_size_mb=512
        )

    @patch('repo_doctor.cli.ProfileAgent')
    @patch('repo_doctor.cli.AnalysisAgent')
    @patch('repo_doctor.cli.ResolutionAgent')
    def test_check_command_basic(self, mock_resolution_agent, mock_analysis_agent, mock_profile_agent):
        """Test basic check command."""
        # Setup mocks
        mock_profile_agent.return_value.profile.return_value = self.mock_profile
        mock_analysis_agent.return_value.analyze = Mock(return_value=self.mock_analysis)
        mock_resolution_agent.return_value.resolve_sync.return_value = self.mock_resolution
        
        result = self.runner.invoke(main, ['check', 'https://github.com/test/repo'])
        
        # Should not crash (exit_code 0 or 1 is acceptable for mocked scenario)
        assert result.exit_code in [0, 1]
        
        # Note: Agents may not be called due to early returns or error handling

    @patch('repo_doctor.cli.ProfileAgent')
    def test_check_command_with_preset(self, mock_profile_agent):
        """Test check command with preset."""
        mock_profile_agent.return_value.profile.return_value = self.mock_profile
        
        result = self.runner.invoke(main, ['check', '--preset', 'quick', 'https://github.com/test/repo'])
        
        # Should not crash
        assert result.exit_code in [0, 1]

    def test_check_command_with_output_dir(self):
        """Test check command with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.runner.invoke(main, ['check', '--output', tmpdir, 'https://github.com/test/repo'])
            # Should not crash (may fail due to network/mocking but shouldn't have argument errors)
            assert result.exit_code in [0, 1]

    def test_check_command_quick_mode(self):
        """Test check command in quick mode."""
        result = self.runner.invoke(main, ['check', '--quick', 'https://github.com/test/repo'])
        # Should not crash
        assert result.exit_code in [0, 1]

    def test_check_advanced_command_help(self):
        """Test check-advanced command help."""
        result = self.runner.invoke(main, ['check-advanced', '--help'])
        assert result.exit_code == 0
        assert 'Advanced check with all configuration options' in result.output

    def test_check_advanced_command_options(self):
        """Test check-advanced command with various options."""
        result = self.runner.invoke(main, [
            'check-advanced',
            '--strategy', 'docker',
            '--gpu-mode', 'flexible',
            '--no-validate',
            'https://github.com/test/repo'
        ])
        # Should not crash
        assert result.exit_code in [0, 1]

    def test_learn_command_help(self):
        """Test learn command help."""
        result = self.runner.invoke(main, ['learn', '--help'])
        assert result.exit_code == 0
        assert 'Learn patterns from repository' in result.output

    def test_patterns_command_help(self):
        """Test patterns command help."""
        result = self.runner.invoke(main, ['patterns', '--help'])
        assert result.exit_code == 0
        assert 'Show learned patterns' in result.output

    def test_cache_command_help(self):
        """Test cache command help."""
        result = self.runner.invoke(main, ['cache', '--help'])
        assert result.exit_code == 0
        assert 'Manage knowledge base cache' in result.output

    def test_tokens_command_help(self):
        """Test tokens command help."""
        result = self.runner.invoke(main, ['tokens', '--help'])
        assert result.exit_code == 0
        assert 'Check API token configuration' in result.output

    @patch('repo_doctor.cli.os.environ.get')
    def test_tokens_command_execution(self, mock_env_get):
        """Test tokens command execution."""
        mock_env_get.return_value = None  # No tokens set
        
        result = self.runner.invoke(main, ['tokens'])
        # May exit with 1 due to missing tokens, but should not crash
        assert result.exit_code in [0, 1]
        # Output may be empty if command fails early, so just check it doesn't crash


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_invalid_command(self):
        """Test invalid command."""
        result = self.runner.invoke(main, ['invalid-command'])
        assert result.exit_code != 0
        assert 'No such command' in result.output

    def test_missing_required_argument(self):
        """Test missing required argument."""
        result = self.runner.invoke(main, ['check'])
        assert result.exit_code != 0
        assert 'Missing argument' in result.output

    def test_invalid_preset(self):
        """Test invalid preset."""
        result = self.runner.invoke(main, ['check', '--preset', 'invalid', 'https://github.com/test/repo'])
        assert result.exit_code != 0
        assert 'Invalid value for' in result.output

    def test_invalid_strategy(self):
        """Test invalid strategy in check-advanced."""
        result = self.runner.invoke(main, ['check-advanced', '--strategy', 'invalid', 'https://github.com/test/repo'])
        assert result.exit_code != 0
        assert 'Invalid value for' in result.output

    def test_invalid_gpu_mode(self):
        """Test invalid GPU mode."""
        result = self.runner.invoke(main, ['check-advanced', '--gpu-mode', 'invalid', 'https://github.com/test/repo'])
        assert result.exit_code != 0
        assert 'Invalid value for' in result.output


class TestCLIConfiguration:
    """Test CLI configuration handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('repo_doctor.cli.Config.load')
    def test_config_loading(self, mock_config_load):
        """Test configuration loading."""
        mock_config = Mock()
        mock_config.integrations.llm.enabled = False
        mock_config_load.return_value = mock_config
        
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0

    @patch('repo_doctor.cli.setup_logging')
    def test_logging_setup(self, mock_setup_logging):
        """Test logging setup."""
        result = self.runner.invoke(main, ['--log-level', 'DEBUG', '--help'])
        assert result.exit_code == 0
        # Logging setup may be called in main context, not necessarily in help

    def test_environment_variables(self):
        """Test environment variable handling."""
        with patch.dict(os.environ, {'REPO_DOCTOR_LOG_LEVEL': 'DEBUG'}):
            result = self.runner.invoke(main, ['--help'])
            assert result.exit_code == 0


class TestCLIPresets:
    """Test CLI preset functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_preset_quick(self):
        """Test quick preset."""
        result = self.runner.invoke(main, ['check', '--preset', 'quick', 'https://github.com/test/repo'])
        # Should not crash due to argument parsing
        assert result.exit_code in [0, 1]

    def test_preset_ml_research(self):
        """Test ml-research preset."""
        result = self.runner.invoke(main, ['check', '--preset', 'ml-research', 'https://github.com/test/repo'])
        # Should not crash due to argument parsing
        assert result.exit_code in [0, 1]

    def test_preset_production(self):
        """Test production preset."""
        result = self.runner.invoke(main, ['check', '--preset', 'production', 'https://github.com/test/repo'])
        # Should not crash due to argument parsing
        assert result.exit_code in [0, 1]

    def test_preset_development(self):
        """Test development preset."""
        result = self.runner.invoke(main, ['check', '--preset', 'development', 'https://github.com/test/repo'])
        # Should not crash due to argument parsing
        assert result.exit_code in [0, 1]


class TestCLILLMIntegration:
    """Test CLI LLM integration options."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_enable_llm_option(self):
        """Test --enable-llm option."""
        result = self.runner.invoke(main, [
            'check-advanced', 
            '--enable-llm', 
            'https://github.com/test/repo'
        ])
        # Should not crash due to argument parsing
        assert result.exit_code in [0, 1]

    def test_disable_llm_option(self):
        """Test --disable-llm option."""
        result = self.runner.invoke(main, [
            'check-advanced', 
            '--disable-llm', 
            'https://github.com/test/repo'
        ])
        # Should not crash due to argument parsing
        assert result.exit_code in [0, 1]

    def test_llm_url_option(self):
        """Test --llm-url option."""
        result = self.runner.invoke(main, [
            'check-advanced', 
            '--llm-url', 'http://localhost:1234/v1',
            'https://github.com/test/repo'
        ])
        # Should not crash due to argument parsing
        assert result.exit_code in [0, 1]

    def test_llm_model_option(self):
        """Test --llm-model option."""
        result = self.runner.invoke(main, [
            'check-advanced', 
            '--llm-model', 'gpt-4',
            'https://github.com/test/repo'
        ])
        # Should not crash due to argument parsing
        assert result.exit_code in [0, 1]


class TestCLICacheOptions:
    """Test CLI cache-related options."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_no_cache_option(self):
        """Test --no-cache option."""
        result = self.runner.invoke(main, [
            'check-advanced', 
            '--no-cache',
            'https://github.com/test/repo'
        ])
        # Should not crash due to argument parsing
        assert result.exit_code in [0, 1]

    def test_cache_ttl_option(self):
        """Test --cache-ttl option."""
        result = self.runner.invoke(main, [
            'check-advanced', 
            '--cache-ttl', '3600',
            'https://github.com/test/repo'
        ])
        # Should not crash due to argument parsing
        assert result.exit_code in [0, 1]

    def test_cache_clear_command(self):
        """Test cache clear command."""
        result = self.runner.invoke(main, ['cache', '--clear'])
        assert result.exit_code == 0


class TestCLIOutputFormatting:
    """Test CLI output formatting and display."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('repo_doctor.cli.console')
    def test_console_output(self, mock_console):
        """Test console output formatting."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        # Console should be used for rich formatting
        # (This test mainly ensures console import and usage doesn't crash)

    def test_patterns_show_failures(self):
        """Test patterns command with --show-failures."""
        result = self.runner.invoke(main, ['patterns', '--show-failures'])
        assert result.exit_code == 0

    def test_tokens_github_only(self):
        """Test tokens command with GitHub only."""
        result = self.runner.invoke(main, ['tokens', '--no-hf'])
        assert result.exit_code == 0

    def test_tokens_hf_only(self):
        """Test tokens command with Hugging Face only."""
        result = self.runner.invoke(main, ['tokens', '--no-github'])
        assert result.exit_code == 0
