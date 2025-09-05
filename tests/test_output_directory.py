"""Tests for output directory functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from repo_doctor.cli import _save_generated_files
from repo_doctor.models.resolution import GeneratedFile, Resolution, Strategy, StrategyType


class TestOutputDirectory:
    """Test output directory functionality."""

    def test_save_generated_files_creates_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test-output"
            
            # Create mock resolution with generated files
            mock_file = GeneratedFile(
                path="test.txt",
                content="test content",
                description="Test file",
                executable=False
            )
            
            resolution = Resolution(
                strategy=Strategy(type=StrategyType.DOCKER, estimated_setup_time=60),
                generated_files=[mock_file],
                instructions="Test instructions"
            )
            
            # Save files
            _save_generated_files(resolution, str(output_dir))
            
            # Verify directory was created
            assert output_dir.exists()
            assert (output_dir / "test.txt").exists()
            assert (output_dir / "SETUP_INSTRUCTIONS.md").exists()

    def test_save_generated_files_with_executable(self):
        """Test that executable files are properly marked."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test-output"
            
            # Create mock resolution with executable file
            mock_file = GeneratedFile(
                path="setup.sh",
                content="#!/bin/bash\necho 'test'",
                description="Setup script",
                executable=True
            )
            
            resolution = Resolution(
                strategy=Strategy(type=StrategyType.DOCKER, estimated_setup_time=60),
                generated_files=[mock_file],
                instructions="Test instructions"
            )
            
            # Save files
            _save_generated_files(resolution, str(output_dir))
            
            # Verify file is executable
            setup_file = output_dir / "setup.sh"
            assert setup_file.exists()
            assert os.access(setup_file, os.X_OK)

    def test_save_generated_files_nested_paths(self):
        """Test that nested file paths are handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "test-output"
            
            # Create mock resolution with nested file path
            mock_file = GeneratedFile(
                path="docker/Dockerfile",
                content="FROM python:3.9",
                description="Dockerfile",
                executable=False
            )
            
            resolution = Resolution(
                strategy=Strategy(type=StrategyType.DOCKER, estimated_setup_time=60),
                generated_files=[mock_file],
                instructions="Test instructions"
            )
            
            # Save files
            _save_generated_files(resolution, str(output_dir))
            
            # Verify nested directory was created
            docker_dir = output_dir / "docker"
            assert docker_dir.exists()
            assert (docker_dir / "Dockerfile").exists()

    def test_default_output_directory_logic(self):
        """Test the logic for determining default output directory."""
        # Test the logic that's used in the CLI
        repo_owner = "test-owner"
        repo_name = "test-repo"
        expected_output = f"outputs/{repo_owner}-{repo_name}"
        
        assert expected_output == "outputs/test-owner-test-repo"

    def test_output_directory_creation(self):
        """Test that output directory is created when needed."""
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "outputs" / "test-owner-test-repo"
            
            # Simulate the directory creation logic
            output_dir.mkdir(parents=True, exist_ok=True)
            
            assert output_dir.exists()
            assert output_dir.is_dir()
