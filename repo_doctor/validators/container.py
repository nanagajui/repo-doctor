"""Container validation for testing generated solutions."""

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker

from ..models.analysis import Analysis
from ..models.resolution import Resolution, ValidationResult, ValidationStatus


class ContainerValidator:
    """Validator for Docker-based solutions."""

    def __init__(self):
        try:
            self.docker_client = docker.from_env()
            self.gpu_available = self._check_gpu_support()
        except docker.errors.DockerException:
            self.docker_client = None
            self.gpu_available = False

    def validate_docker_solution(
        self, dockerfile_path: str, timeout: int = 300
    ) -> ValidationResult:
        """Validate a Docker-based solution."""
        if not self.docker_client:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                error_message="Docker client not available",
            )

        start_time = time.time()
        logs = []
        container_id = None

        try:
            # Build the image
            logs.append("Building Docker image...")
            image, build_logs = self.docker_client.images.build(
                path=str(dockerfile_path).rsplit("/", 1)[0],
                tag="repo-doctor-test",
                rm=True,
            )

            for log in build_logs:
                if "stream" in log:
                    logs.append(log["stream"].strip())

            # Run a test container
            logs.append("Starting test container...")
            container = self.docker_client.containers.run(
                image.id, command="python --version", detach=True, remove=True
            )

            container_id = container.id

            # Wait for container to finish
            result = container.wait(timeout=timeout)

            if result["StatusCode"] == 0:
                logs.append("Container test successful")
                status = ValidationStatus.SUCCESS
                error_message = None
            else:
                logs.append(f"Container exited with code {result['StatusCode']}")
                status = ValidationStatus.FAILED
                error_message = (
                    f"Container failed with exit code {result['StatusCode']}"
                )

        except docker.errors.BuildError as e:
            logs.append(f"Build failed: {str(e)}")
            status = ValidationStatus.FAILED
            error_message = f"Docker build failed: {str(e)}"

        except docker.errors.ContainerError as e:
            logs.append(f"Container error: {str(e)}")
            status = ValidationStatus.FAILED
            error_message = f"Container execution failed: {str(e)}"

        except Exception as e:
            logs.append(f"Validation error: {str(e)}")
            status = ValidationStatus.FAILED
            error_message = f"Validation failed: {str(e)}"

        duration = time.time() - start_time

        return ValidationResult(
            status=status,
            duration=duration,
            logs=logs,
            error_message=error_message,
            container_id=container_id,
        )

    def validate_resolution(
        self, resolution: Resolution, analysis: Analysis, timeout: int = 300
    ) -> ValidationResult:
        """Validate a complete resolution with all generated files."""
        if not self.docker_client:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                error_message="Docker client not available",
            )

        # Create temporary directory for validation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write all generated files
            for generated_file in resolution.generated_files:
                file_path = temp_path / generated_file.path
                file_path.parent.mkdir(parents=True, exist_ok=True)

                with open(file_path, "w") as f:
                    f.write(generated_file.content)

                # Make executable if needed
                if generated_file.executable:
                    file_path.chmod(0o755)

            # Validate the solution
            return self._validate_in_directory(temp_path, analysis, timeout)

    def _validate_in_directory(
        self, directory: Path, analysis: Analysis, timeout: int
    ) -> ValidationResult:
        """Validate solution in a specific directory."""
        start_time = time.time()
        logs = []
        container_id = None
        image_name = f"repo-doctor-test-{int(time.time())}"

        try:
            # Check if Dockerfile exists
            dockerfile_path = directory / "Dockerfile"
            if not dockerfile_path.exists():
                return ValidationResult(
                    status=ValidationStatus.FAILED,
                    error_message="No Dockerfile found in generated files",
                )

            # Build the image
            logs.append("🐳 Building Docker image...")
            image, build_logs = self.docker_client.images.build(
                path=str(directory), tag=image_name, rm=True, pull=True
            )

            # Capture build logs
            for log in build_logs:
                if "stream" in log:
                    log_line = log["stream"].strip()
                    if log_line:
                        logs.append(f"BUILD: {log_line}")

            logs.append("✅ Image built successfully")

            # Run validation tests
            validation_tests = self._get_validation_tests(analysis)

            for test_name, test_command in validation_tests.items():
                logs.append(f"🧪 Running test: {test_name}")

                # Determine if GPU is needed for this test
                gpu_args = {}
                if (
                    analysis.is_gpu_required()
                    and self.gpu_available
                    and "gpu" in test_name.lower()
                ):
                    gpu_args = {
                        "device_requests": [
                            docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                        ]
                    }

                container = self.docker_client.containers.run(
                    image.id, command=test_command, detach=True, remove=True, **gpu_args
                )

                container_id = container.id

                # Wait for test to complete
                result = container.wait(timeout=60)  # Individual test timeout

                if result["StatusCode"] == 0:
                    logs.append(f"✅ {test_name}: PASSED")
                else:
                    # Get container logs for debugging
                    container_logs = container.logs().decode("utf-8")
                    logs.append(
                        f"❌ {test_name}: FAILED (exit code {result['StatusCode']})"
                    )
                    logs.append(f"Container logs: {container_logs[-500:]}")

                    return ValidationResult(
                        status=ValidationStatus.FAILED,
                        duration=time.time() - start_time,
                        logs=logs,
                        error_message=f"Test '{test_name}' failed with exit code {result['StatusCode']}",
                        container_id=container_id,
                    )

            # All tests passed
            logs.append("🎉 All validation tests passed!")

            return ValidationResult(
                status=ValidationStatus.SUCCESS,
                duration=time.time() - start_time,
                logs=logs,
                container_id=container_id,
            )

        except docker.errors.BuildError as e:
            logs.append(f"❌ Build failed: {str(e)}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                duration=time.time() - start_time,
                logs=logs,
                error_message=f"Docker build failed: {str(e)}",
            )

        except Exception as e:
            logs.append(f"❌ Validation error: {str(e)}")
            return ValidationResult(
                status=ValidationStatus.FAILED,
                duration=time.time() - start_time,
                logs=logs,
                error_message=f"Validation failed: {str(e)}",
            )

        finally:
            # Cleanup: remove test image
            try:
                if "image" in locals():
                    self.docker_client.images.remove(image.id, force=True)
            except Exception:
                pass

    def _check_gpu_support(self) -> bool:
        """Check if Docker has GPU support available."""
        try:
            # Try to run nvidia-smi in a container
            container = self.docker_client.containers.run(
                "nvidia/cuda:11.8-base-ubuntu20.04",
                command="nvidia-smi",
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ],
                remove=True,
                detach=True,
            )

            result = container.wait(timeout=30)
            return result["StatusCode"] == 0

        except Exception:
            return False

    def _get_validation_tests(self, analysis: Analysis) -> Dict[str, str]:
        """Get validation tests based on analysis."""
        tests = {
            "python_version": "python --version",
            "pip_functionality": "pip --version",
            "basic_imports": "python -c 'import sys; print(f\"Python {sys.version} ready\")'",
        }

        # Add dependency-specific tests
        python_deps = analysis.get_python_dependencies()

        for dep in python_deps[:5]:  # Test first 5 dependencies
            if dep.name in ["torch", "tensorflow", "jax"]:
                tests[f"import_{dep.name}"] = (
                    f"python -c 'import {dep.name}; print(f\"{dep.name} imported successfully\")'"
                )

        # Add GPU tests if GPU required
        if analysis.is_gpu_required():
            tests["gpu_nvidia_smi"] = "nvidia-smi"

            if any(dep.name == "torch" for dep in python_deps):
                tests["pytorch_gpu"] = (
                    "python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
                )

            if any(dep.name in ["tensorflow", "tensorflow-gpu"] for dep in python_deps):
                tests["tensorflow_gpu"] = (
                    'python -c \'import tensorflow as tf; print(f"GPU devices: {len(tf.config.list_physical_devices("GPU"))}")\''
                )

        return tests

    def cleanup_test_containers(self) -> int:
        """Clean up any leftover test containers."""
        if not self.docker_client:
            return 0

        cleaned = 0
        try:
            # Remove containers with repo-doctor-test prefix
            containers = self.docker_client.containers.list(all=True)
            for container in containers:
                if any("repo-doctor-test" in tag for tag in container.image.tags):
                    container.remove(force=True)
                    cleaned += 1

            # Remove images with repo-doctor-test prefix
            images = self.docker_client.images.list()
            for image in images:
                if any("repo-doctor-test" in tag for tag in image.tags):
                    self.docker_client.images.remove(image.id, force=True)
                    cleaned += 1

        except Exception:
            pass

        return cleaned
