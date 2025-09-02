"""Container validation for testing generated solutions."""

import docker
import time
from typing import Optional, List
from ..models.resolution import ValidationResult, ValidationStatus


class ContainerValidator:
    """Validator for Docker-based solutions."""
    
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
        except docker.errors.DockerException:
            self.docker_client = None
    
    def validate_docker_solution(self, dockerfile_path: str, 
                                timeout: int = 300) -> ValidationResult:
        """Validate a Docker-based solution."""
        if not self.docker_client:
            return ValidationResult(
                status=ValidationStatus.FAILED,
                error_message="Docker client not available"
            )
        
        start_time = time.time()
        logs = []
        container_id = None
        
        try:
            # Build the image
            logs.append("Building Docker image...")
            image, build_logs = self.docker_client.images.build(
                path=str(dockerfile_path).rsplit('/', 1)[0],
                tag="repo-doctor-test",
                rm=True
            )
            
            for log in build_logs:
                if 'stream' in log:
                    logs.append(log['stream'].strip())
            
            # Run a test container
            logs.append("Starting test container...")
            container = self.docker_client.containers.run(
                image.id,
                command="python --version",
                detach=True,
                remove=True
            )
            
            container_id = container.id
            
            # Wait for container to finish
            result = container.wait(timeout=timeout)
            
            if result['StatusCode'] == 0:
                logs.append("Container test successful")
                status = ValidationStatus.SUCCESS
                error_message = None
            else:
                logs.append(f"Container exited with code {result['StatusCode']}")
                status = ValidationStatus.FAILED
                error_message = f"Container failed with exit code {result['StatusCode']}"
            
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
            container_id=container_id
        )
