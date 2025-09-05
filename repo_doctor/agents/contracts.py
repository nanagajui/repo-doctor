"""Agent contract validation and data flow utilities."""

import logging
from typing import Dict, List, Optional

from ..models.analysis import Analysis, DependencyType
from ..models.resolution import Resolution, StrategyType, ValidationStatus
from ..models.system import SystemProfile
from ..utils.logging_config import get_logger


class AgentContractValidator:
    """Validates agent contracts and data flow with enhanced monitoring."""

    @staticmethod
    def validate_system_profile(profile: SystemProfile) -> bool:
        """Validate system profile completeness and consistency."""
        try:
            # Hardware info must be present
            assert profile.hardware.cpu_cores > 0, "CPU cores must be > 0"
            assert profile.hardware.memory_gb > 0, "Memory must be > 0"
            assert profile.hardware.architecture in [
                "x86_64", "arm64", "unknown"
            ], f"Invalid architecture: {profile.hardware.architecture}"
            
            # Software stack must have Python version
            assert profile.software.python_version != "unknown", "Python version must be detected"
            
            # Compute score must be in valid range
            assert 0 <= profile.compute_score <= 100, f"Invalid compute score: {profile.compute_score}"
            
            return True
        except AssertionError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"SystemProfile validation failed: {e}")
            raise ValueError(f"SystemProfile validation failed: {e}")

    @staticmethod
    def validate_analysis(analysis: Analysis) -> bool:
        """Validate analysis completeness and consistency."""
        try:
            # Repository info must be complete
            assert analysis.repository.name, "Repository name is required"
            assert analysis.repository.owner, "Repository owner is required"
            assert analysis.repository.url, "Repository URL is required"
            
            # Dependencies must have valid types
            for dep in analysis.dependencies:
                assert dep.type in [
                    DependencyType.PYTHON, DependencyType.CONDA, 
                    DependencyType.SYSTEM, DependencyType.GPU
                ], f"Invalid dependency type: {dep.type}"
                assert dep.name, "Dependency name is required"
                assert dep.source, "Dependency source is required"
            
            # Compatibility issues must have severity
            for issue in analysis.compatibility_issues:
                assert issue.severity in ["critical", "warning", "info"], \
                    f"Invalid severity: {issue.severity}"
                assert issue.message, "Compatibility issue message is required"
                assert issue.component, "Compatibility issue component is required"
            
            # Confidence score must be valid
            assert 0.0 <= analysis.confidence_score <= 1.0, \
                f"Invalid confidence score: {analysis.confidence_score}"
            
            # Analysis time must be non-negative
            assert analysis.analysis_time >= 0, f"Invalid analysis time: {analysis.analysis_time}"
            
            return True
        except AssertionError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Analysis validation failed: {e}")
            raise ValueError(f"Analysis validation failed: {e}")

    @staticmethod
    def validate_resolution(resolution: Resolution) -> bool:
        """Validate resolution completeness and consistency."""
        try:
            # Strategy must be valid
            assert resolution.strategy.type in [
                StrategyType.DOCKER, StrategyType.CONDA, 
                StrategyType.VENV, StrategyType.DEVCONTAINER
            ], f"Invalid strategy type: {resolution.strategy.type}"
            
            # Generated files must have content
            for file in resolution.generated_files:
                assert file.path, "Generated file path is required"
                assert file.content, "Generated file content is required"
                assert file.description, "Generated file description is required"
            
            # Setup commands must be non-empty if present
            for cmd in resolution.setup_commands:
                assert cmd.strip(), "Setup command cannot be empty"
            
            # Instructions must be provided
            assert resolution.instructions.strip(), "Instructions are required"
            
            # Estimated size must be non-negative
            assert resolution.estimated_size_mb >= 0, \
                f"Invalid estimated size: {resolution.estimated_size_mb}"
            
            return True
        except AssertionError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Resolution validation failed: {e}")
            raise ValueError(f"Resolution validation failed: {e}")


class AgentDataFlow:
    """Manages data flow between agents."""

    @staticmethod
    def profile_to_analysis_context(profile: SystemProfile) -> dict:
        """Convert SystemProfile to analysis context."""
        return {
            "system_capabilities": {
                "has_gpu": profile.has_gpu(),
                "has_cuda": profile.has_cuda(),
                "can_run_containers": profile.can_run_containers(),
                "compute_score": profile.compute_score,
                "gpu_memory_gb": sum(gpu.memory_gb for gpu in profile.hardware.gpus),
                "cuda_version": profile.software.cuda_version,
            },
            "hardware": {
                "cpu_cores": profile.hardware.cpu_cores,
                "memory_gb": profile.hardware.memory_gb,
                "architecture": profile.hardware.architecture,
            },
            "software": {
                "python_version": profile.software.python_version,
                "container_runtime": profile.container_runtime,
            }
        }

    @staticmethod
    def analysis_to_resolution_context(analysis: Analysis) -> dict:
        """Convert Analysis to resolution context."""
        return {
            "repository": {
                "name": analysis.repository.name,
                "owner": analysis.repository.owner,
                "language": analysis.repository.language,
                "has_dockerfile": analysis.repository.has_dockerfile,
                "has_conda_env": analysis.repository.has_conda_env,
            },
            "requirements": {
                "python_version": analysis.python_version_required,
                "cuda_version": analysis.cuda_version_required,
                "min_memory_gb": analysis.min_memory_gb,
                "min_gpu_memory_gb": analysis.min_gpu_memory_gb,
                "gpu_required": analysis.is_gpu_required(),
            },
            "dependencies": [
                {
                    "name": dep.name,
                    "version": dep.version,
                    "type": dep.type.value,
                    "gpu_required": dep.gpu_required,
                }
                for dep in analysis.dependencies
            ],
            "compatibility_issues": [
                {
                    "type": issue.type,
                    "severity": issue.severity,
                    "message": issue.message,
                    "component": issue.component,
                }
                for issue in analysis.compatibility_issues
            ],
            "confidence_score": analysis.confidence_score,
        }

    @staticmethod
    def resolution_to_knowledge_context(resolution: Resolution, analysis: Analysis) -> dict:
        """Convert Resolution and Analysis to knowledge base context."""
        return {
            "repository_key": f"{analysis.repository.owner}/{analysis.repository.name}",
            "strategy_used": resolution.strategy.type.value,
            "success": resolution.is_validated(),
            "files_generated": len(resolution.generated_files),
            "setup_commands": len(resolution.setup_commands),
            "estimated_size_mb": resolution.estimated_size_mb,
            "validation_result": resolution.validation_result.model_dump() if resolution.validation_result else None,
            "analysis_confidence": analysis.confidence_score,
            "compatibility_issues_count": len(analysis.compatibility_issues),
            "critical_issues_count": len(analysis.get_critical_issues()),
        }


class AgentErrorHandler:
    """Standardized error handling for agents."""
    
    logger = get_logger(__name__)

    @staticmethod
    def handle_profile_error(error: Exception, context: str = "") -> SystemProfile:
        """Handle Profile Agent errors with fallback values."""
        from ..models.system import GPUInfo, HardwareInfo, SoftwareStack
        
        # Log the error
        AgentErrorHandler.logger.error(f"Profile Agent error in {context}: {error}", exc_info=error)
        
        # Return minimal valid profile
        return SystemProfile(
            hardware=HardwareInfo(
                cpu_cores=1,
                memory_gb=4.0,
                gpus=[],
                architecture="unknown"
            ),
            software=SoftwareStack(
                python_version="unknown",
                pip_version=None,
                conda_version=None,
                docker_version=None,
                git_version=None,
                cuda_version=None,
            ),
            platform="unknown",
            container_runtime=None,
            compute_score=0.0
        )

    @staticmethod
    def handle_analysis_error(error: Exception, repo_url: str, context: str = "") -> Analysis:
        """Handle Analysis Agent errors with fallback values."""
        from ..models.analysis import RepositoryInfo, CompatibilityIssue
        
        # Log the error
        AgentErrorHandler.logger.error(f"Analysis Agent error in {context}: {error}", exc_info=error)
        
        # Extract basic info from URL
        try:
            import re
            match = re.match(r"https://github\.com/([^/]+)/([^/]+)", repo_url)
            if match:
                owner, name = match.groups()
            else:
                owner, name = "unknown", "unknown"
        except Exception:
            owner, name = "unknown", "unknown"
        
        # Return minimal valid analysis
        return Analysis(
            repository=RepositoryInfo(
                url=repo_url,
                name=name,
                owner=owner,
                description=None,
                stars=0,
                language=None,
                topics=[],
            ),
            dependencies=[],
            python_version_required=None,
            cuda_version_required=None,
            min_memory_gb=0.0,
            min_gpu_memory_gb=0.0,
            compatibility_issues=[
                CompatibilityIssue(
                    type="analysis_error",
                    severity="warning",
                    message=f"Analysis failed: {str(error)}",
                    component="analyzer",
                )
            ],
            analysis_time=0.0,
            confidence_score=0.0,
        )

    @staticmethod
    def handle_resolution_error(error: Exception, analysis: Analysis, context: str = "") -> None:
        """Handle Resolution Agent errors by raising with context."""
        # Log the error
        AgentErrorHandler.logger.error(f"Resolution Agent error in {context}: {error}", exc_info=error)
        
        # Re-raise with additional context
        raise ValueError(
            f"Failed to generate resolution for {analysis.repository.owner}/{analysis.repository.name}: {error}"
        ) from error


class AgentPerformanceMonitor:
    """Monitor agent performance against contracts."""

    def __init__(self, config = None):
        from ..utils.config import Config
        config = config or Config.load()
        
        self.performance_targets = {
            "profile_agent": config.advanced.profile_agent_timeout,
            "analysis_agent": config.advanced.analysis_agent_timeout,
            "resolution_agent": config.advanced.resolution_agent_timeout,
        }

    def check_profile_performance(self, duration: float) -> bool:
        """Check if profile agent meets performance contract."""
        return duration <= self.performance_targets["profile_agent"]

    def check_analysis_performance(self, duration: float) -> bool:
        """Check if analysis agent meets performance contract."""
        return duration <= self.performance_targets["analysis_agent"]

    def check_resolution_performance(self, duration: float) -> bool:
        """Check if resolution agent meets performance contract."""
        return duration <= self.performance_targets["resolution_agent"]

    def get_performance_report(self, agent_name: str, duration: float) -> dict:
        """Get performance report for an agent."""
        target = self.performance_targets.get(agent_name, 0)
        return {
            "agent": agent_name,
            "duration": duration,
            "target": target,
            "meets_target": duration <= target,
            "performance_ratio": duration / target if target > 0 else float('inf'),
        }


class AgentHealthMonitor:
    """Monitor agent health and system status."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_checks = {}
    
    def check_agent_health(self, agent_name: str, status: str, details: Optional[Dict] = None) -> Dict:
        """Check and record agent health status."""
        health_status = {
            "agent": agent_name,
            "status": status,  # "healthy", "degraded", "unhealthy"
            "timestamp": self._get_timestamp(),
            "details": details or {},
        }
        
        self.health_checks[agent_name] = health_status
        
        if status == "unhealthy":
            self.logger.error(f"Agent {agent_name} is unhealthy: {details}")
        elif status == "degraded":
            self.logger.warning(f"Agent {agent_name} is degraded: {details}")
        else:
            self.logger.info(f"Agent {agent_name} is healthy")
            
        return health_status
    
    def get_system_health(self) -> Dict:
        """Get overall system health status."""
        if not self.health_checks:
            return {"status": "unknown", "agents": {}}
        
        agent_statuses = [check["status"] for check in self.health_checks.values()]
        
        if "unhealthy" in agent_statuses:
            overall_status = "unhealthy"
        elif "degraded" in agent_statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
            
        return {
            "status": overall_status,
            "agents": self.health_checks,
            "summary": {
                "healthy": agent_statuses.count("healthy"),
                "degraded": agent_statuses.count("degraded"),
                "unhealthy": agent_statuses.count("unhealthy"),
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
