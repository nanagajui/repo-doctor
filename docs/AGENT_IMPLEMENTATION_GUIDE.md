# Repo Doctor Agent Implementation Guide

## Overview

This guide provides detailed implementation instructions for working with the Repo Doctor agent system. It covers the three core agents (Profile, Analysis, Resolution), their contracts, data flow, and best practices for development and testing.

## Table of Contents

1. [Agent Architecture](#agent-architecture)
2. [Profile Agent Implementation](#profile-agent-implementation)
3. [Analysis Agent Implementation](#analysis-agent-implementation)
4. [Resolution Agent Implementation](#resolution-agent-implementation)
5. [Knowledge System Integration](#knowledge-system-integration)
6. [Contract Validation](#contract-validation)
7. [Error Handling](#error-handling)
8. [Performance Monitoring](#performance-monitoring)
9. [Testing Guidelines](#testing-guidelines)
10. [Development Best Practices](#development-best-practices)

## Agent Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    Repo Doctor Agent System                    │
├─────────────────────────────────────────────────────────────────┤
│  Profile Agent  │  Analysis Agent  │  Resolution Agent         │
│                 │                  │                           │
│ • SystemProfile │ • Analysis       │ • Resolution              │
│ • HardwareInfo  │ • RepositoryInfo │ • Strategy                │
│ • SoftwareStack │ • DependencyInfo │ • GeneratedFile           │
│ • GPUInfo       │ • Compatibility  │ • ValidationResult        │
└─────────────────────────────────────────────────────────────────┘
│                           Knowledge Base                        │
│ • Pattern Storage    • Success Tracking    • Learning System   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Profile Agent** → System capabilities and hardware detection
2. **Analysis Agent** → Repository analysis with system context
3. **Resolution Agent** → Solution generation with analysis context
4. **Knowledge Base** → Learning and pattern storage

## Profile Agent Implementation

### Purpose
The Profile Agent detects and profiles the current system's capabilities, including hardware, software, and container runtime availability.

### Key Responsibilities
- Hardware detection (CPU, memory, GPU, architecture)
- Software stack detection (Python, pip, conda, docker, git, CUDA)
- Container runtime detection (Docker, Podman)
- Compute capability scoring (0-100)

### Implementation Example

```python
from repo_doctor.agents.profile import ProfileAgent
from repo_doctor.agents.contracts import AgentContractValidator

# Initialize agent
agent = ProfileAgent()

# Generate system profile
profile = agent.profile()

# Validate against contracts
AgentContractValidator.validate_system_profile(profile)

# Access system capabilities
print(f"GPU Available: {profile.has_gpu()}")
print(f"CUDA Version: {profile.software.cuda_version}")
print(f"Compute Score: {profile.compute_score}")
```

### Data Contract

**Input**: None (self-contained)

**Output**: `SystemProfile`
```python
class SystemProfile(BaseModel):
    hardware: HardwareInfo
    software: SoftwareStack
    platform: str
    container_runtime: Optional[str]
    compute_score: float
```

### Performance Target
- **Target Time**: < 2 seconds
- **GPU Detection**: < 10 seconds timeout
- **Software Detection**: < 5 seconds timeout per command

### Error Handling
- Graceful degradation with fallback values
- Log warnings for detection failures
- Never raise exceptions that would stop the analysis pipeline

## Analysis Agent Implementation

### Purpose
The Analysis Agent analyzes GitHub repositories to extract dependencies, detect compatibility issues, and understand requirements.

### Key Responsibilities
- Repository metadata extraction
- Multi-source dependency parsing (requirements.txt, setup.py, pyproject.toml)
- Compatibility issue detection
- Documentation analysis (README, CI/CD configs)
- GPU/CUDA requirement detection

### Implementation Example

```python
from repo_doctor.agents.analysis import AnalysisAgent
from repo_doctor.agents.contracts import AgentContractValidator

# Initialize agent
agent = AnalysisAgent(github_token="your_token")

# Analyze repository
analysis = await agent.analyze(
    repo_url="https://github.com/huggingface/transformers",
    system_profile=profile  # Optional system context
)

# Validate against contracts
AgentContractValidator.validate_analysis(analysis)

# Access analysis results
print(f"Dependencies: {len(analysis.dependencies)}")
print(f"Critical Issues: {len(analysis.get_critical_issues())}")
print(f"Confidence Score: {analysis.confidence_score}")
```

### Data Contract

**Input**: 
- `repo_url`: str - GitHub repository URL
- `system_profile`: Optional[SystemProfile] - System context

**Output**: `Analysis`
```python
class Analysis(BaseModel):
    repository: RepositoryInfo
    dependencies: List[DependencyInfo]
    python_version_required: Optional[str]
    cuda_version_required: Optional[str]
    min_memory_gb: float
    min_gpu_memory_gb: float
    compatibility_issues: List[CompatibilityIssue]
    analysis_time: float
    confidence_score: float
```

### Performance Target
- **Target Time**: < 10 seconds total
- **Repository Analysis**: < 5 seconds
- **Dependency Parsing**: < 3 seconds
- **Documentation Analysis**: < 2 seconds

### Error Handling
- Continue analysis even if some components fail
- Record analysis errors as compatibility issues
- Provide partial results rather than failing completely

## Resolution Agent Implementation

### Purpose
The Resolution Agent generates working solutions for repository compatibility issues using various strategies (Docker, Conda, Venv).

### Key Responsibilities
- Strategy selection based on analysis and system capabilities
- Solution generation (environment files, scripts, configurations)
- Container validation and testing
- LLM-enhanced solution generation for complex cases

### Implementation Example

```python
from repo_doctor.agents.resolution import ResolutionAgent
from repo_doctor.agents.contracts import AgentContractValidator

# Initialize agent
agent = ResolutionAgent()

# Generate resolution
resolution = await agent.resolve(
    analysis=analysis,
    preferred_strategy="docker"  # Optional preference
)

# Validate against contracts
AgentContractValidator.validate_resolution(resolution)

# Access resolution results
print(f"Strategy: {resolution.strategy.type}")
print(f"Generated Files: {len(resolution.generated_files)}")
print(f"Setup Commands: {resolution.setup_commands}")

# Validate solution (optional)
validation_result = await agent.validate_solution(
    resolution=resolution,
    analysis=analysis,
    timeout=300
)
```

### Data Contract

**Input**:
- `analysis`: Analysis - Complete repository analysis
- `preferred_strategy`: Optional[str] - User preference

**Output**: `Resolution`
```python
class Resolution(BaseModel):
    strategy: Strategy
    generated_files: List[GeneratedFile]
    setup_commands: List[str]
    validation_result: Optional[ValidationResult]
    instructions: str
    estimated_size_mb: int
```

### Performance Target
- **Strategy Selection**: < 1 second
- **File Generation**: < 2 seconds
- **Validation**: < 300 seconds (5 minutes) timeout

### Error Handling
- Try multiple strategies if preferred strategy fails
- Provide fallback solutions for complex cases
- Use LLM assistance for edge cases when available

## Knowledge System Integration

### Purpose
The Knowledge Base learns from analysis and resolution outcomes to improve future recommendations.

### Key Features
- Pattern storage and retrieval
- Success/failure tracking
- Similarity matching
- Learning from outcomes

### Implementation Example

```python
from repo_doctor.knowledge import KnowledgeBase
from repo_doctor.agents.contracts import AgentDataFlow

# Initialize knowledge base
kb = KnowledgeBase(Path.home() / ".repo-doctor" / "knowledge")

# Record analysis
commit_hash = kb.record_analysis(analysis)

# Record outcome
kb.record_outcome(analysis, resolution, validation_result)

# Get similar analyses
similar = kb.get_similar_analyses(analysis, limit=5)

# Get success patterns
patterns = kb.get_success_patterns("docker")

# Get knowledge context
context = kb.get_knowledge_context(analysis, resolution)
```

### Data Storage Structure
```
knowledge/
├── repos/
│   └── owner/
│       └── name/
│           ├── analyses/
│           │   └── {commit_hash}.json
│           └── solutions/
│               ├── successful/
│               │   └── {solution_id}.json
│               └── failed/
│                   └── {solution_id}.json
├── patterns/
│   ├── proven_fixes.json
│   └── common_failures.json
└── compatibility/
    ├── cuda_matrix.json
    └── python_matrix.json
```

## Contract Validation

### Purpose
Contract validation ensures data integrity and consistency across the agent system.

### Validation Functions

```python
from repo_doctor.agents.contracts import AgentContractValidator

# Validate system profile
AgentContractValidator.validate_system_profile(profile)

# Validate analysis
AgentContractValidator.validate_analysis(analysis)

# Validate resolution
AgentContractValidator.validate_resolution(resolution)
```

### Validation Rules

#### SystemProfile Validation
- CPU cores > 0
- Memory > 0
- Architecture in ["x86_64", "arm64", "unknown"]
- Python version != "unknown"
- Compute score in [0, 100]

#### Analysis Validation
- Repository name, owner, URL present
- Valid dependency types
- Valid compatibility issue severities
- Confidence score in [0.0, 1.0]
- Analysis time >= 0

#### Resolution Validation
- Valid strategy type
- Generated files have content
- Setup commands are non-empty
- Instructions are provided
- Estimated size >= 0

## Error Handling

### Purpose
Standardized error handling ensures graceful degradation and consistent behavior across agents.

### Error Handler Usage

```python
from repo_doctor.agents.contracts import AgentErrorHandler

# Profile agent error handling
try:
    profile = agent.profile()
except Exception as e:
    profile = AgentErrorHandler.handle_profile_error(e, "gpu_detection")

# Analysis agent error handling
try:
    analysis = await agent.analyze(repo_url)
except Exception as e:
    analysis = AgentErrorHandler.handle_analysis_error(e, repo_url, "github_api")

# Resolution agent error handling
try:
    resolution = await agent.resolve(analysis)
except Exception as e:
    AgentErrorHandler.handle_resolution_error(e, analysis, "strategy_selection")
```

### Error Categories
- **Hardware Detection Failure**: Use fallback values
- **Repository Access Failure**: Return minimal RepositoryInfo
- **Strategy Selection Failure**: Try LLM fallback, then raise ValueError
- **File Generation Failure**: Raise ValueError with specific error message

## Performance Monitoring

### Purpose
Performance monitoring ensures agents meet their timing contracts and provides insights for optimization.

### Performance Monitor Usage

```python
from repo_doctor.agents.contracts import AgentPerformanceMonitor

monitor = AgentPerformanceMonitor()

# Check performance
if not monitor.check_profile_performance(duration):
    print(f"Profile agent exceeded target: {duration:.2f}s")

# Get performance report
report = monitor.get_performance_report("analysis_agent", duration)
print(f"Performance ratio: {report['performance_ratio']:.2f}")
```

### Performance Targets
- **Profile Agent**: 2.0 seconds
- **Analysis Agent**: 10.0 seconds
- **Resolution Agent**: 5.0 seconds (excluding validation)

## Testing Guidelines

### Unit Testing

```python
import pytest
from unittest.mock import Mock, patch
from repo_doctor.agents.contracts import AgentContractValidator

def test_profile_agent_contract_compliance():
    """Test that ProfileAgent follows contracts."""
    with patch('repo_doctor.agents.profile.psutil') as mock_psutil:
        mock_psutil.cpu_count.return_value = 4
        mock_psutil.virtual_memory.return_value.total = 8 * 1024**3
        
        agent = ProfileAgent()
        profile = agent.profile()
        
        # Should be valid according to contracts
        AgentContractValidator.validate_system_profile(profile)

def test_analysis_agent_contract_compliance():
    """Test that AnalysisAgent follows contracts."""
    with patch('repo_doctor.agents.analysis.GitHubHelper') as mock_github:
        mock_github.return_value.parse_repo_url.return_value = {
            "owner": "test", "name": "repo"
        }
        
        agent = AnalysisAgent()
        analysis = asyncio.run(agent.analyze("https://github.com/test/repo"))
        
        # Should be valid according to contracts
        AgentContractValidator.validate_analysis(analysis)
```

### Integration Testing

```python
def test_agent_data_flow_integration():
    """Test complete data flow between agents."""
    # Profile to Analysis context
    profile_context = AgentDataFlow.profile_to_analysis_context(profile)
    assert profile_context["system_capabilities"]["has_gpu"] is True
    
    # Analysis to Resolution context
    analysis_context = AgentDataFlow.analysis_to_resolution_context(analysis)
    assert analysis_context["repository"]["name"] == "repo"
    
    # Resolution to Knowledge context
    knowledge_context = AgentDataFlow.resolution_to_knowledge_context(resolution, analysis)
    assert knowledge_context["repository_key"] == "test/repo"
```

### Test Categories
- **Contract Validation Tests**: Verify all validation functions
- **Data Flow Tests**: Test agent-to-agent data transfer
- **Error Handling Tests**: Test all error scenarios
- **Performance Tests**: Verify timing contracts
- **Integration Tests**: Test complete workflows

## Development Best Practices

### 1. Always Validate Data
```python
# Validate input data before processing
AgentContractValidator.validate_analysis(analysis)

# Validate output data after processing
AgentContractValidator.validate_resolution(resolution)
```

### 2. Use Consistent Error Handling
```python
# Use standardized error handlers
try:
    result = agent.process(data)
except Exception as e:
    result = AgentErrorHandler.handle_error(e, context)
```

### 3. Monitor Performance
```python
# Check performance against contracts
start_time = time.time()
result = agent.process(data)
duration = time.time() - start_time

if not monitor.check_performance(duration):
    logger.warning(f"Agent exceeded performance target: {duration:.2f}s")
```

### 4. Log Significant Events
```python
import logging

logger = logging.getLogger(__name__)

# Log analysis start
logger.info(f"Starting analysis for {repo_url}")

# Log analysis completion
logger.info(f"Analysis completed in {analysis_time:.2f}s with confidence {confidence_score:.2f}")
```

### 5. Test All Error Scenarios
```python
def test_error_scenarios():
    """Test all possible error scenarios."""
    # Test hardware detection failure
    with patch('psutil.cpu_count', side_effect=Exception("Hardware detection failed")):
        profile = agent.profile()
        assert profile.hardware.cpu_cores == 1  # Fallback value
    
    # Test GitHub API failure
    with patch('github.Github.get_repo', side_effect=Exception("API failed")):
        analysis = asyncio.run(agent.analyze("https://github.com/test/repo"))
        assert len(analysis.compatibility_issues) > 0  # Should have error issue
```

### 6. Document Any Contract Changes
```python
# When adding new fields to models, update contracts
class Analysis(BaseModel):
    # ... existing fields ...
    new_field: Optional[str] = None  # Added in v1.2.0

# Update validation accordingly
def validate_analysis(analysis: Analysis) -> bool:
    # ... existing validations ...
    # New field validation
    if analysis.new_field is not None:
        assert len(analysis.new_field) > 0, "New field cannot be empty"
```

### 7. Use Type Hints
```python
from typing import List, Optional, Dict, Any

def process_dependencies(
    dependencies: List[DependencyInfo],
    system_context: Optional[Dict[str, Any]] = None
) -> List[CompatibilityIssue]:
    """Process dependencies with proper type hints."""
    # Implementation
```

### 8. Follow Async Patterns
```python
# Use async for I/O operations
async def analyze_repository(self, repo_url: str) -> Analysis:
    """Analyze repository asynchronously."""
    # Parallel analysis
    results = await asyncio.gather(
        self._analyze_dependencies(repo_info),
        self._check_dockerfiles(repo_info),
        self._scan_documentation(repo_info),
        return_exceptions=True,
    )
```

## Troubleshooting

### Common Issues

#### 1. Contract Validation Failures
```python
# Check validation error details
try:
    AgentContractValidator.validate_analysis(analysis)
except ValueError as e:
    print(f"Validation failed: {e}")
    # Fix the issue and retry
```

#### 2. Performance Issues
```python
# Monitor performance and identify bottlenecks
report = monitor.get_performance_report("analysis_agent", duration)
if report["performance_ratio"] > 1.5:
    print("Performance issue detected, consider optimization")
```

#### 3. Error Handling Issues
```python
# Check error handler behavior
try:
    result = agent.process(data)
except Exception as e:
    # Verify error handler provides fallback
    fallback = AgentErrorHandler.handle_error(e, context)
    assert fallback is not None
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug output
agent = ProfileAgent()
profile = agent.profile()  # Will show detailed debug info
```

## Conclusion

This implementation guide provides comprehensive instructions for working with the Repo Doctor agent system. By following these guidelines, developers can ensure consistent behavior, proper error handling, and maintainable code across all agents.

For additional information, refer to:
- [Agent Contracts Documentation](AGENT_CONTRACTS.md)
- [API Reference](../repo_doctor/agents/)
- [Test Examples](../tests/test_agent_contracts.py)
