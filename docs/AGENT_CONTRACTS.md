# Repo Doctor Agent Contracts

## Overview

This document defines the clear data contracts and interfaces between the three core agents in the Repo Doctor system: Profile Agent, Analysis Agent, and Resolution Agent. These contracts ensure consistent data flow, proper validation, and clear expectations for each agent's responsibilities.

## Agent Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Profile Agent  │───▶│ Analysis Agent  │───▶│Resolution Agent │
│                 │    │                 │    │                 │
│ SystemProfile   │    │ Analysis        │    │ Resolution      │
│ HardwareInfo    │    │ RepositoryInfo  │    │ Strategy        │
│ SoftwareStack   │    │ DependencyInfo  │    │ GeneratedFile   │
│ GPUInfo         │    │ Compatibility   │    │ ValidationResult│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 1. Profile Agent Contract

### Input
- **None** (self-contained system profiling)

### Output: SystemProfile
```python
class SystemProfile(BaseModel):
    hardware: HardwareInfo
    software: SoftwareStack
    platform: str
    container_runtime: Optional[str]
    compute_score: float
```

### Key Responsibilities
1. **Hardware Detection**: CPU cores, memory, GPU detection, architecture
2. **Software Stack**: Python version, pip/conda/docker/git versions, CUDA detection
3. **Container Runtime**: Docker/Podman availability detection
4. **Compute Scoring**: Overall system capability assessment (0-100)

### Data Quality Requirements
- All hardware info must be detected or have fallback values
- GPU detection must include memory, driver version, and CUDA compatibility
- Software versions must be extracted from actual command outputs
- Compute score must be calculated consistently across systems

### Error Handling
- Graceful degradation with fallback values
- Log warnings for detection failures
- Never raise exceptions that would stop the analysis pipeline

## 2. Analysis Agent Contract

### Input
- **repo_url**: str - GitHub repository URL
- **SystemProfile**: Optional - For system-aware analysis

### Output: Analysis
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

### Key Responsibilities
1. **Repository Parsing**: Extract metadata, topics, language, stars
2. **Dependency Analysis**: Multi-source parsing (requirements.txt, setup.py, pyproject.toml)
3. **Compatibility Detection**: Version conflicts, CUDA requirements, GPU needs
4. **Documentation Analysis**: README parsing for requirements and setup instructions
5. **CI/CD Analysis**: GitHub Actions, Travis, CircleCI configuration parsing

### Data Quality Requirements
- Dependencies must include source file, version constraints, and GPU requirements
- Compatibility issues must be categorized by severity (critical, warning, info)
- Confidence score must reflect analysis completeness (0.0-1.0)
- Analysis time must be accurately measured

### Error Handling
- Continue analysis even if some components fail
- Record analysis errors as compatibility issues
- Provide partial results rather than failing completely

## 3. Resolution Agent Contract

### Input
- **Analysis**: Complete repository analysis
- **preferred_strategy**: Optional[str] - User preference (docker, conda, venv)

### Output: Resolution
```python
class Resolution(BaseModel):
    strategy: Strategy
    generated_files: List[GeneratedFile]
    setup_commands: List[str]
    validation_result: Optional[ValidationResult]
    instructions: str
    estimated_size_mb: int
```

### Key Responsibilities
1. **Strategy Selection**: Choose best strategy based on analysis and system capabilities
2. **Solution Generation**: Create environment files, scripts, and configurations
3. **Validation**: Test generated solutions in containers
4. **Documentation**: Provide clear setup instructions and troubleshooting

### Data Quality Requirements
- Generated files must be complete and executable
- Setup commands must be tested and validated
- Instructions must be clear and actionable
- Validation results must include detailed logs and error messages

### Error Handling
- Try multiple strategies if preferred strategy fails
- Provide fallback solutions for complex cases
- Use LLM assistance for edge cases when available

## 4. Inter-Agent Data Flow

### Profile → Analysis
- **SystemProfile** provides system context for analysis
- GPU capabilities inform CUDA requirement detection
- Container runtime availability affects strategy recommendations

### Analysis → Resolution
- **Analysis** provides complete repository understanding
- Dependencies inform strategy selection and file generation
- Compatibility issues guide solution approach
- Confidence score affects validation thoroughness

### Resolution → Knowledge Base
- **Resolution** outcomes are recorded for learning
- Success/failure patterns are extracted and stored
- Similar solutions are identified for future recommendations

## 5. Data Validation Contracts

### SystemProfile Validation
```python
def validate_system_profile(profile: SystemProfile) -> bool:
    """Validate system profile completeness and consistency."""
    # Hardware info must be present
    assert profile.hardware.cpu_cores > 0
    assert profile.hardware.memory_gb > 0
    assert profile.hardware.architecture in ["x86_64", "arm64", "unknown"]
    
    # Software stack must have Python version
    assert profile.software.python_version != "unknown"
    
    # Compute score must be in valid range
    assert 0 <= profile.compute_score <= 100
    
    return True
```

### Analysis Validation
```python
def validate_analysis(analysis: Analysis) -> bool:
    """Validate analysis completeness and consistency."""
    # Repository info must be complete
    assert analysis.repository.name
    assert analysis.repository.owner
    assert analysis.repository.url
    
    # Dependencies must have valid types
    for dep in analysis.dependencies:
        assert dep.type in [DependencyType.PYTHON, DependencyType.CONDA, 
                           DependencyType.SYSTEM, DependencyType.GPU]
        assert dep.name
        assert dep.source
    
    # Compatibility issues must have severity
    for issue in analysis.compatibility_issues:
        assert issue.severity in ["critical", "warning", "info"]
        assert issue.message
        assert issue.component
    
    # Confidence score must be valid
    assert 0.0 <= analysis.confidence_score <= 1.0
    
    return True
```

### Resolution Validation
```python
def validate_resolution(resolution: Resolution) -> bool:
    """Validate resolution completeness and consistency."""
    # Strategy must be valid
    assert resolution.strategy.type in [StrategyType.DOCKER, StrategyType.CONDA, 
                                       StrategyType.VENV, StrategyType.DEVCONTAINER]
    
    # Generated files must have content
    for file in resolution.generated_files:
        assert file.path
        assert file.content
        assert file.description
    
    # Setup commands must be non-empty if present
    for cmd in resolution.setup_commands:
        assert cmd.strip()
    
    # Instructions must be provided
    assert resolution.instructions.strip()
    
    return True
```

## 6. Error Handling Contracts

### Profile Agent Errors
- **Hardware Detection Failure**: Use fallback values, log warning
- **GPU Detection Failure**: Continue without GPU info, log warning
- **Software Version Failure**: Use "unknown" version, log warning

### Analysis Agent Errors
- **Repository Access Failure**: Return minimal RepositoryInfo, add compatibility issue
- **Dependency Parsing Failure**: Return empty dependencies list, add compatibility issue
- **GitHub API Failure**: Use URL parsing fallback, log warning

### Resolution Agent Errors
- **Strategy Selection Failure**: Try LLM fallback, then raise ValueError
- **File Generation Failure**: Raise ValueError with specific error message
- **Validation Failure**: Return ValidationResult with error details

## 7. Performance Contracts

### Profile Agent
- **Target Time**: < 2 seconds
- **GPU Detection**: < 10 seconds timeout
- **Software Detection**: < 5 seconds timeout per command

### Analysis Agent
- **Target Time**: < 10 seconds total
- **Repository Analysis**: < 5 seconds
- **Dependency Parsing**: < 3 seconds
- **Documentation Analysis**: < 2 seconds

### Resolution Agent
- **Strategy Selection**: < 1 second
- **File Generation**: < 2 seconds
- **Validation**: < 300 seconds (5 minutes) timeout

## 8. Knowledge Base Integration

### Data Storage Contracts
- **Analysis Storage**: Store complete Analysis objects with timestamps
- **Resolution Storage**: Store Resolution and ValidationResult pairs
- **Pattern Storage**: Extract and store success/failure patterns
- **Cache Management**: TTL-based cache with 7-day expiration

### Learning Contracts
- **Success Pattern Extraction**: Identify common successful configurations
- **Failure Analysis**: Categorize and store failure patterns
- **Similarity Matching**: Find similar repositories and solutions
- **Recommendation Generation**: Suggest strategies based on historical data

## 9. LLM Integration Contracts

### Analysis Agent LLM Usage
- **Documentation Enhancement**: Extract requirements from README files
- **Complex Case Analysis**: Handle edge cases not covered by rules
- **Error Diagnosis**: Analyze validation failures and suggest fixes

### Resolution Agent LLM Usage
- **Strategy Recommendation**: Suggest strategies for complex compatibility cases
- **Solution Enhancement**: Add special instructions for complex setups
- **Error Diagnosis**: Provide detailed analysis of validation failures

### LLM Error Handling
- **LLM Unavailable**: Continue without LLM assistance, log warning
- **LLM Timeout**: Use fallback logic, log warning
- **LLM Error**: Continue with standard analysis, log error

## 10. Testing Contracts

### Unit Testing Requirements
- **Profile Agent**: Test with mock system calls, verify fallback behavior
- **Analysis Agent**: Test with mock GitHub API, verify error handling
- **Resolution Agent**: Test with mock strategies, verify validation

### Integration Testing Requirements
- **End-to-End**: Test complete pipeline with real repositories
- **Error Scenarios**: Test failure modes and recovery
- **Performance**: Verify timing contracts are met

### Validation Testing Requirements
- **Data Contracts**: Verify all data validation functions
- **Error Handling**: Test all error scenarios
- **Knowledge Base**: Test storage and retrieval operations

## Implementation Guidelines

1. **Always validate input data** before processing
2. **Use consistent error handling** across all agents
3. **Log all significant events** for debugging and monitoring
4. **Provide meaningful error messages** for troubleshooting
5. **Test all error scenarios** to ensure graceful degradation
6. **Document any deviations** from these contracts
7. **Update contracts** when adding new features or changing behavior

This contract specification ensures that all agents work together seamlessly while maintaining clear boundaries and responsibilities. Any changes to these contracts should be documented and communicated to all team members.
