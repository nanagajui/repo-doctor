# TEST_ISSUES.md - Machine Readable Test Status

## METADATA
- **Last Updated**: 2025-09-06T12:44:40+09:30
- **Test Environment**: Python 3.12.3
- **Test Command**: `pytest --tb=short -v --durations=10 --ignore=tests/test_learning_system.py --ignore=tests/test_real_repositories.py --ignore=tests/test_llm_with_repos.py`

## TEST_SUMMARY
```json
{
  "total_tests": 258,
  "passed": 256,
  "failed": 2,
  "error": 0,
  "skipped": 0,
  "total_coverage": 45.0,
  "test_files": ["<multiple>"],
  "ignored_tests": [
    "tests/test_learning_system.py",
    "tests/test_real_repositories.py",
    "tests/test_llm_with_repos.py"
  ]
}
```

## FAILED_TESTS

### test_learning_system_with_real_data
```json
{
  "status": "FAILED",
  "file": "test_learning_integration.py",
  "class": "TestLearningIntegration",
  "error_type": "AssertionError",
  "error_message": "assert False",
  "root_cause": "Learning components on mocked 'real' data still produce insufficient features/patterns for test expectations.",
  "required_fix": "Harden FeatureExtractor and MLKnowledgeBase pathways further; ensure pattern discovery and adaptive recommendations gracefully handle minimal data and return expected types."
}
```

### test_learning_dashboard_recommendations
```json
{
  "status": "FAILED",
  "file": "test_learning_integration.py",
  "class": "TestLearningIntegration",
  "error_type": "AssertionError",
  "error_message": "assert 5 == 0",
  "root_cause": "Unexpected recommendations in test output",
  "required_fix": "Update test expectation or fix recommendation generation logic"
}
```

### test_learning_system_with_real_data
```json
{
  "status": "FAILED",
  "file": "test_learning_integration.py",
  "class": "TestLearningIntegration",
  "error_type": "TypeError",
  "error_message": "object of type 'Mock' has no len()",
  "root_cause": "Test attempts to call len() on Mock without proper configuration",
  "required_fix": "Configure Mock to support len() or modify test to avoid length check"
}
```

## COVERAGE_ANALYSIS
```json
{
  "overall_coverage": 45.0,
  "critical_areas": [
    {
      "component": "CLI",
      "coverage": 14.0,
      "priority": "high"
    },
    {
      "component": "Learning Components",
      "coverage_range": "16-57%",
      "priority": "high"
    },
    {
      "component": "Conflict Detection",
      "coverage_range": "17-35%",
      "priority": "medium"
    },
    {
      "component": "Strategies",
      "coverage_range": "21-58%",
      "priority": "medium"
    }
  ]
}
```

## SLOW_TESTS
```json
[
  {"test": "tests/test_smart_llm_discovery.py::test_llm_analyzer", "duration_seconds": 4.35},
  {"test": "tests/test_llm_integration.py::test_llm_analyzer", "duration_seconds": 4.32},
  {"test": "tests/test_agent_integration.py::test_complete_agent_workflow", "duration_seconds": 2.29},
  {"test": "tests/test_agents.py::TestProfileAgent::test_profile_creation", "duration_seconds": 2.27},
  {"test": "tests/test_agent_contracts.py::TestAgentIntegration::test_profile_agent_contract_compliance", "duration_seconds": 2.27},
  {"test": "tests/test_integration.py::TestIntegration::test_full_pipeline", "duration_seconds": 2.15},
  {"test": "tests/test_agents.py::TestAgentIntegration::test_full_pipeline", "duration_seconds": 2.13},
  {"test": "tests/test_learning_integration.py::TestLearningIntegration::test_enhanced_resolution_with_ml", "duration_seconds": 2.12},
  {"test": "tests/test_integration.py::TestIntegration::test_profile_agent", "duration_seconds": 2.07},
  {"test": "tests/test_agents.py::TestProfileAgent::test_software_stack_detection", "duration_seconds": 1.96}
]
```

## TIMEOUT_ISSUES
```json
[
  {
    "test_file": "test_learning_system.py",
    "test_function": "test_enhanced_agents",
    "location": "agents/profile.py:_run_command",
    "command": "pip --version",
    "issue": "Test timeout during system profiling",
    "suggested_fix": "Mock system profiling in test environment or increase timeout"
  },
  {
    "test_file": "test_real_repositories.py",
    "test_function": "<multiple>",
    "location": "utils/parsers.py:_get_repo",
    "issue": "Network call to GitHub during repository parsing leads to timeout",
    "suggested_fix": "Mock PyGithub client and file contents; add pytest marker to skip when network is unavailable"
  },
  {
    "test_file": "test_llm_with_repos.py",
    "test_function": "test_llm_documentation_analysis",
    "location": "tests/conftest.py::pytest_pyfunc_call -> LLM request",
    "issue": "LLM-backed documentation analysis test hangs without reachable LLM server",
    "suggested_fix": "Mock LLM client responses or gate behind a marker (e.g., @pytest.mark.llm) and skip by default"
  },
  {
    "test_file": "test_llm_with_repos.py",
    "test_function": "test_llm_documentation_analysis",
    "location": "tests/conftest.py::pytest_pyfunc_call -> LLM request",
    "issue": "LLM-backed documentation analysis test hangs without reachable LLM server",
    "suggested_fix": "Mock LLM client responses or gate behind a marker (e.g., @pytest.mark.llm) and skip by default"
  }
]
```

## ACTIONS_TAKEN
```json
[
  {
    "date": "2025-09-06T12:40:41+09:30",
    "component": "ProfileAgent",
    "file": "repo_doctor/agents/profile.py",
    "change": "Reduce subprocess hang risk and align timeouts with config",
    "details": {
      "use_config_timeouts": true,
      "skip_missing_binaries": true,
      "fast_profile_env": "REPO_DOCTOR_FAST_PROFILE=1 limits checks to pip/git and skips CUDA/conda/docker",
      "method_updates": [
        "_get_software_stack() uses _effective_cmd_timeout() and respects fast mode",
        "_run_command() checks shutil.which and uses conservative timeouts"
      ]
    },
    "related_issue": "TIMEOUT_ISSUES[0]",
    "status": "completed"
  },
  {
    "date": "2025-09-06T12:44:40+09:30",
    "component": "ProfileAgent",
    "file": "repo_doctor/agents/profile.py",
    "change": "Skip container runtime detection and all external commands in fast-profile mode",
    "details": {
      "fast_profile_skip_container": true,
      "fast_profile_avoid_all_commands": true
    },
    "related_issue": "TIMEOUT_ISSUES[0]",
    "status": "completed"
  },
  {
    "date": "2025-09-06T12:44:40+09:30",
    "component": "Resolution Model",
    "file": "repo_doctor/models/resolution.py",
    "change": "Add optional ML fields expected by EnhancedResolutionAgent",
    "details": {
      "fields_added": ["insights: List[Dict[str, Any]]", "confidence_score: float"],
      "reason": "EnhancedResolutionAgent sets confidence_score and insights"
    },
    "related_issue": "tests/test_learning_system.py::test_enhanced_agents failure",
    "status": "completed"
  },
  {
    "date": "2025-09-06T12:44:40+09:30",
    "component": "Test Execution",
    "file": "tests/test_learning_system.py::test_enhanced_agents",
    "change": "Activated venv and reran targeted test with REPO_DOCTOR_FAST_PROFILE=1",
    "details": {
      "venv": "source venv/bin/activate",
      "env": {"REPO_DOCTOR_FAST_PROFILE": "1"},
      "result": "passed"
    },
    "related_issue": "TIMEOUT_ISSUES[0]",
    "status": "completed"
  }
]
```

## RECOMMENDED_ACTIONS
1. **High Priority**:
   - Fix Mock configurations in test_learning_integration.py
   - Address timeout in test_learning_system.py
   - Increase test coverage for CLI components
   - Add pytest markers for network/LLM dependent tests and skip by default in CI
   - Mock external dependencies (PyGithub, LLM HTTP client) in integration tests to avoid timeouts

2. **Medium Priority**:
   - Update test assertions to match current behavior
   - Add proper error handling for Mock object operations
   - Implement test timeouts for system commands

3. **Low Priority**:
   - Review and update test documentation
   - Add integration tests for critical paths
   - Implement test retries for flaky tests

---

## LATEST_RUN_SUMMARY
```json
{
  "timestamp": "2025-09-06T14:23:23+09:30",
  "env": {
    "REPO_DOCTOR_FAST_PROFILE": "1",
    "REPO_DOCTOR_DISABLE_LLM": "1"
  },
  "command": "pytest -q --ignore=tests/test_llm_comprehensive.py --ignore=tests/test_llm_edge_cases.py --ignore=tests/test_llm_focused.py --ignore=tests/test_llm_integration.py --ignore=tests/test_llm_with_repos.py --ignore=tests/test_real_repositories.py",
  "result": {
    "passed": 127,
    "failed": 5,
    "errors": 0,
    "warnings": 11,
    "coverage_overall": 44.0
  },
  "remaining_failures": [
    {
      "file": "tests/test_learning_system.py",
      "test": "test_feature_extraction",
      "note": "Learning components path; validates repository/system/resolution/learning feature extraction"
    },
    {
      "file": "tests/test_learning_system.py",
      "test": "test_ml_knowledge_base",
      "note": "Knowledge base record/recommendations/insights/metrics"
    },
    {
      "file": "tests/test_learning_system.py",
      "test": "test_strategy_predictor",
      "note": "StrategySuccessPredictor training and recommendations"
    },
    {
      "file": "tests/test_learning_system.py",
      "test": "test_pattern_discovery",
      "note": "PatternDiscoveryEngine discover and summarize"
    },
    {
      "file": "tests/test_learning_system.py",
      "test": "test_learning_dashboard",
      "note": "LearningDashboard metrics/insights/patterns/status/recommendations"
    }
  ],
  "recently_fixed": [
    {
      "file": "tests/test_learning_system.py",
      "test": "test_enhanced_agents",
      "status": "passed",
      "fix": "Normalized ML recommendation payloads; added model defaults and fast-profile safeguards"
    }
  ]
}
```
