### analysis_agent_github_content_timeout
```json
{
  "status": "FAILED",
  "file": "tests/test_integration.py (or related)",
  "class": "<n/a>",
  "error_type": "TimeoutError",
  "error_message": "requests/urllib3 timeout while calling GitHub API in AnalysisAgent._get_file_content",
  "root_cause": "Live network call to GitHub via PyGithub.get_contents during README scan with no token/offline conditions.",
  "required_fix": "Introduce an offline-safe path in AnalysisAgent: short-circuit remote content fetch when no token or when an OFFLINE flag/config is set; add small request timeout and robust exception handling."
}
```

# TEST_ISSUES.md - Machine Readable Test Status

## METADATA
- **Last Updated**: 2025-09-06T20:19:28+09:30
- **Test Environment**: Python 3.12.3
- **Test Command**: `pytest -q --maxfail=50 --disable-warnings --durations=10`

## TEST_SUMMARY
```json
{
  "notes": "Latest focused runs validated LLM suites against live server. Full suite run surfaced a remaining network timeout in AnalysisAgent when parsing GitHub contents.",
  "llm_suites": {
    "files": [
      "tests/test_llm_comprehensive.py",
      "tests/test_llm_edge_cases.py",
      "tests/test_llm_focused.py",
      "tests/test_llm_integration.py",
      "tests/test_llm_with_repos.py",
      "tests/test_smart_llm_discovery.py"
    ],
    "status": "passed"
  },
  "full_run_summary": {
    "passed": "majority",
    "failed": 1,
    "errors": 0,
    "failures": [
      {
        "file": "tests/test_integration.py (or related)",
        "note": "Network timeout in AnalysisAgent._get_file_content via PyGithub during documentation scan"
      }
    ]
  }
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
  ,
  {
    "test_file": "tests/test_integration.py (or related)",
    "test_function": "<multiple>",
    "location": "agents/analysis.py::_get_file_content",
    "issue": "Network call to GitHub via PyGithub.get_contents leads to timeout in full test run",
    "suggested_fix": "Add offline guard (e.g., REPO_DOCTOR_OFFLINE=1) and/or require a token for GitHub fetch, apply short per-request timeout and catch exceptions to continue gracefully"
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
  ,
  {
    "date": "2025-09-06T20:05:00+09:30",
    "component": "Learning/FeatureExtractor",
    "file": "repo_doctor/learning/feature_extractor.py",
    "change": "Hardened GPU vendor detection and resolution file introspection; robust outcome parsing",
    "details": {
      "gpu_vendor": "derive from name if vendor missing",
      "files": "use name or path basename",
      "outcome": "handle enums/strings"
    },
    "related_issue": "tests/test_learning_system.py::test_feature_extraction",
    "status": "completed"
  },
  {
    "date": "2025-09-06T20:06:00+09:30",
    "component": "Learning/MLKnowledgeBase",
    "file": "repo_doctor/learning/ml_knowledge_base.py",
    "change": "Avoided base-class method override mismatch by renaming ML-specific pattern updaters",
    "details": {
      "methods": ["_update_success_patterns_ml", "_update_failure_patterns_ml"],
      "call_site": "_update_learning_patterns"
    },
    "related_issue": "tests/test_learning_system.py::test_ml_knowledge_base",
    "status": "completed"
  },
  {
    "date": "2025-09-06T20:08:00+09:30",
    "component": "Tests/Async Harness",
    "file": "tests/conftest.py",
    "change": "Async test runner now uses asyncio.run unless an event loop is already present; LLM base URL defaults to user's live server when unset",
    "details": {
      "pytest_pyfunc_call": "cooperates with pytest-asyncio and avoids event loop conflicts",
      "LLM_BASE_URL": "http://172.29.96.1:1234/v1 if unset"
    },
    "related_issue": "async def tests not supported / LLM hangs",
    "status": "completed"
  },
  {
    "date": "2025-09-06T20:12:00+09:30",
    "component": "LLM Integration",
    "file": "repo_doctor/utils/llm.py",
    "change": "Stabilized LLM client and analyzer for live server + mocks (timeouts, JSON parsing, wait_for caps)",
    "details": {
      "client": "restored constructor defaults; availability fallbacks; tolerant JSON parsing; ClientTimeout",
      "analyzer": "asyncio.wait_for caps and TimeoutError handling"
    },
    "related_issue": "LLM suites",
    "status": "completed"
  }
]
```

## RECOMMENDED_ACTIONS
1. **High Priority**:
   - Introduce offline guard in `AnalysisAgent._get_file_content` to avoid GitHub calls during tests (e.g., respect `REPO_DOCTOR_OFFLINE=1` or require `github_token`) and set a strict request timeout.
   - Keep LLM tests green by preserving tolerant JSON parsing and timeouts.
   - Re-run full suite to confirm no further network timeouts.

2. **Medium Priority**:
   - Improve CLI coverage and error handling paths.
   - Harden learning components for minimal data scenarios.

3. **Low Priority**:
   - Review test documentation and add integration tests for critical paths.
   - Consider test retries for transient network cases (outside offline mode).

---

## LATEST_RUN_SUMMARY
```json
{
  "timestamp": "2025-09-06T20:19:28+09:30",
  "env": {
    "LLM_BASE_URL": "http://172.29.96.1:1234/v1"
  },
  "llm_run": {
    "command": "pytest -q tests/test_llm_comprehensive.py tests/test_llm_edge_cases.py tests/test_llm_focused.py tests/test_llm_integration.py tests/test_llm_with_repos.py tests/test_smart_llm_discovery.py",
    "result": { "passed": true }
  },
  "full_run": {
    "command": "pytest -q --maxfail=50 --disable-warnings --durations=10",
    "result": {
      "failures": [
        {
          "file": "<integration tests>",
          "root_cause": "GitHub content fetch timeout via PyGithub in AnalysisAgent"
        }
      ]
    }
  }
}
```
