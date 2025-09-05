"""Test configuration and async support without pytest-asyncio.

This conftest provides a lightweight async runner so that tests defined as
`async def` can run even if pytest-asyncio is not available in the environment.
It also normalizes the event loop lifecycle across tests.
"""
from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any, Dict

import pytest


@pytest.fixture
def event_loop():
    """Provide a simple event loop fixture for tests that request it."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Allow running `async def` test functions without pytest-asyncio.

    If the test function is a coroutine function, run it using asyncio.run and
    return True to indicate the call was handled.
    """
    testfunction = pyfuncitem.obj
    if inspect.iscoroutinefunction(testfunction):
        # Run coroutine tests and pass only fixtures explicitly declared
        sig = inspect.signature(testfunction)
        accepted = {k: v for k, v in pyfuncitem.funcargs.items() if k in sig.parameters}
        asyncio.run(testfunction(**accepted))
        return True
    return None


# Ensure environment is predictable for tests that rely on these variables
@pytest.fixture(autouse=True)
def _normalize_env(monkeypatch: pytest.MonkeyPatch):
    # Configure LLM server for WSL-hosted LM Studio on Windows
    monkeypatch.setenv("LLM_BASE_URL", "http://172.29.96.1:1234/v1")
    # Clear API key unless explicitly required
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    yield
