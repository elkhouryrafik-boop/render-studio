"""Shared fixtures.

Most suites exercise the full routing table, including the two experimental
tasks, so the flag is on by default. Tests of the gate itself set or unset
ENABLE_EXPERIMENTAL_TASKS explicitly through monkeypatch.
"""
import pytest


@pytest.fixture(autouse=True)
def _experimental_tasks_on(monkeypatch):
    monkeypatch.setenv("ENABLE_EXPERIMENTAL_TASKS", "1")
