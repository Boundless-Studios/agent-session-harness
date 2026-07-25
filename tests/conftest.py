"""Test-wide isolation from an ambient managed session.

This harness supervises the agents that develop it, so the suite is routinely
run from *inside* a managed session — where `AGENT_SESSION_HARNESS_OWNER_PID`,
`AGENT_SESSION_HARNESS_MANAGED` and friends are already exported and describe a
completely unrelated chain. Code under test reads those variables as ground
truth, so leaving them in place makes results depend on who launched pytest:
`test_acknowledge_writes_bounded_durable_record`, for instance, resolves the
acknowledging pid from the environment and fails with "acknowledgement process
does not match managed child" only when a developer runs it inside a session,
never in CI.

Strip them once, for every test. Anything that needs them sets them explicitly
with `monkeypatch.setenv`, which runs after this fixture.
"""

from __future__ import annotations

import os

import pytest

_HARNESS_ENVIRONMENT_PREFIX = "AGENT_SESSION_HARNESS_"


@pytest.fixture(autouse=True)
def _isolated_from_ambient_harness_environment(monkeypatch) -> None:
    for key in [
        key for key in os.environ if key.startswith(_HARNESS_ENVIRONMENT_PREFIX)
    ]:
        monkeypatch.delenv(key, raising=False)
