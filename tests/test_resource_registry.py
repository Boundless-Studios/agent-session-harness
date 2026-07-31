from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_session_harness.process_identity import ProcessIdentity, ProcessPlatform
from agent_session_harness.resource_guardian import ManagedResource
from agent_session_harness.resource_registry import (
    RegistrationConflictError,
    ResourceRegistry,
)
from agent_session_harness.secure_files import private_file_mode

NOW = datetime(2026, 7, 30, tzinfo=UTC)


def resource(key: str = "child:1") -> ManagedResource:
    return ManagedResource(
        kind="hook-child",
        resource_key=key,
        process_identity=ProcessIdentity(
            platform=ProcessPlatform.LINUX,
            pid=42,
            opaque_start_token="linux:123",
            executable_identity="/usr/bin/python3",
            captured_at=NOW,
        ),
        cleanup_adapter="hook-child-v1",
    )


def test_registration_survives_registry_restart_and_is_private(tmp_path) -> None:
    path = tmp_path / "guardian" / "resources.json"
    first = ResourceRegistry(path)

    registered = first.register(resource(), now=NOW)
    restored = ResourceRegistry(path).list()

    assert restored == [registered]
    assert private_file_mode(path) == 0o600


def test_replacement_token_fences_stale_unregister(tmp_path) -> None:
    registry = ResourceRegistry(tmp_path / "resources.json")
    old = registry.register(resource(), now=NOW)
    replacement = registry.register(resource(), now=NOW)

    with pytest.raises(RegistrationConflictError, match="stale"):
        registry.unregister(
            old.resource.kind, old.resource.resource_key, old.registration_id
        )

    assert registry.list() == [replacement]
    registry.unregister(
        replacement.resource.kind,
        replacement.resource.resource_key,
        replacement.registration_id,
    )
    assert registry.list() == []


def test_corrupt_registry_blocks_mutation_instead_of_becoming_empty(tmp_path) -> None:
    path = tmp_path / "resources.json"
    path.write_text('{"registrations":', encoding="utf-8")
    registry = ResourceRegistry(path)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        registry.list()
    with pytest.raises(RuntimeError, match="invalid JSON"):
        registry.register(resource(), now=NOW)


def test_registry_rejects_unsupported_major_version(tmp_path) -> None:
    path = tmp_path / "resources.json"
    path.write_text('{"schema_version":2,"registrations":[]}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported"):
        ResourceRegistry(path).list()
