from __future__ import annotations

import json
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

    assert registry.is_current(old) is False
    assert registry.is_current(replacement) is True

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


def test_registry_rejects_unknown_persisted_fields(tmp_path) -> None:
    path = tmp_path / "resources.json"
    path.write_text(
        '{"schema_version":1,"registrations":[],"unknown_authority":true}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="contract"):
        ResourceRegistry(path).list()


def test_register_refuses_to_persist_more_than_bounded_capacity(tmp_path) -> None:
    path = tmp_path / "resources.json"
    template = ResourceRegistry(path).register(resource("child:seed"), now=NOW)
    registrations = []
    for index in range(1024):
        payload = template.model_dump(mode="json")
        payload["registration_id"] = f"registration-{index:04d}"
        payload["resource"]["resource_key"] = f"child:{index}"
        registrations.append(payload)
    path.write_text(
        json.dumps({"schema_version": 1, "registrations": registrations}),
        encoding="utf-8",
    )
    registry = ResourceRegistry(path)

    with pytest.raises(ValueError, match="1024"):
        registry.register(resource("child:overflow"), now=NOW)

    assert len(registry.list()) == 1024


def test_package_exports_guardian_service_contracts() -> None:
    import agent_session_harness

    assert agent_session_harness.ResourceRegistry is ResourceRegistry
    assert agent_session_harness.GuardianService.__name__ == "GuardianService"
    assert agent_session_harness.GuardianSingleton.__name__ == "GuardianSingleton"
