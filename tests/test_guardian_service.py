from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_session_harness.guardian_service import GuardianService
from agent_session_harness.process_identity import ProcessIdentity, ProcessPlatform
from agent_session_harness.resource_guardian import (
    GuardianAction,
    GuardianEvidence,
    GuardianObservation,
    LeaseState,
    ManagedOwnerState,
    ManagedResource,
    ProcessIdentityState,
    WorktreeState,
)
from agent_session_harness.resource_registry import ResourceRegistry

NOW = datetime(2026, 7, 30, tzinfo=UTC)


class Lease:
    def __init__(self, *, fail_after: int | None = None):
        self.assertions = 0
        self.fail_after = fail_after

    def assert_current(self) -> None:
        self.assertions += 1
        if self.fail_after is not None and self.assertions > self.fail_after:
            raise RuntimeError("stale guardian lease")


def resource() -> ManagedResource:
    return ManagedResource(
        kind="hook-child",
        resource_key="child:1",
        process_identity=ProcessIdentity(
            platform=ProcessPlatform.LINUX,
            pid=42,
            opaque_start_token="linux:123",
            executable_identity="/usr/bin/python3",
            captured_at=NOW,
        ),
        cleanup_adapter="hook-child-v1",
    )


def missing_observation(managed: ManagedResource) -> GuardianObservation:
    return GuardianObservation(
        resource=managed,
        process_state="missing",
        process_identity_state=ProcessIdentityState.MISSING,
        worktree_state=WorktreeState.NOT_APPLICABLE,
        managed_owner_state=ManagedOwnerState.MISSING,
        lease_state=LeaseState.NOT_APPLICABLE,
        evidence=[GuardianEvidence(source="test", code="process_missing")],
        observed_at=NOW,
    )


def test_service_is_observe_only_and_publishes_decisions(tmp_path) -> None:
    registry = ResourceRegistry(tmp_path / "resources.json")
    registry.register(resource(), now=NOW)
    published = []
    lease = Lease()
    service = GuardianService(
        registry=registry,
        lease=lease,
        observer=missing_observation,
        publish=published.append,
    )

    decisions = service.run_once()

    assert decisions == published
    assert decisions[0].action is GuardianAction.REAP
    assert lease.assertions == 2


def test_lost_lease_blocks_reap_authorizing_publication(tmp_path) -> None:
    registry = ResourceRegistry(tmp_path / "resources.json")
    registry.register(resource(), now=NOW)
    published = []
    service = GuardianService(
        registry=registry,
        lease=Lease(fail_after=1),
        observer=missing_observation,
        publish=published.append,
    )

    with pytest.raises(RuntimeError, match="stale guardian lease"):
        service.run_once()

    assert published == []


def test_observer_failure_becomes_alert_instead_of_reap(tmp_path) -> None:
    registry = ResourceRegistry(tmp_path / "resources.json")
    registry.register(resource(), now=NOW)

    def fail(_resource: ManagedResource) -> GuardianObservation:
        raise PermissionError("denied")

    decisions = GuardianService(
        registry=registry,
        lease=Lease(),
        observer=fail,
        clock=lambda: NOW,
    ).run_once()

    assert decisions[0].action is GuardianAction.ALERT
    assert decisions[0].reason_code == "inspection_failed"
