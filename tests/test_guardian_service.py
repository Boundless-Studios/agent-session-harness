from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_session_harness.guardian_service import GuardianService
from agent_session_harness.guardian_singleton import GuardianLeaseProof
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

    def current_proof(self) -> GuardianLeaseProof:
        self.assertions += 1
        if self.fail_after is not None and self.assertions > self.fail_after:
            raise RuntimeError("stale guardian lease")
        return GuardianLeaseProof(claim_id="claim-1", lease_epoch=7)


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
    assert decisions[0].decision.action is GuardianAction.REAP
    assert decisions[0].registration_id
    assert decisions[0].guardian == GuardianLeaseProof(
        claim_id="claim-1",
        lease_epoch=7,
    )
    assert decisions[0].observe_only is True
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

    assert decisions[0].decision.action is GuardianAction.ALERT
    assert decisions[0].decision.reason_code == "inspection_failed"


def test_replaced_registration_blocks_stale_reap_publication(tmp_path) -> None:
    registry = ResourceRegistry(tmp_path / "resources.json")
    registry.register(resource(), now=NOW)
    published = []

    def replace_during_inspection(managed: ManagedResource) -> GuardianObservation:
        registry.register(managed, now=NOW)
        return missing_observation(managed)

    decisions = GuardianService(
        registry=registry,
        lease=Lease(),
        observer=replace_during_inspection,
        publish=published.append,
    ).run_once()

    assert decisions == []
    assert published == []


def test_mismatched_observation_alerts_for_registered_resource(tmp_path) -> None:
    registry = ResourceRegistry(tmp_path / "resources.json")
    registered = resource()
    registry.register(registered, now=NOW)
    other = registered.model_copy(update={"resource_key": "child:other"})

    publications = GuardianService(
        registry=registry,
        lease=Lease(),
        observer=lambda _managed: missing_observation(other),
        clock=lambda: NOW,
    ).run_once()

    decision = publications[0].decision
    assert decision.resource.resource_key == registered.resource_key
    assert decision.action is GuardianAction.ALERT
    assert decision.reason_code == "inspection_failed"
