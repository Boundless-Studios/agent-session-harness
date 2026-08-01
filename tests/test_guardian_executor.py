from __future__ import annotations

from datetime import UTC, datetime

from agent_session_harness.guardian_executor import (
    CleanupOutcome,
    GuardianExecutor,
)
from agent_session_harness.guardian_service import GuardianPublication
from agent_session_harness.guardian_singleton import (
    GuardianLeaseProof,
    StaleGuardianError,
)
from agent_session_harness.process_identity import ProcessIdentity, ProcessPlatform
from agent_session_harness.resource_guardian import (
    GuardianAction,
    GuardianDecision,
    GuardianEvidence,
    GuardianReasonCode,
    ManagedResource,
)
from agent_session_harness.resource_registry import ResourceRegistry

NOW = datetime(2026, 7, 31, tzinfo=UTC)


class Ownership:
    def __init__(self, proof: GuardianLeaseProof):
        self.proof = proof

    def current_proof(self) -> GuardianLeaseProof:
        return self.proof


class StaleOwnership:
    def current_proof(self) -> GuardianLeaseProof:
        raise StaleGuardianError("lease expired")


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


def publication(registration_id: str) -> GuardianPublication:
    return GuardianPublication(
        decision=GuardianDecision(
            resource={"kind": "hook-child", "resource_key": "child:1"},
            action=GuardianAction.REAP,
            reason_code=GuardianReasonCode.TERMINAL_MANAGED_CHILD,
            evidence=[GuardianEvidence(source="test", code="process_missing")],
            observed_at=NOW,
        ),
        registration_id=registration_id,
        guardian=GuardianLeaseProof(claim_id="claim-1", lease_epoch=7),
    )


def setup_executor(tmp_path):
    registry = ResourceRegistry(tmp_path / "resources.json")
    registration = registry.register(resource(), now=NOW)
    executor = GuardianExecutor(
        registry=registry,
        ownership=Ownership(GuardianLeaseProof(claim_id="claim-1", lease_epoch=7)),
    )
    return executor, registry, registration


def test_observe_only_default_never_invokes_cleanup(tmp_path) -> None:
    executor, _registry, registration = setup_executor(tmp_path)

    result = executor.authorize(publication(registration.registration_id))

    assert result.outcome is CleanupOutcome.OBSERVED


def test_enabled_reason_authorizes_exact_current_registration(tmp_path) -> None:
    executor, _registry, registration = setup_executor(tmp_path)

    result = executor.authorize(
        publication(registration.registration_id),
        observe_only=False,
        enabled_reasons={GuardianReasonCode.TERMINAL_MANAGED_CHILD},
    )

    assert result.outcome is CleanupOutcome.AUTHORIZED
    assert result.authorization is not None
    assert result.authorization.registration == registration
    assert result.authorization.guardian == GuardianLeaseProof(
        claim_id="claim-1",
        lease_epoch=7,
    )


def test_replaced_registration_refuses_stale_cleanup(tmp_path) -> None:
    executor, registry, registration = setup_executor(tmp_path)
    registry.register(resource(), now=NOW)

    result = executor.authorize(
        publication(registration.registration_id),
        observe_only=False,
        enabled_reasons={GuardianReasonCode.TERMINAL_MANAGED_CHILD},
    )

    assert result.outcome is CleanupOutcome.REFUSED_STALE_REGISTRATION


def test_changed_guardian_epoch_refuses_stale_cleanup(tmp_path) -> None:
    executor, _registry, registration = setup_executor(tmp_path)
    executor.ownership.proof = GuardianLeaseProof(claim_id="claim-2", lease_epoch=8)

    result = executor.authorize(
        publication(registration.registration_id),
        observe_only=False,
        enabled_reasons={GuardianReasonCode.TERMINAL_MANAGED_CHILD},
    )

    assert result.outcome is CleanupOutcome.REFUSED_STALE_GUARDIAN


def test_expired_guardian_lease_refuses_cleanup(tmp_path) -> None:
    executor, _registry, registration = setup_executor(tmp_path)
    executor.ownership = StaleOwnership()

    result = executor.authorize(
        publication(registration.registration_id),
        observe_only=False,
        enabled_reasons={GuardianReasonCode.TERMINAL_MANAGED_CHILD},
    )

    assert result.outcome is CleanupOutcome.REFUSED_STALE_GUARDIAN
