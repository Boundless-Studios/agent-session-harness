from datetime import UTC, datetime

import pytest

from agent_session_harness.process_identity import (
    ProcessIdentity,
    ProcessPlatform,
    ProcessState,
)
from agent_session_harness.resource_guardian import (
    GuardianAction,
    GuardianEvidence,
    GuardianObservation,
    LeaseState,
    ManagedOwnerState,
    ManagedResource,
    OwnerLeaseIdentity,
    ProcessIdentityState,
    WorktreeIdentity,
    WorktreeState,
    decide_guardian_action,
)

OBSERVED_AT = datetime(2026, 7, 30, tzinfo=UTC)


def process_identity() -> ProcessIdentity:
    return ProcessIdentity(
        platform=ProcessPlatform.LINUX,
        pid=42,
        opaque_start_token="linux:12345",
        executable_identity="/usr/bin/python3",
        captured_at=OBSERVED_AT,
    )


def observation(**changes: object) -> GuardianObservation:
    values: dict[str, object] = {
        "resource": ManagedResource(
            kind="worktree-tunnel",
            resource_key="tunnel:worktree-7",
            process_identity=process_identity(),
            worktree_identity=WorktreeIdentity(canonical_path="/workspace/worktree-7"),
            owner_lease=OwnerLeaseIdentity(
                owner_id="session-7",
                fencing_token=3,
                expires_at=OBSERVED_AT,
            ),
            cleanup_adapter="tunnel-cleanup-v1",
        ),
        "process_state": ProcessState.RUNNING,
        "process_identity_state": ProcessIdentityState.EXACT,
        "worktree_state": WorktreeState.PRESENT,
        "managed_owner_state": ManagedOwnerState.LIVE,
        "lease_state": LeaseState.ACTIVE,
        "evidence": [
            GuardianEvidence(
                source="process-inspector",
                code="exact_process_live",
            )
        ],
        "observed_at": OBSERVED_AT,
    }
    values.update(changes)
    return GuardianObservation.model_validate(values)


def test_live_managed_owner_is_retained() -> None:
    decision = decide_guardian_action(observation())

    assert decision.action is GuardianAction.RETAIN
    assert decision.reason_code == "live_managed_owner"


def test_deleted_worktree_without_live_owner_is_reaped() -> None:
    decision = decide_guardian_action(
        observation(
            worktree_state=WorktreeState.DELETED,
            managed_owner_state=ManagedOwnerState.MISSING,
        )
    )

    assert decision.action is GuardianAction.REAP
    assert decision.reason_code == "deleted_worktree_without_live_owner"


@pytest.mark.parametrize("process_state", [ProcessState.MISSING, ProcessState.ZOMBIE])
def test_terminal_managed_child_is_reaped(process_state: ProcessState) -> None:
    decision = decide_guardian_action(
        observation(
            process_state=process_state,
            process_identity_state=ProcessIdentityState.MISSING,
            managed_owner_state=ManagedOwnerState.MISSING,
        )
    )

    assert decision.action is GuardianAction.REAP
    assert decision.reason_code == "terminal_managed_child"


def test_pid_reuse_with_active_lease_alerts() -> None:
    decision = decide_guardian_action(
        observation(
            process_identity_state=ProcessIdentityState.MISMATCH,
            managed_owner_state=ManagedOwnerState.MISSING,
        )
    )

    assert decision.action is GuardianAction.ALERT
    assert decision.reason_code == "process_identity_mismatch"


def test_expired_fenced_non_live_identity_is_reaped() -> None:
    decision = decide_guardian_action(
        observation(
            process_state=ProcessState.MISSING,
            process_identity_state=ProcessIdentityState.MISMATCH,
            managed_owner_state=ManagedOwnerState.MISSING,
            lease_state=LeaseState.EXPIRED,
        )
    )

    assert decision.action is GuardianAction.REAP
    assert decision.reason_code == "expired_fenced_identity"


def test_unknown_identity_alerts() -> None:
    decision = decide_guardian_action(
        observation(
            process_state=ProcessState.UNKNOWN,
            process_identity_state=ProcessIdentityState.UNKNOWN,
            managed_owner_state=ManagedOwnerState.UNKNOWN,
        )
    )

    assert decision.action is GuardianAction.ALERT
    assert decision.reason_code == "ambiguous_identity"


def test_failed_inspection_alerts_even_when_worktree_is_deleted() -> None:
    decision = decide_guardian_action(
        observation(
            worktree_state=WorktreeState.DELETED,
            managed_owner_state=ManagedOwnerState.MISSING,
            evidence=[
                GuardianEvidence(
                    source="process-inspector",
                    code="inspection_failed",
                    detail="permission denied",
                )
            ],
        )
    )

    assert decision.action is GuardianAction.ALERT
    assert decision.reason_code == "inspection_failed"


def test_live_unowned_process_alerts_instead_of_reaping() -> None:
    decision = decide_guardian_action(
        observation(
            managed_owner_state=ManagedOwnerState.MISSING,
            lease_state=LeaseState.EXPIRED,
        )
    )

    assert decision.action is GuardianAction.ALERT
    assert decision.reason_code == "live_unowned_resource"


def test_every_decision_preserves_observation_evidence() -> None:
    source = observation()

    decision = decide_guardian_action(source)

    assert decision.evidence[: len(source.evidence)] == source.evidence
    assert decision.evidence[-1].source == "guardian-policy"
    assert decision.evidence[-1].code == decision.reason_code
    assert decision.observed_at == source.observed_at
