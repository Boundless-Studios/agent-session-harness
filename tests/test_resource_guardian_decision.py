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
            process_state=ProcessState.MISSING,
            process_identity_state=ProcessIdentityState.MISSING,
            worktree_state=WorktreeState.DELETED,
            managed_owner_state=ManagedOwnerState.MISSING,
            lease_state=LeaseState.EXPIRED,
        )
    )

    assert decision.action is GuardianAction.REAP
    assert decision.reason_code == "deleted_worktree_without_live_owner"


def test_deleted_worktree_with_exact_live_process_alerts() -> None:
    decision = decide_guardian_action(
        observation(
            worktree_state=WorktreeState.DELETED,
            managed_owner_state=ManagedOwnerState.MISSING,
        )
    )

    assert decision.action is GuardianAction.ALERT
    assert decision.reason_code == "live_unowned_resource"


def test_deleted_worktree_with_active_lease_alerts() -> None:
    decision = decide_guardian_action(
        observation(
            process_state=ProcessState.MISSING,
            process_identity_state=ProcessIdentityState.MISSING,
            worktree_state=WorktreeState.DELETED,
            managed_owner_state=ManagedOwnerState.MISSING,
        )
    )

    assert decision.action is GuardianAction.ALERT
    assert decision.reason_code == "active_owner_lease"


@pytest.mark.parametrize("process_state", [ProcessState.MISSING, ProcessState.ZOMBIE])
def test_terminal_managed_child_is_reaped(process_state: ProcessState) -> None:
    decision = decide_guardian_action(
        observation(
            process_state=process_state,
            process_identity_state=ProcessIdentityState.MISSING,
            managed_owner_state=ManagedOwnerState.MISSING,
            lease_state=LeaseState.EXPIRED,
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
    assert decision.resource.kind == source.resource.kind
    assert decision.resource.resource_key == source.resource.resource_key
    assert decision.observed_at == source.observed_at


@pytest.mark.parametrize(
    ("resource", "changes"),
    [
        (
            ManagedResource(
                kind="process-only",
                resource_key="process:1",
                process_identity=process_identity(),
                cleanup_adapter="cleanup-v1",
            ),
            {"worktree_state": WorktreeState.DELETED},
        ),
        (
            ManagedResource(
                kind="worktree-only",
                resource_key="worktree:1",
                worktree_identity=WorktreeIdentity(canonical_path="/workspace/one"),
                cleanup_adapter="cleanup-v1",
            ),
            {
                "process_state": ProcessState.MISSING,
                "process_identity_state": ProcessIdentityState.MISMATCH,
            },
        ),
        (
            ManagedResource(
                kind="process-only",
                resource_key="process:2",
                process_identity=process_identity(),
                cleanup_adapter="cleanup-v1",
            ),
            {"lease_state": LeaseState.EXPIRED},
        ),
    ],
)
def test_missing_registered_identity_cannot_manufacture_reap_authority(
    resource: ManagedResource,
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "resource": resource,
        "process_state": None,
        "process_identity_state": ProcessIdentityState.NOT_APPLICABLE,
        "worktree_state": WorktreeState.NOT_APPLICABLE,
        "managed_owner_state": ManagedOwnerState.MISSING,
        "lease_state": LeaseState.NOT_APPLICABLE,
    }
    values.update(changes)
    decision = decide_guardian_action(observation(**values))

    assert decision.action is GuardianAction.ALERT
    assert decision.reason_code == "identity_state_mismatch"


@pytest.mark.parametrize(
    ("changes", "reason_code", "required_evidence"),
    [
        (
            {
                "process_state": ProcessState.MISSING,
                "process_identity_state": ProcessIdentityState.MISSING,
                "worktree_state": WorktreeState.DELETED,
                "managed_owner_state": ManagedOwnerState.MISSING,
                "lease_state": LeaseState.EXPIRED,
            },
            "deleted_worktree_without_live_owner",
            {"process_missing", "worktree_deleted", "lease_expired"},
        ),
        (
            {
                "process_state": ProcessState.MISSING,
                "process_identity_state": ProcessIdentityState.MISMATCH,
                "managed_owner_state": ManagedOwnerState.MISSING,
                "lease_state": LeaseState.EXPIRED,
            },
            "expired_fenced_identity",
            {"process_missing", "process_identity_mismatch", "lease_expired"},
        ),
        (
            {
                "process_state": ProcessState.ZOMBIE,
                "process_identity_state": ProcessIdentityState.MISSING,
                "managed_owner_state": ManagedOwnerState.MISSING,
                "lease_state": LeaseState.EXPIRED,
            },
            "terminal_managed_child",
            {"process_zombie", "process_identity_missing"},
        ),
    ],
)
def test_reap_decisions_include_normalized_proof(
    changes: dict[str, object],
    reason_code: str,
    required_evidence: set[str],
) -> None:
    decision = decide_guardian_action(observation(**changes))

    assert decision.action is GuardianAction.REAP
    assert decision.reason_code == reason_code
    assert required_evidence <= {item.code for item in decision.evidence}


@pytest.mark.parametrize(
    ("process_state", "process_identity_state", "lease_state"),
    [
        (
            ProcessState.MISSING,
            ProcessIdentityState.MISSING,
            LeaseState.NOT_APPLICABLE,
        ),
        (
            ProcessState.MISSING,
            ProcessIdentityState.NOT_APPLICABLE,
            LeaseState.EXPIRED,
        ),
        (
            ProcessState.MISSING,
            ProcessIdentityState.EXACT,
            LeaseState.EXPIRED,
        ),
    ],
)
def test_registered_identity_requires_complete_coherent_state(
    process_state: ProcessState,
    process_identity_state: ProcessIdentityState,
    lease_state: LeaseState,
) -> None:
    decision = decide_guardian_action(
        observation(
            process_state=process_state,
            process_identity_state=process_identity_state,
            managed_owner_state=ManagedOwnerState.MISSING,
            lease_state=lease_state,
        )
    )

    assert decision.action is GuardianAction.ALERT
    assert decision.reason_code == "identity_state_mismatch"
