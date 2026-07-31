from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_session_harness.process_identity import (
    ProcessIdentity,
    ProcessPlatform,
    ProcessState,
)
from agent_session_harness.resource_guardian import (
    GuardianAction,
    GuardianDecision,
    GuardianEvidence,
    GuardianObservation,
    LeaseState,
    ManagedOwnerState,
    ManagedResource,
    OwnerLeaseIdentity,
    WorktreeIdentity,
    WorktreeState,
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


def managed_resource(**changes: object) -> ManagedResource:
    values: dict[str, object] = {
        "kind": "worktree-tunnel",
        "resource_key": "tunnel:worktree-7",
        "process_identity": process_identity(),
        "worktree_identity": WorktreeIdentity(
            canonical_path="/workspace/worktree-7",
        ),
        "owner_lease": OwnerLeaseIdentity(
            owner_id="session-7",
            fencing_token=3,
            expires_at=OBSERVED_AT,
        ),
        "cleanup_adapter": "tunnel-cleanup-v1",
    }
    values.update(changes)
    return ManagedResource.model_validate(values)


def test_managed_resource_contract_round_trips_portable_identities() -> None:
    resource = managed_resource()

    restored = ManagedResource.model_validate_json(resource.model_dump_json())

    assert restored.schema_version == 1
    assert restored.kind == "worktree-tunnel"
    assert restored.resource_key == "tunnel:worktree-7"
    assert restored.process_identity == process_identity()
    assert restored.worktree_identity is not None
    assert restored.owner_lease is not None
    assert restored.cleanup_adapter == "tunnel-cleanup-v1"


def test_managed_resource_requires_an_identity() -> None:
    with pytest.raises(ValidationError, match="identity"):
        managed_resource(
            process_identity=None,
            worktree_identity=None,
            owner_lease=None,
        )


def test_observation_and_decision_contracts_record_stable_evidence() -> None:
    evidence = GuardianEvidence(
        source="process-inspector",
        code="exact_process_live",
        detail="pid=42",
    )
    observation = GuardianObservation(
        resource=managed_resource(),
        process_state=ProcessState.RUNNING,
        worktree_state=WorktreeState.PRESENT,
        managed_owner_state=ManagedOwnerState.LIVE,
        lease_state=LeaseState.ACTIVE,
        evidence=[evidence],
        observed_at=OBSERVED_AT,
    )
    decision = GuardianDecision(
        action=GuardianAction.RETAIN,
        reason_code="live_managed_owner",
        evidence=observation.evidence,
        observed_at=OBSERVED_AT,
    )

    assert decision.model_dump(mode="json") == {
        "schema_version": 1,
        "action": "retain",
        "reason_code": "live_managed_owner",
        "evidence": [
            {
                "schema_version": 1,
                "source": "process-inspector",
                "code": "exact_process_live",
                "detail": "pid=42",
            }
        ],
        "observed_at": "2026-07-30T00:00:00Z",
    }


@pytest.mark.parametrize(
    "model",
    [
        managed_resource().model_copy(update={"schema_version": 2}),
        GuardianEvidence(
            source="inspection",
            code="unknown",
        ).model_copy(update={"schema_version": 2}),
    ],
)
def test_contracts_reject_unsupported_major_versions(model: object) -> None:
    contract = type(model)

    with pytest.raises(ValidationError):
        contract.model_validate(model.model_dump())


@pytest.mark.parametrize(
    ("contract", "values"),
    [
        (
            OwnerLeaseIdentity,
            {
                "owner_id": "session-7",
                "fencing_token": 1,
                "expires_at": datetime(2026, 7, 30),
            },
        ),
        (
            GuardianDecision,
            {
                "action": GuardianAction.ALERT,
                "reason_code": "inspection_failed",
                "evidence": [],
                "observed_at": datetime(2026, 7, 30),
            },
        ),
    ],
)
def test_contracts_require_timezone_aware_timestamps(
    contract: type[object],
    values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        contract.model_validate(values)


def test_contracts_preserve_additive_fields() -> None:
    resource = ManagedResource.model_validate(
        {
            **managed_resource().model_dump(),
            "future_registration_epoch": 9,
        }
    )

    assert resource.model_dump()["future_registration_epoch"] == 9
