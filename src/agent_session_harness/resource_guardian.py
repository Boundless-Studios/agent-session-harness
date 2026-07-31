"""Portable managed-resource and guardian-decision contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .process_identity import ManagedResourceReference, ProcessIdentity, ProcessState

SchemaVersion = Literal[1]
BoundedText = Annotated[str, Field(min_length=1, max_length=512)]


def _timezone_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class GuardianAction(StrEnum):
    RETAIN = "retain"
    ALERT = "alert"
    REAP = "reap"


class GuardianReasonCode(StrEnum):
    INSPECTION_FAILED = "inspection_failed"
    AMBIGUOUS_IDENTITY = "ambiguous_identity"
    LIVE_MANAGED_OWNER = "live_managed_owner"
    DELETED_WORKTREE_WITHOUT_LIVE_OWNER = "deleted_worktree_without_live_owner"
    EXPIRED_FENCED_IDENTITY = "expired_fenced_identity"
    TERMINAL_MANAGED_CHILD = "terminal_managed_child"
    PROCESS_IDENTITY_MISMATCH = "process_identity_mismatch"
    LIVE_UNOWNED_RESOURCE = "live_unowned_resource"
    ACTIVE_OWNER_LEASE = "active_owner_lease"
    IDENTITY_STATE_MISMATCH = "identity_state_mismatch"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class WorktreeState(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    PRESENT = "present"
    DELETED = "deleted"
    UNKNOWN = "unknown"


class ManagedOwnerState(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    LIVE = "live"
    MISSING = "missing"
    UNKNOWN = "unknown"


class LeaseState(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    ACTIVE = "active"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class ProcessIdentityState(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    EXACT = "exact"
    MISSING = "missing"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class WorktreeIdentity(_ContractModel):
    schema_version: SchemaVersion = 1
    canonical_path: BoundedText


class OwnerLeaseIdentity(_ContractModel):
    schema_version: SchemaVersion = 1
    owner_id: BoundedText
    fencing_token: Annotated[int, Field(ge=1)]
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _timezone_aware(value)


class ManagedResource(_ContractModel):
    schema_version: SchemaVersion = 1
    kind: BoundedText
    resource_key: BoundedText
    process_identity: ProcessIdentity | None = None
    worktree_identity: WorktreeIdentity | None = None
    owner_lease: OwnerLeaseIdentity | None = None
    cleanup_adapter: BoundedText

    @model_validator(mode="after")
    def require_identity(self) -> Self:
        if (
            self.process_identity is None
            and self.worktree_identity is None
            and self.owner_lease is None
        ):
            raise ValueError("managed resource requires at least one identity")
        return self


class GuardianEvidence(_ContractModel):
    schema_version: SchemaVersion = 1
    source: BoundedText
    code: BoundedText
    detail: BoundedText | None = None


class GuardianObservation(_ContractModel):
    schema_version: SchemaVersion = 1
    resource: ManagedResource
    process_state: ProcessState | None = None
    process_identity_state: ProcessIdentityState = ProcessIdentityState.NOT_APPLICABLE
    worktree_state: WorktreeState = WorktreeState.NOT_APPLICABLE
    managed_owner_state: ManagedOwnerState = ManagedOwnerState.NOT_APPLICABLE
    lease_state: LeaseState = LeaseState.NOT_APPLICABLE
    evidence: Annotated[list[GuardianEvidence], Field(min_length=1, max_length=26)]
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _timezone_aware(value)


class GuardianDecision(_ContractModel):
    schema_version: SchemaVersion = 1
    resource: ManagedResourceReference
    action: GuardianAction
    reason_code: GuardianReasonCode
    evidence: Annotated[list[GuardianEvidence], Field(min_length=1, max_length=32)]
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _timezone_aware(value)


def decide_guardian_action(observation: GuardianObservation) -> GuardianDecision:
    """Return a fail-safe action without executing resource cleanup."""

    evidence_codes = {item.code for item in observation.evidence}
    if "inspection_failed" in evidence_codes:
        return _decision(
            observation,
            GuardianAction.ALERT,
            GuardianReasonCode.INSPECTION_FAILED,
        )

    if (
        observation.process_state is ProcessState.UNKNOWN
        or observation.process_identity_state is ProcessIdentityState.UNKNOWN
        or observation.worktree_state is WorktreeState.UNKNOWN
        or observation.managed_owner_state is ManagedOwnerState.UNKNOWN
        or observation.lease_state is LeaseState.UNKNOWN
    ):
        return _decision(
            observation,
            GuardianAction.ALERT,
            GuardianReasonCode.AMBIGUOUS_IDENTITY,
        )

    if _has_identity_state_mismatch(observation):
        return _decision(
            observation,
            GuardianAction.ALERT,
            GuardianReasonCode.IDENTITY_STATE_MISMATCH,
        )

    if observation.managed_owner_state is ManagedOwnerState.LIVE:
        return _decision(
            observation,
            GuardianAction.RETAIN,
            GuardianReasonCode.LIVE_MANAGED_OWNER,
        )

    if (
        observation.process_state is ProcessState.RUNNING
        and observation.process_identity_state is ProcessIdentityState.EXACT
    ):
        return _decision(
            observation,
            GuardianAction.ALERT,
            GuardianReasonCode.LIVE_UNOWNED_RESOURCE,
        )

    if (
        observation.lease_state is LeaseState.EXPIRED
        and observation.process_state is not ProcessState.RUNNING
        and observation.process_identity_state is ProcessIdentityState.MISMATCH
    ):
        return _decision(
            observation,
            GuardianAction.REAP,
            GuardianReasonCode.EXPIRED_FENCED_IDENTITY,
        )

    if observation.process_identity_state is ProcessIdentityState.MISMATCH:
        return _decision(
            observation,
            GuardianAction.ALERT,
            GuardianReasonCode.PROCESS_IDENTITY_MISMATCH,
        )

    if observation.lease_state is LeaseState.ACTIVE:
        return _decision(
            observation,
            GuardianAction.ALERT,
            GuardianReasonCode.ACTIVE_OWNER_LEASE,
        )

    if (
        observation.worktree_state is WorktreeState.DELETED
        and observation.managed_owner_state is ManagedOwnerState.MISSING
        and observation.process_state in {ProcessState.MISSING, ProcessState.ZOMBIE}
    ):
        return _decision(
            observation,
            GuardianAction.REAP,
            GuardianReasonCode.DELETED_WORKTREE_WITHOUT_LIVE_OWNER,
        )

    if (
        observation.resource.process_identity is not None
        and observation.process_state in {ProcessState.MISSING, ProcessState.ZOMBIE}
    ):
        return _decision(
            observation,
            GuardianAction.REAP,
            GuardianReasonCode.TERMINAL_MANAGED_CHILD,
        )

    return _decision(
        observation,
        GuardianAction.ALERT,
        GuardianReasonCode.INSUFFICIENT_EVIDENCE,
    )


def _decision(
    observation: GuardianObservation,
    action: GuardianAction,
    reason_code: GuardianReasonCode,
) -> GuardianDecision:
    return GuardianDecision(
        resource=ManagedResourceReference(
            kind=observation.resource.kind,
            resource_key=observation.resource.resource_key,
        ),
        action=action,
        reason_code=reason_code,
        evidence=[
            *observation.evidence,
            *_state_evidence(observation),
            GuardianEvidence(
                source="guardian-policy",
                code=reason_code,
            ),
        ],
        observed_at=observation.observed_at,
    )


def _has_identity_state_mismatch(observation: GuardianObservation) -> bool:
    resource = observation.resource
    process_state_is_incomplete = resource.process_identity is not None and (
        observation.process_state is None
        or observation.process_identity_state is ProcessIdentityState.NOT_APPLICABLE
    )
    process_state_is_contradictory = (
        observation.process_state is ProcessState.MISSING
        and observation.process_identity_state is ProcessIdentityState.EXACT
    )
    return (
        process_state_is_incomplete
        or process_state_is_contradictory
        or (
            resource.process_identity is None
            and (
                observation.process_state is not None
                or observation.process_identity_state
                is not ProcessIdentityState.NOT_APPLICABLE
            )
        )
        or (
            (resource.worktree_identity is None)
            != (observation.worktree_state is WorktreeState.NOT_APPLICABLE)
        )
        or (
            (resource.owner_lease is None)
            != (observation.lease_state is LeaseState.NOT_APPLICABLE)
        )
    )


def _state_evidence(observation: GuardianObservation) -> list[GuardianEvidence]:
    process_state = (
        observation.process_state.value
        if observation.process_state is not None
        else "not_applicable"
    )
    return [
        GuardianEvidence(
            source="guardian-observation", code=f"process_{process_state}"
        ),
        GuardianEvidence(
            source="guardian-observation",
            code=f"process_identity_{observation.process_identity_state.value}",
        ),
        GuardianEvidence(
            source="guardian-observation",
            code=f"worktree_{observation.worktree_state.value}",
        ),
        GuardianEvidence(
            source="guardian-observation",
            code=f"managed_owner_{observation.managed_owner_state.value}",
        ),
        GuardianEvidence(
            source="guardian-observation",
            code=f"lease_{observation.lease_state.value}",
        ),
    ]
