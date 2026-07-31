"""Portable managed-resource and guardian-decision contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .process_identity import ProcessIdentity, ProcessState

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
    evidence: Annotated[list[GuardianEvidence], Field(min_length=1, max_length=31)]
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _timezone_aware(value)


class GuardianDecision(_ContractModel):
    schema_version: SchemaVersion = 1
    action: GuardianAction
    reason_code: BoundedText
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
        return _decision(observation, GuardianAction.ALERT, "inspection_failed")

    if (
        observation.process_state is ProcessState.UNKNOWN
        or observation.process_identity_state is ProcessIdentityState.UNKNOWN
        or observation.worktree_state is WorktreeState.UNKNOWN
        or observation.managed_owner_state is ManagedOwnerState.UNKNOWN
        or observation.lease_state is LeaseState.UNKNOWN
    ):
        return _decision(observation, GuardianAction.ALERT, "ambiguous_identity")

    if observation.managed_owner_state is ManagedOwnerState.LIVE:
        return _decision(observation, GuardianAction.RETAIN, "live_managed_owner")

    if (
        observation.worktree_state is WorktreeState.DELETED
        and observation.managed_owner_state is ManagedOwnerState.MISSING
    ):
        return _decision(
            observation,
            GuardianAction.REAP,
            "deleted_worktree_without_live_owner",
        )

    if (
        observation.lease_state is LeaseState.EXPIRED
        and observation.process_state is not ProcessState.RUNNING
        and observation.process_identity_state
        in {ProcessIdentityState.MISSING, ProcessIdentityState.MISMATCH}
    ):
        return _decision(
            observation,
            GuardianAction.REAP,
            "expired_fenced_identity",
        )

    if (
        observation.resource.process_identity is not None
        and observation.process_state in {ProcessState.MISSING, ProcessState.ZOMBIE}
    ):
        return _decision(
            observation,
            GuardianAction.REAP,
            "terminal_managed_child",
        )

    if observation.process_identity_state is ProcessIdentityState.MISMATCH:
        return _decision(
            observation,
            GuardianAction.ALERT,
            "process_identity_mismatch",
        )

    if (
        observation.process_state is ProcessState.RUNNING
        and observation.process_identity_state is ProcessIdentityState.EXACT
    ):
        return _decision(
            observation,
            GuardianAction.ALERT,
            "live_unowned_resource",
        )

    return _decision(observation, GuardianAction.ALERT, "insufficient_evidence")


def _decision(
    observation: GuardianObservation,
    action: GuardianAction,
    reason_code: str,
) -> GuardianDecision:
    return GuardianDecision(
        action=action,
        reason_code=reason_code,
        evidence=[
            *observation.evidence,
            GuardianEvidence(
                source="guardian-policy",
                code=reason_code,
            ),
        ],
        observed_at=observation.observed_at,
    )
