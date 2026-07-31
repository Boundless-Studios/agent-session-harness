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
    worktree_state: WorktreeState = WorktreeState.NOT_APPLICABLE
    managed_owner_state: ManagedOwnerState = ManagedOwnerState.NOT_APPLICABLE
    lease_state: LeaseState = LeaseState.NOT_APPLICABLE
    evidence: Annotated[list[GuardianEvidence], Field(min_length=1, max_length=32)]
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
