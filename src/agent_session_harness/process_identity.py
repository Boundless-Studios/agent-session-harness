"""Versioned process identity and observation contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SchemaVersion = Literal[1]
BoundedText = Annotated[str, Field(min_length=1, max_length=512)]


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class ProcessPlatform(StrEnum):
    LINUX = "linux"
    DARWIN = "darwin"


class ProcessState(StrEnum):
    RUNNING = "running"
    ZOMBIE = "zombie"
    MISSING = "missing"
    UNKNOWN = "unknown"


class ProcessIdentity(_ContractModel):
    schema_version: SchemaVersion = 1
    platform: ProcessPlatform
    pid: Annotated[int, Field(gt=0)]
    opaque_start_token: BoundedText
    executable_identity: BoundedText
    captured_at: datetime

    @field_validator("captured_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)


class ManagedResourceReference(_ContractModel):
    schema_version: SchemaVersion = 1
    kind: BoundedText
    resource_key: BoundedText


class ProcessEvidence(_ContractModel):
    schema_version: SchemaVersion = 1
    source: BoundedText
    code: BoundedText
    detail: BoundedText | None = None


class ProcessObservation(_ContractModel):
    schema_version: SchemaVersion = 1
    identity: ProcessIdentity
    state: ProcessState
    parent_identity: ProcessIdentity | None = None
    managed_resource: ManagedResourceReference | None = None
    evidence: Annotated[list[ProcessEvidence], Field(max_length=32)]
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        return _require_timezone(value)


def same_process(left: ProcessIdentity, right: ProcessIdentity) -> bool:
    """Return whether two identities describe the same process lifetime."""

    return (
        left.platform == right.platform
        and left.pid == right.pid
        and left.opaque_start_token == right.opaque_start_token
    )
