"""Sanitized lifecycle event schema."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import EventType, Runtime


ACTIVITY_EVENT_TYPES = frozenset(
    {
        EventType.TURN_STARTED,
        EventType.TURN_IDLE,
        EventType.TOOL_STARTED,
        EventType.TOOL_FINISHED,
        EventType.TOOL_FAILED,
        EventType.SUBAGENT_STARTED,
        EventType.SUBAGENT_FINISHED,
        EventType.CRITICAL_ENTERED,
        EventType.CRITICAL_EXITED,
    }
)

# Portable mutation boundaries understood by every supported runtime. Critical
# lifecycle records must use this shared vocabulary so arbitrary tool metadata
# cannot hold the supervisor in a permanently busy state.
ALLOWED_CRITICAL_SECTION_NAMES = frozenset(
    {
        "checkpoint",
        "git-write",
        "database-migration",
        "deployment",
        "external-effect",
        "process-launch",
    }
)

_CRITICAL_SECTION_EVENT_TYPES = frozenset(
    {EventType.CRITICAL_ENTERED, EventType.CRITICAL_EXITED}
)


class LifecycleEvent(BaseModel):
    """One privacy-preserving runtime lifecycle transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    event_id: str = Field(min_length=1, max_length=160)
    runtime: Runtime
    chain_id: str = Field(min_length=1, max_length=160)
    conversation_id: str = Field(min_length=1, max_length=160)
    generation: int = Field(ge=0)
    event_type: EventType
    timestamp: datetime
    cwd: Path
    owner_pid: int = Field(gt=0)
    activity_id: str | None = Field(default=None, max_length=160)
    name: str | None = Field(default=None, max_length=128)

    @field_validator("timestamp")
    @classmethod
    def require_aware_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("cwd", mode="before")
    @classmethod
    def require_absolute_cwd(cls, value: object) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            raise ValueError("cwd must be absolute")
        return path.resolve()

    @field_validator("activity_id", "name")
    @classmethod
    def reject_blank_optional_identifiers(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("identifier must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_activity_identity(self) -> "LifecycleEvent":
        if self.event_type in ACTIVITY_EVENT_TYPES and self.activity_id is None:
            raise ValueError("activity_id is required for activity events")
        if (
            self.event_type in _CRITICAL_SECTION_EVENT_TYPES
            and self.name not in ALLOWED_CRITICAL_SECTION_NAMES
        ):
            allowed = ", ".join(sorted(ALLOWED_CRITICAL_SECTION_NAMES))
            raise ValueError(f"critical section name must be one of: {allowed}")
        return self
