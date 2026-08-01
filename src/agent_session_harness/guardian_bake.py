"""Typed, privacy-preserving evidence contracts for a guardian bake."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

SchemaVersion = Literal[1]
BoundedText = Annotated[str, Field(min_length=1, max_length=512)]

_AUTHORIZATION = re.compile(
    r"""(?ix)
    \bAuthorization\s*(?::|=)\s*
    (?:(?:Bearer|Basic)\s+)?
    (?:"[^"]*"|'[^']*'|\S+)
    """
)
_CREDENTIAL = re.compile(
    r"""(?ix)
    ["']?[A-Za-z0-9_-]*(?:token|api[_-]?key|password|secret)["']?
    \s*(?:=|:)\s*
    (?:"[^"]*"|'[^']*'|[^\s,}]+)
    """
)
_HOME_PATH = re.compile(r"(?:/Users|/home)/[^/\s]+")
_ARGUMENT_VALUE = re.compile(
    r"""(?x)
    (?<!\S)-{1,2}[A-Za-z0-9_-]+
    (?:=|\s+)(?:"[^"]*"|'[^']*'|\S+)
    """
)
_COMMAND_VALUE = re.compile(r"(?is)\b(?:argv|command(?:_line)?)\s*(?:=|:)\s*.*$")


def redact_guardian_text(value: str) -> str:
    """Remove secrets, user path prefixes, and command argument values."""

    redacted = _AUTHORIZATION.sub("Authorization: [REDACTED]", value)
    redacted = _CREDENTIAL.sub("credential=[REDACTED]", redacted)
    redacted = _HOME_PATH.sub("/[HOME]", redacted)
    redacted = _ARGUMENT_VALUE.sub("--argument [REDACTED]", redacted)
    return _COMMAND_VALUE.sub("command=[REDACTED]", redacted)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class ObservationWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    started_at: datetime
    ends_at: datetime

    _normalize_started = field_validator("started_at")(_utc)
    _normalize_ends = field_validator("ends_at")(_utc)

    @model_validator(mode="after")
    def require_positive_window(self) -> Self:
        if self.ends_at <= self.started_at:
            raise ValueError("observation window ends after it starts")
        return self


class ResourceHighWaterMarks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observed: int = Field(ge=0)
    managed: int = Field(ge=0)
    ambiguous: int = Field(ge=0)


class UsageHighWaterMarks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_bytes: int = Field(ge=0)
    cpu_percent: float = Field(ge=0)


class UsageSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_bytes: int = Field(ge=0)
    cpu_percent: float = Field(ge=0)


class GuardianHighWaterMarks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resources: ResourceHighWaterMarks
    usage: UsageHighWaterMarks


class GuardianBakeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: BoundedText
    performed: bool
    live_resource: bool
    evidence: Annotated[list[BoundedText], Field(min_length=1, max_length=64)]

    @field_validator("reason_code")
    @classmethod
    def redact_reason(cls, value: str) -> str:
        return redact_guardian_text(value)

    @field_validator("evidence")
    @classmethod
    def redact_evidence(cls, values: list[str]) -> list[str]:
        return [redact_guardian_text(value) for value in values]


class GuardianBakeError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: BoundedText
    error_type: BoundedText

    @field_validator("stage", "error_type")
    @classmethod
    def redact_text(cls, value: str) -> str:
        return redact_guardian_text(value)


class GuardianBakeReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = 1
    guardian_version: BoundedText
    platform: Literal["darwin", "linux"]
    observation_window: ObservationWindow
    heartbeat_at: datetime
    usage_before: UsageSnapshot
    usage_after: UsageSnapshot
    high_water_marks: GuardianHighWaterMarks
    reap_decisions: Annotated[list[GuardianBakeDecision], Field(max_length=1024)]
    refused_decisions: Annotated[list[GuardianBakeDecision], Field(max_length=1024)]
    errors: Annotated[list[GuardianBakeError], Field(max_length=1024)]
    deduplication_key: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    _normalize_heartbeat = field_validator("heartbeat_at")(_utc)

    @field_validator("guardian_version")
    @classmethod
    def redact_version(cls, value: str) -> str:
        return redact_guardian_text(value)

    @model_validator(mode="after")
    def require_heartbeat_in_window(self) -> Self:
        if not (
            self.observation_window.started_at
            <= self.heartbeat_at
            <= self.observation_window.ends_at
        ):
            raise ValueError("heartbeat must fall within the observation window")
        return self

    @model_validator(mode="after")
    def require_content_deduplication_key(self, info: ValidationInfo) -> Self:
        if info.context and info.context.get("skip_deduplication_check"):
            return self
        if self.deduplication_key != self._content_deduplication_key():
            raise ValueError("deduplication key does not match report content")
        return self

    def _content_deduplication_key(self) -> str:
        content = self.model_dump(
            mode="json",
            exclude={"heartbeat_at", "deduplication_key"},
        )
        encoded = json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def build(cls, **values: object) -> GuardianBakeReport:
        candidate = cls.model_validate(
            {**values, "deduplication_key": "0" * 64},
            context={"skip_deduplication_check": True},
        )
        return candidate.model_copy(
            update={"deduplication_key": candidate._content_deduplication_key()}
        )
