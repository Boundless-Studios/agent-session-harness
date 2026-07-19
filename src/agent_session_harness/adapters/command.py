"""Bounded JSON protocol for executable checkpoint adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import re
import subprocess
from typing import Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent_session_harness.capsule import HandoffCapsule


_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[-_]?key|credential|password|secret|token)\s*[:=]\s*\S+"
)


def sanitize_error(value: str, *, max_length: int = 240) -> str:
    """Bound an adapter diagnostic and remove assignment-shaped credentials."""

    printable = "".join(
        character if character.isprintable() else " " for character in value
    )
    collapsed = " ".join(printable.split())
    redacted = _SENSITIVE_ASSIGNMENT.sub("credential=[redacted]", collapsed)
    return redacted[:max_length] or "adapter reported failure"


class AdapterOperation(str, Enum):
    WRITE = "write"
    READ = "read"
    ACKNOWLEDGE = "acknowledge"


class AdapterRequest(BaseModel):
    """One versioned idempotent checkpoint adapter request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    operation: AdapterOperation
    idempotency_key: str = Field(min_length=1, max_length=256)
    capsule: HandoffCapsule

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


class AdapterResponse(BaseModel):
    """The exact normalized response returned by every checkpoint adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    fingerprint: str | None = Field(max_length=64)
    retryable: bool
    error: str | None

    @field_validator("error")
    @classmethod
    def bound_error(cls, value: str | None) -> str | None:
        return None if value is None else sanitize_error(value)


class CheckpointAdapter(Protocol):
    name: str

    def execute(self, request: AdapterRequest) -> AdapterResponse: ...


def _failure(error: str, *, retryable: bool) -> AdapterResponse:
    return AdapterResponse(
        ok=False,
        fingerprint=None,
        retryable=retryable,
        error=error,
    )


@dataclass(frozen=True)
class CommandAdapter:
    """Invoke one executable adapter directly, without shell interpolation."""

    name: str
    argv: tuple[str, ...]
    timeout_seconds: float = 5.0
    max_response_bytes: int = 64 * 1024
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("adapter name must not be blank")
        if not self.argv or any(not argument for argument in self.argv):
            raise ValueError("adapter argv must contain non-empty arguments")
        if self.timeout_seconds <= 0:
            raise ValueError("adapter timeout must be positive")
        if self.max_response_bytes <= 0:
            raise ValueError("adapter response bound must be positive")

    def execute(self, request: AdapterRequest) -> AdapterResponse:
        try:
            completed = subprocess.run(
                list(self.argv),
                input=request.canonical_bytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout_seconds,
                shell=False,
                env=dict(self.env),
            )
        except subprocess.TimeoutExpired:
            return _failure("adapter timed out", retryable=True)
        except OSError:
            return _failure("adapter could not be executed", retryable=False)

        if completed.returncode != 0:
            return _failure(
                f"adapter exited with status {completed.returncode}",
                retryable=True,
            )
        if len(completed.stdout) > self.max_response_bytes:
            return _failure(
                f"adapter response exceeded {self.max_response_bytes} bytes",
                retryable=False,
            )
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
            response = AdapterResponse.model_validate(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError):
            return _failure("adapter returned malformed JSON", retryable=False)

        if response.ok and response.fingerprint != request.capsule.fingerprint:
            return _failure("adapter returned the wrong fingerprint", retryable=False)
        return response
