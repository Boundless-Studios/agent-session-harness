"""Durable state and locking contracts for supervised daemons."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from .process_identity import ProcessIdentity
from .secure_files import (
    _open_private,
    atomic_write_private_text,
    private_exists,
    read_private_text,
)

MAX_STATE_BYTES = 64 * 1024
MAX_LOCK_OWNER_BYTES = 16 * 1024


class DaemonLifecyclePhase(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class DaemonDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    daemon_key: str = Field(min_length=1, max_length=256)
    argv: tuple[str, ...] = Field(min_length=1, max_length=256)
    cwd: Path

    @field_validator("argv")
    @classmethod
    def require_bounded_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or len(item) > 4096 for item in value):
            raise ValueError("daemon argv entries must be nonempty and bounded")
        return value


class DaemonLifecycleRecord(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    schema_version: Literal[1] = 1
    daemon_key: str = Field(min_length=1, max_length=256)
    phase: DaemonLifecyclePhase
    generation: int = Field(ge=0)
    changed_at: datetime
    detail: str | None = Field(default=None, max_length=1024)
    process_identity: ProcessIdentity | None = None

    @field_validator("changed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def require_running_identity(self) -> DaemonLifecycleRecord:
        if self.phase is DaemonLifecyclePhase.RUNNING and self.process_identity is None:
            raise ValueError("running state requires process identity")
        return self


class LockOwner(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    pid: int = Field(gt=0)
    purpose: str = Field(min_length=1, max_length=256)
    acquired_at: datetime

    @field_validator("acquired_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)


class LockBusyError(TimeoutError):
    def __init__(self, owner: LockOwner | None):
        super().__init__("daemon lifecycle lock is busy")
        self.owner = owner


class DaemonLifecycleStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def publish(self, record: DaemonLifecycleRecord) -> None:
        encoded = record.model_dump_json() + "\n"
        if len(encoded.encode("utf-8")) > MAX_STATE_BYTES:
            raise ValueError(f"daemon lifecycle state exceeds {MAX_STATE_BYTES} bytes")
        atomic_write_private_text(self.path, encoded)

    def read(self) -> DaemonLifecycleRecord | None:
        if not private_exists(self.path):
            return None
        try:
            payload = json.loads(
                read_private_text(self.path, max_bytes=MAX_STATE_BYTES)
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError("daemon lifecycle state contains invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("daemon lifecycle state is not an object")
        if payload.get("schema_version") != 1:
            raise RuntimeError("daemon lifecycle state has unsupported schema version")
        try:
            return DaemonLifecycleRecord.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError("daemon lifecycle state violates its contract") from exc


class OwnerDiagnosticLock:
    def __init__(self, path: str | Path, *, purpose: str):
        self.path = Path(path)
        self.purpose = purpose

    def acquire(self, *, timeout: float) -> AbstractContextManager[LockOwner]:
        if timeout < 0:
            raise ValueError("lock timeout cannot be negative")
        return self._acquire(timeout)

    @contextmanager
    def _acquire(self, timeout: float) -> Iterator[LockOwner]:
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError("exclusive file locking is unavailable") from exc

        descriptor = _open_private(self.path, os.O_RDWR, create=True)
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise LockBusyError(self._read_owner()) from None
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

            owner = LockOwner(
                pid=os.getpid(),
                purpose=self.purpose,
                acquired_at=datetime.now(UTC),
            )
            handle.seek(0)
            handle.truncate()
            handle.write(owner.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            try:
                yield owner
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass

    def _read_owner(self) -> LockOwner | None:
        try:
            payload = json.loads(
                read_private_text(self.path, max_bytes=MAX_LOCK_OWNER_BYTES)
            )
            return LockOwner.model_validate(payload)
        except (
            OSError,
            RuntimeError,
            ValueError,
            ValidationError,
            json.JSONDecodeError,
        ):
            return None
