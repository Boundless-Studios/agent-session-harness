"""Locked, idempotent JSONL queue for checkpoint mirror requests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent_session_harness.adapters.command import (
    AdapterRequest,
    CheckpointAdapter,
    sanitize_error,
)


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.chmod(path, 0o600)
    with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError("exclusive file locking is unavailable") from exc
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


class OutboxEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    adapter: str = Field(min_length=1, max_length=160)
    request: AdapterRequest
    enqueued_at: datetime


class DeadLetterEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    adapter: str = Field(min_length=1, max_length=160)
    request: AdapterRequest
    enqueued_at: datetime
    failed_at: datetime
    error: str = Field(min_length=1, max_length=240)

    @field_validator("error", mode="before")
    @classmethod
    def clean_error(cls, value: object) -> str:
        return sanitize_error(str(value))


@dataclass(frozen=True)
class ReplayResult:
    attempted: int
    succeeded: int
    retained: int
    dead_lettered: int


class MirrorOutbox:
    """Persist mirror operations and replay them in their enqueue order."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        dead_letter_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.dead_letter_path = (
            Path(dead_letter_path)
            if dead_letter_path is not None
            else self.path.with_suffix(self.path.suffix + ".dead")
        )
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @property
    def depth(self) -> int:
        return len(self.pending())

    def enqueue(self, adapter: str, request: AdapterRequest) -> bool:
        with _exclusive_lock(self.lock_path):
            entries = self._read(self.path, OutboxEntry)
            pair = (adapter, request.idempotency_key)
            if any(
                (entry.adapter, entry.request.idempotency_key) == pair
                for entry in entries
            ):
                return False
            entry = OutboxEntry(
                schema_version=1,
                adapter=adapter,
                request=request,
                enqueued_at=datetime.now(timezone.utc),
            )
            self._append(self.path, entry)
            return True

    def pending(self) -> tuple[OutboxEntry, ...]:
        with _exclusive_lock(self.lock_path):
            return tuple(self._read(self.path, OutboxEntry))

    def dead_letters(self) -> tuple[DeadLetterEntry, ...]:
        with _exclusive_lock(self.lock_path):
            return tuple(self._read(self.dead_letter_path, DeadLetterEntry))

    def replay(
        self,
        adapters: Mapping[str, CheckpointAdapter],
    ) -> ReplayResult:
        with _exclusive_lock(self.lock_path):
            entries = self._read(self.path, OutboxEntry)
            retained: list[OutboxEntry] = []
            dead_letters: list[DeadLetterEntry] = []
            succeeded = 0
            for entry in entries:
                adapter = adapters.get(entry.adapter)
                if adapter is None:
                    retained.append(entry)
                    continue
                try:
                    response = adapter.execute(entry.request)
                except Exception:
                    retained.append(entry)
                    continue

                expected = entry.request.capsule.fingerprint
                if response.ok and response.fingerprint == expected:
                    succeeded += 1
                elif response.ok:
                    dead_letters.append(
                        self._dead_letter(
                            entry,
                            "adapter returned the wrong fingerprint",
                        )
                    )
                elif response.retryable:
                    retained.append(entry)
                else:
                    dead_letters.append(
                        self._dead_letter(
                            entry,
                            response.error or "adapter reported failure",
                        )
                    )

            for dead_letter in dead_letters:
                self._append(self.dead_letter_path, dead_letter)
            self._replace(self.path, retained)
            return ReplayResult(
                attempted=len(entries),
                succeeded=succeeded,
                retained=len(retained),
                dead_lettered=len(dead_letters),
            )

    @staticmethod
    def _dead_letter(entry: OutboxEntry, error: str) -> DeadLetterEntry:
        return DeadLetterEntry(
            schema_version=1,
            adapter=entry.adapter,
            request=entry.request,
            enqueued_at=entry.enqueued_at,
            failed_at=datetime.now(timezone.utc),
            error=error,
        )

    @staticmethod
    def _encoded(model: BaseModel) -> str:
        return json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def _append(cls, path: Path, model: BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.chmod(path, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(cls._encoded(model) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _read(path: Path, model_type: type[BaseModel]) -> list:
        if not path.exists():
            return []
        result = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                result.append(model_type.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                raise ValueError(f"invalid queue record at line {line_number}") from exc
        return result

    @classmethod
    def _replace(cls, path: Path, entries: list[OutboxEntry]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
        )
        temporary_path = Path(temporary_name)
        try:
            os.chmod(temporary_path, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(cls._encoded(entry) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            os.chmod(path, 0o600)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
