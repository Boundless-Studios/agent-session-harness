"""Locked, idempotent JSONL queue for checkpoint mirror requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent_session_harness.adapters.command import (
    AdapterRequest,
    CheckpointAdapter,
    sanitize_error,
)
from agent_session_harness.secure_files import (
    append_private_text,
    atomic_write_private_text,
    exclusive_lock,
    private_exists,
    read_private_text,
)


DEFAULT_REPLAY_ATTEMPTS = 100
DEFAULT_MAX_QUEUE_BYTES = 8 * 1024 * 1024


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
        max_queue_bytes: int = DEFAULT_MAX_QUEUE_BYTES,
    ) -> None:
        if max_queue_bytes <= 0:
            raise ValueError("maximum queue bytes must be positive")
        self.path = Path(path)
        self.dead_letter_path = (
            Path(dead_letter_path)
            if dead_letter_path is not None
            else self.path.with_suffix(self.path.suffix + ".dead")
        )
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.max_queue_bytes = max_queue_bytes

    @property
    def depth(self) -> int:
        return len(self.pending())

    def enqueue(self, adapter: str, request: AdapterRequest) -> bool:
        with exclusive_lock(self.lock_path):
            entries = self._read(
                self.path,
                OutboxEntry,
                max_bytes=self.max_queue_bytes,
            )
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
            encoded_size = len((self._encoded(entry) + "\n").encode("utf-8"))
            existing_size = sum(
                len((self._encoded(current) + "\n").encode("utf-8"))
                for current in entries
            )
            if existing_size + encoded_size > self.max_queue_bytes:
                raise ValueError(f"mirror outbox exceeds {self.max_queue_bytes} bytes")
            self._append(self.path, entry)
            return True

    def pending(self) -> tuple[OutboxEntry, ...]:
        with exclusive_lock(self.lock_path):
            return tuple(
                self._read(
                    self.path,
                    OutboxEntry,
                    max_bytes=self.max_queue_bytes,
                )
            )

    def dead_letters(self) -> tuple[DeadLetterEntry, ...]:
        with exclusive_lock(self.lock_path):
            return tuple(
                self._read(
                    self.dead_letter_path,
                    DeadLetterEntry,
                    max_bytes=self.max_queue_bytes,
                )
            )

    def replay(
        self,
        adapters: Mapping[str, CheckpointAdapter],
        *,
        max_attempts: int = DEFAULT_REPLAY_ATTEMPTS,
    ) -> ReplayResult:
        if max_attempts <= 0:
            raise ValueError("maximum replay attempts must be positive")
        with exclusive_lock(self.lock_path):
            entries = self._read(
                self.path,
                OutboxEntry,
                max_bytes=self.max_queue_bytes,
            )
            batch = entries[:max_attempts]
            retained: list[OutboxEntry] = []
            dead_letters: list[DeadLetterEntry] = []
            succeeded = 0
            for entry in batch:
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

            retained.extend(entries[len(batch) :])

            existing_dead_letters = self._read(
                self.dead_letter_path,
                DeadLetterEntry,
                max_bytes=self.max_queue_bytes,
            )
            dead_letter_size = sum(
                len((self._encoded(entry) + "\n").encode("utf-8"))
                for entry in existing_dead_letters
            )
            new_dead_letter_size = sum(
                len((self._encoded(entry) + "\n").encode("utf-8"))
                for entry in dead_letters
            )
            if dead_letter_size + new_dead_letter_size > self.max_queue_bytes:
                raise ValueError(
                    f"mirror dead-letter queue exceeds {self.max_queue_bytes} bytes"
                )
            for dead_letter in dead_letters:
                self._append(self.dead_letter_path, dead_letter)
            self._replace(self.path, retained)
            return ReplayResult(
                attempted=len(batch),
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
        append_private_text(path, cls._encoded(model) + "\n")

    @staticmethod
    def _read(
        path: Path,
        model_type: type[BaseModel],
        *,
        max_bytes: int,
    ) -> list:
        if not private_exists(path):
            return []
        result = []
        for line_number, line in enumerate(
            read_private_text(path, max_bytes=max_bytes).splitlines(),
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
        encoded = "".join(cls._encoded(entry) + "\n" for entry in entries)
        atomic_write_private_text(path, encoded)
