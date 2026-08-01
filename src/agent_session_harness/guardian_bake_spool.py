"""Private, bounded, restart-safe guardian bake report spool."""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .guardian_bake import GuardianBakeReport
from .secure_files import (
    atomic_write_private_text,
    exclusive_lock,
    private_exists,
    read_private_text,
)

DEFAULT_MAX_BYTES = 8 * 1_048_576
DEFAULT_MAX_RECORDS = 1024


class GuardianBakeSpoolRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: Annotated[str, Field(min_length=16, max_length=256)]
    report: GuardianBakeReport
    first_seen_at: datetime
    last_seen_at: datetime
    repeat_count: int = Field(ge=1)
    delivered_at: datetime | None = None


class _SpoolDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    records: list[GuardianBakeSpoolRecord]

    @model_validator(mode="after")
    def require_unique_record_ids(self) -> Self:
        record_ids = [record.record_id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("guardian bake spool contains duplicate record IDs")
        return self


class GuardianBakeSpool:
    """Persist reports before delivery without dropping pending evidence."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_records: int = DEFAULT_MAX_RECORDS,
    ):
        if max_bytes <= 0 or max_records <= 0:
            raise ValueError("spool bounds must be positive")
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.max_bytes = max_bytes
        self.max_records = max_records

    def append(
        self,
        report: GuardianBakeReport,
        *,
        now: datetime | None = None,
    ) -> GuardianBakeSpoolRecord:
        observed_at = (now or datetime.now(UTC)).astimezone(UTC)
        with exclusive_lock(self.lock_path):
            document = self._read()
            records = list(document.records)
            if (
                records
                and records[-1].delivered_at is None
                and records[-1].report.deduplication_key == report.deduplication_key
            ):
                compacted = records[-1].model_copy(
                    update={
                        "report": report,
                        "last_seen_at": observed_at,
                        "repeat_count": records[-1].repeat_count + 1,
                    }
                )
                records[-1] = compacted
                self._write(_SpoolDocument(records=records))
                return compacted

            while len(records) >= self.max_records:
                delivered_index = next(
                    (
                        index
                        for index, record in enumerate(records)
                        if record.delivered_at is not None
                    ),
                    None,
                )
                if delivered_index is None:
                    raise ValueError(
                        "guardian bake spool is full of undelivered evidence"
                    )
                del records[delivered_index]

            created = GuardianBakeSpoolRecord(
                record_id=secrets.token_urlsafe(24),
                report=report,
                first_seen_at=observed_at,
                last_seen_at=observed_at,
                repeat_count=1,
            )
            records.append(created)
            self._write(_SpoolDocument(records=records))
            return created

    def acknowledge(
        self,
        record_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        delivered_at = (now or datetime.now(UTC)).astimezone(UTC)
        with exclusive_lock(self.lock_path):
            document = self._read()
            found = False
            records: list[GuardianBakeSpoolRecord] = []
            for record in document.records:
                if record.record_id == record_id:
                    found = True
                    record = record.model_copy(update={"delivered_at": delivered_at})
                records.append(record)
            if not found:
                raise KeyError(f"unknown guardian bake record: {record_id}")
            self._write(_SpoolDocument(records=records))

    def list(self) -> list[GuardianBakeSpoolRecord]:
        with exclusive_lock(self.lock_path):
            return list(self._read().records)

    def pending(self) -> list[GuardianBakeSpoolRecord]:
        return [record for record in self.list() if record.delivered_at is None]

    def _read(self) -> _SpoolDocument:
        if not private_exists(self.path):
            return _SpoolDocument(records=[])
        try:
            payload = json.loads(read_private_text(self.path, max_bytes=self.max_bytes))
        except json.JSONDecodeError as exc:
            raise RuntimeError("guardian bake spool contains invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("guardian bake spool is not an object")
        if payload.get("schema_version") != 1:
            raise RuntimeError("guardian bake spool has unsupported schema version")
        try:
            document = _SpoolDocument.model_validate(payload)
        except ValidationError as exc:
            message = str(exc)
            if "duplicate record IDs" in message:
                raise RuntimeError(
                    "guardian bake spool contains duplicate record IDs"
                ) from exc
            raise RuntimeError("guardian bake spool violates its contract") from exc
        if len(document.records) > self.max_records:
            raise RuntimeError("guardian bake spool exceeds its record bound")
        return document

    def _write(self, document: _SpoolDocument) -> None:
        encoded = document.model_dump_json() + "\n"
        if len(encoded.encode("utf-8")) > self.max_bytes:
            raise ValueError(f"guardian bake spool exceeds {self.max_bytes} bytes")
        atomic_write_private_text(self.path, encoded)
