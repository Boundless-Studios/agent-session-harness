"""Versioned process identity and observation contracts."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol

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


@dataclass(frozen=True)
class NativeProcessRecord:
    pid: int
    parent_pid: int | None
    state: ProcessState
    opaque_start_token: str
    executable_identity: str | None


@dataclass(frozen=True)
class NativeReadResult:
    state: ProcessState
    record: NativeProcessRecord | None
    evidence_code: str

    @classmethod
    def present(cls, record: NativeProcessRecord) -> NativeReadResult:
        return cls(
            state=record.state,
            record=record,
            evidence_code="native_record",
        )

    @classmethod
    def missing(cls) -> NativeReadResult:
        return cls(
            state=ProcessState.MISSING,
            record=None,
            evidence_code="process_missing",
        )

    @classmethod
    def unknown(cls, code: str) -> NativeReadResult:
        return cls(
            state=ProcessState.UNKNOWN,
            record=None,
            evidence_code=code,
        )


class NativeProcessReader(Protocol):
    platform: ProcessPlatform

    def read(self, pid: int) -> NativeReadResult: ...


class ProcessInspector:
    def __init__(
        self,
        reader: NativeProcessReader,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.reader = reader
        self.clock = clock

    def capture(self, pid: int) -> ProcessIdentity | None:
        result = self.reader.read(pid)
        record = result.record
        if (
            result.state in {ProcessState.MISSING, ProcessState.UNKNOWN}
            or record is None
            or not record.executable_identity
        ):
            return None
        return self._identity(record)

    def observe(
        self,
        expected: ProcessIdentity,
        managed_resource: ManagedResourceReference | None = None,
    ) -> ProcessObservation:
        if expected.platform is not self.reader.platform:
            return self._observation(
                expected,
                ProcessState.UNKNOWN,
                managed_resource=managed_resource,
                evidence=[
                    ProcessEvidence(
                        source="adapter",
                        code="platform_mismatch",
                    )
                ],
            )

        result = self.reader.read(expected.pid)
        record = result.record
        if record is None:
            return self._observation(
                expected,
                result.state,
                managed_resource=managed_resource,
                evidence=[
                    ProcessEvidence(
                        source="native",
                        code=result.evidence_code,
                    )
                ],
            )
        if record.opaque_start_token != expected.opaque_start_token:
            return self._observation(
                expected,
                ProcessState.MISSING,
                managed_resource=managed_resource,
                evidence=[
                    ProcessEvidence(
                        source="comparison",
                        code="pid_reused",
                    )
                ],
            )
        if record.state is ProcessState.ZOMBIE:
            return self._observation(
                expected,
                ProcessState.ZOMBIE,
                managed_resource=managed_resource,
            )
        if not record.executable_identity:
            return self._observation(
                expected,
                ProcessState.UNKNOWN,
                managed_resource=managed_resource,
                evidence=[
                    ProcessEvidence(
                        source="native",
                        code="executable_unavailable",
                    )
                ],
            )

        evidence: list[ProcessEvidence] = []
        if record.executable_identity != expected.executable_identity:
            evidence.append(
                ProcessEvidence(
                    source="comparison",
                    code="executable_changed",
                    detail=record.executable_identity,
                )
            )
        parent_identity, parent_unavailable = self._capture_parent(record.parent_pid)
        if parent_unavailable:
            evidence.append(
                ProcessEvidence(
                    source="native",
                    code="parent_unavailable",
                )
            )
        return self._observation(
            expected,
            ProcessState.RUNNING,
            parent_identity=parent_identity,
            managed_resource=managed_resource,
            evidence=evidence,
        )

    def _capture_parent(
        self,
        pid: int | None,
    ) -> tuple[ProcessIdentity | None, bool]:
        if pid is None or pid <= 0:
            return None, False
        result = self.reader.read(pid)
        record = result.record
        if (
            record is None
            or record.state in {ProcessState.MISSING, ProcessState.UNKNOWN}
            or not record.executable_identity
        ):
            return None, True
        return self._identity(record), False

    def _identity(self, record: NativeProcessRecord) -> ProcessIdentity:
        if record.executable_identity is None:
            raise ValueError("executable identity is unavailable")
        return ProcessIdentity(
            platform=self.reader.platform,
            pid=record.pid,
            opaque_start_token=record.opaque_start_token,
            executable_identity=record.executable_identity,
            captured_at=self.clock(),
        )

    def _observation(
        self,
        identity: ProcessIdentity,
        state: ProcessState,
        *,
        parent_identity: ProcessIdentity | None = None,
        managed_resource: ManagedResourceReference | None = None,
        evidence: list[ProcessEvidence] | None = None,
    ) -> ProcessObservation:
        return ProcessObservation(
            identity=identity,
            state=state,
            parent_identity=parent_identity,
            managed_resource=managed_resource,
            evidence=evidence or [],
            observed_at=self.clock(),
        )


class LinuxProcessReader:
    platform = ProcessPlatform.LINUX

    def __init__(
        self,
        read_stat: Callable[[int], str] | None = None,
        read_executable: Callable[[int], str] | None = None,
    ) -> None:
        self._read_stat = read_stat or self._read_proc_stat
        self._read_executable = read_executable or self._read_proc_executable

    def read(self, pid: int) -> NativeReadResult:
        try:
            encoded = self._read_stat(pid)
        except (FileNotFoundError, ProcessLookupError):
            return NativeReadResult.missing()
        except (OSError, ValueError):
            return NativeReadResult.unknown("stat_unavailable")

        record = self._parse_stat(pid, encoded)
        if record is None:
            return NativeReadResult.unknown("stat_malformed")
        if record.state is ProcessState.ZOMBIE:
            return NativeReadResult.present(record)

        try:
            executable = self._read_executable(pid)
        except (FileNotFoundError, ProcessLookupError):
            return NativeReadResult.missing()
        except (OSError, ValueError):
            return NativeReadResult.unknown("executable_unavailable")
        if not executable:
            return NativeReadResult.unknown("executable_unavailable")
        return NativeReadResult.present(
            NativeProcessRecord(
                pid=record.pid,
                parent_pid=record.parent_pid,
                state=record.state,
                opaque_start_token=record.opaque_start_token,
                executable_identity=executable,
            )
        )

    @staticmethod
    def _parse_stat(
        expected_pid: int,
        encoded: str,
    ) -> NativeProcessRecord | None:
        command_start = encoded.find("(")
        command_end = encoded.rfind(")")
        if command_start < 1 or command_end <= command_start:
            return None
        try:
            parsed_pid = int(encoded[:command_start].strip())
        except ValueError:
            return None
        if parsed_pid != expected_pid:
            return None
        fields = encoded[command_end + 1 :].split()
        if len(fields) <= 19 or len(fields[0]) != 1:
            return None
        try:
            parent_pid = int(fields[1])
            start_ticks = int(fields[19])
        except ValueError:
            return None
        if parent_pid < 0 or start_ticks <= 0:
            return None
        return NativeProcessRecord(
            pid=expected_pid,
            parent_pid=parent_pid,
            state=(ProcessState.ZOMBIE if fields[0] == "Z" else ProcessState.RUNNING),
            opaque_start_token=f"linux:{start_ticks}",
            executable_identity=None,
        )

    @staticmethod
    def _read_proc_stat(pid: int) -> str:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")

    @staticmethod
    def _read_proc_executable(pid: int) -> str:
        return os.readlink(f"/proc/{pid}/exe")
