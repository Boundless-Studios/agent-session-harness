# Process Identity Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish versioned macOS/Linux process identity and observation contracts that fail safely under PID reuse, zombies, disappearance, and unavailable native inspection.

**Architecture:** A focused `process_identity.py` module owns public Pydantic contracts, native-reader protocols, Linux/macOS adapters, comparison, and observation. `process.py` retains launch/guardian responsibilities and delegates its legacy persisted fingerprint to the new canonical capture implementation. Higher-level guardian and repository policy remain out of scope.

**Tech Stack:** Python 3.11+, Pydantic v2, `/proc`, macOS `libproc` through `ctypes`, pytest, Ruff, Hatchling.

**Design:** `docs/superpowers/specs/2026-07-30-process-identity-contract-design.md`

---

## File map

- Create `src/agent_session_harness/process_identity.py`: public contracts,
  platform readers, inspector, comparison, and stable legacy fingerprint.
- Create `tests/test_process_identity_contract.py`: schema and comparison tests.
- Create `tests/test_process_identity_observation.py`: reader-neutral observation
  semantics with deterministic fakes.
- Create `tests/test_process_identity_linux.py`: Linux parser fixtures and live
  smoke test.
- Create `tests/test_process_identity_darwin.py`: Darwin native-record fixtures
  and live macOS smoke test.
- Modify `src/agent_session_harness/process.py`: delegate birth-token capture
  and remove duplicate Linux/Darwin parsing.
- Modify `src/agent_session_harness/__init__.py`: export the public contracts
  and inspector.
- Modify `.github/workflows/ci.yml`: include the Darwin smoke test in the
  macOS runtime job.
- Modify `README.md`: document capture/observe usage and safety semantics.

### Task 1: Public schema and identity comparison

**Files:**
- Create: `src/agent_session_harness/process_identity.py`
- Create: `tests/test_process_identity_contract.py`

- [ ] **Step 1: Write failing contract tests**

```python
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_session_harness.process_identity import (
    ManagedResourceReference,
    ProcessIdentity,
    ProcessObservation,
    ProcessPlatform,
    ProcessState,
    same_process,
)


def identity(**changes: object) -> ProcessIdentity:
    values = {
        "schema_version": 1,
        "platform": ProcessPlatform.LINUX,
        "pid": 42,
        "opaque_start_token": "linux:12345",
        "executable_identity": "/usr/bin/python3",
        "captured_at": datetime(2026, 7, 30, tzinfo=UTC),
    }
    values.update(changes)
    return ProcessIdentity.model_validate(values)


def test_contract_accepts_additive_fields_and_round_trips_them() -> None:
    parsed = ProcessIdentity.model_validate(
        {**identity().model_dump(), "future_native_clock": "boot-7"}
    )
    assert parsed.model_dump()["future_native_clock"] == "boot-7"


def test_contract_rejects_unsupported_major_version() -> None:
    with pytest.raises(ValidationError):
        identity(schema_version=2)


def test_same_process_uses_pid_platform_and_start_token_not_executable() -> None:
    original = identity()
    after_exec = identity(executable_identity="/usr/bin/codex")
    successor = identity(opaque_start_token="linux:12346")
    assert same_process(original, after_exec)
    assert not same_process(original, successor)


def test_observation_contains_typed_managed_resource_reference() -> None:
    reference = ManagedResourceReference(
        schema_version=1,
        kind="runtime",
        resource_key="chain-1:0",
    )
    observation = ProcessObservation(
        schema_version=1,
        identity=identity(),
        state=ProcessState.RUNNING,
        managed_resource=reference,
        evidence=[],
        observed_at=datetime(2026, 7, 30, tzinfo=UTC),
    )
    assert observation.managed_resource == reference
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_process_identity_contract.py
```

Expected: collection fails because
`agent_session_harness.process_identity` does not exist.

- [ ] **Step 3: Implement the minimal public models and comparison**

```python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SchemaVersion = Literal[1]
BoundedText = Annotated[str, Field(min_length=1, max_length=512)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class ProcessPlatform(StrEnum):
    LINUX = "linux"
    DARWIN = "darwin"


class ProcessState(StrEnum):
    RUNNING = "running"
    ZOMBIE = "zombie"
    MISSING = "missing"
    UNKNOWN = "unknown"


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value


class ProcessIdentity(ContractModel):
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


class ManagedResourceReference(ContractModel):
    schema_version: SchemaVersion = 1
    kind: BoundedText
    resource_key: BoundedText


class ProcessEvidence(ContractModel):
    schema_version: SchemaVersion = 1
    source: BoundedText
    code: BoundedText
    detail: BoundedText | None = None


class ProcessObservation(ContractModel):
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
    return (
        left.platform == right.platform
        and left.pid == right.pid
        and left.opaque_start_token == right.opaque_start_token
    )
```

- [ ] **Step 4: Run the contract tests and verify GREEN**

Run:

```bash
uv run pytest -q tests/test_process_identity_contract.py
```

Expected: all contract tests pass.

- [ ] **Step 5: Commit the schema**

```bash
git add src/agent_session_harness/process_identity.py tests/test_process_identity_contract.py
git commit -m "feat: define process identity contracts"
```

### Task 2: Reader-neutral observation semantics

**Files:**
- Modify: `src/agent_session_harness/process_identity.py`
- Create: `tests/test_process_identity_observation.py`

- [ ] **Step 1: Write failing observer tests**

```python
from dataclasses import replace
from datetime import UTC, datetime

from agent_session_harness.process_identity import (
    ManagedResourceReference,
    NativeProcessRecord,
    NativeReadResult,
    ProcessInspector,
    ProcessPlatform,
    ProcessState,
)


class FakeReader:
    platform = ProcessPlatform.LINUX

    def __init__(self, results: dict[int, NativeReadResult]) -> None:
        self.results = results

    def read(self, pid: int) -> NativeReadResult:
        return self.results[pid]


NOW = datetime(2026, 7, 30, tzinfo=UTC)
RUNNING = NativeProcessRecord(
    pid=42,
    parent_pid=7,
    state=ProcessState.RUNNING,
    opaque_start_token="linux:100",
    executable_identity="/usr/bin/python3",
)


def inspector(result: NativeReadResult) -> ProcessInspector:
    return ProcessInspector(
        reader=FakeReader(
            {
                42: result,
                7: NativeReadResult.present(
                    NativeProcessRecord(
                        pid=7,
                        parent_pid=1,
                        state=ProcessState.RUNNING,
                        opaque_start_token="linux:10",
                        executable_identity="/sbin/launchd",
                    )
                ),
            }
        ),
        clock=lambda: NOW,
    )


def test_pid_reuse_marks_expected_identity_missing() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[42] = NativeReadResult.present(
        replace(RUNNING, opaque_start_token="linux:101")
    )
    observed = subject.observe(expected)
    assert observed.state is ProcessState.MISSING
    assert [item.code for item in observed.evidence] == ["pid_reused"]


def test_disappeared_process_is_missing() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[42] = NativeReadResult.missing()
    assert subject.observe(expected).state is ProcessState.MISSING


def test_zombie_never_counts_as_running() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[42] = NativeReadResult.present(
        replace(RUNNING, state=ProcessState.ZOMBIE, executable_identity=None)
    )
    assert subject.observe(expected).state is ProcessState.ZOMBIE


def test_unknown_is_preserved_and_resource_reference_is_attached() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[42] = NativeReadResult.unknown("permission_denied")
    reference = ManagedResourceReference(
        kind="runtime",
        resource_key="chain-1:0",
    )
    observed = subject.observe(expected, managed_resource=reference)
    assert observed.state is ProcessState.UNKNOWN
    assert observed.managed_resource == reference


def test_platform_mismatch_is_unknown_without_native_read() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    foreign = expected.model_copy(update={"platform": ProcessPlatform.DARWIN})
    observed = subject.observe(foreign)
    assert observed.state is ProcessState.UNKNOWN
    assert [item.code for item in observed.evidence] == ["platform_mismatch"]


def test_executable_change_preserves_lifetime_and_records_evidence() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[42] = NativeReadResult.present(
        replace(RUNNING, executable_identity="/usr/bin/codex")
    )
    observed = subject.observe(expected)
    assert observed.state is ProcessState.RUNNING
    assert [item.code for item in observed.evidence] == ["executable_changed"]
    assert observed.parent_identity is not None
    assert observed.parent_identity.pid == 7
```

- [ ] **Step 2: Run the observer tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_process_identity_observation.py
```

Expected: collection fails because the native-record, result, and inspector
types are not implemented.

- [ ] **Step 3: Implement native result types and `ProcessInspector`**

Add:

```python
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol


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
    def present(cls, record: NativeProcessRecord) -> "NativeReadResult":
        return cls(record.state, record, "native_record")

    @classmethod
    def missing(cls) -> "NativeReadResult":
        return cls(ProcessState.MISSING, None, "process_missing")

    @classmethod
    def unknown(cls, code: str) -> "NativeReadResult":
        return cls(ProcessState.UNKNOWN, None, code)


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
            return ProcessObservation(
                identity=expected,
                state=ProcessState.UNKNOWN,
                managed_resource=managed_resource,
                evidence=[
                    ProcessEvidence(
                        source="adapter",
                        code="platform_mismatch",
                    )
                ],
                observed_at=self.clock(),
            )
        result = self.reader.read(expected.pid)
        evidence: list[ProcessEvidence] = []
        state = result.state
        parent = None
        record = result.record
        if record is None:
            evidence.append(
                ProcessEvidence(source="native", code=result.evidence_code)
            )
        elif record.opaque_start_token != expected.opaque_start_token:
            state = ProcessState.MISSING
            evidence.append(ProcessEvidence(source="comparison", code="pid_reused"))
        elif record.state is ProcessState.ZOMBIE:
            state = ProcessState.ZOMBIE
        elif not record.executable_identity:
            state = ProcessState.UNKNOWN
            evidence.append(
                ProcessEvidence(source="native", code="executable_unavailable")
            )
        else:
            state = ProcessState.RUNNING
            if record.executable_identity != expected.executable_identity:
                evidence.append(
                    ProcessEvidence(
                        source="comparison",
                        code="executable_changed",
                        detail=record.executable_identity,
                    )
                )
            parent = self._capture_parent(record.parent_pid)
        return ProcessObservation(
            identity=expected,
            state=state,
            parent_identity=parent,
            managed_resource=managed_resource,
            evidence=evidence,
            observed_at=self.clock(),
        )

    def _identity(self, record: NativeProcessRecord) -> ProcessIdentity:
        assert record.executable_identity is not None
        return ProcessIdentity(
            platform=self.reader.platform,
            pid=record.pid,
            opaque_start_token=record.opaque_start_token,
            executable_identity=record.executable_identity,
            captured_at=self.clock(),
        )

    def _capture_parent(self, pid: int | None) -> ProcessIdentity | None:
        if pid is None or pid <= 0:
            return None
        result = self.reader.read(pid)
        if result.record is None or not result.record.executable_identity:
            return None
        return self._identity(result.record)
```

- [ ] **Step 4: Run observer and contract tests**

Run:

```bash
uv run pytest -q tests/test_process_identity_contract.py tests/test_process_identity_observation.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit observation semantics**

```bash
git add src/agent_session_harness/process_identity.py tests/test_process_identity_observation.py
git commit -m "feat: observe expected process identities"
```

### Task 3: Linux `/proc` adapter

**Files:**
- Modify: `src/agent_session_harness/process_identity.py`
- Create: `tests/test_process_identity_linux.py`

- [ ] **Step 1: Write failing Linux fixture tests**

Use fixture text with a command containing both spaces and parentheses:

```python
import os

from agent_session_harness.process_identity import (
    LinuxProcessReader,
    ProcessInspector,
    ProcessState,
)


def proc_stat(
    *,
    pid: int = 42,
    command: str = "python (worker)",
    state: str = "S",
    parent_pid: int = 7,
    start_ticks: int = 12345,
) -> str:
    suffix = [
        state,
        str(parent_pid),
        "0", "0", "0", "0", "0", "0", "0", "0",
        "0", "0", "0", "0", "0", "0", "0", "0",
        str(start_ticks),
    ]
    return f"{pid} ({command}) {' '.join(suffix)}\n"


def test_linux_parser_preserves_start_ticks_after_complex_command() -> None:
    reader = LinuxProcessReader(
        read_stat=lambda pid: proc_stat(),
        read_executable=lambda pid: "/usr/bin/python3",
    )
    record = reader.read(42).record
    assert record is not None
    assert record.opaque_start_token == "linux:12345"
    assert record.parent_pid == 7


def test_linux_zombie_does_not_require_executable_link() -> None:
    reader = LinuxProcessReader(
        read_stat=lambda pid: proc_stat(state="Z"),
        read_executable=lambda pid: (_ for _ in ()).throw(FileNotFoundError()),
    )
    record = reader.read(42).record
    assert record is not None
    assert record.state is ProcessState.ZOMBIE


def test_linux_permission_and_malformed_output_are_unknown() -> None:
    denied = LinuxProcessReader(
        read_stat=lambda pid: (_ for _ in ()).throw(PermissionError()),
        read_executable=lambda pid: "/usr/bin/python3",
    )
    malformed = LinuxProcessReader(
        read_stat=lambda pid: "42 malformed",
        read_executable=lambda pid: "/usr/bin/python3",
    )
    assert denied.read(42).state is ProcessState.UNKNOWN
    assert malformed.read(42).state is ProcessState.UNKNOWN


def test_linux_live_process_smoke() -> None:
    reader = LinuxProcessReader()
    identity = ProcessInspector(reader).capture(os.getpid())
    assert identity is not None
    assert identity.pid == os.getpid()
```

- [ ] **Step 2: Run Linux tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_process_identity_linux.py
```

Expected: collection fails because `LinuxProcessReader` is missing.

- [ ] **Step 3: Implement `LinuxProcessReader`**

```python
from pathlib import Path


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
        command_end = encoded.rfind(")")
        if command_end < 0:
            return NativeReadResult.unknown("stat_malformed")
        fields = encoded[command_end + 2 :].split()
        if len(fields) <= 19:
            return NativeReadResult.unknown("stat_malformed")
        state = ProcessState.ZOMBIE if fields[0] == "Z" else ProcessState.RUNNING
        try:
            parent_pid = int(fields[1])
            start_ticks = int(fields[19])
        except ValueError:
            return NativeReadResult.unknown("stat_malformed")
        executable = None
        if state is not ProcessState.ZOMBIE:
            try:
                executable = self._read_executable(pid)
            except (FileNotFoundError, ProcessLookupError):
                return NativeReadResult.missing()
            except OSError:
                return NativeReadResult.unknown("executable_unavailable")
            if not executable:
                return NativeReadResult.unknown("executable_unavailable")
        return NativeReadResult.present(
            NativeProcessRecord(
                pid=pid,
                parent_pid=parent_pid,
                state=state,
                opaque_start_token=f"linux:{start_ticks}",
                executable_identity=executable,
            )
        )

    @staticmethod
    def _read_proc_stat(pid: int) -> str:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")

    @staticmethod
    def _read_proc_executable(pid: int) -> str:
        return str(Path(f"/proc/{pid}/exe").resolve(strict=True))
```

- [ ] **Step 4: Run Linux, observer, and contract tests**

Run:

```bash
uv run pytest -q tests/test_process_identity_contract.py tests/test_process_identity_observation.py tests/test_process_identity_linux.py
```

Expected: all tests pass on Linux; the live smoke test is skipped on non-Linux
using `pytest.mark.skipif(sys.platform != "linux", ...)`.

- [ ] **Step 5: Commit the Linux adapter**

```bash
git add src/agent_session_harness/process_identity.py tests/test_process_identity_linux.py
git commit -m "feat: inspect Linux process identities"
```

### Task 4: macOS `libproc` adapter

**Files:**
- Modify: `src/agent_session_harness/process_identity.py`
- Create: `tests/test_process_identity_darwin.py`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Write failing Darwin fixture tests**

```python
import os
import sys

import pytest

from agent_session_harness.process_identity import (
    DarwinNativeInfo,
    DarwinProcessReader,
    ProcessInspector,
    ProcessState,
)


def info(
    *,
    pid: int = 42,
    parent_pid: int = 7,
    status: int = 2,
    start_seconds: int = 100,
    start_microseconds: int = 4,
) -> DarwinNativeInfo:
    return DarwinNativeInfo(
        pid=pid,
        parent_pid=parent_pid,
        status=status,
        start_seconds=start_seconds,
        start_microseconds=start_microseconds,
    )


def test_darwin_microseconds_distinguish_same_second_successors() -> None:
    first = DarwinProcessReader(
        read_info=lambda pid: info(start_microseconds=4),
        read_path=lambda pid: "/usr/bin/python3",
    ).read(42).record
    second = DarwinProcessReader(
        read_info=lambda pid: info(start_microseconds=5),
        read_path=lambda pid: "/usr/bin/python3",
    ).read(42).record
    assert first is not None and second is not None
    assert first.opaque_start_token != second.opaque_start_token


def test_darwin_zombie_does_not_require_process_path() -> None:
    reader = DarwinProcessReader(
        read_info=lambda pid: info(status=5),
        read_path=lambda pid: (_ for _ in ()).throw(ProcessLookupError()),
    )
    assert reader.read(42).state is ProcessState.ZOMBIE


def test_darwin_permission_and_malformed_info_are_unknown() -> None:
    denied = DarwinProcessReader(
        read_info=lambda pid: (_ for _ in ()).throw(PermissionError()),
        read_path=lambda pid: "/usr/bin/python3",
    )
    malformed = DarwinProcessReader(
        read_info=lambda pid: info(pid=99),
        read_path=lambda pid: "/usr/bin/python3",
    )
    path_denied = DarwinProcessReader(
        read_info=lambda pid: info(),
        read_path=lambda pid: (_ for _ in ()).throw(PermissionError()),
    )
    assert denied.read(42).state is ProcessState.UNKNOWN
    assert malformed.read(42).state is ProcessState.UNKNOWN
    assert path_denied.read(42).state is ProcessState.UNKNOWN


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin smoke test")
def test_darwin_live_process_smoke() -> None:
    identity = ProcessInspector(DarwinProcessReader()).capture(os.getpid())
    assert identity is not None
    assert identity.pid == os.getpid()
```

- [ ] **Step 2: Run Darwin tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_process_identity_darwin.py
```

Expected: collection fails because the Darwin reader types are missing.

- [ ] **Step 3: Implement Darwin native info and reader**

Move the existing `_DarwinProcessInfo` structure from `process.py` into
`process_identity.py`. Add:

```python
import ctypes
import errno


@dataclass(frozen=True)
class DarwinNativeInfo:
    pid: int
    parent_pid: int
    status: int
    start_seconds: int
    start_microseconds: int


class DarwinProcessReader:
    platform = ProcessPlatform.DARWIN
    _ZOMBIE_STATUS = 5

    def __init__(
        self,
        read_info: Callable[[int], DarwinNativeInfo] | None = None,
        read_path: Callable[[int], str] | None = None,
    ) -> None:
        self._read_info = read_info or self._read_libproc_info
        self._read_path = read_path or self._read_libproc_path

    def read(self, pid: int) -> NativeReadResult:
        try:
            info = self._read_info(pid)
        except (FileNotFoundError, ProcessLookupError):
            return NativeReadResult.missing()
        except (OSError, ValueError):
            return NativeReadResult.unknown("proc_info_unavailable")
        if (
            info.pid != pid
            or info.start_seconds <= 0
            or info.start_microseconds < 0
        ):
            return NativeReadResult.unknown("proc_info_malformed")
        state = (
            ProcessState.ZOMBIE
            if info.status == self._ZOMBIE_STATUS
            else ProcessState.RUNNING
        )
        executable = None
        if state is not ProcessState.ZOMBIE:
            try:
                executable = self._read_path(pid)
            except (FileNotFoundError, ProcessLookupError):
                return NativeReadResult.missing()
            except OSError:
                return NativeReadResult.unknown("executable_unavailable")
            if not executable:
                return NativeReadResult.unknown("executable_unavailable")
        return NativeReadResult.present(
            NativeProcessRecord(
                pid=pid,
                parent_pid=info.parent_pid,
                state=state,
                opaque_start_token=(
                    f"darwin:{info.start_seconds}:{info.start_microseconds}"
                ),
                executable_identity=executable,
            )
        )
```

Implement `_read_libproc_info` with the existing `proc_pidinfo` setup and
`_read_libproc_path` with `proc_pidpath`:

```python
class _DarwinProcessInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


def _raise_libproc_error(pid: int) -> None:
    error = ctypes.get_errno()
    if error == errno.ESRCH:
        raise ProcessLookupError(error, "process disappeared", pid)
    if error in {errno.EACCES, errno.EPERM}:
        raise PermissionError(error, "process inspection denied", pid)
    raise OSError(error or errno.EIO, "libproc inspection failed", pid)


def _load_libproc() -> ctypes.CDLL:
    return ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)


def _read_libproc_info(pid: int) -> DarwinNativeInfo:
    library = _load_libproc()
    proc_pidinfo = library.proc_pidinfo
    proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    proc_pidinfo.restype = ctypes.c_int
    native = _DarwinProcessInfo()
    size = ctypes.sizeof(native)
    ctypes.set_errno(0)
    result = proc_pidinfo(pid, 3, 0, ctypes.byref(native), size)
    if result != size:
        _raise_libproc_error(pid)
    return DarwinNativeInfo(
        pid=int(native.pbi_pid),
        parent_pid=int(native.pbi_ppid),
        status=int(native.pbi_status),
        start_seconds=int(native.pbi_start_tvsec),
        start_microseconds=int(native.pbi_start_tvusec),
    )


def _read_libproc_path(pid: int) -> str:
    library = _load_libproc()
    proc_pidpath = library.proc_pidpath
    proc_pidpath.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    proc_pidpath.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(4096)
    ctypes.set_errno(0)
    result = proc_pidpath(pid, buffer, len(buffer))
    if result <= 0:
        _raise_libproc_error(pid)
    return buffer.value.decode("utf-8")
```

Bind these two methods to `DarwinProcessReader` as
`_read_libproc_info = staticmethod(_read_libproc_info)` and
`_read_libproc_path = staticmethod(_read_libproc_path)`. `ESRCH` becomes
`missing`; permission and structurally invalid results become `unknown`.

- [ ] **Step 4: Add the Darwin smoke test to macOS CI**

Change the macOS command to:

```yaml
run: >-
  python -m pytest -q
  tests/test_process_identity_darwin.py
  tests/test_interactive_pty.py
  tests/test_supervisor.py::test_public_process_group_probe_uses_the_live_group_session
  tests/test_supervisor.py::test_public_process_group_probe_fails_closed_when_group_is_gone
```

- [ ] **Step 5: Run deterministic Darwin tests**

Run:

```bash
uv run pytest -q tests/test_process_identity_darwin.py
```

Expected on macOS: all fixture and live tests pass. Expected on Linux: fixture
tests pass and only the live smoke test skips.

- [ ] **Step 6: Commit the Darwin adapter**

```bash
git add src/agent_session_harness/process_identity.py tests/test_process_identity_darwin.py .github/workflows/ci.yml
git commit -m "feat: inspect Darwin process identities"
```

### Task 5: Existing driver delegation and public documentation

**Files:**
- Modify: `src/agent_session_harness/process.py`
- Modify: `src/agent_session_harness/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_process_identity_observation.py`
- Modify: `tests/test_supervisor.py`

- [ ] **Step 1: Write the failing delegation and stable-fingerprint tests**

```python
import hashlib
from datetime import UTC, datetime

from agent_session_harness import process as process_module
from agent_session_harness.process_identity import (
    ProcessIdentity,
    ProcessPlatform,
    legacy_process_fingerprint,
)


def fingerprint_identity() -> ProcessIdentity:
    return ProcessIdentity(
        platform=ProcessPlatform.LINUX,
        pid=42,
        opaque_start_token="linux:12345",
        executable_identity="/usr/bin/python3",
        captured_at=datetime(2026, 7, 30, tzinfo=UTC),
    )


def test_legacy_fingerprint_preserves_existing_registry_value() -> None:
    identity = fingerprint_identity()
    assert legacy_process_fingerprint(identity) == hashlib.sha256(
        b"42:linux:12345"
    ).hexdigest()


def test_posix_driver_uses_shared_identity_capture(monkeypatch) -> None:
    identity = fingerprint_identity()
    monkeypatch.setattr(
        process_module,
        "capture_process_identity",
        lambda pid: identity,
    )
    assert process_module.PosixProcessDriver._process_identity(42) == (
        legacy_process_fingerprint(identity)
    )
```

- [ ] **Step 2: Run delegation tests and verify RED**

Run:

```bash
uv run pytest -q tests/test_process_identity_observation.py -k fingerprint
uv run pytest -q tests/test_supervisor.py -k process_identity
```

Expected: tests fail because `legacy_process_fingerprint` and the driver
delegation are missing.

- [ ] **Step 3: Implement stable fingerprint and driver delegation**

Add to `process_identity.py`:

```python
import hashlib
import sys


def legacy_process_fingerprint(identity: ProcessIdentity) -> str:
    value = f"{identity.pid}:{identity.opaque_start_token}"
    return hashlib.sha256(value.encode()).hexdigest()


def native_process_reader() -> NativeProcessReader | None:
    if sys.platform.startswith("linux"):
        return LinuxProcessReader()
    if sys.platform == "darwin":
        return DarwinProcessReader()
    return None


def capture_process_identity(pid: int) -> ProcessIdentity | None:
    reader = native_process_reader()
    return ProcessInspector(reader).capture(pid) if reader is not None else None
```

Replace `PosixProcessDriver._process_identity` with:

```python
@staticmethod
def _process_identity(pid: int) -> str | None:
    identity = capture_process_identity(pid)
    return legacy_process_fingerprint(identity) if identity is not None else None
```

Delete `_kernel_process_birth`, `_darwin_process_birth`, and the duplicate
`_DarwinProcessInfo` structure from `process.py`. Remove imports used only by
those implementations.

- [ ] **Step 4: Export the public API**

Add explicit exports in `src/agent_session_harness/__init__.py`:

```python
from .process_identity import (
    ManagedResourceReference,
    ProcessEvidence,
    ProcessIdentity,
    ProcessInspector,
    ProcessObservation,
    ProcessPlatform,
    ProcessState,
    capture_process_identity,
    same_process,
)
```

- [ ] **Step 5: Document capture and observation**

Add a README section showing:

```python
from agent_session_harness import (
    ManagedResourceReference,
    ProcessInspector,
)
from agent_session_harness.process_identity import native_process_reader

reader = native_process_reader()
if reader is None:
    raise RuntimeError("native process identity is unavailable")
inspector = ProcessInspector(reader)
identity = inspector.capture(pid)
if identity is not None:
    observation = inspector.observe(
        identity,
        managed_resource=ManagedResourceReference(
            kind="runtime",
            resource_key="chain-1:0",
        ),
    )
```

State explicitly that only `running` is live, while `unknown` is neither live
evidence nor permission to reap.

- [ ] **Step 6: Run focused regression tests**

Run:

```bash
uv run pytest -q tests/test_process_identity_contract.py tests/test_process_identity_observation.py tests/test_process_identity_linux.py tests/test_process_identity_darwin.py tests/test_supervisor.py
```

Expected: all applicable tests pass, with only platform-opposite smoke tests
skipped.

- [ ] **Step 7: Commit integration and docs**

```bash
git add src/agent_session_harness/process.py src/agent_session_harness/process_identity.py src/agent_session_harness/__init__.py tests/test_process_identity_observation.py tests/test_supervisor.py README.md
git commit -m "refactor: share canonical process identity inspection"
```

### Task 6: Full verification and publication

**Files:**
- Verify all changed files

- [ ] **Step 1: Run the complete test suite**

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run lint and formatting checks**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: both commands exit zero.

- [ ] **Step 3: Build the package**

```bash
uv run python -m build
```

Expected: source distribution and wheel are created successfully.

- [ ] **Step 4: Confirm boundary hygiene**

```bash
rg -n "gaia|github|linear|beads|mutagen|gmake" src/agent_session_harness/process_identity.py tests/test_process_identity_*.py
```

Expected: no matches.

- [ ] **Step 5: Confirm the branch is clean and publish**

```bash
git status --short
git push -u origin bou-2709-process-identity-contract
gh pr create --base main --title "BOU-2709: publish process identity contract"
```

Expected: a ready-for-review upstream PR with Linux, macOS, build, and review
checks running.
