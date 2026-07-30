import os
import sys

import pytest

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
    fields = [
        state,
        str(parent_pid),
        *(["0"] * 17),
        str(start_ticks),
    ]
    return f"{pid} ({command}) {' '.join(fields)}\n"


def reader_for(
    encoded: str,
    executable: str = "/usr/bin/python3",
) -> LinuxProcessReader:
    return LinuxProcessReader(
        read_stat=lambda pid: encoded,
        read_executable=lambda pid: executable,
    )


def test_linux_parser_preserves_start_ticks_after_complex_command() -> None:
    record = reader_for(proc_stat()).read(42).record

    assert record is not None
    assert record.opaque_start_token == "linux:12345"
    assert record.parent_pid == 7
    assert record.state is ProcessState.RUNNING


def test_linux_start_ticks_distinguish_same_second_successors() -> None:
    first = reader_for(proc_stat(start_ticks=12345)).read(42).record
    second = reader_for(proc_stat(start_ticks=12346)).read(42).record

    assert first is not None and second is not None
    assert first.opaque_start_token != second.opaque_start_token


def test_linux_zombie_does_not_require_executable_link() -> None:
    reader = LinuxProcessReader(
        read_stat=lambda pid: proc_stat(state="Z"),
        read_executable=lambda pid: (_ for _ in ()).throw(FileNotFoundError()),
    )

    result = reader.read(42)

    assert result.state is ProcessState.ZOMBIE
    assert result.record is not None
    assert result.record.executable_identity is None


def test_linux_missing_stat_is_missing() -> None:
    reader = LinuxProcessReader(
        read_stat=lambda pid: (_ for _ in ()).throw(FileNotFoundError()),
        read_executable=lambda pid: "/usr/bin/python3",
    )

    assert reader.read(42).state is ProcessState.MISSING


@pytest.mark.parametrize(
    "read_stat",
    [
        lambda pid: (_ for _ in ()).throw(PermissionError()),
        lambda pid: "42 malformed",
        lambda pid: proc_stat(parent_pid=7).replace(" 7 ", " invalid ", 1),
    ],
)
def test_linux_unavailable_or_malformed_stat_is_unknown(read_stat) -> None:
    reader = LinuxProcessReader(
        read_stat=read_stat,
        read_executable=lambda pid: "/usr/bin/python3",
    )

    assert reader.read(42).state is ProcessState.UNKNOWN


def test_linux_executable_disappearance_is_missing() -> None:
    reader = LinuxProcessReader(
        read_stat=lambda pid: proc_stat(),
        read_executable=lambda pid: (_ for _ in ()).throw(FileNotFoundError()),
    )

    assert reader.read(42).state is ProcessState.MISSING


def test_linux_executable_permission_failure_is_unknown() -> None:
    reader = LinuxProcessReader(
        read_stat=lambda pid: proc_stat(),
        read_executable=lambda pid: (_ for _ in ()).throw(PermissionError()),
    )

    assert reader.read(42).state is ProcessState.UNKNOWN


@pytest.mark.skipif(sys.platform != "linux", reason="Linux smoke test")
def test_linux_live_process_smoke() -> None:
    identity = ProcessInspector(LinuxProcessReader()).capture(os.getpid())

    assert identity is not None
    assert identity.pid == os.getpid()
