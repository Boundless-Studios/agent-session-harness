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
    first = (
        DarwinProcessReader(
            read_info=lambda pid: info(start_microseconds=4),
            read_path=lambda pid: "/usr/bin/python3",
        )
        .read(42)
        .record
    )
    second = (
        DarwinProcessReader(
            read_info=lambda pid: info(start_microseconds=5),
            read_path=lambda pid: "/usr/bin/python3",
        )
        .read(42)
        .record
    )

    assert first is not None and second is not None
    assert first.opaque_start_token != second.opaque_start_token


def test_darwin_zombie_does_not_require_process_path() -> None:
    reader = DarwinProcessReader(
        read_info=lambda pid: info(status=5),
        read_path=lambda pid: (_ for _ in ()).throw(ProcessLookupError()),
    )

    assert reader.read(42).state is ProcessState.ZOMBIE


def test_darwin_disappeared_process_is_missing() -> None:
    reader = DarwinProcessReader(
        read_info=lambda pid: (_ for _ in ()).throw(ProcessLookupError()),
        read_path=lambda pid: "/usr/bin/python3",
    )

    assert reader.read(42).state is ProcessState.MISSING


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
