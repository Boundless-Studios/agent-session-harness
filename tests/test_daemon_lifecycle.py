from __future__ import annotations

import math
import multiprocessing
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_session_harness.daemon_lifecycle import (
    DaemonDefinition,
    DaemonLifecyclePhase,
    DaemonLifecycleRecord,
    DaemonLifecycleStore,
    LockBusyError,
    OwnerDiagnosticLock,
)
from agent_session_harness.process_identity import ProcessIdentity, ProcessPlatform

NOW = datetime(2026, 8, 13, tzinfo=UTC)


def _hold_lock(path: str, ready: multiprocessing.synchronize.Event) -> None:
    with OwnerDiagnosticLock(Path(path), purpose="first starter").acquire(timeout=1):
        ready.set()
        time.sleep(2)


def record(generation: int = 1) -> DaemonLifecycleRecord:
    return DaemonLifecycleRecord(
        daemon_key="pr-dashboard",
        phase=DaemonLifecyclePhase.STARTING,
        generation=generation,
        changed_at=NOW,
        detail="spawn pending",
    )


def test_lifecycle_record_round_trips_through_private_atomic_store(tmp_path) -> None:
    path = tmp_path / "daemon" / "state.json"
    store = DaemonLifecycleStore(path)

    store.publish(record())

    assert DaemonLifecycleStore(path).read() == record()
    assert path.stat().st_mode & 0o777 == 0o600


def test_lifecycle_store_rejects_corrupt_and_unknown_schema(tmp_path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"schema_version":', encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid JSON"):
        DaemonLifecycleStore(path).read()

    path.write_text('{"schema_version":2}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsupported schema"):
        DaemonLifecycleStore(path).read()


def test_lifecycle_store_reads_additive_future_fields(tmp_path) -> None:
    path = tmp_path / "state.json"
    payload = record().model_dump_json()[:-1] + ',"unknown_authority":true}'
    path.write_text(payload, encoding="utf-8")

    restored = DaemonLifecycleStore(path).read()

    assert restored is not None
    assert restored.model_extra == {"unknown_authority": True}


def test_running_state_requires_process_identity() -> None:
    with pytest.raises(ValueError, match="running state requires process identity"):
        record().model_copy(
            update={"phase": DaemonLifecyclePhase.RUNNING}
        ).model_validate(
            record().model_dump()
            | {"phase": DaemonLifecyclePhase.RUNNING, "process_identity": None}
        )


def test_daemon_definition_requires_direct_nonempty_argv(tmp_path) -> None:
    definition = DaemonDefinition(
        daemon_key="pr-dashboard",
        argv=("python3", "-m", "agentic_pr_dash.server"),
        cwd=tmp_path,
    )

    assert definition.argv[0] == "python3"
    with pytest.raises(ValueError):
        DaemonDefinition(daemon_key="bad", argv=(), cwd=tmp_path)


def test_running_state_round_trips_process_identity(tmp_path) -> None:
    identity = ProcessIdentity(
        platform=ProcessPlatform.DARWIN,
        pid=42,
        opaque_start_token="darwin:123",
        executable_identity="/usr/bin/python3",
        captured_at=NOW,
    )
    running = record().model_copy(
        update={
            "phase": DaemonLifecyclePhase.RUNNING,
            "process_identity": identity,
        }
    )
    path = tmp_path / "state.json"

    DaemonLifecycleStore(path).publish(running)

    assert DaemonLifecycleStore(path).read() == running


def test_competing_lock_times_out_with_owner_diagnostics(tmp_path) -> None:
    path = tmp_path / "daemon.lock"
    ready = multiprocessing.Event()
    process = multiprocessing.Process(target=_hold_lock, args=(str(path), ready))
    process.start()
    try:
        assert ready.wait(timeout=2)

        started = time.monotonic()
        with pytest.raises(LockBusyError) as captured:
            with OwnerDiagnosticLock(path, purpose="second starter").acquire(
                timeout=0.1
            ):
                pytest.fail("competing owner acquired the lock")

        assert time.monotonic() - started < 1
        assert captured.value.owner is not None
        assert captured.value.owner.pid == process.pid
        assert captured.value.owner.purpose == "first starter"
        assert captured.value.owner.acquired_at.tzinfo is not None
    finally:
        process.terminate()
        process.join(timeout=2)


def test_released_lock_can_be_reacquired_with_new_owner(tmp_path) -> None:
    path = tmp_path / "daemon.lock"

    with OwnerDiagnosticLock(path, purpose="first").acquire(timeout=0.1):
        pass
    with OwnerDiagnosticLock(path, purpose="second").acquire(timeout=0.1) as owner:
        assert owner.pid > 0
        assert owner.purpose == "second"


@pytest.mark.parametrize("timeout", [math.nan, math.inf, -math.inf])
def test_lock_rejects_non_finite_timeout(tmp_path, timeout) -> None:
    with pytest.raises(ValueError, match="finite"):
        OwnerDiagnosticLock(tmp_path / "lock", purpose="test").acquire(timeout=timeout)


def test_publish_revalidates_untrusted_model_copy(tmp_path) -> None:
    invalid = record().model_copy(update={"phase": DaemonLifecyclePhase.RUNNING})

    with pytest.raises(ValueError, match="running state requires process identity"):
        DaemonLifecycleStore(tmp_path / "state.json").publish(invalid)

    assert not (tmp_path / "state.json").exists()


def test_daemon_definition_requires_absolute_cwd(tmp_path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        DaemonDefinition(daemon_key="bad", argv=("python3",), cwd="relative")


@pytest.mark.parametrize("argv", [("bad\x00arg",), ("python3", "bad\x00arg")])
def test_daemon_definition_rejects_nul_argv(tmp_path, argv) -> None:
    with pytest.raises(ValueError, match="NUL"):
        DaemonDefinition(daemon_key="bad", argv=argv, cwd=tmp_path)
