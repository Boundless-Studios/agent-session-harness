from __future__ import annotations

import os
import multiprocessing
import sys
from datetime import UTC, datetime
from pathlib import Path

from agent_session_harness.daemon_lifecycle import (
    DaemonDefinition,
    DaemonLifecyclePhase,
    DaemonLifecycleRecord,
    DaemonLifecycleStore,
)
from agent_session_harness.daemon_supervisor import DaemonSupervisor
from agent_session_harness.process_identity import (
    ProcessIdentity,
    ProcessPlatform,
)


def definition(tmp_path) -> DaemonDefinition:
    return DaemonDefinition(
        daemon_key="sleeper",
        argv=(sys.executable, "-c", "import time; time.sleep(60)"),
        cwd=tmp_path,
    )


def supervisor(tmp_path) -> DaemonSupervisor:
    return DaemonSupervisor(
        definition(tmp_path),
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "lifecycle.lock",
        lock_timeout=1,
        startup_probe_seconds=0.05,
        stop_timeout=1,
    )


def _start_worker(root: str, queue: multiprocessing.Queue) -> None:
    queue.put(supervisor(Path(root)).start().process_identity.pid)


def test_start_is_idempotent_and_stop_is_durable(tmp_path) -> None:
    service = supervisor(tmp_path)
    started = service.start()
    try:
        assert started.phase is DaemonLifecyclePhase.RUNNING
        assert started.process_identity is not None
        assert service.start().process_identity == started.process_identity

        stopped = DaemonSupervisor(
            definition(tmp_path),
            state_path=tmp_path / "state.json",
            lock_path=tmp_path / "lifecycle.lock",
            lock_timeout=1,
            startup_probe_seconds=0.05,
            stop_timeout=1,
        ).stop()

        assert stopped.phase is DaemonLifecyclePhase.STOPPED
        assert stopped.generation == started.generation
        assert stopped.process_identity is None
    finally:
        if started.process_identity is not None:
            try:
                os.kill(started.process_identity.pid, 9)
            except ProcessLookupError:
                pass


def test_concurrent_starters_publish_exactly_one_process(tmp_path) -> None:
    queue = multiprocessing.Queue()
    workers = [
        multiprocessing.Process(target=_start_worker, args=(str(tmp_path), queue))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=3)

    pids = [queue.get(timeout=1) for _ in workers]
    try:
        assert len(set(pids)) == 1
        assert all(worker.exitcode == 0 for worker in workers)
    finally:
        supervisor(tmp_path).stop()


def test_restart_advances_generation_and_replaces_process(tmp_path) -> None:
    service = supervisor(tmp_path)
    first = service.start()
    try:
        second = service.restart()
        assert second.phase is DaemonLifecyclePhase.RUNNING
        assert second.generation == first.generation + 1
        assert second.process_identity is not None
        assert first.process_identity is not None
        assert second.process_identity != first.process_identity
    finally:
        service.stop()


def test_stop_never_signals_a_reused_pid(tmp_path) -> None:
    platform = (
        ProcessPlatform.DARWIN if sys.platform == "darwin" else ProcessPlatform.LINUX
    )
    state = DaemonLifecycleRecord(
        daemon_key="sleeper",
        phase=DaemonLifecyclePhase.RUNNING,
        generation=1,
        changed_at=datetime.now(UTC),
        process_identity=ProcessIdentity(
            platform=platform,
            pid=os.getpid(),
            opaque_start_token="not-this-process",
            executable_identity=sys.executable,
            captured_at=datetime.now(UTC),
        ),
    )
    DaemonLifecycleStore(tmp_path / "state.json").publish(state)

    stopped = supervisor(tmp_path).stop()

    assert stopped.phase is DaemonLifecyclePhase.STOPPED
    assert stopped.detail == "tracked process lifetime is absent"
    assert os.getpid() == state.process_identity.pid


def test_start_recovers_stale_dead_state(tmp_path) -> None:
    stale = DaemonLifecycleRecord(
        daemon_key="sleeper",
        phase=DaemonLifecyclePhase.FAILED,
        generation=4,
        changed_at=datetime.now(UTC),
        detail="previous launch died",
    )
    DaemonLifecycleStore(tmp_path / "state.json").publish(stale)
    service = supervisor(tmp_path)

    started = service.start()
    try:
        assert started.phase is DaemonLifecyclePhase.RUNNING
        assert started.generation == 5
    finally:
        service.stop()


def test_stop_escalates_after_bounded_grace_period(tmp_path) -> None:
    stubborn = DaemonDefinition(
        daemon_key="stubborn",
        argv=(
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
        ),
        cwd=tmp_path,
    )
    service = DaemonSupervisor(
        stubborn,
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "lifecycle.lock",
        startup_probe_seconds=0.05,
        stop_timeout=0.05,
        kill_timeout=1,
    )
    service.start()

    stopped = service.stop()

    assert stopped.phase is DaemonLifecyclePhase.STOPPED
