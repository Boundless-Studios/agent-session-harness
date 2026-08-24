from __future__ import annotations

import math
import multiprocessing
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from agent_session_harness.daemon_lifecycle import (
    DaemonDefinition,
    DaemonLifecyclePhase,
    DaemonLifecycleRecord,
    DaemonLifecycleStore,
)
from agent_session_harness.daemon_supervisor import (
    DaemonIdentityUnknownError,
    DaemonLaunchError,
    DaemonSupervisor,
    _OwnedChildState,
)
from agent_session_harness.process_identity import (
    ProcessIdentity,
    ProcessPlatform,
    ProcessState,
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
        assert service.status().phase is DaemonLifecyclePhase.RUNNING

        stopped = service.stop()

        assert stopped.phase is DaemonLifecyclePhase.STOPPED
        assert stopped.generation == started.generation
        assert stopped.process_identity is None
    finally:
        if started.process_identity is not None:
            try:
                os.kill(started.process_identity.pid, 9)
            except ProcessLookupError:
                pass


def test_status_without_state_is_durably_stopped(tmp_path) -> None:
    service = supervisor(tmp_path)

    status = service.status()

    assert status.phase is DaemonLifecyclePhase.STOPPED
    assert status.generation == 0
    assert service.store.read() == status


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
        try:
            os.killpg(pids[0], 9)
        except ProcessLookupError:
            pass


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


def test_state_for_another_daemon_is_rejected_without_signaling(tmp_path) -> None:
    foreign = DaemonLifecycleRecord(
        daemon_key="foreign",
        phase=DaemonLifecyclePhase.FAILED,
        generation=1,
        changed_at=datetime.now(UTC),
    )
    DaemonLifecycleStore(tmp_path / "state.json").publish(foreign)

    with pytest.raises(RuntimeError, match="different daemon"):
        supervisor(tmp_path).start()

    assert DaemonLifecycleStore(tmp_path / "state.json").read() == foreign


def test_process_creation_failure_persists_failed_state(tmp_path) -> None:
    missing = DaemonDefinition(
        daemon_key="missing",
        argv=(str(tmp_path / "does-not-exist"),),
        cwd=tmp_path,
    )
    service = DaemonSupervisor(
        missing,
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "lock",
    )

    with pytest.raises(Exception, match="process creation failed"):
        service.start()

    state = DaemonLifecycleStore(tmp_path / "state.json").read()
    assert state is not None
    assert state.phase is DaemonLifecyclePhase.FAILED


def test_detached_controller_refuses_pid_only_stop(tmp_path) -> None:
    owner = supervisor(tmp_path)
    owner.start()
    detached = supervisor(tmp_path)
    try:
        with pytest.raises(DaemonIdentityUnknownError, match="another controller"):
            detached.stop()
    finally:
        owner.stop()


def test_takeover_controller_can_stop_identity_verified_process(tmp_path) -> None:
    owner = supervisor(tmp_path)
    running = owner.start()
    takeover = DaemonSupervisor(
        definition(tmp_path),
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "lifecycle.lock",
        lock_timeout=1,
        startup_probe_seconds=0.05,
        stop_timeout=1,
        allow_process_takeover=True,
    )

    assert running.process_identity is not None
    stopped = takeover.stop()

    assert stopped.phase is DaemonLifecyclePhase.STOPPED


def test_takeover_controller_stops_live_group_with_zombie_leader(
    tmp_path, monkeypatch
) -> None:
    takeover = DaemonSupervisor(
        definition(tmp_path),
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "lifecycle.lock",
        allow_process_takeover=True,
    )
    identity = ProcessIdentity(
        platform=(
            ProcessPlatform.DARWIN
            if sys.platform == "darwin"
            else ProcessPlatform.LINUX
        ),
        pid=12345,
        opaque_start_token="token",
        executable_identity=sys.executable,
        captured_at=datetime.now(UTC),
    )
    takeover.store.publish(
        DaemonLifecycleRecord(
            daemon_key="sleeper",
            phase=DaemonLifecyclePhase.FAILED,
            generation=1,
            changed_at=datetime.now(UTC),
            process_identity=identity,
        )
    )
    monkeypatch.setattr(
        "agent_session_harness.daemon_supervisor.observe_process_identity",
        lambda _identity: Mock(state=ProcessState.ZOMBIE),
    )
    monkeypatch.setattr(
        takeover,
        "_process_group_has_live_members",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(takeover, "_wait_group_absent", lambda *_args: True)
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    stopped = takeover.stop()

    assert stopped.phase is DaemonLifecyclePhase.STOPPED
    assert signals == [(identity.pid, signal.SIGTERM)]


def test_start_persists_recovered_running_phase(tmp_path) -> None:
    owner = supervisor(tmp_path)
    running = owner.start()
    stopping = running.model_copy(update={"phase": DaemonLifecyclePhase.STOPPING})
    DaemonLifecycleStore(tmp_path / "state.json").publish(stopping)
    try:
        recovered = supervisor(tmp_path).start()

        assert recovered.phase is DaemonLifecyclePhase.RUNNING
        assert DaemonLifecycleStore(tmp_path / "state.json").read() == recovered
    finally:
        owner.stop()


@pytest.mark.parametrize("timeout", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field", ["lock_timeout", "startup_probe_seconds", "stop_timeout", "kill_timeout"]
)
def test_supervisor_rejects_non_finite_timeouts(tmp_path, field, timeout) -> None:
    with pytest.raises(ValueError, match="finite"):
        DaemonSupervisor(
            definition(tmp_path),
            state_path=tmp_path / "state.json",
            lock_path=tmp_path / "lock",
            **{field: timeout},
        )


@pytest.mark.parametrize("field", ["stop_timeout", "kill_timeout"])
def test_supervisor_rejects_zero_stop_deadlines(tmp_path, field) -> None:
    with pytest.raises(ValueError, match="positive"):
        DaemonSupervisor(
            definition(tmp_path),
            state_path=tmp_path / "state.json",
            lock_path=tmp_path / "lock",
            **{field: 0},
        )


def test_startup_failure_cleans_group_before_reaping_leader(tmp_path) -> None:
    forking = DaemonDefinition(
        daemon_key="forking",
        argv=(
            sys.executable,
            "-c",
            "import os,time; p=os.fork(); time.sleep(60) if p == 0 else None",
        ),
        cwd=tmp_path,
    )
    service = DaemonSupervisor(
        forking,
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "lock",
        startup_probe_seconds=0.1,
        stop_timeout=0.05,
        kill_timeout=1,
    )

    with pytest.raises(DaemonLaunchError, match="startup identity probe"):
        service.start()

    assert service._child is None


def test_running_publish_failure_cleans_owned_group(tmp_path, monkeypatch) -> None:
    service = supervisor(tmp_path)
    original_publish = service.store.publish

    def fail_running(state):
        if state.phase is DaemonLifecyclePhase.RUNNING:
            raise OSError("disk full")
        original_publish(state)

    monkeypatch.setattr(service.store, "publish", fail_running)

    with pytest.raises(OSError, match="disk full"):
        service.start()

    assert service._child is None
    state = DaemonLifecycleStore(tmp_path / "state.json").read()
    assert state is not None
    assert state.phase is DaemonLifecyclePhase.FAILED
    assert state.process_identity is None


def test_publish_and_cleanup_failure_preserves_captured_identity(
    tmp_path, monkeypatch
) -> None:
    service = supervisor(tmp_path)
    original_publish = service.store.publish

    def fail_running(state):
        if state.phase is DaemonLifecyclePhase.RUNNING:
            raise OSError("disk full")
        original_publish(state)

    monkeypatch.setattr(service.store, "publish", fail_running)
    monkeypatch.setattr(
        service,
        "_terminate_owned_child",
        lambda _child: (_ for _ in ()).throw(TimeoutError("cleanup timed out")),
    )

    with pytest.raises(TimeoutError, match="cleanup timed out"):
        service.start()

    state = DaemonLifecycleStore(tmp_path / "state.json").read()
    assert state is not None
    assert state.phase is DaemonLifecyclePhase.FAILED
    assert state.process_identity is not None


def test_process_group_probe_timeout_fails_closed(monkeypatch) -> None:
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(("/bin/ps",), 0.01)

    monkeypatch.setattr(subprocess, "run", timeout)

    assert DaemonSupervisor._process_group_has_live_members(42, timeout=0.01)


def test_missing_owned_pid_refuses_group_probe_or_signal(tmp_path, monkeypatch) -> None:
    service = supervisor(tmp_path)
    running = service.start()
    assert running.process_identity is not None
    child_pid = running.process_identity.pid
    real_killpg = os.killpg
    group_probes = []
    signals = []
    monkeypatch.setattr(
        "agent_session_harness.daemon_supervisor.observe_process_identity",
        lambda _identity: type("Observation", (), {"state": ProcessState.MISSING})(),
    )
    monkeypatch.setattr(
        service,
        "_process_group_has_live_members",
        lambda *_args, **_kwargs: group_probes.append(True) or True,
    )
    monkeypatch.setattr(os, "killpg", lambda *_args: signals.append(True))

    try:
        with pytest.raises(DaemonIdentityUnknownError, match="missing"):
            service.stop()

        assert group_probes == []
        assert signals == []
    finally:
        real_killpg(child_pid, 9)


def test_start_rejects_captured_identity_after_child_is_missing(
    tmp_path, monkeypatch
) -> None:
    service = supervisor(tmp_path)
    captured = ProcessIdentity(
        platform=(
            ProcessPlatform.DARWIN
            if sys.platform == "darwin"
            else ProcessPlatform.LINUX
        ),
        pid=4242,
        opaque_start_token="replacement",
        executable_identity=sys.executable,
        captured_at=datetime.now(UTC),
    )
    signals = []
    monkeypatch.setattr(
        "agent_session_harness.daemon_supervisor.capture_process_identity",
        lambda _pid: captured,
    )
    monkeypatch.setattr(
        service,
        "_owned_child_state",
        lambda _child: _OwnedChildState.MISSING,
    )
    monkeypatch.setattr(os, "killpg", lambda *_args: signals.append(True))

    with pytest.raises(DaemonLaunchError, match="startup identity probe"):
        service.start()

    assert signals == []
    state = DaemonLifecycleStore(tmp_path / "state.json").read()
    assert state is not None
    assert state.phase is DaemonLifecyclePhase.FAILED
    assert state.process_identity is None
