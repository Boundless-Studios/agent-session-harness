from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from agent_coordinator import ClaimConflictError

from agent_session_harness.activity import ActivitySnapshot, Quiescence
from agent_session_harness.capsule import HandoffCapsule
from agent_session_harness.coordinator import (
    ClaimHandle,
    CoordinatorAdapter,
    FenceResult,
)
from agent_session_harness.models import Confidence


NOW = datetime(2026, 7, 19, 6, 0, tzinfo=timezone.utc)


def _modules():
    try:
        process = importlib.import_module("agent_session_harness.process")
        supervisor = importlib.import_module("agent_session_harness.supervisor")
    except ModuleNotFoundError:
        pytest.fail("rotation supervisor is not implemented")
    return process, supervisor


def _activity(quiescence: Quiescence) -> ActivitySnapshot:
    active = frozenset({"active"}) if quiescence is Quiescence.BUSY else frozenset()
    return ActivitySnapshot(
        quiescence=quiescence,
        active_turn_ids=frozenset(),
        active_tool_ids=active,
        active_subagent_ids=frozenset(),
        active_critical_section_ids=frozenset(),
        processed_event_count=1,
        last_event_at=NOW,
        integrity_warnings=(),
    )


class FakeUsageReader:
    def __init__(self, supervisor_module, *, percent=75.0, confident=True):
        self.supervisor_module = supervisor_module
        self.percent = percent
        self.confident = confident

    def sample(self, _process):
        return self.supervisor_module.UsageObservation(
            conversation_id="native-conversation-0",
            context_percent=self.percent,
            confidence=(Confidence.CONFIDENT if self.confident else Confidence.UNKNOWN),
        )


class FakeCheckpointManager:
    def __init__(self, supervisor_module, root, *, crash_once=False):
        self.supervisor_module = supervisor_module
        self.root = root
        self.crash_once = crash_once
        self.receipts = {}
        self.calls = []

    def checkpoint(self, request):
        self.calls.append(request.idempotency_key)
        capsule = HandoffCapsule(
            schema_version=1,
            chain_id=request.chain_id,
            predecessor_conversation_id=request.predecessor_conversation_id,
            target_generation=request.target_generation,
            task_ids={"test": "task-1"},
            objective="Test crash-safe rotation.",
            exact_next_action="Continue the focused supervisor test.",
            completed_criteria=("predecessor drained",),
            remaining_criteria=("successor acknowledged",),
            repository_path=self.root,
            branch="test-branch",
            head="deadbeef",
            dirty_paths=(),
            file_anchors=("tests/test_supervisor.py",),
            symbol_anchors=("Supervisor.tick",),
            test_results={"focused": "running"},
            decisions=("fresh launch",),
            blockers=(),
            process_summaries={"test": "idle"},
            created_at=NOW,
        )
        capsule_path = self.root / "capsule.json"
        capsule_path.write_bytes(capsule.canonical_bytes() + b"\n")
        receipt = self.receipts.setdefault(
            request.idempotency_key,
            self.supervisor_module.VerifiedCheckpoint(
                verified=True,
                fingerprint=capsule.fingerprint,
                path=capsule_path,
            ),
        )
        if self.crash_once:
            self.crash_once = False
            raise RuntimeError("checkpoint crash")
        return receipt


class FakeCoordinator:
    def __init__(
        self, *, crash_effect=None, stale_on_heartbeat=False, stale_on_fence=False
    ):
        self.epoch = 0
        self.active = None
        self.released = {}
        self.calls = []
        self.crash_effect = crash_effect
        self.stale_on_heartbeat = stale_on_heartbeat
        self.stale_on_fence = stale_on_fence

    def claim(self, **kwargs):
        owner = kwargs["owner_session_id"]
        self.calls.append(("claim", owner))
        if self.active is not None and self.active.owner_session_id == owner:
            handle = self.active
        else:
            self.epoch += 1
            handle = ClaimHandle(
                claim_id=f"claim-{self.epoch}",
                lease_epoch=self.epoch,
                task_type=kwargs["task_type"],
                task_id=kwargs["task_id"],
                task_fingerprint=kwargs["fingerprint"],
                owner_session_id=owner,
            )
            self.active = handle
        if self.crash_effect == "claim-initial" and owner.endswith(":0"):
            self.crash_effect = None
            raise RuntimeError("claim crash")
        if self.crash_effect == "claim-successor" and owner.endswith(":1"):
            self.crash_effect = None
            raise RuntimeError("claim crash")
        return handle

    def fence(self, handle, **_kwargs):
        self.calls.append(("fence", handle.owner_session_id))
        if self.stale_on_fence:
            from agent_session_harness.coordinator import StaleOwnerError

            raise StaleOwnerError("stale owner fence")
        result = self.released.setdefault(
            handle.claim_id,
            FenceResult(
                claim_id=handle.claim_id,
                lease_epoch=handle.lease_epoch,
                release_reason="context-rotation",
            ),
        )
        self.active = None
        if self.crash_effect == "fence":
            self.crash_effect = None
            raise RuntimeError("fence crash")
        return result

    def heartbeat(self, handle, **_kwargs):
        self.calls.append(("heartbeat", handle.owner_session_id))
        if self.stale_on_heartbeat:
            from agent_session_harness.coordinator import StaleOwnerError

            raise StaleOwnerError("stale owner lease")
        return handle


class FakeProcessDriver:
    def __init__(self, process_module, *, crash_effect=None):
        self.process_module = process_module
        self.processes = {}
        self.active_pids = set()
        self.max_active = 0
        self.calls = []
        self.crash_effect = crash_effect

    def start_fresh(self, request):
        self.calls.append(("start", request.generation, request.runtime_args))
        process = self.processes.get(request.generation)
        if process is None:
            pid = 1000 + request.generation
            process = self.process_module.ManagedProcess(
                pid=pid,
                process_group_id=pid,
                registry_key=f"{request.chain_id}:{request.generation}",
            )
            self.processes[request.generation] = process
            self.active_pids.add(pid)
            self.max_active = max(self.max_active, len(self.active_pids))
        if self.crash_effect == "start-initial" and request.generation == 0:
            self.crash_effect = None
            raise RuntimeError("start crash")
        if self.crash_effect == "start-successor" and request.generation == 1:
            self.crash_effect = None
            raise RuntimeError("start crash")
        return process

    def graceful_stop(self, process, _timeout_seconds):
        self.calls.append(("stop", process.pid))
        self.active_pids.discard(process.pid)
        if self.crash_effect == "stop":
            self.crash_effect = None
            raise RuntimeError("stop crash")
        return 0

    def is_alive(self, process):
        return process.pid in self.active_pids


def _supervisor(
    tmp_path,
    *,
    crash_effect=None,
    checkpoint_crash=False,
    stale_on_heartbeat=False,
    stale_on_fence=False,
    heartbeat_interval_seconds=20.0,
):
    process, supervisor = _modules()
    driver = FakeProcessDriver(process, crash_effect=crash_effect)
    coordinator = FakeCoordinator(
        crash_effect=crash_effect,
        stale_on_heartbeat=stale_on_heartbeat,
        stale_on_fence=stale_on_fence,
    )
    checkpoints = FakeCheckpointManager(
        supervisor, tmp_path, crash_once=checkpoint_crash
    )
    kwargs = {
        "runtime": "codex",
        "chain_id": "chain-1",
        "cwd": tmp_path,
        "task_type": "linear",
        "task_id": "BOU-2195",
        "task_fingerprint": "task-fingerprint",
        "executable": "codex",
        "runtime_args": ("--full-auto",),
        "state_path": tmp_path / "supervisor.json",
        "process_driver": driver,
        "usage_reader": FakeUsageReader(supervisor),
        "checkpoint_manager": checkpoints,
        "coordinator": coordinator,
        "heartbeat_interval_seconds": heartbeat_interval_seconds,
    }
    return supervisor.Supervisor(**kwargs), kwargs, driver, coordinator, checkpoints


@pytest.mark.parametrize(
    "runtime,args",
    [
        ("claude", ("--continue",)),
        ("claude", ("--resume=abc",)),
        ("claude", ("--resume", "abc")),
        ("codex", ("resume", "--last")),
        ("codex", ("--last=abc",)),
    ],
)
def test_fresh_launch_rejects_native_resume_arguments(tmp_path, runtime, args) -> None:
    process, _supervisor_module = _modules()
    with pytest.raises(ValueError, match="fresh"):
        process.LaunchRequest(
            runtime=runtime,
            chain_id="chain-1",
            generation=1,
            cwd=tmp_path,
            executable=runtime,
            runtime_args=args,
            environment={},
        )


def test_controlled_launch_environment_preserves_cli_identity_and_rejects_noise(
    monkeypatch,
) -> None:
    process, _supervisor_module = _modules()
    monkeypatch.setenv("HOME", "/tmp/fake-home")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/fake-config")
    monkeypatch.setenv("UNRELATED_PRIVATE_VALUE", "do-not-copy")

    environment = process.PosixProcessDriver._environment({})

    assert environment["HOME"] == "/tmp/fake-home"
    assert environment["XDG_CONFIG_HOME"] == "/tmp/fake-config"
    assert "UNRELATED_PRIVATE_VALUE" not in environment
    with pytest.raises(ValueError, match="allowlisted"):
        process.PosixProcessDriver._environment({"UNRELATED_PRIVATE_VALUE": "x"})


def test_restored_process_identity_fails_closed_on_pid_reuse(
    tmp_path, monkeypatch
) -> None:
    process, _supervisor_module = _modules()
    driver = process.PosixProcessDriver(tmp_path)
    restored = process.ManagedProcess(
        pid=4242,
        process_group_id=4242,
        registry_key="chain-1:0",
        identity="original-birth",
        command_digest="command-digest",
    )
    monkeypatch.setattr(driver, "_pid_exists", lambda _pid: True)
    monkeypatch.setattr(driver, "_process_identity", lambda _pid: "different-birth")

    assert driver.is_alive(restored) is False

    unverified = process.ManagedProcess(
        pid=4242,
        process_group_id=4242,
        registry_key="chain-1:0",
        identity=None,
        command_digest="command-digest",
    )
    with pytest.raises(RuntimeError, match="identity"):
        driver.is_alive(unverified)


def test_process_birth_identity_distinguishes_same_second_processes() -> None:
    process, _supervisor_module = _modules()
    children = [
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(4)
    ]
    try:
        identities = [
            process.PosixProcessDriver._process_identity(child.pid)
            for child in children
        ]
        assert all(identity is not None for identity in identities)
        assert len(set(identities)) == len(identities)
    finally:
        for child in children:
            child.terminate()
        for child in children:
            child.wait(timeout=2)


def test_launch_guardian_rejects_a_superseded_intent(tmp_path, monkeypatch) -> None:
    process, _supervisor_module = _modules()
    registry_path = tmp_path / "process.json"
    intent_path = tmp_path / "process.intent"
    process.PosixProcessDriver._write_intent(
        intent_path,
        key="chain-1:0",
        command_digest="new-command",
        launch_nonce="new-nonce",
    )
    monkeypatch.setattr(
        process.PosixProcessDriver,
        "_process_identity",
        lambda _pid: "birth-identity",
    )

    with pytest.raises(RuntimeError, match="no longer current"):
        process.register_guarded_process(
            registry_path=registry_path,
            intent_path=intent_path,
            registry_key="chain-1:0",
            command_digest="old-command",
            launch_nonce="old-nonce",
        )

    assert not registry_path.exists()


def test_launch_guardian_exec_failure_is_not_reported_as_active(tmp_path) -> None:
    process, _supervisor_module = _modules()
    driver = process.PosixProcessDriver(tmp_path)
    request = process.LaunchRequest(
        runtime="codex",
        chain_id="chain-exec-failure",
        generation=0,
        cwd=tmp_path,
        executable=str(tmp_path / "missing-runtime"),
    )

    with pytest.raises(RuntimeError, match="exited before becoming ready"):
        driver.start_fresh(request)


def test_recent_unspawned_launch_intent_recovers_in_the_same_call(tmp_path) -> None:
    process, _supervisor_module = _modules()
    driver = process.PosixProcessDriver(tmp_path, startup_timeout_seconds=0.1)
    request = process.LaunchRequest(
        runtime="codex",
        chain_id="chain-recent-intent",
        generation=0,
        cwd=tmp_path,
        executable=sys.executable,
        runtime_args=("-c", "import time; time.sleep(5)"),
    )
    key = "chain-recent-intent:0"
    registry_path = driver._registry_path(key)
    intent_path = registry_path.with_suffix(".intent")
    argv = [request.executable, *request.runtime_args]
    command_digest = (
        __import__("hashlib")
        .sha256(
            json.dumps(
                {"argv": argv, "cwd": str(request.cwd)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        .hexdigest()
    )
    process.PosixProcessDriver._write_intent(
        intent_path,
        key=key,
        command_digest=command_digest,
        launch_nonce="recent-unspawned",
    )

    managed = driver.start_fresh(request)
    try:
        assert driver.is_alive(managed)
    finally:
        driver.graceful_stop(managed, 1)


def test_guardian_stops_runtime_after_supervisor_heartbeat_expires(tmp_path) -> None:
    state_path = tmp_path / "supervisor.json"
    claims_path = tmp_path / "claims.jsonl"
    process_state = tmp_path / "process-state"
    started_path = tmp_path / "runtime-started"
    stopped_path = tmp_path / "runtime-stopped"
    managed_path = tmp_path / "managed.json"
    runtime = tmp_path / "runtime.py"
    runtime.write_text(
        "import pathlib, signal, sys, time\n"
        "started, stopped = map(pathlib.Path, sys.argv[1:3])\n"
        "started.write_text('started')\n"
        "def stop(*_args):\n"
        "    stopped.write_text('stopped')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "while True: time.sleep(0.02)\n",
        encoding="utf-8",
    )
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import json, pathlib, sys, time\n"
        "from datetime import datetime, timezone\n"
        "from agent_session_harness.coordinator import CoordinatorAdapter\n"
        "from agent_session_harness.process import LaunchRequest, PosixProcessDriver\n"
        "from agent_session_harness.supervisor import SupervisorSnapshot\n"
        "root, state_name, claims_name, process_name, runtime_name, started, stopped, managed_name = sys.argv[1:]\n"
        "root_path = pathlib.Path(root)\n"
        "state = pathlib.Path(state_name)\n"
        "driver = PosixProcessDriver(process_name, startup_timeout_seconds=1)\n"
        "request = LaunchRequest(runtime='codex', chain_id='watchdog-chain', generation=0, "
        "cwd=root_path, executable=sys.executable, runtime_args=(runtime_name, started, stopped), "
        "environment={'AGENT_SESSION_HARNESS_STATE_PATH': str(state), "
        "'AGENT_SESSION_HARNESS_CHAIN_ID': 'watchdog-chain', "
        "'AGENT_SESSION_HARNESS_GENERATION': '0', "
        "'AGENT_SESSION_HARNESS_WATCHDOG_TIMEOUT_SECONDS': '0.4'})\n"
        "claim = CoordinatorAdapter.from_path(claims_name).claim(task_type='linear', "
        "task_id='BOU-2195', fingerprint='fingerprint', "
        "owner_session_id='watchdog-chain:0', owner_pid=__import__('os').getpid(), "
        "runtime='codex', worktree_path=str(root_path), lease_seconds=2)\n"
        "state.write_text(SupervisorSnapshot(runtime='codex', chain_id='watchdog-chain', "
        "generation=0, phase='launching', owner_session_id='watchdog-chain:0', claim=claim, "
        "last_heartbeat_at=datetime.now(timezone.utc)).model_dump_json())\n"
        "managed = driver.start_fresh(request)\n"
        "state.write_text(SupervisorSnapshot(runtime='codex', chain_id='watchdog-chain', "
        "generation=0, phase='running', owner_session_id='watchdog-chain:0', claim=claim, "
        "process_pid=managed.pid, process_group_id=managed.process_group_id, "
        "process_registry_key=managed.registry_key, process_identity=managed.identity, "
        "process_command_digest=managed.command_digest, process_launch_nonce=managed.launch_nonce, "
        "last_heartbeat_at=datetime.now(timezone.utc)).model_dump_json())\n"
        "pathlib.Path(managed_name).write_text(json.dumps({'pid': managed.pid, 'pgid': managed.process_group_id}))\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    launcher_process = subprocess.Popen(
        [
            sys.executable,
            str(launcher),
            str(tmp_path),
            str(state_path),
            str(claims_path),
            str(process_state),
            str(runtime),
            str(started_path),
            str(stopped_path),
            str(managed_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    managed: dict[str, int] | None = None
    try:
        deadline = time.monotonic() + 5
        while not (started_path.exists() and managed_path.exists()):
            if launcher_process.poll() is not None:
                error = launcher_process.stderr.read().decode("utf-8", errors="replace")
                raise AssertionError(f"launcher exited early: {error}")
            if time.monotonic() >= deadline:
                raise AssertionError("watchdog runtime did not start")
            time.sleep(0.02)
        managed = json.loads(managed_path.read_text(encoding="utf-8"))
        launcher_process.kill()
        launcher_process.wait(timeout=2)

        competitor = CoordinatorAdapter.from_path(claims_path)
        with pytest.raises(ClaimConflictError):
            competitor.claim(
                task_type="linear",
                task_id="BOU-2195",
                fingerprint="fingerprint",
                owner_session_id="competitor:0",
                owner_pid=os.getpid(),
                runtime="codex",
                worktree_path=str(tmp_path),
                lease_seconds=2,
            )

        deadline = time.monotonic() + 3
        stopped_before_claim = False
        while True:
            try:
                successor = competitor.claim(
                    task_type="linear",
                    task_id="BOU-2195",
                    fingerprint="fingerprint",
                    owner_session_id="competitor:0",
                    owner_pid=os.getpid(),
                    runtime="codex",
                    worktree_path=str(tmp_path),
                    lease_seconds=2,
                )
                stopped_before_claim = stopped_path.exists()
                break
            except ClaimConflictError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        assert successor.lease_epoch == 2
        assert stopped_before_claim is True
        assert stopped_path.read_text(encoding="utf-8") == "stopped"
    finally:
        if launcher_process.poll() is None:
            launcher_process.kill()
            launcher_process.wait(timeout=2)
        if managed is not None:
            try:
                os.killpg(managed["pgid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_supervisor_watchdog_preempts_claim_lease_with_shutdown_margin(
    tmp_path,
) -> None:
    managed, kwargs, _driver, _coordinator, _checkpoints = _supervisor(tmp_path)

    request = managed._launch_request(generation=0)

    assert (
        request.environment["AGENT_SESSION_HARNESS_WATCHDOG_TIMEOUT_SECONDS"] == "57.0"
    )
    kwargs["heartbeat_interval_seconds"] = 57.0
    with pytest.raises(ValueError, match="watchdog shutdown margin"):
        type(managed)(**kwargs)


def test_supervisor_warns_drains_waits_then_rotates_without_overlap(tmp_path) -> None:
    managed, _kwargs, driver, coordinator, checkpoints = _supervisor(tmp_path)
    managed.start()
    managed.usage_reader.percent = 65.0
    assert managed.tick(_activity(Quiescence.BUSY)).phase.value == "warning"
    managed.usage_reader.percent = 70.0
    assert managed.tick(_activity(Quiescence.BUSY)).phase.value == "draining"
    managed.usage_reader.percent = 10.0
    assert managed.tick(_activity(Quiescence.BUSY)).phase.value == "draining"

    awaiting = managed.tick(_activity(Quiescence.IDLE))

    assert awaiting.phase.value == "awaiting_ack"
    assert awaiting.generation == 1
    assert driver.max_active == 1
    assert [call[0] for call in coordinator.calls] == ["claim", "fence", "claim"]
    assert [call[0] for call in driver.calls] == ["start", "stop", "start"]
    assert len(set(checkpoints.calls)) == 1
    assert managed.can_dispatch is False
    assert "resume" not in driver.calls[-1][2]
    assert "--continue" not in driver.calls[-1][2]


def test_successor_acknowledgement_requires_expected_generation_and_fingerprint(
    tmp_path,
) -> None:
    managed, _kwargs, _driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    snapshot = managed.tick(_activity(Quiescence.IDLE))

    with pytest.raises(ValueError, match="fingerprint"):
        managed.acknowledge(
            generation=1,
            fingerprint="wrong",
            conversation_id="native-conversation-1",
            owner_pid=1001,
        )
    with pytest.raises(ValueError, match="generation"):
        managed.acknowledge(
            generation=2,
            fingerprint=snapshot.checkpoint_fingerprint,
            conversation_id="native-conversation-1",
            owner_pid=1001,
        )

    running = managed.acknowledge(
        generation=1,
        fingerprint=snapshot.checkpoint_fingerprint,
        conversation_id="native-conversation-1",
        owner_pid=1001,
    )
    assert running.phase.value == "running"
    assert running.conversation_id == "native-conversation-1"
    assert managed.can_dispatch is True


def test_successor_acknowledgement_is_bound_to_child_pid_and_heartbeats_claim(
    tmp_path,
) -> None:
    managed, _kwargs, _driver, coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    snapshot = managed.tick(_activity(Quiescence.IDLE))

    with pytest.raises(ValueError, match="process"):
        managed.acknowledge(
            generation=1,
            fingerprint=snapshot.checkpoint_fingerprint,
            conversation_id="native-conversation-1",
            owner_pid=9999,
        )
    running = managed.acknowledge(
        generation=1,
        fingerprint=snapshot.checkpoint_fingerprint,
        conversation_id="native-conversation-1",
        owner_pid=1001,
    )

    assert running.phase.value == "running"
    assert coordinator.calls[-1] == ("heartbeat", "chain-1:1")


def test_successor_acknowledgement_rejects_tampered_capsule(tmp_path) -> None:
    managed, _kwargs, _driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    snapshot = managed.tick(_activity(Quiescence.IDLE))
    assert snapshot.checkpoint_path is not None
    snapshot.checkpoint_path.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="capsule"):
        managed.acknowledge(
            generation=1,
            fingerprint=snapshot.checkpoint_fingerprint,
            conversation_id="native-conversation-1",
            owner_pid=1001,
        )

    assert managed.snapshot.phase.value == "awaiting_ack"


def test_competing_supervisors_refresh_under_one_transition_lock(tmp_path) -> None:
    first, kwargs, driver, coordinator, _checkpoints = _supervisor(tmp_path)
    second = type(first)(**kwargs)

    first.start()
    second.start()

    assert [call[0] for call in coordinator.calls] == ["claim"]
    assert [call[0] for call in driver.calls] == ["start"]


def test_periodic_heartbeat_and_stale_owner_failure_are_fail_closed(tmp_path) -> None:
    managed, _kwargs, _driver, coordinator, _checkpoints = _supervisor(
        tmp_path, heartbeat_interval_seconds=0.0
    )
    managed.start()
    managed.tick(_activity(Quiescence.BUSY))
    assert coordinator.calls[1] == ("heartbeat", "chain-1:0")

    stale, _kwargs, driver, _coordinator, _checkpoints = _supervisor(
        tmp_path / "stale",
        stale_on_heartbeat=True,
        heartbeat_interval_seconds=0.0,
    )
    stale.start()
    with pytest.raises(RuntimeError, match="stale"):
        stale.tick(_activity(Quiescence.BUSY))

    assert stale.snapshot.phase.value == "blocked"
    assert driver.active_pids == set()


def test_dead_managed_process_is_detected_before_more_dispatch(tmp_path) -> None:
    managed, _kwargs, driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    driver.active_pids.clear()

    with pytest.raises(RuntimeError, match="managed process is not live"):
        managed.tick(_activity(Quiescence.IDLE))

    assert managed.snapshot.phase.value == "blocked"
    assert managed.can_dispatch is False


def test_stale_owner_during_fence_stops_predecessor_fail_closed(tmp_path) -> None:
    managed, _kwargs, driver, _coordinator, _checkpoints = _supervisor(
        tmp_path, stale_on_fence=True
    )
    managed.start()

    with pytest.raises(RuntimeError, match="stale"):
        managed.tick(_activity(Quiescence.IDLE))

    assert managed.snapshot.phase.value == "blocked"
    assert driver.active_pids == set()


def test_unknown_usage_or_activity_never_terminates_a_session(tmp_path) -> None:
    managed, _kwargs, driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    managed.usage_reader.confident = False

    assert managed.tick(_activity(Quiescence.IDLE)).phase.value == "running"
    managed.usage_reader.confident = True
    assert managed.tick(_activity(Quiescence.UNKNOWN)).phase.value == "draining"
    assert [call[0] for call in driver.calls] == ["start"]


def test_terminal_shutdown_is_persisted_and_clears_owned_process(tmp_path) -> None:
    managed, kwargs, driver, coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()

    snapshot = managed.shutdown()

    assert snapshot.phase.value == "blocked"
    assert snapshot.claim is None
    assert snapshot.process_pid is None
    assert managed.current_process is None
    assert driver.active_pids == set()
    assert coordinator.active is None

    recovered = type(managed)(**kwargs)
    assert recovered.snapshot.phase.value == "blocked"
    assert recovered.current_process is None
    assert recovered.start().phase.value == "blocked"


@pytest.mark.parametrize("failed_effect", ["claim", "launch"])
def test_terminal_shutdown_recovers_resources_after_initial_effect_failure(
    tmp_path, monkeypatch, failed_effect
) -> None:
    managed, _kwargs, driver, coordinator, _checkpoints = _supervisor(tmp_path)
    original_effect = managed._effect

    def fail_after_effect(effect, status, *, generation):
        original_effect(effect, status, generation=generation)
        if effect == failed_effect and status == "completed":
            raise RuntimeError(f"{failed_effect} completion failed")

    monkeypatch.setattr(managed, "_effect", fail_after_effect)
    with pytest.raises(RuntimeError, match="completion failed"):
        managed.start()
    monkeypatch.setattr(managed, "_effect", original_effect)

    snapshot = managed.shutdown()

    assert snapshot.phase.value == "blocked"
    assert snapshot.claim is None
    assert driver.active_pids == set()
    assert coordinator.active is None


def test_terminal_shutdown_preserves_in_memory_launch_when_persist_fails(
    tmp_path, monkeypatch
) -> None:
    managed, _kwargs, driver, coordinator, _checkpoints = _supervisor(tmp_path)
    original_persist = managed._persist
    failed = False

    def fail_process_persist():
        nonlocal failed
        if managed.snapshot.process_pid is not None and not failed:
            failed = True
            raise RuntimeError("process persist failed")
        original_persist()

    monkeypatch.setattr(managed, "_persist", fail_process_persist)
    with pytest.raises(RuntimeError, match="process persist failed"):
        managed.start()
    monkeypatch.setattr(managed, "_persist", original_persist)

    snapshot = managed.shutdown()

    assert snapshot.claim is None
    assert snapshot.process_pid is None
    assert driver.active_pids == set()
    assert coordinator.active is None


def test_terminal_shutdown_retains_claim_until_fencing_can_be_retried(tmp_path) -> None:
    managed, _kwargs, driver, coordinator, _checkpoints = _supervisor(
        tmp_path, crash_effect="fence"
    )
    managed.start()

    with pytest.raises(RuntimeError, match="fencing failed"):
        managed.shutdown()

    assert managed.snapshot.phase.value == "blocked"
    assert managed.snapshot.claim is not None
    assert managed.snapshot.process_pid is None
    assert driver.active_pids == set()

    recovered = managed.shutdown()
    assert recovered.claim is None
    assert coordinator.active is None


def test_terminal_shutdown_stops_successor_after_launch_completion_failure(
    tmp_path, monkeypatch
) -> None:
    managed, _kwargs, driver, coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    original_effect = managed._effect

    def fail_successor_launch_effect(effect, status, *, generation):
        original_effect(effect, status, generation=generation)
        if effect == "launch" and status == "completed" and generation == 1:
            raise RuntimeError("successor launch completion failed")

    monkeypatch.setattr(managed, "_effect", fail_successor_launch_effect)
    with pytest.raises(RuntimeError, match="successor launch completion failed"):
        managed.tick(_activity(Quiescence.IDLE))
    monkeypatch.setattr(managed, "_effect", original_effect)

    snapshot = managed.shutdown()

    assert snapshot.claim is None
    assert snapshot.process_pid is None
    assert driver.active_pids == set()
    assert coordinator.active is None


@pytest.mark.parametrize("failed_effect", ["stop", "fence"])
def test_terminal_shutdown_cleans_resources_when_effect_journaling_fails(
    tmp_path, monkeypatch, failed_effect
) -> None:
    managed, _kwargs, driver, coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    original_effect = managed._effect

    def fail_started_effect(effect, status, *, generation):
        if effect == failed_effect and status == "started":
            raise OSError(f"{failed_effect} journal unavailable")
        original_effect(effect, status, generation=generation)

    monkeypatch.setattr(managed, "_effect", fail_started_effect)

    with pytest.raises(RuntimeError, match="effect journaling failed"):
        managed.shutdown()

    assert managed.snapshot.claim is None
    assert managed.snapshot.process_pid is None
    assert driver.active_pids == set()
    assert coordinator.active is None


def test_terminal_shutdown_uses_known_resources_when_state_is_corrupt(tmp_path) -> None:
    managed, _kwargs, driver, coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    managed.state_path.write_text("{corrupt", encoding="utf-8")

    with pytest.raises(RuntimeError, match="state refresh failed after cleanup"):
        managed.shutdown()

    assert managed.snapshot.claim is None
    assert managed.snapshot.process_pid is None
    assert driver.active_pids == set()
    assert coordinator.active is None


@pytest.mark.parametrize("crash_effect", ["claim-initial", "start-initial"])
def test_initial_launch_recovers_without_duplicate_process(
    tmp_path, crash_effect
) -> None:
    managed, kwargs, driver, _coordinator, _checkpoints = _supervisor(
        tmp_path, crash_effect=crash_effect
    )
    with pytest.raises(RuntimeError, match="crash"):
        managed.start()

    recovered = type(managed)(**kwargs)
    snapshot = recovered.start()

    assert snapshot.phase.value == "running"
    assert driver.max_active == 1
    assert len(driver.processes) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", "BOU-9999"),
        ("task_fingerprint", "different-task"),
        ("executable", "different-codex"),
        ("runtime_args", ("--different",)),
    ],
)
def test_recovery_rejects_a_different_immutable_run_spec(
    tmp_path, field, value
) -> None:
    managed, kwargs, _driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    changed = {**kwargs, field: value}

    with pytest.raises(ValueError, match="run specification"):
        type(managed)(**changed)

    managed.shutdown()


def test_acknowledgement_completion_crash_recovers_and_clears_stale_record(
    tmp_path, monkeypatch
) -> None:
    managed, kwargs, _driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    _process_module, supervisor_module = _modules()
    managed.start()
    snapshot = managed.tick(_activity(Quiescence.IDLE))
    supervisor_module.write_acknowledgement(
        state_path=managed.state_path,
        generation=1,
        fingerprint=snapshot.checkpoint_fingerprint,
        conversation_id="native-conversation-1",
        owner_pid=1001,
    )
    original_effect = managed._effect

    def crash_after_acknowledgement(effect, status, *, generation):
        original_effect(effect, status, generation=generation)
        if effect == "acknowledge" and status == "completed":
            raise RuntimeError("acknowledgement crash")

    monkeypatch.setattr(managed, "_effect", crash_after_acknowledgement)
    with pytest.raises(RuntimeError, match="acknowledgement crash"):
        managed.tick(_activity(Quiescence.IDLE))

    recovered = type(managed)(**kwargs)
    running = recovered.start()
    lifecycle = recovered.lifecycle_path.read_text(encoding="utf-8")

    assert running.phase.value == "running"
    assert not supervisor_module.acknowledgement_path(managed.state_path).exists()
    assert lifecycle.count('"event_type":"handoff.acknowledged"') == 1


def test_stale_ack_clear_never_unlinks_a_newer_acknowledgement(tmp_path) -> None:
    _process_module, supervisor = _modules()
    state_path = tmp_path / "supervisor.json"
    target = supervisor.acknowledgement_path(state_path)
    old = supervisor.AcknowledgementRecord(
        generation=1,
        fingerprint="1" * 64,
        conversation_id="conversation-1",
        owner_pid=1001,
    )
    newer = supervisor.AcknowledgementRecord(
        generation=2,
        fingerprint="2" * 64,
        conversation_id="conversation-2",
        owner_pid=1002,
    )
    target.write_text(newer.model_dump_json() + "\n", encoding="utf-8")

    supervisor.clear_acknowledgement(state_path, expected=old)

    persisted = supervisor.AcknowledgementRecord.model_validate_json(
        target.read_text(encoding="utf-8")
    )
    assert persisted == newer
    assert json.loads(target.read_text(encoding="utf-8"))["generation"] == 2


@pytest.mark.parametrize(
    ("crash_effect", "checkpoint_crash"),
    [
        (None, True),
        ("fence", False),
        ("stop", False),
        ("claim-successor", False),
        ("start-successor", False),
    ],
)
def test_rotation_recovers_idempotently_after_each_effect_crash(
    tmp_path, crash_effect, checkpoint_crash
) -> None:
    managed, kwargs, driver, coordinator, checkpoints = _supervisor(
        tmp_path,
        crash_effect=crash_effect,
        checkpoint_crash=checkpoint_crash,
    )
    managed.start()
    with pytest.raises(RuntimeError, match="crash"):
        managed.tick(_activity(Quiescence.IDLE))

    recovered = type(managed)(**kwargs)
    snapshot = recovered.tick(_activity(Quiescence.IDLE))

    assert snapshot.phase.value == "awaiting_ack"
    assert driver.max_active == 1
    assert len(checkpoints.receipts) == 1
    assert len(coordinator.released) == 1
    assert len(driver.processes) == 2
