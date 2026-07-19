from __future__ import annotations

from datetime import datetime, timezone
import importlib
from pathlib import Path

import pytest

from agent_session_harness.activity import ActivitySnapshot, Quiescence
from agent_session_harness.coordinator import ClaimHandle, FenceResult
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
    def __init__(self, supervisor_module, *, crash_once=False):
        self.supervisor_module = supervisor_module
        self.crash_once = crash_once
        self.receipts = {}
        self.calls = []

    def checkpoint(self, request):
        self.calls.append(request.idempotency_key)
        receipt = self.receipts.setdefault(
            request.idempotency_key,
            self.supervisor_module.VerifiedCheckpoint(
                verified=True,
                fingerprint="capsule-fingerprint",
                path=Path("/tmp/capsule.json"),
            ),
        )
        if self.crash_once:
            self.crash_once = False
            raise RuntimeError("checkpoint crash")
        return receipt


class FakeCoordinator:
    def __init__(self, *, crash_effect=None):
        self.epoch = 0
        self.active = None
        self.released = {}
        self.calls = []
        self.crash_effect = crash_effect

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
        if self.crash_effect == "claim-successor" and owner.endswith(":1"):
            self.crash_effect = None
            raise RuntimeError("claim crash")
        return handle

    def fence(self, handle, **_kwargs):
        self.calls.append(("fence", handle.owner_session_id))
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


def _supervisor(tmp_path, *, crash_effect=None, checkpoint_crash=False):
    process, supervisor = _modules()
    driver = FakeProcessDriver(process, crash_effect=crash_effect)
    coordinator = FakeCoordinator(crash_effect=crash_effect)
    checkpoints = FakeCheckpointManager(supervisor, crash_once=checkpoint_crash)
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
    }
    return supervisor.Supervisor(**kwargs), kwargs, driver, coordinator, checkpoints


@pytest.mark.parametrize(
    "runtime,args",
    [
        ("claude", ("--continue",)),
        ("claude", ("--resume", "abc")),
        ("codex", ("resume", "--last")),
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
        )
    with pytest.raises(ValueError, match="generation"):
        managed.acknowledge(
            generation=2,
            fingerprint=snapshot.checkpoint_fingerprint,
            conversation_id="native-conversation-1",
        )

    running = managed.acknowledge(
        generation=1,
        fingerprint=snapshot.checkpoint_fingerprint,
        conversation_id="native-conversation-1",
    )
    assert running.phase.value == "running"
    assert running.conversation_id == "native-conversation-1"
    assert managed.can_dispatch is True


def test_unknown_usage_or_activity_never_terminates_a_session(tmp_path) -> None:
    managed, _kwargs, driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    managed.usage_reader.confident = False

    assert managed.tick(_activity(Quiescence.IDLE)).phase.value == "running"
    managed.usage_reader.confident = True
    assert managed.tick(_activity(Quiescence.UNKNOWN)).phase.value == "draining"
    assert [call[0] for call in driver.calls] == ["start"]


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
