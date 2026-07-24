from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from agent_coordinator import JsonlClaimStore, TaskCoordinator

from agent_session_harness.activity import Quiescence
from agent_session_harness.adapters.codex import CodexUsageReader
from agent_session_harness.adapters.command import (
    AdapterOperation,
    AdapterResponse,
)
from agent_session_harness.capsule import HandoffCapsule
from agent_session_harness.checkpoint import CheckpointManager
from agent_session_harness.coordinator import CoordinatorAdapter
from agent_session_harness.events import LifecycleEvent
from agent_session_harness.ledger import EventLedger
from agent_session_harness.models import Confidence, EventType
from agent_session_harness.outbox import MirrorOutbox
from agent_session_harness.process import PosixProcessDriver
from agent_session_harness.supervisor import (
    Supervisor,
    UsageObservation,
    VerifiedCheckpoint,
    acknowledgement_path,
)

FAKE_RUNTIME = Path(__file__).parent / "fixtures" / "fake_runtime.py"


class RolloutUsageReader:
    def __init__(self, root: Path):
        self.root = root

    def sample(self, process) -> UsageObservation:
        generation = int(process.registry_key.rsplit(":", 1)[1])
        usage = CodexUsageReader().read_file(self.root / f"rollout-{generation}.jsonl")
        return UsageObservation(
            conversation_id=usage.session_id,
            context_percent=usage.context_percent,
            confidence=usage.confidence,
            context_tokens=usage.context_tokens,
            window_tokens=usage.window_tokens,
            cumulative_tokens=usage.incremental_total_tokens,
        )


class CapsuleFileAdapter:
    name = "capsule-file"

    def __init__(self, path: Path, *, fail_acknowledgement: bool = False):
        self.path = path
        self.calls: list[AdapterOperation] = []
        self.fail_acknowledgement = fail_acknowledgement

    def execute(self, request) -> AdapterResponse:
        self.calls.append(request.operation)
        if (
            request.operation is AdapterOperation.ACKNOWLEDGE
            and self.fail_acknowledgement
        ):
            return AdapterResponse(
                ok=False,
                fingerprint=None,
                retryable=True,
                error="required adapter unavailable",
            )
        if request.operation is AdapterOperation.WRITE:
            self.path.write_bytes(request.capsule.canonical_bytes() + b"\n")
            fingerprint = request.capsule.fingerprint
        elif request.operation in {
            AdapterOperation.READ,
            AdapterOperation.ACKNOWLEDGE,
        }:
            stored = HandoffCapsule.model_validate_json(
                self.path.read_text(encoding="utf-8")
            )
            fingerprint = stored.fingerprint
        else:
            return AdapterResponse(
                ok=False,
                fingerprint=None,
                retryable=False,
                error="unsupported test adapter operation",
            )
        return AdapterResponse(
            ok=True,
            fingerprint=fingerprint,
            retryable=False,
            error=None,
        )


class CapsuleManager:
    def __init__(self, root: Path, *, fail_acknowledgement: bool = False):
        self.root = root
        self.adapter = CapsuleFileAdapter(
            root / "capsule.json",
            fail_acknowledgement=fail_acknowledgement,
        )
        self.manager = CheckpointManager(
            required_adapters=(self.adapter,),
            mirror_adapters=(),
            outbox=MirrorOutbox(root / "mirror-outbox.jsonl"),
        )
        self.calls: list[str] = []

    def checkpoint(self, request) -> VerifiedCheckpoint:
        self.calls.append(request.idempotency_key)
        capsule = HandoffCapsule(
            schema_version=1,
            chain_id=request.chain_id,
            predecessor_conversation_id=request.predecessor_conversation_id,
            target_generation=request.target_generation,
            task_ids={"linear": "BOU-2195", "bead": "bou-parent.3"},
            objective="Rotate without losing durable implementation state.",
            exact_next_action="Run the remaining harness integration tests.",
            completed_criteria=("usage accounting verified",),
            remaining_criteria=("integration tests green",),
            repository_path=self.root.resolve(),
            branch="bou-2195-agent-session-harness",
            head="deadbeef",
            dirty_paths=(),
            file_anchors=("src/agent_session_harness/supervisor.py",),
            symbol_anchors=("Supervisor.tick",),
            test_results={"focused": "passed"},
            decisions=("fresh successor only",),
            blockers=(),
            process_summaries={"pytest": "idle"},
            created_at=datetime.now(tz=timezone.utc),
        )
        result = self.manager.checkpoint(
            capsule,
            idempotency_key=request.idempotency_key,
        )
        return VerifiedCheckpoint(
            verified=result.verified,
            fingerprint=result.fingerprint,
            path=self.adapter.path,
        )

    def acknowledge(self, capsule, *, idempotency_key) -> bool:
        return self.manager.acknowledge(
            capsule,
            idempotency_key=idempotency_key,
        ).verified


def _wait_for(path: Path, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.02)


def _wait_for_quiescence(
    ledger: EventLedger,
    expected: Quiescence,
    *,
    timeout: float = 5.0,
):
    deadline = time.monotonic() + timeout
    while True:
        now = datetime.now(tz=timezone.utc)
        snapshot = ledger.materialize(now=now, stale_after_seconds=30)
        if snapshot.quiescence is expected:
            return snapshot
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"timed out waiting for lifecycle quiescence {expected.value}: "
                f"{snapshot.model_dump()}"
            )
        time.sleep(0.02)


def _set_context_tokens(path: Path, *, total_tokens: int) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    token_rows = [
        row
        for row in rows
        if row.get("type") == "event_msg"
        and row.get("payload", {}).get("type") == "token_count"
    ]
    if not token_rows:
        raise AssertionError(f"rollout has no token row: {path}")
    info = token_rows[-1]["payload"]["info"]
    info["total_token_usage"] = {"total_tokens": total_tokens}
    info["last_token_usage"] = {"total_tokens": total_tokens}
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_fake_runtime_rotates_once_to_a_fresh_acknowledged_successor(tmp_path) -> None:
    state_path = tmp_path / "supervisor.json"
    lifecycle_path = state_path.with_suffix(state_path.suffix + ".lifecycle")
    ledger = EventLedger(lifecycle_path)
    driver = PosixProcessDriver(tmp_path / "process-state")
    checkpoint_manager = CapsuleManager(tmp_path)
    supervisor = Supervisor(
        runtime="codex",
        chain_id="chain-1",
        cwd=tmp_path,
        task_type="linear",
        task_id="BOU-2195",
        task_fingerprint="task-fingerprint",
        executable=sys.executable,
        runtime_args=(str(FAKE_RUNTIME), "--root", str(tmp_path)),
        state_path=state_path,
        process_driver=driver,
        usage_reader=RolloutUsageReader(tmp_path),
        checkpoint_manager=checkpoint_manager,
        coordinator=CoordinatorAdapter(
            TaskCoordinator(
                JsonlClaimStore(tmp_path / "claims.jsonl"),
                pid_is_live=lambda _pid: True,
            )
        ),
        stop_timeout_seconds=2,
    )

    try:
        supervisor.start()
        rollout_path = tmp_path / "rollout-0.jsonl"
        _wait_for(rollout_path)
        busy = _wait_for_quiescence(ledger, Quiescence.BUSY)

        warning = supervisor.tick(busy)
        assert warning.phase.value == "warning"
        assert warning.context_percent == 65.0

        _set_context_tokens(rollout_path, total_tokens=140)
        draining = supervisor.tick(busy)
        assert draining.phase.value == "draining"
        assert draining.context_percent == 70.0

        (tmp_path / "finish-activity-0").write_text("finish\n", encoding="utf-8")
        idle = _wait_for_quiescence(ledger, Quiescence.IDLE)
        awaiting = supervisor.tick(idle)
        assert awaiting.phase.value == "awaiting_ack"
        assert awaiting.generation == 1
        assert awaiting.claim is not None and awaiting.claim.lease_epoch == 2

        _wait_for(acknowledgement_path(state_path))
        running = supervisor.tick(idle)
        assert running.phase.value == "running"
        assert running.conversation_id == "native-conversation-1"
        assert running.context_confidence is Confidence.UNKNOWN

        _wait_for(tmp_path / "continuations.jsonl")
        history = [
            json.loads(line)
            for line in (tmp_path / "history.jsonl").read_text().splitlines()
        ]
        starts = [entry for entry in history if entry["event"] == "started"]
        stops = [entry for entry in history if entry["event"] == "stopped"]
        continuation = [
            json.loads(line)
            for line in (tmp_path / "continuations.jsonl").read_text().splitlines()
        ]
        claims = [
            json.loads(line)
            for line in (tmp_path / "claims.jsonl").read_text().splitlines()
            if '"event": "claimed"' in line
        ]
        lifecycle = [
            LifecycleEvent.model_validate(json.loads(line))
            for line in lifecycle_path.read_text(encoding="utf-8").splitlines()
        ]
        checkpoint_events = [
            event
            for event in lifecycle
            if event.event_type is EventType.HANDOFF_CHECKPOINTED
        ]
        acknowledgement_events = [
            event
            for event in lifecycle
            if event.event_type is EventType.HANDOFF_ACKNOWLEDGED
        ]
        stop_responses = [
            json.loads(line)
            for line in (tmp_path / "stop-responses.jsonl").read_text().splitlines()
        ]

        assert [entry["conversation_id"] for entry in starts] == [
            "native-conversation-0",
            "native-conversation-1",
        ]
        assert len({entry["chain_id"] for entry in starts}) == 1
        assert [entry["generation"] for entry in starts] == [0, 1]
        assert [entry["generation"] for entry in stops] == [0]
        assert starts[0]["pid"] != starts[1]["pid"]
        stopped_at = datetime.fromisoformat(stops[0]["timestamp"])
        successor_started_at = datetime.fromisoformat(starts[1]["timestamp"])
        assert stopped_at <= successor_started_at
        assert len(claims) == 2
        assert [entry["claim"]["lease_epoch"] for entry in claims] == [1, 2]
        assert len(continuation) == 1
        assert continuation[0]["exact_next_action"] == (
            "Run the remaining harness integration tests."
        )

        assert checkpoint_manager.calls == ["chain-1:1"]
        assert checkpoint_manager.adapter.calls == [
            AdapterOperation.WRITE,
            AdapterOperation.READ,
            AdapterOperation.ACKNOWLEDGE,
        ]
        assert len(checkpoint_events) == 1
        assert len(acknowledgement_events) == 1
        assert [response["exit_code"] for response in stop_responses] == [2, 0]
        assert "Context rotation is draining" in stop_responses[0]["stderr"]
        assert (
            sum(event.event_type is EventType.HANDOFF_REQUESTED for event in lifecycle)
            == 1
        )
        assert checkpoint_events[0].timestamp <= stopped_at
        assert successor_started_at <= acknowledgement_events[0].timestamp
        assert (
            sum(event.event_type is EventType.TOOL_STARTED for event in lifecycle) == 1
        )
        assert (
            sum(event.event_type is EventType.TOOL_FINISHED for event in lifecycle) == 1
        )
        assert sum(event.event_type is EventType.TURN_IDLE for event in lifecycle) == 1
        assert not acknowledgement_path(state_path).exists()
    finally:
        process = supervisor.current_process
        if process is not None and driver.is_alive(process):
            driver.graceful_stop(process, 2)


def test_successor_cannot_run_when_required_ack_adapter_is_unavailable(
    tmp_path,
) -> None:
    state_path = tmp_path / "supervisor.json"
    ledger = EventLedger(state_path.with_suffix(state_path.suffix + ".lifecycle"))
    driver = PosixProcessDriver(tmp_path / "process-state")
    checkpoint_manager = CapsuleManager(tmp_path, fail_acknowledgement=True)
    supervisor = Supervisor(
        runtime="codex",
        chain_id="chain-required-ack",
        cwd=tmp_path,
        task_type="linear",
        task_id="BOU-2195",
        task_fingerprint="task-fingerprint",
        executable=sys.executable,
        runtime_args=(str(FAKE_RUNTIME), "--root", str(tmp_path)),
        state_path=state_path,
        process_driver=driver,
        usage_reader=RolloutUsageReader(tmp_path),
        checkpoint_manager=checkpoint_manager,
        coordinator=CoordinatorAdapter(
            TaskCoordinator(
                JsonlClaimStore(tmp_path / "claims.jsonl"),
                pid_is_live=lambda _pid: True,
            )
        ),
        stop_timeout_seconds=2,
    )

    try:
        supervisor.start()
        rollout_path = tmp_path / "rollout-0.jsonl"
        _wait_for(rollout_path)
        busy = _wait_for_quiescence(ledger, Quiescence.BUSY)
        supervisor.tick(busy)
        _set_context_tokens(rollout_path, total_tokens=140)
        assert supervisor.tick(busy).phase.value == "draining"
        (tmp_path / "finish-activity-0").write_text("finish\n", encoding="utf-8")
        idle = _wait_for_quiescence(ledger, Quiescence.IDLE)
        assert supervisor.tick(idle).phase.value == "awaiting_ack"
        _wait_for(acknowledgement_path(state_path))

        assert not (tmp_path / "continuations.jsonl").exists()
        retried = supervisor.tick(idle)
        assert retried.phase.value == "awaiting_ack"
        assert retried.successor_attempt == 1
        _wait_for(acknowledgement_path(state_path))
        with pytest.raises(RuntimeError, match="retry budget"):
            supervisor.tick(idle)
        assert supervisor.snapshot.phase.value == "blocked"
        assert not (tmp_path / "continuations.jsonl").exists()
    finally:
        supervisor.shutdown()
