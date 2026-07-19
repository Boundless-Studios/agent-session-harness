from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from agent_coordinator import JsonlClaimStore, TaskCoordinator

from agent_session_harness.activity import Quiescence
from agent_session_harness.adapters.codex import CodexUsageReader
from agent_session_harness.capsule import HandoffCapsule
from agent_session_harness.coordinator import CoordinatorAdapter
from agent_session_harness.events import LifecycleEvent
from agent_session_harness.ledger import EventLedger
from agent_session_harness.models import Confidence
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
        )


class CapsuleManager:
    def __init__(self, root: Path):
        self.root = root
        self.calls: dict[str, VerifiedCheckpoint] = {}

    def checkpoint(self, request) -> VerifiedCheckpoint:
        existing = self.calls.get(request.idempotency_key)
        if existing is not None:
            return existing
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
        path = self.root / "capsule.json"
        path.write_bytes(capsule.canonical_bytes() + b"\n")
        verified = HandoffCapsule.model_validate_json(path.read_text())
        receipt = VerifiedCheckpoint(
            verified=verified.fingerprint == capsule.fingerprint,
            fingerprint=capsule.fingerprint,
            path=path,
        )
        self.calls[request.idempotency_key] = receipt
        return receipt


def _wait_for(path: Path, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.02)


def _tool_event(
    *, root: Path, event_id: str, event_type: str, timestamp: datetime
) -> LifecycleEvent:
    return LifecycleEvent(
        schema_version=1,
        event_id=event_id,
        runtime="codex",
        chain_id="chain-1",
        conversation_id="native-conversation-0",
        generation=0,
        event_type=event_type,
        timestamp=timestamp,
        cwd=root,
        owner_pid=1234,
        activity_id="tool-1",
        name="fake-tool",
    )


def test_fake_runtime_rotates_once_to_a_fresh_acknowledged_successor(tmp_path) -> None:
    state_path = tmp_path / "supervisor.json"
    lifecycle_path = state_path.with_suffix(state_path.suffix + ".lifecycle")
    ledger = EventLedger(lifecycle_path)
    driver = PosixProcessDriver(tmp_path / "process-state")
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
        checkpoint_manager=CapsuleManager(tmp_path),
        coordinator=CoordinatorAdapter(
            TaskCoordinator(
                JsonlClaimStore(tmp_path / "claims.jsonl"),
                pid_is_live=lambda _pid: True,
            )
        ),
        stop_timeout_seconds=2,
    )

    supervisor.start()
    _wait_for(tmp_path / "rollout-0.jsonl")
    now = datetime.now(tz=timezone.utc)
    ledger.append(
        _tool_event(
            root=tmp_path,
            event_id="tool-start",
            event_type="tool.started",
            timestamp=now,
        )
    )
    busy = ledger.materialize(now=now, stale_after_seconds=30)
    assert busy.quiescence is Quiescence.BUSY
    assert supervisor.tick(busy).phase.value == "draining"

    finished_at = datetime.now(tz=timezone.utc)
    ledger.append(
        _tool_event(
            root=tmp_path,
            event_id="tool-finish",
            event_type="tool.finished",
            timestamp=finished_at,
        )
    )
    idle = ledger.materialize(now=finished_at, stale_after_seconds=30)
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

    assert [entry["conversation_id"] for entry in starts] == [
        "native-conversation-0",
        "native-conversation-1",
    ]
    assert len({entry["chain_id"] for entry in starts}) == 1
    assert [entry["generation"] for entry in starts] == [0, 1]
    assert stops[0]["generation"] == 0
    assert len(claims) == 2
    assert [entry["claim"]["lease_epoch"] for entry in claims] == [1, 2]
    assert len(continuation) == 1
    assert continuation[0]["exact_next_action"] == (
        "Run the remaining harness integration tests."
    )
    assert not acknowledgement_path(state_path).exists()

    assert supervisor.current_process is not None
    driver.graceful_stop(supervisor.current_process, 2)
