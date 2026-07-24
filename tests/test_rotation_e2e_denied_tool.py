"""End-to-end proof that a denied tool no longer wedges rotation (BOU-2236).

The ledger-level tests prove the fold is correct. This proves the thing that
actually matters: a real `Supervisor`, driving a real child process, over a real
ledger, still completes a full rotation to an acknowledged generation 1 when a
permission gate denied a call at PreToolUse and no finish event can ever arrive.

Before the fix this test hangs at `_wait_for_quiescence(..., IDLE)` and fails:
the leaked `tool-denied` start keeps `has_active` true forever, quiescence never
reaches IDLE, and `DRAINING` only leaves for IDLE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_coordinator import JsonlClaimStore, TaskCoordinator
from test_rotation_e2e import (  # type: ignore[import-not-found]
    FAKE_RUNTIME,
    CapsuleManager,
    RolloutUsageReader,
    _set_context_tokens,
    _wait_for,
    _wait_for_quiescence,
)

from agent_session_harness.activity import Quiescence
from agent_session_harness.capsule import HandoffCapsule
from agent_session_harness.coordinator import CoordinatorAdapter
from agent_session_harness.ledger import EventLedger
from agent_session_harness.process import PosixProcessDriver
from agent_session_harness.supervisor import Supervisor, acknowledgement_path


def test_rotation_completes_even_though_a_denied_tool_never_closed(tmp_path) -> None:
    # Marker file rather than an env var: the supervisor curates the child's
    # environment through an allowlist, so an exported var never reaches it.
    (tmp_path / "deny-a-tool").write_text("deny\n", encoding="utf-8")

    state_path = tmp_path / "supervisor.json"
    lifecycle_path = state_path.with_suffix(state_path.suffix + ".lifecycle")
    ledger = EventLedger(lifecycle_path)
    driver = PosixProcessDriver(tmp_path / "process-state")
    supervisor = Supervisor(
        runtime="codex",
        chain_id="chain-1",
        cwd=tmp_path,
        task_type="linear",
        task_id="BOU-2236",
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

    try:
        supervisor.start()
        rollout_path = tmp_path / "rollout-0.jsonl"
        _wait_for(rollout_path)
        busy = _wait_for_quiescence(ledger, Quiescence.BUSY)

        # The denial is present and unbalanced before anything else happens.
        assert "tool-denied" in busy.active_tool_ids

        supervisor.tick(busy)
        _set_context_tokens(rollout_path, total_tokens=140)
        draining = supervisor.tick(busy)
        assert draining.phase.value == "draining"

        # Only `tool-0` is ever finished; `tool-denied` never will be. Reaching
        # IDLE at all is the fix -- this is where the bug used to hang.
        (tmp_path / "finish-activity-0").write_text("finish\n", encoding="utf-8")
        idle = _wait_for_quiescence(ledger, Quiescence.IDLE)
        assert idle.active_tool_ids == frozenset()
        assert "tool-denied" in idle.reaped_tool_ids

        awaiting = supervisor.tick(idle)
        assert awaiting.phase.value == "awaiting_ack"
        assert awaiting.generation == 1

        _wait_for(acknowledgement_path(state_path))
        running = supervisor.tick(idle)
        assert running.phase.value == "running"
        assert running.generation == 1

        # A real fresh successor process, not a resumed transcript.
        _wait_for(tmp_path / "continuations.jsonl")
        history = [
            json.loads(line)
            for line in (tmp_path / "history.jsonl").read_text().splitlines()
        ]
        starts = [entry for entry in history if entry["event"] == "started"]
        assert [entry["generation"] for entry in starts] == [0, 1]
        assert [entry["conversation_id"] for entry in starts] == [
            "native-conversation-0",
            "native-conversation-1",
        ]

        # And the checkpoint the whole feature exists to produce actually landed.
        assert running.checkpoint_fingerprint
        capsule_path = Path(str(running.checkpoint_path))
        assert capsule_path.is_file()
        HandoffCapsule.model_validate_json(capsule_path.read_text(encoding="utf-8"))
    finally:
        process = supervisor.current_process
        if process is not None and driver.is_alive(process):
            driver.graceful_stop(process, 2)
