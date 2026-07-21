"""Turn-idle reconciliation for tool starts a permission gate never closes.

BOU-2236. Claude Code fires `PreToolUse` (-> `tool.started`) BEFORE the
permission gate runs. When a gate such as warden denies the call the tool never
executes, so neither `PostToolUse` (-> `tool.finished`) nor `PostToolUseFailure`
(-> `tool.failed`) is ever emitted. The start is never balanced, `has_active`
stays true for the life of the session, quiescence can never return `IDLE`, and
`DRAINING` only leaves for `IDLE` -- so rotation is blocked permanently. A
single denied command is enough.

`Stop` (-> `turn.idle`) is the runtime's own statement that the turn finished,
which means no tool belonging to that turn can still be in flight. Reaping
outstanding tools there restores the invariant without needing a
denial-specific signal the runtime does not emit.

Observed in the wild before the fix: four live gaia chains parked in `DRAINING`
at 89-135% reported context with `generation: 0` and `checkpoint_fingerprint:
null`, each carrying 6-11 unbalanced tool starts that mapped one-to-one onto
warden denials in the session transcript.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib
from pathlib import Path

import pytest


NOW = datetime(2026, 7, 21, 3, 0, tzinfo=timezone.utc)


def _modules():
    try:
        events = importlib.import_module("agent_session_harness.events")
        ledger = importlib.import_module("agent_session_harness.ledger")
        activity = importlib.import_module("agent_session_harness.activity")
    except ModuleNotFoundError:
        pytest.fail("lifecycle ledger modules are not implemented")
    return events, ledger, activity


def _event(events, cwd: Path, event_id: str, event_type: str, **overrides):
    payload = {
        "schema_version": 1,
        "event_id": event_id,
        "runtime": "claude",
        "chain_id": "chain-1",
        "conversation_id": "conversation-1",
        "generation": 0,
        "event_type": event_type,
        "timestamp": NOW,
        "cwd": cwd,
        "owner_pid": 4321,
        "activity_id": "activity-1",
        "name": "Bash",
    }
    payload.update(overrides)
    return events.LifecycleEvent.model_validate(payload)


def test_turn_idle_reaps_tools_a_denied_call_never_closed(tmp_path) -> None:
    events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(
        _event(events, tmp_path, "event-1", "turn.started", activity_id="turn-1")
    )
    # Ran and completed normally.
    ledger.append(
        _event(events, tmp_path, "event-2", "tool.started", activity_id="tool-ok")
    )
    ledger.append(
        _event(events, tmp_path, "event-3", "tool.finished", activity_id="tool-ok")
    )
    # Denied at the PreToolUse gate: a start with no possible finish.
    ledger.append(
        _event(events, tmp_path, "event-4", "tool.started", activity_id="tool-denied")
    )

    busy = ledger.materialize(now=NOW + timedelta(seconds=1), stale_after_seconds=30)
    assert busy.active_tool_ids == frozenset({"tool-denied"})
    assert busy.quiescence is activity.Quiescence.BUSY

    ledger.append(
        _event(
            events,
            tmp_path,
            "event-5",
            "turn.idle",
            activity_id="turn-1",
            timestamp=NOW + timedelta(seconds=2),
        )
    )
    snapshot = ledger.materialize(
        now=NOW + timedelta(seconds=3), stale_after_seconds=30
    )

    assert snapshot.active_tool_ids == frozenset(), (
        "a PreToolUse-denied tool is still counted as running after the turn "
        "ended -- quiescence can never reach IDLE and rotation stays wedged"
    )
    assert snapshot.active_turn_ids == frozenset()
    assert snapshot.quiescence is activity.Quiescence.IDLE
    # Reaping is routine reconciliation, not a ledger fault. A gating warning
    # would hold quiescence at UNKNOWN and re-block the very rotation this fixes.
    assert snapshot.integrity_warnings == ()
    assert snapshot.reaped_tool_ids == frozenset({"tool-denied"})


def test_reaped_tools_stay_reaped_across_repeated_materialize(tmp_path) -> None:
    """`materialize` re-folds retained history on every read.

    The reap must therefore be a property of the fold, not a one-shot mutation,
    or the leak reappears on the very next governor tick.
    """
    events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(
        _event(events, tmp_path, "event-0", "turn.started", activity_id="turn-1")
    )
    ledger.append(
        _event(events, tmp_path, "event-1", "tool.started", activity_id="tool-denied")
    )
    ledger.append(
        _event(
            events,
            tmp_path,
            "event-2",
            "turn.idle",
            activity_id="turn-1",
            timestamp=NOW + timedelta(seconds=1),
        )
    )

    for offset in (2, 3, 4):
        snapshot = ledger.materialize(
            now=NOW + timedelta(seconds=offset), stale_after_seconds=30
        )
        assert snapshot.active_tool_ids == frozenset()
        assert snapshot.quiescence is activity.Quiescence.IDLE


def test_a_tool_started_after_turn_idle_is_still_tracked(tmp_path) -> None:
    """Reaping must not swallow work from the NEXT turn.

    The fold is ordered, so a start arriving after the turn.idle belongs to a
    later turn and must keep quiescence BUSY.
    """
    events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(
        _event(events, tmp_path, "event-0", "turn.started", activity_id="turn-1")
    )
    ledger.append(
        _event(events, tmp_path, "event-1", "tool.started", activity_id="tool-denied")
    )
    ledger.append(
        _event(
            events,
            tmp_path,
            "event-2",
            "turn.idle",
            activity_id="turn-1",
            timestamp=NOW + timedelta(seconds=1),
        )
    )
    ledger.append(
        _event(
            events,
            tmp_path,
            "event-3",
            "tool.started",
            activity_id="tool-next-turn",
            timestamp=NOW + timedelta(seconds=2),
        )
    )

    snapshot = ledger.materialize(
        now=NOW + timedelta(seconds=3), stale_after_seconds=30
    )

    assert snapshot.active_tool_ids == frozenset({"tool-next-turn"})
    assert snapshot.quiescence is activity.Quiescence.BUSY


def test_turn_idle_does_not_reap_a_live_background_subagent(tmp_path) -> None:
    """Reaping is scoped to tools; an outstanding subagent still blocks IDLE.

    A background subagent can outlive the main turn, so turn.idle must not be
    read as "the whole session is quiet". Subagent tracking is what keeps
    quiescence BUSY in that case -- which is also precisely why reaping tools
    at turn.idle is safe.
    """
    events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(
        _event(events, tmp_path, "event-1", "turn.started", activity_id="turn-1")
    )
    ledger.append(
        _event(events, tmp_path, "event-2", "subagent.started", activity_id="agent-1")
    )
    ledger.append(
        _event(events, tmp_path, "event-3", "tool.started", activity_id="tool-denied")
    )
    ledger.append(
        _event(
            events,
            tmp_path,
            "event-4",
            "turn.idle",
            activity_id="turn-1",
            timestamp=NOW + timedelta(seconds=1),
        )
    )

    snapshot = ledger.materialize(
        now=NOW + timedelta(seconds=2), stale_after_seconds=30
    )

    assert snapshot.active_subagent_ids == frozenset({"agent-1"})
    assert snapshot.quiescence is activity.Quiescence.BUSY


def test_critical_section_still_blocks_idle_after_turn_idle(tmp_path) -> None:
    """A critical section is an explicit "do not rotate through me" claim."""
    events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(
        _event(
            events,
            tmp_path,
            "event-1",
            "critical_section.entered",
            activity_id="crit-1",
            name="checkpoint",
        )
    )
    ledger.append(
        _event(events, tmp_path, "event-0", "turn.started", activity_id="turn-1")
    )
    ledger.append(
        _event(events, tmp_path, "event-2", "tool.started", activity_id="tool-denied")
    )
    ledger.append(
        _event(
            events,
            tmp_path,
            "event-3",
            "turn.idle",
            activity_id="turn-1",
            timestamp=NOW + timedelta(seconds=1),
        )
    )

    snapshot = ledger.materialize(
        now=NOW + timedelta(seconds=2), stale_after_seconds=30
    )

    assert snapshot.active_critical_section_ids == frozenset({"crit-1"})
    assert snapshot.quiescence is activity.Quiescence.BUSY
