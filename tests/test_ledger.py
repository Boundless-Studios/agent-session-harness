from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib
from pathlib import Path
import stat

import pytest


NOW = datetime(2026, 7, 19, 3, 0, tzinfo=timezone.utc)


def _modules():
    try:
        models = importlib.import_module("agent_session_harness.models")
        events = importlib.import_module("agent_session_harness.events")
        ledger = importlib.import_module("agent_session_harness.ledger")
        activity = importlib.import_module("agent_session_harness.activity")
    except ModuleNotFoundError:
        pytest.fail("lifecycle ledger modules are not implemented")
    return models, events, ledger, activity


def _event(events, cwd: Path, event_id: str, event_type: str, **overrides):
    payload = {
        "schema_version": 1,
        "event_id": event_id,
        "runtime": "codex",
        "chain_id": "chain-1",
        "conversation_id": "conversation-1",
        "generation": 0,
        "event_type": event_type,
        "timestamp": NOW,
        "cwd": cwd,
        "owner_pid": 4321,
        "activity_id": "activity-1",
        "name": "Read",
    }
    payload.update(overrides)
    return events.LifecycleEvent.model_validate(payload)


def test_ledger_deduplicates_events_and_materializes_quiescence(tmp_path) -> None:
    _models, events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    started = _event(events, tmp_path, "event-1", "tool.started")
    finished = _event(
        events,
        tmp_path,
        "event-2",
        "tool.finished",
        timestamp=NOW + timedelta(seconds=1),
    )

    ledger.append(started)
    ledger.append(started)
    ledger.append(finished)
    snapshot = ledger.materialize(
        now=NOW + timedelta(seconds=2),
        stale_after_seconds=30,
    )

    assert snapshot.processed_event_count == 2
    assert snapshot.active_tool_ids == frozenset()
    assert snapshot.integrity_warnings == ()
    assert snapshot.quiescence is activity.Quiescence.IDLE
    assert stat.S_IMODE(ledger.path.stat().st_mode) == 0o600


def test_ledger_reports_busy_only_for_fresh_consistent_activity(tmp_path) -> None:
    _models, events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(_event(events, tmp_path, "event-1", "subagent.started"))

    fresh = ledger.materialize(
        now=NOW + timedelta(seconds=2),
        stale_after_seconds=30,
    )
    stale = ledger.materialize(
        now=NOW + timedelta(seconds=31),
        stale_after_seconds=30,
    )

    assert fresh.active_subagent_ids == frozenset({"activity-1"})
    assert fresh.quiescence is activity.Quiescence.BUSY
    assert stale.quiescence is activity.Quiescence.UNKNOWN


def test_finish_without_start_makes_quiescence_unknown(tmp_path) -> None:
    _models, events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(_event(events, tmp_path, "event-1", "tool.finished"))

    snapshot = ledger.materialize(
        now=NOW + timedelta(seconds=1),
        stale_after_seconds=30,
    )

    assert snapshot.quiescence is activity.Quiescence.UNKNOWN
    assert any("finish without start" in item for item in snapshot.integrity_warnings)


def test_repeated_activity_ids_balance_instead_of_wedging_quiescence(
    tmp_path,
) -> None:
    """BOU-2207: two concurrent calls may share a derived activity id.

    When a runtime supplies no tool-use id the hook derives one from the call's
    own fields, so identical calls in one prompt collide. Set-tracking would
    treat the second finish as a finish-without-start, and that warning latches
    quiescence to UNKNOWN for the life of the supervisor — rotation would never
    run again. Counting makes the pair balance exactly.
    """
    _models, events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    shared = "derived:collision"
    for index, event_type in enumerate(
        ("tool.started", "tool.started", "tool.finished", "tool.finished")
    ):
        ledger.append(
            _event(
                events,
                tmp_path,
                f"event-{index}",
                event_type,
                activity_id=shared,
            )
        )

    snapshot = ledger.materialize(
        now=NOW + timedelta(seconds=1),
        stale_after_seconds=30,
    )

    assert snapshot.active_tool_ids == frozenset()
    assert not [
        item for item in snapshot.integrity_warnings if "finish without start" in item
    ]
    assert snapshot.quiescence is activity.Quiescence.IDLE


def test_unbalanced_repeated_activity_id_still_reports_outstanding_work(
    tmp_path,
) -> None:
    """Counting must not under-report: two starts and one finish is still busy."""
    _models, events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    shared = "derived:collision"
    for index, event_type in enumerate(
        ("tool.started", "tool.started", "tool.finished")
    ):
        ledger.append(
            _event(
                events,
                tmp_path,
                f"event-{index}",
                event_type,
                activity_id=shared,
            )
        )

    snapshot = ledger.materialize(
        now=NOW + timedelta(seconds=1),
        stale_after_seconds=30,
    )

    assert snapshot.active_tool_ids == frozenset({shared})
    assert snapshot.quiescence is not activity.Quiescence.IDLE


def test_corrupt_line_is_retained_as_integrity_warning(tmp_path) -> None:
    _models, events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(_event(events, tmp_path, "event-1", "turn.started"))
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"truncated":\n')

    snapshot = ledger.materialize(
        now=NOW + timedelta(seconds=1),
        stale_after_seconds=30,
    )

    assert snapshot.quiescence is activity.Quiescence.UNKNOWN
    assert any("invalid JSON" in item for item in snapshot.integrity_warnings)


def test_one_corrupt_line_does_not_latch_quiescence_to_unknown(tmp_path) -> None:
    """BOU-2208 latch 1: a parse warning must age out, not wedge the supervisor.

    `_cached_warnings` is only cleared on file replacement, so a single
    unparseable line used to pin quiescence to UNKNOWN for the supervisor's
    whole life. `DRAINING` only leaves for `IDLE`, so the session would sit
    above the rotate threshold forever.
    """
    _models, events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(_event(events, tmp_path, "event-1", "tool.started"))
    with ledger.path.open("a", encoding="utf-8") as handle:
        handle.write('{"truncated":\n')

    corrupt = ledger.materialize(
        now=NOW + timedelta(seconds=1),
        stale_after_seconds=30,
    )

    assert corrupt.quiescence is activity.Quiescence.UNKNOWN

    ledger.append(
        _event(
            events,
            tmp_path,
            "event-2",
            "tool.finished",
            timestamp=NOW + timedelta(seconds=60),
        )
    )
    recovered = ledger.materialize(
        now=NOW + timedelta(seconds=61),
        stale_after_seconds=30,
    )

    assert recovered.active_tool_ids == frozenset()
    assert recovered.quiescence is activity.Quiescence.IDLE
    assert any("invalid JSON" in item for item in recovered.integrity_warnings)


def test_one_unmatched_finish_does_not_latch_quiescence_to_unknown(tmp_path) -> None:
    """BOU-2208 latch 1: finish-without-start is recomputed from retained history.

    Hooks installed mid-session or an interrupted tool leave exactly one
    unmatched finish, and because the warning is re-derived on every
    `materialize` it never expires on its own.
    """
    _models, events, ledger_module, activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(_event(events, tmp_path, "event-1", "tool.finished"))

    early = ledger.materialize(
        now=NOW + timedelta(seconds=1),
        stale_after_seconds=30,
    )

    assert early.quiescence is activity.Quiescence.UNKNOWN

    for index, (event_type, offset_seconds) in enumerate(
        (("turn.started", 60), ("turn.idle", 61)),
        start=2,
    ):
        ledger.append(
            _event(
                events,
                tmp_path,
                f"event-{index}",
                event_type,
                activity_id="turn-1",
                timestamp=NOW + timedelta(seconds=offset_seconds),
            )
        )
    recovered = ledger.materialize(
        now=NOW + timedelta(seconds=62),
        stale_after_seconds=30,
    )

    assert recovered.active_turn_ids == frozenset()
    assert recovered.quiescence is activity.Quiescence.IDLE


def test_truncated_ledger_keeps_gating_quiescence(tmp_path, monkeypatch) -> None:
    """Aging out parse warnings must not loosen a genuinely untrustworthy read."""
    _models, events, ledger_module, activity = _modules()
    monkeypatch.setattr(ledger_module, "MAX_LEDGER_EVENTS", 1)
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(_event(events, tmp_path, "event-1", "tool.started"))
    ledger.append(
        _event(
            events,
            tmp_path,
            "event-2",
            "tool.finished",
            timestamp=NOW + timedelta(seconds=1),
        )
    )

    snapshot = ledger.materialize(
        now=NOW + timedelta(days=1),
        stale_after_seconds=30,
    )

    assert snapshot.quiescence is activity.Quiescence.UNKNOWN
    assert any("event limit" in item for item in snapshot.integrity_warnings)


def test_repeated_materialization_reads_only_the_new_ledger_tail(
    tmp_path,
    monkeypatch,
) -> None:
    _models, events, ledger_module, _activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    offsets: list[int] = []
    original = ledger_module.read_private_text_incremental

    def tracking_read(path, **kwargs):
        offsets.append(kwargs["offset"])
        return original(path, **kwargs)

    monkeypatch.setattr(ledger_module, "read_private_text_incremental", tracking_read)
    ledger.append(_event(events, tmp_path, "event-1", "tool.started"))
    ledger.materialize(now=NOW, stale_after_seconds=30)
    ledger.append(
        _event(
            events,
            tmp_path,
            "event-2",
            "tool.finished",
            timestamp=NOW + timedelta(seconds=1),
        )
    )
    snapshot = ledger.materialize(
        now=NOW + timedelta(seconds=2),
        stale_after_seconds=30,
    )

    assert offsets[0] == 0
    assert offsets[1] > 0
    assert snapshot.quiescence.value == "idle"


def test_ledger_refuses_to_append_past_its_byte_bound(
    tmp_path,
    monkeypatch,
) -> None:
    _models, events, ledger_module, _activity = _modules()
    ledger = ledger_module.EventLedger(tmp_path / "events.jsonl")
    ledger.append(_event(events, tmp_path, "event-1", "tool.started"))
    monkeypatch.setattr(
        ledger_module,
        "MAX_LEDGER_BYTES",
        ledger.path.stat().st_size + 1,
    )

    with pytest.raises(ValueError, match="byte limit"):
        ledger.append(_event(events, tmp_path, "event-2", "tool.finished"))
