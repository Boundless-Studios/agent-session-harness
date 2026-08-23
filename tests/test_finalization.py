from pathlib import Path

import pytest

from agent_session_harness import finalization
from agent_session_harness.finalization import (
    FinalizationPhase,
    FinalizationStore,
)


def test_begin_is_idempotent_and_requires_a_human_summary(tmp_path: Path) -> None:
    store = FinalizationStore(tmp_path / "finalization.json")

    first = store.begin("session-1", "Implemented PI lifecycle reliability.")
    second = store.begin("session-1", "Implemented PI lifecycle reliability.")

    assert first == second
    assert first.phase is FinalizationPhase.ACTIVE


def test_block_survives_restart_until_matching_acknowledgement(tmp_path: Path) -> None:
    path = tmp_path / "finalization.json"
    store = FinalizationStore(path)
    store.begin("session-1", "Implemented PI lifecycle reliability.")
    blocked = store.record_block("dispatch-7", "Open and settle the pull request.")

    assert blocked.phase is FinalizationPhase.BLOCKED
    assert blocked.pending_block == "Open and settle the pull request."

    restarted = FinalizationStore(path)
    assert restarted.load().block_dispatch_id == "dispatch-7"
    assert restarted.acknowledge_block("dispatch-other").pending_block is not None
    assert restarted.acknowledge_block("dispatch-7").pending_block is None


def test_retro_summary_and_finalization_are_exactly_once(tmp_path: Path) -> None:
    store = FinalizationStore(tmp_path / "finalization.json")
    store.begin("session-1", "Implemented PI lifecycle reliability.")

    assert store.mark_retro_submitted().retro_submitted is True
    assert store.mark_retro_submitted().retro_submitted is True
    assert store.mark_summary_surfaced().summary_surfaced is True
    assert store.mark_summary_surfaced().summary_surfaced is True
    assert store.finalize().phase is FinalizationPhase.FINALIZED
    assert store.finalize().phase is FinalizationPhase.FINALIZED


def test_finalize_requires_no_block_and_completed_human_artifacts(
    tmp_path: Path,
) -> None:
    store = FinalizationStore(tmp_path / "finalization.json")
    store.begin("session-1", "Implemented PI lifecycle reliability.")

    with pytest.raises(ValueError, match="retro and summary"):
        store.finalize()

    store.mark_retro_submitted()
    store.mark_summary_surfaced()
    store.record_block("dispatch-7", "Settle required review.")

    with pytest.raises(ValueError, match="pending block"):
        store.finalize()


def test_finalized_record_cannot_be_reopened_by_a_stale_runtime(tmp_path: Path) -> None:
    store = FinalizationStore(tmp_path / "finalization.json")
    store.begin("session-1", "Done.")
    store.mark_retro_submitted()
    store.mark_summary_surfaced()
    store.finalize()

    with pytest.raises(ValueError, match="after finalization"):
        store.record_block("late-dispatch", "late block")

    assert store.load().phase is FinalizationPhase.FINALIZED


def test_updates_are_validated_before_they_replace_durable_state(
    tmp_path: Path,
) -> None:
    store = FinalizationStore(tmp_path / "finalization.json")
    original = store.begin("session-1", "Done.")

    with pytest.raises(ValueError):
        store.record_block("x" * 161, "blocked")

    assert store.load() == original


def test_record_block_rejects_a_blank_message_without_changing_state(
    tmp_path: Path,
) -> None:
    store = FinalizationStore(tmp_path / "finalization.json")
    original = store.begin("session-1", "Done.")

    with pytest.raises(ValueError):
        store.record_block("dispatch-7", " \t\n")

    assert store.load() == original


def test_load_rejects_state_larger_than_the_durable_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "finalization.json"
    path.write_text(" " * 129, encoding="utf-8")
    monkeypatch.setattr(finalization, "MAX_FINALIZATION_STATE_BYTES", 128)

    with pytest.raises(ValueError, match="private file exceeds 128 bytes"):
        FinalizationStore(path).load()


def test_future_schema_version_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "finalization.json"
    path.write_text(
        '{"schema_version":2,"session_id":"session-1","summary":"Done.",'
        '"phase":"active","pending_block":null,"block_dispatch_id":null,'
        '"retro_submitted":false,"summary_surfaced":false}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        FinalizationStore(path).load()
