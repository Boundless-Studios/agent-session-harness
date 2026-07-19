from __future__ import annotations

from datetime import datetime, timezone
import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError


def _modules():
    try:
        return (
            importlib.import_module("agent_session_harness.models"),
            importlib.import_module("agent_session_harness.events"),
        )
    except ModuleNotFoundError:
        pytest.fail("agent_session_harness lifecycle models are not implemented")


def _payload(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "event_id": "event-1",
        "runtime": "claude",
        "chain_id": "chain-1",
        "conversation_id": "conversation-1",
        "generation": 0,
        "event_type": "tool.started",
        "timestamp": "2026-07-18T20:00:00-07:00",
        "cwd": str(tmp_path),
        "owner_pid": 1234,
        "activity_id": "tool-1",
        "name": "Read",
    }


def test_lifecycle_event_requires_sanitized_versioned_identity(tmp_path) -> None:
    models, events = _modules()
    event = events.LifecycleEvent.model_validate(_payload(tmp_path))

    assert event.schema_version == 1
    assert event.event_type is models.EventType.TOOL_STARTED
    assert event.runtime is models.Runtime.CLAUDE
    assert event.timestamp == datetime(2026, 7, 19, 3, 0, tzinfo=timezone.utc)
    assert event.cwd == tmp_path.resolve()

    required = {
        "schema_version",
        "event_id",
        "runtime",
        "chain_id",
        "conversation_id",
        "generation",
        "event_type",
        "timestamp",
        "cwd",
        "owner_pid",
    }
    assert required <= set(events.LifecycleEvent.model_fields)


def test_lifecycle_event_rejects_unknown_types_and_private_payloads(tmp_path) -> None:
    _models, events = _modules()
    unknown = _payload(tmp_path)
    unknown["event_type"] = "tool.teleported"
    with pytest.raises(ValidationError):
        events.LifecycleEvent.model_validate(unknown)

    private = _payload(tmp_path)
    private["prompt_text"] = "do not persist me"
    with pytest.raises(ValidationError):
        events.LifecycleEvent.model_validate(private)

    arbitrary = _payload(tmp_path)
    arbitrary["payload"] = {"tool_input": "secret"}
    with pytest.raises(ValidationError):
        events.LifecycleEvent.model_validate(arbitrary)

    assert not {
        "environment",
        "payload",
        "prompt_text",
        "tool_input",
        "transcript",
    } & set(events.LifecycleEvent.model_fields)


@pytest.mark.parametrize(
    "event_type",
    [
        "turn.started",
        "turn.idle",
        "tool.started",
        "tool.finished",
        "tool.failed",
        "subagent.started",
        "subagent.finished",
        "critical_section.entered",
        "critical_section.exited",
    ],
)
def test_activity_events_require_an_opaque_activity_id(tmp_path, event_type) -> None:
    _models, events = _modules()
    payload = _payload(tmp_path)
    payload["event_type"] = event_type
    payload["activity_id"] = None

    with pytest.raises(ValidationError, match="activity_id"):
        events.LifecycleEvent.model_validate(payload)


def test_lifecycle_event_rejects_naive_timestamp_and_relative_cwd(tmp_path) -> None:
    _models, events = _modules()
    naive = _payload(tmp_path)
    naive["timestamp"] = "2026-07-18T20:00:00"
    with pytest.raises(ValidationError, match="timezone"):
        events.LifecycleEvent.model_validate(naive)

    relative = _payload(tmp_path)
    relative["cwd"] = "relative/path"
    with pytest.raises(ValidationError, match="absolute"):
        events.LifecycleEvent.model_validate(relative)
