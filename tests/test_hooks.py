from __future__ import annotations

from datetime import datetime, timezone
import importlib
import io
import json

import pytest


NOW = datetime(2026, 7, 19, 4, 0, tzinfo=timezone.utc)


def _modules():
    try:
        native = importlib.import_module("agent_session_harness.hooks.native")
        command = importlib.import_module("agent_session_harness.hooks.command")
    except ModuleNotFoundError:
        pytest.fail("native lifecycle hooks are not implemented")
    return native, command


@pytest.mark.parametrize("runtime", ["claude", "codex"])
@pytest.mark.parametrize(
    ("hook_name", "expected_type", "activity_id"),
    [
        ("SessionStart", "session.started", None),
        ("UserPromptSubmit", "turn.started", "turn-1"),
        ("PreToolUse", "tool.started", "tool-1"),
        ("PostToolUse", "tool.finished", "tool-1"),
        ("PostToolUseFailure", "tool.failed", "tool-1"),
        ("SubagentStart", "subagent.started", "agent-1"),
        ("SubagentStop", "subagent.finished", "agent-1"),
        ("Stop", "turn.idle", "turn-1"),
        ("PreCompact", "context.pre_compact", None),
        ("SessionEnd", "session.ended", None),
    ],
)
def test_native_events_normalize_to_sanitized_lifecycle(
    tmp_path, runtime, hook_name, expected_type, activity_id
) -> None:
    native, _command = _modules()
    payload = {
        "hook_event_name": hook_name,
        "session_id": "conversation-1",
        "cwd": str(tmp_path),
        "timestamp": NOW.isoformat(),
        "turn_id": "turn-1",
        "tool_use_id": "tool-1",
        "tool_name": "Read",
        "agent_id": "agent-1",
        "prompt": "discard this private value",
        "tool_input": {"path": "/private/value"},
    }

    event = native.normalize_native_event(
        runtime=runtime,
        payload=payload,
        chain_id="chain-1",
        generation=0,
        owner_pid=1234,
    )

    assert event.event_type.value == expected_type
    assert event.activity_id == activity_id
    assert event.conversation_id == "conversation-1"
    assert "discard this private value" not in event.model_dump_json()
    assert "/private/value" not in event.model_dump_json()


@pytest.mark.parametrize("runtime", ["claude", "codex"])
def test_stop_handshake_blocks_once_until_checkpoint_is_verified(runtime) -> None:
    native, _command = _modules()

    running = native.stop_handshake(
        runtime=runtime,
        draining=False,
        checkpoint_verified=False,
        recursion_active=False,
        already_requested=False,
        required_fields=("beads", "local-ledger"),
    )
    request = native.stop_handshake(
        runtime=runtime,
        draining=True,
        checkpoint_verified=False,
        recursion_active=False,
        already_requested=False,
        required_fields=("beads", "local-ledger"),
    )
    recursive = native.stop_handshake(
        runtime=runtime,
        draining=True,
        checkpoint_verified=False,
        recursion_active=True,
        already_requested=True,
        required_fields=("beads", "local-ledger"),
    )
    verified = native.stop_handshake(
        runtime=runtime,
        draining=True,
        checkpoint_verified=True,
        recursion_active=False,
        already_requested=True,
        required_fields=("beads", "local-ledger"),
    )

    assert running.get("decision") != "block"
    assert request["decision"] == "block"
    assert "beads" in request["reason"]
    assert request["continue"] is True
    assert recursive.get("decision") != "block"
    assert "already requested" in recursive["systemMessage"]
    assert verified.get("decision") != "block"


def test_hook_command_requires_managed_mode_and_appends_locally(tmp_path) -> None:
    _native, command = _modules()
    payload = {
        "hook_event_name": "SessionStart",
        "session_id": "conversation-1",
        "cwd": str(tmp_path),
        "timestamp": NOW.isoformat(),
    }
    stdin = io.StringIO(json.dumps(payload))
    stdout = io.StringIO()
    environ = {
        "AGENT_SESSION_HARNESS_MANAGED": "1",
        "AGENT_SESSION_HARNESS_CHAIN_ID": "chain-1",
        "AGENT_SESSION_HARNESS_GENERATION": "0",
        "AGENT_SESSION_HARNESS_LEDGER": str(tmp_path / "events.jsonl"),
        "AGENT_SESSION_HARNESS_OWNER_PID": "1234",
    }

    assert (
        command.run_hook(runtime="claude", stdin=stdin, stdout=stdout, environ=environ)
        == 0
    )
    response = json.loads(stdout.getvalue())
    assert response["ok"] is True
    assert (tmp_path / "events.jsonl").read_text().count("session.started") == 1

    with pytest.raises(RuntimeError, match="managed"):
        command.run_hook(
            runtime="claude",
            stdin=io.StringIO(json.dumps(payload)),
            stdout=io.StringIO(),
            environ={},
        )


def test_hook_command_rejects_oversized_input(tmp_path) -> None:
    _native, command = _modules()
    with pytest.raises(ValueError, match="large"):
        command.run_hook(
            runtime="codex",
            stdin=io.StringIO("x" * 1_048_577),
            stdout=io.StringIO(),
            environ={"AGENT_SESSION_HARNESS_MANAGED": "1"},
        )
