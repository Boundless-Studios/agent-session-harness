from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib
import io
import json
from pathlib import Path
import stat
import threading
import time

import pytest


NOW = datetime(2026, 7, 19, 4, 0, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parent / "fixtures" / "hooks"


def _modules():
    try:
        native = importlib.import_module("agent_session_harness.hooks.native")
        command = importlib.import_module("agent_session_harness.hooks.command")
    except ModuleNotFoundError:
        pytest.fail("native lifecycle hooks are not implemented")
    return native, command


def test_unmanaged_hook_is_a_silent_noop() -> None:
    _native, command = _modules()
    stdin = io.StringIO("not even JSON")
    stdout = io.StringIO()

    assert (
        command.run_hook(
            runtime="codex",
            stdin=stdin,
            stdout=stdout,
            environ={},
        )
        == 0
    )
    assert stdout.getvalue() == ""
    assert stdin.tell() == 0


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


@pytest.mark.parametrize(
    "hook_name",
    [
        "SubagentStart",
        "SubagentStop",
    ],
)
def test_concurrent_subagent_activity_without_an_opaque_id_fails_closed(
    tmp_path,
    hook_name,
) -> None:
    """Subagent events still fail closed — they do carry `agent_id`.

    Tool events deliberately no longer do; see the companion test below.
    """
    native, _command = _modules()
    payload = {
        "hook_event_name": hook_name,
        "session_id": "conversation-1",
        "cwd": str(tmp_path),
        "timestamp": NOW.isoformat(),
    }

    with pytest.raises(ValueError, match="activity ID"):
        native.normalize_native_event(
            runtime="codex",
            payload=payload,
            chain_id="chain-1",
            generation=0,
            owner_pid=1234,
        )


@pytest.mark.parametrize(
    "hook_name",
    ["PreToolUse", "PostToolUse", "PostToolUseFailure"],
)
def test_tool_hooks_derive_an_activity_id_rather_than_blocking(
    tmp_path,
    hook_name,
) -> None:
    """BOU-2207: failing closed here blocks every tool call in a managed session.

    Claude Code's documented tool payloads carry no tool-use identifier, and a
    raised ValueError becomes exit 2, which for PreToolUse blocks the tool. The
    harness must not depend on an undocumented field.
    """
    native, _command = _modules()
    payload = {
        "hook_event_name": hook_name,
        "session_id": "conversation-1",
        "cwd": str(tmp_path),
        "timestamp": NOW.isoformat(),
        "tool_name": "Bash",
    }

    event = native.normalize_native_event(
        runtime="claude",
        payload=payload,
        chain_id="chain-1",
        generation=0,
        owner_pid=1234,
    )

    assert event.activity_id
    assert event.activity_id.startswith("derived:")


def test_pre_and_post_tool_hooks_derive_the_same_activity_id(tmp_path) -> None:
    """The ledger pairs starts against finishes; a mismatch never reaches IDLE.

    If Pre and Post derived different ids, `active_tools` would never empty and
    rotation would stall forever — trading a blocking bug for a silent one.
    """
    native, _command = _modules()

    def _event(hook_name: str):
        return native.normalize_native_event(
            runtime="claude",
            payload={
                "hook_event_name": hook_name,
                "session_id": "conversation-1",
                "cwd": str(tmp_path),
                "timestamp": NOW.isoformat(),
                "prompt_id": "prompt-1",
                "tool_name": "Bash",
            },
            chain_id="chain-1",
            generation=0,
            owner_pid=1234,
        )

    assert _event("PreToolUse").activity_id == _event("PostToolUse").activity_id


def test_supplied_tool_use_id_wins_over_the_derived_id(tmp_path) -> None:
    native, _command = _modules()

    event = native.normalize_native_event(
        runtime="claude",
        payload={
            "hook_event_name": "PreToolUse",
            "session_id": "conversation-1",
            "cwd": str(tmp_path),
            "timestamp": NOW.isoformat(),
            "tool_name": "Bash",
            "tool_use_id": "toolu_01REAL",
        },
        chain_id="chain-1",
        generation=0,
        owner_pid=1234,
    )

    assert event.activity_id == "toolu_01REAL"


@pytest.mark.parametrize("runtime", ["claude", "codex"])
def test_stop_handshake_encodes_runtime_native_block_response(runtime) -> None:
    native, _command = _modules()
    request = native.stop_handshake(
        runtime=runtime,
        draining=True,
        checkpoint_verified=False,
        already_requested=False,
        required_fields=("beads", "local-ledger"),
    )
    expected = json.loads((FIXTURES / f"{runtime}-stop-block.json").read_text())

    assert request.exit_code == expected["exit_code"]
    assert request.stdout == expected["stdout"]
    assert request.stderr == expected["stderr"]


def _write_supervisor_state(
    path: Path,
    *,
    runtime: str,
    phase: str,
    generation: int = 0,
    checkpoint_fingerprint: str | None = None,
    process_pid: int = 1234,
) -> None:
    supervisor = importlib.import_module("agent_session_harness.supervisor")
    snapshot = supervisor.SupervisorSnapshot(
        runtime=runtime,
        chain_id="chain-1",
        generation=generation,
        phase=phase,
        process_pid=process_pid,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )
    path.write_text(snapshot.model_dump_json() + "\n", encoding="utf-8")
    path.chmod(0o600)


def _stop_environment(tmp_path: Path, state_path: Path, *, generation: int = 0):
    return {
        "AGENT_SESSION_HARNESS_MANAGED": "1",
        "AGENT_SESSION_HARNESS_CHAIN_ID": "chain-1",
        "AGENT_SESSION_HARNESS_GENERATION": str(generation),
        "AGENT_SESSION_HARNESS_LEDGER": str(tmp_path / "events.jsonl"),
        "AGENT_SESSION_HARNESS_OWNER_PID": "1234",
        "AGENT_SESSION_HARNESS_STATE_PATH": str(state_path),
        "AGENT_SESSION_HARNESS_REQUIRED_CHECKPOINTS": "beads,local-ledger",
    }


def _run_stop(
    command,
    *,
    runtime: str,
    tmp_path: Path,
    environ,
    timestamp: datetime = NOW,
):
    payload = {
        "hook_event_name": "Stop",
        "session_id": "conversation-1",
        "cwd": str(tmp_path),
        "timestamp": timestamp.isoformat(),
        "turn_id": "turn-1",
        "prompt": "private prompt must never persist",
        "tool_input": {"secret": "private tool input"},
    }
    stdout = io.StringIO()
    stderr = io.StringIO()
    exit_code = command.run_hook(
        runtime=runtime,
        stdin=io.StringIO(json.dumps(payload)),
        stdout=stdout,
        stderr=stderr,
        environ=environ,
    )
    return exit_code, stdout.getvalue(), stderr.getvalue()


@pytest.mark.parametrize("runtime", ["claude", "codex"])
@pytest.mark.parametrize("phase", ["draining", "checkpointing"])
def test_first_draining_stop_blocks_without_idle_and_repeat_allows(
    tmp_path: Path,
    runtime: str,
    phase: str,
) -> None:
    _native, command = _modules()
    state_path = tmp_path / "supervisor.json"
    _write_supervisor_state(state_path, runtime=runtime, phase=phase)
    environ = _stop_environment(tmp_path, state_path)
    # Static flags are deliberately contradictory; state is authoritative.
    environ["AGENT_SESSION_HARNESS_CHECKPOINT_VERIFIED"] = "1"

    first_code, first_stdout, first_stderr = _run_stop(
        command,
        runtime=runtime,
        tmp_path=tmp_path,
        environ=environ,
    )

    expected = json.loads((FIXTURES / f"{runtime}-stop-block.json").read_text())
    assert first_code == expected["exit_code"]
    assert (json.loads(first_stdout) if first_stdout else None) == expected["stdout"]
    assert first_stderr == expected["stderr"]
    ledger_text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert ledger_text.count("handoff.requested") == 1
    assert "turn.idle" not in ledger_text
    assert "private prompt" not in ledger_text
    assert "private tool input" not in ledger_text
    marker_path = command.stop_request_path(state_path)
    assert stat.S_IMODE(marker_path.stat().st_mode) == 0o600

    repeat_code, repeat_stdout, repeat_stderr = _run_stop(
        command,
        runtime=runtime,
        tmp_path=tmp_path,
        environ=environ,
    )

    assert repeat_code == 0
    if runtime == "claude":
        assert "already requested" in json.loads(repeat_stdout)["systemMessage"]
        assert repeat_stderr == ""
    else:
        assert repeat_stdout == ""
        assert "already requested" in repeat_stderr
    ledger_text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert ledger_text.count("handoff.requested") == 1
    assert ledger_text.count("turn.idle") == 1

    _run_stop(
        command,
        runtime=runtime,
        tmp_path=tmp_path,
        environ=environ,
        timestamp=NOW + timedelta(seconds=1),
    )
    ledger_text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert ledger_text.count("turn.idle") == 1

    _write_supervisor_state(
        state_path,
        runtime=runtime,
        phase=phase,
        generation=1,
    )
    next_environ = _stop_environment(tmp_path, state_path, generation=1)
    next_code, next_stdout, next_stderr = _run_stop(
        command,
        runtime=runtime,
        tmp_path=tmp_path,
        environ=next_environ,
        timestamp=NOW + timedelta(seconds=2),
    )
    assert next_code == expected["exit_code"]
    assert (json.loads(next_stdout) if next_stdout else None) == expected["stdout"]
    assert next_stderr == expected["stderr"]
    ledger_text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert ledger_text.count("handoff.requested") == 2


@pytest.mark.parametrize(
    ("phase", "checkpoint_fingerprint"),
    [("running", None), ("checkpointed", "f" * 64)],
)
@pytest.mark.parametrize("runtime", ["claude", "codex"])
def test_normal_or_verified_stop_allows_and_appends_idle(
    tmp_path: Path,
    runtime: str,
    phase: str,
    checkpoint_fingerprint: str | None,
) -> None:
    _native, command = _modules()
    state_path = tmp_path / "supervisor.json"
    _write_supervisor_state(
        state_path,
        runtime=runtime,
        phase=phase,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )
    environ = _stop_environment(tmp_path, state_path)
    environ["AGENT_SESSION_HARNESS_DRAINING"] = "1"

    exit_code, stdout, stderr = _run_stop(
        command,
        runtime=runtime,
        tmp_path=tmp_path,
        environ=environ,
    )

    assert exit_code == 0
    assert stderr == ""
    if runtime == "claude":
        assert json.loads(stdout) == {"continue": True}
    else:
        assert stdout == ""
    ledger_text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert ledger_text.count("turn.idle") == 1
    assert "handoff.requested" not in ledger_text


def test_stop_rejects_an_owner_other_than_the_managed_runtime(tmp_path: Path) -> None:
    _native, command = _modules()
    state_path = tmp_path / "supervisor.json"
    _write_supervisor_state(
        state_path,
        runtime="codex",
        phase="draining",
        process_pid=4321,
    )
    environ = _stop_environment(tmp_path, state_path)

    with pytest.raises(RuntimeError, match="managed Stop event"):
        _run_stop(
            command,
            runtime="codex",
            tmp_path=tmp_path,
            environ=environ,
        )

    assert not (tmp_path / "events.jsonl").exists()


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
    assert stdout.getvalue() == ""
    assert (tmp_path / "events.jsonl").read_text().count("session.started") == 1

    unmanaged_stdout = io.StringIO()
    assert (
        command.run_hook(
            runtime="claude",
            stdin=io.StringIO(json.dumps(payload)),
            stdout=unmanaged_stdout,
            environ={},
        )
        == 0
    )
    assert unmanaged_stdout.getvalue() == ""


@pytest.mark.parametrize("runtime", ["claude", "codex"])
def test_session_start_auto_acknowledges_verified_successor(
    tmp_path: Path,
    runtime: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _native, command = _modules()
    capsule_module = importlib.import_module("agent_session_harness.capsule")
    supervisor = importlib.import_module("agent_session_harness.supervisor")
    capsule = capsule_module.HandoffCapsule(
        schema_version=1,
        chain_id="chain-1",
        predecessor_conversation_id="conversation-0",
        target_generation=1,
        task_ids={"linear": "BOU-2195", "bead": "bead-1"},
        objective="Continue the verified task.",
        exact_next_action="Run the focused successor test.",
        completed_criteria=("predecessor stopped",),
        remaining_criteria=("successor acknowledged",),
        repository_path=tmp_path,
        branch="test-branch",
        head="deadbeef",
        dirty_paths=(),
        file_anchors=("tests/test_hooks.py",),
        symbol_anchors=("run_hook",),
        test_results={"focused": "running"},
        decisions=("fresh successor only",),
        blockers=(),
        process_summaries={"predecessor": "stopped"},
        created_at=NOW,
    )
    capsule_path = tmp_path / "capsule.json"
    capsule_path.write_bytes(capsule.canonical_bytes() + b"\n")
    state_path = tmp_path / "supervisor.json"
    launching = supervisor.SupervisorSnapshot(
        runtime=runtime,
        chain_id="chain-1",
        generation=1,
        phase="launching",
        checkpoint_fingerprint=capsule.fingerprint,
        checkpoint_path=capsule_path,
    )
    ready = launching.model_copy(
        update={
            "phase": supervisor.SupervisorPhase.AWAITING_ACK,
            "process_pid": 1234,
        }
    )
    state_path.write_text(launching.model_dump_json() + "\n", encoding="utf-8")
    state_path.chmod(0o600)
    payload = {
        "hook_event_name": "SessionStart",
        "session_id": "conversation-1",
        "cwd": str(tmp_path),
        "timestamp": NOW.isoformat(),
    }
    environ = _stop_environment(tmp_path, state_path, generation=1)
    environ.update(
        {
            "AGENT_SESSION_HARNESS_CAPSULE_PATH": str(capsule_path),
            "AGENT_SESSION_HARNESS_CAPSULE_FINGERPRINT": capsule.fingerprint,
            "AGENT_SESSION_HARNESS_TARGET_GENERATION": "1",
        }
    )

    first_read = threading.Event()
    original_read = command._read_snapshot
    read_count = 0

    def observed_read(path):
        nonlocal read_count
        snapshot = original_read(path)
        read_count += 1
        if read_count == 1:
            first_read.set()
        return snapshot

    monkeypatch.setattr(command, "_read_snapshot", observed_read)

    def publish_ready_state() -> None:
        assert first_read.wait(timeout=1)
        secure_files = importlib.import_module("agent_session_harness.secure_files")
        with secure_files.exclusive_lock(
            state_path.with_suffix(state_path.suffix + ".lock")
        ):
            secure_files.atomic_write_private_text(
                state_path,
                ready.model_dump_json() + "\n",
            )
        deadline = time.monotonic() + 1
        while supervisor.read_acknowledgement(state_path) is None:
            if time.monotonic() >= deadline:
                raise AssertionError("successor did not publish its acknowledgement")
            time.sleep(0.01)
        running = ready.model_copy(
            update={
                "phase": supervisor.SupervisorPhase.RUNNING,
                "conversation_id": "conversation-1",
            }
        )
        with secure_files.exclusive_lock(
            state_path.with_suffix(state_path.suffix + ".lock")
        ):
            secure_files.atomic_write_private_text(
                state_path,
                running.model_dump_json() + "\n",
            )

    publisher = threading.Thread(target=publish_ready_state)
    publisher.start()

    try:
        exit_code = command.run_hook(
            runtime=runtime,
            stdin=io.StringIO(json.dumps(payload)),
            stdout=io.StringIO(),
            environ=environ,
        )
    finally:
        publisher.join(timeout=1)

    acknowledgement = supervisor.read_acknowledgement(state_path)
    assert exit_code == 0
    assert acknowledgement is not None
    assert acknowledgement.conversation_id == "conversation-1"
    assert acknowledgement.generation == 1
    assert acknowledgement.fingerprint == capsule.fingerprint
    assert acknowledgement.owner_pid == 1234
    assert read_count >= 2
    assert (tmp_path / "events.jsonl").read_text().count("session.started") == 1


def test_session_start_blocks_when_durable_acknowledgement_never_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _native, command = _modules()
    supervisor = importlib.import_module("agent_session_harness.supervisor")
    capsule_module = importlib.import_module("agent_session_harness.capsule")
    capsule = capsule_module.HandoffCapsule(
        schema_version=1,
        chain_id="chain-1",
        predecessor_conversation_id="conversation-0",
        target_generation=1,
        task_ids={"bead": "bead-1"},
        objective="Continue safely.",
        exact_next_action="Do not run before ACK.",
        completed_criteria=(),
        remaining_criteria=("durable ACK",),
        repository_path=tmp_path,
        branch="test-branch",
        head="deadbeef",
        dirty_paths=(),
        file_anchors=(),
        symbol_anchors=(),
        test_results={},
        decisions=(),
        blockers=(),
        process_summaries={},
        created_at=NOW,
    )
    capsule_path = tmp_path / "capsule.json"
    capsule_path.write_bytes(capsule.canonical_bytes() + b"\n")
    state_path = tmp_path / "supervisor.json"
    snapshot = supervisor.SupervisorSnapshot(
        runtime="codex",
        chain_id="chain-1",
        generation=1,
        phase="awaiting_ack",
        process_pid=1234,
        checkpoint_fingerprint=capsule.fingerprint,
        checkpoint_path=capsule_path,
    )
    state_path.write_text(snapshot.model_dump_json() + "\n", encoding="utf-8")
    state_path.chmod(0o600)
    environ = _stop_environment(tmp_path, state_path, generation=1)
    environ.update(
        {
            "AGENT_SESSION_HARNESS_CAPSULE_PATH": str(capsule_path),
            "AGENT_SESSION_HARNESS_CAPSULE_FINGERPRINT": capsule.fingerprint,
            "AGENT_SESSION_HARNESS_TARGET_GENERATION": "1",
        }
    )
    monkeypatch.setattr(command, "SUCCESSOR_ACK_TIMEOUT_SECONDS", 0.02)

    def abort_successor(**_kwargs) -> None:
        raise RuntimeError("unacknowledged successor was terminated")

    monkeypatch.setattr(command, "_abort_unacknowledged_successor", abort_successor)

    with pytest.raises(RuntimeError, match="successor was terminated"):
        command.run_hook(
            runtime="codex",
            stdin=io.StringIO(
                json.dumps(
                    {
                        "hook_event_name": "SessionStart",
                        "session_id": "conversation-1",
                        "cwd": str(tmp_path),
                        "timestamp": NOW.isoformat(),
                    }
                )
            ),
            stdout=io.StringIO(),
            environ=environ,
        )


def test_failed_successor_session_start_requests_abort_and_never_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _native, command = _modules()
    process = importlib.import_module("agent_session_harness.process")
    state_path = tmp_path / "supervisor.json"
    environ = _stop_environment(tmp_path, state_path, generation=1)
    events = importlib.import_module("agent_session_harness.events")
    event = events.LifecycleEvent(
        schema_version=1,
        event_id="session-start:chain-1:1",
        runtime="claude",
        chain_id="chain-1",
        conversation_id="conversation-1",
        generation=1,
        event_type="session.started",
        timestamp=NOW,
        cwd=tmp_path,
        owner_pid=1234,
    )

    class AbortWaitObserved(RuntimeError):
        pass

    monkeypatch.setattr(
        command.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(AbortWaitObserved("waiting")),
    )

    with pytest.raises(AbortWaitObserved, match="waiting"):
        command._abort_unacknowledged_successor(event=event, environment=environ)

    marker = process.read_runtime_abort(state_path)
    assert marker is not None
    assert marker.chain_id == "chain-1"
    assert marker.generation == 1
    assert marker.owner_pid == 1234


def test_abort_never_returns_when_durable_marker_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _native, command = _modules()
    events = importlib.import_module("agent_session_harness.events")
    event = events.LifecycleEvent(
        schema_version=1,
        event_id="session-start:chain-1:1:disk-full",
        runtime="claude",
        chain_id="chain-1",
        conversation_id="conversation-1",
        generation=1,
        event_type="session.started",
        timestamp=NOW,
        cwd=tmp_path,
        owner_pid=1234,
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        command,
        "write_runtime_abort",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(command.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(
        command.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(RuntimeError("still blocked")),
    )

    with pytest.raises(RuntimeError, match="still blocked"):
        command._abort_unacknowledged_successor(
            event=event,
            environment=_stop_environment(
                tmp_path, tmp_path / "state.json", generation=1
            ),
        )

    assert signals == [(1234, command.signal.SIGUSR1)]


def test_successor_user_prompt_is_blocked_until_durable_ack(tmp_path: Path) -> None:
    _native, command = _modules()
    state_path = tmp_path / "supervisor.json"
    _write_supervisor_state(
        state_path,
        runtime="claude",
        phase="awaiting_ack",
        generation=1,
    )
    environ = _stop_environment(tmp_path, state_path, generation=1)
    environ["AGENT_SESSION_HARNESS_CAPSULE_PATH"] = str(tmp_path / "capsule.json")
    environ["AGENT_SESSION_HARNESS_CAPSULE_FINGERPRINT"] = "f" * 64
    environ["AGENT_SESSION_HARNESS_TARGET_GENERATION"] = "1"
    stdout = io.StringIO()

    exit_code = command.run_hook(
        runtime="claude",
        stdin=io.StringIO(
            json.dumps(
                {
                    "hook_event_name": "UserPromptSubmit",
                    "session_id": "conversation-1",
                    "cwd": str(tmp_path),
                    "timestamp": NOW.isoformat(),
                    "prompt": "must not persist",
                }
            )
        ),
        stdout=stdout,
        environ=environ,
    )

    assert exit_code == 0
    assert json.loads(stdout.getvalue())["decision"] == "block"
    assert not (tmp_path / "events.jsonl").exists()


def test_hook_command_rejects_oversized_input(tmp_path) -> None:
    _native, command = _modules()
    with pytest.raises(ValueError, match="large"):
        command.run_hook(
            runtime="codex",
            stdin=io.StringIO("x" * 1_048_577),
            stdout=io.StringIO(),
            environ={"AGENT_SESSION_HARNESS_MANAGED": "1"},
        )
