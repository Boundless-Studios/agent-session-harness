from __future__ import annotations

import json
from pathlib import Path
import stat

from agent_session_harness import cli
from agent_session_harness.supervisor import SupervisorSnapshot


FIXTURES = Path(__file__).parent / "fixtures"


def _json_stdout(capsys) -> dict[str, object]:
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out)


def test_doctor_json_is_deterministic_and_never_launches_a_model(capsys) -> None:
    assert cli.main(["doctor", "--json"]) == 0

    payload = _json_stdout(capsys)
    assert payload["schema_version"] == 1
    assert payload["ok"] is True
    assert payload["checks"]["package"] == "0.1.0"
    assert "summary" not in payload


def test_inspect_reads_native_usage_as_stable_json(capsys) -> None:
    assert (
        cli.main(
            [
                "inspect",
                "--runtime",
                "claude",
                "--path",
                str(FIXTURES / "claude" / "duplicate-message.jsonl"),
                "--window-tokens",
                "200000",
                "--json",
            ]
        )
        == 0
    )

    payload = _json_stdout(capsys)
    assert payload["runtime"] == "claude"
    assert payload["unique_messages"] == 1
    assert payload["context_percent"] == 62.5


def test_report_projects_supervisor_state_without_model_work(tmp_path, capsys) -> None:
    state_path = tmp_path / "supervisor.json"
    snapshot = SupervisorSnapshot(
        runtime="codex",
        chain_id="chain-1",
        generation=1,
        phase="awaiting_ack",
        owner_session_id="chain-1:1",
        context_percent=70.0,
        checkpoint_fingerprint="fingerprint",
        checkpoint_path=tmp_path / "capsule.json",
    )
    state_path.write_text(snapshot.model_dump_json(), encoding="utf-8")

    assert cli.main(["report", "--state", str(state_path), "--json"]) == 0

    payload = _json_stdout(capsys)
    assert payload == {
        "active": {
            "critical_sections": 0,
            "subagents": 0,
            "tools": 0,
            "turns": 0,
        },
        "chain_id": "chain-1",
        "checkpoint_fingerprint": "fingerprint",
        "confidence": "unknown",
        "context_percent": 70.0,
        "conversation_id": None,
        "generation": 1,
        "outbox_depth": 0,
        "quiescence": "unknown",
        "runtime": "codex",
        "schema_version": 1,
        "state": "awaiting_ack",
    }


def test_hook_and_hook_installer_commands_round_trip(
    tmp_path, monkeypatch, capsys
) -> None:
    manifest = tmp_path / "hooks.json"
    manifest.write_text('{"hooks":{}}\n', encoding="utf-8")
    monkeypatch.setenv("AGENT_SESSION_HARNESS_MANAGED", "1")
    monkeypatch.setenv("AGENT_SESSION_HARNESS_CHAIN_ID", "chain-1")
    monkeypatch.setenv("AGENT_SESSION_HARNESS_GENERATION", "0")
    monkeypatch.setenv("AGENT_SESSION_HARNESS_LEDGER", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("AGENT_SESSION_HARNESS_OWNER_PID", "1234")
    monkeypatch.setattr(
        "sys.stdin",
        __import__("io").StringIO(
            json.dumps(
                {
                    "hook_event_name": "SessionStart",
                    "session_id": "conversation-1",
                    "cwd": str(tmp_path),
                    "timestamp": "2026-07-19T07:00:00Z",
                }
            )
        ),
    )

    assert cli.main(["hook", "--runtime", "codex"]) == 0
    assert _json_stdout(capsys)["ok"] is True
    assert (
        cli.main(
            [
                "hooks",
                "install",
                "--runtime",
                "codex",
                "--path",
                str(manifest),
                "--json",
            ]
        )
        == 0
    )
    assert _json_stdout(capsys)["installed"] is True
    assert (
        cli.main(
            [
                "hooks",
                "check",
                "--runtime",
                "codex",
                "--path",
                str(manifest),
                "--json",
            ]
        )
        == 0
    )
    assert _json_stdout(capsys)["installed"] is True
    assert (
        cli.main(
            [
                "hooks",
                "uninstall",
                "--runtime",
                "codex",
                "--path",
                str(manifest),
                "--json",
            ]
        )
        == 0
    )
    assert _json_stdout(capsys)["installed"] is False


def test_acknowledge_writes_bounded_durable_record(tmp_path, capsys) -> None:
    state_path = tmp_path / "supervisor.json"
    snapshot = SupervisorSnapshot(
        runtime="claude",
        chain_id="chain-1",
        generation=1,
        phase="awaiting_ack",
        checkpoint_fingerprint="fingerprint",
        checkpoint_path=tmp_path / "capsule.json",
    )
    state_path.write_text(snapshot.model_dump_json(), encoding="utf-8")

    assert (
        cli.main(
            [
                "acknowledge",
                "--state",
                str(state_path),
                "--generation",
                "1",
                "--fingerprint",
                "fingerprint",
                "--conversation-id",
                "conversation-1",
                "--json",
            ]
        )
        == 0
    )

    payload = _json_stdout(capsys)
    acknowledgement_path = Path(payload["path"])
    record = json.loads(acknowledgement_path.read_text())
    assert record["generation"] == 1
    assert record["fingerprint"] == "fingerprint"
    assert stat.S_IMODE(acknowledgement_path.stat().st_mode) == 0o600


def test_empty_outbox_replay_and_supervise_preflight_are_safe(tmp_path, capsys) -> None:
    assert (
        cli.main(
            [
                "outbox",
                "replay",
                "--path",
                str(tmp_path / "outbox.jsonl"),
                "--json",
            ]
        )
        == 0
    )
    assert _json_stdout(capsys)["attempted"] == 0

    assert (
        cli.main(
            [
                "supervise",
                "--runtime",
                "codex",
                "--cwd",
                str(tmp_path),
                "--chain-id",
                "chain-1",
                "--task-type",
                "linear",
                "--task-id",
                "BOU-2195",
                "--task-fingerprint",
                "task-fingerprint",
                "--state",
                str(tmp_path / "supervisor.json"),
                "--check",
                "--required-capabilities-known",
                "--json",
            ]
        )
        == 0
    )
    preflight = _json_stdout(capsys)
    assert preflight["ready"] is True
    assert preflight["observe_only"] is False
