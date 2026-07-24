from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from agent_session_harness import cli
from agent_session_harness.adapters import beads
from agent_session_harness.adapters.command import sanitize_error

FIXTURES = Path(__file__).parent / "fixtures" / "adapters"
SUCCESS = {
    "ok": True,
    "fingerprint": "a" * 64,
    "retryable": False,
    "error": None,
}

_FAKE_BD = """\
import json, os, pathlib, sys
state = pathlib.Path(os.environ['FAKE_BD_STATE'])
log = pathlib.Path(os.environ['FAKE_BD_LOG'])
if os.environ.get('BD_NO_DAEMON') != '1': raise SystemExit(3)
with log.open('a') as handle: handle.write(json.dumps(sys.argv[1:]) + '\\n')
if os.environ.get('FAKE_BD_LOCK') == '1':
    print('database is locked: credential=do-not-leak', file=sys.stderr)
    raise SystemExit(1)
record = json.loads(state.read_text())
if sys.argv[1] == 'show':
    print(json.dumps([record]))
elif sys.argv[1] == 'update':
    note = sys.argv[sys.argv.index('--append-notes') + 1]
    if os.environ.get('FAKE_BD_DROP_UPDATE') != '1':
        record['notes'] = record.get('notes', '') + '\\n' + note
        state.write_text(json.dumps(record) + '\\n')
    print(json.dumps([record]))
else:
    raise SystemExit(2)
"""


def _request(operation: str = "write") -> dict[str, object]:
    payload = json.loads(
        (FIXTURES / "checkpoint-request-v1.json").read_text(encoding="utf-8")
    )
    payload["operation"] = operation
    return payload


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    executable = tmp_path / "fake_bd.py"
    state = tmp_path / "bead.json"
    log = tmp_path / "argv.jsonl"
    state.write_text(
        json.dumps({"id": "bead-1", "notes": "existing note"}) + "\n",
        encoding="utf-8",
    )
    executable.write_text(_FAKE_BD, encoding="utf-8")
    monkeypatch.setenv("FAKE_BD_STATE", str(state))
    monkeypatch.setenv("FAKE_BD_LOG", str(log))
    client = beads.BdClient(
        argv=(sys.executable, str(executable)),
        cwd=tmp_path,
        timeout_seconds=5,
    )
    return client, state, log


def test_write_read_and_acknowledge_are_idempotent(tmp_path, monkeypatch) -> None:
    client, state, log = _client(tmp_path, monkeypatch)

    first = beads.handle_request(_request("write"), client)
    repeated = beads.handle_request(_request("write"), client)
    read_back = beads.handle_request(_request("read"), client)
    acknowledged = beads.handle_request(_request("acknowledge"), client)
    repeated_ack = beads.handle_request(_request("acknowledge"), client)

    assert first == repeated == read_back == SUCCESS
    assert acknowledged == repeated_ack == SUCCESS
    notes = json.loads(state.read_text(encoding="utf-8"))["notes"]
    assert notes.count("agent-session-harness checkpoint") == 1
    assert notes.count("agent-session-harness acknowledgement") == 1
    assert "Run the adapter contract tests." in notes
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    updates = [call for call in calls if call[0] == "update"]
    assert len(updates) == 2
    assert all("--append-notes" in call and "--json" in call for call in updates)
    assert all("--no-daemon" not in call for call in calls)


def test_requires_a_bounded_bead_id(tmp_path, monkeypatch) -> None:
    client, _state, _log = _client(tmp_path, monkeypatch)
    request = _request()
    request["capsule"]["task_ids"].pop("bead")

    response = beads.handle_request(request, client)

    assert response["ok"] is False
    assert response["retryable"] is False
    assert "bead" in response["error"]


def test_lock_contention_is_retryable_and_sanitized(tmp_path, monkeypatch) -> None:
    client, _state, _log = _client(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_BD_LOCK", "1")

    response = beads.handle_request(_request(), client)

    assert response["ok"] is False
    assert response["retryable"] is True
    assert "do-not-leak" not in response["error"]


def test_rejects_oversized_command_output_without_buffering_it(
    tmp_path, monkeypatch
) -> None:
    client, state, _log = _client(tmp_path, monkeypatch)
    state.write_text(
        json.dumps({"id": "bead-1", "notes": "x" * 1_024}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(beads, "MAX_BD_OUTPUT_BYTES", 128)

    response = beads.handle_request(_request(), client)

    assert response["ok"] is False
    assert response["retryable"] is False
    assert response["error"] == "bd command output exceeded limit"


def test_write_requires_exact_read_back(tmp_path, monkeypatch) -> None:
    client, _state, _log = _client(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_BD_DROP_UPDATE", "1")

    response = beads.handle_request(_request(), client)

    assert response["ok"] is False
    assert response["retryable"] is True
    assert response["fingerprint"] is None


@pytest.mark.parametrize(
    "notes",
    (
        "agent-session-harness checkpoint\nchain: chain-1\n",
        (
            "agent-session-harness checkpoint\nchain: chain-1\n"
            "generation: 1\nfingerprint: "
            + ("a" * 64)
            + "\nidempotency: chain-1:1\nobjective: wrong\n"
            "handoff-action: wrong\nhead: deadbeef\n\n```json\n{}\n```"
        ),
    ),
)
def test_read_rejects_marker_only_or_tampered_capsule(
    tmp_path, monkeypatch, notes
) -> None:
    client, state, _log = _client(tmp_path, monkeypatch)
    state.write_text(
        json.dumps({"id": "bead-1", "notes": notes}) + "\n",
        encoding="utf-8",
    )

    response = beads.handle_request(_request("read"), client)

    assert response["ok"] is False
    assert response["retryable"] is True
    assert response["fingerprint"] is None


def test_rejects_malformed_request_without_running_bd(tmp_path, monkeypatch) -> None:
    client, _state, log = _client(tmp_path, monkeypatch)

    response = beads.handle_request({"schema_version": 1}, client)

    assert response["ok"] is False
    assert response["retryable"] is False
    assert not log.exists()


def test_diagnostics_use_the_shared_credential_redaction() -> None:
    assert beads.sanitize_error is sanitize_error

    redacted = sanitize_error("transport failed with AWS_SECRET_ACCESS_KEY=do-not-leak")

    assert "do-not-leak" not in redacted
    assert "credential=[redacted]" in redacted


def test_cli_reports_a_bounded_failure_for_an_invalid_request(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"not json")))

    assert cli.main(["adapter", "beads"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["retryable"] is False
