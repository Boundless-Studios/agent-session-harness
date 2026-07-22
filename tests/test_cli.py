from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
import sys

import pytest

from agent_session_harness import cli
from agent_session_harness.activity import Quiescence
from agent_session_harness.capsule import HandoffCapsule
from agent_session_harness.events import LifecycleEvent
from agent_session_harness.hooks.install import HookInstaller
from agent_session_harness.ledger import EventLedger
from agent_session_harness.models import EventType
from agent_session_harness.outbox import MirrorOutbox
from agent_session_harness.supervisor import SupervisorPhase, SupervisorSnapshot


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


def test_doctor_reports_managed_policy_only_when_capabilities_are_known(
    tmp_path: Path, capsys
) -> None:
    config = tmp_path / ".agent-session-harness.toml"
    config.write_text("observe_only = false\n", encoding="utf-8")

    assert cli.main(["doctor", "--config", str(config), "--json"]) == 0
    unknown = _json_stdout(capsys)
    assert unknown["checks"]["observe_only"] is True

    assert (
        cli.main(
            [
                "doctor",
                "--config",
                str(config),
                "--required-capabilities-known",
                "--json",
            ]
        )
        == 0
    )
    known = _json_stdout(capsys)
    assert known["checks"]["observe_only"] is False


def _interval_args(
    *,
    adapter_timeout_seconds: float = 5.0,
    lease_seconds: int = 60,
    poll_seconds: float = 1.0,
) -> argparse.Namespace:
    return argparse.Namespace(
        poll_seconds=poll_seconds,
        lease_seconds=lease_seconds,
        stop_timeout_seconds=10.0,
        stale_after_seconds=None,
        max_ticks=0,
        adapter_timeout_seconds=adapter_timeout_seconds,
        heartbeat_interval_seconds=None,
        successor_retry_limit=1,
    )


def test_supervise_rejects_cumulative_checkpoint_adapter_budget() -> None:
    with pytest.raises(ValueError, match="checkpoint adapter budget"):
        cli._validate_supervise_intervals(
            _interval_args(),
            required_adapter_count=6,
            mirror_adapter_count=0,
        )


def test_supervise_rejects_cumulative_acknowledgement_adapter_budget() -> None:
    with pytest.raises(ValueError, match="acknowledgement adapter budget"):
        cli._validate_supervise_intervals(
            _interval_args(adapter_timeout_seconds=20.0, lease_seconds=120),
            required_adapter_count=1,
            mirror_adapter_count=1,
        )


def test_supervise_accepts_bounded_multi_adapter_budget() -> None:
    cli._validate_supervise_intervals(
        _interval_args(adapter_timeout_seconds=10.0),
        required_adapter_count=1,
        mirror_adapter_count=1,
    )


def test_supervise_acknowledgement_budget_includes_successor_readiness() -> None:
    with pytest.raises(ValueError, match="acknowledgement adapter budget"):
        cli._validate_supervise_intervals(
            _interval_args(adapter_timeout_seconds=11.0),
            required_adapter_count=1,
            mirror_adapter_count=1,
        )


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


def test_inspect_without_window_override_uses_rollout_model(
    tmp_path: Path, capsys
) -> None:
    rollout = tmp_path / "long-context.jsonl"
    rollout.write_text(
        json.dumps(
            {
                "type": "assistant",
                "sessionId": "long-context",
                "timestamp": "2026-07-22T00:00:00Z",
                "message": {
                    "id": "msg-long-context",
                    "model": "claude-opus-4-8[1m]",
                    "usage": {
                        "input_tokens": 100_000,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        cli.main(
            [
                "inspect",
                "--runtime",
                "claude",
                "--path",
                str(rollout),
                "--json",
            ]
        )
        == 0
    )

    payload = _json_stdout(capsys)
    assert payload["window_tokens"] == 1_000_000
    assert payload["context_percent"] == 10.0


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
        "context_tokens": None,
        "context_percent": 70.0,
        "conversation_id": None,
        "cumulative_tokens": None,
        "generation": 1,
        "liveness_alarm": None,
        "outbox_depth": 0,
        "quiescence": "unknown",
        # BOU-2236: no ledger was supplied, so nothing could be reconciled.
        "reaped_tools": 0,
        "runtime": "codex",
        # No ledger was supplied, so no hook has ever been seen to report
        # (BOU-2222): "quiescence unknown" alone never said that out loud.
        "runtime_liveness": "never_reported",
        "schema_version": 1,
        "state": "awaiting_ack",
        "usage_alarm": None,
        "window_tokens": None,
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
        == 1
    )
    assert _json_stdout(capsys)["installed"] is False


def test_acknowledge_writes_bounded_durable_record(tmp_path, capsys) -> None:
    state_path = tmp_path / "supervisor.json"
    capsule = HandoffCapsule(
        schema_version=1,
        chain_id="chain-1",
        predecessor_conversation_id="conversation-0",
        target_generation=1,
        task_ids={"linear": "BOU-2195"},
        objective="Continue the durable handoff test.",
        exact_next_action="Acknowledge the capsule.",
        completed_criteria=("checkpoint written",),
        remaining_criteria=("successor acknowledged",),
        repository_path=tmp_path,
        branch="test-branch",
        head="deadbeef",
        dirty_paths=(),
        file_anchors=("tests/test_cli.py",),
        symbol_anchors=("test_acknowledge_writes_bounded_durable_record",),
        test_results={"focused": "running"},
        decisions=("bind acknowledgement to the child",),
        blockers=(),
        process_summaries={"pytest": "running"},
        created_at=datetime(2026, 7, 19, 7, 0, tzinfo=timezone.utc),
    )
    capsule_path = tmp_path / "capsule.json"
    capsule_path.write_bytes(capsule.canonical_bytes() + b"\n")
    snapshot = SupervisorSnapshot(
        runtime="claude",
        chain_id="chain-1",
        generation=1,
        phase="awaiting_ack",
        process_pid=os.getpid(),
        checkpoint_fingerprint=capsule.fingerprint,
        checkpoint_path=capsule_path,
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
                capsule.fingerprint,
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
    assert record["fingerprint"] == capsule.fingerprint
    assert record["owner_pid"] == os.getpid()
    assert stat.S_IMODE(acknowledgement_path.stat().st_mode) == 0o600


def test_acknowledge_forwards_an_explicit_owner_pid(
    tmp_path, monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_write_acknowledgement(**kwargs):
        captured.update(kwargs)
        return tmp_path / "ack.json"

    monkeypatch.setattr(cli, "write_acknowledgement", fake_write_acknowledgement)

    assert (
        cli.main(
            [
                "acknowledge",
                "--state",
                str(tmp_path / "supervisor.json"),
                "--generation",
                "1",
                "--fingerprint",
                "f" * 64,
                "--conversation-id",
                "conversation-1",
                "--owner-pid",
                "4321",
                "--json",
            ]
        )
        == 0
    )

    assert captured["owner_pid"] == 4321
    assert _json_stdout(capsys)["ok"] is True


def test_acknowledge_uses_managed_owner_pid_environment(
    tmp_path, monkeypatch, capsys
) -> None:
    captured: dict[str, object] = {}

    def fake_write_acknowledgement(**kwargs):
        captured.update(kwargs)
        return tmp_path / "ack.json"

    monkeypatch.setattr(cli, "write_acknowledgement", fake_write_acknowledgement)
    monkeypatch.setenv("AGENT_SESSION_HARNESS_OWNER_PID", "2468")

    assert (
        cli.main(
            [
                "acknowledge",
                "--state",
                str(tmp_path / "supervisor.json"),
                "--generation",
                "1",
                "--fingerprint",
                "f" * 64,
                "--conversation-id",
                "conversation-1",
                "--json",
            ]
        )
        == 0
    )

    assert captured["owner_pid"] == 2468
    assert _json_stdout(capsys)["ok"] is True


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


def test_runtime_environment_rejects_harness_control_keys(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_SESSION_HARNESS_STATE_PATH", "/tmp/attacker-state")

    with pytest.raises(ValueError, match="reserved"):
        cli._runtime_environment(["AGENT_SESSION_HARNESS_STATE_PATH"])


def test_automatic_mirror_replay_is_one_attempt_and_fail_open() -> None:
    calls: list[int] = []

    class BrokenManager:
        def replay_mirrors(self, *, max_attempts):
            calls.append(max_attempts)
            raise ValueError("queue exceeds 64 bytes")

    error = cli._replay_mirrors_fail_open(BrokenManager())

    assert calls == [1]
    assert error == "queue exceeds 64 bytes"


def test_doctor_checks_versions_logs_permissions_hooks_and_adapter_contracts(
    tmp_path, monkeypatch, capsys
) -> None:
    runtime_args = tmp_path / "runtime-args"
    runtime = tmp_path / "codex"
    runtime.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {json.dumps(str(runtime_args))}\n"
        "printf 'codex 9.9.0\\n'\n",
        encoding="utf-8",
    )
    runtime.chmod(0o700)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

    log_root = tmp_path / "logs"
    log_root.mkdir()
    (log_root / "session.jsonl").write_text("{}\n", encoding="utf-8")
    state_path = tmp_path / "supervisor.json"
    state_path.write_text("{}\n", encoding="utf-8")
    state_path.chmod(0o600)
    manifest = tmp_path / "hooks.json"
    manifest.write_text('{"hooks":{}}\n', encoding="utf-8")
    HookInstaller(runtime="codex", path=manifest).install()

    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)\n"
        "print(json.dumps({'ok': True, "
        "'fingerprint': request['capsule']['fingerprint'], "
        "'retryable': False, 'error': None}))\n",
        encoding="utf-8",
    )
    adapter_spec = "local=" + json.dumps([sys.executable, str(adapter)])

    assert (
        cli.main(
            [
                "doctor",
                "--runtime",
                "codex",
                "--log-root",
                str(log_root),
                "--state-path",
                str(state_path),
                "--hook-manifest",
                str(manifest),
                "--adapter",
                adapter_spec,
                "--json",
            ]
        )
        == 0
    )

    payload = _json_stdout(capsys)
    checks = payload["checks"]
    assert checks["runtime"]["version"] == "codex 9.9.0"
    assert checks["logs"]["jsonl_count"] == 1
    assert checks["state"]["restrictive"] is True
    assert checks["hook_manifest"]["installed"] is True
    assert checks["adapters"]["local"]["contract"] is True
    assert "version" in checks["coordinator"]
    assert runtime_args.read_text(encoding="utf-8") == "--version\n"


def test_doctor_accepts_stateful_read_miss_and_routes_explicit_adapter_env(
    tmp_path, monkeypatch, capsys
) -> None:
    observed = tmp_path / "observed-token"
    adapter = tmp_path / "stateful_adapter.py"
    adapter.write_text(
        "import json, os, pathlib, sys\n"
        "json.load(sys.stdin)\n"
        "pathlib.Path(sys.argv[1]).write_text("
        "os.environ.get('ADAPTER_CONTRACT_TOKEN', 'missing'))\n"
        "print(json.dumps({'ok': False, 'fingerprint': None, "
        "'retryable': True, 'error': 'checkpoint not found'}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ADAPTER_CONTRACT_TOKEN", "explicit-token")
    adapter_spec = "stateful=" + json.dumps(
        [sys.executable, str(adapter), str(observed)]
    )

    assert (
        cli.main(
            [
                "doctor",
                "--adapter",
                adapter_spec,
                "--adapter-env",
                "stateful=ADAPTER_CONTRACT_TOKEN",
                "--json",
            ]
        )
        == 0
    )

    payload = _json_stdout(capsys)
    result = payload["checks"]["adapters"]["stateful"]
    assert result["contract"] is True
    assert result["readback"] is False
    assert observed.read_text(encoding="utf-8") == "explicit-token"


def test_doctor_rejects_an_existing_group_readable_state_file(tmp_path, capsys) -> None:
    state_path = tmp_path / "supervisor.json"
    state_path.write_text("{}\n", encoding="utf-8")
    state_path.chmod(0o640)

    assert cli.main(["doctor", "--state-path", str(state_path), "--json"]) == 1

    payload = _json_stdout(capsys)
    assert payload["checks"]["state"]["restrictive"] is False


def test_human_diagnostics_use_stderr_while_json_uses_stdout(capsys) -> None:
    assert cli.main(["doctor"]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "schema_version" in captured.err


def test_non_check_supervise_runs_a_real_bounded_supervisor_loop(
    tmp_path, capsys
) -> None:
    marker = tmp_path / "runtime-started"
    runtime = tmp_path / "runtime.py"
    runtime.write_text(
        "import pathlib, signal, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text('started')\n"
        "signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))\n"
        "while True: time.sleep(0.01)\n",
        encoding="utf-8",
    )
    usage = tmp_path / "usage.py"
    usage.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'conversation_id': 'conversation-0', "
        "'context_percent': 10.0, 'confidence': 'confident'}))\n",
        encoding="utf-8",
    )
    capsule = tmp_path / "capsule.py"
    capsule.write_text("print('{}')\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.py"
    checkpoint.write_text("print('{}')\n", encoding="utf-8")
    safety_request = tmp_path / "safety-request.json"
    safety = tmp_path / "safety.py"
    safety.write_text(
        "import json, pathlib, sys\n"
        "request = json.load(sys.stdin)\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(request))\n"
        "print(json.dumps({'schema_version': 1, 'status': 'quiescent', "
        "'active_critical_sections': [], 'warnings': []}))\n",
        encoding="utf-8",
    )

    arguments = [
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
        "--executable",
        sys.executable,
        "--runtime-arg",
        str(runtime),
        "--runtime-arg",
        str(marker),
        "--usage-adapter",
        json.dumps([sys.executable, str(usage)]),
        "--capsule-adapter",
        json.dumps([sys.executable, str(capsule)]),
        "--safety-adapter",
        json.dumps([sys.executable, str(safety), str(safety_request)]),
        "--required-adapter",
        "local=" + json.dumps([sys.executable, str(checkpoint)]),
        "--coordinator-store",
        str(tmp_path / "claims.jsonl"),
        "--outbox",
        str(tmp_path / "outbox.jsonl"),
        "--poll-seconds",
        "0.05",
        "--max-ticks",
        "1",
        "--required-capabilities-known",
        "--json",
    ]

    assert cli.main(arguments) == 0

    payload = _json_stdout(capsys)
    assert payload["mode"] == "managed"
    assert payload["ticks"] == 1
    assert payload["state"] == "blocked"
    assert payload["generation"] == 0
    assert payload["process_pid"] is None
    persisted = SupervisorSnapshot.model_validate_json(
        (tmp_path / "supervisor.json").read_text(encoding="utf-8")
    )
    assert persisted.phase.value == "blocked"
    assert persisted.claim is None
    assert persisted.process_pid is None
    assert marker.read_text() == "started"
    observed_safety = json.loads(safety_request.read_text(encoding="utf-8"))
    assert observed_safety == {
        "schema_version": 1,
        "operation": "probe",
        "cwd": str(tmp_path),
        "chain_id": "chain-1",
        "generation": 0,
        "process_group_id": observed_safety["process_group_id"],
    }
    assert isinstance(observed_safety["process_group_id"], int)
    assert observed_safety["process_group_id"] > 0

    assert cli.main(arguments) == 2
    blocked = _json_stdout(capsys)
    assert blocked["ok"] is False
    assert blocked["error"]["type"] == "RuntimeError"
    assert "blocked" in blocked["error"]["message"]


def test_supervise_returns_success_when_runtime_exits_cleanly(
    tmp_path, capsys, monkeypatch
) -> None:
    environment_marker = tmp_path / "runtime-environment.json"
    runtime = tmp_path / "runtime.py"
    runtime.write_text(
        "import json, os, pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({"
        "'docker_host': os.environ.get('DOCKER_HOST'), "
        "'unrelated': os.environ.get('UNRELATED_PRIVATE_VALUE')}))\n"
        "time.sleep(0.2)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCKER_HOST", "ssh://docker.example")
    monkeypatch.setenv("UNRELATED_PRIVATE_VALUE", "must-not-leak")
    usage = tmp_path / "usage.py"
    usage.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'conversation_id': 'conversation-0', "
        "'context_percent': 10.0, 'confidence': 'confident'}))\n",
        encoding="utf-8",
    )
    inert = json.dumps([sys.executable, "-c", "print('{}')"])
    state_path = tmp_path / "supervisor.json"

    assert (
        cli.main(
            [
                "supervise",
                "--runtime",
                "codex",
                "--cwd",
                str(tmp_path),
                "--chain-id",
                "chain-clean-exit",
                "--task-type",
                "linear",
                "--task-id",
                "BOU-2195",
                "--task-fingerprint",
                "task-fingerprint",
                "--state",
                str(state_path),
                "--executable",
                sys.executable,
                "--runtime-arg",
                str(runtime),
                "--runtime-arg",
                str(environment_marker),
                "--runtime-env",
                "DOCKER_HOST",
                "--usage-adapter",
                json.dumps([sys.executable, str(usage)]),
                "--capsule-adapter",
                inert,
                "--required-adapter",
                "local=" + inert,
                "--poll-seconds",
                "0.05",
                "--max-ticks",
                "10",
                "--required-capabilities-known",
                "--json",
            ]
        )
        == 0
    )

    payload = _json_stdout(capsys)
    assert payload["state"] == "completed"
    assert payload["process_pid"] is None
    persisted = SupervisorSnapshot.model_validate_json(
        state_path.read_text(encoding="utf-8")
    )
    assert persisted.phase is SupervisorPhase.COMPLETED
    assert persisted.claim is None
    assert json.loads(environment_marker.read_text(encoding="utf-8")) == {
        "docker_host": "ssh://docker.example",
        "unrelated": None,
    }
    durable_files = "".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in tmp_path.rglob("*")
        if path.is_file() and path != environment_marker
    )
    assert "ssh://docker.example" not in durable_files
    assert "must-not-leak" not in durable_files


def test_supervise_merges_busy_project_safety_before_tick(
    tmp_path, capsys, monkeypatch
) -> None:
    state_path = tmp_path / "supervisor.json"
    lifecycle_path = state_path.with_suffix(state_path.suffix + ".lifecycle")
    EventLedger(lifecycle_path).append(
        LifecycleEvent(
            schema_version=1,
            event_id="session-started",
            runtime="codex",
            chain_id="chain-safety",
            conversation_id="conversation-0",
            generation=0,
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(tz=timezone.utc),
            cwd=tmp_path,
            owner_pid=os.getpid(),
        )
    )
    safety = tmp_path / "safety.py"
    safety.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'schema_version': 1, 'status': 'busy', "
        "'active_critical_sections': ['git-index-lock'], 'warnings': []}))\n",
        encoding="utf-8",
    )
    instances = []

    class FakeSupervisor:
        def __init__(self, **kwargs) -> None:
            self.chain_id = kwargs["chain_id"]
            self.lifecycle_path = lifecycle_path
            self.snapshot = SupervisorSnapshot(
                runtime="codex",
                chain_id=self.chain_id,
                phase="running",
                run_spec_fingerprint="a" * 64,
                process_group_id=4242,
            )
            self.activity = None
            instances.append(self)

        def start(self):
            return self.snapshot

        def tick(self, activity) -> None:
            self.activity = activity
            return self.snapshot

        def shutdown(self) -> None:
            self.snapshot = self.snapshot.model_copy(
                update={"phase": SupervisorPhase.BLOCKED}
            )

    monkeypatch.setattr(cli, "Supervisor", FakeSupervisor)
    inert = json.dumps([sys.executable, "-c", "pass"])

    assert (
        cli.main(
            [
                "supervise",
                "--runtime",
                "codex",
                "--cwd",
                str(tmp_path),
                "--chain-id",
                "chain-safety",
                "--task-type",
                "linear",
                "--task-id",
                "BOU-2195",
                "--task-fingerprint",
                "fingerprint",
                "--state",
                str(state_path),
                "--executable",
                sys.executable,
                "--usage-adapter",
                inert,
                "--capsule-adapter",
                inert,
                "--safety-adapter",
                json.dumps([sys.executable, str(safety)]),
                "--required-adapter",
                "local=" + inert,
                "--poll-seconds",
                "0.001",
                "--max-ticks",
                "1",
                "--required-capabilities-known",
                "--json",
            ]
        )
        == 0
    )

    _json_stdout(capsys)
    assert instances[0].activity.quiescence is Quiescence.BUSY
    assert instances[0].activity.active_critical_section_ids == frozenset(
        {"git-index-lock"}
    )


def test_supervise_prioritizes_successor_ack_before_sleep_safety_or_replay(
    tmp_path, capsys, monkeypatch
) -> None:
    state_path = tmp_path / "supervisor.json"
    lifecycle_path = state_path.with_suffix(state_path.suffix + ".lifecycle")
    replay_calls: list[object] = []
    sleep_calls: list[float] = []

    class FakeSupervisor:
        def __init__(self, **kwargs) -> None:
            self.chain_id = kwargs["chain_id"]
            self.lifecycle_path = lifecycle_path
            self.snapshot = SupervisorSnapshot(
                runtime="claude",
                chain_id=self.chain_id,
                generation=1,
                phase="awaiting_ack",
                run_spec_fingerprint="a" * 64,
            )

        def start(self):
            return self.snapshot

        def tick(self, _activity):
            return self.snapshot

        def shutdown(self) -> None:
            self.snapshot = self.snapshot.model_copy(
                update={"phase": SupervisorPhase.BLOCKED}
            )

    monkeypatch.setattr(cli, "Supervisor", FakeSupervisor)
    monkeypatch.setattr(
        cli,
        "_replay_mirrors_fail_open",
        lambda manager: replay_calls.append(manager),
    )
    monkeypatch.setattr(cli.time, "sleep", sleep_calls.append)
    inert = json.dumps([sys.executable, "-c", "pass"])

    assert (
        cli.main(
            [
                "supervise",
                "--runtime",
                "claude",
                "--cwd",
                str(tmp_path),
                "--chain-id",
                "chain-ack-priority",
                "--task-type",
                "linear",
                "--task-id",
                "BOU-2195",
                "--task-fingerprint",
                "fingerprint",
                "--state",
                str(state_path),
                "--executable",
                sys.executable,
                "--usage-adapter",
                inert,
                "--capsule-adapter",
                inert,
                "--safety-adapter",
                inert,
                "--required-adapter",
                "local=" + inert,
                "--poll-seconds",
                "10",
                "--max-ticks",
                "1",
                "--required-capabilities-known",
                "--json",
            ]
        )
        == 0
    )

    _json_stdout(capsys)
    assert len(replay_calls) == 1
    assert 10.0 not in sleep_calls


def test_json_mode_emits_a_stable_error_object(tmp_path, capsys) -> None:
    assert (
        cli.main(
            [
                "report",
                "--state",
                str(tmp_path / "missing-state.json"),
                "--json",
            ]
        )
        == 2
    )

    payload = _json_stdout(capsys)
    assert payload["schema_version"] == 1
    assert payload["ok"] is False
    assert payload["error"]["type"] == "FileNotFoundError"
    assert len(payload["error"]["message"]) <= 500


def test_supervise_routes_capsule_required_and_mirror_adapters(
    tmp_path, capsys
) -> None:
    runtime = tmp_path / "runtime.py"
    runtime.write_text(
        "import signal, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))\n"
        "while True: time.sleep(0.01)\n",
        encoding="utf-8",
    )
    usage = tmp_path / "usage.py"
    usage.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'conversation_id': 'conversation-0', "
        "'context_percent': 75.0, 'confidence': 'confident'}))\n",
        encoding="utf-8",
    )
    capsule = tmp_path / "capsule.py"
    capsule.write_text(
        "import json, sys\n"
        "request = json.load(sys.stdin)['checkpoint']\n"
        "print(json.dumps({'capsule': {"
        "'schema_version': 1, 'chain_id': request['chain_id'], "
        "'predecessor_conversation_id': request['predecessor_conversation_id'], "
        "'target_generation': request['target_generation'], "
        "'task_ids': {'linear': 'BOU-2195'}, 'objective': 'Continue.', "
        "'exact_next_action': 'Resume the task.', 'completed_criteria': [], "
        "'remaining_criteria': ['finish'], 'repository_path': sys.argv[1], "
        "'branch': 'test', 'head': 'deadbeef', 'dirty_paths': [], "
        "'file_anchors': [], 'symbol_anchors': [], "
        "'test_results': {'focused': 'running'}, 'decisions': [], "
        "'blockers': [], 'process_summaries': {}, "
        "'created_at': '2026-07-19T07:00:00Z'}}))\n",
        encoding="utf-8",
    )
    required_state = tmp_path / "required.json"
    required = tmp_path / "required.py"
    required.write_text(
        "import json, pathlib, sys\n"
        "request = json.load(sys.stdin)\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "if request['operation'] == 'write': "
        "path.write_text(json.dumps(request['capsule']))\n"
        "stored = json.loads(path.read_text())\n"
        "print(json.dumps({'ok': True, 'fingerprint': stored['fingerprint'], "
        "'retryable': False, 'error': None}))\n",
        encoding="utf-8",
    )
    mirror = tmp_path / "mirror.py"
    mirror.write_text(
        "import json, sys\n"
        "json.load(sys.stdin)\n"
        "print(json.dumps({'ok': False, 'fingerprint': None, "
        "'retryable': True, 'error': 'offline'}))\n",
        encoding="utf-8",
    )
    state_path = tmp_path / "supervisor.json"
    lifecycle_path = state_path.with_suffix(state_path.suffix + ".lifecycle")
    EventLedger(lifecycle_path).append(
        LifecycleEvent(
            schema_version=1,
            event_id="session-started",
            runtime="codex",
            chain_id="chain-1",
            conversation_id="conversation-0",
            generation=0,
            event_type=EventType.SESSION_STARTED,
            timestamp=datetime.now(tz=timezone.utc),
            cwd=tmp_path,
            owner_pid=os.getpid(),
        )
    )
    EventLedger(lifecycle_path).append(
        LifecycleEvent(
            schema_version=1,
            event_id="handoff-requested",
            runtime="codex",
            chain_id="chain-1",
            conversation_id="conversation-0",
            generation=0,
            event_type=EventType.HANDOFF_REQUESTED,
            timestamp=datetime.now(tz=timezone.utc),
            cwd=tmp_path,
            owner_pid=os.getpid(),
        )
    )
    outbox_path = tmp_path / "outbox.jsonl"

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
                str(state_path),
                "--executable",
                sys.executable,
                "--runtime-arg",
                str(runtime),
                "--usage-adapter",
                json.dumps([sys.executable, str(usage)]),
                "--capsule-adapter",
                json.dumps([sys.executable, str(capsule), str(tmp_path)]),
                "--required-adapter",
                "local="
                + json.dumps([sys.executable, str(required), str(required_state)]),
                "--mirror-adapter",
                "remote=" + json.dumps([sys.executable, str(mirror)]),
                "--coordinator-store",
                str(tmp_path / "claims.jsonl"),
                "--outbox",
                str(outbox_path),
                "--poll-seconds",
                "0.05",
                "--max-ticks",
                "1",
                "--required-capabilities-known",
                "--json",
            ]
        )
        == 0
    )

    payload = _json_stdout(capsys)
    assert payload["generation"] == 1
    assert payload["state"] == "blocked"
    assert required_state.is_file()
    capsule_paths = list((tmp_path / "capsules").glob("*.json"))
    assert len(capsule_paths) == 1
    assert stat.S_IMODE(capsule_paths[0].stat().st_mode) == 0o600
    assert MirrorOutbox(outbox_path).depth == 1
