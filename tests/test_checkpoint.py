from __future__ import annotations

import importlib
import json
import os
import stat
import sys
import threading
from pathlib import Path
from typing import Callable

import pytest
from test_capsule import capsule_payload

FAKE_ADAPTER = r"""
import json
from pathlib import Path
import sys
import time

mode, state_name, log_name = sys.argv[1:4]
request = json.load(sys.stdin)
state_path = Path(state_name)
log_path = Path(log_name)
log_path.parent.mkdir(parents=True, exist_ok=True)
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n")

if set(request) != {"schema_version", "operation", "idempotency_key", "capsule"}:
    raise SystemExit(9)
if mode == "timeout":
    time.sleep(2)
if mode == "nonzero":
    print("sensitive=do-not-copy", file=sys.stderr)
    raise SystemExit(7)
if mode == "malformed":
    print("{broken")
    raise SystemExit(0)
if mode == "oversized":
    print(json.dumps({
        "ok": False,
        "fingerprint": None,
        "retryable": False,
        "error": "x" * 4096,
    }))
    raise SystemExit(0)
if mode == "stderr-flood":
    sys.stderr.write("x" * (2 * 1024 * 1024))
    sys.stderr.flush()
    time.sleep(2)

fingerprint = request["capsule"]["fingerprint"]
if request["operation"] == "write":
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(request["capsule"]), encoding="utf-8")
elif request["operation"] == "read":
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    fingerprint = stored["fingerprint"]
elif request["operation"] != "acknowledge":
    raise SystemExit(8)

if mode == "wrong-fingerprint":
    fingerprint = "0" * 64
print(json.dumps({
    "ok": True,
    "fingerprint": fingerprint,
    "retryable": False,
    "error": None,
}))
"""


def _modules():
    try:
        return (
            importlib.import_module("agent_session_harness.capsule"),
            importlib.import_module("agent_session_harness.adapters.command"),
            importlib.import_module("agent_session_harness.checkpoint"),
            importlib.import_module("agent_session_harness.outbox"),
        )
    except ModuleNotFoundError:
        pytest.fail("checkpoint adapters and orchestration are not implemented")


def _capsule(capsule_module, tmp_path: Path):
    return capsule_module.HandoffCapsule.model_validate(capsule_payload(tmp_path))


def _write_fake_adapter(tmp_path: Path) -> Path:
    script = tmp_path / "fake adapter.py"
    script.write_text(FAKE_ADAPTER, encoding="utf-8")
    return script


def _command_adapter(command, tmp_path: Path, mode: str = "success"):
    script = _write_fake_adapter(tmp_path)
    state_path = tmp_path / "state dir" / "capsule.json"
    log_path = tmp_path / "state dir" / "requests.jsonl"
    adapter = command.CommandAdapter(
        name="fake",
        argv=(sys.executable, str(script), mode, str(state_path), str(log_path)),
        timeout_seconds=0.05 if mode == "timeout" else 2.0,
        max_response_bytes=1024,
        max_stderr_bytes=1024,
    )
    return adapter, log_path


def _request(command, handoff, operation: str, key: str = "handoff-8"):
    return command.AdapterRequest(
        schema_version=1,
        operation=operation,
        idempotency_key=key,
        capsule=handoff,
    )


def test_command_adapter_writes_reads_and_acknowledges_by_argv(tmp_path) -> None:
    capsule, command, _checkpoint, _outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    adapter, log_path = _command_adapter(command, tmp_path)

    responses = [
        adapter.execute(_request(command, handoff, operation))
        for operation in ("write", "read", "acknowledge")
    ]

    assert all(response.ok for response in responses)
    assert [response.fingerprint for response in responses] == [
        handoff.fingerprint,
        handoff.fingerprint,
        handoff.fingerprint,
    ]
    requests = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [request["operation"] for request in requests] == [
        "write",
        "read",
        "acknowledge",
    ]
    assert all(
        set(request) == {"schema_version", "operation", "idempotency_key", "capsule"}
        for request in requests
    )


@pytest.mark.parametrize(
    "diagnostic",
    (
        "aws_secret_access_key=must-not-leak",
        "github_token_value=must-not-leak",
        "proxy_authorization=must-not-leak",
    ),
)
def test_adapter_diagnostics_redact_prefixed_and_suffixed_credentials(
    diagnostic: str,
) -> None:
    _capsule_module, command, _checkpoint, _outbox = _modules()

    assert command.sanitize_error(diagnostic) == "credential=[redacted]"


def test_json_command_inherits_only_controlled_adapter_environment(
    tmp_path, monkeypatch
) -> None:
    _capsule_module, command, _checkpoint, _outbox = _modules()
    script = tmp_path / "echo_env.py"
    script.write_text(
        """
import json
import os
import sys

json.load(sys.stdin)
print(json.dumps({
    "home": os.environ.get("HOME"),
    "path": os.environ.get("PATH"),
    "linear": os.environ.get("LINEAR_API_KEY"),
    "beads": os.environ.get("BEADS_DB"),
    "beads_secret": os.environ.get("BEADS_REMOTE_TOKEN"),
    "bd_secret": os.environ.get("BD_PASSWORD"),
    "xdg_secret": os.environ.get("XDG_PRIVATE_SECRET"),
    "unrelated": os.environ.get("UNRELATED_PRIVATE_SECRET"),
    "override": os.environ.get("HARNESS_EXPLICIT_OVERRIDE"),
}))
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path / "controlled-home"))
    monkeypatch.setenv("PATH", os.environ["PATH"])
    monkeypatch.setenv("LINEAR_API_KEY", "linear-test-key")
    monkeypatch.setenv("BEADS_DB", str(tmp_path / "beads.db"))
    monkeypatch.setenv("BEADS_REMOTE_TOKEN", "must-not-leak")
    monkeypatch.setenv("BD_PASSWORD", "must-not-leak")
    monkeypatch.setenv("XDG_PRIVATE_SECRET", "must-not-leak")
    monkeypatch.setenv("UNRELATED_PRIVATE_SECRET", "must-not-leak")

    response = command.JsonCommand(
        name="environment echo",
        argv=(sys.executable, str(script)),
        env={"HARNESS_EXPLICIT_OVERRIDE": "explicit-value"},
    ).execute({"schema_version": 1})

    assert response == {
        "home": str(tmp_path / "controlled-home"),
        "path": os.environ["PATH"],
        "linear": None,
        "beads": str(tmp_path / "beads.db"),
        "beads_secret": None,
        "bd_secret": None,
        "xdg_secret": None,
        "unrelated": None,
        "override": "explicit-value",
    }

    privileged = command.JsonCommand(
        name="environment echo with explicit inheritance",
        argv=(sys.executable, str(script)),
        inherit_env=("LINEAR_API_KEY",),
    ).execute({"schema_version": 1})
    assert privileged["linear"] == "linear-test-key"
    assert privileged["unrelated"] is None


@pytest.mark.parametrize(
    ("mode", "retryable", "message"),
    [
        ("timeout", True, "adapter timed out"),
        ("nonzero", True, "adapter exited with status 7"),
        ("malformed", False, "adapter returned malformed JSON"),
        ("oversized", False, "adapter response exceeded 1024 bytes"),
        ("wrong-fingerprint", False, "adapter returned the wrong fingerprint"),
        (
            "stderr-flood",
            False,
            "adapter diagnostic output exceeded 1024 bytes",
        ),
    ],
)
def test_command_adapter_normalizes_transport_and_contract_failures(
    tmp_path,
    mode,
    retryable,
    message,
) -> None:
    capsule, command, _checkpoint, _outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    adapter, _log_path = _command_adapter(command, tmp_path, mode)

    response = adapter.execute(_request(command, handoff, "write"))

    assert response.ok is False
    assert response.fingerprint is None
    assert response.retryable is retryable
    assert response.error == message
    assert "do-not-copy" not in response.error


class RecordingAdapter:
    def __init__(
        self,
        name: str,
        calls: list[tuple[str, str, str]],
        respond: Callable,
    ) -> None:
        self.name = name
        self.calls = calls
        self.respond = respond

    def execute(self, request):
        self.calls.append((self.name, request.operation.value, request.idempotency_key))
        return self.respond(request)


def _success(command, request):
    return command.AdapterResponse(
        ok=True,
        fingerprint=request.capsule.fingerprint,
        retryable=False,
        error=None,
    )


def test_checkpoint_writes_every_required_adapter_before_exact_readback(
    tmp_path,
) -> None:
    capsule, command, checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    calls: list[tuple[str, str, str]] = []
    local = RecordingAdapter("local", calls, lambda request: _success(command, request))
    beads = RecordingAdapter("beads", calls, lambda request: _success(command, request))
    manager = checkpoint.CheckpointManager(
        required_adapters=(local, beads),
        mirror_adapters=(),
        outbox=outbox.MirrorOutbox(tmp_path / "mirror.jsonl"),
    )

    result = manager.checkpoint(handoff, idempotency_key="handoff-8")

    assert result.verified is True
    assert result.fingerprint == handoff.fingerprint
    assert calls == [
        ("local", "write", "handoff-8"),
        ("beads", "write", "handoff-8"),
        ("local", "read", "handoff-8"),
        ("beads", "read", "handoff-8"),
    ]


def test_required_readback_mismatch_prevents_verification(tmp_path) -> None:
    capsule, command, checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    calls: list[tuple[str, str, str]] = []

    def respond(request):
        if request.operation.value == "read":
            return command.AdapterResponse(
                ok=True,
                fingerprint="0" * 64,
                retryable=False,
                error=None,
            )
        return _success(command, request)

    required = RecordingAdapter("required", calls, respond)
    manager = checkpoint.CheckpointManager(
        required_adapters=(required,),
        mirror_adapters=(),
        outbox=outbox.MirrorOutbox(tmp_path / "mirror.jsonl"),
    )

    result = manager.checkpoint(handoff, idempotency_key="handoff-8")

    assert result.verified is False
    assert result.fingerprint == handoff.fingerprint


def test_acknowledgement_is_required_and_mirror_failures_are_queued(tmp_path) -> None:
    capsule, command, checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    calls: list[tuple[str, str, str]] = []
    required = RecordingAdapter(
        "required", calls, lambda request: _success(command, request)
    )
    mirror = RecordingAdapter(
        "linear",
        calls,
        lambda _request: command.AdapterResponse(
            ok=False,
            fingerprint=None,
            retryable=True,
            error="temporarily unavailable",
        ),
    )
    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")
    manager = checkpoint.CheckpointManager(
        required_adapters=(required,),
        mirror_adapters=(mirror,),
        outbox=queue,
    )

    result = manager.acknowledge(handoff, idempotency_key="handoff-8:ack")

    assert result.verified is True
    assert calls == [
        ("required", "acknowledge", "handoff-8:ack"),
        ("linear", "acknowledge", "handoff-8:ack"),
    ]
    assert queue.depth == 1
    pending = queue.pending()[0]
    assert pending.request.operation.value == "acknowledge"
    assert pending.request.idempotency_key == "handoff-8:ack"


@pytest.mark.parametrize("operation", ["checkpoint", "acknowledge"])
def test_raising_mirror_is_fail_open_and_queued(tmp_path, operation) -> None:
    capsule, command, checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    calls: list[tuple[str, str, str]] = []
    required = RecordingAdapter(
        "required", calls, lambda request: _success(command, request)
    )

    class RaisingMirror:
        name = "linear"

        def execute(self, _request):
            raise RuntimeError("LINEAR_API_KEY=must-not-leak")

    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")
    manager = checkpoint.CheckpointManager(
        required_adapters=(required,),
        mirror_adapters=(RaisingMirror(),),
        outbox=queue,
    )

    if operation == "checkpoint":
        result = manager.checkpoint(handoff, idempotency_key="handoff-8")
    else:
        result = manager.acknowledge(handoff, idempotency_key="handoff-8:ack")

    assert result.verified is True
    assert len(result.mirror_attempts) == 1
    assert result.mirror_attempts[0].response.retryable is True
    assert result.mirror_attempts[0].response.error == "credential=[redacted]"
    assert queue.depth == 1
    assert queue.pending()[0].request.operation.value == operation.replace(
        "checkpoint", "write"
    )


def test_mirror_enqueue_failure_is_fail_open_but_observable(tmp_path) -> None:
    capsule, command, checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    calls: list[tuple[str, str, str]] = []
    required = RecordingAdapter(
        "required", calls, lambda request: _success(command, request)
    )
    mirror = RecordingAdapter(
        "linear",
        calls,
        lambda _request: command.AdapterResponse(
            ok=False,
            fingerprint=None,
            retryable=True,
            error="temporarily unavailable",
        ),
    )
    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")

    def fail_enqueue(_adapter, _request):
        raise OSError("LINEAR_API_KEY=must-not-leak")

    queue.enqueue = fail_enqueue
    manager = checkpoint.CheckpointManager(
        required_adapters=(required,),
        mirror_adapters=(mirror,),
        outbox=queue,
    )

    result = manager.checkpoint(handoff, idempotency_key="handoff-8")

    assert result.verified is True
    assert result.mirror_attempts[0].enqueue_error == (
        "mirror retry enqueue failed: credential=[redacted]"
    )


def test_required_acknowledgement_mismatch_prevents_completion(tmp_path) -> None:
    capsule, command, checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    calls: list[tuple[str, str, str]] = []
    required = RecordingAdapter(
        "required",
        calls,
        lambda _request: command.AdapterResponse(
            ok=True,
            fingerprint="0" * 64,
            retryable=False,
            error=None,
        ),
    )
    manager = checkpoint.CheckpointManager(
        required_adapters=(required,),
        mirror_adapters=(),
        outbox=outbox.MirrorOutbox(tmp_path / "mirror.jsonl"),
    )

    result = manager.acknowledge(handoff, idempotency_key="handoff-8:ack")

    assert result.verified is False
    assert calls == [("required", "acknowledge", "handoff-8:ack")]


def test_mirror_failure_is_deduplicated_without_changing_required_verification(
    tmp_path,
) -> None:
    capsule, command, checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    calls: list[tuple[str, str, str]] = []
    required = RecordingAdapter(
        "required", calls, lambda request: _success(command, request)
    )
    mirror = RecordingAdapter(
        "linear",
        calls,
        lambda _request: command.AdapterResponse(
            ok=False,
            fingerprint=None,
            retryable=True,
            error="temporarily unavailable",
        ),
    )
    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")
    manager = checkpoint.CheckpointManager(
        required_adapters=(required,),
        mirror_adapters=(mirror,),
        outbox=queue,
    )

    first = manager.checkpoint(handoff, idempotency_key="handoff-8")
    second = manager.checkpoint(handoff, idempotency_key="handoff-8")

    assert first.verified is True
    assert second.verified is True
    assert queue.depth == 1
    assert stat.S_IMODE(queue.path.stat().st_mode) == 0o600
    pending = queue.pending()
    assert pending[0].adapter == "linear"
    assert pending[0].request.operation.value == "write"
    assert pending[0].request.idempotency_key == "handoff-8"
    encoded = queue.path.read_text(encoding="utf-8").strip()
    assert encoded == json.dumps(
        json.loads(encoded), sort_keys=True, separators=(",", ":")
    )


def test_checkpoint_manager_replays_retained_mirror_work(tmp_path) -> None:
    capsule, command, checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    calls: list[tuple[str, str, str]] = []
    online = False

    def mirror_response(request):
        if online:
            return _success(command, request)
        return command.AdapterResponse(
            ok=False,
            fingerprint=None,
            retryable=True,
            error="temporarily unavailable",
        )

    required = RecordingAdapter(
        "required", calls, lambda request: _success(command, request)
    )
    mirror = RecordingAdapter("linear", calls, mirror_response)
    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")
    manager = checkpoint.CheckpointManager(
        required_adapters=(required,),
        mirror_adapters=(mirror,),
        outbox=queue,
    )
    manager.checkpoint(handoff, idempotency_key="handoff-8")
    assert queue.depth == 1

    online = True
    result = manager.replay_mirrors()

    assert result.succeeded == 1
    assert result.retained == 0
    assert queue.depth == 0


def test_outbox_deduplicates_only_the_adapter_and_idempotency_pair(tmp_path) -> None:
    capsule, command, _checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")
    request = _request(command, handoff, "write", key="handoff-8")

    assert queue.enqueue("linear", request) is True
    assert queue.enqueue("linear", request) is False
    assert queue.enqueue("audit", request) is True
    assert queue.depth == 2


def test_outbox_replays_oldest_first_and_dead_letters_nonretryable_failures(
    tmp_path,
) -> None:
    capsule, command, _checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")
    for key in ("first", "second", "third"):
        queue.enqueue("linear", _request(command, handoff, "write", key=key))

    calls: list[tuple[str, str, str]] = []

    def respond(request):
        if request.idempotency_key == "first":
            return command.AdapterResponse(
                ok=False,
                fingerprint=None,
                retryable=True,
                error="try again",
            )
        if request.idempotency_key == "third":
            return command.AdapterResponse(
                ok=False,
                fingerprint=None,
                retryable=False,
                error="  permanent\nfailure\x00  ",
            )
        return _success(command, request)

    adapter = RecordingAdapter("linear", calls, respond)

    result = queue.replay({"linear": adapter})

    assert [call[2] for call in calls] == ["first", "second", "third"]
    assert result.attempted == 3
    assert result.succeeded == 1
    assert result.retained == 1
    assert result.dead_lettered == 1
    assert [entry.request.idempotency_key for entry in queue.pending()] == ["first"]
    dead_letters = queue.dead_letters()
    assert len(dead_letters) == 1
    assert dead_letters[0].request.idempotency_key == "third"
    assert dead_letters[0].error == "permanent failure"
    assert stat.S_IMODE(queue.dead_letter_path.stat().st_mode) == 0o600


def test_outbox_replay_bounds_each_batch_and_retains_the_remainder(tmp_path) -> None:
    capsule, command, _checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")
    for key in ("first", "second", "third"):
        queue.enqueue("mirror", _request(command, handoff, "write", key=key))
    calls: list[str] = []

    class SuccessfulAdapter:
        name = "mirror"

        def execute(self, request):
            calls.append(request.idempotency_key)
            return _success(command, request)

    result = queue.replay({"mirror": SuccessfulAdapter()}, max_attempts=2)

    assert result.attempted == 2
    assert result.succeeded == 2
    assert result.retained == 1
    assert calls == ["first", "second"]
    assert [entry.request.idempotency_key for entry in queue.pending()] == ["third"]


def test_outbox_replay_does_not_hold_queue_lock_during_adapter_execution(
    tmp_path,
) -> None:
    capsule, command, _checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")
    queue.enqueue("mirror", _request(command, handoff, "write", key="first"))
    adapter_started = threading.Event()
    release_adapter = threading.Event()
    replay_finished = threading.Event()
    enqueue_finished = threading.Event()

    class BlockingAdapter:
        name = "mirror"

        def execute(self, request):
            adapter_started.set()
            assert release_adapter.wait(timeout=2)
            return _success(command, request)

    def replay() -> None:
        queue.replay({"mirror": BlockingAdapter()}, max_attempts=1)
        replay_finished.set()

    replay_thread = threading.Thread(target=replay)
    replay_thread.start()
    assert adapter_started.wait(timeout=1)

    def enqueue() -> None:
        queue.enqueue("mirror", _request(command, handoff, "write", key="second"))
        enqueue_finished.set()

    enqueue_thread = threading.Thread(target=enqueue)
    enqueue_thread.start()
    try:
        assert enqueue_finished.wait(timeout=0.5), (
            "live supervision must be able to enqueue while a mirror call is blocked"
        )
    finally:
        release_adapter.set()
        replay_thread.join(timeout=2)
        enqueue_thread.join(timeout=2)

    assert replay_finished.is_set()
    assert [entry.request.idempotency_key for entry in queue.pending()] == ["second"]


def test_concurrent_outbox_replayers_claim_each_entry_once(tmp_path) -> None:
    capsule, command, _checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")
    queue.enqueue("mirror", _request(command, handoff, "write", key="one-call"))
    adapter_started = threading.Event()
    release_adapter = threading.Event()
    second_finished = threading.Event()
    calls: list[str] = []

    class BlockingAdapter:
        name = "mirror"

        def execute(self, request):
            calls.append(request.idempotency_key)
            adapter_started.set()
            assert release_adapter.wait(timeout=2)
            return _success(command, request)

    adapter = BlockingAdapter()
    first = threading.Thread(
        target=lambda: queue.replay({"mirror": adapter}, max_attempts=1)
    )

    def second_replay() -> None:
        queue.replay({"mirror": adapter}, max_attempts=1)
        second_finished.set()

    first.start()
    assert adapter_started.wait(timeout=1)
    second = threading.Thread(target=second_replay)
    second.start()
    try:
        assert second_finished.wait(timeout=0.5)
    finally:
        release_adapter.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert calls == ["one-call"]
    assert queue.depth == 0


def test_outbox_rejects_reads_larger_than_its_queue_bound(tmp_path) -> None:
    _capsule_module, _command, _checkpoint, outbox = _modules()
    path = tmp_path / "mirror.jsonl"
    path.write_text("x" * 65, encoding="utf-8")
    path.chmod(0o600)
    queue = outbox.MirrorOutbox(path, max_queue_bytes=64)

    with pytest.raises(ValueError, match="exceeds.*64"):
        queue.pending()


def test_outbox_treats_a_successful_wrong_fingerprint_as_nonretryable(tmp_path) -> None:
    capsule, command, _checkpoint, outbox = _modules()
    handoff = _capsule(capsule, tmp_path)
    queue = outbox.MirrorOutbox(tmp_path / "mirror.jsonl")
    queue.enqueue("linear", _request(command, handoff, "write"))
    calls: list[tuple[str, str, str]] = []
    adapter = RecordingAdapter(
        "linear",
        calls,
        lambda _request: command.AdapterResponse(
            ok=True,
            fingerprint="0" * 64,
            retryable=False,
            error=None,
        ),
    )

    result = queue.replay({"linear": adapter})

    assert result.dead_lettered == 1
    assert queue.depth == 0
    assert queue.dead_letters()[0].error == "adapter returned the wrong fingerprint"
