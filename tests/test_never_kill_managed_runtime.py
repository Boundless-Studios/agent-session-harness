"""BOU-2389: the harness must never kill a managed runtime for an infra fault.

A network outage stalls every adapter to its full timeout, which stretches the
supervisor tick past the watchdog budget.  The guardian derives its kill
deadline solely from `last_heartbeat_at`, so it cannot tell "supervisor is dead"
from "supervisor is slow" — and it SIGKILLed healthy interactive sessions.

The governing rule is that the managed runtime's lifetime belongs to the user.
The harness may stop *supervising* a session; it may never *terminate* one for
an infrastructure-side fault.  Fail open on the runtime, fail closed on the
claim.

Deliberate terminations survive unchanged: context rotation replaces a session
with a successor carrying its context, and an unacknowledged successor is a
generation that never became the user's session at all.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest
from test_supervisor import _activity, _modules, _supervisor

from agent_session_harness.activity import Quiescence
from agent_session_harness.coordinator import StaleOwnerError

CHILD_LIFETIME_SECONDS = 1.0


def _guardian():
    return importlib.import_module("agent_session_harness.guardian")


def _child():
    """A runtime that outlives every kill deadline under test."""

    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({CHILD_LIFETIME_SECONDS})"],
        start_new_session=True,
    )


def _state(
    tmp_path,
    *,
    chain_id: str,
    generation: int = 0,
    phase: str = "running",
    process_pid: int,
    heartbeat_age_seconds: float = 0.0,
    owner_session_id: str | None = None,
):
    state_path = tmp_path / "supervisor.json"
    heartbeat = datetime.now(tz=UTC) - timedelta(seconds=heartbeat_age_seconds)
    state_path.write_text(
        json.dumps(
            {
                "claim": {
                    "owner_session_id": (
                        owner_session_id
                        if owner_session_id is not None
                        else f"{chain_id}:{generation}"
                    )
                },
                "chain_id": chain_id,
                "generation": generation,
                "phase": phase,
                "process_pid": process_pid,
                "last_heartbeat_at": heartbeat.isoformat(),
            }
        ),
        encoding="utf-8",
    )
    state_path.chmod(0o600)
    return state_path


# --------------------------------------------------------------------------
# Guardian: infrastructure faults must never terminate the runtime.
# --------------------------------------------------------------------------


def test_guardian_does_not_kill_when_the_watchdog_deadline_expires(
    tmp_path,
) -> None:
    """The reported BOU-2389 crash: `-9 (watchdog_expired)` on a healthy session.

    A stalled supervisor loop is not a dead supervisor.  The heartbeat is
    emitted by the same thread that runs every blocking adapter, so a stale
    heartbeat is a diagnostic signal, never grounds for killing the user's
    session.  Ten minutes stale against a 0.05s timeout is as expired as it
    gets, and the child still finishes on its own.
    """
    process, _supervisor_module = _modules()
    guardian = _guardian()
    state_path = _state(
        tmp_path,
        chain_id="chain-watchdog",
        process_pid=99999,
        heartbeat_age_seconds=600.0,
    )
    child = _child()

    terminal = guardian._watch_child(
        child,
        process_pid=99999,
        chain_id="chain-watchdog",
        generation=0,
        state_path=state_path,
        timeout_seconds=0.05,
    )

    assert terminal.reason is not process.ExitReason.WATCHDOG_EXPIRED
    assert terminal.reason is process.ExitReason.NATURAL
    assert terminal.return_code == 0


def test_guardian_does_not_kill_when_the_heartbeat_cannot_be_parsed(tmp_path) -> None:
    """The BOU-2366 grace floor does not cover this path.

    `_read_watchdog_state` returns `None` — not a deadline — whenever the state
    file cannot be read or its `last_heartbeat_at` cannot be parsed (a torn
    read, a null heartbeat mid-rotation, malformed JSON).  On `None` the
    deadline is never refreshed, so the entry deadline stands and expires.  The
    grace floor only applies to the float branch, so it never sees this one.
    """
    process, _supervisor_module = _modules()
    guardian = _guardian()
    state_path = tmp_path / "supervisor.json"
    state_path.write_text(
        json.dumps(
            {
                "claim": {"owner_session_id": "chain-null-hb:0"},
                "chain_id": "chain-null-hb",
                "generation": 0,
                "phase": "running",
                "process_pid": 99999,
                "last_heartbeat_at": None,  # unparseable -> _read_watchdog_state None
            }
        ),
        encoding="utf-8",
    )
    state_path.chmod(0o600)
    child = _child()

    terminal = guardian._watch_child(
        child,
        process_pid=99999,
        chain_id="chain-null-hb",
        generation=0,
        state_path=state_path,
        timeout_seconds=0.05,
    )

    assert terminal.reason is not process.ExitReason.WATCHDOG_EXPIRED
    assert terminal.reason is process.ExitReason.NATURAL
    assert terminal.return_code == 0


def test_guardian_does_not_kill_when_supervisor_state_is_unreadable(tmp_path) -> None:
    """An unparseable or mismatched state file says nothing about the runtime.

    `STATE_INVALID` describes a supervisor-side bookkeeping fault.  Answering it
    by killing the user's session destroys work to protect an invariant the
    session was never violating.
    """
    process, _supervisor_module = _modules()
    guardian = _guardian()
    state_path = _state(
        tmp_path,
        chain_id="chain-written",
        process_pid=99999,
    )
    child = _child()

    terminal = guardian._watch_child(
        child,
        process_pid=99999,
        chain_id="chain-expected",  # identity mismatch -> STATE_INVALID today
        generation=0,
        state_path=state_path,
        timeout_seconds=30.0,
    )

    assert terminal.reason is not process.ExitReason.STATE_INVALID
    assert terminal.reason is process.ExitReason.NATURAL
    assert terminal.return_code == 0


def test_guardian_does_not_kill_when_the_supervisor_phase_is_blocked(tmp_path) -> None:
    """`blocked` is where a supervisor lands after a fault, not a user request.

    Rotation's `stopping` is a deliberate handoff and still terminates; the two
    were previously collapsed into a single `SUPERVISOR_STOP` kill.
    """
    process, _supervisor_module = _modules()
    guardian = _guardian()
    state_path = _state(
        tmp_path,
        chain_id="chain-blocked",
        phase="blocked",
        process_pid=99999,
    )
    child = _child()

    terminal = guardian._watch_child(
        child,
        process_pid=99999,
        chain_id="chain-blocked",
        generation=0,
        state_path=state_path,
        timeout_seconds=30.0,
    )

    assert terminal.reason is not process.ExitReason.SUPERVISOR_STOP
    assert terminal.reason is process.ExitReason.NATURAL
    assert terminal.return_code == 0


# --------------------------------------------------------------------------
# Guardian: deliberate terminations must keep working.
# --------------------------------------------------------------------------


def test_guardian_still_terminates_for_rotation_stopping_phase(tmp_path) -> None:
    """Rotation replaces a session with a successor; that stop must still land."""
    process, _supervisor_module = _modules()
    guardian = _guardian()
    state_path = _state(
        tmp_path,
        chain_id="chain-stopping",
        phase="stopping",
        process_pid=88888,
    )
    child = _child()

    terminal = guardian._watch_child(
        child,
        process_pid=88888,
        chain_id="chain-stopping",
        generation=0,
        state_path=state_path,
        timeout_seconds=30.0,
    )

    assert terminal.reason is process.ExitReason.SUPERVISOR_STOP
    assert terminal.return_code != 0


def test_guardian_still_terminates_an_unacknowledged_successor(tmp_path) -> None:
    """A successor that never acknowledged never became the user's session."""
    process, _supervisor_module = _modules()
    guardian = _guardian()
    state_path = _state(
        tmp_path,
        chain_id="chain-ack",
        process_pid=77777,
    )
    child = _child()

    terminal = guardian._watch_child(
        child,
        process_pid=77777,
        chain_id="chain-ack",
        generation=0,
        state_path=state_path,
        timeout_seconds=30.0,
        acknowledgement_abort_requested=lambda: True,
    )

    assert terminal.reason is process.ExitReason.ACKNOWLEDGEMENT_FAILED
    assert terminal.return_code != 0


# --------------------------------------------------------------------------
# Supervisor: fault paths must detach, not kill.
# --------------------------------------------------------------------------


def test_losing_the_claim_does_not_stop_a_live_runtime(tmp_path) -> None:
    """Fail closed on the claim, fail open on the runtime.

    Losing the coordinator lease means this supervisor must stop *supervising*.
    It does not mean the user's session should die.
    """
    managed, _kwargs, driver, _coordinator, _checkpoints = _supervisor(
        tmp_path,
        stale_on_heartbeat=True,
        heartbeat_interval_seconds=0.0,  # heartbeat every tick so the claim is lost now
    )
    managed.start()
    launched = managed.current_process
    assert launched is not None

    with pytest.raises(StaleOwnerError):
        managed.tick(_activity(Quiescence.BUSY))

    assert managed.snapshot.phase.value == "blocked"

    assert ("stop", launched.pid) not in driver.calls
    assert launched.pid in driver.active_pids


def test_successor_abort_intent_survives_a_supervisor_restart(tmp_path) -> None:
    """The abort must outlive the process that started it.

    Second-round review finding on PR #21: the in-memory provenance flag is
    reinitialized by a fresh `Supervisor`, so a supervisor killed between
    `_persist_blocked()` and the CLI's `finally` came back, saw BLOCKED with a
    live process, and detached the unacknowledged successor it had been
    aborting. Adoption is now also derived from durable state -- past
    generation 0, a null `conversation_id` means this generation never
    acknowledged, and only an acknowledgement makes a successor the user's.
    """
    _process, supervisor_module = _modules()
    managed, kwargs, driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    launched = managed.current_process
    assert launched is not None

    # A successor generation that never acknowledged, blocked by a failed abort.
    managed.snapshot = managed.snapshot.model_copy(
        update={
            "phase": supervisor_module.SupervisorPhase.BLOCKED,
            "generation": 1,
            "conversation_id": None,
        }
    )
    managed._persist()

    # A brand new Supervisor over the same state: the flag resets to False.
    restarted = type(managed)(**kwargs)
    assert restarted._successor_abort_pending is False
    restarted.current_process = launched

    restarted.shutdown()

    assert ("stop", launched.pid) in driver.calls, (
        "a restarted supervisor must still finish the abort, not detach it"
    )


def test_a_predecessor_being_rotated_out_is_never_detached(tmp_path) -> None:
    """`STOPPING` is a deliberate rotation stop, not a session to preserve.

    Second-round review finding on PR #21: if `graceful_stop()` raises while
    the predecessor is still live, the CLI's `finally` reached shutdown with
    STOPPING persisted. Treating that as adopted cancelled the rotation and
    left the superseded predecessor running.
    """
    _process, supervisor_module = _modules()
    managed, _kwargs, driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    launched = managed.current_process
    assert launched is not None

    managed.snapshot = managed.snapshot.model_copy(
        update={"phase": supervisor_module.SupervisorPhase.STOPPING}
    )
    managed._persist()  # shutdown() re-reads the snapshot from disk

    managed.shutdown()

    assert ("stop", launched.pid) in driver.calls, (
        "a predecessor mid-rotation must still be stopped, not detached"
    )


def test_a_failed_successor_abort_still_terminates_its_runtime(tmp_path) -> None:
    """`BLOCKED` alone must not be read as "this is the user's session".

    Review finding on PR #21: when `_retry_successor_unlocked()` cannot stop a
    live successor it persists BLOCKED and raises, and the CLI's `finally`
    then calls shutdown. Classifying adoption from the phase alone made that
    BLOCKED look adopted, so the abort silently became a detach and the
    successor that never became the user's session was left running.
    """
    _process, supervisor_module = _modules()
    managed, _kwargs, driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    launched = managed.current_process
    assert launched is not None

    # Stand in for "the abort could not stop its runtime": these two lines are
    # exactly what the except branch in _retry_successor_unlocked leaves behind.
    managed._successor_abort_pending = True
    managed.snapshot = managed.snapshot.model_copy(
        update={"phase": supervisor_module.SupervisorPhase.BLOCKED}
    )

    managed.shutdown()

    assert ("stop", launched.pid) in driver.calls, (
        "an aborted successor must still be stopped, not detached"
    )
    assert launched.pid not in driver.active_pids


def test_detach_survives_an_unwritable_announcement(tmp_path, monkeypatch) -> None:
    """A cosmetic print must never strand the claim.

    Review finding on PR #21: `_announce_alarm` runs inside shutdown, ahead of
    coordinator fencing. A redirected stderr whose reader has exited raises
    BrokenPipeError on write, which would abort the shutdown and leave the
    claim held until lease expiry -- failing OPEN on the claim.
    """
    _process, supervisor_module = _modules()
    managed, _kwargs, _driver, coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()

    def explode(_message: str) -> None:
        raise BrokenPipeError("stderr consumer has exited")

    monkeypatch.setattr(
        supervisor_module, "_stderr_belongs_to_someone_else", lambda: False
    )
    monkeypatch.setattr(supervisor_module.sys, "stderr", _ExplodingStream())

    managed.shutdown()

    assert coordinator.active is None, (
        "the claim must be fenced even when the detach announcement cannot print"
    )
    assert managed.snapshot.claim is None


def test_a_detached_projection_does_not_leak_into_the_next_invocation(
    tmp_path, capsys
) -> None:
    """The projection describes ONE invocation's detach.

    Second-round review finding on PR #21: nothing cleared the module-global,
    so a later unrelated failure in the same process gained a stale
    `supervision` document -- and, worse, had its own diagnostics suppressed as
    if a runtime still owned the terminal. The test suite runs `main()`
    repeatedly in one process, which is exactly the shape that exposes it.
    """
    cli = importlib.import_module("agent_session_harness.cli")
    cli._record_detached_projection({"supervision_alarm": "stale from an old run"})

    # Any failing invocation will do; this one cannot find its state file.
    assert (
        cli.main(["report", "--state", str(tmp_path / "missing.json"), "--json"]) == 2
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "supervision" not in payload, (
        "a stale detach must not be attributed to an unrelated later failure"
    )


def test_json_errors_are_suppressed_while_a_detached_runtime_owns_the_tty(
    monkeypatch, capsys
) -> None:
    """The JSON error path needs the same terminal guard as every other write.

    Second-round review finding on PR #21: the plain-text branch honoured
    terminal ownership but the JSON branch printed unconditionally, so
    `supervise --json` still wrote over the detached runtime's foreground TUI.
    """
    cli = importlib.import_module("agent_session_harness.cli")
    monkeypatch.setattr(cli, "_stdio_belongs_to_someone_else", lambda _json: True)

    def explode(_args):
        # What `_run_supervise` does on the detach path: record the projection,
        # then let the exception continue. Recording it before `main()` would
        # not survive -- main() clears the projection on entry by design.
        cli._record_detached_projection({"supervision_alarm": "still running"})
        raise RuntimeError("adapter timed out")

    monkeypatch.setattr(cli, "_run_report", explode)
    assert cli.main(["report", "--state", "/nonexistent", "--json"]) == 2

    assert capsys.readouterr().out == "", (
        "nothing may be written while a detached runtime owns the terminal"
    )


class _ExplodingStream:
    """A stderr whose reader has gone away."""

    def isatty(self) -> bool:
        return False

    def write(self, _text: str) -> int:
        raise BrokenPipeError("stderr consumer has exited")

    def flush(self) -> None:
        raise BrokenPipeError("stderr consumer has exited")


def test_terminal_shutdown_leaves_a_live_runtime_running(tmp_path) -> None:
    """The supervisor going down must not take the user's session with it.

    BOU-2208 already documented this wound from the other end: an unhandled
    adapter error unwound `tick()` into the CLI's `finally: managed.shutdown()`,
    "which persists BLOCKED and kills the runtime".  That fixed one trigger;
    this removes the kill.
    """
    managed, _kwargs, driver, _coordinator, _checkpoints = _supervisor(tmp_path)
    managed.start()
    launched = managed.current_process
    assert launched is not None

    managed.shutdown()

    assert ("stop", launched.pid) not in driver.calls
    assert launched.pid in driver.active_pids
