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
