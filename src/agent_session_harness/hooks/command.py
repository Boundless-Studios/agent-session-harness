"""Bounded stdin-to-ledger hook command."""

from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, NoReturn, TextIO

from pydantic import BaseModel, ConfigDict, Field

from ..events import LifecycleEvent
from ..ledger import EventLedger
from ..models import EventType, Runtime
from ..process import write_runtime_abort
from ..secure_files import (
    atomic_write_private_text,
    exclusive_lock,
    lexical_absolute,
    private_exists,
    read_private_text,
)
from ..supervisor import (
    SupervisorPhase,
    SupervisorSnapshot,
    write_acknowledgement,
)
from .native import (
    NativeHookResponse,
    handoff_requested_event,
    normalize_native_event,
    repeated_stop_idle_event,
    stop_handshake,
)

MAX_INPUT_BYTES = 1_048_576
SUCCESSOR_READY_TIMEOUT_SECONDS = 2.0
SUCCESSOR_READY_POLL_SECONDS = 0.02
SUCCESSOR_ACK_TIMEOUT_SECONDS = 30.0


class StopRequestMarker(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    chain_id: str = Field(min_length=1)
    generation: int = Field(ge=0)
    requested_at: datetime
    idle_emitted: bool = False


def run_hook(
    *,
    runtime: str,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    environment = environ if environ is not None else os.environ
    if environment.get("AGENT_SESSION_HARNESS_MANAGED") != "1":
        return 0
    encoded = stdin.read(MAX_INPUT_BYTES + 1)
    if len(encoded.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("native hook input is too large")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("native hook input is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("native hook input must be a JSON object")

    chain_id = environment.get("AGENT_SESSION_HARNESS_CHAIN_ID", "")
    ledger_path = environment.get("AGENT_SESSION_HARNESS_LEDGER", "")
    if not chain_id or not ledger_path:
        raise RuntimeError("managed hook environment is incomplete")
    generation = int(environment.get("AGENT_SESSION_HARNESS_GENERATION", "0"))
    owner_pid = int(
        environment.get("AGENT_SESSION_HARNESS_OWNER_PID", str(os.getppid()))
    )
    event = normalize_native_event(
        runtime=runtime,
        payload=payload,
        chain_id=chain_id,
        generation=generation,
        owner_pid=owner_pid,
    )
    ledger = EventLedger(Path(ledger_path))

    if event.event_type is EventType.SESSION_STARTED:
        try:
            ledger.append(event)
            _acknowledge_verified_successor(event=event, environment=environment)
        except Exception:
            if _successor_ack_required(environment):
                _abort_unacknowledged_successor(
                    event=event,
                    environment=environment,
                )
            raise
        response = NativeHookResponse(exit_code=0, stdout={"ok": True})
    elif event.event_type is not EventType.TURN_IDLE:
        if (
            event.event_type is EventType.TURN_STARTED
            and _successor_ack_required(environment)
            and not _successor_dispatch_allowed(event, environment)
        ):
            response = _blocked_successor_prompt(event.runtime)
        else:
            ledger.append(event)
            response = NativeHookResponse(exit_code=0, stdout={"ok": True})
    else:
        response = _handle_stop(
            runtime=Runtime(runtime),
            event=event,
            ledger=ledger,
            environment=environment,
        )
    _write_response(response, stdout=stdout, stderr=stderr)
    return response.exit_code


def _successor_ack_required(environment: Mapping[str, str]) -> bool:
    return any(
        environment.get(key, "")
        for key in (
            "AGENT_SESSION_HARNESS_CAPSULE_PATH",
            "AGENT_SESSION_HARNESS_CAPSULE_FINGERPRINT",
            "AGENT_SESSION_HARNESS_TARGET_GENERATION",
        )
    )


def _abort_unacknowledged_successor(
    *,
    event: LifecycleEvent,
    environment: Mapping[str, str],
) -> NoReturn:
    """Request guardian termination and never release the queued first prompt."""

    state_path: Path | None = None
    try:
        state_path = _state_path(environment, event.cwd)
        write_runtime_abort(
            state_path=state_path,
            chain_id=event.chain_id,
            generation=event.generation,
            owner_pid=event.owner_pid,
        )
    except (OSError, RuntimeError, ValueError):
        pass

    # The guardian is a separate process group, so SIGUSR1 is an independent
    # fail-closed channel when the state filesystem is unavailable.
    try:
        os.kill(event.owner_pid, signal.SIGUSR1)
    except OSError:
        pass

    # Hooks normally inherit the managed runtime's process group. Kill it
    # immediately only after binding that group to the exact supervisor state;
    # the durable guardian marker above remains the fallback for runtimes that
    # isolate hooks into their own process group.
    try:
        if state_path is not None:
            snapshot = _read_snapshot(state_path)
            _validate_snapshot(snapshot, event=event, runtime=event.runtime)
            current_group = os.getpgrp()
            if snapshot.process_group_id == current_group:
                os.killpg(current_group, signal.SIGKILL)
    except (OSError, RuntimeError, ValueError):
        pass

    while True:
        time.sleep(SUCCESSOR_READY_POLL_SECONDS)


def _successor_dispatch_allowed(
    event: LifecycleEvent,
    environment: Mapping[str, str],
) -> bool:
    try:
        snapshot = _read_snapshot(_state_path(environment, event.cwd))
        _validate_snapshot(snapshot, event=event, runtime=event.runtime)
        return (
            snapshot.phase is SupervisorPhase.RUNNING
            and snapshot.conversation_id == event.conversation_id
            and snapshot.checkpoint_fingerprint
            == environment.get("AGENT_SESSION_HARNESS_CAPSULE_FINGERPRINT")
            and str(snapshot.checkpoint_path)
            == environment.get("AGENT_SESSION_HARNESS_CAPSULE_PATH")
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _blocked_successor_prompt(runtime: Runtime) -> NativeHookResponse:
    reason = "Verified handoff acknowledgement is not complete; prompt blocked."
    if runtime is Runtime.CLAUDE:
        return NativeHookResponse(
            exit_code=0,
            stdout={"decision": "block", "reason": reason},
        )
    return NativeHookResponse(exit_code=2, stderr=reason + "\n")


def _acknowledge_verified_successor(
    *,
    event: LifecycleEvent,
    environment: Mapping[str, str],
) -> None:
    capsule_keys = (
        "AGENT_SESSION_HARNESS_CAPSULE_PATH",
        "AGENT_SESSION_HARNESS_CAPSULE_FINGERPRINT",
        "AGENT_SESSION_HARNESS_TARGET_GENERATION",
    )
    values = tuple(environment.get(key, "") for key in capsule_keys)
    if not any(values):
        return
    if not all(values):
        raise RuntimeError("managed successor environment is incomplete")

    state_path = _state_path(environment, event.cwd)
    snapshot = _await_successor_snapshot(state_path, event=event)
    _validate_snapshot(snapshot, event=event, runtime=event.runtime)
    try:
        target_generation = int(values[2])
    except ValueError as exc:
        raise RuntimeError("managed successor generation is invalid") from exc
    capsule_path = lexical_absolute(values[0])
    if (
        target_generation != event.generation
        or snapshot.checkpoint_path != capsule_path
        or snapshot.checkpoint_fingerprint != values[1]
    ):
        raise RuntimeError("managed successor capsule does not match supervisor state")
    if (
        snapshot.phase is SupervisorPhase.RUNNING
        and snapshot.conversation_id == event.conversation_id
    ):
        return
    if snapshot.phase is not SupervisorPhase.AWAITING_ACK:
        raise RuntimeError("managed successor is not awaiting acknowledgement")
    try:
        write_acknowledgement(
            state_path=state_path,
            generation=event.generation,
            fingerprint=values[1],
            conversation_id=event.conversation_id,
            owner_pid=event.owner_pid,
        )
    except ValueError:
        refreshed = _read_snapshot(state_path)
        if (
            refreshed.phase is SupervisorPhase.RUNNING
            and refreshed.runtime is event.runtime
            and refreshed.chain_id == event.chain_id
            and refreshed.generation == event.generation
            and refreshed.process_pid == event.owner_pid
            and refreshed.conversation_id == event.conversation_id
            and refreshed.checkpoint_fingerprint == values[1]
            and refreshed.checkpoint_path == capsule_path
        ):
            return
        raise
    _await_durable_acknowledgement(
        state_path,
        event=event,
        fingerprint=values[1],
        capsule_path=capsule_path,
    )


def _await_durable_acknowledgement(
    state_path: Path,
    *,
    event: LifecycleEvent,
    fingerprint: str,
    capsule_path: Path,
) -> None:
    deadline = time.monotonic() + SUCCESSOR_ACK_TIMEOUT_SECONDS
    while True:
        snapshot = _read_snapshot(state_path)
        if (
            snapshot.phase is SupervisorPhase.RUNNING
            and snapshot.runtime is event.runtime
            and snapshot.chain_id == event.chain_id
            and snapshot.generation == event.generation
            and snapshot.process_pid == event.owner_pid
            and snapshot.conversation_id == event.conversation_id
            and snapshot.checkpoint_fingerprint == fingerprint
            and snapshot.checkpoint_path == capsule_path
        ):
            return
        if snapshot.phase is not SupervisorPhase.AWAITING_ACK:
            raise RuntimeError(
                "managed successor durable acknowledgement was not accepted"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError("managed successor durable acknowledgement timed out")
        time.sleep(SUCCESSOR_READY_POLL_SECONDS)


def _await_successor_snapshot(
    state_path: Path,
    *,
    event: LifecycleEvent,
) -> SupervisorSnapshot:
    deadline = time.monotonic() + SUCCESSOR_READY_TIMEOUT_SECONDS
    while True:
        snapshot = _read_snapshot(state_path)
        if (
            snapshot.runtime is not event.runtime
            or snapshot.chain_id != event.chain_id
            or snapshot.generation != event.generation
        ):
            raise RuntimeError("supervisor state does not match managed SessionStart")
        if (
            snapshot.phase is not SupervisorPhase.LAUNCHING
            or snapshot.process_pid is not None
        ):
            return snapshot
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "managed successor did not become ready for acknowledgement"
            )
        time.sleep(SUCCESSOR_READY_POLL_SECONDS)


def stop_request_path(state_path: str | os.PathLike[str]) -> Path:
    path = Path(state_path)
    return path.with_suffix(path.suffix + ".stop-request")


def _handle_stop(
    *,
    runtime: Runtime,
    event: LifecycleEvent,
    ledger: EventLedger,
    environment: Mapping[str, str],
) -> NativeHookResponse:
    state_path = _state_path(environment, event.cwd)
    snapshot = _read_snapshot(state_path)
    _validate_snapshot(snapshot, event=event, runtime=runtime)
    draining = snapshot.phase in {
        SupervisorPhase.DRAINING,
        SupervisorPhase.CHECKPOINTING,
    }
    checkpoint_verified = (
        snapshot.checkpoint_fingerprint is not None
        and snapshot.phase
        not in {SupervisorPhase.DRAINING, SupervisorPhase.CHECKPOINTING}
    )
    already_requested = False
    if draining and not checkpoint_verified:
        already_requested = not _record_stop_request(
            state_path=state_path,
            snapshot=snapshot,
            ledger=ledger,
            request_event=handoff_requested_event(event),
            idle_event=repeated_stop_idle_event(event),
        )
    else:
        ledger.append(event)
    return stop_handshake(
        runtime=runtime,
        draining=draining,
        checkpoint_verified=checkpoint_verified,
        already_requested=already_requested,
        required_fields=tuple(
            item.strip()
            for item in environment.get(
                "AGENT_SESSION_HARNESS_REQUIRED_CHECKPOINTS", ""
            ).split(",")
            if item.strip()
        ),
    )


def _state_path(environment: Mapping[str, str], cwd: Path) -> Path:
    raw_path = environment.get("AGENT_SESSION_HARNESS_STATE_PATH", "")
    if not raw_path:
        raise RuntimeError("managed Stop hook requires supervisor state")
    path = Path(raw_path).expanduser()
    return lexical_absolute(path if path.is_absolute() else cwd / path)


def _read_snapshot(path: Path) -> SupervisorSnapshot:
    with exclusive_lock(path.with_suffix(path.suffix + ".lock")):
        try:
            encoded = read_private_text(path)
        except FileNotFoundError as exc:
            raise RuntimeError("supervisor state is unavailable") from exc
        return SupervisorSnapshot.model_validate_json(encoded)


def _validate_snapshot(
    snapshot: SupervisorSnapshot,
    *,
    event: LifecycleEvent,
    runtime: Runtime,
) -> None:
    if (
        snapshot.runtime is not runtime
        or snapshot.chain_id != event.chain_id
        or snapshot.generation != event.generation
        or snapshot.process_pid != event.owner_pid
    ):
        raise RuntimeError("supervisor state does not match managed Stop event")


def _record_stop_request(
    *,
    state_path: Path,
    snapshot: SupervisorSnapshot,
    ledger: EventLedger,
    request_event: LifecycleEvent,
    idle_event: LifecycleEvent,
) -> bool:
    marker_path = stop_request_path(state_path)
    lock_path = marker_path.with_suffix(marker_path.suffix + ".lock")
    with exclusive_lock(lock_path):
        existing = _read_marker(marker_path)
        if (
            existing is not None
            and existing.chain_id == snapshot.chain_id
            and existing.generation == snapshot.generation
        ):
            if not existing.idle_emitted:
                ledger.append(idle_event)
                _atomic_write(
                    marker_path,
                    existing.model_copy(update={"idle_emitted": True}).model_dump_json()
                    + "\n",
                )
            return False
        ledger.append(request_event)
        marker = StopRequestMarker(
            chain_id=snapshot.chain_id,
            generation=snapshot.generation,
            requested_at=request_event.timestamp,
        )
        _atomic_write(marker_path, marker.model_dump_json() + "\n")
        return True


def _read_marker(path: Path) -> StopRequestMarker | None:
    if not private_exists(path):
        return None
    return StopRequestMarker.model_validate_json(read_private_text(path))


def _atomic_write(path: Path, value: str) -> None:
    atomic_write_private_text(path, value)


def _write_response(
    response: NativeHookResponse,
    *,
    stdout: TextIO,
    stderr: TextIO | None,
) -> None:
    if response.stdout is not None:
        stdout.write(
            json.dumps(response.stdout, sort_keys=True, separators=(",", ":")) + "\n"
        )
        stdout.flush()
    if response.stderr and stderr is not None:
        stderr.write(response.stderr)
        stderr.flush()
