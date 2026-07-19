"""Durable runtime guardian with lease-heartbeat enforcement."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Mapping

from .process import register_guarded_process, unregister_guarded_process
from .secure_files import lexical_absolute, read_private_text


WATCHDOG_POLL_MAX_SECONDS = 0.1
TERMINATE_GRACE_SECONDS = 1.0
KILL_WAIT_SECONDS = 1.0
WATCHDOG_SHUTDOWN_SLACK_SECONDS = 0.9
WATCHDOG_SHUTDOWN_MARGIN_SECONDS = (
    WATCHDOG_POLL_MAX_SECONDS
    + TERMINATE_GRACE_SECONDS
    + KILL_WAIT_SECONDS
    + WATCHDOG_SHUTDOWN_SLACK_SECONDS
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-session-harness-guardian")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--registry-key", required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--command-digest", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("guardian requires a child command")

    timeout = _watchdog_timeout(os.environ)
    guardian_pid = os.getpid()
    child_environment = dict(os.environ)
    child_environment["AGENT_SESSION_HARNESS_OWNER_PID"] = str(guardian_pid)
    child = subprocess.Popen(
        command,
        cwd=args.cwd,
        env=child_environment,
        start_new_session=True,
    )

    process = None
    try:
        process = register_guarded_process(
            registry_path=Path(args.registry),
            intent_path=Path(args.intent),
            registry_key=args.registry_key,
            command_digest=args.command_digest,
            launch_nonce=args.launch_nonce,
            process_group_id=child.pid,
        )
        return _watch_child(
            child,
            process_pid=guardian_pid,
            chain_id=_required_environment("AGENT_SESSION_HARNESS_CHAIN_ID"),
            generation=int(_required_environment("AGENT_SESSION_HARNESS_GENERATION")),
            state_path=_state_path(os.environ),
            timeout_seconds=timeout,
        )
    finally:
        if child.poll() is None:
            _terminate_child(child)
        if process is not None:
            unregister_guarded_process(
                registry_path=Path(args.registry),
                process=process,
            )


def _watch_child(
    child: subprocess.Popen[bytes],
    *,
    process_pid: int,
    chain_id: str,
    generation: int,
    state_path: Path | None,
    timeout_seconds: float,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    parent_pid = os.getppid()
    interval = min(WATCHDOG_POLL_MAX_SECONDS, timeout_seconds / 4)
    while child.poll() is None:
        if state_path is None:
            if os.getppid() == parent_pid:
                deadline = time.monotonic() + timeout_seconds
        else:
            status = _read_watchdog_state(
                state_path,
                process_pid=process_pid,
                chain_id=chain_id,
                generation=generation,
                timeout_seconds=timeout_seconds,
            )
            if status is False:
                _terminate_child(child)
                break
            if isinstance(status, float):
                deadline = status
        if time.monotonic() >= deadline:
            _terminate_child(child)
            break
        time.sleep(interval)
    return int(child.wait())


def _read_watchdog_state(
    path: Path,
    *,
    process_pid: int,
    chain_id: str,
    generation: int,
    timeout_seconds: float,
) -> float | bool | None:
    try:
        payload = json.loads(read_private_text(path))
        claim = payload["claim"]
        owner_session_id = f"{chain_id}:{generation}"
        if (
            payload.get("chain_id") != chain_id
            or payload.get("generation") != generation
            or payload.get("phase") == "blocked"
            or not isinstance(claim, dict)
            or claim.get("owner_session_id") != owner_session_id
            or payload.get("process_pid") not in {None, process_pid}
        ):
            return False
        heartbeat = datetime.fromisoformat(str(payload["last_heartbeat_at"]))
        if heartbeat.tzinfo is None or heartbeat.utcoffset() is None:
            return None
        age = (datetime.now(tz=timezone.utc) - heartbeat).total_seconds()
        if age < -5:
            return None
        return time.monotonic() + max(0.0, timeout_seconds - max(0.0, age))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _terminate_child(child: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=KILL_WAIT_SECONDS)


def _watchdog_timeout(environment: Mapping[str, str]) -> float:
    raw = environment.get("AGENT_SESSION_HARNESS_WATCHDOG_TIMEOUT_SECONDS", "60")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("guardian watchdog timeout must be numeric") from exc
    if timeout <= 0:
        raise ValueError("guardian watchdog timeout must be positive")
    return timeout


def _state_path(environment: Mapping[str, str]) -> Path | None:
    raw = environment.get("AGENT_SESSION_HARNESS_STATE_PATH")
    return lexical_absolute(raw) if raw else None


def _required_environment(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"guardian requires {key}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
