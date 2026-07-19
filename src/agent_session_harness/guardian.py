"""Durable runtime guardian with lease-heartbeat enforcement."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import termios
import time
from typing import Callable, Mapping

from .process import (
    ExitReason,
    ProcessExit,
    read_runtime_abort,
    record_guarded_exit,
    register_guarded_process,
    unregister_guarded_process,
)
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


class _TerminalLease:
    """Temporarily hand the controlling terminal to one runtime process group."""

    def __init__(self, fd: int, supervisor_pgid: int, attributes: list[object]):
        self.fd = fd
        self.supervisor_pgid = supervisor_pgid
        self.attributes = attributes
        self.restored = False

    @classmethod
    def capture(cls) -> _TerminalLease | None:
        fd = 0
        if not os.isatty(fd):
            return None
        try:
            return cls(fd, os.tcgetpgrp(fd), termios.tcgetattr(fd))
        except OSError as exc:
            raise RuntimeError(
                "interactive guardian cannot inspect its terminal"
            ) from exc

    def foreground(self, process_group_id: int) -> None:
        previous = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
        try:
            os.tcsetpgrp(self.fd, process_group_id)
            os.killpg(process_group_id, signal.SIGCONT)
        except OSError as exc:
            raise RuntimeError(
                "interactive guardian cannot transfer terminal ownership"
            ) from exc
        finally:
            signal.signal(signal.SIGTTOU, previous)

    def restore(self) -> None:
        if self.restored:
            return
        previous = signal.signal(signal.SIGTTOU, signal.SIG_IGN)
        try:
            os.tcsetpgrp(self.fd, self.supervisor_pgid)
            termios.tcsetattr(self.fd, termios.TCSANOW, self.attributes)
        except OSError as exc:
            raise RuntimeError(
                "interactive guardian cannot restore terminal ownership"
            ) from exc
        finally:
            signal.signal(signal.SIGTTOU, previous)
        self.restored = True


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
    terminal_lease = _TerminalLease.capture()
    child: subprocess.Popen[bytes] | None = None
    child_session_id: int | None = None
    process = None
    acknowledgement_abort_requested = False

    def request_acknowledgement_abort(_signum, _frame) -> None:
        nonlocal acknowledgement_abort_requested
        acknowledgement_abort_requested = True

    previous_abort_handler = signal.signal(
        signal.SIGUSR1,
        request_acknowledgement_abort,
    )
    try:
        child = subprocess.Popen(
            command,
            cwd=args.cwd,
            env=child_environment,
            process_group=0,
        )
        child_session_id = os.getsid(child.pid)
        if terminal_lease is not None:
            terminal_lease.foreground(child.pid)
        process = register_guarded_process(
            registry_path=Path(args.registry),
            intent_path=Path(args.intent),
            registry_key=args.registry_key,
            command_digest=args.command_digest,
            launch_nonce=args.launch_nonce,
            process_group_id=child.pid,
        )
        terminal_exit = _watch_child(
            child,
            process_pid=guardian_pid,
            chain_id=_required_environment("AGENT_SESSION_HARNESS_CHAIN_ID"),
            generation=int(_required_environment("AGENT_SESSION_HARNESS_GENERATION")),
            state_path=_state_path(os.environ),
            timeout_seconds=timeout,
            process_group_session_id=child_session_id,
            acknowledgement_abort_requested=(lambda: acknowledgement_abort_requested),
        )
        if terminal_lease is not None:
            terminal_lease.restore()
        record_guarded_exit(
            registry_path=Path(args.registry),
            process=process,
            terminal=terminal_exit,
        )
        return terminal_exit.return_code
    finally:
        if child is not None and child.poll() is None:
            _terminate_child(child)
        if child is not None:
            _drain_process_group(
                child.pid,
                expected_session_id=child_session_id,
            )
        try:
            if terminal_lease is not None:
                terminal_lease.restore()
        finally:
            try:
                if process is not None:
                    unregister_guarded_process(
                        registry_path=Path(args.registry),
                        process=process,
                    )
            finally:
                signal.signal(signal.SIGUSR1, previous_abort_handler)


def _watch_child(
    child: subprocess.Popen[bytes],
    *,
    process_pid: int,
    chain_id: str,
    generation: int,
    state_path: Path | None,
    timeout_seconds: float,
    process_group_session_id: int | None = None,
    acknowledgement_abort_requested: Callable[[], bool] | None = None,
) -> ProcessExit:
    deadline = time.monotonic() + timeout_seconds
    parent_pid = os.getppid()
    interval = min(WATCHDOG_POLL_MAX_SECONDS, timeout_seconds / 4)
    reason = ExitReason.NATURAL
    while child.poll() is None:
        if (
            acknowledgement_abort_requested is not None
            and acknowledgement_abort_requested()
        ):
            reason = ExitReason.ACKNOWLEDGEMENT_FAILED
            _terminate_child(child)
            break
        if _child_was_stopped(child.pid):
            # A nested managed runtime cannot safely hand shell job control back
            # without also suspending the outer supervisor and its lease. Keep
            # terminal input live instead of leaving both layers deadlocked.
            try:
                os.killpg(child.pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        if state_path is None:
            if os.getppid() == parent_pid:
                deadline = time.monotonic() + timeout_seconds
        else:
            abort_request = read_runtime_abort(state_path)
            if (
                abort_request is not None
                and abort_request.chain_id == chain_id
                and abort_request.generation == generation
                and abort_request.owner_pid == process_pid
            ):
                reason = ExitReason.ACKNOWLEDGEMENT_FAILED
                _terminate_child(child)
                break
            status = _read_watchdog_state(
                state_path,
                process_pid=process_pid,
                chain_id=chain_id,
                generation=generation,
                timeout_seconds=timeout_seconds,
            )
            if isinstance(status, ExitReason):
                reason = status
                _terminate_child(child)
                break
            if isinstance(status, float):
                deadline = status
        if time.monotonic() >= deadline:
            reason = ExitReason.WATCHDOG_EXPIRED
            _terminate_child(child)
            break
        time.sleep(interval)
    return_code = int(child.wait())
    if reason is ExitReason.NATURAL and state_path is not None:
        final_status = _read_watchdog_state(
            state_path,
            process_pid=process_pid,
            chain_id=chain_id,
            generation=generation,
            timeout_seconds=timeout_seconds,
        )
        if isinstance(final_status, ExitReason):
            reason = final_status
    if not _drain_process_group(
        child.pid,
        expected_session_id=process_group_session_id,
    ):
        reason = ExitReason.PROCESS_GROUP_UNVERIFIED
    return ProcessExit(return_code=return_code, reason=reason)


def _child_was_stopped(pid: int) -> bool:
    """Consume a child stop notification without reaping terminal exit state."""

    try:
        status = os.waitid(os.P_PID, pid, os.WSTOPPED | os.WNOHANG)
    except (AttributeError, ChildProcessError, OSError):
        return False
    return status is not None and status.si_code == os.CLD_STOPPED


def _read_watchdog_state(
    path: Path,
    *,
    process_pid: int,
    chain_id: str,
    generation: int,
    timeout_seconds: float,
) -> float | ExitReason | None:
    try:
        payload = json.loads(read_private_text(path))
        claim = payload["claim"]
        owner_session_id = f"{chain_id}:{generation}"
        if (
            payload.get("chain_id") != chain_id
            or payload.get("generation") != generation
            or not isinstance(claim, dict)
            or claim.get("owner_session_id") != owner_session_id
            or payload.get("process_pid") not in {None, process_pid}
        ):
            return ExitReason.STATE_INVALID
        if payload.get("phase") in {"blocked", "stopping"}:
            return ExitReason.SUPERVISOR_STOP
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


def _drain_process_group(
    process_group_id: int,
    *,
    expected_session_id: int | None = None,
) -> bool:
    """Drain only descendants that still belong to the child's original session."""

    required_session_id = (
        process_group_id if expected_session_id is None else expected_session_id
    )
    members = _verified_process_group_members(
        process_group_id,
        expected_session_id=required_session_id,
    )
    if members is None:
        return False
    if not members:
        return True
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if _await_empty_process_group(
        process_group_id,
        timeout_seconds=TERMINATE_GRACE_SECONDS,
        expected_session_id=required_session_id,
    ):
        return True
    members = _verified_process_group_members(
        process_group_id,
        expected_session_id=required_session_id,
    )
    if members is None:
        return False
    if not members:
        return True
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return _await_empty_process_group(
        process_group_id,
        timeout_seconds=KILL_WAIT_SECONDS,
        expected_session_id=required_session_id,
    )


def _await_empty_process_group(
    process_group_id: int,
    *,
    timeout_seconds: float,
    expected_session_id: int | None = None,
) -> bool:
    required_session_id = (
        process_group_id if expected_session_id is None else expected_session_id
    )
    deadline = time.monotonic() + timeout_seconds
    while True:
        members = _verified_process_group_members(
            process_group_id,
            expected_session_id=required_session_id,
        )
        if members is None:
            return False
        if not members:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def verified_process_group_members(process_group_id: int) -> set[int] | None:
    """Return live members only when the kernel still binds the group to one session."""

    if isinstance(process_group_id, bool) or process_group_id <= 0:
        return None
    try:
        session_id = os.getsid(process_group_id)
    except (OSError, PermissionError, ProcessLookupError):
        return None
    return _verified_process_group_members(
        process_group_id,
        expected_session_id=session_id,
    )


def _verified_process_group_members(
    process_group_id: int,
    *,
    expected_session_id: int | None = None,
) -> set[int] | None:
    required_session_id = (
        process_group_id if expected_session_id is None else expected_session_id
    )
    try:
        completed = subprocess.run(
            ["/bin/ps", "-ax", "-o", "pid=", "-o", "pgid=", "-o", "stat="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    members: set[int] = set()
    for line in completed.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) < 2:
            return None
        try:
            pid = int(fields[0])
            pgid = int(fields[1])
        except ValueError:
            return None
        if pgid != process_group_id:
            continue
        try:
            actual_session_id = os.getsid(pid)
        except ProcessLookupError:
            continue
        except (OSError, PermissionError):
            return None
        if actual_session_id != required_session_id:
            return None
        state = fields[2] if len(fields) == 3 else ""
        if not state.startswith("Z"):
            members.add(pid)
    return members


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
