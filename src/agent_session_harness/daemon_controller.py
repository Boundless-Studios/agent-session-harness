"""Private local RPC boundary for persistent daemon ownership."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .daemon_lifecycle import (
    DaemonDefinition,
    DaemonLifecyclePhase,
    DaemonLifecycleRecord,
    OwnerDiagnosticLock,
)
from .daemon_supervisor import DaemonSupervisor

DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024
DEFAULT_CONTROLLER_STARTUP_TIMEOUT = 5.0
DEFAULT_CONTROLLER_REQUEST_TIMEOUT = 10.0


class DaemonOperation(StrEnum):
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    STATUS = "status"


class DaemonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    operation: DaemonOperation
    definition: DaemonDefinition
    allow_process_takeover: bool = False


class DaemonResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    record: DaemonLifecycleRecord


class _ControllerProbe(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    kind: Literal["probe"] = "probe"
    state_directory: Path


class _ControllerProbeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    kind: Literal["probe"] = "probe"
    state_directory: Path


class _ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    error: str = Field(min_length=1, max_length=1024)


class DaemonControllerClient:
    def __init__(
        self,
        socket_path: str | Path,
        *,
        timeout: float = DEFAULT_CONTROLLER_REQUEST_TIMEOUT,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout
        self.max_message_bytes = max_message_bytes

    def request(self, request: DaemonRequest) -> DaemonResponse:
        exclude = None if request.allow_process_takeover else {"allow_process_takeover"}
        payload = request.model_dump_json(exclude=exclude).encode() + b"\n"
        if len(payload) > self.max_message_bytes:
            raise ValueError(
                f"daemon controller request exceeds {self.max_message_bytes} bytes"
            )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(str(self.socket_path))
            connection.sendall(payload)
            raw = _read_message(connection, self.max_message_bytes)
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("daemon controller returned invalid JSON") from exc
        if isinstance(decoded, dict) and "error" in decoded:
            error = _ErrorResponse.model_validate(decoded)
            raise RuntimeError(error.error)
        return DaemonResponse.model_validate(decoded)


def ensure_controller(
    socket_path: str | Path,
    state_directory: str | Path,
    *,
    startup_timeout: float = DEFAULT_CONTROLLER_STARTUP_TIMEOUT,
    allow_process_takeover: bool = False,
) -> None:
    """Ensure a detached controller is accepting local requests."""
    resolved_socket = Path(socket_path)
    resolved_state = Path(state_directory)
    if _controller_available(resolved_socket, resolved_state):
        return
    command = [
        sys.executable,
        "-m",
        "agent_session_harness.daemon_controller",
        "serve",
        "--socket",
        str(resolved_socket),
        "--state-directory",
        str(resolved_state),
    ]
    if allow_process_takeover:
        command.append("--takeover")
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if _controller_available(resolved_socket, resolved_state):
            return
        time.sleep(0.01)
    raise TimeoutError("daemon controller did not become ready before the deadline")


class DaemonControllerServer:
    def __init__(
        self,
        *,
        socket_path: str | Path,
        state_directory: str | Path,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        allow_process_takeover: bool = False,
        **supervisor_options: float,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.state_directory = Path(state_directory)
        self.max_message_bytes = max_message_bytes
        self.allow_process_takeover = allow_process_takeover
        self.supervisor_options = supervisor_options
        self._supervisors: dict[str, DaemonSupervisor] = {}
        self._definitions: dict[str, DaemonDefinition] = {}
        self._ready = threading.Event()
        self._closing = threading.Event()
        self._listener: socket.socket | None = None
        self._startup_error: BaseException | None = None
        self._owner_lock = OwnerDiagnosticLock(
            self.socket_path.with_suffix(".lock"),
            purpose="daemon-controller",
        )

    def serve(self) -> None:
        self._prepare_directories()
        with self._owner_lock.acquire(timeout=0):
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._listener = listener
            try:
                self._remove_stale_socket()
                listener.bind(str(self.socket_path))
                os.chmod(self.socket_path, 0o600)
                listener.listen()
                listener.settimeout(0.1)
                self._ready.set()
                while not self._closing.is_set():
                    try:
                        connection, _ = listener.accept()
                    except TimeoutError:
                        continue
                    with connection:
                        connection.settimeout(5)
                        self._serve_connection(connection)
            except BaseException as exc:
                self._startup_error = exc
                raise
            finally:
                self._ready.set()
                listener.close()
                self._listener = None
                self._stop_owned_daemons()
                self._unlink_socket()

    def wait_until_ready(self, *, timeout: float) -> bool:
        observed = self._ready.wait(timeout)
        return observed and self._startup_error is None and self.socket_path.exists()

    def close(self) -> None:
        self._closing.set()

    def _serve_connection(self, connection: socket.socket) -> None:
        try:
            raw = _read_message(connection, self.max_message_bytes)
            decoded = json.loads(raw)
            if isinstance(decoded, dict) and decoded.get("kind") == "probe":
                probe = _ControllerProbe.model_validate(decoded)
                expected = self.state_directory.resolve(strict=False)
                if probe.state_directory.resolve(strict=False) != expected:
                    raise RuntimeError("daemon controller state directory mismatch")
                response: BaseModel = _ControllerProbeResponse(state_directory=expected)
            else:
                request = DaemonRequest.model_validate(decoded)
                response = DaemonResponse(record=self._dispatch(request))
        except (ValidationError, ValueError, RuntimeError, OSError) as exc:
            response = _ErrorResponse(error=str(exc) or type(exc).__name__)
        try:
            connection.sendall(response.model_dump_json().encode() + b"\n")
        except OSError:
            pass

    def _dispatch(self, request: DaemonRequest) -> DaemonLifecycleRecord:
        key = request.definition.daemon_key
        definition = self._definitions.get(key)
        supervisor = self._supervisors.get(key)
        if supervisor is None:
            supervisor = self._new_supervisor(request.definition)
            self._definitions[key] = request.definition
            self._supervisors[key] = supervisor
        elif definition is None:
            raise RuntimeError("daemon controller definition ownership is inconsistent")
        if request.allow_process_takeover:
            supervisor.allow_process_takeover = True
        if definition is not None and definition != request.definition:
            return self._dispatch_definition_drift(
                request,
                supervisor,
                definition,
            )
        return self._dispatch_owned(request, supervisor)

    def _dispatch_owned(
        self,
        request: DaemonRequest,
        supervisor: DaemonSupervisor,
    ) -> DaemonLifecycleRecord:
        if request.operation is DaemonOperation.START:
            return supervisor.start()
        if request.operation is DaemonOperation.STOP:
            record = supervisor.stop()
            self._forget(request.definition.daemon_key)
            return record
        if request.operation is DaemonOperation.STATUS:
            return supervisor.status()
        return supervisor.restart()

    def _dispatch_definition_drift(
        self,
        request: DaemonRequest,
        supervisor: DaemonSupervisor,
        previous_definition: DaemonDefinition,
    ) -> DaemonLifecycleRecord:
        """Reconcile a changed definition while retaining identity fencing.

        The supervisor that owns the existing key remains the authority for
        observing or stopping its process.  A stopped key can be replaced by
        the requested definition.  A running key is restarted through the old
        supervisor first, then the new definition is started; if that launch
        fails, the old definition is started again before the error escapes.
        """
        current = supervisor.status()
        if (
            current.phase is DaemonLifecyclePhase.RUNNING
            or current.process_identity is not None
            or supervisor.owns_live_child()
        ):
            if request.operation is DaemonOperation.STATUS:
                return current
            if request.operation is DaemonOperation.STOP:
                record = supervisor.stop()
                self._forget(request.definition.daemon_key)
                return record
            return self._restart_definition(
                request,
                supervisor,
                previous_definition,
            )

        if request.operation is DaemonOperation.STOP:
            record = supervisor.stop()
            self._forget(request.definition.daemon_key)
            return record

        replacement = self._new_supervisor(request.definition)
        key = request.definition.daemon_key
        self._definitions[key] = request.definition
        self._supervisors[key] = replacement
        return self._dispatch_owned(request, replacement)

    def _restart_definition(
        self,
        request: DaemonRequest,
        supervisor: DaemonSupervisor,
        previous_definition: DaemonDefinition,
    ) -> DaemonLifecycleRecord:
        key = request.definition.daemon_key
        supervisor.stop()
        replacement = self._new_supervisor(request.definition)
        self._definitions[key] = request.definition
        self._supervisors[key] = replacement
        try:
            return replacement.start()
        except Exception as start_error:
            try:
                observed = replacement.status()
            except Exception as observe_error:
                raise RuntimeError(
                    f"{start_error}; definition recovery status failed: "
                    f"{type(observe_error).__name__}: {observe_error}"
                ) from start_error
            if (
                observed.phase is DaemonLifecyclePhase.RUNNING
                or observed.process_identity is not None
                or replacement.owns_live_child()
            ):
                raise
            self._definitions[key] = previous_definition
            self._supervisors[key] = supervisor
            try:
                supervisor.start()
            except Exception as rollback_error:
                raise RuntimeError(
                    f"{start_error}; definition rollback failed: "
                    f"{type(rollback_error).__name__}: {rollback_error}"
                ) from start_error
            raise

    def _forget(self, key: str) -> None:
        self._definitions.pop(key, None)
        self._supervisors.pop(key, None)

    def _new_supervisor(self, definition: DaemonDefinition) -> DaemonSupervisor:
        stem = hashlib.sha256(definition.daemon_key.encode()).hexdigest()
        return DaemonSupervisor(
            definition,
            state_path=self.state_directory / f"{stem}.json",
            lock_path=self.state_directory / f"{stem}.lock",
            allow_process_takeover=self.allow_process_takeover,
            **self.supervisor_options,
        )

    def _prepare_directories(self) -> None:
        _require_private_directory(self.socket_path.parent)
        try:
            self.state_directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _require_private_directory(self.state_directory)

    def _remove_stale_socket(self) -> None:
        try:
            mode = self.socket_path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(mode):
            raise RuntimeError("daemon controller path is not a socket")
        self.socket_path.unlink()

    def _stop_owned_daemons(self) -> None:
        for supervisor in self._supervisors.values():
            try:
                supervisor.stop()
            except (RuntimeError, OSError, TimeoutError):
                pass

    def _unlink_socket(self) -> None:
        try:
            if stat.S_ISSOCK(self.socket_path.lstat().st_mode):
                self.socket_path.unlink()
        except FileNotFoundError:
            pass


def _read_message(connection: socket.socket, max_bytes: int) -> bytes:
    deadline = time.monotonic() + 5
    payload = bytearray()
    while b"\n" not in payload:
        if time.monotonic() >= deadline:
            raise TimeoutError("daemon controller message timed out")
        chunk = connection.recv(min(4096, max_bytes + 1 - len(payload)))
        if not chunk:
            raise RuntimeError("daemon controller message ended before newline")
        payload.extend(chunk)
        if len(payload) > max_bytes:
            raise ValueError(f"daemon controller message exceeds {max_bytes} bytes")
    message, _, trailing = payload.partition(b"\n")
    if trailing:
        raise ValueError("daemon controller accepts one request per connection")
    return bytes(message)


def _controller_available(socket_path: Path, state_directory: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(DEFAULT_CONTROLLER_REQUEST_TIMEOUT)
            connection.connect(str(socket_path))
            connection.sendall(
                _ControllerProbe(state_directory=state_directory)
                .model_dump_json()
                .encode()
                + b"\n"
            )
            raw = _read_message(connection, DEFAULT_MAX_MESSAGE_BYTES)
    except OSError:
        return False
    decoded = json.loads(raw)
    if isinstance(decoded, dict) and "error" in decoded:
        error = _ErrorResponse.model_validate(decoded)
        raise RuntimeError(error.error)
    _ControllerProbeResponse.model_validate(decoded)
    return True


def _require_private_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"daemon controller directory does not exist: {path}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"daemon controller path is not a directory: {path}")
    if metadata.st_uid != os.getuid():
        raise RuntimeError(f"daemon controller directory has a different owner: {path}")
    if metadata.st_mode & 0o077:
        raise RuntimeError(f"daemon controller directory must be private: {path}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="agent-session-harness-daemon-controller")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--socket", required=True)
    serve.add_argument("--state-directory", required=True)
    serve.add_argument("--takeover", action="store_true")
    args = parser.parse_args(argv)
    DaemonControllerServer(
        socket_path=args.socket,
        state_directory=args.state_directory,
        allow_process_takeover=args.takeover,
    ).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
