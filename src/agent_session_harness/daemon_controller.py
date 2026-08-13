"""Private local RPC boundary for persistent daemon ownership."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .daemon_lifecycle import (
    DaemonDefinition,
    DaemonLifecycleRecord,
    OwnerDiagnosticLock,
)
from .daemon_supervisor import DaemonSupervisor

DEFAULT_MAX_MESSAGE_BYTES = 64 * 1024


class DaemonOperation(StrEnum):
    START = "start"
    STOP = "stop"
    RESTART = "restart"


class DaemonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    operation: DaemonOperation
    definition: DaemonDefinition


class DaemonResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    record: DaemonLifecycleRecord


class _ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    error: str = Field(min_length=1, max_length=1024)


class DaemonControllerClient:
    def __init__(
        self,
        socket_path: str | Path,
        *,
        timeout: float = 5,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.timeout = timeout
        self.max_message_bytes = max_message_bytes

    def request(self, request: DaemonRequest) -> DaemonResponse:
        payload = request.model_dump_json().encode() + b"\n"
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


class DaemonControllerServer:
    def __init__(
        self,
        *,
        socket_path: str | Path,
        state_directory: str | Path,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        **supervisor_options: float,
    ) -> None:
        self.socket_path = Path(socket_path)
        self.state_directory = Path(state_directory)
        self.max_message_bytes = max_message_bytes
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
            request = DaemonRequest.model_validate_json(raw)
            response: BaseModel = DaemonResponse(record=self._dispatch(request))
        except (ValidationError, ValueError, RuntimeError, OSError) as exc:
            response = _ErrorResponse(error=str(exc) or type(exc).__name__)
        try:
            connection.sendall(response.model_dump_json().encode() + b"\n")
        except OSError:
            pass

    def _dispatch(self, request: DaemonRequest) -> DaemonLifecycleRecord:
        definition = self._definitions.get(request.definition.daemon_key)
        if definition is not None and definition != request.definition:
            raise RuntimeError("daemon definition changed while controller owns it")
        supervisor = self._supervisors.get(request.definition.daemon_key)
        if supervisor is None:
            supervisor = self._new_supervisor(request.definition)
            self._definitions[request.definition.daemon_key] = request.definition
            self._supervisors[request.definition.daemon_key] = supervisor
        if request.operation is DaemonOperation.START:
            return supervisor.start()
        if request.operation is DaemonOperation.STOP:
            return supervisor.stop()
        return supervisor.restart()

    def _new_supervisor(self, definition: DaemonDefinition) -> DaemonSupervisor:
        stem = hashlib.sha256(definition.daemon_key.encode()).hexdigest()
        return DaemonSupervisor(
            definition,
            state_path=self.state_directory / f"{stem}.json",
            lock_path=self.state_directory / f"{stem}.lock",
            **self.supervisor_options,
        )

    def _prepare_directories(self) -> None:
        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.socket_path.parent, 0o700)
        os.chmod(self.state_directory, 0o700)

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
