"""Bounded JSON protocol for executable checkpoint adapters."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agent_session_harness.capsule import HandoffCapsule

_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(?:[a-z0-9]+[-_])*(?:api[-_]?key|authorization|credential|password|secret|token)"
    r"(?:[-_][a-z0-9]+)*"
    r"\s*[:=]\s*\S+"
)
_INHERITED_ENVIRONMENT_KEYS = frozenset(
    {
        "CURL_CA_BUNDLE",
        "BD_NO_DAEMON",
        "BEADS_ACTOR",
        "BEADS_DB",
        "BEADS_DIR",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "NO_COLOR",
        "NO_PROXY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SHELL",
        "SOPS_AGE_KEY_FILE",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TERM",
        "TMPDIR",
        "TZ",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "XDG_STATE_HOME",
    }
)


def _controlled_environment(
    overrides: Mapping[str, str],
    inherit_env: tuple[str, ...],
) -> dict[str, str]:
    """Build the small environment required by local checkpoint adapters."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key in _INHERITED_ENVIRONMENT_KEYS
    }
    environment.update(
        {key: os.environ[key] for key in inherit_env if key in os.environ}
    )
    environment.update(overrides)
    return environment


def sanitize_error(value: str, *, max_length: int = 240) -> str:
    """Bound an adapter diagnostic and remove assignment-shaped credentials."""

    printable = "".join(
        character if character.isprintable() else " " for character in value
    )
    collapsed = " ".join(printable.split())
    redacted = _SENSITIVE_ASSIGNMENT.sub("credential=[redacted]", collapsed)
    return redacted[:max_length] or "adapter reported failure"


class AdapterOperation(str, Enum):
    WRITE = "write"
    READ = "read"
    ACKNOWLEDGE = "acknowledge"


class AdapterRequest(BaseModel):
    """One versioned idempotent checkpoint adapter request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    operation: AdapterOperation
    idempotency_key: str = Field(min_length=1, max_length=256)
    capsule: HandoffCapsule

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


class AdapterResponse(BaseModel):
    """The exact normalized response returned by every checkpoint adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: bool
    fingerprint: str | None = Field(max_length=64)
    retryable: bool
    error: str | None

    @field_validator("error")
    @classmethod
    def bound_error(cls, value: str | None) -> str | None:
        return None if value is None else sanitize_error(value)


class CheckpointAdapter(Protocol):
    name: str

    def execute(self, request: AdapterRequest) -> AdapterResponse: ...


@dataclass(frozen=True)
class _BoundedResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_overflow: bool
    stderr_overflow: bool


def _run_bounded(
    argv: tuple[str, ...],
    *,
    request: bytes,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    environment: Mapping[str, str],
) -> _BoundedResult:
    process = subprocess.Popen(
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=dict(environment),
        start_new_session=True,
    )
    stdout = bytearray()
    stderr = bytearray()
    stdout_overflow = threading.Event()
    stderr_overflow = threading.Event()

    def drain(
        stream,
        target: bytearray,
        limit: int,
        overflow: threading.Event,
    ) -> None:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            remaining = limit + 1 - len(target)
            if remaining > 0:
                target.extend(chunk[:remaining])
            if len(chunk) > remaining or len(target) > limit:
                overflow.set()

    def write_request() -> None:
        assert process.stdin is not None
        try:
            process.stdin.write(request)
            process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            process.stdin.close()

    assert process.stdout is not None
    assert process.stderr is not None
    threads = (
        threading.Thread(
            target=drain,
            args=(process.stdout, stdout, max_stdout_bytes, stdout_overflow),
            daemon=True,
        ),
        threading.Thread(
            target=drain,
            args=(process.stderr, stderr, max_stderr_bytes, stderr_overflow),
            daemon=True,
        ),
        threading.Thread(target=write_request, daemon=True),
    )
    for thread in threads:
        thread.start()

    deadline = time.monotonic() + timeout_seconds
    while process.poll() is None:
        if stdout_overflow.is_set() or stderr_overflow.is_set():
            _kill_process_group(process)
            break
        if time.monotonic() >= deadline:
            _kill_process_group(process)
            for thread in threads:
                thread.join(timeout=1)
            raise subprocess.TimeoutExpired(list(argv), timeout_seconds)
        time.sleep(0.005)
    for thread in threads:
        thread.join(timeout=1)
    return _BoundedResult(
        returncode=int(process.wait()),
        stdout=bytes(stdout[:max_stdout_bytes]),
        stderr=bytes(stderr[:max_stderr_bytes]),
        stdout_overflow=stdout_overflow.is_set(),
        stderr_overflow=stderr_overflow.is_set(),
    )


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        process.kill()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


@dataclass(frozen=True)
class JsonCommand:
    """Invoke a generic bounded JSON adapter without shell interpolation."""

    name: str
    argv: tuple[str, ...]
    timeout_seconds: float = 5.0
    max_response_bytes: int = 1024 * 1024
    max_stderr_bytes: int = 64 * 1024
    env: Mapping[str, str] | None = None
    inherit_env: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("command name must not be blank")
        if not self.argv or any(not argument for argument in self.argv):
            raise ValueError("command argv must contain non-empty arguments")
        if self.timeout_seconds <= 0:
            raise ValueError("command timeout must be positive")
        if self.max_response_bytes <= 0:
            raise ValueError("command response bound must be positive")
        if self.max_stderr_bytes <= 0:
            raise ValueError("command diagnostic bound must be positive")
        if any(not key or "=" in key for key in self.inherit_env):
            raise ValueError("inherited environment keys must be valid names")

    def execute(self, payload: Mapping[str, object]) -> dict[str, object]:
        request = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        try:
            completed = _run_bounded(
                self.argv,
                request=request,
                timeout_seconds=self.timeout_seconds,
                max_stdout_bytes=self.max_response_bytes,
                max_stderr_bytes=self.max_stderr_bytes,
                environment=_controlled_environment(
                    self.env or {},
                    self.inherit_env,
                ),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{self.name} timed out") from exc
        except OSError as exc:
            raise RuntimeError(f"{self.name} could not be executed") from exc

        if completed.stdout_overflow:
            raise RuntimeError(
                f"{self.name} response exceeded {self.max_response_bytes} bytes"
            )
        if completed.stderr_overflow:
            raise RuntimeError(
                f"{self.name} diagnostic output exceeded {self.max_stderr_bytes} bytes"
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{self.name} exited with status {completed.returncode}")
        try:
            response = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{self.name} returned malformed JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"{self.name} response must be a JSON object")
        return response


def _failure(error: str, *, retryable: bool) -> AdapterResponse:
    return AdapterResponse(
        ok=False,
        fingerprint=None,
        retryable=retryable,
        error=error,
    )


@dataclass(frozen=True)
class CommandAdapter:
    """Invoke one executable adapter directly, without shell interpolation."""

    name: str
    argv: tuple[str, ...]
    timeout_seconds: float = 5.0
    max_response_bytes: int = 64 * 1024
    max_stderr_bytes: int = 64 * 1024
    env: Mapping[str, str] | None = None
    inherit_env: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("adapter name must not be blank")
        if not self.argv or any(not argument for argument in self.argv):
            raise ValueError("adapter argv must contain non-empty arguments")
        if self.timeout_seconds <= 0:
            raise ValueError("adapter timeout must be positive")
        if self.max_response_bytes <= 0:
            raise ValueError("adapter response bound must be positive")
        if self.max_stderr_bytes <= 0:
            raise ValueError("adapter diagnostic bound must be positive")
        if any(not key or "=" in key for key in self.inherit_env):
            raise ValueError("inherited environment keys must be valid names")

    def execute(self, request: AdapterRequest) -> AdapterResponse:
        try:
            completed = _run_bounded(
                self.argv,
                request=request.canonical_bytes(),
                timeout_seconds=self.timeout_seconds,
                max_stdout_bytes=self.max_response_bytes,
                max_stderr_bytes=self.max_stderr_bytes,
                environment=_controlled_environment(
                    self.env or {},
                    self.inherit_env,
                ),
            )
        except subprocess.TimeoutExpired:
            return _failure("adapter timed out", retryable=True)
        except OSError:
            return _failure("adapter could not be executed", retryable=False)

        if completed.stdout_overflow:
            return _failure(
                f"adapter response exceeded {self.max_response_bytes} bytes",
                retryable=False,
            )
        if completed.stderr_overflow:
            return _failure(
                f"adapter diagnostic output exceeded {self.max_stderr_bytes} bytes",
                retryable=False,
            )
        if completed.returncode != 0:
            return _failure(
                f"adapter exited with status {completed.returncode}",
                retryable=True,
            )
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
            response = AdapterResponse.model_validate(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError):
            return _failure("adapter returned malformed JSON", retryable=False)

        if response.ok and response.fingerprint != request.capsule.fingerprint:
            return _failure("adapter returned the wrong fingerprint", retryable=False)
        return response
