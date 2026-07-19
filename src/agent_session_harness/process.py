"""Fresh-only child process drivers for managed coding-agent sessions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Iterator, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import Runtime


_BASE_ENVIRONMENT = frozenset(
    {
        "COLORTERM",
        "LANG",
        "LC_ALL",
        "NO_COLOR",
        "PATH",
        "SHELL",
        "SSH_AUTH_SOCK",
        "TERM",
        "TERM_PROGRAM",
        "TMPDIR",
        "USER",
    }
)


class LaunchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: Runtime
    chain_id: str = Field(min_length=1)
    generation: int = Field(ge=0)
    cwd: Path
    executable: str = Field(min_length=1)
    runtime_args: tuple[str, ...] = ()
    environment: dict[str, str] = Field(default_factory=dict)
    capsule_path: Path | None = None
    capsule_fingerprint: str | None = None
    handoff_message: str | None = Field(default=None, max_length=500)

    @field_validator("cwd", mode="before")
    @classmethod
    def absolute_cwd(cls, value: object) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            raise ValueError("launch cwd must be absolute")
        return path.resolve()

    @model_validator(mode="after")
    def reject_resume_arguments(self) -> LaunchRequest:
        lowered = {argument.strip().lower() for argument in self.runtime_args}
        forbidden = (
            {"--continue", "-c", "--resume"}
            if self.runtime is Runtime.CLAUDE
            else {"resume", "--last"}
        )
        if lowered & forbidden:
            raise ValueError("fresh successor launch cannot contain resume arguments")
        return self


@dataclass
class ManagedProcess:
    pid: int
    process_group_id: int
    registry_key: str
    handle: subprocess.Popen[bytes] | None = field(default=None, repr=False)


class ProcessDriver(Protocol):
    def start_fresh(self, request: LaunchRequest) -> ManagedProcess: ...

    def graceful_stop(
        self, process: ManagedProcess, timeout_seconds: float
    ) -> int: ...

    def is_alive(self, process: ManagedProcess) -> bool: ...


class PosixProcessDriver:
    """Idempotent POSIX launcher keyed by chain and generation."""

    def __init__(self, state_dir: str | os.PathLike[str]):
        self.state_dir = Path(state_dir).expanduser()

    def start_fresh(self, request: LaunchRequest) -> ManagedProcess:
        key = f"{request.chain_id}:{request.generation}"
        registry_path = self._registry_path(key)
        with _file_lock(registry_path.with_suffix(".lock")):
            existing = self._read_registry(registry_path, key)
            if existing is not None and self.is_alive(existing):
                return existing

            argv = [request.executable, *request.runtime_args]
            if request.handoff_message:
                argv.append(request.handoff_message)
            environment = self._environment(request.environment)
            process = subprocess.Popen(
                argv,
                cwd=request.cwd,
                env=environment,
                start_new_session=True,
            )
            managed = ManagedProcess(
                pid=process.pid,
                process_group_id=process.pid,
                registry_key=key,
                handle=process,
            )
            self._write_registry(registry_path, managed)
            return managed

    def graceful_stop(
        self, process: ManagedProcess, timeout_seconds: float
    ) -> int:
        if timeout_seconds < 0:
            raise ValueError("stop timeout must be non-negative")
        if self.is_alive(process):
            try:
                os.killpg(process.process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout_seconds
        while self.is_alive(process) and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.is_alive(process):
            try:
                os.killpg(process.process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return_code = 0
        if process.handle is not None:
            try:
                return_code = process.handle.wait(timeout=1)
            except subprocess.TimeoutExpired:
                return_code = -signal.SIGKILL
        registry_path = self._registry_path(process.registry_key)
        with _file_lock(registry_path.with_suffix(".lock")):
            if registry_path.exists():
                registry_path.unlink()
        return return_code

    def is_alive(self, process: ManagedProcess) -> bool:
        if process.handle is not None:
            return process.handle.poll() is None
        try:
            os.kill(process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _registry_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.state_dir / "processes" / f"{digest}.json"

    @staticmethod
    def _read_registry(path: Path, key: str) -> ManagedProcess | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("registry_key") != key:
                return None
            return ManagedProcess(
                pid=int(payload["pid"]),
                process_group_id=int(payload["process_group_id"]),
                registry_key=key,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _write_registry(path: Path, process: ManagedProcess) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            {
                "pid": process.pid,
                "process_group_id": process.process_group_id,
                "registry_key": process.registry_key,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.chmod(path, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _environment(overrides: dict[str, str]) -> dict[str, str]:
        invalid = [
            key
            for key in overrides
            if key not in _BASE_ENVIRONMENT
            and not key.startswith("AGENT_SESSION_HARNESS_")
        ]
        if invalid:
            raise ValueError(f"launch environment key is not allowlisted: {invalid[0]}")
        environment = {
            key: value for key, value in os.environ.items() if key in _BASE_ENVIRONMENT
        }
        environment.update(overrides)
        return environment


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    os.chmod(path, 0o600)
    with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError("exclusive file locking is unavailable") from exc
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
