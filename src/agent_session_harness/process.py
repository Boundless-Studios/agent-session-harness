"""Fresh-only child process drivers for managed coding-agent sessions."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Protocol
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import Runtime
from .secure_files import (
    atomic_write_private_text,
    exclusive_lock,
    lexical_absolute,
    private_exists,
    private_unlink,
    read_private_text,
)


_BASE_ENVIRONMENT = frozenset(
    {
        "COLORTERM",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "GIT_CONFIG_GLOBAL",
        "GH_CONFIG_DIR",
        "HOME",
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
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    }
)


class _DarwinProcessInfo(ctypes.Structure):
    """Subset of macOS ``proc_bsdinfo`` containing microsecond birth time."""

    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


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
        lowered = tuple(argument.strip().lower() for argument in self.runtime_args)
        if self.runtime is Runtime.CLAUDE:
            rejected = any(
                argument in {"--continue", "-c", "--resume"}
                or argument.startswith("--continue=")
                or argument.startswith("--resume=")
                for argument in lowered
            )
        else:
            rejected = any(
                argument in {"resume", "--last"} or argument.startswith("--last=")
                for argument in lowered
            )
        if rejected:
            raise ValueError("fresh successor launch cannot contain resume arguments")
        return self


@dataclass
class ManagedProcess:
    pid: int
    process_group_id: int
    registry_key: str
    identity: str | None = None
    command_digest: str | None = None
    launch_nonce: str | None = None
    handle: subprocess.Popen[bytes] | None = field(default=None, repr=False)


class ProcessDriver(Protocol):
    def start_fresh(self, request: LaunchRequest) -> ManagedProcess: ...

    def graceful_stop(self, process: ManagedProcess, timeout_seconds: float) -> int: ...

    def is_alive(self, process: ManagedProcess) -> bool: ...


class PosixProcessDriver:
    """Idempotent POSIX launcher keyed by chain and generation."""

    def __init__(
        self,
        state_dir: str | os.PathLike[str],
        *,
        startup_timeout_seconds: float = 2.0,
    ):
        if startup_timeout_seconds <= 0:
            raise ValueError("startup timeout must be positive")
        self.state_dir = lexical_absolute(state_dir)
        self.startup_timeout_seconds = startup_timeout_seconds

    def start_fresh(self, request: LaunchRequest) -> ManagedProcess:
        key = f"{request.chain_id}:{request.generation}"
        registry_path = self._registry_path(key)
        intent_path = registry_path.with_suffix(".intent")
        lock_path = registry_path.with_suffix(".lock")
        argv = [request.executable, *request.runtime_args]
        if request.handoff_message:
            argv.append(request.handoff_message)
        command_digest = hashlib.sha256(
            json.dumps(
                {"argv": argv, "cwd": str(request.cwd)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        environment = self._environment(request.environment)
        environment.setdefault("AGENT_SESSION_HARNESS_MANAGED", "1")
        environment.setdefault("AGENT_SESSION_HARNESS_CHAIN_ID", request.chain_id)
        environment.setdefault(
            "AGENT_SESSION_HARNESS_GENERATION",
            str(request.generation),
        )
        handle: subprocess.Popen[bytes] | None = None

        with exclusive_lock(lock_path):
            existing = self._read_registry(registry_path, key)
            if (
                existing is not None
                and existing.command_digest is not None
                and existing.command_digest != command_digest
            ):
                raise RuntimeError("managed generation already has a different command")
            if existing is not None and self.is_alive(existing):
                return existing
            if existing is not None and private_exists(registry_path):
                private_unlink(registry_path)

            intent = self._read_intent(intent_path)
            if intent is not None and intent.get("command_digest") != command_digest:
                raise RuntimeError("launch intent already has a different command")
            nonce = str(intent.get("launch_nonce")) if intent else uuid.uuid4().hex
            should_spawn = intent is None
            if intent is not None:
                age = time.time() - float(intent.get("created_at", 0))
                if age >= self.startup_timeout_seconds:
                    guardian_pid = self._guardian_pid_for_nonce(nonce)
                    if guardian_pid is not None:
                        raise RuntimeError(
                            "launch guardian exists but has not registered; refusing duplicate"
                        )
                    nonce = uuid.uuid4().hex
                    should_spawn = True
            if should_spawn:
                self._write_intent(
                    intent_path,
                    key=key,
                    command_digest=command_digest,
                    launch_nonce=nonce,
                )

        managed: ManagedProcess | None = None
        if not should_spawn:
            managed = self._await_registry(registry_path, key, nonce)
            if managed is None:
                with exclusive_lock(lock_path):
                    existing = self._read_registry(registry_path, key)
                    if existing is not None and self.is_alive(existing):
                        managed = existing
                    else:
                        current_intent = self._read_intent(intent_path)
                        if (
                            current_intent is None
                            or current_intent.get("command_digest") != command_digest
                            or current_intent.get("launch_nonce") != nonce
                        ):
                            raise RuntimeError(
                                "launch intent changed while awaiting registration"
                            )
                        if self._guardian_pid_for_nonce(nonce) is not None:
                            raise RuntimeError(
                                "launch guardian exists but has not registered; "
                                "refusing duplicate"
                            )
                        nonce = uuid.uuid4().hex
                        self._write_intent(
                            intent_path,
                            key=key,
                            command_digest=command_digest,
                            launch_nonce=nonce,
                        )
                        should_spawn = True
        if should_spawn:
            guardian_argv = [
                sys.executable,
                "-m",
                "agent_session_harness.guardian",
                "--registry",
                str(registry_path),
                "--intent",
                str(intent_path),
                "--registry-key",
                key,
                "--launch-nonce",
                nonce,
                "--command-digest",
                command_digest,
                "--cwd",
                str(request.cwd),
                "--",
                *argv,
            ]
            handle = subprocess.Popen(
                guardian_argv,
                cwd=request.cwd,
                env=environment,
                start_new_session=True,
            )
        if managed is None:
            managed = self._await_registry(registry_path, key, nonce)
        if managed is None:
            if handle is not None and handle.poll() is not None:
                raise RuntimeError(
                    f"launch guardian exited before becoming ready: {handle.returncode}"
                )
            raise RuntimeError("launch guardian did not register in time")
        if handle is not None and handle.pid == managed.pid:
            managed.handle = handle
        self._await_process_readiness(managed, registry_path)
        with exclusive_lock(lock_path):
            current_intent = self._read_intent(intent_path)
            if (
                current_intent is not None
                and current_intent.get("launch_nonce") == nonce
            ):
                private_unlink(intent_path)
        return managed

    def _await_process_readiness(
        self,
        process: ManagedProcess,
        registry_path: Path,
    ) -> None:
        deadline = time.monotonic() + min(0.1, self.startup_timeout_seconds)
        while True:
            if not self.is_alive(process):
                with exclusive_lock(registry_path.with_suffix(".lock")):
                    registered = self._read_registry(
                        registry_path,
                        process.registry_key,
                    )
                    if registered is not None and self._same_process(
                        registered, process
                    ):
                        private_unlink(registry_path)
                raise RuntimeError("launch guardian exited before becoming ready")
            if time.monotonic() >= deadline:
                return
            time.sleep(0.01)

    def graceful_stop(self, process: ManagedProcess, timeout_seconds: float) -> int:
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
        with exclusive_lock(registry_path.with_suffix(".lock")):
            registered = self._read_registry(registry_path, process.registry_key)
            if registered is not None and self._same_process(registered, process):
                private_unlink(registry_path)
        return return_code

    def is_alive(self, process: ManagedProcess) -> bool:
        if process.handle is not None:
            return process.handle.poll() is None
        if not self._pid_exists(process.pid):
            return False
        if process.identity is None:
            raise RuntimeError("restored process identity is unavailable")
        current_identity = self._process_identity(process.pid)
        if current_identity is None:
            raise RuntimeError("restored process identity cannot be verified")
        return current_identity == process.identity

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _process_identity(pid: int) -> str | None:
        birth = PosixProcessDriver._kernel_process_birth(pid)
        if birth is None:
            return None
        return hashlib.sha256(f"{pid}:{birth}".encode("utf-8")).hexdigest()

    @staticmethod
    def _kernel_process_birth(pid: int) -> str | None:
        if sys.platform.startswith("linux"):
            try:
                encoded = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            except OSError:
                return None
            command_end = encoded.rfind(")")
            if command_end < 0:
                return None
            fields = encoded[command_end + 2 :].split()
            return f"linux:{fields[19]}" if len(fields) > 19 else None
        if sys.platform == "darwin":
            return PosixProcessDriver._darwin_process_birth(pid)
        return None

    @staticmethod
    def _darwin_process_birth(pid: int) -> str | None:
        try:
            library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            proc_pidinfo = library.proc_pidinfo
            proc_pidinfo.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            proc_pidinfo.restype = ctypes.c_int
            info = _DarwinProcessInfo()
            size = ctypes.sizeof(info)
            result = proc_pidinfo(
                pid,
                3,
                0,
                ctypes.byref(info),
                size,
            )
        except (AttributeError, OSError):
            return None
        if result != size or info.pbi_pid != pid or info.pbi_start_tvsec == 0:
            return None
        return f"darwin:{info.pbi_start_tvsec}:{info.pbi_start_tvusec}"

    def _registry_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.state_dir / "processes" / f"{digest}.json"

    @staticmethod
    def _read_registry(path: Path, key: str) -> ManagedProcess | None:
        if not private_exists(path):
            return None
        try:
            payload = json.loads(read_private_text(path))
            if payload.get("registry_key") != key:
                return None
            return ManagedProcess(
                pid=int(payload["pid"]),
                process_group_id=int(payload["process_group_id"]),
                registry_key=key,
                identity=str(payload["identity"]),
                command_digest=str(payload["command_digest"]),
                launch_nonce=str(payload["launch_nonce"]),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def _await_registry(
        self,
        path: Path,
        key: str,
        nonce: str,
    ) -> ManagedProcess | None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            process = self._read_registry(path, key)
            if process is not None and process.launch_nonce == nonce:
                return process
            time.sleep(0.02)
        return None

    @staticmethod
    def _read_intent(path: Path) -> dict[str, object] | None:
        if not private_exists(path):
            return None
        try:
            payload = json.loads(read_private_text(path))
        except (json.JSONDecodeError, OSError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _write_intent(
        path: Path,
        *,
        key: str,
        command_digest: str,
        launch_nonce: str,
    ) -> None:
        _atomic_private_json(
            path,
            {
                "schema_version": 1,
                "registry_key": key,
                "command_digest": command_digest,
                "launch_nonce": launch_nonce,
                "created_at": time.time(),
            },
        )

    @staticmethod
    def _guardian_pid_for_nonce(nonce: str) -> int | None:
        try:
            completed = subprocess.run(
                ["/bin/ps", "-ax", "-o", "pid=,command="],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=2,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError("cannot verify stale launch intent") from None
        if completed.returncode != 0:
            raise RuntimeError("cannot verify stale launch intent")
        for line in completed.stdout.splitlines():
            if "agent_session_harness.guardian" not in line or nonce not in line:
                continue
            raw_pid = line.strip().split(maxsplit=1)[0]
            try:
                return int(raw_pid)
            except ValueError:
                continue
        return None

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

    @staticmethod
    def _same_process(left: ManagedProcess, right: ManagedProcess) -> bool:
        return (
            left.pid == right.pid
            and left.process_group_id == right.process_group_id
            and left.registry_key == right.registry_key
            and left.identity == right.identity
            and left.command_digest == right.command_digest
            and left.launch_nonce == right.launch_nonce
        )


def register_guarded_process(
    *,
    registry_path: Path,
    intent_path: Path,
    registry_key: str,
    command_digest: str,
    launch_nonce: str,
    process_group_id: int | None = None,
) -> ManagedProcess:
    with exclusive_lock(registry_path.with_suffix(".lock")):
        intent = PosixProcessDriver._read_intent(intent_path)
        if (
            intent is None
            or intent.get("registry_key") != registry_key
            or intent.get("command_digest") != command_digest
            or intent.get("launch_nonce") != launch_nonce
        ):
            raise RuntimeError("launch guardian intent is no longer current")
        identity = PosixProcessDriver._process_identity(os.getpid())
        if identity is None:
            raise RuntimeError("launch guardian cannot determine process identity")
        process = ManagedProcess(
            pid=os.getpid(),
            process_group_id=(
                os.getpgrp() if process_group_id is None else process_group_id
            ),
            registry_key=registry_key,
            identity=identity,
            command_digest=command_digest,
            launch_nonce=launch_nonce,
        )
        _atomic_private_json(
            registry_path,
            {
                "pid": process.pid,
                "process_group_id": process.process_group_id,
                "registry_key": process.registry_key,
                "identity": process.identity,
                "command_digest": process.command_digest,
                "launch_nonce": process.launch_nonce,
            },
        )
    return process


def unregister_guarded_process(
    *,
    registry_path: Path,
    process: ManagedProcess,
) -> None:
    """Remove only the registry record still owned by this guardian."""

    with exclusive_lock(registry_path.with_suffix(".lock")):
        registered = PosixProcessDriver._read_registry(
            registry_path,
            process.registry_key,
        )
        if registered is not None and PosixProcessDriver._same_process(
            registered,
            process,
        ):
            private_unlink(registry_path)


def _atomic_private_json(path: Path, payload: dict[str, object]) -> None:
    atomic_write_private_text(
        path,
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    )
