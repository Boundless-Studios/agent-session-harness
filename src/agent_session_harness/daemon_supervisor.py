"""Fenced lifecycle operations for one directly executed daemon."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from .daemon_lifecycle import (
    DaemonDefinition,
    DaemonLifecyclePhase,
    DaemonLifecycleRecord,
    DaemonLifecycleStore,
    OwnerDiagnosticLock,
)
from .process_identity import (
    ProcessIdentity,
    ProcessState,
    capture_process_identity,
    observe_process_identity,
)


class DaemonLaunchError(RuntimeError):
    """A child could not become a verified running daemon."""


class DaemonIdentityUnknownError(RuntimeError):
    """The tracked process lifetime cannot be verified safely."""


class DaemonStopTimeoutError(TimeoutError):
    """A verified daemon did not stop within its bounded deadline."""


class DaemonSupervisor:
    def __init__(
        self,
        definition: DaemonDefinition,
        *,
        state_path: str | Path,
        lock_path: str | Path,
        lock_timeout: float = 5,
        startup_probe_seconds: float = 0.1,
        stop_timeout: float = 5,
        kill_timeout: float = 2,
    ) -> None:
        timeouts = (
            lock_timeout,
            startup_probe_seconds,
            stop_timeout,
            kill_timeout,
        )
        if not all(math.isfinite(value) for value in timeouts):
            raise ValueError("daemon lifecycle timeouts must be finite")
        if min(timeouts) < 0:
            raise ValueError("daemon lifecycle timeouts cannot be negative")
        if stop_timeout == 0 or kill_timeout == 0:
            raise ValueError("daemon stop and kill timeouts must be positive")
        self.definition = definition
        self.store = DaemonLifecycleStore(state_path)
        self.lock = OwnerDiagnosticLock(lock_path, purpose=definition.daemon_key)
        self.lock_timeout = lock_timeout
        self.startup_probe_seconds = startup_probe_seconds
        self.stop_timeout = stop_timeout
        self.kill_timeout = kill_timeout
        self._child: subprocess.Popen[bytes] | None = None

    def start(self) -> DaemonLifecycleRecord:
        with self.lock.acquire(timeout=self.lock_timeout):
            current = self.store.read()
            running = self._verified_running(current)
            if running is not None:
                return running
            generation = (current.generation if current is not None else 0) + 1
            return self._start_locked(generation)

    def stop(self) -> DaemonLifecycleRecord:
        with self.lock.acquire(timeout=self.lock_timeout):
            return self._stop_locked()

    def restart(self) -> DaemonLifecycleRecord:
        with self.lock.acquire(timeout=self.lock_timeout):
            current = self.store.read()
            prior_generation = current.generation if current is not None else 0
            self._stop_locked()
            return self._start_locked(prior_generation + 1)

    def _start_locked(self, generation: int) -> DaemonLifecycleRecord:
        self._publish(DaemonLifecyclePhase.STARTING, generation)
        try:
            child = subprocess.Popen(
                self.definition.argv,
                cwd=self.definition.cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self._publish(
                DaemonLifecyclePhase.FAILED,
                generation,
                detail=f"process creation failed: {type(exc).__name__}",
            )
            raise DaemonLaunchError("daemon process creation failed") from exc
        self._child = child
        time.sleep(self.startup_probe_seconds)
        identity = capture_process_identity(child.pid)
        if identity is None:
            self._terminate_owned_child(child)
            self._publish(
                DaemonLifecyclePhase.FAILED,
                generation,
                detail="child exited or identity capture failed",
            )
            raise DaemonLaunchError("daemon failed its startup identity probe")
        try:
            return self._publish(
                DaemonLifecyclePhase.RUNNING,
                generation,
                process_identity=identity,
            )
        except BaseException:
            self._terminate_owned_child(child)
            raise

    def _stop_locked(self) -> DaemonLifecycleRecord:
        current = self.store.read()
        self._require_matching_daemon(current)
        if current is None:
            return self._publish(DaemonLifecyclePhase.STOPPED, 0)
        identity = current.process_identity
        if identity is None:
            return self._publish(DaemonLifecyclePhase.STOPPED, current.generation)
        observation = observe_process_identity(identity)
        if observation.state is ProcessState.UNKNOWN:
            raise DaemonIdentityUnknownError(
                "daemon process identity is unknown; refusing to signal"
            )
        if observation.state in {ProcessState.MISSING, ProcessState.ZOMBIE}:
            child = self._owned_child(identity.pid)
            if child is not None and self._process_group_has_live_members(
                child.pid, timeout=min(self.stop_timeout, 0.25)
            ):
                self._terminate_owned_child(child)
            else:
                self._reap_owned_child(identity.pid)
            return self._publish(
                DaemonLifecyclePhase.STOPPED,
                current.generation,
                detail="tracked process lifetime is absent",
            )

        child = self._owned_child(identity.pid)
        if child is None:
            raise DaemonIdentityUnknownError(
                "daemon is owned by another controller; refusing PID-only signal"
            )
        self._publish(
            DaemonLifecyclePhase.STOPPING,
            current.generation,
            process_identity=identity,
        )
        try:
            os.killpg(identity.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if self._wait_group_absent(child, self.stop_timeout):
            return self._publish(DaemonLifecyclePhase.STOPPED, current.generation)
        try:
            os.killpg(identity.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self._wait_group_absent(child, self.kill_timeout):
            return self._publish(DaemonLifecyclePhase.STOPPED, current.generation)
        self._publish(
            DaemonLifecyclePhase.FAILED,
            current.generation,
            detail="process group remained after stop deadlines",
            process_identity=identity,
        )
        raise DaemonStopTimeoutError("daemon did not stop before the deadline")

    @staticmethod
    def _wait_group_absent(child: subprocess.Popen[bytes], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if not DaemonSupervisor._process_group_has_live_members(
                child.pid, timeout=min(remaining, 0.25)
            ):
                child.wait(timeout=0)
                return True
            time.sleep(0.01)

    @staticmethod
    def _process_group_has_live_members(
        process_group_id: int, *, timeout: float
    ) -> bool:
        try:
            result = subprocess.run(
                ("/bin/ps", "-axo", "pgid=,stat="),
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return True
        for line in result.stdout.splitlines():
            fields = line.split()
            if (
                len(fields) >= 2
                and fields[0] == str(process_group_id)
                and not fields[1].startswith("Z")
            ):
                return True
        return False

    def _owned_child(self, pid: int) -> subprocess.Popen[bytes] | None:
        child = self._child
        return child if child is not None and child.pid == pid else None

    def _reap_owned_child(self, pid: int) -> None:
        child = self._owned_child(pid)
        if child is not None and child.poll() is not None:
            child.wait()
            self._child = None

    def _terminate_owned_child(self, child: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait()
            return
        if not self._wait_group_absent(child, self.stop_timeout):
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not self._wait_group_absent(child, self.kill_timeout):
                raise DaemonStopTimeoutError(
                    "failed launch process group survived cleanup deadlines"
                )
        self._child = None

    def _require_matching_daemon(self, current: DaemonLifecycleRecord | None) -> None:
        if current is not None and current.daemon_key != self.definition.daemon_key:
            raise RuntimeError("lifecycle state belongs to a different daemon")

    def _verified_running(
        self, current: DaemonLifecycleRecord | None
    ) -> DaemonLifecycleRecord | None:
        self._require_matching_daemon(current)
        if current is None or current.process_identity is None:
            return None
        observation = observe_process_identity(current.process_identity)
        if observation.state is ProcessState.RUNNING:
            normalized = current.model_copy(
                update={"phase": DaemonLifecyclePhase.RUNNING}
            )
            self.store.publish(normalized)
            return normalized
        if observation.state is ProcessState.UNKNOWN:
            raise DaemonIdentityUnknownError(
                "daemon process identity is unknown; refusing to start"
            )
        return None

    def _publish(
        self,
        phase: DaemonLifecyclePhase,
        generation: int,
        *,
        detail: str | None = None,
        process_identity: ProcessIdentity | None = None,
    ) -> DaemonLifecycleRecord:
        record = DaemonLifecycleRecord(
            daemon_key=self.definition.daemon_key,
            phase=phase,
            generation=generation,
            changed_at=datetime.now(UTC),
            detail=detail,
            process_identity=process_identity,
        )
        self.store.publish(record)
        return record
