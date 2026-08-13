"""Fenced lifecycle operations for one directly executed daemon."""

from __future__ import annotations

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
        if min(lock_timeout, startup_probe_seconds, stop_timeout, kill_timeout) < 0:
            raise ValueError("daemon lifecycle timeouts cannot be negative")
        self.definition = definition
        self.store = DaemonLifecycleStore(state_path)
        self.lock = OwnerDiagnosticLock(lock_path, purpose=definition.daemon_key)
        self.lock_timeout = lock_timeout
        self.startup_probe_seconds = startup_probe_seconds
        self.stop_timeout = stop_timeout
        self.kill_timeout = kill_timeout

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
        child = subprocess.Popen(
            self.definition.argv,
            cwd=self.definition.cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(self.startup_probe_seconds)
        identity = capture_process_identity(child.pid)
        if child.poll() is not None or identity is None:
            self._publish(
                DaemonLifecyclePhase.FAILED,
                generation,
                detail="child exited or identity capture failed",
            )
            if child.poll() is None:
                child.terminate()
            raise DaemonLaunchError("daemon failed its startup identity probe")
        return self._publish(
            DaemonLifecyclePhase.RUNNING,
            generation,
            process_identity=identity,
        )

    def _stop_locked(self) -> DaemonLifecycleRecord:
        current = self.store.read()
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
            return self._publish(
                DaemonLifecyclePhase.STOPPED,
                current.generation,
                detail="tracked process lifetime is absent",
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
        if self._wait_absent(identity, self.stop_timeout):
            return self._publish(DaemonLifecyclePhase.STOPPED, current.generation)
        try:
            os.killpg(identity.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self._wait_absent(identity, self.kill_timeout):
            return self._publish(DaemonLifecyclePhase.STOPPED, current.generation)
        raise DaemonStopTimeoutError("daemon did not stop before the deadline")

    @staticmethod
    def _wait_absent(identity: ProcessIdentity, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = observe_process_identity(identity).state
            if state in {ProcessState.MISSING, ProcessState.ZOMBIE}:
                return True
            if state is ProcessState.UNKNOWN:
                raise DaemonIdentityUnknownError(
                    "daemon identity became unknown while stopping"
                )
            time.sleep(0.01)
        return False

    def _verified_running(
        self, current: DaemonLifecycleRecord | None
    ) -> DaemonLifecycleRecord | None:
        if current is None or current.process_identity is None:
            return None
        observation = observe_process_identity(current.process_identity)
        if observation.state is ProcessState.RUNNING:
            return current.model_copy(update={"phase": DaemonLifecyclePhase.RUNNING})
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
