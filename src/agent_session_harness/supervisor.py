"""Crash-recoverable supervision for exactly-once fresh-session rotation."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
import hmac
import json
import os
from pathlib import Path
import tempfile
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .activity import ActivitySnapshot, Quiescence
from .coordinator import ClaimHandle, CoordinatorAdapter
from .models import Confidence, Runtime
from .process import LaunchRequest, ManagedProcess, ProcessDriver


class SupervisorPhase(str, Enum):
    INITIAL = "initial"
    RUNNING = "running"
    WARNING = "warning"
    DRAINING = "draining"
    CHECKPOINTING = "checkpointing"
    CHECKPOINTED = "checkpointed"
    FENCING = "fencing"
    FENCED = "fenced"
    STOPPING = "stopping"
    STOPPED = "stopped"
    CLAIMING = "claiming"
    LAUNCHING = "launching"
    AWAITING_ACK = "awaiting_ack"


class UsageObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str = Field(min_length=1)
    context_percent: float = Field(ge=0)
    confidence: Confidence


class CheckpointRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chain_id: str
    predecessor_conversation_id: str
    target_generation: int
    idempotency_key: str


class VerifiedCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verified: bool
    fingerprint: str = Field(min_length=1)
    path: Path


class AcknowledgementRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    generation: int = Field(ge=0)
    fingerprint: str = Field(min_length=1, max_length=64)
    conversation_id: str = Field(min_length=1, max_length=160)


class UsageReader(Protocol):
    def sample(self, process: ManagedProcess) -> UsageObservation: ...


class CheckpointManager(Protocol):
    def checkpoint(self, request: CheckpointRequest) -> VerifiedCheckpoint: ...


class SupervisorSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    runtime: Runtime
    chain_id: str
    generation: int = Field(default=0, ge=0)
    phase: SupervisorPhase = SupervisorPhase.INITIAL
    owner_session_id: str | None = None
    conversation_id: str | None = None
    claim: ClaimHandle | None = None
    process_pid: int | None = None
    process_group_id: int | None = None
    process_registry_key: str | None = None
    context_percent: float | None = Field(default=None, ge=0)
    context_confidence: Confidence = Confidence.UNKNOWN
    checkpoint_fingerprint: str | None = None
    checkpoint_path: Path | None = None
    warning_emitted: bool = False


class Supervisor:
    def __init__(
        self,
        *,
        runtime: str | Runtime,
        chain_id: str,
        cwd: str | os.PathLike[str],
        task_type: str,
        task_id: str,
        task_fingerprint: str,
        executable: str,
        runtime_args: tuple[str, ...],
        state_path: str | os.PathLike[str],
        process_driver: ProcessDriver,
        usage_reader: UsageReader,
        checkpoint_manager: CheckpointManager,
        coordinator: CoordinatorAdapter,
        warn_percent: float = 65.0,
        rotate_percent: float = 70.0,
        lease_seconds: int = 60,
        stop_timeout_seconds: float = 10.0,
    ):
        if not 0 < warn_percent < rotate_percent <= 100:
            raise ValueError("thresholds must satisfy 0 < warn < rotate <= 100")
        self.runtime = Runtime(runtime)
        self.chain_id = chain_id
        self.cwd = Path(cwd).expanduser().resolve()
        self.task_type = task_type
        self.task_id = task_id
        self.task_fingerprint = task_fingerprint
        self.executable = executable
        self.runtime_args = runtime_args
        self.state_path = Path(state_path)
        self.effect_path = self.state_path.with_suffix(
            self.state_path.suffix + ".events"
        )
        self.process_driver = process_driver
        self.usage_reader = usage_reader
        self.checkpoint_manager = checkpoint_manager
        self.coordinator = coordinator
        self.warn_percent = warn_percent
        self.rotate_percent = rotate_percent
        self.lease_seconds = lease_seconds
        self.stop_timeout_seconds = stop_timeout_seconds
        self.snapshot = self._load()
        self.current_process = self._restore_process(self.snapshot)

    @property
    def can_dispatch(self) -> bool:
        return self.snapshot.phase in {
            SupervisorPhase.RUNNING,
            SupervisorPhase.WARNING,
        }

    def start(self) -> SupervisorSnapshot:
        if self.snapshot.phase is not SupervisorPhase.INITIAL:
            return self.snapshot
        owner_session_id = self._owner_session_id(0)
        self._effect("claim", "started", generation=0)
        claim = self._claim(owner_session_id)
        self._effect("claim", "completed", generation=0)
        request = self._launch_request(generation=0)
        self._effect("launch", "started", generation=0)
        process = self.process_driver.start_fresh(request)
        self._effect("launch", "completed", generation=0)
        self.current_process = process
        self.snapshot = self.snapshot.model_copy(
            update={
                "generation": 0,
                "phase": SupervisorPhase.RUNNING,
                "owner_session_id": owner_session_id,
                "claim": claim,
                **self._process_fields(process),
            }
        )
        self._persist()
        return self.snapshot

    def tick(self, activity: ActivitySnapshot) -> SupervisorSnapshot:
        if self.snapshot.phase is SupervisorPhase.INITIAL:
            raise RuntimeError("supervisor must be started before ticking")
        if self.snapshot.phase is SupervisorPhase.AWAITING_ACK:
            acknowledgement = read_acknowledgement(self.state_path)
            if acknowledgement is None:
                return self.snapshot
            snapshot = self.acknowledge(
                generation=acknowledgement.generation,
                fingerprint=acknowledgement.fingerprint,
                conversation_id=acknowledgement.conversation_id,
            )
            clear_acknowledgement(self.state_path)
            return snapshot
        if self.snapshot.phase in {
            SupervisorPhase.RUNNING,
            SupervisorPhase.WARNING,
            SupervisorPhase.DRAINING,
        }:
            self._observe_usage()
            if self.snapshot.phase is SupervisorPhase.DRAINING:
                if activity.quiescence is not Quiescence.IDLE:
                    return self.snapshot
                self._set_phase(SupervisorPhase.CHECKPOINTING)
        return self._advance_rotation()

    def acknowledge(
        self,
        *,
        generation: int,
        fingerprint: str | None,
        conversation_id: str,
    ) -> SupervisorSnapshot:
        if self.snapshot.phase is not SupervisorPhase.AWAITING_ACK:
            raise ValueError("no successor acknowledgement is expected")
        if generation != self.snapshot.generation:
            raise ValueError("acknowledgement generation does not match")
        if fingerprint != self.snapshot.checkpoint_fingerprint:
            raise ValueError("acknowledgement fingerprint does not match")
        self._effect("acknowledge", "started", generation=generation)
        self.snapshot = self.snapshot.model_copy(
            update={
                "phase": SupervisorPhase.RUNNING,
                "conversation_id": conversation_id,
                "context_percent": None,
                "context_confidence": Confidence.UNKNOWN,
                "warning_emitted": False,
            }
        )
        self._persist()
        self._effect("acknowledge", "completed", generation=generation)
        return self.snapshot

    def _observe_usage(self) -> None:
        if self.current_process is None:
            raise RuntimeError("managed process metadata is unavailable")
        sample = self.usage_reader.sample(self.current_process)
        self.snapshot = self.snapshot.model_copy(
            update={
                "conversation_id": sample.conversation_id,
                "context_percent": sample.context_percent,
                "context_confidence": sample.confidence,
            }
        )
        if sample.confidence is not Confidence.CONFIDENT:
            self._persist()
            return
        if (
            self.snapshot.phase is SupervisorPhase.RUNNING
            and sample.context_percent >= self.warn_percent
        ):
            self.snapshot = self.snapshot.model_copy(
                update={
                    "phase": SupervisorPhase.WARNING,
                    "warning_emitted": True,
                }
            )
        if (
            self.snapshot.phase in {SupervisorPhase.RUNNING, SupervisorPhase.WARNING}
            and sample.context_percent >= self.rotate_percent
        ):
            self.snapshot = self.snapshot.model_copy(
                update={"phase": SupervisorPhase.DRAINING}
            )
        self._persist()

    def _advance_rotation(self) -> SupervisorSnapshot:
        while True:
            phase = self.snapshot.phase
            if phase is SupervisorPhase.CHECKPOINTING:
                predecessor = self.snapshot.conversation_id
                if predecessor is None:
                    return self.snapshot
                target = self.snapshot.generation + 1
                request = CheckpointRequest(
                    chain_id=self.chain_id,
                    predecessor_conversation_id=predecessor,
                    target_generation=target,
                    idempotency_key=f"{self.chain_id}:{target}",
                )
                self._effect("checkpoint", "started", generation=target)
                receipt = self.checkpoint_manager.checkpoint(request)
                if not receipt.verified:
                    self._effect("checkpoint", "failed", generation=target)
                    return self.snapshot
                self._effect("checkpoint", "completed", generation=target)
                self.snapshot = self.snapshot.model_copy(
                    update={
                        "phase": SupervisorPhase.CHECKPOINTED,
                        "checkpoint_fingerprint": receipt.fingerprint,
                        "checkpoint_path": receipt.path,
                    }
                )
                self._persist()
                continue
            if phase is SupervisorPhase.CHECKPOINTED:
                self._set_phase(SupervisorPhase.FENCING)
                continue
            if phase is SupervisorPhase.FENCING:
                claim = self._required_claim()
                self._effect("fence", "started", generation=self.snapshot.generation)
                self.coordinator.fence(claim)
                self._effect("fence", "completed", generation=self.snapshot.generation)
                self._set_phase(SupervisorPhase.FENCED)
                continue
            if phase is SupervisorPhase.FENCED:
                self._set_phase(SupervisorPhase.STOPPING)
                continue
            if phase is SupervisorPhase.STOPPING:
                process = self._required_process()
                self._effect("stop", "started", generation=self.snapshot.generation)
                self.process_driver.graceful_stop(process, self.stop_timeout_seconds)
                if self.process_driver.is_alive(process):
                    raise RuntimeError("predecessor process remains live after stop")
                self._effect("stop", "completed", generation=self.snapshot.generation)
                self.current_process = None
                self.snapshot = self.snapshot.model_copy(
                    update={
                        "phase": SupervisorPhase.STOPPED,
                        "process_pid": None,
                        "process_group_id": None,
                        "process_registry_key": None,
                    }
                )
                self._persist()
                continue
            if phase is SupervisorPhase.STOPPED:
                self._set_phase(SupervisorPhase.CLAIMING)
                continue
            if phase is SupervisorPhase.CLAIMING:
                target = self.snapshot.generation + 1
                owner_session_id = self._owner_session_id(target)
                self._effect("claim", "started", generation=target)
                claim = self._claim(owner_session_id)
                self._effect("claim", "completed", generation=target)
                self.snapshot = self.snapshot.model_copy(
                    update={
                        "generation": target,
                        "phase": SupervisorPhase.LAUNCHING,
                        "owner_session_id": owner_session_id,
                        "conversation_id": None,
                        "claim": claim,
                    }
                )
                self._persist()
                continue
            if phase is SupervisorPhase.LAUNCHING:
                request = self._launch_request(generation=self.snapshot.generation)
                self._effect("launch", "started", generation=self.snapshot.generation)
                process = self.process_driver.start_fresh(request)
                self._effect("launch", "completed", generation=self.snapshot.generation)
                self.current_process = process
                self.snapshot = self.snapshot.model_copy(
                    update={
                        "phase": SupervisorPhase.AWAITING_ACK,
                        **self._process_fields(process),
                    }
                )
                self._persist()
                continue
            return self.snapshot

    def _claim(self, owner_session_id: str) -> ClaimHandle:
        return self.coordinator.claim(
            task_type=self.task_type,
            task_id=self.task_id,
            fingerprint=self.task_fingerprint,
            owner_session_id=owner_session_id,
            owner_pid=os.getpid(),
            runtime=self.runtime,
            worktree_path=str(self.cwd),
            lease_seconds=self.lease_seconds,
        )

    def _launch_request(self, *, generation: int) -> LaunchRequest:
        environment = {
            "AGENT_SESSION_HARNESS_MANAGED": "1",
            "AGENT_SESSION_HARNESS_CHAIN_ID": self.chain_id,
            "AGENT_SESSION_HARNESS_GENERATION": str(generation),
            "AGENT_SESSION_HARNESS_LEDGER": str(
                self.state_path.with_suffix(self.state_path.suffix + ".lifecycle")
            ),
            "AGENT_SESSION_HARNESS_STATE_PATH": str(self.state_path),
        }
        message = None
        if generation > 0:
            if (
                self.snapshot.checkpoint_path is None
                or self.snapshot.checkpoint_fingerprint is None
            ):
                raise RuntimeError("successor launch requires a verified checkpoint")
            environment.update(
                {
                    "AGENT_SESSION_HARNESS_CAPSULE_PATH": str(
                        self.snapshot.checkpoint_path
                    ),
                    "AGENT_SESSION_HARNESS_CAPSULE_FINGERPRINT": (
                        self.snapshot.checkpoint_fingerprint
                    ),
                    "AGENT_SESSION_HARNESS_TARGET_GENERATION": str(generation),
                }
            )
            message = (
                "Read and acknowledge the verified handoff capsule, then continue "
                "its exact next action."
            )
        return LaunchRequest(
            runtime=self.runtime,
            chain_id=self.chain_id,
            generation=generation,
            cwd=self.cwd,
            executable=self.executable,
            runtime_args=self.runtime_args,
            environment=environment,
            capsule_path=self.snapshot.checkpoint_path,
            capsule_fingerprint=self.snapshot.checkpoint_fingerprint,
            handoff_message=message,
        )

    def _set_phase(self, phase: SupervisorPhase) -> None:
        self.snapshot = self.snapshot.model_copy(update={"phase": phase})
        self._persist()

    def _load(self) -> SupervisorSnapshot:
        if not self.state_path.exists():
            return SupervisorSnapshot(runtime=self.runtime, chain_id=self.chain_id)
        with _lock(self.state_path.with_suffix(self.state_path.suffix + ".lock")):
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        snapshot = SupervisorSnapshot.model_validate(payload)
        if snapshot.runtime is not self.runtime or snapshot.chain_id != self.chain_id:
            raise ValueError("supervisor state does not match requested runtime/chain")
        return snapshot

    def _persist(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with _lock(self.state_path.with_suffix(self.state_path.suffix + ".lock")):
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.state_path.parent,
                prefix=f".{self.state_path.name}.",
                text=True,
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            self.snapshot.model_dump(mode="json"),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary_path, 0o600)
                os.replace(temporary_path, self.state_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

    def _effect(self, effect: str, status: str, *, generation: int) -> None:
        payload = {
            "schema_version": 1,
            "effect": effect,
            "status": status,
            "chain_id": self.chain_id,
            "generation": generation,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.effect_path.parent.mkdir(parents=True, exist_ok=True)
        with _lock(self.effect_path.with_suffix(self.effect_path.suffix + ".lock")):
            descriptor = os.open(
                self.effect_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            os.chmod(self.effect_path, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _owner_session_id(self, generation: int) -> str:
        return f"{self.chain_id}:{generation}"

    def _required_claim(self) -> ClaimHandle:
        if self.snapshot.claim is None:
            raise RuntimeError("coordinator claim metadata is unavailable")
        return self.snapshot.claim

    def _required_process(self) -> ManagedProcess:
        if self.current_process is None:
            raise RuntimeError("managed process metadata is unavailable")
        return self.current_process

    @staticmethod
    def _process_fields(process: ManagedProcess) -> dict[str, object]:
        return {
            "process_pid": process.pid,
            "process_group_id": process.process_group_id,
            "process_registry_key": process.registry_key,
        }

    @staticmethod
    def _restore_process(snapshot: SupervisorSnapshot) -> ManagedProcess | None:
        if (
            snapshot.process_pid is None
            or snapshot.process_group_id is None
            or snapshot.process_registry_key is None
        ):
            return None
        return ManagedProcess(
            pid=snapshot.process_pid,
            process_group_id=snapshot.process_group_id,
            registry_key=snapshot.process_registry_key,
        )


@contextmanager
def _lock(path: Path):
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


def acknowledgement_path(state_path: str | os.PathLike[str]) -> Path:
    path = Path(state_path)
    return path.with_suffix(path.suffix + ".ack")


def write_acknowledgement(
    *,
    state_path: str | os.PathLike[str],
    generation: int,
    fingerprint: str,
    conversation_id: str,
) -> Path:
    state = Path(state_path)
    with _lock(state.with_suffix(state.suffix + ".lock")):
        snapshot = SupervisorSnapshot.model_validate_json(
            state.read_text(encoding="utf-8")
        )
        if snapshot.phase is not SupervisorPhase.AWAITING_ACK:
            raise ValueError("supervisor is not awaiting an acknowledgement")
        if generation != snapshot.generation:
            raise ValueError("acknowledgement generation does not match")
        if snapshot.checkpoint_fingerprint is None or not hmac.compare_digest(
            fingerprint, snapshot.checkpoint_fingerprint
        ):
            raise ValueError("acknowledgement fingerprint does not match")
        record = AcknowledgementRecord(
            generation=generation,
            fingerprint=fingerprint,
            conversation_id=conversation_id,
        )
        target = acknowledgement_path(state)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, target)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return target


def read_acknowledgement(
    state_path: str | os.PathLike[str],
) -> AcknowledgementRecord | None:
    target = acknowledgement_path(state_path)
    if not target.exists():
        return None
    with _lock(target.with_suffix(target.suffix + ".lock")):
        return AcknowledgementRecord.model_validate_json(
            target.read_text(encoding="utf-8")
        )


def clear_acknowledgement(state_path: str | os.PathLike[str]) -> None:
    target = acknowledgement_path(state_path)
    with _lock(target.with_suffix(target.suffix + ".lock")):
        if target.exists():
            target.unlink()
