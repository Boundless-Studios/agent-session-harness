"""Crash-recoverable supervision for exactly-once fresh-session rotation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .activity import ActivitySnapshot, Quiescence
from .capsule import HandoffCapsule
from .coordinator import ClaimHandle, CoordinatorAdapter, StaleOwnerError
from .events import LifecycleEvent
from .guardian import WATCHDOG_SHUTDOWN_MARGIN_SECONDS
from .ledger import EventLedger
from .models import Confidence, EventType, Runtime
from .process import (
    ExitReason,
    LaunchRequest,
    ManagedProcess,
    ProcessDriver,
)
from .secure_files import (
    append_private_text,
    atomic_write_private_text,
    exclusive_lock,
    lexical_absolute,
    private_exists,
    private_unlink,
    read_private_text,
)


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
    COMPLETED = "completed"
    BLOCKED = "blocked"


class UsageObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str = Field(min_length=1)
    context_percent: float = Field(ge=0)
    confidence: Confidence
    context_tokens: int | None = Field(default=None, ge=0)
    window_tokens: int | None = Field(default=None, gt=0)
    cumulative_tokens: int | None = Field(default=None, ge=0)


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
    fingerprint: str = Field(min_length=64, max_length=64)
    conversation_id: str = Field(min_length=1, max_length=160)
    owner_pid: int = Field(gt=0)


class UsageReader(Protocol):
    def sample(self, process: ManagedProcess) -> UsageObservation: ...


class CheckpointManager(Protocol):
    def checkpoint(self, request: CheckpointRequest) -> VerifiedCheckpoint: ...

    def acknowledge(
        self,
        capsule: HandoffCapsule,
        *,
        idempotency_key: str,
    ) -> bool: ...


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
    process_identity: str | None = None
    process_command_digest: str | None = None
    process_launch_nonce: str | None = None
    run_spec_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)
    context_percent: float | None = Field(default=None, ge=0)
    context_confidence: Confidence = Confidence.UNKNOWN
    context_tokens: int | None = Field(default=None, ge=0)
    window_tokens: int | None = Field(default=None, gt=0)
    cumulative_tokens: int | None = Field(default=None, ge=0)
    last_heartbeat_at: datetime | None = None
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
        runtime_environment: dict[str, str] | None = None,
        state_path: str | os.PathLike[str],
        process_driver: ProcessDriver,
        usage_reader: UsageReader,
        checkpoint_manager: CheckpointManager,
        coordinator: CoordinatorAdapter,
        warn_percent: float = 65.0,
        rotate_percent: float = 70.0,
        lease_seconds: int = 60,
        heartbeat_interval_seconds: float = 20.0,
        stop_timeout_seconds: float = 10.0,
    ):
        if not 0 < warn_percent < rotate_percent <= 100:
            raise ValueError("thresholds must satisfy 0 < warn < rotate <= 100")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if not 0 <= heartbeat_interval_seconds < lease_seconds:
            raise ValueError(
                "heartbeat interval must satisfy 0 <= interval < lease seconds"
            )
        watchdog_timeout_seconds = (
            float(lease_seconds) - WATCHDOG_SHUTDOWN_MARGIN_SECONDS
        )
        if watchdog_timeout_seconds <= 0 or (
            heartbeat_interval_seconds >= watchdog_timeout_seconds
        ):
            raise ValueError(
                "heartbeat interval and lease must preserve the watchdog shutdown "
                "margin"
            )
        self.runtime = Runtime(runtime)
        self.chain_id = chain_id
        self.cwd = Path(cwd).expanduser().resolve()
        self.task_type = task_type
        self.task_id = task_id
        self.task_fingerprint = task_fingerprint
        self.executable = executable
        self.runtime_args = runtime_args
        self.runtime_environment = dict(runtime_environment or {})
        reserved_environment_keys = sorted(
            key
            for key in self.runtime_environment
            if key.startswith("AGENT_SESSION_HARNESS_")
        )
        if reserved_environment_keys:
            raise ValueError(
                f"reserved runtime environment key: {reserved_environment_keys[0]}"
            )
        self.state_path = lexical_absolute(state_path)
        self.effect_path = self.state_path.with_suffix(
            self.state_path.suffix + ".events"
        )
        self.transition_lock_path = self.state_path.with_suffix(
            self.state_path.suffix + ".transition.lock"
        )
        self.lifecycle_path = self.state_path.with_suffix(
            self.state_path.suffix + ".lifecycle"
        )
        self.process_driver = process_driver
        self.usage_reader = usage_reader
        self.checkpoint_manager = checkpoint_manager
        self.coordinator = coordinator
        self.warn_percent = warn_percent
        self.rotate_percent = rotate_percent
        self.lease_seconds = lease_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.watchdog_timeout_seconds = watchdog_timeout_seconds
        self.stop_timeout_seconds = stop_timeout_seconds
        self.run_spec_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "runtime": self.runtime.value,
                    "chain_id": self.chain_id,
                    "cwd": str(self.cwd),
                    "task_type": self.task_type,
                    "task_id": self.task_id,
                    "task_fingerprint": self.task_fingerprint,
                    "executable": self.executable,
                    "runtime_args": self.runtime_args,
                    "runtime_environment_keys": sorted(self.runtime_environment),
                    "runtime_environment_fingerprint": hashlib.sha256(
                        json.dumps(
                            sorted(self.runtime_environment.items()),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.snapshot = self._load()
        self.current_process = self._restore_process(self.snapshot)

    @property
    def can_dispatch(self) -> bool:
        return self.snapshot.phase in {
            SupervisorPhase.RUNNING,
            SupervisorPhase.WARNING,
        }

    def start(self) -> SupervisorSnapshot:
        with exclusive_lock(self.transition_lock_path):
            self._refresh()
            return self._start_unlocked()

    def _start_unlocked(self) -> SupervisorSnapshot:
        if self.snapshot.phase is SupervisorPhase.COMPLETED:
            self._finalize_completed_cleanup()
            return self.snapshot
        if self.snapshot.phase is SupervisorPhase.INITIAL:
            self._persist()
            owner_session_id = self._owner_session_id(0)
            self._effect("claim", "started", generation=0)
            claimed_at = self._now()
            claim = self._claim(owner_session_id, now=claimed_at)
            self.snapshot = self.snapshot.model_copy(
                update={
                    "generation": 0,
                    "phase": SupervisorPhase.LAUNCHING,
                    "owner_session_id": owner_session_id,
                    "claim": claim,
                    "last_heartbeat_at": claimed_at,
                }
            )
            self._persist()
            self._effect("claim", "completed", generation=0)
        if (
            self.snapshot.phase is not SupervisorPhase.LAUNCHING
            or self.snapshot.generation != 0
        ):
            self._ensure_active_process_is_live()
            return self.snapshot
        request = self._launch_request(generation=0)
        self._effect("launch", "started", generation=0)
        process = self.process_driver.start_fresh(request)
        self.current_process = process
        self.snapshot = self.snapshot.model_copy(
            update={
                "phase": SupervisorPhase.RUNNING,
                **self._process_fields(process),
            }
        )
        self._persist()
        self._effect("launch", "completed", generation=0)
        return self.snapshot

    def tick(self, activity: ActivitySnapshot) -> SupervisorSnapshot:
        with exclusive_lock(self.transition_lock_path):
            self._refresh()
            return self._tick_unlocked(activity)

    def shutdown(self) -> SupervisorSnapshot:
        """Stop the managed child and durably fail closed for terminal CLI exit."""

        with exclusive_lock(self.transition_lock_path):
            known_snapshot = self.snapshot
            known_process = self.current_process
            refresh_error: Exception | None = None
            try:
                self._refresh()
            except (OSError, ValueError) as exc:
                refresh_error = exc
                self.snapshot = known_snapshot
                self.current_process = known_process
            updates: dict[str, object] = {}
            if self.snapshot.claim is None and known_snapshot.claim is not None:
                updates.update(
                    {
                        "generation": known_snapshot.generation,
                        "owner_session_id": known_snapshot.owner_session_id,
                        "claim": known_snapshot.claim,
                        "last_heartbeat_at": known_snapshot.last_heartbeat_at,
                    }
                )
            if self.current_process is None and known_process is not None:
                self.current_process = known_process
                updates.update(
                    {
                        "generation": known_snapshot.generation,
                        **self._process_fields(known_process),
                    }
                )
            if updates:
                self.snapshot = self.snapshot.model_copy(update=updates)
            snapshot = self._shutdown_unlocked()
            if refresh_error is not None:
                raise RuntimeError(
                    "terminal shutdown state refresh failed after cleanup"
                ) from refresh_error
            return snapshot

    def _shutdown_unlocked(self) -> SupervisorSnapshot:
        if self.snapshot.phase is SupervisorPhase.COMPLETED:
            self._finalize_completed_cleanup()
            return self.snapshot
        process = self.current_process
        stop_error: Exception | None = None
        journal_error: Exception | None = None
        exit_cleanup_error: Exception | None = None

        def record_effect(effect: str, status: str) -> None:
            nonlocal journal_error
            try:
                self._effect(
                    effect,
                    status,
                    generation=self.snapshot.generation,
                )
            except (OSError, RuntimeError) as exc:
                if journal_error is None:
                    journal_error = exc

        if process is not None:
            self.snapshot = self.snapshot.model_copy(
                update={"phase": SupervisorPhase.BLOCKED}
            )
            self._persist()
        if process is not None and self.process_driver.is_alive(process):
            record_effect("stop", "started")
            try:
                self.process_driver.graceful_stop(
                    process,
                    self.stop_timeout_seconds,
                )
            except (OSError, RuntimeError) as exc:
                stop_error = exc
            if self.process_driver.is_alive(process):
                self.snapshot = self.snapshot.model_copy(
                    update={"phase": SupervisorPhase.BLOCKED}
                )
                self._persist()
                raise RuntimeError(
                    "managed process remains live after terminal shutdown"
                ) from stop_error
            record_effect("stop", "completed")

        if process is not None:
            try:
                self.process_driver.clear_exit_status(process)
            except (OSError, RuntimeError, ValueError) as exc:
                exit_cleanup_error = exc

        if exit_cleanup_error is None:
            self.current_process = None
        fence_error: Exception | None = None
        had_claim = self.snapshot.claim is not None
        claim_released = not had_claim
        if had_claim:
            record_effect("fence", "started")
            try:
                self.coordinator.fence(self.snapshot.claim)
            except StaleOwnerError:
                claim_released = True
            except (OSError, RuntimeError) as exc:
                fence_error = exc
            else:
                claim_released = True
        update: dict[str, object] = {"phase": SupervisorPhase.BLOCKED}
        if exit_cleanup_error is None:
            update.update(self._cleared_process_fields())
        if claim_released:
            update.update(
                {
                    "owner_session_id": None,
                    "claim": None,
                    "last_heartbeat_at": None,
                }
            )
        self.snapshot = self.snapshot.model_copy(update=update)
        self._persist()
        if fence_error is not None:
            raise RuntimeError(
                "managed process stopped but coordinator fencing failed"
            ) from fence_error
        if had_claim and claim_released:
            record_effect("fence", "completed")
        if journal_error is not None:
            raise RuntimeError(
                "terminal shutdown effect journaling failed after cleanup"
            ) from journal_error
        if exit_cleanup_error is not None:
            raise RuntimeError(
                "terminal exit record cleanup failed"
            ) from exit_cleanup_error
        return self.snapshot

    def _tick_unlocked(self, activity: ActivitySnapshot) -> SupervisorSnapshot:
        if self.snapshot.phase is SupervisorPhase.INITIAL:
            raise RuntimeError("supervisor must be started before ticking")
        self._ensure_active_process_is_live()
        if self.snapshot.phase is SupervisorPhase.COMPLETED:
            return self.snapshot
        self._heartbeat_if_due()
        if self.snapshot.phase is SupervisorPhase.AWAITING_ACK:
            acknowledgement = read_acknowledgement(self.state_path)
            if acknowledgement is None:
                return self.snapshot
            snapshot = self._acknowledge_unlocked(
                generation=acknowledgement.generation,
                fingerprint=acknowledgement.fingerprint,
                conversation_id=acknowledgement.conversation_id,
                owner_pid=acknowledgement.owner_pid,
            )
            clear_acknowledgement(self.state_path, expected=acknowledgement)
            return snapshot
        if self.snapshot.phase in {
            SupervisorPhase.RUNNING,
            SupervisorPhase.WARNING,
            SupervisorPhase.DRAINING,
        }:
            self._observe_usage()
            if self.snapshot.phase is SupervisorPhase.DRAINING:
                if (
                    self.snapshot.context_confidence is not Confidence.CONFIDENT
                    or self.snapshot.generation
                    not in activity.handoff_requested_generations
                    or activity.quiescence is not Quiescence.IDLE
                ):
                    return self.snapshot
                self._set_phase(SupervisorPhase.CHECKPOINTING)
        return self._advance_rotation()

    def acknowledge(
        self,
        *,
        generation: int,
        fingerprint: str | None,
        conversation_id: str,
        owner_pid: int,
    ) -> SupervisorSnapshot:
        with exclusive_lock(self.transition_lock_path):
            self._refresh()
            return self._acknowledge_unlocked(
                generation=generation,
                fingerprint=fingerprint,
                conversation_id=conversation_id,
                owner_pid=owner_pid,
            )

    def _acknowledge_unlocked(
        self,
        *,
        generation: int,
        fingerprint: str | None,
        conversation_id: str,
        owner_pid: int,
    ) -> SupervisorSnapshot:
        if self.snapshot.phase is not SupervisorPhase.AWAITING_ACK:
            raise ValueError("no successor acknowledgement is expected")
        if generation != self.snapshot.generation:
            raise ValueError("acknowledgement generation does not match")
        if fingerprint != self.snapshot.checkpoint_fingerprint:
            raise ValueError("acknowledgement fingerprint does not match")
        if owner_pid != self.snapshot.process_pid:
            raise ValueError("acknowledgement process does not match managed child")
        capsule = self._verified_capsule(fingerprint)
        self._heartbeat_if_due(force=True)
        self._effect("acknowledge", "started", generation=generation)
        if not self.checkpoint_manager.acknowledge(
            capsule,
            idempotency_key=f"{self.chain_id}:{generation}:ack",
        ):
            raise RuntimeError("required checkpoint acknowledgement was not verified")
        EventLedger(self.lifecycle_path).append(
            LifecycleEvent(
                schema_version=1,
                event_id=f"handoff-ack:{self.chain_id}:{generation}",
                runtime=self.runtime,
                chain_id=self.chain_id,
                conversation_id=conversation_id,
                generation=generation,
                event_type=EventType.HANDOFF_ACKNOWLEDGED,
                timestamp=self._now(),
                cwd=self.cwd,
                owner_pid=owner_pid,
                name=capsule.fingerprint,
            )
        )
        self.snapshot = self.snapshot.model_copy(
            update={
                "phase": SupervisorPhase.RUNNING,
                "conversation_id": conversation_id,
                "context_percent": None,
                "context_confidence": Confidence.UNKNOWN,
                "context_tokens": None,
                "window_tokens": None,
                "cumulative_tokens": None,
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
        if sample.confidence is not Confidence.CONFIDENT:
            self.snapshot = self.snapshot.model_copy(
                update={"context_confidence": sample.confidence}
            )
            self._persist()
            return
        self.snapshot = self.snapshot.model_copy(
            update={
                "conversation_id": sample.conversation_id,
                "context_percent": sample.context_percent,
                "context_confidence": sample.confidence,
                "context_tokens": sample.context_tokens,
                "window_tokens": sample.window_tokens,
                "cumulative_tokens": sample.cumulative_tokens,
            }
        )
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
                EventLedger(self.lifecycle_path).append(
                    LifecycleEvent(
                        schema_version=1,
                        event_id=f"handoff-checkpoint:{self.chain_id}:{target}",
                        runtime=self.runtime,
                        chain_id=self.chain_id,
                        conversation_id=predecessor,
                        generation=target,
                        event_type=EventType.HANDOFF_CHECKPOINTED,
                        timestamp=self._now(),
                        cwd=self.cwd,
                        owner_pid=os.getpid(),
                        name=receipt.fingerprint,
                    )
                )
                self.snapshot = self.snapshot.model_copy(
                    update={
                        "phase": SupervisorPhase.CHECKPOINTED,
                        "checkpoint_fingerprint": receipt.fingerprint,
                        "checkpoint_path": receipt.path,
                    }
                )
                self._persist()
                self._heartbeat_if_due()
                continue
            if phase is SupervisorPhase.CHECKPOINTED:
                self._set_phase(SupervisorPhase.FENCING)
                continue
            if phase is SupervisorPhase.FENCING:
                claim = self._required_claim()
                self._effect("fence", "started", generation=self.snapshot.generation)
                try:
                    self.coordinator.fence(claim)
                except StaleOwnerError:
                    self._fail_closed_for_stale_owner()
                    raise
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
                self.process_driver.clear_exit_status(process)
                self.current_process = None
                self.snapshot = self.snapshot.model_copy(
                    update={
                        "phase": SupervisorPhase.STOPPED,
                        "process_pid": None,
                        "process_group_id": None,
                        "process_registry_key": None,
                        "process_identity": None,
                        "process_command_digest": None,
                        "process_launch_nonce": None,
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
                claimed_at = self._now()
                claim = self._claim(owner_session_id, now=claimed_at)
                self.snapshot = self.snapshot.model_copy(
                    update={
                        "generation": target,
                        "phase": SupervisorPhase.LAUNCHING,
                        "owner_session_id": owner_session_id,
                        "conversation_id": None,
                        "claim": claim,
                        "last_heartbeat_at": claimed_at,
                    }
                )
                self._persist()
                self._effect("claim", "completed", generation=target)
                continue
            if phase is SupervisorPhase.LAUNCHING:
                request = self._launch_request(generation=self.snapshot.generation)
                self._effect("launch", "started", generation=self.snapshot.generation)
                process = self.process_driver.start_fresh(request)
                self.current_process = process
                self.snapshot = self.snapshot.model_copy(
                    update={
                        "phase": SupervisorPhase.AWAITING_ACK,
                        **self._process_fields(process),
                    }
                )
                self._persist()
                self._effect("launch", "completed", generation=self.snapshot.generation)
                self._heartbeat_if_due()
                continue
            return self.snapshot

    def _claim(self, owner_session_id: str, *, now: datetime) -> ClaimHandle:
        return self.coordinator.claim(
            task_type=self.task_type,
            task_id=self.task_id,
            fingerprint=self.task_fingerprint,
            owner_session_id=owner_session_id,
            owner_pid=os.getpid(),
            runtime=self.runtime,
            worktree_path=str(self.cwd),
            lease_seconds=self.lease_seconds,
            now=now,
        )

    def _launch_request(self, *, generation: int) -> LaunchRequest:
        environment = dict(self.runtime_environment)
        environment.update(
            {
                "AGENT_SESSION_HARNESS_MANAGED": "1",
                "AGENT_SESSION_HARNESS_CHAIN_ID": self.chain_id,
                "AGENT_SESSION_HARNESS_GENERATION": str(generation),
                "AGENT_SESSION_HARNESS_LEDGER": str(
                    self.state_path.with_suffix(self.state_path.suffix + ".lifecycle")
                ),
                "AGENT_SESSION_HARNESS_STATE_PATH": str(self.state_path),
                "AGENT_SESSION_HARNESS_WATCHDOG_TIMEOUT_SECONDS": str(
                    self.watchdog_timeout_seconds
                ),
            }
        )
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
            allowed_environment_keys=frozenset(self.runtime_environment),
            capsule_path=self.snapshot.checkpoint_path,
            capsule_fingerprint=self.snapshot.checkpoint_fingerprint,
            handoff_message=message,
        )

    def _set_phase(self, phase: SupervisorPhase) -> None:
        self.snapshot = self.snapshot.model_copy(update={"phase": phase})
        self._persist()

    def _refresh(self) -> None:
        refreshed = self._load()
        if (
            self.current_process is not None
            and refreshed.process_pid == self.current_process.pid
            and refreshed.process_registry_key == self.current_process.registry_key
        ):
            process = self.current_process
        else:
            process = self._restore_process(refreshed)
        self.snapshot = refreshed
        self.current_process = process
        acknowledgement = read_acknowledgement(self.state_path)
        if acknowledgement is not None and (
            refreshed.phase is not SupervisorPhase.AWAITING_ACK
            or acknowledgement.generation < refreshed.generation
        ):
            clear_acknowledgement(self.state_path, expected=acknowledgement)

    def _heartbeat_if_due(self, *, force: bool = False) -> None:
        if self.snapshot.phase not in {
            SupervisorPhase.RUNNING,
            SupervisorPhase.WARNING,
            SupervisorPhase.DRAINING,
            SupervisorPhase.CHECKPOINTING,
            SupervisorPhase.CHECKPOINTED,
            SupervisorPhase.LAUNCHING,
            SupervisorPhase.AWAITING_ACK,
        }:
            return
        claim = self.snapshot.claim
        if claim is None:
            raise RuntimeError("active supervisor phase has no coordinator claim")
        now = self._now()
        last = self.snapshot.last_heartbeat_at
        if (
            not force
            and last is not None
            and now - last < timedelta(seconds=self.heartbeat_interval_seconds)
        ):
            return
        try:
            refreshed = self.coordinator.heartbeat(
                claim,
                lease_seconds=self.lease_seconds,
                now=now,
            )
        except StaleOwnerError:
            self._fail_closed_for_stale_owner()
            raise
        self.snapshot = self.snapshot.model_copy(
            update={"claim": refreshed, "last_heartbeat_at": now}
        )
        self._persist()

    def _ensure_active_process_is_live(self) -> None:
        if self.snapshot.phase not in {
            SupervisorPhase.RUNNING,
            SupervisorPhase.WARNING,
            SupervisorPhase.DRAINING,
            SupervisorPhase.CHECKPOINTING,
            SupervisorPhase.CHECKPOINTED,
            SupervisorPhase.FENCING,
            SupervisorPhase.FENCED,
            SupervisorPhase.AWAITING_ACK,
        }:
            return
        process = self.current_process
        try:
            live = process is not None and self.process_driver.is_alive(process)
        except (OSError, RuntimeError, ValueError):
            self._persist_blocked()
            raise
        if live:
            return
        try:
            terminal = (
                self.process_driver.exit_status(process)
                if process is not None
                else None
            )
        except (OSError, RuntimeError, ValueError):
            self._persist_blocked()
            raise
        if (
            terminal is not None
            and terminal.return_code == 0
            and terminal.reason is ExitReason.NATURAL
            and process is not None
            and self.snapshot.phase
            in {SupervisorPhase.RUNNING, SupervisorPhase.WARNING}
        ):
            self._complete_clean_exit(process)
            return
        self._persist_blocked()
        if terminal is not None:
            raise RuntimeError(
                "managed process exited with status "
                f"{terminal.return_code} ({terminal.reason.value}); "
                "supervisor blocked"
            )
        raise RuntimeError("managed process is not live; supervisor blocked")

    def _complete_clean_exit(self, process: ManagedProcess) -> None:
        self._effect(
            "terminal-exit",
            "started",
            generation=self.snapshot.generation,
        )
        claim = self.snapshot.claim
        if claim is not None:
            try:
                self.coordinator.fence(claim)
            except StaleOwnerError:
                pass
            except (OSError, RuntimeError):
                self.snapshot = self.snapshot.model_copy(
                    update={"phase": SupervisorPhase.BLOCKED}
                )
                self._persist()
                raise
        # Persist completion while retaining the process identity. If exit-record
        # cleanup or the final state write is interrupted, a restarted supervisor
        # can safely retry cleanup without treating the generation as active.
        self.snapshot = self.snapshot.model_copy(
            update={
                "phase": SupervisorPhase.COMPLETED,
                "owner_session_id": None,
                "claim": None,
                "last_heartbeat_at": None,
            }
        )
        self._persist()
        try:
            self._finalize_completed_cleanup()
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("completed process exit record cleanup failed") from exc
        self._effect(
            "terminal-exit",
            "completed",
            generation=self.snapshot.generation,
        )

    def _finalize_completed_cleanup(self) -> None:
        if self.snapshot.phase is not SupervisorPhase.COMPLETED:
            return
        process = self.current_process
        if process is None:
            return
        self.process_driver.clear_exit_status(process)
        self.current_process = None
        self.snapshot = self.snapshot.model_copy(update=self._cleared_process_fields())
        self._persist()

    def _fail_closed_for_stale_owner(self) -> None:
        self.snapshot = self.snapshot.model_copy(
            update={"phase": SupervisorPhase.BLOCKED}
        )
        self._persist()
        process = self.current_process
        if process is None:
            return
        try:
            self.process_driver.graceful_stop(process, self.stop_timeout_seconds)
        finally:
            if not self.process_driver.is_alive(process):
                try:
                    self.process_driver.clear_exit_status(process)
                except (OSError, RuntimeError, ValueError):
                    # Keep the process identity in the blocked snapshot so a
                    # later shutdown can retry exact-record cleanup.
                    return
                self.current_process = None
                self.snapshot = self.snapshot.model_copy(
                    update=self._cleared_process_fields()
                )
                self._persist()

    def _persist_blocked(self) -> None:
        self.snapshot = self.snapshot.model_copy(
            update={"phase": SupervisorPhase.BLOCKED}
        )
        self._persist()

    def _verified_capsule(self, fingerprint: str | None) -> HandoffCapsule:
        if self.snapshot.checkpoint_path is None or fingerprint is None:
            raise ValueError("acknowledgement requires a checkpoint capsule")
        try:
            capsule = HandoffCapsule.model_validate_json(
                read_private_text(self.snapshot.checkpoint_path)
            )
        except (OSError, ValueError) as exc:
            raise ValueError("checkpoint capsule cannot be verified") from exc
        if not hmac.compare_digest(capsule.fingerprint, fingerprint):
            raise ValueError("checkpoint capsule fingerprint does not match")
        if capsule.chain_id != self.chain_id:
            raise ValueError("checkpoint capsule chain does not match")
        if capsule.target_generation != self.snapshot.generation:
            raise ValueError("checkpoint capsule generation does not match")
        return capsule

    @staticmethod
    def _now() -> datetime:
        return datetime.now(tz=timezone.utc)

    def _load(self) -> SupervisorSnapshot:
        if not private_exists(self.state_path):
            return SupervisorSnapshot(
                runtime=self.runtime,
                chain_id=self.chain_id,
                run_spec_fingerprint=self.run_spec_fingerprint,
            )
        with exclusive_lock(
            self.state_path.with_suffix(self.state_path.suffix + ".lock")
        ):
            payload = json.loads(read_private_text(self.state_path))
        snapshot = SupervisorSnapshot.model_validate(payload)
        if snapshot.runtime is not self.runtime or snapshot.chain_id != self.chain_id:
            raise ValueError("supervisor state does not match requested runtime/chain")
        if snapshot.run_spec_fingerprint != self.run_spec_fingerprint:
            raise ValueError("supervisor state run specification does not match")
        return snapshot

    def _persist(self) -> None:
        with exclusive_lock(
            self.state_path.with_suffix(self.state_path.suffix + ".lock")
        ):
            atomic_write_private_text(
                self.state_path,
                json.dumps(
                    self.snapshot.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
            )

    def _effect(self, effect: str, status: str, *, generation: int) -> None:
        payload = {
            "schema_version": 1,
            "effect_id": f"{self.chain_id}:{generation}:{effect}",
            "effect": effect,
            "status": status,
            "chain_id": self.chain_id,
            "generation": generation,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with exclusive_lock(
            self.effect_path.with_suffix(self.effect_path.suffix + ".lock")
        ):
            append_private_text(self.effect_path, encoded + "\n")

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
            "process_identity": process.identity,
            "process_command_digest": process.command_digest,
            "process_launch_nonce": process.launch_nonce,
        }

    @staticmethod
    def _cleared_process_fields() -> dict[str, object]:
        return {
            "process_pid": None,
            "process_group_id": None,
            "process_registry_key": None,
            "process_identity": None,
            "process_command_digest": None,
            "process_launch_nonce": None,
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
            identity=snapshot.process_identity,
            command_digest=snapshot.process_command_digest,
            launch_nonce=snapshot.process_launch_nonce,
        )


def acknowledgement_path(state_path: str | os.PathLike[str]) -> Path:
    path = lexical_absolute(state_path)
    return path.with_suffix(path.suffix + ".ack")


def write_acknowledgement(
    *,
    state_path: str | os.PathLike[str],
    generation: int,
    fingerprint: str,
    conversation_id: str,
    owner_pid: int | None = None,
) -> Path:
    state = lexical_absolute(state_path)
    acknowledged_pid = os.getpid() if owner_pid is None else owner_pid
    with exclusive_lock(state.with_suffix(state.suffix + ".lock")):
        snapshot = SupervisorSnapshot.model_validate_json(read_private_text(state))
        if snapshot.phase is not SupervisorPhase.AWAITING_ACK:
            raise ValueError("supervisor is not awaiting an acknowledgement")
        if generation != snapshot.generation:
            raise ValueError("acknowledgement generation does not match")
        if snapshot.checkpoint_fingerprint is None or not hmac.compare_digest(
            fingerprint, snapshot.checkpoint_fingerprint
        ):
            raise ValueError("acknowledgement fingerprint does not match")
        if snapshot.process_pid != acknowledged_pid:
            raise ValueError("acknowledgement process does not match managed child")
        if snapshot.checkpoint_path is None:
            raise ValueError("acknowledgement requires a checkpoint capsule")
        try:
            capsule = HandoffCapsule.model_validate_json(
                read_private_text(snapshot.checkpoint_path)
            )
        except (OSError, ValueError) as exc:
            raise ValueError("checkpoint capsule cannot be verified") from exc
        if not hmac.compare_digest(capsule.fingerprint, fingerprint):
            raise ValueError("checkpoint capsule fingerprint does not match")
        if capsule.chain_id != snapshot.chain_id:
            raise ValueError("checkpoint capsule chain does not match")
        if capsule.target_generation != generation:
            raise ValueError("checkpoint capsule generation does not match")
        record = AcknowledgementRecord(
            generation=generation,
            fingerprint=fingerprint,
            conversation_id=conversation_id,
            owner_pid=acknowledged_pid,
        )
        target = acknowledgement_path(state)
        with exclusive_lock(target.with_suffix(target.suffix + ".lock")):
            atomic_write_private_text(target, record.model_dump_json() + "\n")
        return target


def read_acknowledgement(
    state_path: str | os.PathLike[str],
) -> AcknowledgementRecord | None:
    target = acknowledgement_path(state_path)
    with exclusive_lock(target.with_suffix(target.suffix + ".lock")):
        if not private_exists(target):
            return None
        return AcknowledgementRecord.model_validate_json(read_private_text(target))


def clear_acknowledgement(
    state_path: str | os.PathLike[str],
    *,
    expected: AcknowledgementRecord | None = None,
) -> None:
    target = acknowledgement_path(state_path)
    with exclusive_lock(target.with_suffix(target.suffix + ".lock")):
        if not private_exists(target):
            return
        if expected is not None:
            current = AcknowledgementRecord.model_validate_json(
                read_private_text(target)
            )
            if current != expected:
                return
        private_unlink(target)
