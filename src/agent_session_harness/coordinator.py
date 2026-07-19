"""Thin fenced-ownership adapter over agent-coordinator v0.2.0."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from agent_coordinator import (
    JsonlClaimStore,
    OwnerIdentity,
    StaleClaimError,
    TaskCoordinator,
    TaskIdentity,
)
from pydantic import BaseModel, ConfigDict, Field

from .models import Runtime


class StaleOwnerError(RuntimeError):
    """The caller no longer owns the current fenced task generation."""


class ClaimHandle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    lease_epoch: int = Field(ge=1)
    task_type: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    task_fingerprint: str = Field(min_length=1)
    owner_session_id: str = Field(min_length=1)


class FenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    lease_epoch: int
    release_reason: str


class CoordinatorAdapter:
    def __init__(self, coordinator: TaskCoordinator):
        self.coordinator = coordinator

    @classmethod
    def from_path(cls, path: str | Path) -> CoordinatorAdapter:
        return cls(TaskCoordinator(JsonlClaimStore(path)))

    def claim(
        self,
        *,
        task_type: str,
        task_id: str,
        fingerprint: str,
        owner_session_id: str,
        owner_pid: int | None,
        runtime: str | Runtime,
        worktree_path: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> ClaimHandle:
        runtime_value = Runtime(runtime)
        task = TaskIdentity(
            task_type=task_type,
            task_id=task_id,
            fingerprint=fingerprint,
        )
        owner = OwnerIdentity(
            session_id=owner_session_id,
            pid=owner_pid,
            agent=runtime_value.value,
            worktree_path=worktree_path,
        )
        record = self.coordinator.claim_task(
            task,
            owner,
            lease_seconds=lease_seconds,
            now=now,
        )
        return ClaimHandle(
            claim_id=record.claim_id,
            lease_epoch=record.lease_epoch,
            task_type=task_type,
            task_id=task_id,
            task_fingerprint=fingerprint,
            owner_session_id=owner_session_id,
        )

    def heartbeat(
        self,
        handle: ClaimHandle,
        *,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> ClaimHandle:
        self._assert_current(handle, now=now)
        try:
            record = self.coordinator.heartbeat_claim(
                handle.claim_id,
                owner_session_id=handle.owner_session_id,
                lease_epoch=handle.lease_epoch,
                lease_seconds=lease_seconds,
                now=now,
            )
        except (StaleClaimError, KeyError, PermissionError, ValueError) as exc:
            raise StaleOwnerError(f"stale owner: {exc}") from exc
        return handle.model_copy(update={"lease_epoch": record.lease_epoch})

    def fence(
        self,
        handle: ClaimHandle,
        *,
        now: datetime | None = None,
    ) -> FenceResult:
        decision = self.coordinator.status(self._task(handle), now=now)
        current = decision.claim
        if (
            current is not None
            and current.claim_id == handle.claim_id
            and current.lease_epoch == handle.lease_epoch
            and current.release_reason == "context-rotation"
        ):
            return FenceResult(
                claim_id=current.claim_id,
                lease_epoch=current.lease_epoch,
                release_reason="context-rotation",
            )
        self._assert_current(handle, now=now)
        try:
            record = self.coordinator.release_claim(
                handle.claim_id,
                owner_session_id=handle.owner_session_id,
                lease_epoch=handle.lease_epoch,
                reason="context-rotation",
                now=now,
            )
        except (StaleClaimError, KeyError, PermissionError, ValueError) as exc:
            raise StaleOwnerError(f"stale owner: {exc}") from exc
        return FenceResult(
            claim_id=record.claim_id,
            lease_epoch=record.lease_epoch,
            release_reason=record.release_reason or "context-rotation",
        )

    def _assert_current(
        self, handle: ClaimHandle, *, now: datetime | None
    ) -> None:
        decision = self.coordinator.status(self._task(handle), now=now)
        current = decision.claim
        if (
            current is None
            or current.claim_id != handle.claim_id
            or current.lease_epoch != handle.lease_epoch
            or current.status != "active"
        ):
            raise StaleOwnerError(
                f"stale owner lease: claim={handle.claim_id} epoch={handle.lease_epoch}"
            )

    @staticmethod
    def _task(handle: ClaimHandle) -> TaskIdentity:
        return TaskIdentity(
            task_type=handle.task_type,
            task_id=handle.task_id,
            fingerprint=handle.task_fingerprint,
        )
