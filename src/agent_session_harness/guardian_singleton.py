"""Fenced per-user ownership for the resource guardian service."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from agent_coordinator import (
    ClaimConflictError,
    OwnerIdentity,
    StaleClaimError,
    TaskCoordinator,
    TaskIdentity,
)
from pydantic import BaseModel, ConfigDict, Field

from .coordinator import _SecureJsonlClaimStore


class DuplicateGuardianError(RuntimeError):
    """Another guardian owns the current per-user lease."""


class StaleGuardianError(RuntimeError):
    """The guardian has been superseded or its lease has expired."""


class GuardianLeaseHandle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    lease_epoch: int = Field(ge=1)
    owner_session_id: str = Field(min_length=1)


class GuardianRelease(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    lease_epoch: int = Field(ge=1)
    release_reason: str = "guardian-shutdown"


class GuardianOwnership:
    """Bind a fenced handle to the no-argument service lease contract."""

    def __init__(
        self,
        singleton: GuardianSingleton,
        handle: GuardianLeaseHandle,
        clock: Callable[[], datetime],
    ):
        self.singleton = singleton
        self.handle = handle
        self.clock = clock

    def assert_current(self) -> None:
        self.singleton.assert_current(self.handle, now=self.clock())


class GuardianSingleton:
    """Own one durable, lease-epoch-fenced guardian claim per OS user."""

    def __init__(self, coordinator: TaskCoordinator, *, user_id: int | None = None):
        self.coordinator = coordinator
        self.user_id = os.getuid() if user_id is None else user_id
        self.task = TaskIdentity(
            task_type="resource-guardian",
            task_id=str(self.user_id),
            fingerprint="resource-guardian-policy-v1",
        )

    @classmethod
    def from_path(cls, path: str | Path) -> GuardianSingleton:
        return cls(
            TaskCoordinator(
                _SecureJsonlClaimStore(path),
                pid_is_live=lambda _pid: True,
            )
        )

    def acquire(
        self,
        *,
        owner_session_id: str,
        owner_pid: int | None,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> GuardianLeaseHandle:
        try:
            claim = self.coordinator.claim_task(
                self.task,
                OwnerIdentity(
                    session_id=owner_session_id,
                    pid=owner_pid,
                    agent="resource-guardian",
                ),
                lease_seconds=lease_seconds,
                now=now,
            )
        except ClaimConflictError as exc:
            raise DuplicateGuardianError(f"active guardian exists: {exc}") from exc
        return GuardianLeaseHandle(
            claim_id=claim.claim_id,
            lease_epoch=claim.lease_epoch,
            owner_session_id=owner_session_id,
        )

    def bind(
        self,
        handle: GuardianLeaseHandle,
        *,
        clock: Callable[[], datetime],
    ) -> GuardianOwnership:
        return GuardianOwnership(self, handle, clock)

    def heartbeat(
        self,
        handle: GuardianLeaseHandle,
        *,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> GuardianLeaseHandle:
        self.assert_current(handle, now=now)
        try:
            claim = self.coordinator.heartbeat_claim(
                handle.claim_id,
                owner_session_id=handle.owner_session_id,
                lease_epoch=handle.lease_epoch,
                lease_seconds=lease_seconds,
                now=now,
            )
        except (KeyError, PermissionError, StaleClaimError, ValueError) as exc:
            raise StaleGuardianError(f"stale guardian: {exc}") from exc
        return handle.model_copy(update={"lease_epoch": claim.lease_epoch})

    def assert_current(
        self,
        handle: GuardianLeaseHandle,
        *,
        now: datetime | None = None,
    ) -> None:
        current = self.coordinator.status(self.task, now=now).claim
        if (
            current is None
            or current.claim_id != handle.claim_id
            or current.lease_epoch != handle.lease_epoch
            or current.status != "active"
        ):
            raise StaleGuardianError(
                f"stale guardian lease: claim={handle.claim_id} "
                f"epoch={handle.lease_epoch}"
            )

    def release(
        self,
        handle: GuardianLeaseHandle,
        *,
        now: datetime | None = None,
    ) -> GuardianRelease:
        self.assert_current(handle, now=now)
        try:
            claim = self.coordinator.release_claim(
                handle.claim_id,
                owner_session_id=handle.owner_session_id,
                lease_epoch=handle.lease_epoch,
                reason="guardian-shutdown",
                now=now,
            )
        except (KeyError, PermissionError, StaleClaimError, ValueError) as exc:
            raise StaleGuardianError(f"stale guardian: {exc}") from exc
        return GuardianRelease(
            claim_id=claim.claim_id,
            lease_epoch=claim.lease_epoch,
        )
