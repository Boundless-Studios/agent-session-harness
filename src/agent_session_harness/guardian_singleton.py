"""Fenced per-user ownership for the resource guardian service."""

from __future__ import annotations

import os
import pwd
import uuid
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


class GuardianLeaseProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    lease_epoch: int = Field(ge=1)


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

    def current_proof(self) -> GuardianLeaseProof:
        self.singleton.assert_current(self.handle, now=self.clock())
        return GuardianLeaseProof(
            claim_id=self.handle.claim_id,
            lease_epoch=self.handle.lease_epoch,
        )


class GuardianSingleton:
    """Own one durable, lease-epoch-fenced guardian claim per OS user."""

    _CONSTRUCTION_TOKEN = object()

    def __init__(
        self,
        coordinator: TaskCoordinator,
        *,
        _construction_token: object,
    ):
        if _construction_token is not self._CONSTRUCTION_TOKEN:
            raise TypeError("use GuardianSingleton.for_current_user()")
        self.coordinator = coordinator
        self.user_id = os.getuid()
        self.task = TaskIdentity(
            task_type="resource-guardian",
            task_id=str(self.user_id),
            fingerprint="resource-guardian-policy-v1",
        )

    @classmethod
    def for_current_user(
        cls,
    ) -> GuardianSingleton:
        return cls._from_state_root(cls.canonical_state_root())

    @staticmethod
    def canonical_state_root() -> Path:
        """Return one environment-independent state root for the effective UID."""

        account_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        return account_home / ".local" / "state" / "agent-session-harness" / "guardian"

    @classmethod
    def _for_test(cls, state_root: str | Path) -> GuardianSingleton:
        return cls._from_state_root(Path(state_root))

    @classmethod
    def _from_state_root(cls, root: Path) -> GuardianSingleton:
        return cls(
            TaskCoordinator(
                _SecureJsonlClaimStore(root / "singleton-claims.jsonl"),
                pid_is_live=lambda _pid: True,
            ),
            _construction_token=cls._CONSTRUCTION_TOKEN,
        )

    def acquire(
        self,
        *,
        owner_session_id: str,
        owner_pid: int | None,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> GuardianLeaseHandle:
        owner_incarnation_id = f"{owner_session_id}:{uuid.uuid4().hex}"
        try:
            claim = self.coordinator.claim_task(
                self.task,
                OwnerIdentity(
                    session_id=owner_incarnation_id,
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
            owner_session_id=owner_incarnation_id,
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
