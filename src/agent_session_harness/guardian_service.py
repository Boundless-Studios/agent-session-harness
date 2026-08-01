"""Observe-only orchestration for registered managed resources."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from .process_identity import ProcessState
from .resource_guardian import (
    GuardianAction,
    GuardianDecision,
    GuardianEvidence,
    GuardianObservation,
    LeaseState,
    ManagedOwnerState,
    ManagedResource,
    ProcessIdentityState,
    WorktreeState,
    decide_guardian_action,
)
from .resource_registry import ResourceRegistry


class GuardianLease(Protocol):
    def assert_current(self) -> None: ...


class GuardianService:
    """Evaluate durable registrations without executing cleanup."""

    def __init__(
        self,
        *,
        registry: ResourceRegistry,
        lease: GuardianLease,
        observer: Callable[[ManagedResource], GuardianObservation],
        publish: Callable[[GuardianDecision], None] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.registry = registry
        self.lease = lease
        self.observer = observer
        self.publish = publish or (lambda _decision: None)
        self.clock = clock

    def run_once(self) -> list[GuardianDecision]:
        self.lease.assert_current()
        decisions: list[GuardianDecision] = []
        for registration in self.registry.list():
            observation = self._observe(registration.resource)
            decision = decide_guardian_action(observation)
            if decision.action is GuardianAction.REAP:
                self.lease.assert_current()
                if not self.registry.is_current(registration):
                    continue
            self.publish(decision)
            decisions.append(decision)
        return decisions

    def _observe(self, resource: ManagedResource) -> GuardianObservation:
        try:
            return self.observer(resource)
        except Exception as exc:
            return GuardianObservation(
                resource=resource,
                process_state=(
                    ProcessState.UNKNOWN
                    if resource.process_identity is not None
                    else None
                ),
                process_identity_state=(
                    ProcessIdentityState.UNKNOWN
                    if resource.process_identity is not None
                    else ProcessIdentityState.NOT_APPLICABLE
                ),
                worktree_state=(
                    WorktreeState.UNKNOWN
                    if resource.worktree_identity is not None
                    else WorktreeState.NOT_APPLICABLE
                ),
                managed_owner_state=ManagedOwnerState.UNKNOWN,
                lease_state=(
                    LeaseState.UNKNOWN
                    if resource.owner_lease is not None
                    else LeaseState.NOT_APPLICABLE
                ),
                evidence=[
                    GuardianEvidence(
                        source="guardian-service",
                        code="inspection_failed",
                        detail=type(exc).__name__,
                    )
                ],
                observed_at=self.clock(),
            )
