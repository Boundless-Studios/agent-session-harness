"""Observe-only orchestration for registered managed resources."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .guardian_singleton import GuardianLeaseProof
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
    def current_proof(self) -> GuardianLeaseProof: ...


class GuardianPublication(BaseModel):
    """Observe-only output carrying the fences required by later execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    decision: GuardianDecision
    registration_id: str = Field(min_length=16, max_length=256)
    guardian: GuardianLeaseProof
    observe_only: Literal[True] = True


class GuardianService:
    """Evaluate durable registrations without executing cleanup."""

    def __init__(
        self,
        *,
        registry: ResourceRegistry,
        lease: GuardianLease,
        observer: Callable[[ManagedResource], GuardianObservation],
        publish: Callable[[GuardianPublication], None] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.registry = registry
        self.lease = lease
        self.observer = observer
        self.publish = publish or (lambda _decision: None)
        self.clock = clock

    def run_once(self) -> list[GuardianPublication]:
        initial_proof = self.lease.current_proof()
        publications: list[GuardianPublication] = []
        for registration in self.registry.list():
            observation = self._observe(registration.resource)
            decision = decide_guardian_action(observation)
            proof = initial_proof
            if decision.action is GuardianAction.REAP:
                if not self.registry.is_current(registration):
                    continue
                proof = self.lease.current_proof()
            publication = GuardianPublication(
                decision=decision,
                registration_id=registration.registration_id,
                guardian=proof,
            )
            self.publish(publication)
            publications.append(publication)
        return publications

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
