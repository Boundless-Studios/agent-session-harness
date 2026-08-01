"""Fence-validating execution boundary for guardian cleanup adapters."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from .guardian_service import GuardianPublication
from .guardian_singleton import GuardianLeaseProof
from .resource_guardian import GuardianAction, GuardianReasonCode
from .resource_registry import ResourceRegistration, ResourceRegistry


class CleanupOutcome(StrEnum):
    OBSERVED = "observed"
    AUTHORIZED = "authorized"
    REFUSED_STALE_REGISTRATION = "refused_stale_registration"
    REFUSED_STALE_GUARDIAN = "refused_stale_guardian"


class CleanupAuthorization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    registration: ResourceRegistration
    guardian: GuardianLeaseProof


class CleanupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: CleanupOutcome
    authorization: CleanupAuthorization | None = None


class CurrentGuardianOwnership(Protocol):
    def current_proof(self) -> GuardianLeaseProof: ...


class GuardianExecutor:
    """Execute only explicitly enabled, still-fenced cleanup decisions."""

    def __init__(
        self,
        *,
        registry: ResourceRegistry,
        ownership: CurrentGuardianOwnership,
    ):
        self.registry = registry
        self.ownership = ownership

    def authorize(
        self,
        publication: GuardianPublication,
        *,
        observe_only: bool = True,
        enabled_reasons: set[GuardianReasonCode] | None = None,
    ) -> CleanupResult:
        decision = publication.decision
        if (
            observe_only
            or decision.action is not GuardianAction.REAP
            or decision.reason_code not in (enabled_reasons or set())
        ):
            return CleanupResult(outcome=CleanupOutcome.OBSERVED)

        current_proof = self.ownership.current_proof()
        if current_proof != publication.guardian:
            return CleanupResult(outcome=CleanupOutcome.REFUSED_STALE_GUARDIAN)

        registration = self.registry.resolve_current(
            decision.resource.kind,
            decision.resource.resource_key,
            publication.registration_id,
        )
        if registration is None:
            return CleanupResult(outcome=CleanupOutcome.REFUSED_STALE_REGISTRATION)

        return CleanupResult(
            outcome=CleanupOutcome.AUTHORIZED,
            authorization=CleanupAuthorization(
                registration=registration,
                guardian=current_proof,
            ),
        )
