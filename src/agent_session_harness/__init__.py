"""Public package boundary for agent-session-harness."""

from .guardian_service import GuardianPublication, GuardianService
from .guardian_singleton import (
    DuplicateGuardianError,
    GuardianLeaseHandle,
    GuardianLeaseProof,
    GuardianOwnership,
    GuardianSingleton,
    StaleGuardianError,
)
from .process_identity import (
    ManagedResourceReference,
    ProcessEvidence,
    ProcessIdentity,
    ProcessInspector,
    ProcessObservation,
    ProcessPlatform,
    ProcessState,
    capture_process_identity,
    same_process,
)
from .resource_guardian import (
    GuardianAction,
    GuardianDecision,
    GuardianEvidence,
    GuardianObservation,
    GuardianReasonCode,
    LeaseState,
    ManagedOwnerState,
    ManagedResource,
    OwnerLeaseIdentity,
    ProcessIdentityState,
    WorktreeIdentity,
    WorktreeState,
    decide_guardian_action,
)
from .resource_registry import (
    RegistrationConflictError,
    ResourceRegistration,
    ResourceRegistry,
)

__version__ = "0.1.0"

__all__ = [
    "DuplicateGuardianError",
    "GuardianAction",
    "GuardianDecision",
    "GuardianEvidence",
    "GuardianLeaseHandle",
    "GuardianLeaseProof",
    "GuardianObservation",
    "GuardianOwnership",
    "GuardianPublication",
    "GuardianReasonCode",
    "GuardianService",
    "GuardianSingleton",
    "LeaseState",
    "ManagedOwnerState",
    "ManagedResource",
    "ManagedResourceReference",
    "OwnerLeaseIdentity",
    "ProcessEvidence",
    "ProcessIdentity",
    "ProcessIdentityState",
    "ProcessInspector",
    "ProcessObservation",
    "ProcessPlatform",
    "ProcessState",
    "RegistrationConflictError",
    "ResourceRegistration",
    "ResourceRegistry",
    "StaleGuardianError",
    "WorktreeIdentity",
    "WorktreeState",
    "__version__",
    "capture_process_identity",
    "decide_guardian_action",
    "same_process",
]
