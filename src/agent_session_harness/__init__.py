"""Public package boundary for agent-session-harness."""

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

__version__ = "0.1.0"

__all__ = [
    "GuardianAction",
    "GuardianDecision",
    "GuardianEvidence",
    "GuardianObservation",
    "GuardianReasonCode",
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
    "WorktreeIdentity",
    "WorktreeState",
    "__version__",
    "capture_process_identity",
    "decide_guardian_action",
    "same_process",
]
