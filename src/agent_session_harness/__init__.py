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

__version__ = "0.1.0"

__all__ = [
    "ManagedResourceReference",
    "ProcessEvidence",
    "ProcessIdentity",
    "ProcessInspector",
    "ProcessObservation",
    "ProcessPlatform",
    "ProcessState",
    "__version__",
    "capture_process_identity",
    "same_process",
]
