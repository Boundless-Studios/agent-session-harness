"""Shared request/response handling for durable checkpoint adapters.

Every checkpoint adapter (beads, Linear, and anything a host adds) speaks the
same bounded JSON protocol, so validation, task-id resolution, and the
normalized success/failure shape live here exactly once.

Validation here is deliberately structural rather than a full
`HandoffCapsule` parse: an adapter must mirror and echo the fingerprint the
supervisor computed, never recompute or silently correct it.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Pattern

from .command import AdapterResponse, sanitize_error

MAX_INPUT_BYTES = 1_048_576
FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")

_CAPSULE_LINE_LIMITS = (
    ("chain_id", 160),
    ("objective", 4000),
    ("exact_next_action", 4000),
    ("head", 128),
)


@dataclass(frozen=True)
class CheckpointRequest:
    """One validated adapter request, reduced to what adapters actually use."""

    operation: str
    idempotency_key: str
    capsule: dict[str, Any]
    fingerprint: str
    task_id: str


def bounded_line(value: object, label: str, limit: int) -> str:
    """Collapse whitespace and require a bounded, non-empty single line."""

    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > limit:
        raise ValueError(f"{label} is empty or too long")
    return normalized


def validate_checkpoint_request(
    request: Mapping[str, object],
    *,
    task_id_keys: tuple[str, ...],
    task_id_label: str,
    task_id_pattern: Pattern[str] | None = None,
) -> CheckpointRequest:
    """Validate the shared envelope and resolve this adapter's task identity."""

    if not isinstance(request, Mapping) or request.get("schema_version") != 1:
        raise ValueError("request must use schema_version 1")
    operation = request.get("operation")
    if operation not in {"write", "read", "acknowledge"}:
        raise ValueError("request operation is invalid")
    idempotency_key = bounded_line(
        request.get("idempotency_key"), "idempotency_key", 256
    )
    capsule = request.get("capsule")
    if not isinstance(capsule, dict) or capsule.get("schema_version") != 1:
        raise ValueError("request capsule must use schema_version 1")
    fingerprint = str(capsule.get("fingerprint") or "")
    if not FINGERPRINT.fullmatch(fingerprint):
        raise ValueError("capsule fingerprint must be lowercase SHA-256")
    task_ids = capsule.get("task_ids")
    if not isinstance(task_ids, dict):
        raise ValueError("capsule task_ids must be an object")
    generation = capsule.get("target_generation")
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < 1
    ):
        raise ValueError("capsule target_generation must be a positive integer")
    for key, limit in _CAPSULE_LINE_LIMITS:
        bounded_line(capsule.get(key), f"capsule {key}", limit)
    return CheckpointRequest(
        operation=str(operation),
        idempotency_key=idempotency_key,
        capsule=capsule,
        fingerprint=fingerprint,
        task_id=_resolve_task_id(
            task_ids,
            keys=task_id_keys,
            label=task_id_label,
            pattern=task_id_pattern,
        ),
    )


def _resolve_task_id(
    task_ids: Mapping[str, object],
    *,
    keys: tuple[str, ...],
    label: str,
    pattern: Pattern[str] | None,
) -> str:
    raw: object = None
    for key in keys:
        if task_ids.get(key):
            raw = task_ids[key]
            break
    if pattern is None:
        return bounded_line(raw, label, 160)
    value = str(raw or "")
    if not pattern.fullmatch(value):
        raise ValueError(f"capsule requires a bounded {label}")
    return value


def success(fingerprint: str) -> dict[str, object]:
    """Return the normalized success response for a mirrored checkpoint."""

    return AdapterResponse(
        ok=True,
        fingerprint=fingerprint,
        retryable=False,
        error=None,
    ).model_dump(mode="json")


def failure(error: str, *, retryable: bool) -> dict[str, object]:
    """Return the normalized failure response, with diagnostics redacted."""

    return AdapterResponse(
        ok=False,
        fingerprint=None,
        retryable=retryable,
        error=sanitize_error(error),
    ).model_dump(mode="json")


def read_stdin_request() -> dict[str, object]:
    """Read one bounded JSON request object from standard input."""

    encoded = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(encoded) > MAX_INPUT_BYTES:
        raise ValueError("request exceeds input limit")
    request = json.loads(encoded)
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    return request


def emit_response(response: Mapping[str, object]) -> None:
    """Write one canonical adapter response to standard output."""

    json.dump(dict(response), sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
