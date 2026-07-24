"""Canonical, privacy-preserving handoff capsules."""

from __future__ import annotations

import hmac
import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

ProcessState = Literal[
    "idle",
    "running",
    "blocked",
    "unknown",
    "stopped",
    "passed",
    "failed",
]

_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:[a-z0-9]+[-_])*(?:api[-_]?key|authorization|credential|password|secret|token)"
    r"(?:[-_][a-z0-9]+)*"
    r"\s*[:=]\s*\S+"
)

_LIST_ITEM_LIMITS = {
    "completed_criteria": 1000,
    "remaining_criteria": 1000,
    "dirty_paths": 1024,
    "file_anchors": 1024,
    "symbol_anchors": 1024,
    "decisions": 1000,
    "blockers": 1000,
}

_MAPPING_ENTRY_LIMITS = {
    "task_ids": (64, 160),
    "test_results": (512, 512),
    "process_summaries": (64, 32),
}


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class HandoffCapsule(BaseModel):
    """The complete bounded state needed by one successor conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    chain_id: str = Field(min_length=1, max_length=160)
    predecessor_conversation_id: str = Field(min_length=1, max_length=160)
    target_generation: int = Field(ge=1)
    task_ids: dict[str, str] = Field(max_length=16)
    objective: str = Field(min_length=1, max_length=4000)
    exact_next_action: str = Field(min_length=1, max_length=4000)
    completed_criteria: tuple[str, ...] = Field(max_length=32)
    remaining_criteria: tuple[str, ...] = Field(max_length=32)
    repository_path: Path
    branch: str = Field(min_length=1, max_length=512)
    head: str = Field(min_length=1, max_length=128)
    dirty_paths: tuple[str, ...] = Field(max_length=128)
    file_anchors: tuple[str, ...] = Field(max_length=128)
    symbol_anchors: tuple[str, ...] = Field(max_length=128)
    test_results: dict[str, str] = Field(max_length=64)
    decisions: tuple[str, ...] = Field(max_length=32)
    blockers: tuple[str, ...] = Field(max_length=32)
    process_summaries: dict[str, ProcessState] = Field(max_length=16)
    created_at: datetime
    fingerprint: str = Field(default="", max_length=64)

    @field_validator("repository_path")
    @classmethod
    def require_absolute_repository_path(cls, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            raise ValueError("repository_path must be absolute")
        return path.resolve()

    @field_validator("repository_path", mode="before")
    @classmethod
    def bound_repository_path(cls, value: object) -> object:
        if len(str(value)) > 4096:
            raise ValueError("repository_path exceeds 4096 characters")
        return value

    @field_validator("created_at")
    @classmethod
    def require_aware_utc_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator(
        "chain_id",
        "predecessor_conversation_id",
        "task_ids",
        "objective",
        "exact_next_action",
        "completed_criteria",
        "remaining_criteria",
        "repository_path",
        "branch",
        "head",
        "dirty_paths",
        "file_anchors",
        "symbol_anchors",
        "decisions",
        "blockers",
        "test_results",
        "process_summaries",
        "fingerprint",
        mode="before",
    )
    @classmethod
    def reject_credential_shaped_text(cls, value: object) -> object:
        if isinstance(value, (str, Path)):
            candidates = (str(value),)
        elif isinstance(value, dict):
            candidates = tuple(
                candidate
                for key, item in value.items()
                for candidate in (str(key), str(item), f"{key}={item}")
            )
        elif isinstance(value, (list, tuple)):
            candidates = tuple(str(item) for item in value)
        else:
            return value
        if any(_CREDENTIAL_ASSIGNMENT.search(candidate) for candidate in candidates):
            raise ValueError("capsule must not contain credential-shaped text")
        return value

    @field_validator(
        "completed_criteria",
        "remaining_criteria",
        "dirty_paths",
        "file_anchors",
        "symbol_anchors",
        "decisions",
        "blockers",
    )
    @classmethod
    def reject_blank_list_items(
        cls, values: tuple[str, ...], info: ValidationInfo
    ) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("capsule list items must not be blank")
        limit = _LIST_ITEM_LIMITS[info.field_name]
        if any(len(value) > limit for value in values):
            raise ValueError(
                f"capsule {info.field_name} items exceed {limit} characters"
            )
        return values

    @field_validator("task_ids", "test_results", "process_summaries")
    @classmethod
    def reject_blank_mapping_entries(
        cls, values: dict[str, str], info: ValidationInfo
    ) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("capsule mapping entries must not be blank")
        key_limit, value_limit = _MAPPING_ENTRY_LIMITS[info.field_name]
        if any(
            len(key) > key_limit or len(value) > value_limit
            for key, value in values.items()
        ):
            raise ValueError(f"capsule {info.field_name} entries exceed their limits")
        return values

    @model_validator(mode="after")
    def calculate_and_verify_fingerprint(self) -> "HandoffCapsule":
        expected = sha256(self.fingerprint_payload_bytes()).hexdigest()
        if self.fingerprint and not hmac.compare_digest(self.fingerprint, expected):
            raise ValueError("capsule fingerprint does not match its canonical payload")
        object.__setattr__(self, "fingerprint", expected)
        return self

    def fingerprint_payload_bytes(self) -> bytes:
        """Return canonical JSON used as the fingerprint input."""

        return _canonical_bytes(self.model_dump(mode="json", exclude={"fingerprint"}))

    def canonical_bytes(self) -> bytes:
        """Return the complete capsule as compact deterministic JSON."""

        return _canonical_bytes(self.model_dump(mode="json"))
