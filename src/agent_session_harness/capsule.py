"""Canonical, privacy-preserving handoff capsules."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import hmac
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    task_ids: dict[str, str]
    objective: str = Field(min_length=1, max_length=4000)
    exact_next_action: str = Field(min_length=1, max_length=4000)
    completed_criteria: tuple[str, ...]
    remaining_criteria: tuple[str, ...]
    repository_path: Path
    branch: str = Field(min_length=1, max_length=512)
    head: str = Field(min_length=1, max_length=128)
    dirty_paths: tuple[str, ...]
    file_anchors: tuple[str, ...]
    symbol_anchors: tuple[str, ...]
    test_results: dict[str, str]
    decisions: tuple[str, ...]
    blockers: tuple[str, ...]
    process_summaries: dict[str, str]
    created_at: datetime
    fingerprint: str = Field(default="", max_length=64)

    @field_validator("repository_path")
    @classmethod
    def require_absolute_repository_path(cls, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            raise ValueError("repository_path must be absolute")
        return path.resolve()

    @field_validator("created_at")
    @classmethod
    def require_aware_utc_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(timezone.utc)

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
    def reject_blank_list_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("capsule list items must not be blank")
        return values

    @field_validator("task_ids", "test_results", "process_summaries")
    @classmethod
    def reject_blank_mapping_entries(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("capsule mapping entries must not be blank")
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
