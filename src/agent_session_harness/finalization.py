"""Durable, provider-neutral session finalization state."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .secure_files import (
    atomic_write_private_text,
    exclusive_lock,
    private_exists,
    read_private_text,
)


class FinalizationPhase(StrEnum):
    ACTIVE = "active"
    BLOCKED = "blocked"
    FINALIZED = "finalized"


class FinalizationRecord(BaseModel):
    """One session's idempotent finalization progress."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    session_id: str = Field(min_length=1, max_length=160)
    summary: str = Field(min_length=1, max_length=20_000)
    phase: FinalizationPhase = FinalizationPhase.ACTIVE
    pending_block: str | None = Field(default=None, max_length=20_000)
    block_dispatch_id: str | None = Field(default=None, max_length=160)
    retro_submitted: bool = False
    summary_surfaced: bool = False

    @model_validator(mode="after")
    def require_consistent_block(self) -> Self:
        if (self.pending_block is None) != (self.block_dispatch_id is None):
            raise ValueError("pending block and dispatch id must be set together")
        if self.phase is FinalizationPhase.BLOCKED and self.pending_block is None:
            raise ValueError("blocked finalization requires a pending block")
        return self


class FinalizationStore:
    """Atomically persist one finalization record across runtime replacement."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def load(self) -> FinalizationRecord:
        if not private_exists(self.path):
            raise FileNotFoundError(self.path)
        return FinalizationRecord.model_validate_json(read_private_text(self.path))

    def begin(self, session_id: str, summary: str) -> FinalizationRecord:
        normalized_summary = summary.strip()
        candidate = FinalizationRecord(
            session_id=session_id,
            summary=normalized_summary,
        )
        with exclusive_lock(self.lock_path):
            if private_exists(self.path):
                current = self.load()
                if current.session_id != session_id or current.summary != normalized_summary:
                    raise ValueError("finalization already belongs to another request")
                return current
            self._write(candidate)
            return candidate

    def record_block(self, dispatch_id: str, message: str) -> FinalizationRecord:
        def transition(current: FinalizationRecord) -> dict[str, object]:
            if current.phase is FinalizationPhase.FINALIZED:
                raise ValueError("cannot record a block after finalization")
            return {
                "phase": FinalizationPhase.BLOCKED,
                "pending_block": message.strip(),
                "block_dispatch_id": dispatch_id,
            }

        return self._transition(transition)

    def acknowledge_block(self, dispatch_id: str) -> FinalizationRecord:
        def transition(current: FinalizationRecord) -> dict[str, object]:
            if current.block_dispatch_id != dispatch_id:
                return {}
            return {
                "phase": FinalizationPhase.ACTIVE,
                "pending_block": None,
                "block_dispatch_id": None,
            }

        return self._transition(transition)

    def mark_retro_submitted(self) -> FinalizationRecord:
        return self._update(retro_submitted=True)

    def mark_summary_surfaced(self) -> FinalizationRecord:
        return self._update(summary_surfaced=True)

    def finalize(self) -> FinalizationRecord:
        def transition(current: FinalizationRecord) -> dict[str, object]:
            if current.pending_block is not None:
                raise ValueError("cannot finalize with a pending block")
            if not current.retro_submitted or not current.summary_surfaced:
                raise ValueError("cannot finalize before retro and summary are complete")
            return {"phase": FinalizationPhase.FINALIZED}

        return self._transition(transition)

    def _update(self, **changes: object) -> FinalizationRecord:
        return self._transition(lambda _current: changes)

    def _transition(
        self,
        transition: Callable[[FinalizationRecord], dict[str, object]],
    ) -> FinalizationRecord:
        with exclusive_lock(self.lock_path):
            current = self.load()
            changes = transition(current)
            updated = FinalizationRecord.model_validate(
                {**current.model_dump(mode="python"), **changes}
            )
            self._write(updated)
            return updated

    def _write(self, record: FinalizationRecord) -> None:
        encoded = json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        atomic_write_private_text(self.path, encoded + "\n")
