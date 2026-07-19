"""Materialized runtime activity and quiescence state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Quiescence(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ActivitySnapshot:
    quiescence: Quiescence
    active_turn_ids: frozenset[str]
    active_tool_ids: frozenset[str]
    active_subagent_ids: frozenset[str]
    active_critical_section_ids: frozenset[str]
    processed_event_count: int
    last_event_at: datetime | None
    integrity_warnings: tuple[str, ...]

    @property
    def active_count(self) -> int:
        return sum(
            len(values)
            for values in (
                self.active_turn_ids,
                self.active_tool_ids,
                self.active_subagent_ids,
                self.active_critical_section_ids,
            )
        )
