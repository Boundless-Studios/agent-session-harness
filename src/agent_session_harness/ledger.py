"""Append-only local lifecycle event ledger."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

from pydantic import ValidationError

from .activity import ActivitySnapshot, Quiescence
from .events import LifecycleEvent
from .models import EventType
from .secure_files import (
    append_private_text,
    exclusive_lock,
    private_exists,
    read_private_text,
)


class EventLedger:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def append(self, event: LifecycleEvent) -> None:
        encoded = json.dumps(
            event.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        with exclusive_lock(self.lock_path):
            append_private_text(self.path, encoded + "\n")

    def materialize(
        self,
        *,
        now: datetime,
        stale_after_seconds: float,
    ) -> ActivitySnapshot:
        events, warnings = self._read_events()
        seen: set[str] = set()
        active_turns: set[str] = set()
        active_tools: set[str] = set()
        active_subagents: set[str] = set()
        active_critical_sections: set[str] = set()
        last_event_at: datetime | None = None
        processed = 0

        starts = {
            EventType.TURN_STARTED: (active_turns, "turn"),
            EventType.TOOL_STARTED: (active_tools, "tool"),
            EventType.SUBAGENT_STARTED: (active_subagents, "subagent"),
            EventType.CRITICAL_ENTERED: (active_critical_sections, "critical section"),
        }
        finishes = {
            EventType.TURN_IDLE: (active_turns, "turn"),
            EventType.TOOL_FINISHED: (active_tools, "tool"),
            EventType.TOOL_FAILED: (active_tools, "tool"),
            EventType.SUBAGENT_FINISHED: (active_subagents, "subagent"),
            EventType.CRITICAL_EXITED: (active_critical_sections, "critical section"),
        }

        for event in events:
            if event.event_id in seen:
                continue
            seen.add(event.event_id)
            processed += 1
            if last_event_at is None or event.timestamp > last_event_at:
                last_event_at = event.timestamp

            if event.event_type in starts:
                target, _label = starts[event.event_type]
                target.add(event.activity_id or "")
            elif event.event_type in finishes:
                target, label = finishes[event.event_type]
                activity_id = event.activity_id or ""
                if activity_id not in target:
                    warnings.append(
                        f"{label} finish without start: {activity_id or 'missing'}"
                    )
                else:
                    target.remove(activity_id)

        active_groups = (
            active_turns,
            active_tools,
            active_subagents,
            active_critical_sections,
        )
        quiescence = self._quiescence(
            warnings=warnings,
            last_event_at=last_event_at,
            now=now,
            stale_after_seconds=stale_after_seconds,
            has_active=any(active_groups),
        )
        return ActivitySnapshot(
            quiescence=quiescence,
            active_turn_ids=frozenset(active_turns),
            active_tool_ids=frozenset(active_tools),
            active_subagent_ids=frozenset(active_subagents),
            active_critical_section_ids=frozenset(active_critical_sections),
            processed_event_count=processed,
            last_event_at=last_event_at,
            integrity_warnings=tuple(warnings),
        )

    def _read_events(self) -> tuple[list[LifecycleEvent], list[str]]:
        with exclusive_lock(self.lock_path):
            if not private_exists(self.path):
                return [], []
            lines = read_private_text(self.path).splitlines()

        events: list[LifecycleEvent] = []
        warnings: list[str] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"line {line_number}: invalid JSON")
                continue
            try:
                events.append(LifecycleEvent.model_validate(payload))
            except ValidationError:
                warnings.append(f"line {line_number}: invalid lifecycle event")
        return events, warnings

    @staticmethod
    def _quiescence(
        *,
        warnings: list[str],
        last_event_at: datetime | None,
        now: datetime,
        stale_after_seconds: float,
        has_active: bool,
    ) -> Quiescence:
        if warnings or last_event_at is None:
            return Quiescence.UNKNOWN
        age_seconds = (now - last_event_at).total_seconds()
        if age_seconds < 0 or age_seconds > stale_after_seconds:
            return Quiescence.UNKNOWN
        return Quiescence.BUSY if has_active else Quiescence.IDLE
