"""Append-only local lifecycle event ledger."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

from pydantic import ValidationError

from .activity import ActivitySnapshot, Quiescence, RuntimeLiveness
from .events import LifecycleEvent
from .models import EventType
from .secure_files import (
    append_private_text,
    exclusive_lock,
    private_file_size,
    private_exists,
    read_private_text_incremental,
)


MAX_LEDGER_BYTES = 16 * 1_048_576
MAX_LEDGER_EVENTS = 50_000
MAX_EVENT_BYTES = 64 * 1024
MAX_RETAINED_WARNINGS = 256

# BOU-2222: the supervisor appends these itself, so they are proof that the
# supervisor is alive, never that the runtime's hooks are. A successor whose
# hooks are completely dead still has an acknowledgement in its ledger, and
# counting it would hide the fault from the first rotation onwards.
SUPERVISOR_AUTHORED_EVENT_TYPES = frozenset(
    {
        EventType.HANDOFF_CHECKPOINTED,
        EventType.HANDOFF_ACKNOWLEDGED,
        EventType.HANDOFF_FENCED,
    }
)


@dataclass(frozen=True)
class _IntegrityWarning:
    """One ledger integrity finding plus how long it may gate quiescence.

    BOU-2208: warnings used to be a flat list of strings that gated quiescence
    for the supervisor's whole life, because the cache is only cleared when the
    file is replaced. One unparseable line or one unmatched finish therefore
    pinned quiescence to UNKNOWN forever, and `DRAINING` only leaves for `IDLE`,
    so the session sat above the rotate threshold and never rotated.

    A parse problem says "some events near here may be missing right now", not
    "this ledger is permanently untrustworthy". Transient findings keep gating
    only while they are within the same staleness window the rest of quiescence
    uses; a genuinely truncated or unreadable read stays sticky.
    """

    message: str
    observed_at: datetime
    sticky: bool = False

    def gates(self, *, now: datetime, stale_after_seconds: float) -> bool:
        if self.sticky:
            return True
        return (now - self.observed_at).total_seconds() <= stale_after_seconds


class EventLedger:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._file_identity: tuple[int, int] | None = None
        self._read_offset = 0
        self._line_count = 0
        self._cached_events: list[LifecycleEvent] = []
        self._cached_warnings: list[_IntegrityWarning] = []

    def append(self, event: LifecycleEvent) -> None:
        encoded = json.dumps(
            event.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        encoded_bytes = (encoded + "\n").encode("utf-8")
        if len(encoded_bytes) > MAX_EVENT_BYTES:
            raise ValueError("lifecycle event exceeds byte limit")
        with exclusive_lock(self.lock_path):
            current_size = (
                private_file_size(self.path) if private_exists(self.path) else 0
            )
            if current_size + len(encoded_bytes) > MAX_LEDGER_BYTES:
                raise ValueError("lifecycle ledger exceeds byte limit")
            append_private_text(self.path, encoded_bytes.decode("utf-8"))

    def materialize(
        self,
        *,
        now: datetime,
        stale_after_seconds: float,
    ) -> ActivitySnapshot:
        events, warnings = self._read_events(now=now)
        seen: set[str] = set()
        # Counted, not set-tracked: when a runtime supplies no tool-use id the
        # hook derives one from the call's own fields, so two identical calls in
        # one prompt share an id. Under set semantics the second finish would
        # look like a finish-without-start, and that warning latches quiescence
        # to UNKNOWN for the life of the supervisor -- rotation would never run.
        # Counting makes repeats balance exactly. Quiescence only asks whether
        # anything is outstanding, so this does not change its meaning.
        active_turns: Counter[str] = Counter()
        active_tools: Counter[str] = Counter()
        active_subagents: Counter[str] = Counter()
        active_critical_sections: Counter[str] = Counter()
        handoff_requested_generations: set[int] = set()
        last_event_at: datetime | None = None
        last_hook_event_at: datetime | None = None
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
            if event.event_type not in SUPERVISOR_AUTHORED_EVENT_TYPES and (
                last_hook_event_at is None or event.timestamp > last_hook_event_at
            ):
                last_hook_event_at = event.timestamp

            if event.event_type in starts:
                target, _label = starts[event.event_type]
                target[event.activity_id or ""] += 1
            elif event.event_type in finishes:
                target, label = finishes[event.event_type]
                activity_id = event.activity_id or ""
                if target[activity_id] <= 0:
                    # Counter lookups insert a zero key; drop it so an unmatched
                    # finish cannot leave phantom outstanding work behind.
                    target.pop(activity_id, None)
                    # Stamped with the event's own time, not the read time: this
                    # warning is re-derived from retained history on every
                    # materialize, so a read-time stamp would keep it forever
                    # fresh and permanently gate quiescence.
                    warnings.append(
                        _IntegrityWarning(
                            message=(
                                f"{label} finish without start: "
                                f"{activity_id or 'missing'}"
                            ),
                            observed_at=event.timestamp,
                        )
                    )
                else:
                    target[activity_id] -= 1
                    if target[activity_id] == 0:
                        del target[activity_id]
            elif event.event_type is EventType.HANDOFF_REQUESTED:
                handoff_requested_generations.add(event.generation)

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
            runtime_liveness=self._runtime_liveness(
                last_hook_event_at=last_hook_event_at,
                now=now,
                stale_after_seconds=stale_after_seconds,
                has_active=any(active_groups),
            ),
            active_turn_ids=frozenset(active_turns),
            active_tool_ids=frozenset(active_tools),
            active_subagent_ids=frozenset(active_subagents),
            active_critical_section_ids=frozenset(active_critical_sections),
            processed_event_count=processed,
            last_event_at=last_event_at,
            integrity_warnings=tuple(warning.message for warning in warnings),
            handoff_requested_generations=frozenset(handoff_requested_generations),
        )

    def _read_events(
        self,
        *,
        now: datetime,
    ) -> tuple[list[LifecycleEvent], list[_IntegrityWarning]]:
        with exclusive_lock(self.lock_path):
            if not private_exists(self.path):
                self._reset_cache()
                return [], []
            try:
                tail, offset, identity, reset = read_private_text_incremental(
                    self.path,
                    offset=self._read_offset,
                    expected_identity=self._file_identity,
                    max_bytes=MAX_LEDGER_BYTES,
                )
            except (UnicodeDecodeError, ValueError):
                # Recomputed on every call rather than cached, so it clears by
                # itself as soon as the file becomes readable again.
                return [], [
                    _IntegrityWarning(
                        message="lifecycle ledger exceeds bounds or is unreadable",
                        observed_at=now,
                        sticky=True,
                    )
                ]

        if reset:
            self._reset_cache()
        lines = tail.splitlines()
        if tail and not tail.endswith("\n"):
            self._record_warning("lifecycle ledger has a partial final line", now=now)
        for line_number, line in enumerate(lines, start=self._line_count + 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                self._record_warning(f"line {line_number}: invalid JSON", now=now)
                continue
            try:
                self._cached_events.append(LifecycleEvent.model_validate(payload))
            except ValidationError:
                self._record_warning(
                    f"line {line_number}: invalid lifecycle event",
                    now=now,
                )
            if len(self._cached_events) > MAX_LEDGER_EVENTS:
                # A truncated read really is permanently untrustworthy: the
                # events beyond the bound are never materialized, so outstanding
                # work can be invisible for as long as this ledger is in use.
                self._record_warning(
                    "lifecycle ledger exceeds event limit",
                    now=now,
                    sticky=True,
                )
                break
        self._line_count += len(lines)
        self._read_offset = offset
        self._file_identity = identity
        return list(self._cached_events), list(self._cached_warnings)

    def _record_warning(
        self,
        message: str,
        *,
        now: datetime,
        sticky: bool = False,
    ) -> None:
        # Refresh rather than drop a repeat: "partial final line" can be
        # observed again on a later read, and a re-observed finding must start
        # gating again instead of inheriting an already-expired timestamp.
        for index, cached in enumerate(self._cached_warnings):
            if cached.message == message:
                self._cached_warnings[index] = _IntegrityWarning(
                    message=message,
                    observed_at=now,
                    sticky=cached.sticky or sticky,
                )
                return
        self._cached_warnings.append(
            _IntegrityWarning(message=message, observed_at=now, sticky=sticky)
        )
        if len(self._cached_warnings) > MAX_RETAINED_WARNINGS:
            transient = next(
                (
                    index
                    for index, cached in enumerate(self._cached_warnings)
                    if not cached.sticky
                ),
                None,
            )
            del self._cached_warnings[0 if transient is None else transient]

    def _reset_cache(self) -> None:
        self._file_identity = None
        self._read_offset = 0
        self._line_count = 0
        self._cached_events.clear()
        self._cached_warnings.clear()

    @staticmethod
    def _runtime_liveness(
        *,
        last_hook_event_at: datetime | None,
        now: datetime,
        stale_after_seconds: float,
        has_active: bool,
    ) -> RuntimeLiveness:
        """Classify the reporting path itself, independently of quiescence.

        Integrity warnings are deliberately not consulted: an unparseable line
        says the ledger is briefly untrustworthy, not that the runtime stopped
        talking. Folding them in here would re-create the BOU-2208 latch under
        a new name.
        """
        if last_hook_event_at is None:
            return RuntimeLiveness.NEVER_REPORTED
        age_seconds = (now - last_hook_event_at).total_seconds()
        if age_seconds <= stale_after_seconds:
            # Includes a negative age from clock skew: the events are fresh
            # enough to prove the hooks are alive, and quiescence separately
            # refuses to trust their ordering.
            return RuntimeLiveness.REPORTING
        return (
            RuntimeLiveness.SILENT_ACTIVE if has_active else RuntimeLiveness.SILENT_IDLE
        )

    @staticmethod
    def _quiescence(
        *,
        warnings: list[_IntegrityWarning],
        last_event_at: datetime | None,
        now: datetime,
        stale_after_seconds: float,
        has_active: bool,
    ) -> Quiescence:
        gating = any(
            warning.gates(now=now, stale_after_seconds=stale_after_seconds)
            for warning in warnings
        )
        if gating or last_event_at is None:
            return Quiescence.UNKNOWN
        age_seconds = (now - last_event_at).total_seconds()
        if age_seconds < 0 or age_seconds > stale_after_seconds:
            return Quiescence.UNKNOWN
        return Quiescence.BUSY if has_active else Quiescence.IDLE
