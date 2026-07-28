"""Materialized runtime activity and quiescence state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Quiescence(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    UNKNOWN = "unknown"


class RuntimeLiveness(str, Enum):
    """Whether the runtime's own hooks are still reporting at all.

    BOU-2222: quiescence answers "is work outstanding", and it collapses every
    way of not knowing into UNKNOWN. A session whose hooks broke -- never
    installed, misconfigured, crashed bridge -- is therefore indistinguishable
    from a session that is merely between turns, and it parks in DRAINING
    forever with no finding to show for it. Liveness answers the separate
    question "is the reporting path itself working", so that silence can be
    named as a fault instead of read as calm.
    """

    REPORTING = "reporting"
    NEVER_REPORTED = "never_reported"
    SILENT_IDLE = "silent_idle"
    SILENT_ACTIVE = "silent_active"

    @property
    def is_faulted(self) -> bool:
        return self is not RuntimeLiveness.REPORTING


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
    handoff_requested_generations: frozenset[int] = frozenset()
    # Generations in which the runtime announced its OWN context compaction
    # (`context.pre_compact`). This is the only context signal that does not
    # depend on usage sampling, so it is the one that can release a drain when
    # sampling has gone non-confident (BOU-2565).
    pre_compact_generations: frozenset[int] = frozenset()
    # How many compaction events the ledger has EVER reported. Membership above
    # is a fold over retained history, so it stays true forever once a
    # generation has compacted once — it answers "has this generation ever
    # compacted", never "is this a new signal". A consumer that must act once
    # per event needs a watermark it can compare against what it already acted
    # on; this is that watermark. Monotonic and bounded (BOU-2565 review).
    pre_compact_seen: int = 0
    # Same watermark treatment for handoff requests. `handoff_requested_generations`
    # is also a retained fold, so consent given BEFORE a self-compaction would
    # otherwise still authorise a rotation AFTER it — in the same generation, off
    # a request the runtime made about different work (BOU-2565 review).
    handoff_requested_seen: int = 0
    runtime_liveness: RuntimeLiveness = RuntimeLiveness.REPORTING
    # Tool starts closed by turn-idle reconciliation rather than by their own
    # finish event (BOU-2236). Reported for observability only -- these are NOT
    # counted as active. A non-empty set here is the fingerprint of a permission
    # gate that denies calls at PreToolUse, which is normal, not a fault.
    reaped_tool_ids: frozenset[str] = frozenset()

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
