"""Shared enums and small value objects."""

from __future__ import annotations

from enum import Enum


class Runtime(str, Enum):
    CLAUDE = "claude"
    CODEX = "codex"


class EventType(str, Enum):
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"
    TURN_STARTED = "turn.started"
    TURN_IDLE = "turn.idle"
    TOOL_STARTED = "tool.started"
    TOOL_FINISHED = "tool.finished"
    TOOL_FAILED = "tool.failed"
    SUBAGENT_STARTED = "subagent.started"
    SUBAGENT_FINISHED = "subagent.finished"
    PRE_COMPACT = "context.pre_compact"
    COMPACTED = "context.compacted"
    CRITICAL_ENTERED = "critical_section.entered"
    CRITICAL_EXITED = "critical_section.exited"
    HANDOFF_REQUESTED = "handoff.requested"
    HANDOFF_CHECKPOINTED = "handoff.checkpointed"
    HANDOFF_FENCED = "handoff.fenced"
    HANDOFF_ACKNOWLEDGED = "handoff.acknowledged"
