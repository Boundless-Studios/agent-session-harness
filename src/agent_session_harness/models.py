"""Shared enums and small value objects."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Runtime(str, Enum):
    CLAUDE = "claude"
    CODEX = "codex"


class Confidence(str, Enum):
    CONFIDENT = "confident"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


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


class UsageSample(BaseModel):
    """Sanitized cumulative and current-context usage for one conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: Runtime
    conversation_id: str = Field(min_length=1)
    observed_at: datetime
    unique_messages: int = Field(ge=0)
    cumulative_input_tokens: int = Field(ge=0)
    cumulative_output_tokens: int = Field(ge=0)
    cumulative_cache_creation_tokens: int = Field(ge=0)
    cumulative_cache_read_tokens: int = Field(ge=0)
    latest_input_tokens: int = Field(ge=0)
    latest_output_tokens: int = Field(ge=0)
    latest_cache_creation_tokens: int = Field(ge=0)
    latest_cache_read_tokens: int = Field(ge=0)
    context_tokens: int = Field(ge=0)
    window_tokens: int = Field(gt=0)
    context_percent: float = Field(ge=0)
    confidence: Confidence
    message_keys: tuple[str, ...] = ()
    tool_counts: dict[str, int] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def cumulative_total_tokens(self) -> int:
        return (
            self.cumulative_input_tokens
            + self.cumulative_output_tokens
            + self.cumulative_cache_creation_tokens
            + self.cumulative_cache_read_tokens
        )
