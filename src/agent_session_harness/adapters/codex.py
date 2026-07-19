"""Codex rollout accounting with fork-lineage baseline subtraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from ..models import Confidence, Runtime


class CodexSessionUsage(BaseModel):
    """Sanitized usage for one Codex session in a fork lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime: Runtime = Runtime.CODEX
    session_id: str = Field(min_length=1)
    parent_session_id: str | None = None
    started_at: datetime
    observed_at: datetime
    final_input_tokens: int = Field(ge=0)
    final_cached_input_tokens: int = Field(ge=0)
    final_output_tokens: int = Field(ge=0)
    final_reasoning_output_tokens: int = Field(ge=0)
    final_total_tokens: int = Field(ge=0)
    baseline_input_tokens: int | None = Field(default=None, ge=0)
    baseline_cached_input_tokens: int | None = Field(default=None, ge=0)
    baseline_output_tokens: int | None = Field(default=None, ge=0)
    baseline_reasoning_output_tokens: int | None = Field(default=None, ge=0)
    baseline_total_tokens: int | None = Field(default=None, ge=0)
    incremental_input_tokens: int | None = Field(default=None, ge=0)
    incremental_cached_input_tokens: int | None = Field(default=None, ge=0)
    incremental_output_tokens: int | None = Field(default=None, ge=0)
    incremental_reasoning_output_tokens: int | None = Field(default=None, ge=0)
    incremental_total_tokens: int | None = Field(default=None, ge=0)
    context_tokens: int = Field(ge=0)
    window_tokens: int = Field(gt=0)
    context_percent: float = Field(ge=0)
    confidence: Confidence
    warnings: tuple[str, ...] = ()


class CodexLineageUsage(BaseModel):
    """Ordered sessions and corrected totals for one Codex fork lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sessions: tuple[CodexSessionUsage, ...]
    naive_total_tokens: int = Field(ge=0)
    incremental_total_tokens: int | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class _TokenDimensions:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_payload(cls, payload: object) -> _TokenDimensions:
        values = payload if isinstance(payload, dict) else {}
        return cls(
            input_tokens=_nonnegative_int(values.get("input_tokens")),
            cached_input_tokens=_nonnegative_int(values.get("cached_input_tokens")),
            output_tokens=_nonnegative_int(values.get("output_tokens")),
            reasoning_output_tokens=_nonnegative_int(
                values.get("reasoning_output_tokens")
            ),
            total_tokens=_nonnegative_int(values.get("total_tokens")),
        )


@dataclass(frozen=True)
class _TokenEvent:
    observed_at: datetime
    cumulative: _TokenDimensions
    latest: _TokenDimensions
    window_tokens: int


@dataclass(frozen=True)
class _RawSession:
    path: Path
    session_id: str
    parent_session_id: str | None
    started_at: datetime
    events: tuple[_TokenEvent, ...]
    warnings: tuple[str, ...]


class CodexUsageReader:
    """Read only metadata and token-count events from Codex rollout files."""

    def read_file(self, path: str | os.PathLike[str]) -> CodexSessionUsage:
        return self._to_usage(self._read_raw(Path(path)))

    def read_lineage(
        self, paths: Iterable[str | os.PathLike[str]]
    ) -> CodexLineageUsage:
        raw_sessions = tuple(self._read_raw(Path(path)) for path in paths)
        by_id: dict[str, _RawSession] = {}
        for session in raw_sessions:
            if session.session_id in by_id:
                raise ValueError(f"duplicate Codex session ID: {session.session_id}")
            by_id[session.session_id] = session

        ordered: list[_RawSession] = []
        visit_state: dict[str, int] = {}

        def visit(session: _RawSession) -> None:
            state = visit_state.get(session.session_id, 0)
            if state == 1:
                raise ValueError("Codex session lineage contains a cycle")
            if state == 2:
                return
            visit_state[session.session_id] = 1
            if session.parent_session_id in by_id:
                visit(by_id[session.parent_session_id])
            visit_state[session.session_id] = 2
            ordered.append(session)

        for raw in sorted(
            raw_sessions, key=lambda item: (item.started_at, item.session_id)
        ):
            visit(raw)

        sessions = tuple(self._to_usage(raw) for raw in ordered)
        corrected = [item.incremental_total_tokens for item in sessions]
        return CodexLineageUsage(
            sessions=sessions,
            naive_total_tokens=sum(item.final_total_tokens for item in sessions),
            incremental_total_tokens=(
                sum(value for value in corrected if value is not None)
                if all(value is not None for value in corrected)
                else None
            ),
        )

    def _read_raw(self, path: Path) -> _RawSession:
        session_id: str | None = None
        parent_session_id: str | None = None
        started_at: datetime | None = None
        events: list[_TokenEvent] = []
        warnings: list[str] = []

        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                if (
                    b'"session_meta"' not in raw_line
                    and b'"token_count"' not in raw_line
                ):
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    warnings.append(f"invalid usage JSON at byte offset {offset}")
                    continue
                if not isinstance(row, dict):
                    continue
                payload = row.get("payload")
                if not isinstance(payload, dict):
                    continue
                if row.get("type") == "session_meta":
                    raw_id = payload.get("id")
                    if raw_id:
                        session_id = str(raw_id)
                    started_at = _timestamp(
                        payload.get("timestamp") or row.get("timestamp"), path
                    )
                    parent_session_id = _parent_id(payload.get("source"))
                    continue
                if (
                    row.get("type") != "event_msg"
                    or payload.get("type") != "token_count"
                ):
                    continue
                info = payload.get("info")
                if not isinstance(info, dict):
                    warnings.append(f"missing token info at byte offset {offset}")
                    continue
                window_tokens = _nonnegative_int(info.get("model_context_window"))
                if window_tokens == 0:
                    warnings.append(f"missing context window at byte offset {offset}")
                    continue
                events.append(
                    _TokenEvent(
                        observed_at=_timestamp(row.get("timestamp"), path),
                        cumulative=_TokenDimensions.from_payload(
                            info.get("total_token_usage")
                        ),
                        latest=_TokenDimensions.from_payload(
                            info.get("last_token_usage")
                        ),
                        window_tokens=window_tokens,
                    )
                )

        fallback_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        return _RawSession(
            path=path,
            session_id=session_id or path.stem,
            parent_session_id=parent_session_id,
            started_at=started_at or fallback_time,
            events=tuple(sorted(events, key=lambda event: event.observed_at)),
            warnings=tuple(warnings),
        )

    def _to_usage(self, raw: _RawSession) -> CodexSessionUsage:
        warnings = list(raw.warnings)
        confidence = Confidence.CONFIDENT
        if not raw.events:
            warnings.append("no token-count events found")
            confidence = Confidence.UNKNOWN
            final_event = _TokenEvent(
                observed_at=raw.started_at,
                cumulative=_TokenDimensions(),
                latest=_TokenDimensions(),
                window_tokens=1,
            )
        else:
            final_event = raw.events[-1]

        baseline: _TokenDimensions | None
        if raw.parent_session_id is None:
            baseline = _TokenDimensions()
        else:
            inherited = [
                event for event in raw.events if event.observed_at < raw.started_at
            ]
            baseline = inherited[-1].cumulative if inherited else None
            if baseline is None:
                warnings.append(
                    "fork baseline is unavailable; incremental usage unknown"
                )
                confidence = Confidence.DEGRADED

        if warnings and confidence is Confidence.CONFIDENT:
            confidence = Confidence.DEGRADED
        final = final_event.cumulative
        incremental = _subtract(final, baseline)
        if baseline is not None and incremental is None:
            warnings.append("cumulative usage fell below fork baseline")
            confidence = Confidence.DEGRADED

        return CodexSessionUsage(
            session_id=raw.session_id,
            parent_session_id=raw.parent_session_id,
            started_at=raw.started_at,
            observed_at=final_event.observed_at,
            final_input_tokens=final.input_tokens,
            final_cached_input_tokens=final.cached_input_tokens,
            final_output_tokens=final.output_tokens,
            final_reasoning_output_tokens=final.reasoning_output_tokens,
            final_total_tokens=final.total_tokens,
            baseline_input_tokens=_dimension(baseline, "input_tokens"),
            baseline_cached_input_tokens=_dimension(baseline, "cached_input_tokens"),
            baseline_output_tokens=_dimension(baseline, "output_tokens"),
            baseline_reasoning_output_tokens=_dimension(
                baseline, "reasoning_output_tokens"
            ),
            baseline_total_tokens=_dimension(baseline, "total_tokens"),
            incremental_input_tokens=_dimension(incremental, "input_tokens"),
            incremental_cached_input_tokens=_dimension(
                incremental, "cached_input_tokens"
            ),
            incremental_output_tokens=_dimension(incremental, "output_tokens"),
            incremental_reasoning_output_tokens=_dimension(
                incremental, "reasoning_output_tokens"
            ),
            incremental_total_tokens=_dimension(incremental, "total_tokens"),
            context_tokens=final_event.latest.total_tokens,
            window_tokens=final_event.window_tokens,
            context_percent=(
                100.0 * final_event.latest.total_tokens / final_event.window_tokens
            ),
            confidence=confidence,
            warnings=tuple(warnings),
        )


def _subtract(
    final: _TokenDimensions, baseline: _TokenDimensions | None
) -> _TokenDimensions | None:
    if baseline is None:
        return None
    values = {
        name: getattr(final, name) - getattr(baseline, name)
        for name in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        )
    }
    if any(value < 0 for value in values.values()):
        return None
    return _TokenDimensions(**values)


def _dimension(value: _TokenDimensions | None, name: str) -> int | None:
    return getattr(value, name) if value is not None else None


def _nonnegative_int(value: object) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _parent_id(source: object) -> str | None:
    if not isinstance(source, dict):
        return None
    subagent = source.get("subagent")
    if not isinstance(subagent, dict):
        return None
    thread_spawn = subagent.get("thread_spawn")
    if not isinstance(thread_spawn, dict):
        return None
    raw_parent = thread_spawn.get("parent_thread_id")
    return str(raw_parent) if raw_parent else None


def _timestamp(value: Any, path: Path) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
