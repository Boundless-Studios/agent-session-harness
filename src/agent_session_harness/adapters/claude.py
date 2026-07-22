"""Claude Code usage accounting without retaining conversation text."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..models import Confidence, Runtime, UsageSample
from .discovery import iter_files


class ClaudeDiscovery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[Path, ...]
    ambiguous: bool
    error: str | None = None


# Context window per model identity, as recorded in `message.model`. This is a
# fallback when the caller has no authoritative launch/config window: Claude
# Code omits the `[1m]` selection from rollout model names, so a bare identity
# alone cannot distinguish 200k from 1M context.
_BASE_CONTEXT_WINDOW_TOKENS = 200_000
_LONG_CONTEXT_SUFFIX = "[1m]"
_LONG_CONTEXT_WINDOW_TOKENS = 1_000_000

CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": _BASE_CONTEXT_WINDOW_TOKENS,
    "claude-opus-4-8": _BASE_CONTEXT_WINDOW_TOKENS,
    "claude-sonnet-5": _BASE_CONTEXT_WINDOW_TOKENS,
    "claude-fable-5": _BASE_CONTEXT_WINDOW_TOKENS,
    "claude-haiku-4-5": _BASE_CONTEXT_WINDOW_TOKENS,
}


def resolve_window_tokens(model: str | None) -> int | None:
    """Context window for a model identity, or None when it is unrecognized.

    Handles the `[1m]` long-context suffix generically so a newly released
    long-context variant resolves correctly without a table entry.
    """
    if not model:
        return None
    normalized = model.strip()
    if not normalized:
        return None
    if normalized.endswith(_LONG_CONTEXT_SUFFIX):
        return _LONG_CONTEXT_WINDOW_TOKENS
    exact = CONTEXT_WINDOWS.get(normalized)
    if exact is not None:
        return exact
    return None


@dataclass
class _MessageUsage:
    key: str
    order: int
    observed_at: datetime
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    model: str | None = None
    tools: set[tuple[str, str]] = field(default_factory=set)
    # True when the row carried no `message.id` and the key had to be derived
    # from its byte offset, so this record cannot be deduplicated reliably.
    derived_key: bool = False

    def merge(
        self,
        *,
        observed_at: datetime,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int,
        tools: set[tuple[str, str]],
        model: str | None = None,
    ) -> None:
        # A later row for the same message id carries the authoritative model.
        if model:
            self.model = model
        self.observed_at = max(self.observed_at, observed_at)
        self.input_tokens = max(self.input_tokens, input_tokens)
        self.output_tokens = max(self.output_tokens, output_tokens)
        self.cache_creation_tokens = max(
            self.cache_creation_tokens, cache_creation_tokens
        )
        self.cache_read_tokens = max(self.cache_read_tokens, cache_read_tokens)
        self.tools.update(tools)


class ClaudeUsageReader:
    def __init__(self, *, window_tokens: int | None):
        """Read Claude usage with an optional authoritative context window.

        Claude Code omits context-variant suffixes from rollout model names, so
        an explicit launch/config window wins when present. Without one, the
        latest rollout model is used as a fallback.
        """
        if window_tokens is not None and window_tokens <= 0:
            raise ValueError("window_tokens must be positive")
        self.window_tokens = window_tokens

    def read_file(self, path: str | os.PathLike[str]) -> UsageSample:
        rollout_path = Path(path)
        records: dict[str, _MessageUsage] = {}
        conversation_id: str | None = None
        warnings: list[str] = []
        # BOU-2208: a single sticky `degraded` flag never recovered, and the
        # whole rollout is re-read on every sample, so one unreadable row pinned
        # every future sample to DEGRADED — which the supervisor discards
        # entirely, so context percent stopped updating and rotation never
        # fired. Split the flag by what the finding actually invalidates.
        structurally_degraded = False
        stale_tail = False

        with rollout_path.open("rb") as handle:
            order = 0
            while True:
                offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                try:
                    payload = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    warnings.append(f"invalid JSON at byte offset {offset}")
                    stale_tail = True
                    continue
                if not isinstance(payload, dict) or payload.get("type") != "assistant":
                    continue
                message = payload.get("message")
                if not isinstance(message, dict):
                    warnings.append(
                        f"missing assistant message at byte offset {offset}"
                    )
                    stale_tail = True
                    continue
                usage = message.get("usage")
                if not isinstance(usage, dict):
                    warnings.append(f"missing usage at byte offset {offset}")
                    stale_tail = True
                    continue

                row_conversation = str(payload.get("sessionId") or rollout_path.stem)
                if conversation_id is None:
                    conversation_id = row_conversation
                elif conversation_id != row_conversation:
                    # Structural: no later row can make a mixed rollout coherent.
                    structurally_degraded = True
                    warnings.append("multiple conversation IDs in one rollout")

                raw_message_id = message.get("id")
                derived_key = not raw_message_id
                if raw_message_id:
                    key = f"message:{row_conversation}:{raw_message_id}"
                else:
                    key = f"offset:{offset}"
                    warnings.append(f"missing message id at byte offset {offset}")
                stale_tail = False

                observed_at = self._timestamp(payload.get("timestamp"), rollout_path)
                values = self._usage_values(usage)
                tools = self._safe_tools(message.get("content"), key)
                raw_model = message.get("model")
                row_model = raw_model if isinstance(raw_model, str) else None
                existing = records.get(key)
                if existing is None:
                    records[key] = _MessageUsage(
                        key=key,
                        order=order,
                        observed_at=observed_at,
                        input_tokens=values[0],
                        output_tokens=values[1],
                        cache_creation_tokens=values[2],
                        cache_read_tokens=values[3],
                        model=row_model,
                        tools=tools,
                        derived_key=derived_key,
                    )
                    order += 1
                else:
                    existing.merge(
                        observed_at=observed_at,
                        input_tokens=values[0],
                        output_tokens=values[1],
                        cache_creation_tokens=values[2],
                        cache_read_tokens=values[3],
                        tools=tools,
                        model=row_model,
                    )

        ordered = sorted(records.values(), key=lambda item: item.order)
        if not ordered:
            observed_at = datetime.fromtimestamp(
                rollout_path.stat().st_mtime, tz=timezone.utc
            )
            return UsageSample(
                runtime=Runtime.CLAUDE,
                conversation_id=conversation_id or rollout_path.stem,
                observed_at=observed_at,
                unique_messages=0,
                cumulative_input_tokens=0,
                cumulative_output_tokens=0,
                cumulative_cache_creation_tokens=0,
                cumulative_cache_read_tokens=0,
                latest_input_tokens=0,
                latest_output_tokens=0,
                latest_cache_creation_tokens=0,
                latest_cache_read_tokens=0,
                context_tokens=0,
                window_tokens=self.window_tokens,
                context_percent=0,
                confidence=Confidence.UNKNOWN,
                warnings=tuple(warnings or ["no usage messages found"]),
            )

        latest = max(ordered, key=lambda item: (item.observed_at, item.order))
        # An explicit launch/config window wins because the rollout may omit its
        # context variant. Otherwise the latest assistant message is the best
        # available fallback; a mid-session `/model` switch is routine.
        window_tokens = self.window_tokens or resolve_window_tokens(latest.model)
        if window_tokens is None:
            raise ValueError("Claude context window is unknown")
        # Only findings that invalidate the figures being reported may degrade
        # the sample: a mixed rollout, an unreadable tail (so the numbers are
        # stale), or a latest record that could not be keyed by message id (so
        # it may be a duplicate). A skipped row that a later good row supersedes
        # is recorded as a warning and nothing more.
        degraded = structurally_degraded or stale_tail or latest.derived_key
        context_tokens = (
            latest.input_tokens
            + latest.output_tokens
            + latest.cache_creation_tokens
            + latest.cache_read_tokens
        )
        tool_counts: dict[str, int] = {}
        for record in ordered:
            for _identity, name in record.tools:
                tool_counts[name] = tool_counts.get(name, 0) + 1

        return UsageSample(
            runtime=Runtime.CLAUDE,
            conversation_id=conversation_id or rollout_path.stem,
            observed_at=latest.observed_at,
            unique_messages=len(ordered),
            cumulative_input_tokens=sum(item.input_tokens for item in ordered),
            cumulative_output_tokens=sum(item.output_tokens for item in ordered),
            cumulative_cache_creation_tokens=sum(
                item.cache_creation_tokens for item in ordered
            ),
            cumulative_cache_read_tokens=sum(
                item.cache_read_tokens for item in ordered
            ),
            latest_input_tokens=latest.input_tokens,
            latest_output_tokens=latest.output_tokens,
            latest_cache_creation_tokens=latest.cache_creation_tokens,
            latest_cache_read_tokens=latest.cache_read_tokens,
            context_tokens=context_tokens,
            window_tokens=window_tokens,
            context_percent=100.0 * context_tokens / window_tokens,
            confidence=(Confidence.DEGRADED if degraded else Confidence.CONFIDENT),
            message_keys=tuple(item.key for item in ordered),
            tool_counts=dict(sorted(tool_counts.items())),
            warnings=tuple(warnings),
        )

    def discover(
        self,
        *,
        projects_root: str | os.PathLike[str],
        cwd: str | os.PathLike[str],
    ) -> ClaudeDiscovery:
        root = Path(projects_root).expanduser()
        resolved_cwd = Path(cwd).expanduser().resolve()
        slug = str(resolved_cwd).replace(os.sep, "-")
        candidates: list[Path] = []
        for path in iter_files(root):
            relative_parts = path.relative_to(root).parts
            slug_match = bool(relative_parts and relative_parts[0] == slug)
            if slug_match or self._contains_cwd(path, resolved_cwd):
                candidates.append(path.resolve())
        ordered = tuple(sorted(candidates))
        if len(ordered) > 1:
            return ClaudeDiscovery(
                candidates=ordered,
                ambiguous=True,
                error="multiple Claude conversations match cwd",
            )
        if not ordered:
            return ClaudeDiscovery(
                candidates=(),
                ambiguous=False,
                error="no Claude conversation matches cwd",
            )
        return ClaudeDiscovery(candidates=ordered, ambiguous=False)

    @staticmethod
    def _contains_cwd(path: Path, cwd: Path) -> bool:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    raw_cwd = payload.get("cwd")
                    if raw_cwd and Path(str(raw_cwd)).expanduser().resolve() == cwd:
                        return True
        except OSError:
            return False
        return False

    @staticmethod
    def _timestamp(raw: object, path: Path) -> datetime:
        if isinstance(raw, str):
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    return parsed.astimezone(timezone.utc)
            except ValueError:
                pass
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

    @staticmethod
    def _usage_values(usage: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            max(0, int(usage.get("input_tokens", 0) or 0)),
            max(0, int(usage.get("output_tokens", 0) or 0)),
            max(0, int(usage.get("cache_creation_input_tokens", 0) or 0)),
            max(0, int(usage.get("cache_read_input_tokens", 0) or 0)),
        )

    @staticmethod
    def _safe_tools(content: object, message_key: str) -> set[tuple[str, str]]:
        tools: set[tuple[str, str]] = set()
        if not isinstance(content, list):
            return tools
        for index, item in enumerate(content):
            if not isinstance(item, dict) or item.get("type") != "tool_use":
                continue
            name = " ".join(str(item.get("name") or "unknown").split())[:128]
            identity = str(item.get("id") or f"{message_key}:{index}")
            tools.add((identity, name))
        return tools
