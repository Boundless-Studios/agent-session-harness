"""Bounded usage sampling for one exactly-identified managed conversation.

The adapter resolves the conversation from the harness lifecycle ledger,
locates its native rollout, strips the rollout down to accounting metadata,
and hands only that reduced copy to a usage reader. Ambiguity of any kind
resolves to an unknown observation so an automatic rotation never looks safe.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..secure_files import read_private_text
from .claude import resolve_window_tokens
from .discovery import LIFECYCLE_SUFFIX, iter_files, select_conversation_rollout

MAX_INPUT_BYTES = 1_048_576
MAX_LEDGER_BYTES = 4 * 1_048_576
MAX_ROLLOUT_BYTES = 128 * 1_048_576
MAX_LEDGER_EVENTS = 20_000
_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_REQUEST_KEYS = frozenset({"schema_version", "operation", "process"})
_PROCESS_KEYS = frozenset(
    {
        "pid",
        "process_group_id",
        "registry_key",
        "identity",
        "command_digest",
        "launch_nonce",
    }
)

ReaderFactory = Callable[..., Any]


def unknown_observation() -> dict[str, object]:
    """Return the only safe result when usage cannot be attributed exactly."""

    return {
        "conversation_id": "unresolved",
        "context_percent": 0.0,
        "confidence": "unknown",
        "context_tokens": None,
        "window_tokens": None,
        "cumulative_tokens": None,
    }


def sample_usage(
    request: Mapping[str, object],
    *,
    ledger_paths: Sequence[str | os.PathLike[str]] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    claude_roots: Sequence[str | os.PathLike[str]] | None = None,
    codex_roots: Sequence[str | os.PathLike[str]] | None = None,
    claude_window_tokens: int | None = None,
    claude_fallback_window_tokens: int | None = None,
    max_rollout_bytes: int = MAX_ROLLOUT_BYTES,
    claude_reader_factory: ReaderFactory | None = None,
    codex_reader_factory: ReaderFactory | None = None,
) -> dict[str, object]:
    """Resolve and sample one managed conversation, or return bounded unknown."""

    try:
        process = _validate_request(request)
        worktree = _absolute_directory(cwd or Path.cwd())
        paths = tuple(
            Path(path) for path in (ledger_paths or _discover_ledgers(worktree))
        )
        runtime, conversation_id = _select_conversation(
            paths,
            process=process,
            cwd=worktree,
        )
        roots = _runtime_roots("claude") if claude_roots is None else claude_roots
        if runtime == "codex":
            roots = _runtime_roots("codex") if codex_roots is None else codex_roots
        rollout = select_conversation_rollout(
            tuple(roots),
            conversation_id=conversation_id,
            substring_match=runtime == "codex",
        )
        decoded = read_private_text(rollout, max_bytes=max_rollout_bytes)
        sanitized = _sanitize_rollout(decoded, runtime=runtime)
        window_tokens: int | None = None
        if runtime == "claude":
            window_tokens = _claude_window_tokens(
                sanitized,
                explicit=claude_window_tokens,
                fallback=claude_fallback_window_tokens,
            )
        if claude_reader_factory is None or codex_reader_factory is None:
            installed_claude, installed_codex = _installed_reader_factories()
            claude_reader_factory = claude_reader_factory or installed_claude
            codex_reader_factory = codex_reader_factory or installed_codex
        observation = _read_sanitized_rollout(
            sanitized,
            runtime=runtime,
            claude_window_tokens=window_tokens,
            claude_reader_factory=claude_reader_factory,
            codex_reader_factory=codex_reader_factory,
        )
        if runtime == "claude":
            observed_id = str(observation.conversation_id)
            cumulative = int(observation.cumulative_total_tokens)
        else:
            observed_id = str(observation.session_id)
            incremental = observation.incremental_total_tokens
            cumulative = int(incremental) if incremental is not None else None
        if observed_id != conversation_id:
            raise ValueError("rollout conversation does not match lifecycle ledger")
        return {
            "conversation_id": conversation_id,
            "context_percent": float(observation.context_percent),
            "confidence": str(observation.confidence.value),
            "context_tokens": int(observation.context_tokens),
            "window_tokens": int(observation.window_tokens),
            "cumulative_tokens": cumulative,
        }
    except Exception:
        # Usage ambiguity must never make an automatic rotation look safe. Keep
        # filesystem paths, conversation fragments, and parser diagnostics private.
        return unknown_observation()


def _validate_request(request: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(request, Mapping) or set(request) != _REQUEST_KEYS:
        raise ValueError("invalid usage request")
    if request.get("schema_version") != 1 or request.get("operation") != "sample":
        raise ValueError("invalid usage request")
    raw_process = request.get("process")
    if not isinstance(raw_process, Mapping) or set(raw_process) != _PROCESS_KEYS:
        raise ValueError("invalid usage process")
    process = dict(raw_process)
    pid = process.get("pid")
    process_group_id = process.get("process_group_id")
    registry_key = process.get("registry_key")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError("invalid managed owner")
    if (
        not isinstance(process_group_id, int)
        or isinstance(process_group_id, bool)
        or process_group_id <= 0
    ):
        raise ValueError("invalid managed process group")
    if not isinstance(registry_key, str) or not registry_key or len(registry_key) > 321:
        raise ValueError("invalid managed registry key")
    for key in ("identity", "command_digest", "launch_nonce"):
        value = process.get(key)
        if value is not None and (not isinstance(value, str) or len(value) > 512):
            raise ValueError("invalid managed process metadata")
    return process


def _select_conversation(
    paths: Sequence[Path],
    *,
    process: Mapping[str, object],
    cwd: Path,
) -> tuple[str, str]:
    if not paths:
        raise ValueError("lifecycle ledger is unavailable")
    registry_key = str(process["registry_key"])
    chain_id, separator, raw_generation = registry_key.rpartition(":")
    if not separator or not chain_id:
        raise ValueError("managed registry key has no generation")
    try:
        generation = int(raw_generation)
    except ValueError as exc:
        raise ValueError("managed registry generation is invalid") from exc
    if generation < 0:
        raise ValueError("managed registry generation is invalid")

    identities: set[tuple[str, str]] = set()
    event_count = 0
    for path in paths:
        decoded = read_private_text(path, max_bytes=MAX_LEDGER_BYTES)
        for raw_line in decoded.splitlines():
            if not raw_line.strip():
                continue
            event_count += 1
            if event_count > MAX_LEDGER_EVENTS:
                raise ValueError("lifecycle ledger exceeds event limit")
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError("lifecycle ledger is malformed") from exc
            if not isinstance(event, dict):
                raise ValueError("lifecycle ledger is malformed")
            if (
                event.get("owner_pid") != process["pid"]
                or event.get("generation") != generation
                or event.get("chain_id") != chain_id
            ):
                continue
            event_cwd = Path(str(event.get("cwd") or ""))
            if not event_cwd.is_absolute() or event_cwd.resolve() != cwd:
                continue
            runtime = event.get("runtime")
            conversation_id = event.get("conversation_id")
            if runtime not in {"claude", "codex"}:
                raise ValueError("lifecycle runtime is invalid")
            if not isinstance(conversation_id, str) or not _CONVERSATION_ID.fullmatch(
                conversation_id
            ):
                raise ValueError("lifecycle conversation is invalid")
            identities.add((runtime, conversation_id))
    if len(identities) != 1:
        raise ValueError("managed conversation is missing or ambiguous")
    return next(iter(identities))


def _discover_ledgers(cwd: Path) -> tuple[Path, ...]:
    return iter_files(cwd / ".agent-session-harness", LIFECYCLE_SUFFIX)


def _runtime_roots(runtime: str) -> tuple[Path, ...]:
    home = Path(os.environ.get("HOME") or Path.home())
    if runtime == "claude":
        config = Path(os.environ.get("CLAUDE_CONFIG_DIR") or home / ".claude")
        return (config / "projects",)
    config = Path(os.environ.get("CODEX_HOME") or home / ".codex")
    return (config / "sessions",)


def _sanitize_rollout(decoded: str, *, runtime: str) -> str:
    sanitized: list[dict[str, object]] = []
    for raw_line in decoded.splitlines():
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError("rollout contains invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError("rollout row is invalid")
        safe = _safe_claude_row(row) if runtime == "claude" else _safe_codex_row(row)
        if safe is not None:
            sanitized.append(safe)
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in sanitized
    )


def _safe_claude_row(row: Mapping[str, object]) -> dict[str, object] | None:
    if row.get("type") != "assistant":
        return None
    if not isinstance(row.get("message"), dict):
        raise ValueError("Claude assistant row is missing message metadata")
    message = row["message"]
    usage = message.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("Claude assistant row is missing usage metadata")
    safe_message: dict[str, object] = {
        "usage": {
            key: _required_nonnegative_int(usage.get(key), label=key)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        }
    }
    message_id = _safe_identifier(message.get("id"), 160)
    if message_id:
        safe_message["id"] = message_id
    model = _safe_identifier(message.get("model"), 160)
    if model:
        safe_message["model"] = model
    tools: list[dict[str, str]] = []
    content = message.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool_use":
                continue
            tools.append(
                {
                    "type": "tool_use",
                    "id": _safe_identifier(item.get("id"), 160) or "unknown",
                    "name": _safe_identifier(item.get("name"), 128) or "unknown",
                }
            )
    safe_message["content"] = tools
    safe: dict[str, object] = {"type": "assistant", "message": safe_message}
    session_id = _safe_identifier(row.get("sessionId"), 160)
    if session_id:
        safe["sessionId"] = session_id
    timestamp = _safe_timestamp(row.get("timestamp"))
    if timestamp:
        safe["timestamp"] = timestamp
    return safe


def _safe_codex_row(row: Mapping[str, object]) -> dict[str, object] | None:
    payload = row.get("payload")
    if not isinstance(payload, dict):
        if row.get("type") in {"session_meta", "event_msg"}:
            raise ValueError("Codex event is missing payload metadata")
        return None
    timestamp = _safe_timestamp(row.get("timestamp"))
    if row.get("type") == "session_meta":
        safe_payload: dict[str, object] = {}
        session_id = _safe_identifier(payload.get("id"), 160)
        if session_id:
            safe_payload["id"] = session_id
        payload_timestamp = _safe_timestamp(payload.get("timestamp"))
        if payload_timestamp:
            safe_payload["timestamp"] = payload_timestamp
        parent = _codex_parent_id(payload.get("source"))
        if parent:
            safe_payload["source"] = {
                "subagent": {"thread_spawn": {"parent_thread_id": parent}}
            }
        safe: dict[str, object] = {"type": "session_meta", "payload": safe_payload}
        if timestamp:
            safe["timestamp"] = timestamp
        return safe
    if row.get("type") != "event_msg" or payload.get("type") != "token_count":
        return None
    info = payload.get("info")
    if not isinstance(info, dict):
        raise ValueError("Codex token event is missing usage metadata")
    safe_info: dict[str, object] = {
        "model_context_window": _required_positive_int(
            info.get("model_context_window"), label="model_context_window"
        ),
        "total_token_usage": _safe_token_dimensions(info.get("total_token_usage")),
        "last_token_usage": _safe_token_dimensions(info.get("last_token_usage")),
    }
    safe = {
        "type": "event_msg",
        "payload": {"type": "token_count", "info": safe_info},
    }
    if timestamp:
        safe["timestamp"] = timestamp
    return safe


def _safe_token_dimensions(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError("Codex token dimensions are unavailable")
    return {
        key: _required_nonnegative_int(value.get(key), label=key)
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        )
    }


def _safe_identifier(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    if not normalized or len(normalized) > limit:
        return None
    return normalized


def _safe_timestamp(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if parsed.tzinfo is not None else None


def _codex_parent_id(source: object) -> str | None:
    if not isinstance(source, dict):
        return None
    subagent = source.get("subagent")
    if not isinstance(subagent, dict):
        return None
    spawn = subagent.get("thread_spawn")
    if not isinstance(spawn, dict):
        return None
    return _safe_identifier(spawn.get("parent_thread_id"), 160)


def _required_nonnegative_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _required_positive_int(value: object, *, label: str) -> int:
    result = _required_nonnegative_int(value, label=label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _read_sanitized_rollout(
    sanitized: str,
    *,
    runtime: str,
    claude_window_tokens: int | None,
    claude_reader_factory: ReaderFactory,
    codex_reader_factory: ReaderFactory,
) -> Any:
    descriptor, raw_path = tempfile.mkstemp(
        prefix="agent-session-harness-usage-", suffix=".jsonl"
    )
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(sanitized)
            handle.flush()
        if runtime == "claude":
            if claude_window_tokens is None:
                raise ValueError("Claude context window is unknown")
            reader = claude_reader_factory(window_tokens=claude_window_tokens)
        else:
            reader = codex_reader_factory()
        return reader.read_file(path)
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _installed_reader_factories() -> tuple[ReaderFactory, ReaderFactory]:
    from .claude import ClaudeUsageReader
    from .codex import CodexUsageReader

    return ClaudeUsageReader, CodexUsageReader


def _claude_window_tokens(
    sanitized: str, *, explicit: int | None, fallback: int | None
) -> int:
    """Resolve an authoritative or model-derived Claude context window.

    An explicit launch/config value is authoritative because Claude Code omits
    context-variant suffixes from rollout model names. The model table and then
    `fallback` apply only when no explicit value is available.
    """

    if explicit is not None:
        if explicit <= 0:
            raise ValueError("Claude context window must be positive")
        return explicit

    models: set[str] = set()
    for raw_line in sanitized.splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        message = row.get("message") if isinstance(row, dict) else None
        model = message.get("model") if isinstance(message, dict) else None
        if isinstance(model, str) and model:
            models.add(model)
    if len(models) != 1:
        raise ValueError("Claude model identity is unavailable or ambiguous")
    resolved = resolve_window_tokens(next(iter(models)))
    if resolved is not None:
        return resolved
    if fallback is None:
        raise ValueError("Claude context window is unknown")
    if fallback <= 0:
        raise ValueError("Claude context window must be positive")
    return fallback


def _absolute_directory(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("working directory is invalid")
    return resolved


def main(
    *,
    ledger_paths: Sequence[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    claude_roots: Sequence[str] | None = None,
    codex_roots: Sequence[str] | None = None,
    claude_window_tokens: int | None = None,
    claude_fallback_window_tokens: int | None = None,
    max_rollout_bytes: int = MAX_ROLLOUT_BYTES,
) -> int:
    """Read one sample request from stdin and write one bounded observation."""

    try:
        encoded = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        if len(encoded) > MAX_INPUT_BYTES:
            raise ValueError("request exceeds input limit")
        request = json.loads(encoded)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        result = sample_usage(
            request,
            ledger_paths=tuple(ledger_paths) if ledger_paths else None,
            cwd=cwd or Path.cwd(),
            claude_roots=claude_roots,
            codex_roots=codex_roots,
            claude_window_tokens=claude_window_tokens,
            claude_fallback_window_tokens=claude_fallback_window_tokens,
            max_rollout_bytes=max_rollout_bytes,
        )
    except Exception:
        result = unknown_observation()
    json.dump(result, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0
