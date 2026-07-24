"""Normalize native hook payloads and encode runtime control responses."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from ..events import LifecycleEvent
from ..models import EventType, Runtime

_EVENT_TYPES = {
    "SessionStart": EventType.SESSION_STARTED,
    "SessionEnd": EventType.SESSION_ENDED,
    "UserPromptSubmit": EventType.TURN_STARTED,
    "PreToolUse": EventType.TOOL_STARTED,
    "PostToolUse": EventType.TOOL_FINISHED,
    "PostToolUseFailure": EventType.TOOL_FAILED,
    "SubagentStart": EventType.SUBAGENT_STARTED,
    "SubagentStop": EventType.SUBAGENT_FINISHED,
    "Stop": EventType.TURN_IDLE,
    "PreCompact": EventType.PRE_COMPACT,
}


@dataclass(frozen=True, slots=True)
class NativeHookResponse:
    """One runtime-native hook result, separated by output channel."""

    exit_code: int
    stdout: dict[str, object] | None = None
    stderr: str = ""


def normalize_native_event(
    *,
    runtime: str | Runtime,
    payload: Mapping[str, object],
    chain_id: str,
    generation: int,
    owner_pid: int,
) -> LifecycleEvent:
    """Select only lifecycle metadata from a native hook payload."""

    runtime_value = Runtime(runtime)
    hook_name = str(
        payload.get("hook_event_name")
        or payload.get("event_name")
        or payload.get("event")
        or ""
    )
    try:
        event_type = _EVENT_TYPES[hook_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported native hook event: {hook_name or 'missing'}"
        ) from exc

    conversation_id = str(
        payload.get("session_id")
        or payload.get("conversation_id")
        or payload.get("thread_id")
        or ""
    )
    if not conversation_id:
        raise ValueError("native hook payload is missing a conversation ID")
    cwd = Path(str(payload.get("cwd") or ""))
    if not cwd.is_absolute():
        raise ValueError("native hook payload is missing an absolute cwd")
    timestamp = _timestamp(payload.get("timestamp"))
    activity_id = _activity_id(hook_name, payload, conversation_id)
    name = None
    if event_type in {
        EventType.TOOL_STARTED,
        EventType.TOOL_FINISHED,
        EventType.TOOL_FAILED,
    }:
        raw_name = payload.get("tool_name") or payload.get("name")
        name = _normalized_name(raw_name)
    elif event_type in {EventType.SUBAGENT_STARTED, EventType.SUBAGENT_FINISHED}:
        raw_name = payload.get("agent_type") or payload.get("name")
        name = _normalized_name(raw_name)

    identity = "|".join(
        (
            runtime_value.value,
            conversation_id,
            str(generation),
            hook_name,
            activity_id or "",
            timestamp.isoformat(),
        )
    )
    native_id = payload.get("event_id") or payload.get("hook_id")
    event_id = (
        str(native_id) if native_id else hashlib.sha256(identity.encode()).hexdigest()
    )
    return LifecycleEvent(
        schema_version=1,
        event_id=event_id,
        runtime=runtime_value,
        chain_id=chain_id,
        conversation_id=conversation_id,
        generation=generation,
        event_type=event_type,
        timestamp=timestamp,
        cwd=cwd,
        owner_pid=owner_pid,
        activity_id=activity_id,
        name=name,
    )


def stop_handshake(
    *,
    runtime: str | Runtime,
    draining: bool,
    checkpoint_verified: bool,
    already_requested: bool,
    required_fields: Sequence[str],
) -> NativeHookResponse:
    """Return one native-compatible continuation request at a drain boundary."""

    runtime_value = Runtime(runtime)
    if not draining or checkpoint_verified:
        return _allow_response(runtime_value)
    if already_requested:
        return _repeat_response(runtime_value)
    fields = _checkpoint_fields(required_fields)
    reason = (
        "Context rotation is draining. Persist the exact next action and "
        f"acceptance state to: {fields}. Do not start new work."
    )
    if runtime_value is Runtime.CLAUDE:
        return NativeHookResponse(
            exit_code=0,
            stdout={"decision": "block", "reason": reason},
        )
    return NativeHookResponse(exit_code=2, stderr=reason + "\n")


def handoff_requested_event(stop_event: LifecycleEvent) -> LifecycleEvent:
    """Derive one sanitized, generation-stable handoff request event."""

    if stop_event.event_type is not EventType.TURN_IDLE:
        raise ValueError("handoff request must derive from a Stop event")
    identity = "|".join(
        (
            stop_event.runtime.value,
            stop_event.chain_id,
            stop_event.conversation_id,
            str(stop_event.generation),
            EventType.HANDOFF_REQUESTED.value,
        )
    )
    return LifecycleEvent(
        schema_version=stop_event.schema_version,
        event_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        runtime=stop_event.runtime,
        chain_id=stop_event.chain_id,
        conversation_id=stop_event.conversation_id,
        generation=stop_event.generation,
        event_type=EventType.HANDOFF_REQUESTED,
        timestamp=stop_event.timestamp,
        cwd=stop_event.cwd,
        owner_pid=stop_event.owner_pid,
    )


def repeated_stop_idle_event(stop_event: LifecycleEvent) -> LifecycleEvent:
    """Give recursive Stop idles a stable identity so retries stay idempotent."""

    if stop_event.event_type is not EventType.TURN_IDLE:
        raise ValueError("recursive Stop idle must derive from a Stop event")
    identity = "|".join(
        (
            stop_event.runtime.value,
            stop_event.chain_id,
            stop_event.conversation_id,
            str(stop_event.generation),
            EventType.TURN_IDLE.value,
            stop_event.activity_id or "",
            "recursive",
        )
    )
    return stop_event.model_copy(
        update={"event_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()}
    )


def _allow_response(runtime: Runtime) -> NativeHookResponse:
    if runtime is Runtime.CLAUDE:
        return NativeHookResponse(exit_code=0, stdout={"continue": True})
    return NativeHookResponse(exit_code=0)


def _repeat_response(runtime: Runtime) -> NativeHookResponse:
    message = "Durable checkpoint was already requested for this generation."
    if runtime is Runtime.CLAUDE:
        return NativeHookResponse(
            exit_code=0,
            stdout={"continue": True, "systemMessage": message},
        )
    return NativeHookResponse(exit_code=0, stderr=message + "\n")


def _checkpoint_fields(values: Sequence[str]) -> str:
    safe_values = []
    for value in values:
        normalized = " ".join(str(value).split())
        if (
            normalized
            and len(normalized) <= 64
            and all(
                character.isalnum() or character in "-_." for character in normalized
            )
        ):
            safe_values.append(normalized)
    return ", ".join(safe_values) if safe_values else "configured adapters"


def _derived_activity_id(payload: Mapping[str, object], conversation_id: str) -> str:
    """Stable tool-call id for runtimes that do not supply one.

    Built only from fields a PreToolUse and its matching PostToolUse both carry,
    so the pair resolves to the same id. Deliberately excludes the tool payload:
    the harness never reads prompt or tool argument text, and the granularity is
    not needed. Quiescence asks only whether anything is outstanding, and the
    ledger counts activity, so two different calls sharing an id still balance
    (+1 +1 -1 -1). Finer identity would buy nothing and would cost the boundary.
    """
    material = "|".join(
        (
            conversation_id,
            str(payload.get("prompt_id") or ""),
            str(payload.get("tool_name") or payload.get("name") or ""),
        )
    )
    return f"derived:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _activity_id(
    hook_name: str, payload: Mapping[str, object], conversation_id: str
) -> str | None:
    if hook_name in {"UserPromptSubmit", "Stop"}:
        return str(payload.get("turn_id") or f"turn:{conversation_id}")
    if hook_name in {"PreToolUse", "PostToolUse", "PostToolUseFailure"}:
        raw_value = payload.get("tool_use_id") or payload.get("call_id")
        if raw_value:
            return str(raw_value)
        # Claude Code's documented PreToolUse/PostToolUse payloads carry no
        # tool-use identifier. Raising here returns exit 2, which for PreToolUse
        # BLOCKS the tool call — so a managed session could run no tools at all.
        # Never depend on an undocumented field; derive a stable id instead.
        #
        # The id must be identical for the Pre and Post of one call, because the
        # ledger pairs starts against finishes to decide quiescence. Calls that
        # share a prompt and tool name collide; the ledger counts activity
        # rather than set-tracking it, so collisions balance instead of wedging
        # quiescence. Codex still supplies call_id and is unaffected.
        return _derived_activity_id(payload, conversation_id)
    if hook_name in {"SubagentStart", "SubagentStop"}:
        raw_value = payload.get("agent_id") or payload.get("subagent_id")
        if not raw_value:
            raise ValueError("native subagent hook is missing an activity ID")
        return str(raw_value)
    return None


def _normalized_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())[:128]
    return normalized or None


def _timestamp(value: object) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("native timestamp must include a timezone")
            return parsed.astimezone(timezone.utc)
        except ValueError as exc:
            raise ValueError("invalid native hook timestamp") from exc
    return datetime.now(tz=timezone.utc)
