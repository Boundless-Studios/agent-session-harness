"""Normalize native hook payloads and encode runtime control responses."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
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
    recursion_active: bool,
    already_requested: bool,
    required_fields: Sequence[str],
) -> dict[str, object]:
    """Return one native-compatible continuation request at a drain boundary."""

    Runtime(runtime)
    if not draining or checkpoint_verified:
        return {"continue": True}
    if recursion_active or already_requested:
        return {
            "continue": True,
            "systemMessage": "Durable checkpoint was already requested for this generation.",
        }
    fields = ", ".join(required_fields) if required_fields else "configured adapters"
    return {
        "decision": "block",
        "continue": True,
        "reason": (
            "Context rotation is draining. Persist the exact next action and "
            f"acceptance state to: {fields}. Do not start new work."
        ),
    }


def _activity_id(
    hook_name: str, payload: Mapping[str, object], conversation_id: str
) -> str | None:
    if hook_name in {"UserPromptSubmit", "Stop"}:
        return str(payload.get("turn_id") or f"turn:{conversation_id}")
    if hook_name in {"PreToolUse", "PostToolUse", "PostToolUseFailure"}:
        raw_value = payload.get("tool_use_id") or payload.get("call_id")
        return str(raw_value) if raw_value else f"tool:{conversation_id}"
    if hook_name in {"SubagentStart", "SubagentStop"}:
        raw_value = payload.get("agent_id") or payload.get("subagent_id")
        return str(raw_value) if raw_value else f"subagent:{conversation_id}"
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
