from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import signal
import time

from agent_session_harness.hooks.command import run_hook
from agent_session_harness.hooks.native import normalize_native_event
from agent_session_harness.ledger import EventLedger


def append(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args, _handoff = parser.parse_known_args()
    root = Path(args.root)
    generation = int(os.environ["AGENT_SESSION_HARNESS_GENERATION"])
    chain_id = os.environ["AGENT_SESSION_HARNESS_CHAIN_ID"]
    owner_pid = int(os.environ["AGENT_SESSION_HARNESS_OWNER_PID"])
    if owner_pid <= 0:
        raise RuntimeError("managed owner PID is invalid")
    conversation_id = f"native-conversation-{generation}"
    now = datetime.now(tz=timezone.utc).isoformat()
    ledger = EventLedger(os.environ["AGENT_SESSION_HARNESS_LEDGER"])

    def emit(hook_event_name: str, **metadata: object) -> None:
        ledger.append(
            normalize_native_event(
                runtime="codex",
                payload={
                    "hook_event_name": hook_event_name,
                    "session_id": conversation_id,
                    "cwd": str(Path.cwd()),
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    **metadata,
                },
                chain_id=chain_id,
                generation=generation,
                owner_pid=os.getpid(),
            )
        )

    def stop_hook() -> int:
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = run_hook(
            runtime="codex",
            stdin=io.StringIO(
                json.dumps(
                    {
                        "hook_event_name": "Stop",
                        "session_id": conversation_id,
                        "cwd": str(Path.cwd()),
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                        "turn_id": "turn-0",
                    }
                )
            ),
            stdout=stdout,
            stderr=stderr,
            environ=os.environ,
        )
        append(
            root / "stop-responses.jsonl",
            {
                "exit_code": exit_code,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
            },
        )
        return exit_code

    def session_start_hook() -> int:
        return run_hook(
            runtime="codex",
            stdin=io.StringIO(
                json.dumps(
                    {
                        "hook_event_name": "SessionStart",
                        "session_id": conversation_id,
                        "cwd": str(Path.cwd()),
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    }
                )
            ),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            environ=os.environ,
        )

    if session_start_hook() != 0:
        raise RuntimeError("SessionStart hook failed")
    if generation == 0:
        emit("UserPromptSubmit", turn_id="turn-0")
        emit("PreToolUse", tool_use_id="tool-0", tool_name="fake-tool")
    append(
        root / "history.jsonl",
        {
            "event": "started",
            "chain_id": chain_id,
            "conversation_id": conversation_id,
            "generation": generation,
            "pid": os.getpid(),
            "timestamp": now,
        },
    )
    rollout = root / f"rollout-{generation}.jsonl"
    rows: list[dict[str, object]] = []
    if generation > 0:
        rows.append(
            {
                "timestamp": now,
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {"total_tokens": 150},
                        "last_token_usage": {"total_tokens": 150},
                        "model_context_window": 200,
                    },
                },
            }
        )
    meta: dict[str, object] = {
        "id": conversation_id,
        "timestamp": now,
        "cwd": str(Path.cwd()),
    }
    if generation > 0:
        meta["source"] = {
            "subagent": {"thread_spawn": {"parent_thread_id": "native-conversation-0"}}
        }
    rows.append({"timestamp": now, "type": "session_meta", "payload": meta})
    total = 130 if generation == 0 else 180
    latest = 130 if generation == 0 else 30
    rows.append(
        {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {"total_tokens": total},
                    "last_token_usage": {"total_tokens": latest},
                    "model_context_window": 200,
                },
            },
        }
    )
    rollout.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    if generation > 0:
        fingerprint = os.environ["AGENT_SESSION_HARNESS_CAPSULE_FINGERPRINT"]
        capsule_path = Path(os.environ["AGENT_SESSION_HARNESS_CAPSULE_PATH"])
        capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
        append(
            root / "continuations.jsonl",
            {
                "generation": generation,
                "fingerprint": fingerprint,
                "exact_next_action": capsule["exact_next_action"],
            },
        )

    def stop(_signal: int, _frame: object) -> None:
        append(
            root / "history.jsonl",
            {
                "event": "stopped",
                "chain_id": chain_id,
                "conversation_id": conversation_id,
                "generation": generation,
                "pid": os.getpid(),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
        )
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    activity_finished = generation > 0
    finish_path = root / f"finish-activity-{generation}"
    while True:
        if not activity_finished and finish_path.exists():
            emit("PostToolUse", tool_use_id="tool-0", tool_name="fake-tool")
            if stop_hook() != 2:
                raise RuntimeError("first draining Stop was not blocked")
            if stop_hook() != 0:
                raise RuntimeError("recursive Stop was not allowed")
            activity_finished = True
        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
