"""Bounded stdin-to-ledger hook command."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, TextIO

from ..ledger import EventLedger
from .native import normalize_native_event, stop_handshake


MAX_INPUT_BYTES = 1_048_576


def run_hook(
    *,
    runtime: str,
    stdin: TextIO,
    stdout: TextIO,
    environ: Mapping[str, str] | None = None,
) -> int:
    environment = environ if environ is not None else os.environ
    if environment.get("AGENT_SESSION_HARNESS_MANAGED") != "1":
        raise RuntimeError("hook requires managed harness mode")
    encoded = stdin.read(MAX_INPUT_BYTES + 1)
    if len(encoded.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("native hook input is too large")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("native hook input is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("native hook input must be a JSON object")

    chain_id = environment.get("AGENT_SESSION_HARNESS_CHAIN_ID", "")
    ledger_path = environment.get("AGENT_SESSION_HARNESS_LEDGER", "")
    if not chain_id or not ledger_path:
        raise RuntimeError("managed hook environment is incomplete")
    generation = int(environment.get("AGENT_SESSION_HARNESS_GENERATION", "0"))
    owner_pid = int(
        environment.get("AGENT_SESSION_HARNESS_OWNER_PID", str(os.getppid()))
    )
    event = normalize_native_event(
        runtime=runtime,
        payload=payload,
        chain_id=chain_id,
        generation=generation,
        owner_pid=owner_pid,
    )
    EventLedger(Path(ledger_path)).append(event)

    if event.event_type.value == "turn.idle":
        response = stop_handshake(
            runtime=runtime,
            draining=_enabled(environment, "AGENT_SESSION_HARNESS_DRAINING"),
            checkpoint_verified=_enabled(
                environment, "AGENT_SESSION_HARNESS_CHECKPOINT_VERIFIED"
            ),
            recursion_active=_enabled(
                environment, "AGENT_SESSION_HARNESS_STOP_RECURSION"
            ),
            already_requested=(
                environment.get("AGENT_SESSION_HARNESS_STOP_REQUESTED_GENERATION")
                == str(generation)
            ),
            required_fields=tuple(
                item.strip()
                for item in environment.get(
                    "AGENT_SESSION_HARNESS_REQUIRED_CHECKPOINTS", ""
                ).split(",")
                if item.strip()
            ),
        )
    else:
        response = {"ok": True}
    stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
    stdout.flush()
    return 0


def _enabled(environment: Mapping[str, str], name: str) -> bool:
    return environment.get(name, "").lower() in {"1", "true", "yes", "on"}
