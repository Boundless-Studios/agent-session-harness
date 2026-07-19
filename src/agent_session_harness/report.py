"""Deterministic, model-free harness status and diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .activity import Quiescence
from .config import load_config, resolve_config_path
from .ledger import EventLedger
from .models import Confidence, Runtime
from .outbox import MirrorOutbox
from .supervisor import SupervisorSnapshot


class ActiveCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    turns: int = Field(default=0, ge=0)
    tools: int = Field(default=0, ge=0)
    subagents: int = Field(default=0, ge=0)
    critical_sections: int = Field(default=0, ge=0)


class StatusReport(BaseModel):
    """Canonical downstream projection for dashboard and terminal consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    runtime: Runtime
    state: str
    chain_id: str
    conversation_id: str | None
    generation: int = Field(ge=0)
    context_percent: float | None = Field(default=None, ge=0)
    confidence: Confidence
    quiescence: Quiescence
    active: ActiveCounts
    checkpoint_fingerprint: str | None
    outbox_depth: int = Field(ge=0)


def build_report(
    *,
    state_path: str | os.PathLike[str],
    ledger_path: str | os.PathLike[str] | None = None,
    outbox_path: str | os.PathLike[str] | None = None,
    stale_after_seconds: float = 30.0,
    now: datetime | None = None,
) -> StatusReport:
    state = SupervisorSnapshot.model_validate_json(
        Path(state_path).read_text(encoding="utf-8")
    )
    quiescence = Quiescence.UNKNOWN
    active = ActiveCounts()
    if ledger_path is not None:
        snapshot = EventLedger(ledger_path).materialize(
            now=now or datetime.now(tz=timezone.utc),
            stale_after_seconds=stale_after_seconds,
        )
        quiescence = snapshot.quiescence
        active = ActiveCounts(
            turns=len(snapshot.active_turn_ids),
            tools=len(snapshot.active_tool_ids),
            subagents=len(snapshot.active_subagent_ids),
            critical_sections=len(snapshot.active_critical_section_ids),
        )
    outbox_depth = MirrorOutbox(outbox_path).depth if outbox_path else 0
    return StatusReport(
        runtime=state.runtime,
        state=state.phase.value,
        chain_id=state.chain_id,
        conversation_id=state.conversation_id,
        generation=state.generation,
        context_percent=state.context_percent,
        confidence=state.context_confidence,
        quiescence=quiescence,
        active=active,
        checkpoint_fingerprint=state.checkpoint_fingerprint,
        outbox_depth=outbox_depth,
    )


def doctor_report(
    *,
    runtime: str | Runtime | None = None,
    config_path: str | os.PathLike[str] | None = None,
    project_dir: str | os.PathLike[str] | None = None,
    state_dir: str | os.PathLike[str] | None = None,
    hook_manifest: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    checks: dict[str, object] = {
        "package": __version__,
        "python": ".".join(str(value) for value in sys.version_info[:3]),
    }
    ok = True
    resolved = resolve_config_path(
        explicit_path=config_path,
        project_dir=project_dir,
    )
    config = load_config(
        explicit_path=config_path,
        project_dir=project_dir,
        required_capabilities_known=None,
    )
    checks["config"] = str(resolved) if resolved else "defaults"
    checks["observe_only"] = config.observe_only
    if runtime is not None:
        runtime_value = Runtime(runtime)
        executable = shutil.which(runtime_value.value)
        checks["runtime"] = {
            "name": runtime_value.value,
            "available": executable is not None,
            "path": executable,
        }
        ok = ok and executable is not None
    if state_dir is not None:
        path = Path(state_dir).expanduser()
        parent = path if path.exists() else path.parent
        parent_exists = parent.exists()
        parent_writable = os.access(parent, os.W_OK) if parent_exists else False
        checks["state"] = {
            "path": str(path),
            "parent_exists": parent_exists,
            "parent_writable": parent_writable,
        }
        ok = ok and parent_exists and parent_writable
    if hook_manifest is not None:
        path = Path(hook_manifest)
        valid = False
        if path.is_file():
            try:
                valid = isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
            except json.JSONDecodeError:
                valid = False
        checks["hook_manifest"] = {"path": str(path), "valid": valid}
        ok = ok and valid
    return {"schema_version": 1, "ok": ok, "checks": checks}
