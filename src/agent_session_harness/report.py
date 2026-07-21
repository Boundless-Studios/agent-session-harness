"""Deterministic, model-free harness status and diagnostics."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import __version__
from .activity import Quiescence, RuntimeLiveness
from .adapters.command import (
    AdapterOperation,
    AdapterRequest,
    AdapterResponse,
    JsonCommand,
)
from .capsule import HandoffCapsule
from .config import load_config, resolve_config_path
from .hooks.install import HookInstaller
from .ledger import EventLedger
from .models import Confidence, Runtime
from .outbox import MirrorOutbox
from .secure_files import read_private_text
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
    context_tokens: int | None = Field(default=None, ge=0)
    window_tokens: int | None = Field(default=None, gt=0)
    cumulative_tokens: int | None = Field(default=None, ge=0)
    confidence: Confidence
    quiescence: Quiescence
    active: ActiveCounts
    checkpoint_fingerprint: str | None
    outbox_depth: int = Field(ge=0)
    # BOU-2222: quiescence alone cannot tell a dashboard whether the runtime is
    # calm or mute, and a mute runtime never rotates.
    runtime_liveness: RuntimeLiveness = RuntimeLiveness.REPORTING
    liveness_alarm: str | None = None
    usage_alarm: str | None = None
    # BOU-2236: how many tool starts the last turn-idle had to reconcile because
    # a permission gate denied them at PreToolUse, so no finish could ever
    # arrive. Zero on runtimes without such a gate. Non-zero is expected, not a
    # fault -- but a number that climbs every turn is the signal that a gate is
    # denying heavily, which is worth seeing on a dashboard.
    reaped_tools: int = Field(default=0, ge=0)


def build_report(
    *,
    state_path: str | os.PathLike[str],
    ledger_path: str | os.PathLike[str] | None = None,
    outbox_path: str | os.PathLike[str] | None = None,
    stale_after_seconds: float = 30.0,
    now: datetime | None = None,
) -> StatusReport:
    state = SupervisorSnapshot.model_validate_json(read_private_text(state_path))
    quiescence = Quiescence.UNKNOWN
    # No ledger to read is itself the "no hook has ever reported" case.
    runtime_liveness = RuntimeLiveness.NEVER_REPORTED
    active = ActiveCounts()
    reaped_tools = 0
    if ledger_path is not None:
        snapshot = EventLedger(ledger_path).materialize(
            now=now or datetime.now(tz=timezone.utc),
            stale_after_seconds=stale_after_seconds,
        )
        quiescence = snapshot.quiescence
        runtime_liveness = snapshot.runtime_liveness
        active = ActiveCounts(
            turns=len(snapshot.active_turn_ids),
            tools=len(snapshot.active_tool_ids),
            subagents=len(snapshot.active_subagent_ids),
            critical_sections=len(snapshot.active_critical_section_ids),
        )
        reaped_tools = len(snapshot.reaped_tool_ids)
    outbox_depth = MirrorOutbox(outbox_path).depth if outbox_path else 0
    return StatusReport(
        runtime=state.runtime,
        state=state.phase.value,
        chain_id=state.chain_id,
        conversation_id=state.conversation_id,
        generation=state.generation,
        context_percent=state.context_percent,
        context_tokens=state.context_tokens,
        window_tokens=state.window_tokens,
        cumulative_tokens=state.cumulative_tokens,
        confidence=state.context_confidence,
        quiescence=quiescence,
        active=active,
        checkpoint_fingerprint=state.checkpoint_fingerprint,
        outbox_depth=outbox_depth,
        reaped_tools=reaped_tools,
        runtime_liveness=runtime_liveness,
        liveness_alarm=state.liveness_alarm,
        usage_alarm=state.usage_alarm,
    )


def doctor_report(
    *,
    runtime: str | Runtime | None = None,
    config_path: str | os.PathLike[str] | None = None,
    project_dir: str | os.PathLike[str] | None = None,
    state_dir: str | os.PathLike[str] | None = None,
    state_path: str | os.PathLike[str] | None = None,
    log_root: str | os.PathLike[str] | None = None,
    hook_manifest: str | os.PathLike[str] | None = None,
    required_capabilities_known: bool | None = None,
    adapter_commands: Mapping[str, Sequence[str]] | None = None,
    adapter_inherit_env: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    checks: dict[str, object] = {
        "package": __version__,
        "python": ".".join(str(value) for value in sys.version_info[:3]),
    }
    coordinator = _coordinator_check()
    checks["coordinator"] = coordinator
    ok = bool(coordinator["installed"] and coordinator["compatible"])
    resolved = resolve_config_path(
        explicit_path=config_path,
        project_dir=project_dir,
    )
    config = load_config(
        explicit_path=config_path,
        project_dir=project_dir,
        required_capabilities_known=required_capabilities_known,
    )
    checks["config"] = str(resolved) if resolved else "defaults"
    checks["required_capabilities_known"] = required_capabilities_known is True
    checks["observe_only"] = config.observe_only
    if runtime is not None:
        runtime_value = Runtime(runtime)
        runtime_check = _runtime_check(runtime_value)
        checks["runtime"] = runtime_check
        ok = ok and bool(runtime_check["available"] and runtime_check["version"])
        logs = _log_check(runtime_value, log_root)
        checks["logs"] = logs
        ok = ok and bool(logs["available"])
    elif log_root is not None:
        logs = _log_check(None, log_root)
        checks["logs"] = logs
        ok = ok and bool(logs["available"])
    selected_state = state_path if state_path is not None else state_dir
    if selected_state is not None:
        state = _state_check(
            selected_state,
            expect_file=state_path is not None,
        )
        checks["state"] = state
        ok = ok and bool(state["ok"])
    if hook_manifest is not None:
        hook = _hook_check(runtime, hook_manifest)
        checks["hook_manifest"] = hook
        ok = ok and bool(hook["installed"] and hook["composable"])
    if adapter_commands:
        adapters = _adapter_checks(
            adapter_commands,
            project_dir=project_dir,
            inherit_env=adapter_inherit_env or {},
        )
        checks["adapters"] = adapters
        ok = ok and all(
            bool(result["resolved"] and result["contract"])
            for result in adapters.values()
        )
    return {"schema_version": 1, "ok": ok, "checks": checks}


def _coordinator_check() -> dict[str, object]:
    # Keep in step with the pinned `agent-coordinator` in pyproject.toml. The
    # pin and this constant are two halves of one decision: bumping only the
    # pin makes `doctor` report the harness incompatible with the library it
    # actually ships against.
    expected = "0.3.0"
    try:
        installed_version = metadata.version("agent-coordinator")
    except metadata.PackageNotFoundError:
        installed_version = None
    return {
        "installed": installed_version is not None,
        "version": installed_version,
        "expected": expected,
        "compatible": installed_version == expected,
    }


def _runtime_check(runtime: Runtime) -> dict[str, object]:
    executable = shutil.which(runtime.value)
    result: dict[str, object] = {
        "name": runtime.value,
        "available": executable is not None,
        "path": executable,
        "version": None,
    }
    if executable is None:
        return result
    environment = {
        key: os.environ[key]
        for key in ("HOME", "LANG", "LC_ALL", "PATH", "USER")
        if key in os.environ
    }
    try:
        completed = subprocess.run(
            [executable, "--version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
            shell=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result
    output = completed.stdout if completed.stdout else completed.stderr
    version = " ".join(output[:4096].decode("utf-8", errors="replace").split())
    if completed.returncode == 0 and version:
        result["version"] = version[:240]
    result["returncode"] = completed.returncode
    return result


def _log_check(
    runtime: Runtime | None,
    log_root: str | os.PathLike[str] | None,
) -> dict[str, object]:
    if log_root is not None:
        root = Path(log_root).expanduser()
        source = "explicit"
    elif runtime is Runtime.CLAUDE:
        root = Path.home() / ".claude" / "projects"
        source = "default"
    elif runtime is Runtime.CODEX:
        root = Path.home() / ".codex" / "sessions"
        source = "default"
    else:
        raise ValueError("runtime or log_root is required for log discovery")
    available = root.is_dir()
    count = 0
    if available:
        try:
            count = sum(1 for path in root.rglob("*.jsonl") if path.is_file())
        except OSError:
            available = False
    return {
        "root": str(root),
        "source": source,
        "available": available,
        "jsonl_count": count,
    }


def _state_check(
    value: str | os.PathLike[str],
    *,
    expect_file: bool,
) -> dict[str, object]:
    path = Path(value).expanduser()
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        metadata = None
    exists = metadata is not None
    symlink = metadata is not None and stat.S_ISLNK(metadata.st_mode)
    is_file = metadata is not None and stat.S_ISREG(metadata.st_mode)
    is_directory = metadata is not None and stat.S_ISDIR(metadata.st_mode)
    parent = path.parent if expect_file or not is_directory else path
    symlink_parent = _has_symlink_component(parent)
    parent_exists = parent.is_dir()
    parent_writable = parent_exists and os.access(parent, os.W_OK)
    restrictive: bool | None = None
    mode: str | None = None
    if exists and is_file:
        numeric_mode = stat.S_IMODE(metadata.st_mode)
        mode = oct(numeric_mode)
        restrictive = numeric_mode & 0o077 == 0
    valid_kind = not symlink and (not exists or is_file or not expect_file)
    permissions_ok = restrictive is not False
    return {
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "symlink": symlink,
        "symlink_parent": symlink_parent,
        "mode": mode,
        "restrictive": restrictive,
        "parent_exists": parent_exists,
        "parent_writable": parent_writable,
        "ok": (
            valid_kind
            and not symlink_parent
            and permissions_ok
            and parent_exists
            and parent_writable
        ),
    }


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
    return False


def _hook_check(
    runtime: str | Runtime | None,
    hook_manifest: str | os.PathLike[str],
) -> dict[str, object]:
    path = Path(hook_manifest).expanduser()
    result: dict[str, object] = {
        "path": str(path),
        "valid": False,
        "installed": False,
        "composable": False,
        "composed": False,
    }
    if runtime is None:
        result["error"] = "runtime is required to check hooks"
        return result
    try:
        installer = HookInstaller(runtime=runtime, path=path)
        installed = installer.check().installed
        preview = installer.install(dry_run=True)
    except (OSError, ValueError, RuntimeError) as exc:
        result["error"] = str(exc)[:240]
        return result
    result.update(
        {
            "valid": True,
            "installed": installed,
            "composable": preview.installed,
            "composed": installed and not preview.changed,
        }
    )
    return result


def _adapter_checks(
    commands: Mapping[str, Sequence[str]],
    *,
    project_dir: str | os.PathLike[str] | None,
    inherit_env: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, object]]:
    capsule = _doctor_capsule(project_dir)
    request = AdapterRequest(
        schema_version=1,
        operation=AdapterOperation.READ,
        idempotency_key="doctor-contract-v1",
        capsule=capsule,
    )
    results: dict[str, dict[str, object]] = {}
    for name, configured_argv in commands.items():
        argv = tuple(configured_argv)
        executable = shutil.which(str(Path(argv[0]).expanduser())) if argv else None
        result: dict[str, object] = {
            "resolved": executable is not None,
            "path": executable,
            "contract": False,
            "readback": False,
        }
        if executable is not None:
            adapter = JsonCommand(
                name=name,
                argv=(executable, *argv[1:]),
                max_response_bytes=64 * 1024,
                inherit_env=tuple(inherit_env.get(name, ())),
            )
            try:
                payload = adapter.execute(request.model_dump(mode="json"))
                response = AdapterResponse.model_validate(payload)
            except (RuntimeError, ValidationError) as exc:
                result["error"] = str(exc)[:240]
            else:
                matching_readback = bool(
                    response.ok and response.fingerprint == capsule.fingerprint
                )
                result["contract"] = bool(not response.ok or matching_readback)
                result["readback"] = matching_readback
                result["error"] = response.error
        results[name] = result
    return results


def _doctor_capsule(
    project_dir: str | os.PathLike[str] | None,
) -> HandoffCapsule:
    repository_path = Path(project_dir or Path.cwd()).expanduser().resolve()
    return HandoffCapsule(
        schema_version=1,
        chain_id="doctor",
        predecessor_conversation_id="doctor",
        target_generation=1,
        task_ids={"doctor": "contract-v1"},
        objective="Validate an executable checkpoint adapter contract.",
        exact_next_action="Return the capsule fingerprint without model work.",
        completed_criteria=(),
        remaining_criteria=("adapter contract validated",),
        repository_path=repository_path,
        branch="doctor",
        head="doctor",
        dirty_paths=(),
        file_anchors=(),
        symbol_anchors=(),
        test_results={"doctor": "read-only"},
        decisions=("use the read operation for a non-mutating probe",),
        blockers=(),
        process_summaries={},
        created_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
    )
