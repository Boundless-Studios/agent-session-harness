"""Command dispatch for agent-session-harness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

from .adapters import beads as beads_adapter
from .adapters import linear as linear_adapter
from .adapters import project_safety as project_safety_adapter
from .adapters import usage as usage_adapter
from .adapters.claude import ClaudeUsageReader
from .adapters.codex import CodexUsageReader
from .adapters.command import CommandAdapter, JsonCommand, sanitize_error
from .capsule import HandoffCapsule
from .checkpoint import CheckpointManager as DurableCheckpointManager
from .config import load_config
from .coordinator import CoordinatorAdapter
from .guardian import WATCHDOG_SHUTDOWN_MARGIN_SECONDS
from .hooks.command import (
    SUCCESSOR_ACK_TIMEOUT_SECONDS,
    SUCCESSOR_READY_POLL_SECONDS,
    SUCCESSOR_READY_TIMEOUT_SECONDS,
    run_hook,
)
from .hooks.install import HookInstaller
from .ledger import EventLedger
from .models import Runtime
from .outbox import MirrorOutbox
from .process import (
    DEFAULT_PROCESS_STARTUP_TIMEOUT_SECONDS,
    ManagedProcess,
    PosixProcessDriver,
)
from .report import build_report, doctor_report
from .safety import merge_project_safety, sample_project_safety
from .secure_files import atomic_write_private_text
from .supervisor import (
    CheckpointRequest,
    Supervisor,
    SupervisorPhase,
    UsageObservation,
    VerifiedCheckpoint,
    write_acknowledgement,
)


SYNCHRONOUS_ADAPTER_BUDGET_FRACTION = 0.8


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError) as exc:
        message = sanitize_error(str(exc), max_length=500)
        if getattr(args, "json_output", False):
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "ok": False,
                        "error": {
                            "type": type(exc).__name__,
                            "message": message,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(f"agent-session-harness: {message}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-session-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="run read-only diagnostics")
    doctor.add_argument("--runtime", choices=[runtime.value for runtime in Runtime])
    doctor.add_argument("--config")
    doctor.add_argument("--project-dir")
    doctor.add_argument("--state-dir")
    doctor.add_argument("--state-path")
    doctor.add_argument("--log-root")
    doctor.add_argument("--hook-manifest")
    doctor.add_argument("--required-capabilities-known", action="store_true")
    doctor.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=JSON_ARGV",
    )
    doctor.add_argument(
        "--adapter-env",
        action="append",
        default=[],
        metavar="NAME=KEY",
    )
    _add_json(doctor)
    doctor.set_defaults(handler=_run_doctor)

    inspect = subparsers.add_parser("inspect", help="inspect native usage logs")
    inspect.add_argument(
        "--runtime", required=True, choices=[item.value for item in Runtime]
    )
    inspect.add_argument("--path")
    inspect.add_argument("--lineage", action="append", default=[])
    inspect.add_argument("--window-tokens", type=int, default=200_000)
    _add_json(inspect)
    inspect.set_defaults(handler=_run_inspect)

    report = subparsers.add_parser("report", help="project canonical harness status")
    report.add_argument("--state", required=True)
    report.add_argument("--ledger")
    report.add_argument("--outbox")
    report.add_argument("--stale-after-seconds", type=float, default=30.0)
    _add_json(report)
    report.set_defaults(handler=_run_report)

    supervise = subparsers.add_parser(
        "supervise", help="validate or run managed launch"
    )
    supervise.add_argument(
        "--runtime", required=True, choices=[item.value for item in Runtime]
    )
    supervise.add_argument("--cwd", required=True)
    supervise.add_argument("--chain-id", required=True)
    supervise.add_argument("--task-type", required=True)
    supervise.add_argument("--task-id", required=True)
    supervise.add_argument("--task-fingerprint", required=True)
    supervise.add_argument("--state", required=True)
    supervise.add_argument("--config")
    supervise.add_argument("--check", action="store_true")
    supervise.add_argument("--required-capabilities-known", action="store_true")
    supervise.add_argument("--executable")
    supervise.add_argument("--runtime-arg", action="append", default=[])
    supervise.add_argument(
        "--runtime-env",
        action="append",
        default=[],
        metavar="KEY",
        help="inherit one explicitly named environment key into the managed runtime",
    )
    supervise.add_argument("--usage-adapter", metavar="JSON_ARGV")
    supervise.add_argument("--capsule-adapter", metavar="JSON_ARGV")
    supervise.add_argument("--safety-adapter", metavar="JSON_ARGV")
    supervise.add_argument(
        "--required-adapter",
        action="append",
        default=[],
        metavar="NAME=JSON_ARGV",
    )
    supervise.add_argument(
        "--mirror-adapter",
        action="append",
        default=[],
        metavar="NAME=JSON_ARGV",
    )
    supervise.add_argument(
        "--adapter-env",
        action="append",
        default=[],
        metavar="NAME=KEY",
    )
    supervise.add_argument("--coordinator-store")
    supervise.add_argument("--outbox")
    supervise.add_argument("--process-state-dir")
    supervise.add_argument(
        "--process-startup-timeout-seconds",
        type=float,
        default=DEFAULT_PROCESS_STARTUP_TIMEOUT_SECONDS,
    )
    supervise.add_argument("--poll-seconds", type=float, default=1.0)
    supervise.add_argument("--lease-seconds", type=int, default=60)
    supervise.add_argument("--heartbeat-interval-seconds", type=float)
    supervise.add_argument("--stop-timeout-seconds", type=float, default=10.0)
    supervise.add_argument("--stale-after-seconds", type=float)
    supervise.add_argument("--max-ticks", type=int, default=0)
    supervise.add_argument("--adapter-timeout-seconds", type=float, default=5.0)
    supervise.add_argument("--successor-retry-limit", type=int, default=1)
    _add_json(supervise)
    supervise.set_defaults(handler=_run_supervise)

    acknowledge = subparsers.add_parser(
        "acknowledge", help="acknowledge a verified successor handoff"
    )
    acknowledge.add_argument("--state", required=True)
    acknowledge.add_argument("--generation", required=True, type=int)
    acknowledge.add_argument("--fingerprint", required=True)
    acknowledge.add_argument("--conversation-id", required=True)
    acknowledge.add_argument("--owner-pid", type=int)
    _add_json(acknowledge)
    acknowledge.set_defaults(handler=_run_acknowledge)

    hook = subparsers.add_parser("hook", help="record one native hook event")
    hook.add_argument(
        "--runtime", required=True, choices=[item.value for item in Runtime]
    )
    hook.set_defaults(handler=_run_hook)

    hooks = subparsers.add_parser("hooks", help="manage additive native hook fragments")
    hook_actions = hooks.add_subparsers(dest="hook_action", required=True)
    for action in ("install", "check", "uninstall"):
        action_parser = hook_actions.add_parser(action)
        action_parser.add_argument(
            "--runtime", required=True, choices=[item.value for item in Runtime]
        )
        action_parser.add_argument("--path", required=True)
        action_parser.add_argument("--harness-command")
        action_parser.add_argument("--expected-command")
        action_parser.add_argument("--dry-run", action="store_true")
        _add_json(action_parser)
        action_parser.set_defaults(handler=_run_hooks)

    outbox = subparsers.add_parser("outbox", help="manage durable mirror retries")
    outbox_actions = outbox.add_subparsers(dest="outbox_action", required=True)
    replay = outbox_actions.add_parser("replay")
    replay.add_argument("--path", required=True)
    replay.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=JSON_ARGV",
        help='adapter executable argv, for example mirror=["/path/to/adapter"]',
    )
    replay.add_argument(
        "--adapter-env",
        action="append",
        default=[],
        metavar="NAME=KEY",
    )
    _add_json(replay)
    replay.set_defaults(handler=_run_outbox)

    adapter = subparsers.add_parser(
        "adapter",
        help="run one bounded JSON adapter over stdin/stdout",
    )
    adapter_actions = adapter.add_subparsers(dest="adapter_action", required=True)

    usage = adapter_actions.add_parser(
        "usage", help="sample usage for one managed conversation"
    )
    usage.add_argument("--ledger", action="append", default=[])
    usage.add_argument("--cwd", default=None)
    usage.add_argument("--claude-root", action="append")
    usage.add_argument("--codex-root", action="append")
    usage.add_argument(
        "--claude-fallback-window-tokens",
        type=int,
        help=(
            "context window for model identities the rollout names but the "
            "harness does not recognize; never overrides a recognized model"
        ),
    )
    usage.add_argument(
        "--max-rollout-bytes", type=int, default=usage_adapter.MAX_ROLLOUT_BYTES
    )
    usage.set_defaults(handler=_run_adapter)

    project_safety = adapter_actions.add_parser(
        "project-safety", help="probe host repository quiescence"
    )
    project_safety.set_defaults(handler=_run_adapter)

    beads = adapter_actions.add_parser(
        "beads", help="mirror one checkpoint into the beads tracker"
    )
    beads.add_argument("--bd-command", action="append", default=[])
    beads.set_defaults(handler=_run_adapter)

    linear = adapter_actions.add_parser(
        "linear", help="mirror one checkpoint into a Linear issue"
    )
    linear.add_argument(
        "--credential-variable",
        default=linear_adapter.DEFAULT_CREDENTIAL_VARIABLE,
        help="environment variable holding the Linear API key",
    )
    linear.set_defaults(handler=_run_adapter)
    return parser


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", dest="json_output")


def _run_doctor(args: argparse.Namespace) -> int:
    adapter_commands = _parse_named_argv(args.adapter)
    payload = doctor_report(
        runtime=args.runtime,
        config_path=args.config,
        project_dir=args.project_dir,
        state_dir=args.state_dir,
        state_path=args.state_path,
        log_root=args.log_root,
        hook_manifest=args.hook_manifest,
        required_capabilities_known=args.required_capabilities_known,
        adapter_commands=adapter_commands,
        adapter_inherit_env=_parse_named_environment(
            args.adapter_env,
            known_names=set(adapter_commands),
        ),
    )
    _emit(payload, json_output=args.json_output)
    return 0 if payload["ok"] else 1


def _run_inspect(args: argparse.Namespace) -> int:
    if args.runtime == Runtime.CLAUDE.value:
        if not args.path:
            raise ValueError("Claude inspection requires --path")
        usage = ClaudeUsageReader(window_tokens=args.window_tokens).read_file(args.path)
    else:
        paths = args.lineage or ([args.path] if args.path else [])
        if not paths:
            raise ValueError("Codex inspection requires --path or --lineage")
        reader = CodexUsageReader()
        usage = (
            reader.read_lineage(paths) if len(paths) > 1 else reader.read_file(paths[0])
        )
    _emit(usage.model_dump(mode="json"), json_output=args.json_output)
    return 0


def _run_adapter(args: argparse.Namespace) -> int:
    """Dispatch one stdin/stdout adapter; each writes its own bounded JSON."""

    if args.adapter_action == "usage":
        return usage_adapter.main(
            ledger_paths=args.ledger or None,
            cwd=args.cwd,
            claude_roots=args.claude_root,
            codex_roots=args.codex_root,
            claude_fallback_window_tokens=args.claude_fallback_window_tokens,
            max_rollout_bytes=args.max_rollout_bytes,
        )
    if args.adapter_action == "project-safety":
        return project_safety_adapter.main()
    if args.adapter_action == "beads":
        return beads_adapter.main(argv=tuple(args.bd_command) or ("bd",))
    return linear_adapter.main(credential_variable=args.credential_variable)


def _run_report(args: argparse.Namespace) -> int:
    report = build_report(
        state_path=args.state,
        ledger_path=args.ledger,
        outbox_path=args.outbox,
        stale_after_seconds=args.stale_after_seconds,
    )
    _emit(report.model_dump(mode="json"), json_output=args.json_output)
    return 0


def _run_supervise(args: argparse.Namespace) -> int:
    _validate_process_startup_timeout(args.process_startup_timeout_seconds)
    cwd = Path(args.cwd).expanduser().resolve()
    config = load_config(
        explicit_path=args.config,
        project_dir=cwd,
        required_capabilities_known=args.required_capabilities_known,
    )
    ready = cwd.is_dir() and (args.check or not config.observe_only)
    if args.check:
        payload = {
            "schema_version": 1,
            "ready": ready,
            "observe_only": config.observe_only,
            "runtime": args.runtime,
            "chain_id": args.chain_id,
            "state": str(Path(args.state).expanduser()),
            "task": {
                "type": args.task_type,
                "id": args.task_id,
                "fingerprint": args.task_fingerprint,
            },
            "mode": "preflight",
        }
        _emit(payload, json_output=args.json_output)
        return 0 if ready else 1

    if not cwd.is_dir():
        raise ValueError(f"working directory does not exist: {cwd}")
    if config.observe_only:
        raise ValueError(
            "managed supervision requires known capabilities and observe_only=false"
        )
    if args.usage_adapter is None:
        raise ValueError("managed supervision requires --usage-adapter")
    if args.capsule_adapter is None:
        raise ValueError("managed supervision requires --capsule-adapter")

    required_specs = _parse_named_argv(args.required_adapter)
    if not required_specs:
        raise ValueError("managed supervision requires a --required-adapter")
    mirror_specs = _parse_named_argv(args.mirror_adapter)
    _validate_supervise_intervals(
        args,
        required_adapter_count=len(required_specs),
        mirror_adapter_count=len(mirror_specs),
    )
    adapter_names = set(required_specs) | set(mirror_specs)
    adapter_inherit_env = _parse_named_environment(
        args.adapter_env,
        known_names=adapter_names,
    )
    state_path = Path(args.state).expanduser()
    timeout = args.adapter_timeout_seconds
    executable = _resolve_executable(args.executable or args.runtime)
    usage_command = JsonCommand(
        name="usage adapter",
        argv=_resolved_json_argv(args.usage_adapter, label="usage adapter"),
        timeout_seconds=timeout,
    )
    capsule_command = JsonCommand(
        name="capsule adapter",
        argv=_resolved_json_argv(args.capsule_adapter, label="capsule adapter"),
        timeout_seconds=timeout,
    )
    safety_command = (
        JsonCommand(
            name="project safety adapter",
            argv=_resolved_json_argv(args.safety_adapter, label="safety adapter"),
            timeout_seconds=timeout,
            max_response_bytes=64 * 1024,
        )
        if args.safety_adapter is not None
        else None
    )
    required_adapters = tuple(
        CommandAdapter(
            name=name,
            argv=_resolve_argv(argv),
            timeout_seconds=timeout,
            inherit_env=adapter_inherit_env.get(name, ()),
        )
        for name, argv in required_specs.items()
    )
    mirror_adapters = tuple(
        CommandAdapter(
            name=name,
            argv=_resolve_argv(argv),
            timeout_seconds=timeout,
            inherit_env=adapter_inherit_env.get(name, ()),
        )
        for name, argv in mirror_specs.items()
    )
    outbox_path = (
        Path(args.outbox).expanduser()
        if args.outbox
        else state_path.with_suffix(state_path.suffix + ".outbox.jsonl")
    )
    coordinator_path = (
        Path(args.coordinator_store).expanduser()
        if args.coordinator_store
        else state_path.with_suffix(state_path.suffix + ".claims.jsonl")
    )
    process_state_dir = (
        Path(args.process_state_dir).expanduser()
        if args.process_state_dir
        else state_path.parent / "processes"
    )
    process_driver = PosixProcessDriver(
        process_state_dir,
        startup_timeout_seconds=args.process_startup_timeout_seconds,
    )
    checkpoint_manager = _ExecutableCheckpointManager(
        capsule_command=capsule_command,
        durable_manager=DurableCheckpointManager(
            required_adapters=required_adapters,
            mirror_adapters=mirror_adapters,
            outbox=MirrorOutbox(outbox_path),
        ),
        capsule_dir=state_path.parent / "capsules",
    )
    heartbeat_interval = args.heartbeat_interval_seconds
    if heartbeat_interval is None:
        heartbeat_interval = min(20.0, args.lease_seconds / 3)
    managed = Supervisor(
        runtime=args.runtime,
        chain_id=args.chain_id,
        cwd=cwd,
        task_type=args.task_type,
        task_id=args.task_id,
        task_fingerprint=args.task_fingerprint,
        executable=executable,
        runtime_args=tuple(args.runtime_arg),
        runtime_environment=_runtime_environment(args.runtime_env),
        state_path=state_path,
        process_driver=process_driver,
        usage_reader=_ExecutableUsageReader(usage_command),
        checkpoint_manager=checkpoint_manager,
        coordinator=CoordinatorAdapter.from_path(coordinator_path),
        warn_percent=config.governor.warn_percent,
        rotate_percent=config.governor.rotate_percent,
        lease_seconds=args.lease_seconds,
        heartbeat_interval_seconds=heartbeat_interval,
        stop_timeout_seconds=args.stop_timeout_seconds,
        successor_retry_limit=args.successor_retry_limit,
        successor_ack_timeout_seconds=SUCCESSOR_ACK_TIMEOUT_SECONDS,
    )
    ticks = 0
    activity_ledger = EventLedger(managed.lifecycle_path)
    interrupted = False
    mirror_replay_error: str | None = None
    try:
        mirror_replay_error = _replay_mirrors_fail_open(
            checkpoint_manager.durable_manager
        )
        started = managed.start()
        if started.phase is SupervisorPhase.BLOCKED:
            raise RuntimeError(
                "supervisor state is blocked; resolve retained ownership and use "
                "a new chain/state path for a new managed run"
            )
        while args.max_ticks == 0 or ticks < args.max_ticks:
            if managed.snapshot.phase is SupervisorPhase.AWAITING_ACK:
                # SessionStart runs before the first prompt, so service its
                # durable acknowledgement ahead of normal cadence. Do not put
                # the hook behind a project probe, mirror replay, or the
                # operator-configured poll interval.
                activity = activity_ledger.materialize(
                    now=datetime.now(tz=timezone.utc),
                    stale_after_seconds=(
                        args.stale_after_seconds
                        if args.stale_after_seconds is not None
                        else config.governor.stale_event_timeout_seconds
                    ),
                )
                snapshot = managed.tick(activity)
                ticks += 1
                if snapshot.phase is SupervisorPhase.COMPLETED:
                    break
                if snapshot.phase is SupervisorPhase.AWAITING_ACK and (
                    args.max_ticks == 0 or ticks < args.max_ticks
                ):
                    time.sleep(SUCCESSOR_READY_POLL_SECONDS)
                continue
            time.sleep(args.poll_seconds)
            activity = activity_ledger.materialize(
                now=datetime.now(tz=timezone.utc),
                stale_after_seconds=(
                    args.stale_after_seconds
                    if args.stale_after_seconds is not None
                    else config.governor.stale_event_timeout_seconds
                ),
            )
            if safety_command is not None:
                activity = merge_project_safety(
                    activity,
                    sample_project_safety(
                        safety_command,
                        cwd=cwd,
                        chain_id=managed.chain_id,
                        generation=managed.snapshot.generation,
                        process_group_id=managed.snapshot.process_group_id,
                    ),
                )
            snapshot = managed.tick(activity)
            ticks += 1
            if snapshot.phase is SupervisorPhase.COMPLETED:
                break
            if snapshot.phase is SupervisorPhase.AWAITING_ACK:
                continue
            replay_error = _replay_mirrors_fail_open(checkpoint_manager.durable_manager)
            if replay_error is not None:
                mirror_replay_error = replay_error
    except KeyboardInterrupt:
        interrupted = True
    finally:
        managed.shutdown()

    snapshot = managed.snapshot
    payload = {
        "schema_version": 1,
        "mode": "managed",
        "ticks": ticks,
        "state": snapshot.phase.value,
        "runtime": snapshot.runtime.value,
        "chain_id": snapshot.chain_id,
        "conversation_id": snapshot.conversation_id,
        "generation": snapshot.generation,
        "process_pid": snapshot.process_pid,
        "interrupted": interrupted,
        "mirror_replay_error": mirror_replay_error,
        # Surfaced so a wedged usage sampler is visible to the operator instead
        # of silently never rotating (BOU-2208).
        "usage_alarm": snapshot.usage_alarm,
        # Surfaced for the same reason: a managed session whose hooks stopped
        # firing never rotates, and without this it looks healthy until it runs
        # out of context (BOU-2222).
        "liveness_alarm": snapshot.liveness_alarm,
    }
    _emit(payload, json_output=args.json_output)
    return 130 if interrupted else 0


def _run_acknowledge(args: argparse.Namespace) -> int:
    owner_pid = args.owner_pid
    if owner_pid is None:
        raw_owner_pid = os.environ.get("AGENT_SESSION_HARNESS_OWNER_PID")
        if raw_owner_pid is None:
            owner_pid = os.getpid()
        else:
            try:
                owner_pid = int(raw_owner_pid)
            except ValueError as exc:
                raise ValueError("managed owner PID must be an integer") from exc
            if owner_pid <= 0:
                raise ValueError("managed owner PID must be positive")
    path = write_acknowledgement(
        state_path=args.state,
        generation=args.generation,
        fingerprint=args.fingerprint,
        conversation_id=args.conversation_id,
        owner_pid=owner_pid,
    )
    _emit(
        {"schema_version": 1, "ok": True, "path": str(path)},
        json_output=args.json_output,
    )
    return 0


def _run_hook(args: argparse.Namespace) -> int:
    return run_hook(
        runtime=args.runtime,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _run_hooks(args: argparse.Namespace) -> int:
    installer = HookInstaller(
        runtime=args.runtime,
        path=args.path,
        harness_command=args.harness_command,
        expected_command=args.expected_command,
    )
    if args.hook_action == "install":
        result = installer.install(dry_run=args.dry_run)
    elif args.hook_action == "uninstall":
        result = installer.uninstall(dry_run=args.dry_run)
    else:
        result = installer.check()
    _emit(
        {"changed": result.changed, "installed": result.installed},
        json_output=args.json_output,
    )
    if args.hook_action == "check" and not result.installed:
        return 1
    return 0


def _run_outbox(args: argparse.Namespace) -> int:
    adapter_commands = _parse_named_argv(args.adapter)
    adapter_inherit_env = _parse_named_environment(
        args.adapter_env,
        known_names=set(adapter_commands),
    )
    adapters = {
        name: CommandAdapter(
            name=name,
            argv=argv,
            inherit_env=adapter_inherit_env.get(name, ()),
        )
        for name, argv in adapter_commands.items()
    }
    result = MirrorOutbox(args.path).replay(adapters)
    _emit(
        {
            "attempted": result.attempted,
            "succeeded": result.succeeded,
            "retained": result.retained,
            "dead_lettered": result.dead_lettered,
        },
        json_output=args.json_output,
    )
    return 0


def _emit(payload: Any, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return
    if isinstance(payload, dict):
        for key in sorted(payload):
            value = payload[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            print(f"{key}: {value}", file=sys.stderr)
        return
    print(payload, file=sys.stderr)


class _ExecutableUsageReader:
    def __init__(self, command: JsonCommand):
        self.command = command

    def sample(self, process: ManagedProcess) -> UsageObservation:
        payload = self.command.execute(
            {
                "schema_version": 1,
                "operation": "sample",
                "process": {
                    "pid": process.pid,
                    "process_group_id": process.process_group_id,
                    "registry_key": process.registry_key,
                    "identity": process.identity,
                    "command_digest": process.command_digest,
                    "launch_nonce": process.launch_nonce,
                },
            }
        )
        return UsageObservation.model_validate(payload)


class _ExecutableCheckpointManager:
    def __init__(
        self,
        *,
        capsule_command: JsonCommand,
        durable_manager: DurableCheckpointManager,
        capsule_dir: Path,
    ) -> None:
        self.capsule_command = capsule_command
        self.durable_manager = durable_manager
        self.capsule_dir = capsule_dir

    def checkpoint(self, request: CheckpointRequest) -> VerifiedCheckpoint:
        response = self.capsule_command.execute(
            {
                "schema_version": 1,
                "operation": "checkpoint",
                "checkpoint": request.model_dump(mode="json"),
            }
        )
        capsule_payload = response.get("capsule", response)
        capsule = HandoffCapsule.model_validate(capsule_payload)
        if capsule.chain_id != request.chain_id:
            raise ValueError("capsule adapter returned the wrong chain")
        if capsule.predecessor_conversation_id != request.predecessor_conversation_id:
            raise ValueError("capsule adapter returned the wrong predecessor")
        if capsule.target_generation != request.target_generation:
            raise ValueError("capsule adapter returned the wrong generation")
        path = self.capsule_dir / (
            f"generation-{request.target_generation}-{capsule.fingerprint}.json"
        )
        _atomic_private_write(path, capsule.canonical_bytes() + b"\n")
        result = self.durable_manager.checkpoint(
            capsule,
            idempotency_key=request.idempotency_key,
        )
        return VerifiedCheckpoint(
            verified=result.verified,
            fingerprint=result.fingerprint,
            path=path,
        )

    def acknowledge(
        self,
        capsule: HandoffCapsule,
        *,
        idempotency_key: str,
    ) -> bool:
        return self.durable_manager.acknowledge(
            capsule,
            idempotency_key=idempotency_key,
        ).verified


def _atomic_private_write(path: Path, value: bytes) -> None:
    atomic_write_private_text(path, value.decode("utf-8"))


def _parse_named_argv(specifications: list[str]) -> dict[str, tuple[str, ...]]:
    parsed: dict[str, tuple[str, ...]] = {}
    for specification in specifications:
        name, separator, encoded_argv = specification.partition("=")
        if not separator or not name.strip():
            raise ValueError("adapter must use NAME=JSON_ARGV")
        name = name.strip()
        if name in parsed:
            raise ValueError(f"duplicate adapter name: {name}")
        parsed[name] = _parse_json_argv(encoded_argv, label=f"adapter {name}")
    return parsed


def _parse_named_environment(
    specifications: list[str],
    *,
    known_names: set[str],
) -> dict[str, tuple[str, ...]]:
    parsed: dict[str, list[str]] = {}
    for specification in specifications:
        name, separator, key = specification.partition("=")
        name = name.strip()
        key = key.strip()
        if not separator or not name or not key or not key.isidentifier():
            raise ValueError("adapter environment must use NAME=KEY")
        if name not in known_names:
            raise ValueError(f"adapter environment names unknown adapter: {name}")
        inherited = parsed.setdefault(name, [])
        if key not in inherited:
            inherited.append(key)
    return {name: tuple(keys) for name, keys in parsed.items()}


def _runtime_environment(keys: list[str]) -> dict[str, str]:
    inherited: dict[str, str] = {}
    seen: set[str] = set()
    for key in keys:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ValueError(f"invalid runtime environment key: {key}")
        if key.startswith("AGENT_SESSION_HARNESS_"):
            raise ValueError(f"reserved runtime environment key: {key}")
        if key in seen:
            raise ValueError(f"duplicate runtime environment key: {key}")
        seen.add(key)
        if key in os.environ:
            inherited[key] = os.environ[key]
    return inherited


def _replay_mirrors_fail_open(manager: DurableCheckpointManager) -> str | None:
    try:
        manager.replay_mirrors(max_attempts=1)
    except Exception as exc:
        return sanitize_error(str(exc), max_length=500)
    return None


def _parse_json_argv(encoded_argv: str, *, label: str) -> tuple[str, ...]:
    try:
        argv = json.loads(encoded_argv)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} argv must be valid JSON") from exc
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
    ):
        raise ValueError(f"{label} argv must be a non-empty JSON string array")
    return tuple(argv)


def _resolved_json_argv(encoded_argv: str, *, label: str) -> tuple[str, ...]:
    return _resolve_argv(_parse_json_argv(encoded_argv, label=label))


def _resolve_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    return (_resolve_executable(argv[0]), *argv[1:])


def _resolve_executable(executable: str) -> str:
    resolved = shutil.which(str(Path(executable).expanduser()))
    if resolved is None:
        raise ValueError(f"executable is not available: {executable}")
    return resolved


def _validate_process_startup_timeout(value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("process startup timeout must be positive and finite")


def _validate_supervise_intervals(
    args: argparse.Namespace,
    *,
    required_adapter_count: int,
    mirror_adapter_count: int,
) -> None:
    _validate_process_startup_timeout(args.process_startup_timeout_seconds)
    if args.poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")
    if args.lease_seconds <= 0:
        raise ValueError("lease seconds must be positive")
    if args.stop_timeout_seconds < 0:
        raise ValueError("stop timeout seconds must be non-negative")
    if args.stale_after_seconds is not None and args.stale_after_seconds <= 0:
        raise ValueError("stale event timeout must be positive")
    if args.max_ticks < 0:
        raise ValueError("max ticks must be non-negative")
    if args.adapter_timeout_seconds <= 0:
        raise ValueError("adapter timeout seconds must be positive")
    if args.successor_retry_limit < 0:
        raise ValueError("successor retry limit must be non-negative")
    watchdog_seconds = args.lease_seconds - WATCHDOG_SHUTDOWN_MARGIN_SECONDS
    if 2 * (args.adapter_timeout_seconds + args.poll_seconds) >= watchdog_seconds:
        raise ValueError(
            "adapter timeout and poll interval must leave at least half the "
            "watchdog lease for supervision"
        )
    checkpoint_seconds = (
        1 + (2 * required_adapter_count) + mirror_adapter_count
    ) * args.adapter_timeout_seconds
    if checkpoint_seconds >= (watchdog_seconds * SYNCHRONOUS_ADAPTER_BUDGET_FRACTION):
        raise ValueError(
            "cumulative checkpoint adapter budget must leave watchdog headroom"
        )
    acknowledgement_seconds = (
        SUCCESSOR_READY_TIMEOUT_SECONDS
        + SUCCESSOR_READY_POLL_SECONDS
        + (
            (required_adapter_count + mirror_adapter_count)
            * args.adapter_timeout_seconds
        )
    )
    if acknowledgement_seconds >= (
        SUCCESSOR_ACK_TIMEOUT_SECONDS * SYNCHRONOUS_ADAPTER_BUDGET_FRACTION
    ):
        raise ValueError(
            "cumulative acknowledgement adapter budget must leave SessionStart headroom"
        )
    if (
        args.heartbeat_interval_seconds is not None
        and not 0 <= args.heartbeat_interval_seconds < args.lease_seconds
    ):
        raise ValueError(
            "heartbeat interval must satisfy 0 <= interval < lease seconds"
        )
