"""Command dispatch for agent-session-harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .adapters.claude import ClaudeUsageReader
from .adapters.codex import CodexUsageReader
from .adapters.command import CommandAdapter
from .config import load_config
from .hooks.command import run_hook
from .hooks.install import HookInstaller
from .models import Runtime
from .outbox import MirrorOutbox
from .report import build_report, doctor_report
from .supervisor import write_acknowledgement


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"agent-session-harness: {exc}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-session-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="run read-only diagnostics")
    doctor.add_argument("--runtime", choices=[runtime.value for runtime in Runtime])
    doctor.add_argument("--config")
    doctor.add_argument("--project-dir")
    doctor.add_argument("--state-dir")
    doctor.add_argument("--hook-manifest")
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
    _add_json(supervise)
    supervise.set_defaults(handler=_run_supervise)

    acknowledge = subparsers.add_parser(
        "acknowledge", help="acknowledge a verified successor handoff"
    )
    acknowledge.add_argument("--state", required=True)
    acknowledge.add_argument("--generation", required=True, type=int)
    acknowledge.add_argument("--fingerprint", required=True)
    acknowledge.add_argument("--conversation-id", required=True)
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
    _add_json(replay)
    replay.set_defaults(handler=_run_outbox)
    return parser


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", dest="json_output")


def _run_doctor(args: argparse.Namespace) -> int:
    payload = doctor_report(
        runtime=args.runtime,
        config_path=args.config,
        project_dir=args.project_dir,
        state_dir=args.state_dir,
        hook_manifest=args.hook_manifest,
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
    cwd = Path(args.cwd).expanduser().resolve()
    config = load_config(
        explicit_path=args.config,
        project_dir=cwd,
        required_capabilities_known=args.required_capabilities_known,
    )
    ready = cwd.is_dir() and (args.check or not config.observe_only)
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
        "mode": "preflight" if args.check else "integration-required",
    }
    _emit(payload, json_output=args.json_output)
    if not args.check:
        print(
            "managed execution is provided by the Supervisor API and a project checkpoint adapter",
            file=sys.stderr,
        )
        return 2
    return 0 if ready else 1


def _run_acknowledge(args: argparse.Namespace) -> int:
    path = write_acknowledgement(
        state_path=args.state,
        generation=args.generation,
        fingerprint=args.fingerprint,
        conversation_id=args.conversation_id,
    )
    _emit(
        {"schema_version": 1, "ok": True, "path": str(path)},
        json_output=args.json_output,
    )
    return 0


def _run_hook(args: argparse.Namespace) -> int:
    return run_hook(runtime=args.runtime, stdin=sys.stdin, stdout=sys.stdout)


def _run_hooks(args: argparse.Namespace) -> int:
    installer = HookInstaller(runtime=args.runtime, path=args.path)
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
    return 0


def _run_outbox(args: argparse.Namespace) -> int:
    adapters = {}
    for specification in args.adapter:
        name, separator, encoded_argv = specification.partition("=")
        if not separator or not name:
            raise ValueError("adapter must use NAME=JSON_ARGV")
        argv = json.loads(encoded_argv)
        if not isinstance(argv, list) or not all(
            isinstance(item, str) for item in argv
        ):
            raise ValueError("adapter argv must be a JSON string array")
        adapters[name] = CommandAdapter(name=name, argv=tuple(argv))
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
            print(f"{key}: {value}")
        return
    print(payload)
