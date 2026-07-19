"""Additive JSON hook-manifest installation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import stat
from typing import Any

from ..models import Runtime
from ..secure_files import (
    atomic_write_private_text,
    private_exists,
    private_file_mode,
    read_private_text,
)


OWNED_MARKER_PREFIX = "AGENT_SESSION_HARNESS_OWNED="
OWNED_MARKER = f"{OWNED_MARKER_PREFIX}v1"
HOOK_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "SubagentStart",
    "SubagentStop",
    "Stop",
    "PreCompact",
    "SessionEnd",
)
DEFAULT_HOOK_TIMEOUT_SECONDS = 5
SESSION_START_HOOK_TIMEOUT_SECONDS = 35


@dataclass(frozen=True)
class HookInstallResult:
    changed: bool
    installed: bool


class HookInstaller:
    def __init__(
        self,
        *,
        runtime: str | Runtime,
        path: str | os.PathLike[str],
        harness_command: str | os.PathLike[str] | None = None,
    ):
        self.runtime = Runtime(runtime)
        self.path = Path(path)
        self.harness_command = self._harness_command(harness_command)
        self.backup_path = self.path.with_suffix(
            self.path.suffix + ".agent-session-harness.bak"
        )

    def check(self) -> HookInstallResult:
        manifest = self._read()
        return HookInstallResult(changed=False, installed=self._is_installed(manifest))

    def install(self, *, dry_run: bool = False) -> HookInstallResult:
        manifest = self._read()
        updated = self._without_owned(manifest)
        hooks = updated.setdefault("hooks", {})
        if not isinstance(hooks, dict):
            raise ValueError("hook manifest 'hooks' value must be an object")
        command = (
            f"{OWNED_MARKER} {shlex.quote(self.harness_command)} "
            f"hook --runtime {self.runtime.value}"
        )
        for event_name in HOOK_EVENTS:
            groups = hooks.setdefault(event_name, [])
            if not isinstance(groups, list):
                raise ValueError(f"hook manifest '{event_name}' value must be an array")
            groups.append(
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": command,
                            "timeout": (
                                SESSION_START_HOOK_TIMEOUT_SECONDS
                                if event_name == "SessionStart"
                                else DEFAULT_HOOK_TIMEOUT_SECONDS
                            ),
                        }
                    ],
                }
            )
        changed = updated != manifest
        if changed and not dry_run:
            self._write(updated)
        return HookInstallResult(changed=changed, installed=True)

    def uninstall(self, *, dry_run: bool = False) -> HookInstallResult:
        manifest = self._read()
        updated = self._without_owned(manifest)
        changed = updated != manifest
        if changed and not dry_run:
            self._write(updated)
        return HookInstallResult(changed=changed, installed=False)

    def _read(self) -> dict[str, Any]:
        if not private_exists(self.path):
            return {}
        try:
            payload = json.loads(read_private_text(self.path))
        except json.JSONDecodeError as exc:
            raise ValueError("hook manifest contains invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("hook manifest must be a JSON object")
        return payload

    def _without_owned(self, manifest: dict[str, Any]) -> dict[str, Any]:
        updated = deepcopy(manifest)
        hooks = updated.get("hooks")
        if not isinstance(hooks, dict):
            return updated
        for event_name in list(hooks):
            groups = hooks[event_name]
            if not isinstance(groups, list):
                continue
            retained_groups: list[Any] = []
            for group in groups:
                if not isinstance(group, dict):
                    retained_groups.append(group)
                    continue
                commands = group.get("hooks")
                if not isinstance(commands, list):
                    retained_groups.append(group)
                    continue
                retained_commands = [
                    entry for entry in commands if not self._owned_entry(entry)
                ]
                if retained_commands:
                    retained_group = deepcopy(group)
                    retained_group["hooks"] = retained_commands
                    retained_groups.append(retained_group)
            if retained_groups:
                hooks[event_name] = retained_groups
            else:
                del hooks[event_name]
        return updated

    @staticmethod
    def _owned_entry(entry: object) -> bool:
        return (
            isinstance(entry, dict)
            and isinstance(entry.get("command"), str)
            and OWNED_MARKER_PREFIX in entry["command"]
        )

    @staticmethod
    def _current_entry(entry: object, event_name: str) -> bool:
        expected_timeout = (
            SESSION_START_HOOK_TIMEOUT_SECONDS
            if event_name == "SessionStart"
            else DEFAULT_HOOK_TIMEOUT_SECONDS
        )
        return (
            isinstance(entry, dict)
            and isinstance(entry.get("command"), str)
            and OWNED_MARKER in entry["command"]
            and entry.get("timeout") == expected_timeout
        )

    @staticmethod
    def _is_installed(manifest: dict[str, Any]) -> bool:
        hooks = manifest.get("hooks")
        if not isinstance(hooks, dict):
            return False
        found: set[str] = set()
        for event_name, groups in hooks.items():
            if not isinstance(groups, list):
                continue
            for group in groups:
                if not isinstance(group, dict) or not isinstance(
                    group.get("hooks"), list
                ):
                    continue
                if any(
                    HookInstaller._current_entry(entry, event_name)
                    for entry in group["hooks"]
                ):
                    found.add(event_name)
        return found == set(HOOK_EVENTS)

    @staticmethod
    def _harness_command(value: str | os.PathLike[str] | None) -> str:
        if value is None:
            return "agent-session-harness"
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("harness command must be an absolute path")
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise ValueError("harness command does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("harness command must be a regular file")
        if not os.access(path, os.X_OK):
            raise ValueError("harness command is not executable")
        return str(path)

    def _write(self, manifest: dict[str, Any]) -> None:
        exists = private_exists(self.path)
        mode = private_file_mode(self.path) if exists else 0o600
        if exists and not private_exists(self.backup_path):
            atomic_write_private_text(
                self.backup_path,
                read_private_text(self.path),
                mode=mode,
            )
        encoded = json.dumps(manifest, indent=2) + "\n"
        atomic_write_private_text(self.path, encoded, mode=mode)
