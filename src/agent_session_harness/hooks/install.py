"""Additive JSON hook-manifest installation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from ..models import Runtime


OWNED_MARKER = "AGENT_SESSION_HARNESS_OWNED=v1"
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


@dataclass(frozen=True)
class HookInstallResult:
    changed: bool
    installed: bool


class HookInstaller:
    def __init__(self, *, runtime: str | Runtime, path: str | os.PathLike[str]):
        self.runtime = Runtime(runtime)
        self.path = Path(path)
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
            f"{OWNED_MARKER} agent-session-harness hook --runtime {self.runtime.value}"
        )
        for event_name in HOOK_EVENTS:
            groups = hooks.setdefault(event_name, [])
            if not isinstance(groups, list):
                raise ValueError(f"hook manifest '{event_name}' value must be an array")
            groups.append(
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": command, "timeout": 5}],
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
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
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
            and OWNED_MARKER in entry["command"]
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
                if any(HookInstaller._owned_entry(entry) for entry in group["hooks"]):
                    found.add(event_name)
        return found == set(HOOK_EVENTS)

    def _write(self, manifest: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = self.path.stat().st_mode & 0o777 if self.path.exists() else 0o600
        if self.path.exists() and not self.backup_path.exists():
            shutil.copy2(self.path, self.backup_path)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", text=True
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, mode)
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
