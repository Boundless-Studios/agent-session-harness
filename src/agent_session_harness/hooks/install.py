"""Additive JSON hook-manifest installation."""

from __future__ import annotations

import json
import os
import shlex
import stat
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
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
    # Per-event explanation, present only for `check` and only when something
    # is wrong. `installed: false` on its own is undiagnosable: it names
    # neither the event nor what differed, so the only way to find out is to
    # read this module's source. See `HookInstaller.diagnose`.
    problems: tuple[HookProblem, ...] = ()


def _describe_command(entry: Any) -> str:
    """A short, recognizable label for a foreign hook entry."""
    if not isinstance(entry, dict):
        return repr(entry)
    command = entry.get("command")
    if not isinstance(command, str) or not command:
        return repr(entry)
    return command if len(command) <= 80 else command[:77] + "..."


@dataclass(frozen=True)
class HookProblem:
    event: str
    reason: str
    detail: str

    def render(self) -> str:
        return f"{self.event}: {self.reason} — {self.detail}"


class HookInstaller:
    def __init__(
        self,
        *,
        runtime: str | Runtime,
        path: str | os.PathLike[str],
        harness_command: str | os.PathLike[str] | None = None,
        expected_command: str | None = None,
    ):
        self.runtime = Runtime(runtime)
        self.path = Path(path)
        self.harness_command = self._harness_command(harness_command)
        self.expected_command = expected_command or (
            f"{OWNED_MARKER} {shlex.quote(self.harness_command)} "
            f"hook --runtime {self.runtime.value}"
        )
        if (
            not self.expected_command
            or len(self.expected_command) > 16_384
            or "\x00" in self.expected_command
            or "\n" in self.expected_command
            or OWNED_MARKER not in self.expected_command.split()
        ):
            raise ValueError("expected hook command is invalid")
        self.backup_path = self.path.with_suffix(
            self.path.suffix + ".agent-session-harness.bak"
        )

    def check(self) -> HookInstallResult:
        manifest = self._read()
        installed = self._is_installed(manifest)
        return HookInstallResult(
            changed=False,
            installed=installed,
            problems=() if installed else self.diagnose(manifest),
        )

    def diagnose(
        self, manifest: dict[str, Any] | None = None
    ) -> tuple[HookProblem, ...]:
        """Explain, per event, why ownership does not verify.

        `_is_installed` compares each event's group for EXACT equality against
        a single-owned-hook group, so any of several unrelated conditions
        produces the same `false`: the event is missing, the owned entry was
        never installed, its timeout drifted, its command drifted, or — the one
        that is genuinely surprising — another hook was appended into the
        harness's own group. That last case cost a real debugging session in
        gaia, where a response-budget guard had been added alongside the owned
        entry for PostToolUse and PreCompact and the only output was
        `{"changed":false,"installed":false}`.
        """
        if manifest is None:
            manifest = self._read()
        hooks = manifest.get("hooks")
        if not isinstance(hooks, dict):
            return (
                HookProblem(
                    event="*",
                    reason="no hook manifest",
                    detail=f"{self.path} has no 'hooks' object",
                ),
            )

        problems: list[HookProblem] = []
        for event_name in HOOK_EVENTS:
            groups = hooks.get(event_name)
            if not isinstance(groups, list):
                problems.append(
                    HookProblem(
                        event=event_name,
                        reason="event missing",
                        detail="no hook groups registered for this event",
                    )
                )
                continue

            expected = self._expected_group(event_name)
            matches = [group for group in groups if group == expected]
            if len(matches) == 1:
                continue
            if len(matches) > 1:
                problems.append(
                    HookProblem(
                        event=event_name,
                        reason="duplicate owned group",
                        detail=f"{len(matches)} identical owned groups; expected exactly 1",
                    )
                )
                continue

            problems.append(self._near_miss_problem(event_name, groups, expected))

        return tuple(problems)

    def _near_miss_problem(
        self,
        event_name: str,
        groups: list[Any],
        expected: dict[str, object],
    ) -> HookProblem:
        """Name the closest thing to the expected group and how it differs."""
        expected_entry = expected["hooks"][0]  # type: ignore[index]
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                continue
            owned = [entry for entry in group["hooks"] if self._owned_entry(entry)]
            if not owned:
                continue

            # The surprising case: correct entry, wrong company.
            if len(group["hooks"]) > 1:
                others = [
                    _describe_command(entry)
                    for entry in group["hooks"]
                    if not self._owned_entry(entry)
                ]
                return HookProblem(
                    event=event_name,
                    reason="owned hook shares its group",
                    detail=(
                        "the owned entry must be the ONLY hook in its "
                        f"matcher:'*' group; also present: {', '.join(others)}. "
                        "Move the other hook(s) into their own group — two "
                        "groups both matching '*' both run."
                    ),
                )
            if group.get("matcher") != "*":
                return HookProblem(
                    event=event_name,
                    reason="wrong matcher",
                    detail=f"expected matcher '*', found {group.get('matcher')!r}",
                )
            entry = owned[0]
            if entry.get("timeout") != expected_entry.get("timeout"):  # type: ignore[union-attr]
                return HookProblem(
                    event=event_name,
                    reason="timeout drift",
                    detail=(
                        f"expected timeout {expected_entry.get('timeout')}, "  # type: ignore[union-attr]
                        f"found {entry.get('timeout')!r}"
                    ),
                )
            if entry.get("command") != expected_entry.get("command"):  # type: ignore[union-attr]
                return HookProblem(
                    event=event_name,
                    reason="command drift",
                    detail="owned entry's command does not match the expected command",
                )
            return HookProblem(
                event=event_name,
                reason="group mismatch",
                detail=f"owned group is not equal to the expected group: {group!r}",
            )

        return HookProblem(
            event=event_name,
            reason="not installed",
            detail="no harness-owned hook entry registered for this event",
        )

    def install(self, *, dry_run: bool = False) -> HookInstallResult:
        manifest = self._read()
        updated = self._without_owned(manifest)
        hooks = updated.setdefault("hooks", {})
        if not isinstance(hooks, dict):
            raise ValueError("hook manifest 'hooks' value must be an object")
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
                            "command": self.expected_command,
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

    def _expected_group(self, event_name: str) -> dict[str, object]:
        expected_timeout = (
            SESSION_START_HOOK_TIMEOUT_SECONDS
            if event_name == "SessionStart"
            else DEFAULT_HOOK_TIMEOUT_SECONDS
        )
        return {
            "matcher": "*",
            "hooks": [
                {
                    "type": "command",
                    "command": self.expected_command,
                    "timeout": expected_timeout,
                }
            ],
        }

    def _is_installed(self, manifest: dict[str, Any]) -> bool:
        hooks = manifest.get("hooks")
        if not isinstance(hooks, dict):
            return False
        owned_count = 0
        for groups in hooks.values():
            if not isinstance(groups, list):
                continue
            for group in groups:
                if not isinstance(group, dict) or not isinstance(
                    group.get("hooks"), list
                ):
                    continue
                owned_count += sum(
                    1 for entry in group["hooks"] if self._owned_entry(entry)
                )
        if owned_count != len(HOOK_EVENTS):
            return False
        return all(
            isinstance(hooks.get(event_name), list)
            and hooks[event_name].count(self._expected_group(event_name)) == 1
            for event_name in HOOK_EVENTS
        )

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
