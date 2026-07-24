"""Durable checkpoint mirroring into a beads (`bd`) issue tracker.

Requires the `bd` CLI on PATH; the `beads` extra exists to document that
dependency and carries no Python packages of its own.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpoint_records import (
    bounded_line,
    emit_response,
    failure,
    read_stdin_request,
    success,
    validate_checkpoint_request,
)
from .command import sanitize_error

MAX_BD_OUTPUT_BYTES = 4 * 1_048_576
_BEAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


class BdCommandError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class BdClient:
    """Invoke ``bd`` with explicit argv and bounded structured output."""

    argv: tuple[str, ...] = ("bd",)
    cwd: Path = Path.cwd()
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.argv or any(not item for item in self.argv):
            raise ValueError("bd argv must contain non-empty arguments")
        if self.timeout_seconds <= 0:
            raise ValueError("bd timeout must be positive")
        object.__setattr__(self, "cwd", Path(self.cwd).expanduser().resolve())

    def show(self, bead_id: str) -> dict[str, Any]:
        payload = self._run("show", bead_id, "--json")
        if isinstance(payload, list) and len(payload) == 1:
            record = payload[0]
        elif isinstance(payload, dict):
            record = payload
        else:
            raise BdCommandError("bd show returned an invalid record", retryable=False)
        if not isinstance(record, dict) or record.get("id") != bead_id:
            raise BdCommandError("bd show returned the wrong bead", retryable=False)
        return record

    def append_notes(self, bead_id: str, note: str) -> None:
        self._run("update", bead_id, "--append-notes", note, "--json")

    def _run(self, *arguments: str) -> object:
        with (
            tempfile.TemporaryFile() as stdout_file,
            tempfile.TemporaryFile() as stderr_file,
        ):
            try:
                environment = os.environ.copy()
                environment["BD_NO_DAEMON"] = "1"
                completed = subprocess.run(
                    [*self.argv, *arguments],
                    cwd=self.cwd,
                    env=environment,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    check=False,
                    timeout=self.timeout_seconds,
                    shell=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise BdCommandError("bd command timed out", retryable=True) from exc
            except OSError as exc:
                raise BdCommandError(
                    "bd executable is unavailable", retryable=False
                ) from exc
            stdout = _read_bounded_command_output(stdout_file)
            stderr = _read_bounded_command_output(stderr_file)
        if completed.returncode != 0:
            detail = sanitize_error(stderr or stdout)
            retryable = not any(
                marker in detail.lower()
                for marker in ("not found", "unknown bead", "invalid id")
            )
            raise BdCommandError(f"bd command failed: {detail}", retryable=retryable)
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise BdCommandError(
                "bd command returned malformed JSON", retryable=False
            ) from exc


def _read_bounded_command_output(handle: Any) -> str:
    handle.seek(0, 2)
    if handle.tell() > MAX_BD_OUTPUT_BYTES:
        raise BdCommandError("bd command output exceeded limit", retryable=False)
    handle.seek(0)
    return handle.read().decode("utf-8", errors="replace")


def handle_request(
    request: Mapping[str, object], client: BdClient
) -> dict[str, object]:
    """Execute one write/read/acknowledge operation without raising to the host."""

    try:
        parsed = validate_checkpoint_request(
            request,
            task_id_keys=("bead",),
            task_id_label="bead task ID",
            task_id_pattern=_BEAD_ID,
        )
        marker = _checkpoint_marker(parsed.capsule, parsed.idempotency_key)
        if parsed.operation == "write":
            return _write_checkpoint(
                client=client,
                bead_id=parsed.task_id,
                capsule=parsed.capsule,
                marker=marker,
                fingerprint=parsed.fingerprint,
            )
        if parsed.operation == "read":
            return _verify_checkpoint(
                client,
                parsed.task_id,
                marker,
                parsed.capsule,
                parsed.fingerprint,
            )
        acknowledgement = _acknowledgement_marker(
            parsed.capsule, parsed.idempotency_key
        )
        return _write_marker(
            client=client,
            bead_id=parsed.task_id,
            marker=acknowledgement,
            fingerprint=parsed.fingerprint,
        )
    except BdCommandError as exc:
        return failure(str(exc), retryable=exc.retryable)
    except (KeyError, TypeError, ValueError) as exc:
        return failure(str(exc), retryable=False)


def _write_checkpoint(
    *,
    client: BdClient,
    bead_id: str,
    capsule: dict[str, Any],
    marker: str,
    fingerprint: str,
) -> dict[str, object]:
    existing = _notes(client.show(bead_id))
    canonical = _canonical_checkpoint_block(marker, capsule)
    if canonical in existing:
        return success(fingerprint)
    client.append_notes(bead_id, canonical)
    return _verify_checkpoint(client, bead_id, marker, capsule, fingerprint)


def _write_marker(
    *, client: BdClient, bead_id: str, marker: str, fingerprint: str
) -> dict[str, object]:
    existing = _notes(client.show(bead_id))
    if marker in existing:
        return success(fingerprint)
    client.append_notes(bead_id, marker)
    return _verify_marker(client, bead_id, marker, fingerprint)


def _verify_marker(
    client: BdClient, bead_id: str, marker: str, fingerprint: str
) -> dict[str, object]:
    if marker not in _notes(client.show(bead_id)):
        return failure("exact beads checkpoint read-back failed", retryable=True)
    return success(fingerprint)


def _canonical_checkpoint_block(marker: str, capsule: dict[str, Any]) -> str:
    canonical_capsule = json.dumps(
        capsule, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return f"{marker}\n\n```json\n{canonical_capsule}\n```"


def _verify_checkpoint(
    client: BdClient,
    bead_id: str,
    marker: str,
    capsule: dict[str, Any],
    fingerprint: str,
) -> dict[str, object]:
    expected = _canonical_checkpoint_block(marker, capsule)
    if expected not in _notes(client.show(bead_id)):
        return failure("exact beads checkpoint read-back failed", retryable=True)
    return success(fingerprint)


def _checkpoint_marker(capsule: dict[str, Any], idempotency_key: str) -> str:
    return "\n".join(
        (
            "agent-session-harness checkpoint",
            f"chain: {bounded_line(capsule['chain_id'], 'chain_id', 160)}",
            f"generation: {capsule['target_generation']}",
            f"fingerprint: {capsule['fingerprint']}",
            f"idempotency: {idempotency_key}",
            f"objective: {bounded_line(capsule['objective'], 'objective', 4000)}",
            "handoff-action: "
            + bounded_line(capsule["exact_next_action"], "exact_next_action", 4000),
            f"head: {bounded_line(capsule['head'], 'head', 128)}",
        )
    )


def _acknowledgement_marker(capsule: dict[str, Any], idempotency_key: str) -> str:
    return "\n".join(
        (
            "agent-session-harness acknowledgement",
            f"chain: {bounded_line(capsule['chain_id'], 'chain_id', 160)}",
            f"generation: {capsule['target_generation']}",
            f"fingerprint: {capsule['fingerprint']}",
            f"idempotency: {idempotency_key}",
        )
    )


def _notes(record: Mapping[str, object]) -> str:
    value = record.get("notes")
    return value if isinstance(value, str) else ""


def resolve_main_checkout(repository_path: Path) -> Path:
    """Prefer the repository whose `.beads` database a worktree links back to."""

    repository = repository_path.expanduser().resolve()
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--git-common-dir"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=2,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return repository
    if completed.returncode != 0:
        return repository
    common = Path(completed.stdout.strip())
    if not common.is_absolute():
        common = (repository / common).resolve()
    main_checkout = common.parent if common.name == ".git" else repository
    return main_checkout if (main_checkout / ".beads").is_dir() else repository


def main(*, argv: tuple[str, ...] = ("bd",)) -> int:
    """Read one checkpoint request from stdin and mirror it into beads."""

    try:
        request = read_stdin_request()
        capsule = request.get("capsule")
        repository = (
            Path(str(capsule.get("repository_path")))
            if isinstance(capsule, dict) and capsule.get("repository_path")
            else Path.cwd()
        )
        client = BdClient(argv=argv, cwd=resolve_main_checkout(repository))
        response = handle_request(request, client)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        response = failure(str(exc), retryable=False)
    emit_response(response)
    return 0
