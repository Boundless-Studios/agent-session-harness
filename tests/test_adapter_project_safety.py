from __future__ import annotations

import io
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_session_harness import cli
from agent_session_harness.adapters import project_safety
from agent_session_harness.safety import (
    ProjectSafetyObservation,
    ProjectSafetyStatus,
    sample_project_safety,
)


def _worktree(tmp_path: Path) -> Path:
    root = tmp_path / "worktree"
    root.mkdir()
    (root / ".git").mkdir()
    return root


def _marker(root: Path, **overrides: object) -> Path:
    directory = root / ".agent-session-harness" / "critical-sections"
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": 2,
        "name": "deployment",
        "pid": os.getpid(),
        "process_identity": "a" * 64,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    payload.update(overrides)
    if payload["schema_version"] == 1 and "process_identity" not in overrides:
        payload.pop("process_identity")
    path = directory / "deployment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_clean_worktree_is_quiescent(tmp_path) -> None:
    result = project_safety.probe_project_safety(_worktree(tmp_path))

    assert result.status is ProjectSafetyStatus.QUIESCENT
    assert result.active_critical_sections == ()
    assert result.warnings == ()


def test_probe_returns_the_shared_observation_model(tmp_path) -> None:
    result = project_safety.probe_project_safety(_worktree(tmp_path))

    assert isinstance(result, ProjectSafetyObservation)
    assert result.schema_version == 1


def test_git_index_lock_is_busy(tmp_path) -> None:
    root = _worktree(tmp_path)
    (root / ".git" / "index.lock").write_text("", encoding="utf-8")

    result = project_safety.probe_project_safety(root)

    assert result.status is ProjectSafetyStatus.BUSY
    assert result.active_critical_sections == ("git-index-lock",)


def test_slow_sibling_stop_hook_process_blocks_rotation(tmp_path, monkeypatch) -> None:
    root = _worktree(tmp_path)
    monkeypatch.setattr(
        project_safety,
        "runtime_process_group_members",
        lambda _pgid: {4242, 4243},
    )

    result = project_safety.probe_project_safety(root, process_group_id=4242)

    assert result.status is ProjectSafetyStatus.BUSY
    assert result.active_critical_sections == ("runtime-child-processes",)


def test_live_critical_marker_is_busy(tmp_path, monkeypatch) -> None:
    root = _worktree(tmp_path)
    _marker(root)
    monkeypatch.setattr(project_safety, "process_identity", lambda _pid: "a" * 64)

    result = project_safety.probe_project_safety(root)

    assert result.status is ProjectSafetyStatus.BUSY
    assert result.active_critical_sections == ("deployment",)


def test_dead_critical_marker_is_stale_but_not_busy(tmp_path, monkeypatch) -> None:
    root = _worktree(tmp_path)
    _marker(root, pid=987654)
    monkeypatch.setattr(project_safety, "pid_status", lambda _pid: "dead")

    result = project_safety.probe_project_safety(root)

    assert result.status is ProjectSafetyStatus.QUIESCENT
    assert result.active_critical_sections == ()
    assert result.warnings == ("stale-critical-marker:deployment",)


def test_recycled_pid_marker_is_stale_but_not_busy(tmp_path, monkeypatch) -> None:
    root = _worktree(tmp_path)
    _marker(root, schema_version=2, process_identity="a" * 64)
    monkeypatch.setattr(project_safety, "pid_status", lambda _pid: "alive")
    monkeypatch.setattr(project_safety, "process_identity", lambda _pid: "b" * 64)

    result = project_safety.probe_project_safety(root)

    assert result.status is ProjectSafetyStatus.QUIESCENT
    assert result.active_critical_sections == ()
    assert result.warnings == ("stale-critical-marker:deployment",)


def test_legacy_marker_without_process_identity_is_unknown(tmp_path) -> None:
    root = _worktree(tmp_path)
    _marker(root, schema_version=1)

    result = project_safety.probe_project_safety(root)

    assert result.status is ProjectSafetyStatus.UNKNOWN
    assert result.active_critical_sections == ()
    assert result.warnings == ("invalid-critical-marker",)


def test_live_marker_with_unverifiable_identity_is_unknown(
    tmp_path, monkeypatch
) -> None:
    root = _worktree(tmp_path)
    _marker(root)
    monkeypatch.setattr(project_safety, "pid_status", lambda _pid: "alive")
    monkeypatch.setattr(project_safety, "process_identity", lambda _pid: None)

    result = project_safety.probe_project_safety(root)

    assert result.status is ProjectSafetyStatus.UNKNOWN
    assert result.active_critical_sections == ()
    assert result.warnings == ("critical-marker-identity-unknown",)


def test_current_process_identity_is_stable() -> None:
    first = project_safety.process_identity(os.getpid())
    second = project_safety.process_identity(os.getpid())

    assert first is not None
    assert first == second
    assert len(first) == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 3},
        {"name": "../../outside"},
        {"pid": "not-a-pid"},
        {"process_identity": "not-an-identity"},
        {"created_at": "not-a-time"},
        {
            "created_at": (
                datetime.now(tz=timezone.utc) + timedelta(minutes=10)
            ).isoformat()
        },
        {"unexpected_private_field": "private"},
    ],
)
def test_malformed_or_private_marker_is_unknown(tmp_path, overrides) -> None:
    root = _worktree(tmp_path)
    _marker(root, **overrides)

    result = project_safety.probe_project_safety(root)

    assert result.status is ProjectSafetyStatus.UNKNOWN
    assert result.active_critical_sections == ()
    assert result.warnings == ("invalid-critical-marker",)


def test_unknown_pid_liveness_is_unknown(tmp_path, monkeypatch) -> None:
    root = _worktree(tmp_path)
    _marker(root)
    monkeypatch.setattr(project_safety, "pid_status", lambda _pid: "unknown")

    result = project_safety.probe_project_safety(root)

    assert result.status is ProjectSafetyStatus.UNKNOWN
    assert result.warnings == ("critical-marker-owner-unknown",)


def test_symlink_marker_directory_is_unknown(tmp_path) -> None:
    root = _worktree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    harness_dir = root / ".agent-session-harness"
    harness_dir.mkdir()
    (harness_dir / "critical-sections").symlink_to(outside, target_is_directory=True)

    result = project_safety.probe_project_safety(root)

    assert result.status is ProjectSafetyStatus.UNKNOWN
    assert result.warnings == ("unsafe-critical-marker-path",)


def test_marker_directory_outside_worktree_is_unknown(tmp_path) -> None:
    root = _worktree(tmp_path)

    result = project_safety.probe_project_safety(root, marker_dir=tmp_path / "outside")

    assert result.status is ProjectSafetyStatus.UNKNOWN
    assert result.warnings == ("unsafe-critical-marker-path",)


def test_worktree_gitdir_file_resolves_relative_target(tmp_path) -> None:
    root = tmp_path / "worktree"
    root.mkdir()
    git_dir = tmp_path / "git-data"
    git_dir.mkdir()
    (root / ".git").write_text("gitdir: ../git-data\n", encoding="utf-8")

    assert project_safety.resolve_git_dir(root) == git_dir.resolve()


def test_probe_satisfies_the_json_safety_command_contract(
    tmp_path, monkeypatch
) -> None:
    root = _worktree(tmp_path)
    (root / ".git" / "index.lock").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        project_safety,
        "runtime_process_group_members",
        lambda pgid: {pgid},
    )

    observation = sample_project_safety(
        project_safety.ProjectSafetyProbe(),
        cwd=root,
        chain_id="chain-1",
        generation=1,
        process_group_id=4242,
    )

    assert observation.status is ProjectSafetyStatus.BUSY
    assert observation.active_critical_sections == ("git-index-lock",)


def test_cli_returns_bounded_unknown_for_an_invalid_request(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        "sys.stdin", io.TextIOWrapper(io.BytesIO(b'{"schema_version":2}'))
    )

    assert cli.main(["adapter", "project-safety"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "status": "unknown",
        "active_critical_sections": [],
        "warnings": ["invalid-request"],
    }
