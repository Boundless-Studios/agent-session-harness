from __future__ import annotations

import json

import pytest

from agent_session_harness.coordinator import CoordinatorAdapter
from agent_session_harness.events import LifecycleEvent
from agent_session_harness.hooks.install import HookInstaller
from agent_session_harness.ledger import EventLedger
from agent_session_harness.outbox import MirrorOutbox
from agent_session_harness.process import PosixProcessDriver
from agent_session_harness.report import build_report, doctor_report
from agent_session_harness.secure_files import (
    UnsafePathError,
    append_private_text,
    atomic_write_private_text,
    exclusive_lock,
    private_unlink,
    read_private_text,
)


def _event(tmp_path) -> LifecycleEvent:
    return LifecycleEvent.model_validate(
        {
            "schema_version": 1,
            "event_id": "event-1",
            "runtime": "codex",
            "chain_id": "chain-1",
            "conversation_id": "conversation-1",
            "generation": 0,
            "event_type": "turn.started",
            "timestamp": "2026-07-19T03:00:00+00:00",
            "cwd": tmp_path,
            "owner_pid": 4321,
            "activity_id": "turn-1",
        }
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda path: append_private_text(path, "new\n"),
        lambda path: atomic_write_private_text(path, "new\n"),
        lambda path: read_private_text(path),
        lambda path: private_unlink(path),
    ],
)
def test_private_file_operations_reject_a_symlink_target(tmp_path, operation) -> None:
    victim = tmp_path / "victim"
    victim.write_text("keep\n", encoding="utf-8")
    target = tmp_path / "state.json"
    target.symlink_to(victim)

    with pytest.raises(UnsafePathError, match="symlink"):
        operation(target)

    assert victim.read_text(encoding="utf-8") == "keep\n"
    assert target.is_symlink()


def test_private_lock_rejects_a_symlink_target(tmp_path) -> None:
    victim = tmp_path / "victim.lock"
    victim.write_text("keep\n", encoding="utf-8")
    lock = tmp_path / "state.lock"
    lock.symlink_to(victim)

    with pytest.raises(UnsafePathError, match="symlink"):
        with exclusive_lock(lock):
            pytest.fail("symlink lock must never be acquired")

    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_private_file_operations_reject_a_symlink_parent(tmp_path) -> None:
    victim_dir = tmp_path / "victim-dir"
    victim_dir.mkdir()
    redirected_parent = tmp_path / "state"
    redirected_parent.symlink_to(victim_dir, target_is_directory=True)

    with pytest.raises(UnsafePathError, match="symlink"):
        append_private_text(redirected_parent / "events.jsonl", "new\n")

    assert not (victim_dir / "events.jsonl").exists()


def test_event_ledger_does_not_follow_a_symlink_target(tmp_path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("keep\n", encoding="utf-8")
    path = tmp_path / "events.jsonl"
    path.symlink_to(victim)

    with pytest.raises(UnsafePathError):
        EventLedger(path).append(_event(tmp_path))

    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_outbox_does_not_follow_a_symlink_target(tmp_path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("keep\n", encoding="utf-8")
    path = tmp_path / "outbox.jsonl"
    path.symlink_to(victim)

    with pytest.raises(UnsafePathError):
        MirrorOutbox(path).pending()

    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_process_registry_does_not_follow_a_symlink_target(tmp_path) -> None:
    victim = tmp_path / "victim"
    victim.write_text(
        json.dumps(
            {
                "pid": 1,
                "process_group_id": 1,
                "registry_key": "chain:0",
                "identity": "identity",
                "command_digest": "digest",
                "launch_nonce": "nonce",
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "registry.json"
    path.symlink_to(victim)

    with pytest.raises(UnsafePathError):
        PosixProcessDriver._read_registry(path, "chain:0")

    assert victim.exists()


def test_coordinator_store_does_not_follow_a_symlink_target(tmp_path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("keep\n", encoding="utf-8")
    path = tmp_path / "claims.jsonl"
    path.symlink_to(victim)
    coordinator = CoordinatorAdapter.from_path(path)

    with pytest.raises(UnsafePathError):
        coordinator.claim(
            task_type="linear",
            task_id="BOU-2195",
            fingerprint="fingerprint",
            owner_session_id="chain-1:0",
            owner_pid=4321,
            runtime="codex",
            worktree_path=str(tmp_path),
            lease_seconds=60,
        )

    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_status_report_does_not_follow_a_symlink_state(tmp_path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("keep\n", encoding="utf-8")
    path = tmp_path / "supervisor.json"
    path.symlink_to(victim)

    with pytest.raises(UnsafePathError):
        build_report(state_path=path)

    assert victim.read_text(encoding="utf-8") == "keep\n"


def test_hook_installer_does_not_follow_a_symlink_manifest(tmp_path) -> None:
    victim = tmp_path / "victim"
    victim.write_text('{"hooks":{}}\n', encoding="utf-8")
    path = tmp_path / "settings.json"
    path.symlink_to(victim)

    with pytest.raises(UnsafePathError):
        HookInstaller(runtime="codex", path=path).install()

    assert victim.read_text(encoding="utf-8") == '{"hooks":{}}\n'


def test_doctor_marks_a_symlink_state_as_unsafe(tmp_path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("{}\n", encoding="utf-8")
    victim.chmod(0o600)
    path = tmp_path / "supervisor.json"
    path.symlink_to(victim)

    report = doctor_report(state_path=path)

    state = report["checks"]["state"]
    assert state["symlink"] is True
    assert state["ok"] is False
