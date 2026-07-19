from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib

import pytest


NOW = datetime(2026, 7, 19, 5, 0, tzinfo=timezone.utc)


def _modules():
    try:
        harness = importlib.import_module("agent_session_harness.coordinator")
        coordinator = importlib.import_module("agent_coordinator")
    except ModuleNotFoundError:
        pytest.fail("released coordinator adapter is not implemented")
    return harness, coordinator


def test_rotation_fences_predecessor_and_advances_lease_epoch(tmp_path) -> None:
    harness, coordinator = _modules()
    service = coordinator.TaskCoordinator(
        coordinator.JsonlClaimStore(tmp_path / "claims.jsonl"),
        pid_is_live=lambda _pid: True,
    )
    adapter = harness.CoordinatorAdapter(service)

    first = adapter.claim(
        task_type="linear",
        task_id="BOU-2195",
        fingerprint="task-fingerprint",
        owner_session_id="conversation-0",
        owner_pid=1234,
        runtime="codex",
        worktree_path=str(tmp_path),
        lease_seconds=60,
        now=NOW,
    )
    heartbeat = adapter.heartbeat(
        first,
        lease_seconds=60,
        now=NOW + timedelta(seconds=10),
    )
    fenced = adapter.fence(first, now=NOW + timedelta(seconds=20))
    repeated_fence = adapter.fence(first, now=NOW + timedelta(seconds=20))
    successor = adapter.claim(
        task_type="linear",
        task_id="BOU-2195",
        fingerprint="task-fingerprint",
        owner_session_id="conversation-1",
        owner_pid=5678,
        runtime="codex",
        worktree_path=str(tmp_path),
        lease_seconds=60,
        now=NOW + timedelta(seconds=21),
    )

    assert heartbeat.claim_id == first.claim_id
    assert heartbeat.lease_epoch == first.lease_epoch
    assert fenced.release_reason == "context-rotation"
    assert repeated_fence == fenced
    assert successor.lease_epoch > first.lease_epoch
    with pytest.raises(harness.StaleOwnerError, match="stale"):
        adapter.heartbeat(
            first,
            lease_seconds=60,
            now=NOW + timedelta(seconds=22),
        )


def test_adapter_state_contains_only_fencing_identity(tmp_path) -> None:
    harness, coordinator = _modules()
    adapter = harness.CoordinatorAdapter(
        coordinator.TaskCoordinator(
            coordinator.JsonlClaimStore(tmp_path / "claims.jsonl")
        )
    )
    handle = adapter.claim(
        task_type="bead",
        task_id="bou-1",
        fingerprint="abc123",
        owner_session_id="conversation-0",
        owner_pid=None,
        runtime="claude",
        worktree_path=str(tmp_path),
        lease_seconds=30,
        now=NOW,
    )

    assert set(type(handle).model_fields) == {
        "claim_id",
        "lease_epoch",
        "task_type",
        "task_id",
        "task_fingerprint",
        "owner_session_id",
    }
