from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_session_harness.guardian_singleton import (
    DuplicateGuardianError,
    GuardianSingleton,
    StaleGuardianError,
)

NOW = datetime(2026, 7, 30, tzinfo=UTC)


def test_active_guardian_rejects_duplicate_owner(tmp_path) -> None:
    first = GuardianSingleton._for_test(tmp_path)
    duplicate = GuardianSingleton._for_test(tmp_path)
    first.acquire(
        owner_session_id="guardian-1",
        owner_pid=100,
        lease_seconds=30,
        now=NOW,
    )

    with pytest.raises(DuplicateGuardianError, match="active"):
        duplicate.acquire(
            owner_session_id="guardian-2",
            owner_pid=200,
            lease_seconds=30,
            now=NOW + timedelta(seconds=1),
        )


def test_same_session_cannot_idempotently_acquire_twice(tmp_path) -> None:
    first = GuardianSingleton._for_test(tmp_path)
    duplicate = GuardianSingleton._for_test(tmp_path)
    first.acquire(
        owner_session_id="shared-session",
        owner_pid=None,
        lease_seconds=30,
        now=NOW,
    )

    with pytest.raises(DuplicateGuardianError, match="active"):
        duplicate.acquire(
            owner_session_id="shared-session",
            owner_pid=None,
            lease_seconds=30,
            now=NOW + timedelta(seconds=1),
        )


def test_expired_guardian_is_fenced_by_higher_epoch_successor(tmp_path) -> None:
    first = GuardianSingleton._for_test(tmp_path)
    successor = GuardianSingleton._for_test(tmp_path)
    first_handle = first.acquire(
        owner_session_id="guardian-1",
        owner_pid=100,
        lease_seconds=10,
        now=NOW,
    )
    successor_handle = successor.acquire(
        owner_session_id="guardian-2",
        owner_pid=200,
        lease_seconds=10,
        now=NOW + timedelta(seconds=11),
    )

    assert successor_handle.lease_epoch > first_handle.lease_epoch
    with pytest.raises(StaleGuardianError, match="stale"):
        first.assert_current(first_handle, now=NOW + timedelta(seconds=11))
    with pytest.raises(StaleGuardianError, match="stale"):
        first.heartbeat(
            first_handle,
            lease_seconds=10,
            now=NOW + timedelta(seconds=12),
        )


def test_heartbeat_and_clean_release_preserve_fencing_identity(tmp_path) -> None:
    singleton = GuardianSingleton._for_test(tmp_path)
    handle = singleton.acquire(
        owner_session_id="guardian-1",
        owner_pid=100,
        lease_seconds=30,
        now=NOW,
    )

    heartbeat = singleton.heartbeat(
        handle,
        lease_seconds=30,
        now=NOW + timedelta(seconds=5),
    )
    singleton.assert_current(heartbeat, now=NOW + timedelta(seconds=6))
    released = singleton.release(heartbeat, now=NOW + timedelta(seconds=7))

    assert heartbeat.claim_id == handle.claim_id
    assert heartbeat.lease_epoch == handle.lease_epoch
    assert released.release_reason == "guardian-shutdown"
    with pytest.raises(StaleGuardianError, match="stale"):
        singleton.assert_current(heartbeat, now=NOW + timedelta(seconds=8))


def test_bound_ownership_exposes_service_lease_contract(tmp_path) -> None:
    singleton = GuardianSingleton._for_test(tmp_path)
    handle = singleton.acquire(
        owner_session_id="guardian-1",
        owner_pid=100,
        lease_seconds=30,
        now=NOW,
    )

    ownership = singleton.bind(handle, clock=lambda: NOW + timedelta(seconds=1))

    proof = ownership.current_proof()
    assert ownership.handle == handle
    assert proof.claim_id == handle.claim_id
    assert proof.lease_epoch == handle.lease_epoch


def test_canonical_user_root_ignores_process_state_environment(monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "/tmp/first-state-root")
    first = GuardianSingleton.canonical_state_root()
    monkeypatch.setenv("XDG_STATE_HOME", "/tmp/second-state-root")
    second = GuardianSingleton.canonical_state_root()

    assert first == second
    assert "agent-session-harness" in first.parts
