from __future__ import annotations

import json
import stat
from datetime import UTC, datetime, timedelta

import pytest

from agent_session_harness.guardian_bake import (
    GuardianBakeReport,
    GuardianHighWaterMarks,
    ObservationWindow,
    ResourceHighWaterMarks,
    UsageHighWaterMarks,
    UsageSnapshot,
)
from agent_session_harness.guardian_bake_spool import GuardianBakeSpool

NOW = datetime(2026, 7, 31, tzinfo=UTC)


def report(memory_bytes: int = 10, heartbeat: datetime = NOW) -> GuardianBakeReport:
    return GuardianBakeReport.build(
        guardian_version="0.1.0",
        platform="darwin",
        observation_window=ObservationWindow(
            started_at=NOW,
            ends_at=NOW + timedelta(days=1),
        ),
        heartbeat_at=heartbeat,
        usage_before=UsageSnapshot(memory_bytes=memory_bytes, cpu_percent=0.5),
        usage_after=UsageSnapshot(memory_bytes=memory_bytes, cpu_percent=0.5),
        high_water_marks=GuardianHighWaterMarks(
            resources=ResourceHighWaterMarks(observed=1, managed=1, ambiguous=0),
            usage=UsageHighWaterMarks(memory_bytes=memory_bytes, cpu_percent=0.5),
        ),
        reap_decisions=[],
        refused_decisions=[],
        errors=[],
    )


def test_spool_persists_pending_reports_across_restart(tmp_path) -> None:
    path = tmp_path / "bake.json"
    created = GuardianBakeSpool(path).append(report(), now=NOW)

    pending = GuardianBakeSpool(path).pending()

    assert [item.record_id for item in pending] == [created.record_id]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_identical_consecutive_state_compacts_repeat_count(tmp_path) -> None:
    spool = GuardianBakeSpool(tmp_path / "bake.json")
    first = spool.append(report(), now=NOW)
    repeated = spool.append(
        report(heartbeat=NOW + timedelta(minutes=1)),
        now=NOW + timedelta(minutes=1),
    )

    assert repeated.record_id == first.record_id
    assert repeated.repeat_count == 2
    assert repeated.last_seen_at == NOW + timedelta(minutes=1)
    assert len(spool.pending()) == 1


def test_acknowledged_records_are_not_pending(tmp_path) -> None:
    spool = GuardianBakeSpool(tmp_path / "bake.json")
    created = spool.append(report(), now=NOW)

    spool.acknowledge(created.record_id, now=NOW + timedelta(minutes=2))

    assert spool.pending() == []
    assert spool.list()[0].delivered_at == NOW + timedelta(minutes=2)


def test_bound_evicts_delivered_records_but_never_pending(tmp_path) -> None:
    spool = GuardianBakeSpool(tmp_path / "bake.json", max_records=2)
    first = spool.append(report(10), now=NOW)
    spool.acknowledge(first.record_id, now=NOW)
    second = spool.append(report(20), now=NOW)
    spool.acknowledge(second.record_id, now=NOW)

    third = spool.append(report(30), now=NOW)

    assert [
        item.report.high_water_marks.usage.memory_bytes for item in spool.list()
    ] == [
        20,
        30,
    ]
    fourth = spool.append(report(40), now=NOW)
    with pytest.raises(ValueError, match="undelivered"):
        spool.append(report(50), now=NOW)
    assert [item.record_id for item in spool.pending()] == [
        third.record_id,
        fourth.record_id,
    ]


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not-json", "invalid JSON"),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "records": [
                        {
                            "record_id": "same-record-id-0001",
                            "report": report().model_dump(mode="json"),
                            "first_seen_at": NOW.isoformat(),
                            "last_seen_at": NOW.isoformat(),
                            "repeat_count": 1,
                            "delivered_at": None,
                        },
                        {
                            "record_id": "same-record-id-0001",
                            "report": report(20).model_dump(mode="json"),
                            "first_seen_at": NOW.isoformat(),
                            "last_seen_at": NOW.isoformat(),
                            "repeat_count": 1,
                            "delivered_at": None,
                        },
                    ],
                }
            ),
            "duplicate record IDs",
        ),
    ],
)
def test_spool_rejects_corrupt_state(tmp_path, payload: str, message: str) -> None:
    path = tmp_path / "bake.json"
    path.write_text(payload)
    path.chmod(0o600)

    with pytest.raises(RuntimeError, match=message):
        GuardianBakeSpool(path).list()


def test_spool_rejects_oversized_state_before_parsing(tmp_path) -> None:
    path = tmp_path / "bake.json"
    path.write_text("x" * 100)
    path.chmod(0o600)

    with pytest.raises(ValueError, match="exceeds 64 bytes"):
        GuardianBakeSpool(path, max_bytes=64).list()


def test_spool_rejects_model_copy_with_stale_deduplication_key(tmp_path) -> None:
    stale = report().model_copy(update={"deduplication_key": "f" * 64})

    with pytest.raises(ValueError, match="deduplication key"):
        GuardianBakeSpool(tmp_path / "bake.json").append(stale, now=NOW)
