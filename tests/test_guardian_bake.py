from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from agent_session_harness.guardian_bake import (
    GuardianBakeDecision,
    GuardianBakeError,
    GuardianBakeReport,
    GuardianHighWaterMarks,
    ObservationWindow,
    ResourceHighWaterMarks,
    UsageHighWaterMarks,
    UsageSnapshot,
    redact_guardian_text,
)

START = datetime(2026, 7, 31, tzinfo=UTC)


def report(*, heartbeat: datetime = START, detail: str = "safe") -> GuardianBakeReport:
    return GuardianBakeReport.build(
        guardian_version="0.1.0",
        platform="linux",
        observation_window=ObservationWindow(
            started_at=START,
            ends_at=START + timedelta(hours=24),
        ),
        heartbeat_at=heartbeat,
        usage_before=UsageSnapshot(memory_bytes=2048, cpu_percent=0.5),
        usage_after=UsageSnapshot(memory_bytes=1024, cpu_percent=0.25),
        high_water_marks=GuardianHighWaterMarks(
            resources=ResourceHighWaterMarks(observed=4, managed=3, ambiguous=1),
            usage=UsageHighWaterMarks(memory_bytes=4096, cpu_percent=1.25),
        ),
        reap_decisions=[
            GuardianBakeDecision(
                reason_code="terminal_managed_child",
                performed=False,
                live_resource=False,
                evidence=["process_missing", detail],
            )
        ],
        refused_decisions=[],
        errors=[GuardianBakeError(stage="inspection", error_type="TimeoutError")],
    )


def test_report_requires_timezone_aware_window() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        ObservationWindow(
            started_at=START.replace(tzinfo=None),
            ends_at=(START + timedelta(days=1)).replace(tzinfo=None),
        )


def test_report_deduplication_ignores_heartbeat_but_tracks_state() -> None:
    first = report()
    repeated = report(heartbeat=START + timedelta(minutes=5))
    changed = report(detail="different")

    assert first.deduplication_key == repeated.deduplication_key
    assert first.deduplication_key != changed.deduplication_key
    assert len(first.deduplication_key) == 64


def test_redactor_removes_credentials_paths_and_arguments() -> None:
    value = (
        "Authorization: Bearer secret-token LINEAR_API_KEY=abc123 "
        'api_key: colon-secret {"token": "json-secret"} password="two words" '
        "/Users/alice/project --password hunter2 -p short-secret "
        "Authorization=Basic basic-secret command: tool positional-secret\n"
        "multiline-secret"
    )

    redacted = redact_guardian_text(value)

    assert "secret-token" not in redacted
    assert "abc123" not in redacted
    assert "alice" not in redacted
    assert "hunter2" not in redacted
    assert "colon-secret" not in redacted
    assert "json-secret" not in redacted
    assert "two words" not in redacted
    assert "short-secret" not in redacted
    assert "positional-secret" not in redacted
    assert "basic-secret" not in redacted
    assert "multiline-secret" not in redacted
    assert "[REDACTED]" in redacted


def test_report_redacts_free_form_evidence_before_model_storage() -> None:
    baked = report(detail="token=secret /home/alice/private --key value")

    encoded = baked.model_dump_json()
    assert "secret" not in encoded
    assert "alice" not in encoded
    assert " value" not in encoded
    assert "[REDACTED]" in encoded


def test_report_rejects_window_or_heartbeat_outside_window() -> None:
    with pytest.raises(ValidationError, match="ends after it starts"):
        ObservationWindow(started_at=START, ends_at=START)

    with pytest.raises(ValidationError, match="heartbeat"):
        report(heartbeat=START + timedelta(days=2))


def test_report_rejects_empty_decision_evidence() -> None:
    with pytest.raises(ValidationError):
        GuardianBakeDecision(
            reason_code="terminal_managed_child",
            performed=True,
            live_resource=False,
            evidence=[],
        )


def test_report_rejects_forged_deduplication_key() -> None:
    baked = report()
    payload = baked.model_dump()
    payload["deduplication_key"] = "f" * 64

    with pytest.raises(ValidationError, match="deduplication key"):
        GuardianBakeReport.model_validate(payload)


def test_report_preserves_before_after_usage() -> None:
    baked = report()

    assert baked.usage_before.memory_bytes == 2048
    assert baked.usage_after.memory_bytes == 1024
