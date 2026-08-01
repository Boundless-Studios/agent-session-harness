from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agent_session_harness.adapters.guardian_bake_linear import (
    TARGET_ISSUE,
    GuardianBakeLinearSink,
)
from agent_session_harness.guardian_bake import (
    GuardianBakeReport,
    GuardianHighWaterMarks,
    ObservationWindow,
    ResourceHighWaterMarks,
    UsageHighWaterMarks,
)
from agent_session_harness.guardian_bake_spool import GuardianBakeSpool

NOW = datetime(2026, 7, 31, tzinfo=UTC)


def report() -> GuardianBakeReport:
    return GuardianBakeReport.build(
        guardian_version="0.1.0",
        platform="linux",
        observation_window=ObservationWindow(
            started_at=NOW,
            ends_at=NOW + timedelta(days=1),
        ),
        heartbeat_at=NOW,
        high_water_marks=GuardianHighWaterMarks(
            resources=ResourceHighWaterMarks(observed=2, managed=1, ambiguous=1),
            usage=UsageHighWaterMarks(memory_bytes=1024, cpu_percent=0.25),
        ),
        reap_decisions=[],
        refused_decisions=[],
        errors=[],
    )


class LinearFake:
    def __init__(self, *, existing_marker: str | None = None, fail_create=False):
        self.comments = [existing_marker] if existing_marker else []
        self.fail_create = fail_create
        self.payloads: list[dict[str, object]] = []

    def __call__(self, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        query = str(payload["query"])
        variables = payload["variables"]
        if "GuardianBakeComments" in query:
            assert isinstance(variables, dict)
            assert variables["issueId"] == TARGET_ISSUE
            return {
                "data": {
                    "issue": {
                        "id": "linear-issue-uuid",
                        "comments": {
                            "nodes": [
                                {"id": str(index), "body": body}
                                for index, body in enumerate(self.comments)
                            ],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        },
                    }
                }
            }
        if self.fail_create:
            raise RuntimeError("offline")
        assert isinstance(variables, dict)
        comment_input = variables["input"]
        assert isinstance(comment_input, dict)
        assert comment_input["issueId"] == "linear-issue-uuid"
        self.comments.append(str(comment_input["body"]))
        return {"data": {"commentCreate": {"success": True, "comment": {"id": "c"}}}}


def test_sink_posts_only_to_fixed_issue_and_acknowledges_after_readback(
    tmp_path,
) -> None:
    spool = GuardianBakeSpool(tmp_path / "bake.json")
    record = spool.append(report(), now=NOW)
    fake = LinearFake()

    delivered = GuardianBakeLinearSink(fake).drain(spool, now=NOW)

    assert delivered == [record.record_id]
    assert spool.pending() == []
    assert all("issueCreate" not in str(payload["query"]) for payload in fake.payloads)
    assert any(record.report.deduplication_key in body for body in fake.comments)


def test_existing_marker_is_idempotent_without_new_comment(tmp_path) -> None:
    spool = GuardianBakeSpool(tmp_path / "bake.json")
    record = spool.append(report(), now=NOW)
    marker = f"guardian-bake:{record.report.deduplication_key}"
    fake = LinearFake(existing_marker=marker)

    GuardianBakeLinearSink(fake).drain(spool, now=NOW)

    assert spool.pending() == []
    assert fake.comments == [marker]


def test_transport_failure_leaves_report_pending(tmp_path) -> None:
    spool = GuardianBakeSpool(tmp_path / "bake.json")
    record = spool.append(report(), now=NOW)

    with pytest.raises(RuntimeError, match="offline"):
        GuardianBakeLinearSink(LinearFake(fail_create=True)).drain(spool, now=NOW)

    assert [item.record_id for item in spool.pending()] == [record.record_id]


def test_sink_has_no_configurable_issue_target() -> None:
    with pytest.raises(TypeError):
        GuardianBakeLinearSink(LinearFake(), issue="BOU-9999")  # type: ignore[call-arg]
