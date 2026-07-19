from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent_session_harness.activity import ActivitySnapshot, Quiescence
from agent_session_harness.safety import (
    ProjectSafetyObservation,
    ProjectSafetyStatus,
    merge_project_safety,
    sample_project_safety,
)


NOW = datetime(2026, 7, 19, 8, 0, tzinfo=timezone.utc)


def _activity(quiescence: Quiescence) -> ActivitySnapshot:
    return ActivitySnapshot(
        quiescence=quiescence,
        active_turn_ids=frozenset(),
        active_tool_ids=(
            frozenset({"tool-1"}) if quiescence is Quiescence.BUSY else frozenset()
        ),
        active_subagent_ids=frozenset(),
        active_critical_section_ids=frozenset(),
        processed_event_count=3,
        last_event_at=NOW,
        integrity_warnings=("event warning",)
        if quiescence is Quiescence.UNKNOWN
        else (),
    )


def _observation(
    status: ProjectSafetyStatus,
    *,
    critical: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
) -> ProjectSafetyObservation:
    return ProjectSafetyObservation(
        schema_version=1,
        status=status,
        active_critical_sections=critical,
        warnings=warnings,
    )


def test_quiescent_project_preserves_idle_runtime_activity() -> None:
    merged = merge_project_safety(
        _activity(Quiescence.IDLE),
        _observation(ProjectSafetyStatus.QUIESCENT),
    )

    assert merged.quiescence is Quiescence.IDLE
    assert merged.active_count == 0


def test_busy_project_blocks_rotation_and_names_critical_sections() -> None:
    merged = merge_project_safety(
        _activity(Quiescence.IDLE),
        _observation(
            ProjectSafetyStatus.BUSY,
            critical=("git-index-lock", "deployment"),
        ),
    )

    assert merged.quiescence is Quiescence.BUSY
    assert merged.active_critical_section_ids == frozenset(
        {"git-index-lock", "deployment"}
    )


def test_busy_project_without_a_name_gets_a_stable_sentinel() -> None:
    merged = merge_project_safety(
        _activity(Quiescence.IDLE),
        _observation(ProjectSafetyStatus.BUSY),
    )

    assert merged.active_critical_section_ids == frozenset({"project-safety-probe"})


@pytest.mark.parametrize(
    ("runtime_state", "project_state"),
    [
        (Quiescence.UNKNOWN, ProjectSafetyStatus.QUIESCENT),
        (Quiescence.IDLE, ProjectSafetyStatus.UNKNOWN),
    ],
)
def test_unknown_runtime_or_project_safety_fails_closed(
    runtime_state, project_state
) -> None:
    merged = merge_project_safety(
        _activity(runtime_state),
        _observation(project_state, warnings=("probe unavailable",)),
    )

    assert merged.quiescence is Quiescence.UNKNOWN
    assert "probe unavailable" in merged.integrity_warnings


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 1,
            "status": "quiescent",
            "active_critical_sections": (),
            "warnings": (),
            "tool_output": "private",
        },
        {
            "schema_version": 1,
            "status": "unknown",
            "active_critical_sections": (),
            "warnings": ("token=must-not-persist",),
        },
        {
            "schema_version": 1,
            "status": "unknown",
            "active_critical_sections": (),
            "warnings": ("aws_secret_access_key=must-not-persist",),
        },
        {
            "schema_version": 1,
            "status": "busy",
            "active_critical_sections": ("github_token_value=must-not-persist",),
            "warnings": (),
        },
        {
            "schema_version": 1,
            "status": "busy",
            "active_critical_sections": ("proxy_authorization=must-not-persist",),
            "warnings": (),
        },
        {
            "schema_version": 1,
            "status": "busy",
            "active_critical_sections": tuple(f"lock-{index}" for index in range(33)),
            "warnings": (),
        },
    ],
)
def test_project_safety_response_is_bounded_and_private(payload) -> None:
    with pytest.raises(ValidationError):
        ProjectSafetyObservation.model_validate(payload)


def test_project_safety_command_receives_only_bounded_identity(tmp_path) -> None:
    class CapturingCommand:
        def __init__(self) -> None:
            self.payload = None

        def execute(self, payload):
            self.payload = payload
            return {
                "schema_version": 1,
                "status": "quiescent",
                "active_critical_sections": [],
                "warnings": [],
            }

    command = CapturingCommand()

    observation = sample_project_safety(
        command,
        cwd=tmp_path,
        chain_id="chain-1",
        generation=3,
        process_group_id=4242,
    )

    assert observation.status is ProjectSafetyStatus.QUIESCENT
    assert command.payload == {
        "schema_version": 1,
        "operation": "probe",
        "cwd": str(tmp_path),
        "chain_id": "chain-1",
        "generation": 3,
        "process_group_id": 4242,
    }


def test_project_safety_command_failure_becomes_unknown_without_diagnostics(
    tmp_path,
) -> None:
    class BrokenCommand:
        def execute(self, _payload):
            raise RuntimeError("token=must-not-persist")

    observation = sample_project_safety(
        BrokenCommand(),
        cwd=tmp_path,
        chain_id="chain-1",
        generation=0,
        process_group_id=4242,
    )

    assert observation.status is ProjectSafetyStatus.UNKNOWN
    assert observation.warnings == ("project safety probe failed",)
