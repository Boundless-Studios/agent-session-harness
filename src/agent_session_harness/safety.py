"""Bounded project-quiescence contract and runtime activity merge."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import re
from typing import Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .activity import ActivitySnapshot, Quiescence


_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:[a-z0-9]+[-_])*(?:api[-_]?key|authorization|credential|password|secret|token)"
    r"(?:[-_][a-z0-9]+)*"
    r"\s*[:=]\s*\S+"
)


class ProjectSafetyStatus(str, Enum):
    QUIESCENT = "quiescent"
    BUSY = "busy"
    UNKNOWN = "unknown"


class ProjectSafetyObservation(BaseModel):
    """One sanitized response from a configured project safety probe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    status: ProjectSafetyStatus
    active_critical_sections: tuple[str, ...] = Field(
        default=(),
        max_length=32,
    )
    warnings: tuple[str, ...] = Field(default=(), max_length=16)

    @field_validator("active_critical_sections", "warnings")
    @classmethod
    def bound_private_text(
        cls,
        values: tuple[str, ...],
        info: ValidationInfo,
    ) -> tuple[str, ...]:
        limit = 160 if info.field_name == "active_critical_sections" else 240
        if any(not value.strip() for value in values):
            raise ValueError("project safety values must not be blank")
        if any(len(value) > limit for value in values):
            raise ValueError(f"project safety values exceed {limit} characters")
        if any(_CREDENTIAL_ASSIGNMENT.search(value) for value in values):
            raise ValueError("project safety response contains credential-shaped text")
        return values


class JsonSafetyCommand(Protocol):
    def execute(self, payload: Mapping[str, object]) -> dict[str, object]: ...


def sample_project_safety(
    command: JsonSafetyCommand,
    *,
    cwd: Path,
    chain_id: str,
    generation: int,
    process_group_id: int | None,
) -> ProjectSafetyObservation:
    """Invoke a project probe and convert every adapter failure to unknown."""

    try:
        if (
            process_group_id is None
            or isinstance(process_group_id, bool)
            or process_group_id <= 0
        ):
            raise ValueError("managed runtime process group is unavailable")
        payload = command.execute(
            {
                "schema_version": 1,
                "operation": "probe",
                "cwd": str(cwd),
                "chain_id": chain_id,
                "generation": generation,
                "process_group_id": process_group_id,
            }
        )
        return ProjectSafetyObservation.model_validate(payload)
    except Exception:
        return ProjectSafetyObservation(
            status=ProjectSafetyStatus.UNKNOWN,
            warnings=("project safety probe failed",),
        )


def merge_project_safety(
    activity: ActivitySnapshot,
    observation: ProjectSafetyObservation,
) -> ActivitySnapshot:
    """Combine native lifecycle state with a fail-closed project probe."""

    if (
        activity.quiescence is Quiescence.UNKNOWN
        or observation.status is ProjectSafetyStatus.UNKNOWN
    ):
        quiescence = Quiescence.UNKNOWN
    elif (
        activity.quiescence is Quiescence.BUSY
        or observation.status is ProjectSafetyStatus.BUSY
    ):
        quiescence = Quiescence.BUSY
    else:
        quiescence = Quiescence.IDLE

    critical_sections = set(activity.active_critical_section_ids)
    critical_sections.update(observation.active_critical_sections)
    if observation.status is ProjectSafetyStatus.BUSY and not critical_sections:
        critical_sections.add("project-safety-probe")

    return ActivitySnapshot(
        quiescence=quiescence,
        active_turn_ids=activity.active_turn_ids,
        active_tool_ids=activity.active_tool_ids,
        active_subagent_ids=activity.active_subagent_ids,
        active_critical_section_ids=frozenset(critical_sections),
        processed_event_count=activity.processed_event_count,
        last_event_at=activity.last_event_at,
        integrity_warnings=activity.integrity_warnings + observation.warnings,
        handoff_requested_generations=activity.handoff_requested_generations,
    )
