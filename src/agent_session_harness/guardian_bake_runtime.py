"""Opt-in configuration, recording, and exit assessment for guardian bakes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .guardian_bake import GuardianBakeReport, ObservationWindow
from .guardian_bake_spool import GuardianBakeSpool, GuardianBakeSpoolRecord
from .secure_files import (
    atomic_write_private_text,
    exclusive_lock,
    private_exists,
    read_private_text,
)

MAX_CONFIG_BYTES = 64 * 1024


class GuardianBakeMode(StrEnum):
    OBSERVE_ONLY = "observe_only"
    REAP = "reap"


class GuardianBakeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1, le=1)
    installed: bool
    enabled: bool
    guardian_version: Annotated[str, Field(min_length=1, max_length=128)]
    observation_window: ObservationWindow
    mode: GuardianBakeMode = GuardianBakeMode.OBSERVE_ONLY
    enabled_reasons: frozenset[Annotated[str, Field(min_length=1, max_length=128)]] = (
        frozenset()
    )
    max_memory_bytes: int = Field(gt=0)
    max_cpu_percent: float = Field(gt=0)

    @model_validator(mode="after")
    def require_reap_allowlist(self) -> Self:
        if self.mode is GuardianBakeMode.REAP and not self.enabled_reasons:
            raise ValueError("reap mode requires enabled reason codes")
        if self.mode is GuardianBakeMode.OBSERVE_ONLY and self.enabled_reasons:
            raise ValueError("observe-only mode cannot enable reap reasons")
        return self

    def active_at(self, now: datetime) -> bool:
        current = now.astimezone(UTC)
        return (
            self.installed
            and self.enabled
            and self.observation_window.started_at
            <= current
            < self.observation_window.ends_at
        )


class GuardianBakeExitAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    failures: list[str]


class GuardianBakeConfigStore:
    """Persist explicit install state separately from the evidence spool."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def install(self, config: GuardianBakeConfig) -> None:
        self.save(config.model_copy(update={"installed": True}))

    def uninstall(self) -> None:
        current = self.load()
        self.save(
            current.model_copy(
                update={
                    "installed": False,
                    "enabled": False,
                    "mode": GuardianBakeMode.OBSERVE_ONLY,
                    "enabled_reasons": frozenset(),
                }
            )
        )

    def save(self, config: GuardianBakeConfig) -> None:
        encoded = config.model_dump_json() + "\n"
        if len(encoded.encode("utf-8")) > MAX_CONFIG_BYTES:
            raise ValueError("guardian bake config exceeds its byte bound")
        with exclusive_lock(self.lock_path):
            atomic_write_private_text(self.path, encoded)

    def load(self) -> GuardianBakeConfig:
        with exclusive_lock(self.lock_path):
            if not private_exists(self.path):
                raise RuntimeError("guardian bake is not installed")
            try:
                payload = json.loads(
                    read_private_text(self.path, max_bytes=MAX_CONFIG_BYTES)
                )
                return GuardianBakeConfig.model_validate(payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "guardian bake config contains invalid JSON"
                ) from exc
            except ValidationError as exc:
                raise RuntimeError(
                    "guardian bake config violates its contract"
                ) from exc


class GuardianBakeRuntime:
    """Persist active bake reports; execution remains a separate fenced boundary."""

    def __init__(self, config: GuardianBakeConfig, spool: GuardianBakeSpool):
        self.config = config
        self.spool = spool

    def enabled_reasons(self, *, now: datetime) -> set[str]:
        if not self.config.active_at(now):
            return set()
        if self.config.mode is GuardianBakeMode.OBSERVE_ONLY:
            return set()
        return set(self.config.enabled_reasons)

    def record(
        self,
        report: GuardianBakeReport,
        *,
        now: datetime,
    ) -> GuardianBakeSpoolRecord | None:
        if not self.config.active_at(now):
            return None
        if report.observation_window != self.config.observation_window:
            raise ValueError("report observation window does not match bake config")
        if report.guardian_version != self.config.guardian_version:
            raise ValueError("report guardian version does not match bake config")
        return self.spool.append(report, now=now)


def assess_bake_exit(
    reports: list[GuardianBakeReport],
    config: GuardianBakeConfig,
) -> GuardianBakeExitAssessment:
    failures: list[str] = []
    decisions = [decision for report in reports for decision in report.reap_decisions]
    if any(decision.performed and decision.live_resource for decision in decisions):
        failures.append("live_resource_reaped")
    validated_reasons = {
        decision.reason_code
        for decision in decisions
        if decision.performed and not decision.live_resource and decision.evidence
    }
    for reason in sorted(config.enabled_reasons - validated_reasons):
        failures.append(f"enabled_reason_unvalidated:{reason}")
    if any(
        report.high_water_marks.usage.memory_bytes > config.max_memory_bytes
        for report in reports
    ):
        failures.append("memory_overhead_exceeded")
    if any(
        report.high_water_marks.usage.cpu_percent > config.max_cpu_percent
        for report in reports
    ):
        failures.append("cpu_overhead_exceeded")
    if not reports:
        failures.append("no_bake_reports")
    return GuardianBakeExitAssessment(passed=not failures, failures=failures)
