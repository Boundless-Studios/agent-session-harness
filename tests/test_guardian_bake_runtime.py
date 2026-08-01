from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent_session_harness.guardian_bake import (
    GuardianBakeDecision,
    GuardianBakeReport,
    GuardianHighWaterMarks,
    ObservationWindow,
    ResourceHighWaterMarks,
    UsageSnapshot,
    UsageHighWaterMarks,
)
from agent_session_harness.guardian_bake_runtime import (
    GuardianBakeConfig,
    GuardianBakeConfigStore,
    GuardianBakeMode,
    GuardianBakeRuntime,
    assess_bake_exit,
)
from agent_session_harness.guardian_bake_spool import GuardianBakeSpool

NOW = datetime(2026, 7, 31, tzinfo=UTC)
WINDOW = ObservationWindow(started_at=NOW, ends_at=NOW + timedelta(days=1))


def config(**changes) -> GuardianBakeConfig:
    values = {
        "installed": True,
        "enabled": True,
        "guardian_version": "0.1.0",
        "observation_window": WINDOW,
        "mode": GuardianBakeMode.OBSERVE_ONLY,
        "enabled_reasons": set(),
        "max_memory_bytes": 4096,
        "max_cpu_percent": 2.0,
    }
    values.update(changes)
    return GuardianBakeConfig(**values)


def report(
    *,
    heartbeat: datetime = NOW,
    memory_bytes: int = 1024,
    cpu_percent: float = 1.0,
    decisions: list[GuardianBakeDecision] | None = None,
) -> GuardianBakeReport:
    return GuardianBakeReport.build(
        guardian_version="0.1.0",
        platform="linux",
        observation_window=WINDOW,
        heartbeat_at=heartbeat,
        usage_before=UsageSnapshot(memory_bytes=memory_bytes, cpu_percent=cpu_percent),
        usage_after=UsageSnapshot(memory_bytes=memory_bytes, cpu_percent=cpu_percent),
        high_water_marks=GuardianHighWaterMarks(
            resources=ResourceHighWaterMarks(observed=1, managed=1, ambiguous=0),
            usage=UsageHighWaterMarks(
                memory_bytes=memory_bytes,
                cpu_percent=cpu_percent,
            ),
        ),
        reap_decisions=decisions or [],
        refused_decisions=[],
        errors=[],
    )


def test_disabled_or_expired_runtime_is_a_noop(tmp_path) -> None:
    spool = GuardianBakeSpool(tmp_path / "spool.json")

    assert (
        GuardianBakeRuntime(config(enabled=False), spool).record(report(), now=NOW)
        is None
    )
    assert (
        GuardianBakeRuntime(config(), spool).record(
            report(heartbeat=WINDOW.ends_at),
            now=WINDOW.ends_at,
        )
        is None
    )
    assert spool.list() == []


def test_observe_only_is_default_and_reap_requires_explicit_reasons(tmp_path) -> None:
    spool = GuardianBakeSpool(tmp_path / "spool.json")

    assert GuardianBakeRuntime(config(), spool).enabled_reasons(now=NOW) == set()
    reap = config(
        mode=GuardianBakeMode.REAP,
        enabled_reasons={"terminal_managed_child"},
    )
    assert GuardianBakeRuntime(reap, spool).enabled_reasons(now=NOW) == {
        "terminal_managed_child"
    }


def test_config_store_install_and_observe_only_rollback_preserve_spool(
    tmp_path,
) -> None:
    config_path = tmp_path / "config.json"
    spool = GuardianBakeSpool(tmp_path / "spool.json")
    spool.append(report(), now=NOW)
    store = GuardianBakeConfigStore(config_path)
    reap = config(
        mode=GuardianBakeMode.REAP,
        enabled_reasons={"terminal_managed_child"},
    )

    store.install(reap)
    store.save(
        reap.model_copy(
            update={
                "mode": GuardianBakeMode.OBSERVE_ONLY,
                "enabled_reasons": frozenset(),
            }
        )
    )
    rolled_back = store.load()
    store.uninstall()

    assert rolled_back.mode is GuardianBakeMode.OBSERVE_ONLY
    assert store.load().installed is False
    assert len(spool.pending()) == 1


def test_active_runtime_records_heartbeat_report_before_any_sink(tmp_path) -> None:
    spool = GuardianBakeSpool(tmp_path / "spool.json")

    created = GuardianBakeRuntime(config(), spool).record(report(), now=NOW)

    assert created is not None
    assert spool.pending()[0].record_id == created.record_id


def test_exit_requires_safe_reaps_reason_coverage_and_bounded_overhead() -> None:
    safe = GuardianBakeDecision(
        reason_code="terminal_managed_child",
        performed=True,
        live_resource=False,
        evidence=["process_missing"],
    )

    passed = assess_bake_exit(
        [report(decisions=[safe])],
        config(
            mode=GuardianBakeMode.REAP,
            enabled_reasons={"terminal_managed_child"},
        ),
    )
    unsafe = assess_bake_exit(
        [
            report(
                memory_bytes=8192,
                decisions=[safe, safe.model_copy(update={"live_resource": True})],
            )
        ],
        config(
            mode=GuardianBakeMode.REAP,
            enabled_reasons={"terminal_managed_child", "dead_worktree"},
        ),
    )

    assert passed.passed is True
    assert passed.failures == []
    assert unsafe.passed is False
    assert set(unsafe.failures) == {
        "live_resource_reaped",
        "enabled_reason_unvalidated:dead_worktree",
        "memory_overhead_exceeded",
    }
