import hashlib
from dataclasses import replace
from datetime import UTC, datetime

import agent_session_harness.process_identity as process_identity_module
from agent_session_harness.process_identity import (
    ManagedResourceReference,
    NativeProcessRecord,
    NativeReadResult,
    ProcessIdentity,
    ProcessInspector,
    ProcessPlatform,
    ProcessState,
    legacy_process_fingerprint,
    observe_process_identity,
)

NOW = datetime(2026, 7, 30, tzinfo=UTC)
RUNNING = NativeProcessRecord(
    pid=42,
    parent_pid=7,
    state=ProcessState.RUNNING,
    opaque_start_token="linux:100",
    executable_identity="/usr/bin/python3",
)
PARENT = NativeProcessRecord(
    pid=7,
    parent_pid=1,
    state=ProcessState.RUNNING,
    opaque_start_token="linux:10",
    executable_identity="/sbin/launchd",
)


class FakeReader:
    platform = ProcessPlatform.LINUX

    def __init__(self, results: dict[int, NativeReadResult]) -> None:
        self.results = results
        self.read_pids: list[int] = []

    def read(self, pid: int) -> NativeReadResult:
        self.read_pids.append(pid)
        return self.results[pid]


def inspector(result: NativeReadResult) -> ProcessInspector:
    return ProcessInspector(
        reader=FakeReader(
            {
                42: result,
                7: NativeReadResult.present(PARENT),
            }
        ),
        clock=lambda: NOW,
    )


def test_legacy_fingerprint_preserves_existing_registry_value() -> None:
    identity = ProcessIdentity(
        platform=ProcessPlatform.LINUX,
        pid=42,
        opaque_start_token="linux:12345",
        executable_identity="/usr/bin/python3",
        captured_at=NOW,
    )

    assert (
        legacy_process_fingerprint(identity)
        == hashlib.sha256(b"42:linux:12345").hexdigest()
    )


def test_capture_returns_trusted_native_identity() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))

    captured = subject.capture(42)

    assert captured is not None
    assert captured.pid == 42
    assert captured.platform is ProcessPlatform.LINUX
    assert captured.opaque_start_token == "linux:100"
    assert captured.executable_identity == "/usr/bin/python3"
    assert captured.captured_at == NOW


def test_capture_does_not_invent_identity_when_inspection_is_unknown() -> None:
    subject = inspector(NativeReadResult.unknown("permission_denied"))

    assert subject.capture(42) is None


def test_public_native_observation_uses_typed_managed_resource(monkeypatch) -> None:
    reader = FakeReader(
        {
            42: NativeReadResult.present(RUNNING),
            7: NativeReadResult.present(PARENT),
        }
    )
    monkeypatch.setattr(
        process_identity_module, "native_process_reader", lambda: reader
    )
    expected = ProcessInspector(reader, clock=lambda: NOW).capture(42)
    assert expected is not None
    resource = ManagedResourceReference(kind="repository_daemon", resource_key="dash")

    observation = observe_process_identity(expected, resource)

    assert observation.state is ProcessState.RUNNING
    assert observation.managed_resource == resource


def test_pid_reuse_marks_expected_identity_missing() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[42] = NativeReadResult.present(
        replace(RUNNING, opaque_start_token="linux:101")
    )

    observed = subject.observe(expected)

    assert observed.state is ProcessState.MISSING
    assert [item.code for item in observed.evidence] == ["pid_reused"]


def test_disappeared_process_is_missing() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[42] = NativeReadResult.missing()

    observed = subject.observe(expected)

    assert observed.state is ProcessState.MISSING
    assert [item.code for item in observed.evidence] == ["process_missing"]


def test_zombie_never_counts_as_running_or_requires_executable() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[42] = NativeReadResult.present(
        replace(
            RUNNING,
            state=ProcessState.ZOMBIE,
            executable_identity=None,
        )
    )

    observed = subject.observe(expected)

    assert observed.state is ProcessState.ZOMBIE


def test_unknown_preserves_typed_managed_resource_reference() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[42] = NativeReadResult.unknown("permission_denied")
    reference = ManagedResourceReference(
        kind="runtime",
        resource_key="chain-1:0",
    )

    observed = subject.observe(expected, managed_resource=reference)

    assert observed.state is ProcessState.UNKNOWN
    assert observed.managed_resource == reference
    assert [item.code for item in observed.evidence] == ["permission_denied"]


def test_platform_mismatch_is_unknown_without_native_read() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.read_pids.clear()
    foreign = expected.model_copy(update={"platform": ProcessPlatform.DARWIN})

    observed = subject.observe(foreign)

    assert observed.state is ProcessState.UNKNOWN
    assert [item.code for item in observed.evidence] == ["platform_mismatch"]
    assert subject.reader.read_pids == []


def test_executable_change_preserves_lifetime_and_records_evidence() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[42] = NativeReadResult.present(
        replace(RUNNING, executable_identity="/usr/bin/codex")
    )

    observed = subject.observe(expected)

    assert observed.state is ProcessState.RUNNING
    assert [item.code for item in observed.evidence] == ["executable_changed"]
    assert observed.parent_identity is not None
    assert observed.parent_identity.pid == 7


def test_parent_failure_is_evidence_not_child_uncertainty() -> None:
    subject = inspector(NativeReadResult.present(RUNNING))
    expected = subject.capture(42)
    assert expected is not None
    subject.reader.results[7] = NativeReadResult.unknown("parent_denied")

    observed = subject.observe(expected)

    assert observed.state is ProcessState.RUNNING
    assert observed.parent_identity is None
    assert [item.code for item in observed.evidence] == ["parent_unavailable"]
