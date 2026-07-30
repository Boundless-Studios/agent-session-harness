from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_session_harness.process_identity import (
    ManagedResourceReference,
    ProcessEvidence,
    ProcessIdentity,
    ProcessObservation,
    ProcessPlatform,
    ProcessState,
    same_process,
)


CAPTURED_AT = datetime(2026, 7, 30, tzinfo=UTC)


def identity(**changes: object) -> ProcessIdentity:
    values: dict[str, object] = {
        "schema_version": 1,
        "platform": ProcessPlatform.LINUX,
        "pid": 42,
        "opaque_start_token": "linux:12345",
        "executable_identity": "/usr/bin/python3",
        "captured_at": CAPTURED_AT,
    }
    values.update(changes)
    return ProcessIdentity.model_validate(values)


def test_contract_accepts_additive_fields_and_round_trips_them() -> None:
    parsed = ProcessIdentity.model_validate(
        {**identity().model_dump(), "future_native_clock": "boot-7"}
    )

    assert parsed.model_dump()["future_native_clock"] == "boot-7"


@pytest.mark.parametrize(
    "model",
    [
        {
            "schema_version": 2,
            "platform": "linux",
            "pid": 42,
            "opaque_start_token": "linux:12345",
            "executable_identity": "/usr/bin/python3",
            "captured_at": CAPTURED_AT,
        },
        {
            "schema_version": 2,
            "kind": "runtime",
            "resource_key": "chain-1:0",
        },
    ],
)
def test_contract_rejects_unsupported_major_version(
    model: dict[str, object],
) -> None:
    contract = (
        ProcessIdentity if "opaque_start_token" in model else ManagedResourceReference
    )

    with pytest.raises(ValidationError):
        contract.model_validate(model)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pid", 0),
        ("opaque_start_token", ""),
        ("executable_identity", ""),
        ("captured_at", datetime(2026, 7, 30)),
    ],
)
def test_identity_rejects_untrustworthy_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        identity(**{field: value})


def test_same_process_uses_pid_platform_and_start_token_not_executable() -> None:
    original = identity()
    after_exec = identity(executable_identity="/usr/bin/codex")
    successor = identity(opaque_start_token="linux:12346")
    foreign_platform = identity(platform=ProcessPlatform.DARWIN)

    assert same_process(original, after_exec)
    assert not same_process(original, successor)
    assert not same_process(original, foreign_platform)


def test_observation_contains_typed_managed_resource_reference() -> None:
    reference = ManagedResourceReference(
        schema_version=1,
        kind="runtime",
        resource_key="chain-1:0",
    )
    evidence = ProcessEvidence(
        schema_version=1,
        source="native",
        code="native_record",
    )

    observation = ProcessObservation(
        schema_version=1,
        identity=identity(),
        state=ProcessState.RUNNING,
        managed_resource=reference,
        evidence=[evidence],
        observed_at=CAPTURED_AT,
        future_observation_id="observation-7",
    )

    assert observation.managed_resource == reference
    assert observation.evidence == [evidence]
    assert observation.model_dump()["future_observation_id"] == "observation-7"
