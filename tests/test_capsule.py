from __future__ import annotations

from hashlib import sha256
import importlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError


def _module():
    try:
        return importlib.import_module("agent_session_harness.capsule")
    except ModuleNotFoundError:
        pytest.fail("canonical handoff capsules are not implemented")


def capsule_payload(repository_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "chain_id": "chain-1",
        "predecessor_conversation_id": "conversation-7",
        "target_generation": 8,
        "task_ids": {"linear": "BOU-2195", "bead": "bou-2195"},
        "objective": "Continue the durable session handoff implementation.",
        "exact_next_action": "Run the focused checkpoint contract tests.",
        "completed_criteria": ["Capsule contract defined"],
        "remaining_criteria": ["Checkpoint adapters verified"],
        "repository_path": str(repository_path),
        "branch": "bou-2195-agent-session-harness",
        "head": "a" * 40,
        "dirty_paths": ["src/agent_session_harness/checkpoint.py"],
        "file_anchors": ["src/agent_session_harness/capsule.py"],
        "symbol_anchors": ["HandoffCapsule.canonical_bytes"],
        "test_results": {
            "pytest tests/test_capsule.py": "passed",
            "ruff check": "passed",
        },
        "decisions": ["Use canonical compact JSON."],
        "blockers": [],
        "process_summaries": {"git": "idle", "pytest": "idle"},
        "created_at": "2026-07-19T03:00:00+00:00",
    }


def test_capsule_has_the_exact_durable_handoff_fields(tmp_path) -> None:
    capsule = _module()

    expected = {
        "schema_version",
        "chain_id",
        "predecessor_conversation_id",
        "target_generation",
        "task_ids",
        "objective",
        "exact_next_action",
        "completed_criteria",
        "remaining_criteria",
        "repository_path",
        "branch",
        "head",
        "dirty_paths",
        "file_anchors",
        "symbol_anchors",
        "test_results",
        "decisions",
        "blockers",
        "process_summaries",
        "created_at",
        "fingerprint",
    }

    handoff = capsule.HandoffCapsule.model_validate(capsule_payload(tmp_path))

    assert set(capsule.HandoffCapsule.model_fields) == expected
    assert handoff.repository_path == tmp_path.resolve()
    assert len(handoff.fingerprint) == 64


def test_capsule_canonical_bytes_and_fingerprint_ignore_dict_key_order(
    tmp_path,
) -> None:
    capsule = _module()
    first_payload = capsule_payload(tmp_path)
    second_payload = dict(reversed(first_payload.items()))
    second_payload["task_ids"] = dict(
        reversed(first_payload["task_ids"].items())  # type: ignore[union-attr]
    )
    second_payload["process_summaries"] = dict(
        reversed(first_payload["process_summaries"].items())  # type: ignore[union-attr]
    )

    first = capsule.HandoffCapsule.model_validate(first_payload)
    second = capsule.HandoffCapsule.model_validate(second_payload)

    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.fingerprint == second.fingerprint
    serialized = json.loads(first.canonical_bytes())
    assert serialized["fingerprint"] == first.fingerprint
    assert first.fingerprint == sha256(first.fingerprint_payload_bytes()).hexdigest()


@pytest.mark.parametrize(
    "field_name",
    [
        "raw_prompt",
        "transcript",
        "secret",
        "environment_dump",
        "tool_input",
    ],
)
def test_capsule_rejects_private_or_unbounded_fields(tmp_path, field_name) -> None:
    capsule = _module()
    payload = capsule_payload(tmp_path)
    payload[field_name] = "must not persist"

    with pytest.raises(ValidationError):
        capsule.HandoffCapsule.model_validate(payload)

    assert field_name not in capsule.HandoffCapsule.model_fields


def test_capsule_rejects_a_supplied_mismatched_fingerprint(tmp_path) -> None:
    capsule = _module()
    payload = capsule_payload(tmp_path)
    payload["fingerprint"] = "0" * 64

    with pytest.raises(ValidationError, match="fingerprint"):
        capsule.HandoffCapsule.model_validate(payload)


def test_capsule_round_trips_canonical_json_with_verified_fingerprint(tmp_path) -> None:
    capsule = _module()
    original = capsule.HandoffCapsule.model_validate(capsule_payload(tmp_path))

    restored = capsule.HandoffCapsule.model_validate_json(original.canonical_bytes())

    assert restored == original
    assert restored.canonical_bytes() == original.canonical_bytes()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("objective", "Continue with api_key=must-not-persist"),
        ("exact_next_action", "Use password:must-not-persist"),
        ("decisions", ["Keep token=must-not-persist"]),
        ("task_ids", {"linear": "token=must-not-persist"}),
        ("dirty_paths", ["secret=must-not-persist"]),
        ("test_results", {"pytest": "secret=must-not-persist"}),
        ("process_summaries", {"worker": "credential=must-not-persist"}),
    ],
)
def test_capsule_rejects_credential_shaped_text(tmp_path, field, value) -> None:
    capsule = _module()
    payload = capsule_payload(tmp_path)
    payload[field] = value

    with pytest.raises(ValidationError, match="credential-shaped"):
        capsule.HandoffCapsule.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_ids", {f"task-{index}": "value" for index in range(17)}),
        ("dirty_paths", ["x" * 1025]),
        ("repository_path", "/" + "x" * 4096),
        ("decisions", [f"decision-{index}" for index in range(33)]),
        ("test_results", {f"test-{index}": "passed" for index in range(65)}),
        ("process_summaries", {f"worker-{index}": "idle" for index in range(17)}),
    ],
)
def test_capsule_rejects_unbounded_operational_collections(
    tmp_path, field, value
) -> None:
    capsule = _module()
    payload = capsule_payload(tmp_path)
    payload[field] = value

    with pytest.raises(ValidationError):
        capsule.HandoffCapsule.model_validate(payload)


def test_capsule_process_summaries_use_an_allowlisted_state(tmp_path) -> None:
    capsule = _module()
    payload = capsule_payload(tmp_path)
    payload["process_summaries"] = {"pytest": "printing arbitrary output"}

    with pytest.raises(ValidationError):
        capsule.HandoffCapsule.model_validate(payload)
