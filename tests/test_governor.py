from dataclasses import replace

import pytest

from agent_session_harness.activity import Quiescence
from agent_session_harness.config import GovernorConfig
from agent_session_harness.governor import (
    Governor,
    GovernorSnapshot,
    GovernorState,
    OperationResult,
    OperatorCommand,
)
from agent_session_harness.models import Confidence


def _governor() -> Governor:
    return Governor(GovernorConfig())


def _snapshot(
    *,
    state: str,
    percent: float = 64.9,
    quiescent: bool = False,
    checkpoint: bool = False,
) -> GovernorSnapshot:
    return GovernorSnapshot(
        state=GovernorState(state),
        context_percent=percent,
        context_confidence=Confidence.CONFIDENT,
        quiescence=Quiescence.IDLE if quiescent else Quiescence.BUSY,
        required_capabilities_known=True,
        observe_only=False,
        checkpoint_result=(
            OperationResult.SUCCEEDED if checkpoint else OperationResult.UNKNOWN
        ),
        fence_result=OperationResult.SUCCEEDED,
    )


@pytest.mark.parametrize(
    ("state", "percent", "quiescent", "checkpoint", "next_state"),
    [
        ("running", 64.9, False, False, "running"),
        ("running", 65.0, False, False, "warning"),
        ("warning", 70.0, False, False, "draining"),
        ("draining", 70.0, True, False, "checkpointing"),
        ("checkpointing", 70.0, True, True, "fenced"),
    ],
)
def test_governor_transition(
    state: str,
    percent: float,
    quiescent: bool,
    checkpoint: bool,
    next_state: str,
) -> None:
    transition = _governor().evaluate(
        _snapshot(
            state=state,
            percent=percent,
            quiescent=quiescent,
            checkpoint=checkpoint,
        )
    )

    assert transition.next_state is GovernorState(next_state)


def test_large_context_jump_begins_draining_immediately() -> None:
    transition = _governor().evaluate(_snapshot(state="running", percent=70.0))

    assert transition.next_state is GovernorState.DRAINING


def test_warning_clears_when_context_falls_below_warning_threshold() -> None:
    transition = _governor().evaluate(_snapshot(state="warning", percent=64.9))

    assert transition.next_state is GovernorState.RUNNING


def test_fenced_session_launches_and_successor_acknowledges() -> None:
    governor = _governor()
    fenced = _snapshot(state="fenced", percent=70.0, quiescent=True, checkpoint=True)

    launching = governor.evaluate(fenced)
    awaiting_ack = governor.evaluate(
        replace(
            fenced,
            state=launching.next_state,
            launch_result=OperationResult.SUCCEEDED,
        )
    )
    running = governor.evaluate(
        replace(
            fenced,
            state=awaiting_ack.next_state,
            acknowledgement_result=OperationResult.SUCCEEDED,
            claim_result=OperationResult.SUCCEEDED,
        )
    )

    assert launching.next_state is GovernorState.LAUNCHING
    assert awaiting_ack.next_state is GovernorState.AWAITING_ACK
    assert running.next_state is GovernorState.RUNNING


@pytest.mark.parametrize("confidence", [Confidence.DEGRADED, Confidence.UNKNOWN])
def test_unconfident_context_cannot_change_threshold_state(
    confidence: Confidence,
) -> None:
    snapshot = replace(
        _snapshot(state="running", percent=80.0),
        context_confidence=confidence,
    )

    transition = _governor().evaluate(snapshot)

    assert transition.next_state is GovernorState.RUNNING
    assert transition.blocked is True


def test_missing_context_cannot_change_threshold_state() -> None:
    snapshot = replace(
        _snapshot(state="running", percent=80.0),
        context_percent=None,
    )

    transition = _governor().evaluate(snapshot)

    assert transition.next_state is GovernorState.RUNNING
    assert transition.blocked is True


@pytest.mark.parametrize(
    ("required_capabilities_known", "observe_only"),
    [(False, False), (True, True)],
)
def test_unsafe_or_observe_only_mode_warns_but_cannot_begin_rotation(
    required_capabilities_known: bool,
    observe_only: bool,
) -> None:
    snapshot = replace(
        _snapshot(state="running", percent=80.0),
        required_capabilities_known=required_capabilities_known,
        observe_only=observe_only,
    )

    transition = _governor().evaluate(snapshot)

    assert transition.next_state is GovernorState.WARNING
    assert transition.blocked is True


@pytest.mark.parametrize(
    ("quiescence", "blocked"),
    [(Quiescence.BUSY, False), (Quiescence.UNKNOWN, True)],
)
def test_draining_waits_for_fresh_consistent_quiescence(
    quiescence: Quiescence,
    blocked: bool,
) -> None:
    snapshot = replace(
        _snapshot(state="draining", percent=80.0),
        quiescence=quiescence,
    )

    transition = _governor().evaluate(snapshot)

    assert transition.next_state is GovernorState.DRAINING
    assert transition.blocked is blocked


@pytest.mark.parametrize(
    ("state", "field", "other", "next_state"),
    [
        ("checkpointing", "checkpoint_result", "fence_result", "checkpointing"),
        ("checkpointing", "fence_result", "checkpoint_result", "checkpointing"),
        ("launching", "launch_result", None, "launching"),
        ("awaiting_ack", "acknowledgement_result", "claim_result", "awaiting_ack"),
        ("awaiting_ack", "claim_result", "acknowledgement_result", "awaiting_ack"),
    ],
)
@pytest.mark.parametrize(
    ("result", "blocked"),
    [
        (OperationResult.UNKNOWN, True),
        (OperationResult.PENDING, False),
        (OperationResult.FAILED, True),
    ],
)
def test_external_operation_failure_or_unknown_never_advances(
    state: str,
    field: str,
    other: str | None,
    next_state: str,
    result: OperationResult,
    blocked: bool,
) -> None:
    snapshot = _snapshot(state=state, percent=80.0, quiescent=True, checkpoint=True)
    updates: dict[str, OperationResult] = {field: result}
    if other is not None:
        updates[other] = OperationResult.SUCCEEDED

    transition = _governor().evaluate(replace(snapshot, **updates))

    assert transition.next_state is GovernorState(next_state)
    assert transition.blocked is blocked


def test_lower_sample_cannot_cancel_rotation_after_draining_starts() -> None:
    transition = _governor().evaluate(
        _snapshot(state="draining", percent=10.0, quiescent=False)
    )

    assert transition.next_state is GovernorState.DRAINING


@pytest.mark.parametrize("state", ["draining", "checkpointing"])
def test_operator_can_cancel_rotation_before_fencing(state: str) -> None:
    snapshot = replace(
        _snapshot(state=state, percent=80.0, quiescent=True, checkpoint=True),
        operator_command=OperatorCommand.CANCEL_ROTATION,
    )

    transition = _governor().evaluate(snapshot)

    assert transition.next_state is GovernorState.RUNNING
    assert transition.blocked is False
