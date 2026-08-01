from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_session_harness.context_advisory import (
    AdvisoryEmissionState,
    AdvisoryTier,
    ContextAdvisoryPolicy,
    ContextObservation,
    evaluate_context_advisory,
)


def observation(tokens: int, window: int = 200_000) -> ContextObservation:
    return ContextObservation(
        session_id="session-1",
        context_tokens=tokens,
        window_tokens=window,
        model="claude",
    )


@pytest.mark.parametrize(
    ("tokens", "tier", "emit"),
    [
        (129_999, AdvisoryTier.QUIET, False),
        (130_000, AdvisoryTier.CHECKPOINT, True),
        (169_999, AdvisoryTier.CHECKPOINT, True),
        (170_000, AdvisoryTier.URGENT, True),
    ],
)
def test_default_policy_has_explicit_boundary_semantics(
    tokens: int,
    tier: AdvisoryTier,
    emit: bool,
) -> None:
    decision = evaluate_context_advisory(observation(tokens))

    assert decision.tier is tier
    assert decision.emit is emit


def test_same_tier_is_not_emitted_twice() -> None:
    previous = AdvisoryEmissionState(
        tier=AdvisoryTier.CHECKPOINT,
        window_tokens=200_000,
    )

    decision = evaluate_context_advisory(observation(140_000), previous=previous)

    assert decision.tier is AdvisoryTier.CHECKPOINT
    assert decision.emit is False


def test_escalation_is_emitted_after_checkpoint() -> None:
    previous = AdvisoryEmissionState(
        tier=AdvisoryTier.CHECKPOINT,
        window_tokens=200_000,
    )

    decision = evaluate_context_advisory(observation(180_000), previous=previous)

    assert decision.tier is AdvisoryTier.URGENT
    assert decision.emit is True


def test_revised_window_invalidates_the_old_emission_tier() -> None:
    previous = AdvisoryEmissionState(
        tier=AdvisoryTier.URGENT,
        window_tokens=200_000,
    )

    decision = evaluate_context_advisory(
        observation(700_000, window=1_000_000),
        previous=previous,
    )

    assert decision.tier is AdvisoryTier.CHECKPOINT
    assert decision.emit is True


def test_decision_exposes_renderer_inputs_without_host_wording() -> None:
    decision = evaluate_context_advisory(
        ContextObservation(
            session_id="session-1",
            context_tokens=220_000,
            window_tokens=200_000,
            model="claude-opus",
            measured=False,
        )
    )

    assert decision.context_percent == 100
    assert decision.headroom_tokens == 0
    assert decision.model == "claude-opus"
    assert decision.measured is False
    assert decision.emission_state() == AdvisoryEmissionState(
        tier=AdvisoryTier.URGENT,
        window_tokens=200_000,
    )


@pytest.mark.parametrize(
    ("checkpoint", "urgent"),
    [(0, 0.8), (0.8, 0.8), (0.9, 0.8), (0.7, 1.1), (float("nan"), 0.8)],
)
def test_policy_rejects_ambiguous_thresholds(
    checkpoint: float,
    urgent: float,
) -> None:
    with pytest.raises(ValidationError):
        ContextAdvisoryPolicy(
            checkpoint_fraction=checkpoint,
            urgent_fraction=urgent,
        )


def test_host_can_select_policy_without_forking_evaluation() -> None:
    decision = evaluate_context_advisory(
        observation(100_000),
        policy=ContextAdvisoryPolicy(
            checkpoint_fraction=0.4,
            urgent_fraction=0.75,
        ),
    )

    assert decision.tier is AdvisoryTier.CHECKPOINT
    assert decision.emit is True
