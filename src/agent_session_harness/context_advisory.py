"""Typed, repository-neutral context advisory decisions.

This module deliberately does not read transcripts, write state, or render host
instructions. Runtime adapters own observations; repositories own wording and
remediation. The harness owns the threshold and de-duplication semantics shared
by both.
"""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AdvisoryTier(StrEnum):
    """Increasing context-pressure levels exposed to host adapters."""

    QUIET = "quiet"
    CHECKPOINT = "checkpoint"
    URGENT = "urgent"


class ContextObservation(BaseModel):
    """One runtime-derived live-context observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    context_tokens: int = Field(ge=0)
    window_tokens: int = Field(gt=0)
    model: str | None = None
    measured: bool = True

    @property
    def context_fraction(self) -> float:
        return self.context_tokens / self.window_tokens


class ContextAdvisoryPolicy(BaseModel):
    """Portable pressure thresholds; host-specific instructions stay outside."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_fraction: float = 0.65
    urgent_fraction: float = 0.85

    @model_validator(mode="after")
    def validate_thresholds(self) -> ContextAdvisoryPolicy:
        values = (self.checkpoint_fraction, self.urgent_fraction)
        if not all(isfinite(value) for value in values):
            raise ValueError("context advisory thresholds must be finite")
        if not 0 < self.checkpoint_fraction < self.urgent_fraction <= 1:
            raise ValueError(
                "context advisory thresholds must satisfy "
                "0 < checkpoint_fraction < urgent_fraction <= 1"
            )
        return self


class AdvisoryEmissionState(BaseModel):
    """The last tier emitted for one session and the window used to derive it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: AdvisoryTier = AdvisoryTier.QUIET
    window_tokens: int | None = Field(default=None, gt=0)


class ContextAdvisoryDecision(BaseModel):
    """Pure decision returned to a runtime- or repository-specific renderer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tier: AdvisoryTier
    emit: bool
    context_tokens: int = Field(ge=0)
    window_tokens: int = Field(gt=0)
    headroom_tokens: int = Field(ge=0)
    context_percent: float = Field(ge=0)
    measured: bool
    model: str | None = None

    def emission_state(self) -> AdvisoryEmissionState:
        return AdvisoryEmissionState(tier=self.tier, window_tokens=self.window_tokens)


def evaluate_context_advisory(
    observation: ContextObservation,
    *,
    policy: ContextAdvisoryPolicy | None = None,
    previous: AdvisoryEmissionState | None = None,
) -> ContextAdvisoryDecision:
    """Evaluate pressure and emit only a newly crossed tier.

    A changed context window invalidates the previous tier. This matters when a
    runtime initially reports a conservative fallback and later supplies its
    authoritative long-context window.
    """

    selected_policy = policy or ContextAdvisoryPolicy()
    fraction = observation.context_fraction
    if fraction >= selected_policy.urgent_fraction:
        tier = AdvisoryTier.URGENT
    elif fraction >= selected_policy.checkpoint_fraction:
        tier = AdvisoryTier.CHECKPOINT
    else:
        tier = AdvisoryTier.QUIET

    prior_tier = AdvisoryTier.QUIET
    if previous is not None and previous.window_tokens == observation.window_tokens:
        prior_tier = previous.tier
    rank = {
        AdvisoryTier.QUIET: 0,
        AdvisoryTier.CHECKPOINT: 1,
        AdvisoryTier.URGENT: 2,
    }
    return ContextAdvisoryDecision(
        tier=tier,
        emit=rank[tier] > rank[prior_tier],
        context_tokens=observation.context_tokens,
        window_tokens=observation.window_tokens,
        headroom_tokens=max(
            observation.window_tokens - observation.context_tokens,
            0,
        ),
        context_percent=min(fraction * 100, 100),
        measured=observation.measured,
        model=observation.model,
    )
