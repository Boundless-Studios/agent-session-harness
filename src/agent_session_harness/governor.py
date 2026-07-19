from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent_session_harness.activity import Quiescence
from agent_session_harness.config import GovernorConfig
from agent_session_harness.models import Confidence


class GovernorState(str, Enum):
    RUNNING = "running"
    WARNING = "warning"
    DRAINING = "draining"
    CHECKPOINTING = "checkpointing"
    FENCED = "fenced"
    LAUNCHING = "launching"
    AWAITING_ACK = "awaiting_ack"


class OperationResult(str, Enum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OperatorCommand(str, Enum):
    CANCEL_ROTATION = "cancel_rotation"


@dataclass(frozen=True, slots=True)
class GovernorSnapshot:
    """All observed state needed for one pure governor evaluation."""

    state: GovernorState
    context_percent: float | None
    context_confidence: Confidence
    quiescence: Quiescence
    required_capabilities_known: bool
    observe_only: bool
    checkpoint_result: OperationResult = OperationResult.UNKNOWN
    fence_result: OperationResult = OperationResult.UNKNOWN
    launch_result: OperationResult = OperationResult.UNKNOWN
    acknowledgement_result: OperationResult = OperationResult.UNKNOWN
    claim_result: OperationResult = OperationResult.UNKNOWN
    operator_command: OperatorCommand | None = None


@dataclass(frozen=True, slots=True)
class Transition:
    previous_state: GovernorState
    next_state: GovernorState
    reason: str
    blocked: bool = False

    @property
    def changed(self) -> bool:
        return self.previous_state is not self.next_state


class Governor:
    """Pure context warning and fresh-session rotation state machine."""

    def __init__(self, config: GovernorConfig) -> None:
        self._config = config

    def evaluate(self, snapshot: GovernorSnapshot) -> Transition:
        if (
            snapshot.operator_command is OperatorCommand.CANCEL_ROTATION
            and snapshot.state in {GovernorState.DRAINING, GovernorState.CHECKPOINTING}
        ):
            return self._move(
                snapshot, GovernorState.RUNNING, "operator canceled rotation"
            )

        if snapshot.state in {GovernorState.RUNNING, GovernorState.WARNING}:
            return self._evaluate_context(snapshot)
        if snapshot.state is GovernorState.DRAINING:
            return self._evaluate_draining(snapshot)
        if snapshot.state is GovernorState.CHECKPOINTING:
            return self._evaluate_checkpointing(snapshot)
        if snapshot.state is GovernorState.FENCED:
            return self._move(
                snapshot,
                GovernorState.LAUNCHING,
                "ownership fenced; launch successor",
            )
        if snapshot.state is GovernorState.LAUNCHING:
            return self._evaluate_launching(snapshot)
        return self._evaluate_acknowledgement(snapshot)

    def _evaluate_context(self, snapshot: GovernorSnapshot) -> Transition:
        if (
            snapshot.context_percent is None
            or snapshot.context_confidence is not Confidence.CONFIDENT
        ):
            return self._stay(snapshot, "context usage is not confident", blocked=True)

        percent = snapshot.context_percent
        if percent >= self._config.rotate_percent:
            if snapshot.observe_only or not snapshot.required_capabilities_known:
                return self._move(
                    snapshot,
                    GovernorState.WARNING,
                    "rotation disabled until required capabilities are known",
                    blocked=True,
                )
            return self._move(
                snapshot,
                GovernorState.DRAINING,
                "rotation threshold reached",
            )
        if percent >= self._config.warn_percent:
            return self._move(
                snapshot, GovernorState.WARNING, "warning threshold reached"
            )
        return self._move(
            snapshot, GovernorState.RUNNING, "context below warning threshold"
        )

    def _evaluate_draining(self, snapshot: GovernorSnapshot) -> Transition:
        if snapshot.observe_only or not snapshot.required_capabilities_known:
            return self._stay(
                snapshot,
                "rotation capabilities became unavailable while draining",
                blocked=True,
            )
        if snapshot.quiescence is Quiescence.UNKNOWN:
            return self._stay(
                snapshot, "activity is stale or inconsistent", blocked=True
            )
        if snapshot.quiescence is Quiescence.BUSY:
            return self._stay(snapshot, "waiting for quiescence")
        return self._move(
            snapshot,
            GovernorState.CHECKPOINTING,
            "fresh consistent quiescence verified",
        )

    def _evaluate_checkpointing(self, snapshot: GovernorSnapshot) -> Transition:
        if snapshot.observe_only or not snapshot.required_capabilities_known:
            return self._stay(
                snapshot,
                "rotation capabilities unavailable during checkpoint",
                blocked=True,
            )
        if snapshot.checkpoint_result is not OperationResult.SUCCEEDED:
            return self._wait_for_operation(
                snapshot,
                snapshot.checkpoint_result,
                "required checkpoint",
            )
        if snapshot.fence_result is not OperationResult.SUCCEEDED:
            return self._wait_for_operation(
                snapshot,
                snapshot.fence_result,
                "coordinator fence",
            )
        return self._move(
            snapshot,
            GovernorState.FENCED,
            "checkpoint verified and ownership fenced",
        )

    def _evaluate_launching(self, snapshot: GovernorSnapshot) -> Transition:
        if snapshot.launch_result is not OperationResult.SUCCEEDED:
            return self._wait_for_operation(
                snapshot,
                snapshot.launch_result,
                "successor launch",
            )
        return self._move(
            snapshot,
            GovernorState.AWAITING_ACK,
            "successor launched",
        )

    def _evaluate_acknowledgement(self, snapshot: GovernorSnapshot) -> Transition:
        if snapshot.acknowledgement_result is not OperationResult.SUCCEEDED:
            return self._wait_for_operation(
                snapshot,
                snapshot.acknowledgement_result,
                "successor acknowledgement",
            )
        if snapshot.claim_result is not OperationResult.SUCCEEDED:
            return self._wait_for_operation(
                snapshot,
                snapshot.claim_result,
                "successor coordinator claim",
            )
        return self._move(
            snapshot,
            GovernorState.RUNNING,
            "successor acknowledged and claimed ownership",
        )

    def _wait_for_operation(
        self,
        snapshot: GovernorSnapshot,
        result: OperationResult,
        operation: str,
    ) -> Transition:
        blocked = result in {OperationResult.UNKNOWN, OperationResult.FAILED}
        return self._stay(snapshot, f"{operation} is {result.value}", blocked=blocked)

    @staticmethod
    def _stay(
        snapshot: GovernorSnapshot,
        reason: str,
        *,
        blocked: bool = False,
    ) -> Transition:
        return Transition(
            previous_state=snapshot.state,
            next_state=snapshot.state,
            reason=reason,
            blocked=blocked,
        )

    @staticmethod
    def _move(
        snapshot: GovernorSnapshot,
        next_state: GovernorState,
        reason: str,
        *,
        blocked: bool = False,
    ) -> Transition:
        return Transition(
            previous_state=snapshot.state,
            next_state=next_state,
            reason=reason,
            blocked=blocked,
        )
