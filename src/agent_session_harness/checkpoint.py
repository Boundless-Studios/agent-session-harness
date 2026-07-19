"""Required checkpoint verification and best-effort mirror orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent_session_harness.adapters.command import (
    AdapterOperation,
    AdapterRequest,
    AdapterResponse,
    CheckpointAdapter,
    sanitize_error,
)
from agent_session_harness.capsule import HandoffCapsule
from agent_session_harness.outbox import MirrorOutbox, ReplayResult


@dataclass(frozen=True)
class AdapterAttempt:
    adapter: str
    operation: AdapterOperation
    response: AdapterResponse
    enqueue_error: str | None = None


@dataclass(frozen=True)
class CheckpointResult:
    verified: bool
    fingerprint: str
    required_attempts: tuple[AdapterAttempt, ...]
    mirror_attempts: tuple[AdapterAttempt, ...]


class CheckpointManager:
    """Write and independently read back every required checkpoint store."""

    def __init__(
        self,
        *,
        required_adapters: Sequence[CheckpointAdapter],
        mirror_adapters: Sequence[CheckpointAdapter],
        outbox: MirrorOutbox,
    ) -> None:
        if not required_adapters:
            raise ValueError("at least one required checkpoint adapter is required")
        names = [adapter.name for adapter in (*required_adapters, *mirror_adapters)]
        if len(names) != len(set(names)):
            raise ValueError("checkpoint adapter names must be unique")
        self.required_adapters = tuple(required_adapters)
        self.mirror_adapters = tuple(mirror_adapters)
        self.outbox = outbox

    def checkpoint(
        self,
        capsule: HandoffCapsule,
        *,
        idempotency_key: str,
    ) -> CheckpointResult:
        required_attempts: list[AdapterAttempt] = []
        writes: list[tuple[CheckpointAdapter, AdapterResponse]] = []
        for adapter in self.required_adapters:
            request = self._request(capsule, AdapterOperation.WRITE, idempotency_key)
            response = adapter.execute(request)
            writes.append((adapter, response))
            required_attempts.append(
                AdapterAttempt(adapter.name, AdapterOperation.WRITE, response)
            )

        reads: list[tuple[CheckpointAdapter, AdapterResponse]] = []
        for adapter in self.required_adapters:
            request = self._request(capsule, AdapterOperation.READ, idempotency_key)
            response = adapter.execute(request)
            reads.append((adapter, response))
            required_attempts.append(
                AdapterAttempt(adapter.name, AdapterOperation.READ, response)
            )

        expected = capsule.fingerprint
        required_verified = all(
            self._matches(response, expected)
            for _adapter, response in (*writes, *reads)
        )

        mirror_attempts: list[AdapterAttempt] = []
        for adapter in self.mirror_adapters:
            request = self._request(capsule, AdapterOperation.WRITE, idempotency_key)
            response = self._execute_mirror(adapter, request)
            enqueue_error = None
            if not self._matches(response, expected):
                enqueue_error = self._enqueue_mirror(adapter.name, request)
            mirror_attempts.append(
                AdapterAttempt(
                    adapter.name,
                    AdapterOperation.WRITE,
                    response,
                    enqueue_error,
                )
            )

        return CheckpointResult(
            verified=required_verified,
            fingerprint=expected,
            required_attempts=tuple(required_attempts),
            mirror_attempts=tuple(mirror_attempts),
        )

    def acknowledge(
        self,
        capsule: HandoffCapsule,
        *,
        idempotency_key: str,
    ) -> CheckpointResult:
        """Durably acknowledge a verified successor in every configured store."""

        expected = capsule.fingerprint
        required_attempts: list[AdapterAttempt] = []
        for adapter in self.required_adapters:
            request = self._request(
                capsule,
                AdapterOperation.ACKNOWLEDGE,
                idempotency_key,
            )
            response = adapter.execute(request)
            required_attempts.append(
                AdapterAttempt(adapter.name, AdapterOperation.ACKNOWLEDGE, response)
            )

        mirror_attempts: list[AdapterAttempt] = []
        for adapter in self.mirror_adapters:
            request = self._request(
                capsule,
                AdapterOperation.ACKNOWLEDGE,
                idempotency_key,
            )
            response = self._execute_mirror(adapter, request)
            enqueue_error = None
            if not self._matches(response, expected):
                enqueue_error = self._enqueue_mirror(adapter.name, request)
            mirror_attempts.append(
                AdapterAttempt(
                    adapter.name,
                    AdapterOperation.ACKNOWLEDGE,
                    response,
                    enqueue_error,
                )
            )

        return CheckpointResult(
            verified=all(
                self._matches(attempt.response, expected)
                for attempt in required_attempts
            ),
            fingerprint=expected,
            required_attempts=tuple(required_attempts),
            mirror_attempts=tuple(mirror_attempts),
        )

    def replay_mirrors(self, *, max_attempts: int = 100) -> ReplayResult:
        """Retry durable mirror work without weakening required-store safety."""

        return self.outbox.replay(
            {adapter.name: adapter for adapter in self.mirror_adapters},
            max_attempts=max_attempts,
        )

    @staticmethod
    def _execute_mirror(
        adapter: CheckpointAdapter,
        request: AdapterRequest,
    ) -> AdapterResponse:
        try:
            return adapter.execute(request)
        except Exception as exc:
            return AdapterResponse(
                ok=False,
                fingerprint=None,
                retryable=True,
                error=sanitize_error(str(exc)),
            )

    def _enqueue_mirror(self, adapter: str, request: AdapterRequest) -> str | None:
        try:
            self.outbox.enqueue(adapter, request)
        except Exception as exc:
            # Mirror persistence is deliberately best-effort. Required stores remain
            # authoritative even when the local retry queue itself is unavailable,
            # but the failure remains observable on the associated adapter attempt.
            return sanitize_error(f"mirror retry enqueue failed: {exc}")
        return None

    @staticmethod
    def _request(
        capsule: HandoffCapsule,
        operation: AdapterOperation,
        idempotency_key: str,
    ) -> AdapterRequest:
        return AdapterRequest(
            schema_version=1,
            operation=operation,
            idempotency_key=idempotency_key,
            capsule=capsule,
        )

    @staticmethod
    def _matches(response: AdapterResponse, expected: str) -> bool:
        return response.ok and response.fingerprint == expected
