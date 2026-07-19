"""Required checkpoint verification and best-effort mirror orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent_session_harness.adapters.command import (
    AdapterOperation,
    AdapterRequest,
    AdapterResponse,
    CheckpointAdapter,
)
from agent_session_harness.capsule import HandoffCapsule
from agent_session_harness.outbox import MirrorOutbox


@dataclass(frozen=True)
class AdapterAttempt:
    adapter: str
    operation: AdapterOperation
    response: AdapterResponse


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
            response = adapter.execute(request)
            mirror_attempts.append(
                AdapterAttempt(adapter.name, AdapterOperation.WRITE, response)
            )
            if not self._matches(response, expected):
                self.outbox.enqueue(adapter.name, request)

        return CheckpointResult(
            verified=required_verified,
            fingerprint=expected,
            required_attempts=tuple(required_attempts),
            mirror_attempts=tuple(mirror_attempts),
        )

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
