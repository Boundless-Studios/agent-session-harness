"""Durable, incarnation-fenced managed-resource registrations."""

from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .resource_guardian import ManagedResource
from .secure_files import (
    atomic_write_private_text,
    exclusive_lock,
    private_exists,
    read_private_text,
)

SchemaVersion = Literal[1]
MAX_REGISTRY_BYTES = 4 * 1_048_576
MAX_REGISTRATIONS = 1024
RegistrationId = Annotated[str, Field(min_length=16, max_length=256)]


class RegistrationConflictError(RuntimeError):
    """A stale registration incarnation attempted to mutate its successor."""


class ResourceRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = 1
    registration_id: RegistrationId
    resource: ManagedResource
    registered_at: datetime

    @field_validator("registered_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(UTC)


class _RegistryDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = 1
    registrations: Annotated[
        list[ResourceRegistration],
        Field(max_length=MAX_REGISTRATIONS),
    ]


class ResourceRegistry:
    """Private atomic registry keyed by a resource's stable logical identity."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def register(
        self,
        resource: ManagedResource,
        *,
        now: datetime | None = None,
    ) -> ResourceRegistration:
        registration = ResourceRegistration(
            registration_id=secrets.token_urlsafe(24),
            resource=resource,
            registered_at=now or datetime.now(UTC),
        )
        with exclusive_lock(self.lock_path):
            document = self._read()
            logical_key = (resource.kind, resource.resource_key)
            retained = [
                item
                for item in document.registrations
                if (item.resource.kind, item.resource.resource_key) != logical_key
            ]
            self._write(
                _RegistryDocument.model_validate(
                    {
                        **document.model_dump(),
                        "registrations": [*retained, registration],
                    }
                )
            )
        return registration

    def unregister(
        self,
        kind: str,
        resource_key: str,
        registration_id: str,
    ) -> None:
        with exclusive_lock(self.lock_path):
            document = self._read()
            logical_key = (kind, resource_key)
            current = next(
                (
                    item
                    for item in document.registrations
                    if (item.resource.kind, item.resource.resource_key) == logical_key
                ),
                None,
            )
            if current is None:
                return
            if current.registration_id != registration_id:
                raise RegistrationConflictError(
                    "stale registration cannot unregister its successor"
                )
            self._write(
                document.model_copy(
                    update={
                        "registrations": [
                            item
                            for item in document.registrations
                            if item is not current
                        ]
                    }
                )
            )

    def list(self) -> list[ResourceRegistration]:
        with exclusive_lock(self.lock_path):
            return list(self._read().registrations)

    def is_current(self, registration: ResourceRegistration) -> bool:
        logical_key = (
            registration.resource.kind,
            registration.resource.resource_key,
        )
        with exclusive_lock(self.lock_path):
            return any(
                (
                    item.resource.kind,
                    item.resource.resource_key,
                    item.registration_id,
                )
                == (*logical_key, registration.registration_id)
                for item in self._read().registrations
            )

    def _read(self) -> _RegistryDocument:
        if not private_exists(self.path):
            return _RegistryDocument(registrations=[])
        try:
            payload = json.loads(
                read_private_text(self.path, max_bytes=MAX_REGISTRY_BYTES)
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError("resource registry contains invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("resource registry is not an object")
        if payload.get("schema_version") != 1:
            raise RuntimeError("resource registry has unsupported schema version")
        try:
            return _RegistryDocument.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError("resource registry violates its contract") from exc

    def _write(self, document: _RegistryDocument) -> None:
        atomic_write_private_text(
            self.path,
            document.model_dump_json() + "\n",
        )
