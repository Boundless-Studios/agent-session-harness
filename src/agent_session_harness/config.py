from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from platformdirs import user_config_path as platform_user_config_path
from pydantic import BaseModel, ConfigDict, Field, model_validator


APPLICATION_NAME = "agent-session-harness"
PROJECT_CONFIG_NAME = ".agent-session-harness.toml"
USER_CONFIG_NAME = "config.toml"


class GovernorConfig(BaseModel):
    """Thresholds and event freshness required by the governor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    warn_percent: float = Field(default=65.0, gt=0.0, le=100.0)
    rotate_percent: float = Field(default=70.0, gt=0.0, le=100.0)
    stale_event_timeout_seconds: float = Field(default=30.0, gt=0.0)

    @model_validator(mode="after")
    def require_warning_before_rotation(self) -> Self:
        if self.warn_percent >= self.rotate_percent:
            raise ValueError("warn_percent must be less than rotate_percent")
        return self


class HarnessConfig(BaseModel):
    """Typed project configuration after capability safety is applied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    governor: GovernorConfig = Field(default_factory=GovernorConfig)
    observe_only: bool = False


def default_user_config_path() -> Path:
    return Path(platform_user_config_path(APPLICATION_NAME)) / USER_CONFIG_NAME


def resolve_config_path(
    *,
    explicit_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    user_path: str | Path | None = None,
) -> Path | None:
    """Return the highest-precedence available config file."""

    if explicit_path is not None:
        explicit = Path(explicit_path).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(
                f"explicit configuration file does not exist: {explicit}"
            )
        return explicit

    project_root = Path.cwd() if project_dir is None else Path(project_dir)
    project = project_root.expanduser() / PROJECT_CONFIG_NAME
    if project.is_file():
        return project

    user = (
        default_user_config_path()
        if user_path is None
        else Path(user_path).expanduser()
    )
    return user if user.is_file() else None


def load_config(
    *,
    explicit_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    user_path: str | Path | None = None,
    required_capabilities_known: bool | None = None,
) -> HarnessConfig:
    """Load one config source and fail closed when capabilities are unknown."""

    source = resolve_config_path(
        explicit_path=explicit_path,
        project_dir=project_dir,
        user_path=user_path,
    )
    if source is None:
        config = HarnessConfig()
    else:
        with source.open("rb") as config_file:
            config = HarnessConfig.model_validate(tomllib.load(config_file))

    if config.observe_only or required_capabilities_known is not True:
        return config.model_copy(update={"observe_only": True})
    return config
