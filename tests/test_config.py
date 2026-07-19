from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_session_harness.config import (
    GovernorConfig,
    load_config,
    resolve_config_path,
)


def _write_config(
    path: Path,
    *,
    warn_percent: float,
    rotate_percent: float,
    stale_event_timeout_seconds: float = 30.0,
    observe_only: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"observe_only = {str(observe_only).lower()}",
                "[governor]",
                f"warn_percent = {warn_percent}",
                f"rotate_percent = {rotate_percent}",
                f"stale_event_timeout_seconds = {stale_event_timeout_seconds}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_resolve_config_path_prefers_explicit_then_project_then_user(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "explicit.toml"
    project_dir = tmp_path / "project"
    project = project_dir / ".agent-session-harness.toml"
    user = tmp_path / "user" / "config.toml"
    for path in (explicit, project, user):
        _write_config(path, warn_percent=60.0, rotate_percent=70.0)

    assert (
        resolve_config_path(
            explicit_path=explicit,
            project_dir=project_dir,
            user_path=user,
        )
        == explicit
    )
    assert resolve_config_path(project_dir=project_dir, user_path=user) == project

    project.unlink()
    assert resolve_config_path(project_dir=project_dir, user_path=user) == user


def test_explicit_config_path_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="explicit configuration"):
        resolve_config_path(
            explicit_path=tmp_path / "missing.toml",
            project_dir=tmp_path,
            user_path=tmp_path / "user.toml",
        )


def test_load_config_uses_safe_defaults_when_no_file_exists(tmp_path: Path) -> None:
    config = load_config(
        project_dir=tmp_path,
        user_path=tmp_path / "missing.toml",
    )

    assert config.governor.warn_percent == 65.0
    assert config.governor.rotate_percent == 70.0
    assert config.governor.stale_event_timeout_seconds == 30.0
    assert config.observe_only is True


def test_known_capabilities_enable_managed_mode_unless_observe_only_is_forced(
    tmp_path: Path,
) -> None:
    managed = load_config(
        project_dir=tmp_path,
        user_path=tmp_path / "missing.toml",
        required_capabilities_known=True,
    )
    forced_path = tmp_path / "forced.toml"
    _write_config(
        forced_path,
        warn_percent=65.0,
        rotate_percent=70.0,
        observe_only=True,
    )
    forced = load_config(
        explicit_path=forced_path,
        project_dir=tmp_path,
        user_path=tmp_path / "missing.toml",
        required_capabilities_known=True,
    )

    assert managed.observe_only is False
    assert forced.observe_only is True


def test_unknown_capabilities_force_observe_only(tmp_path: Path) -> None:
    config_path = tmp_path / "managed.toml"
    _write_config(
        config_path,
        warn_percent=50.0,
        rotate_percent=75.0,
        observe_only=False,
    )

    config = load_config(
        explicit_path=config_path,
        project_dir=tmp_path,
        user_path=tmp_path / "missing.toml",
        required_capabilities_known=None,
    )

    assert config.observe_only is True


@pytest.mark.parametrize(
    ("warn_percent", "rotate_percent"),
    [
        (0.0, 70.0),
        (-1.0, 70.0),
        (65.0, 0.0),
        (65.0, 101.0),
        (101.0, 101.0),
        (65.0, 65.0),
        (75.0, 70.0),
    ],
)
def test_governor_config_rejects_invalid_thresholds(
    warn_percent: float,
    rotate_percent: float,
) -> None:
    with pytest.raises(ValidationError):
        GovernorConfig(
            warn_percent=warn_percent,
            rotate_percent=rotate_percent,
        )


def test_governor_config_rejects_non_positive_stale_event_timeout() -> None:
    with pytest.raises(ValidationError):
        GovernorConfig(stale_event_timeout_seconds=0.0)
