from __future__ import annotations

import importlib
import json
from pathlib import Path
import shlex
import stat

import pytest


def _installer(runtime: str, path: Path):
    try:
        module = importlib.import_module("agent_session_harness.hooks.install")
    except ModuleNotFoundError:
        pytest.fail("hook installer is not implemented")
    return module.HookInstaller(runtime=runtime, path=path)


@pytest.mark.parametrize("runtime", ["claude", "codex"])
def test_install_check_upgrade_and_uninstall_preserve_unrelated_hooks(
    tmp_path, runtime
) -> None:
    path = tmp_path / "settings.json"
    original = {
        "theme": "dark",
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {"type": "command", "command": "existing pre", "timeout": 9}
                    ],
                }
            ],
            "Stop": [{"hooks": [{"type": "command", "command": "existing stop"}]}],
        },
        "custom": {"keep": [1, 2, 3]},
    }
    path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o640)
    installer = _installer(runtime, path)

    preview = installer.install(dry_run=True)
    assert preview.changed is True
    assert json.loads(path.read_text()) == original

    assert installer.install().changed is True
    assert installer.check().installed is True
    assert installer.install().changed is False
    installed = json.loads(path.read_text())
    assert installed["theme"] == "dark"
    assert installed["custom"] == original["custom"]
    assert installed["hooks"]["PreToolUse"][0] == original["hooks"]["PreToolUse"][0]
    assert installed["hooks"]["Stop"][0] == original["hooks"]["Stop"][0]
    session_start_timeout = installed["hooks"]["SessionStart"][-1]["hooks"][0][
        "timeout"
    ]
    stop_timeout = installed["hooks"]["Stop"][-1]["hooks"][0]["timeout"]
    assert session_start_timeout >= 30
    assert session_start_timeout > stop_timeout
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert path.with_suffix(path.suffix + ".agent-session-harness.bak").exists()

    assert installer.uninstall().changed is True
    assert installer.check().installed is False
    restored = json.loads(path.read_text())
    assert restored == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_invalid_manifest_refuses_mutation(tmp_path) -> None:
    path = tmp_path / "settings.json"
    original = "{ invalid\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        _installer("claude", path).install()

    assert path.read_text() == original
    assert not path.with_suffix(path.suffix + ".agent-session-harness.bak").exists()


def test_upgrade_replaces_older_owned_marker_without_duplicates(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": (
                                        "AGENT_SESSION_HARNESS_OWNED=v0 "
                                        "agent-session-harness hook --runtime codex"
                                    ),
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    installer = _installer("codex", path)
    assert installer.install().changed is True
    encoded = path.read_text(encoding="utf-8")

    assert "AGENT_SESSION_HARNESS_OWNED=v0" not in encoded
    assert encoded.count("AGENT_SESSION_HARNESS_OWNED=v1") == 10
    assert installer.install().changed is False


@pytest.mark.parametrize(
    ("event_name", "stale_timeout"),
    [("SessionStart", 5), ("Stop", 35)],
)
def test_check_rejects_stale_event_timeout(
    tmp_path: Path, event_name: str, stale_timeout: int
) -> None:
    path = tmp_path / "settings.json"
    installer = _installer("codex", path)
    installer.install()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["hooks"][event_name][-1]["hooks"][0]["timeout"] = stale_timeout
    path.write_text(json.dumps(manifest), encoding="utf-8")

    assert installer.check().installed is False


def test_installer_can_pin_an_absolute_harness_entrypoint(tmp_path) -> None:
    path = tmp_path / "settings.json"
    executable = tmp_path / "bin with spaces" / "agent-session-harness"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)

    module = importlib.import_module("agent_session_harness.hooks.install")
    installer = module.HookInstaller(
        runtime="codex",
        path=path,
        harness_command=executable,
    )
    installer.install()

    encoded = path.read_text(encoding="utf-8")
    assert (
        f"AGENT_SESSION_HARNESS_OWNED=v1 {shlex.quote(str(executable))} "
        "hook --runtime codex"
    ) in encoded
