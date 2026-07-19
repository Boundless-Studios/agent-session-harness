from __future__ import annotations

import importlib
import json
from pathlib import Path
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
