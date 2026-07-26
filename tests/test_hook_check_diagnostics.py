"""`hooks check` must say WHY ownership does not verify.

`_is_installed` compares each event's group for exact equality, so half a dozen
unrelated conditions all collapse to `installed: false`. That is not enough to
act on: in gaia, a response-budget guard appended into the harness's own group
for PostToolUse and PreCompact broke every fresh agent-ops install, and the
entire diagnostic was `{"changed":false,"installed":false}` — root-causing it
meant reading the installer's source in site-packages.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path


def _module():
    return importlib.import_module("agent_session_harness.hooks.install")


def _installer(path: Path, runtime: str = "claude"):
    return _module().HookInstaller(runtime=runtime, path=path)


def _installed_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "settings.json"
    path.write_text("{}\n", encoding="utf-8")
    _installer(path).install()
    return path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Healthy installs stay quiet
# ---------------------------------------------------------------------------


def test_a_healthy_install_reports_no_problems(tmp_path) -> None:
    path = _installed_manifest(tmp_path)

    result = _installer(path).check()

    assert result.installed is True
    assert result.problems == ()


# ---------------------------------------------------------------------------
# The case that actually happened
# ---------------------------------------------------------------------------


def test_a_foreign_hook_in_the_owned_group_is_named(tmp_path) -> None:
    path = _installed_manifest(tmp_path)
    manifest = _load(path)
    owned_group = next(
        group
        for group in manifest["hooks"]["PostToolUse"]
        if any(
            "AGENT_SESSION_HARNESS_OWNED" in h.get("command", "")
            for h in group["hooks"]
        )
    )
    owned_group["hooks"].insert(
        0,
        {
            "type": "command",
            "command": "python3 response-budget-guard.py",
            "timeout": 10,
        },
    )
    _save(path, manifest)

    result = _installer(path).check()

    assert result.installed is False
    problems = {problem.event: problem for problem in result.problems}
    assert set(problems) == {"PostToolUse"}, "only the broken event should be reported"

    problem = problems["PostToolUse"]
    assert problem.reason == "owned hook shares its group"
    # The operator needs both halves: what intruded, and what to do about it.
    assert "response-budget-guard.py" in problem.detail
    assert "own group" in problem.detail


def test_the_rendered_problem_leads_with_the_event(tmp_path) -> None:
    path = _installed_manifest(tmp_path)
    manifest = _load(path)
    manifest["hooks"]["PreCompact"][0]["hooks"].append(
        {"type": "command", "command": "other", "timeout": 5}
    )
    _save(path, manifest)

    rendered = _installer(path).check().problems[0].render()

    assert rendered.startswith("PreCompact: ")


# ---------------------------------------------------------------------------
# The other ways it can fail — each must be distinguishable
# ---------------------------------------------------------------------------


def test_a_missing_event_is_distinguished_from_a_broken_one(tmp_path) -> None:
    path = _installed_manifest(tmp_path)
    manifest = _load(path)
    del manifest["hooks"]["Stop"]
    _save(path, manifest)

    problems = {p.event: p.reason for p in _installer(path).check().problems}

    assert problems == {"Stop": "event missing"}


def test_timeout_drift_is_named_as_such(tmp_path) -> None:
    path = _installed_manifest(tmp_path)
    manifest = _load(path)
    manifest["hooks"]["Stop"][0]["hooks"][0]["timeout"] = 999
    _save(path, manifest)

    problems = {p.event: p for p in _installer(path).check().problems}

    assert problems["Stop"].reason == "timeout drift"
    assert "999" in problems["Stop"].detail


def test_command_drift_is_named_as_such(tmp_path) -> None:
    path = _installed_manifest(tmp_path)
    manifest = _load(path)
    manifest["hooks"]["Stop"][0]["hooks"][0]["command"] = (
        "AGENT_SESSION_HARNESS_OWNED=v1 something-else"
    )
    _save(path, manifest)

    problems = {p.event: p.reason for p in _installer(path).check().problems}

    assert problems["Stop"] == "command drift"


def test_a_never_installed_manifest_reports_every_event(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{}\n", encoding="utf-8")

    result = _installer(path).check()

    assert result.installed is False
    assert {p.event for p in result.problems} == {"*"}
    assert result.problems[0].reason == "no hook manifest"


def test_a_manifest_with_other_hooks_but_none_owned_names_each_event(tmp_path) -> None:
    path = tmp_path / "settings.json"
    _save(
        path,
        {
            "hooks": {
                "Stop": [
                    {"matcher": "*", "hooks": [{"type": "command", "command": "x"}]}
                ]
            }
        },
    )

    result = _installer(path).check()

    problems = {p.event: p.reason for p in result.problems}
    assert problems["Stop"] == "not installed"
    assert problems["SessionStart"] == "event missing"
    assert len(problems) == len(_module().HOOK_EVENTS)


# ---------------------------------------------------------------------------
# Silence is the bug — `installed: false` must never come back explanation-free
# ---------------------------------------------------------------------------


def test_a_stale_owned_entry_is_not_silently_accepted(tmp_path) -> None:
    """PR #23 review: every event correct, plus one stale owned entry.

    `_is_installed` counts owned entries globally, so the extra one fails the
    check — while a per-event exact-match scan sees nothing wrong and returns
    no problems. That reproduces the exact silence this diagnosis exists to end.
    """
    path = _installed_manifest(tmp_path)
    manifest = _load(path)
    manifest["hooks"]["Stop"].append(
        {
            "matcher": "*",
            "hooks": [
                {
                    "type": "command",
                    "command": "AGENT_SESSION_HARNESS_OWNED=v1 /old/path/harness hook",
                    "timeout": 5,
                }
            ],
        }
    )
    _save(path, manifest)

    result = _installer(path).check()

    assert result.installed is False
    assert result.problems, "a failing check must never report zero problems"
    problems = {p.event: p for p in result.problems}
    assert problems["Stop"].reason == "stale owned entry"


def test_an_owned_entry_on_an_unknown_event_is_reported(tmp_path) -> None:
    path = _installed_manifest(tmp_path)
    manifest = _load(path)
    manifest["hooks"]["NotAnEvent"] = [
        {
            "matcher": "*",
            "hooks": [
                {"type": "command", "command": "AGENT_SESSION_HARNESS_OWNED=v1 x"}
            ],
        }
    ]
    _save(path, manifest)

    result = _installer(path).check()

    assert result.installed is False
    problems = {p.event: p.reason for p in result.problems}
    assert problems["NotAnEvent"] == "owned entry on an unknown event"


def test_every_failing_check_explains_itself(tmp_path) -> None:
    """The invariant behind all of the above, asserted directly."""
    path = _installed_manifest(tmp_path)
    manifest = _load(path)
    manifest["hooks"]["Stop"].append(
        {
            "matcher": "*",
            "hooks": [
                {"type": "command", "command": "AGENT_SESSION_HARNESS_OWNED=v1 stale"}
            ],
        }
    )
    _save(path, manifest)

    result = _installer(path).check()

    assert (result.installed is False) == bool(result.problems)


def test_a_group_with_two_faults_reports_both(tmp_path) -> None:
    """PR #23 review: fixing the named fault must not reveal an unnamed one.

    A group can hold a foreign hook AND a drifted owned entry. Reporting only
    the foreign hook sends the operator to fix it, re-run, and fail again on
    drift that was never mentioned.
    """
    path = _installed_manifest(tmp_path)
    manifest = _load(path)
    group = manifest["hooks"]["Stop"][0]
    group["hooks"][0]["timeout"] = 999
    group["hooks"].append(
        {"type": "command", "command": "python3 intruder.py", "timeout": 10}
    )
    _save(path, manifest)

    problem = {p.event: p for p in _installer(path).check().problems}["Stop"]

    assert "owned hook shares its group" in problem.reason
    assert "timeout drift" in problem.reason
    assert "intruder.py" in problem.detail
    assert "999" in problem.detail


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _break_post_tool_use(tmp_path) -> Path:
    path = _installed_manifest(tmp_path)
    manifest = _load(path)
    manifest["hooks"]["PostToolUse"][0]["hooks"].append(
        {"type": "command", "command": "python3 intruder.py", "timeout": 10}
    )
    _save(path, manifest)
    return path


def test_the_json_payload_carries_the_diagnosis(tmp_path, capsys) -> None:
    """This is the surface the installer prints — it must be self-explaining."""
    from agent_session_harness.cli import main

    path = _break_post_tool_use(tmp_path)

    exit_code = main(
        ["hooks", "check", "--runtime", "claude", "--path", str(path), "--json"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    payload = json.loads(captured.out)
    assert payload["installed"] is False
    assert payload["problems"][0]["event"] == "PostToolUse"
    assert "intruder.py" in payload["problems"][0]["detail"]
    # JSON on stdout, stderr empty: this CLI's existing contract.
    assert captured.err == ""


def test_human_mode_puts_the_diagnosis_on_stderr(tmp_path, capsys) -> None:
    from agent_session_harness.cli import main

    path = _break_post_tool_use(tmp_path)

    exit_code = main(["hooks", "check", "--runtime", "claude", "--path", str(path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "PostToolUse" in captured.err
    assert "intruder.py" in captured.err


def test_human_mode_says_it_once(tmp_path, capsys) -> None:
    """PR #23 review: `problems` in the human payload made `_emit` print the
    same diagnoses again as a raw JSON line, on top of the readable form."""
    from agent_session_harness.cli import main

    path = _break_post_tool_use(tmp_path)

    main(["hooks", "check", "--runtime", "claude", "--path", str(path)])
    captured = capsys.readouterr()

    assert captured.out.count("intruder.py") + captured.err.count("intruder.py") == 1
    assert "problems" not in captured.out


def test_a_passing_cli_check_stays_quiet(tmp_path, capsys) -> None:
    from agent_session_harness.cli import main

    path = _installed_manifest(tmp_path)

    exit_code = main(
        ["hooks", "check", "--runtime", "claude", "--path", str(path), "--json"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "problems" not in json.loads(captured.out)
    assert captured.err == ""
