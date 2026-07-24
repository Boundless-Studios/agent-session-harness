from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_macos_ci_exercises_process_identity_and_interactive_pty() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "macos-runtime:" in workflow
    job = workflow.split("macos-runtime:", maxsplit=1)[1]
    assert "runs-on: macos-15" in job
    assert 'python-version: "3.13"' in job
    assert "tests/test_interactive_pty.py" in job
    assert "test_public_process_group_probe_uses_the_live_group_session" in job
