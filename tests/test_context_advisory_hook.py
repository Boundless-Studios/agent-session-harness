from __future__ import annotations

import io
import json

from agent_session_harness import context_advisory_hook as hook
from agent_session_harness.context_advisory import ContextAdvisoryDecision


def invoke(monkeypatch, tmp_path, payload: object) -> tuple[int, str, dict]:
    state = tmp_path / "advisory.json"
    stdout = io.StringIO()
    monkeypatch.setenv("AGENT_SESSION_HARNESS_CONTEXT_ADVISORY_STATE", str(state))
    monkeypatch.setenv("AGENT_SESSION_HARNESS_CONTEXT_WINDOW_TOKENS", "100")
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(hook.sys, "stdout", stdout)

    result = hook.main()

    ledger = json.loads(state.read_text()) if state.exists() else {}
    return result, stdout.getvalue(), ledger


def test_malformed_or_unscoped_payload_is_a_silent_noop(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO("not-json"))
    monkeypatch.setattr(hook.sys, "stdout", io.StringIO())
    monkeypatch.setenv(
        "AGENT_SESSION_HARNESS_CONTEXT_ADVISORY_STATE",
        str(tmp_path / "advisory.json"),
    )

    assert hook.main() == 0
    assert not (tmp_path / "advisory.json").exists()


def test_crossing_threshold_emits_native_additional_context(
    monkeypatch, tmp_path
) -> None:
    result, output, ledger = invoke(
        monkeypatch,
        tmp_path,
        {
            "session_id": "session-1",
            "tool_name": "Read",
            "tool_response": {"content": "x" * 400},
        },
    )

    response = json.loads(output)
    assert result == 0
    assert response["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert "Context URGENT" in response["hookSpecificOutput"]["additionalContext"]
    assert ledger["session-1"]["warned_tier"] == 2


def test_emitted_tier_is_not_repeated(monkeypatch, tmp_path) -> None:
    payload = {
        "session_id": "session-1",
        "tool_name": "Read",
        "tool_response": {"content": "x" * 400},
    }
    invoke(monkeypatch, tmp_path, payload)

    result, output, _ledger = invoke(monkeypatch, tmp_path, payload)

    assert result == 0
    assert output == ""


def test_repository_can_inject_remediation_wording(monkeypatch, tmp_path) -> None:
    state = tmp_path / "advisory.json"
    stdout = io.StringIO()
    observed: list[ContextAdvisoryDecision] = []
    monkeypatch.setenv("AGENT_SESSION_HARNESS_CONTEXT_ADVISORY_STATE", str(state))
    monkeypatch.setenv("AGENT_SESSION_HARNESS_CONTEXT_WINDOW_TOKENS", "100")
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "session_id": "session-1",
                    "tool_response": {"content": "x" * 400},
                }
            )
        ),
    )
    monkeypatch.setattr(hook.sys, "stdout", stdout)

    def render(decision: ContextAdvisoryDecision) -> str:
        observed.append(decision)
        return "run the repository checkpoint adapter"

    assert hook.main(renderer=render) == 0
    assert observed[0].context_tokens > 0
    assert "run the repository checkpoint adapter" in stdout.getvalue()


def test_reset_removes_only_the_calling_session(monkeypatch, tmp_path) -> None:
    state = tmp_path / "advisory.json"
    state.write_text(json.dumps({"one": {"tokens": 10}, "two": {"tokens": 20}}))
    monkeypatch.setenv("AGENT_SESSION_HARNESS_CONTEXT_ADVISORY_STATE", str(state))
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO('{"session_id":"one"}'))

    assert hook._reset() == 0
    assert json.loads(state.read_text()) == {"two": {"tokens": 20}}


def test_native_1m_claude_5_family_resolves_to_1m_with_no_marker() -> None:
    # claude-opus-5 / claude-sonnet-5 / claude-fable-5 ship natively at 1M --
    # unlike pre-5 Claude models, they never carry a "[1m]"/"-1m" suffix.
    for model in ("claude-opus-5", "claude-sonnet-5", "claude-fable-5"):
        assert hook._context_window_tokens({"model": model}) == 1_000_000


def test_pre_5_claude_model_still_resolves_to_200k() -> None:
    assert (
        hook._context_window_tokens({"model": "claude-sonnet-4-5-20250929"}) == 200_000
    )


def test_explicit_1m_marker_on_native_1m_id_still_resolves_to_1m() -> None:
    assert hook._context_window_tokens({"model": "claude-opus-5[1m]"}) == 1_000_000
