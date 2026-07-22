from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from agent_session_harness.adapters.claude import ClaudeUsageReader
    from agent_session_harness.models import Confidence, Runtime
except ModuleNotFoundError:
    ClaudeUsageReader = None
    Confidence = None
    Runtime = None


FIXTURES = Path(__file__).parent / "fixtures" / "claude"


def _reader(*, window_tokens: int | None):
    assert ClaudeUsageReader is not None, "ClaudeUsageReader is not implemented"
    return ClaudeUsageReader(window_tokens=window_tokens)


def test_duplicate_content_rows_count_one_api_message() -> None:
    usage = _reader(window_tokens=200_000).read_file(
        FIXTURES / "duplicate-message.jsonl"
    )

    assert usage.runtime is Runtime.CLAUDE
    assert usage.conversation_id == "claude-1"
    assert usage.unique_messages == 1
    assert usage.cumulative_input_tokens == 120_000
    assert usage.cumulative_output_tokens == 5_000
    assert usage.context_tokens == 125_000
    assert usage.context_percent == pytest.approx(62.5)
    assert usage.confidence is Confidence.CONFIDENT
    assert usage.tool_counts == {"Bash": 1, "Read": 1}


def test_live_context_includes_input_cache_and_output(tmp_path) -> None:
    # The model is long-context so the 1M window under test is the one the
    # rollout actually names; `window_tokens` is only a fallback for
    # unrecognized identities and no longer overrides the model.
    rollout = tmp_path / "cache.jsonl"
    rollout.write_text(
        '{"type":"assistant","timestamp":"2026-07-19T03:00:00Z",'
        '"sessionId":"claude-cache","cwd":"/tmp/project","message":{'
        '"id":"msg-cache","model":"claude-opus-4-8[1m]","usage":{'
        '"input_tokens":10000,"output_tokens":10000,'
        '"cache_creation_input_tokens":5000,"cache_read_input_tokens":100000},'
        '"content":[]}}\n',
        encoding="utf-8",
    )

    usage = _reader(window_tokens=1_000_000).read_file(rollout)

    assert usage.context_tokens == (
        usage.latest_input_tokens
        + usage.latest_cache_creation_tokens
        + usage.latest_cache_read_tokens
        + usage.latest_output_tokens
    )
    assert usage.context_tokens == 125_000
    assert usage.context_percent == pytest.approx(12.5)


def test_missing_message_ids_are_deterministic_but_degraded() -> None:
    reader = _reader(window_tokens=200_000)
    first = reader.read_file(FIXTURES / "missing-message-id.jsonl")
    second = reader.read_file(FIXTURES / "missing-message-id.jsonl")

    assert first == second
    assert first.unique_messages == 2
    assert first.confidence is Confidence.DEGRADED
    assert len(first.message_keys) == 2
    assert all(key.startswith("offset:") for key in first.message_keys)


def _recovery_row(
    *,
    message_id: str | None,
    timestamp: str,
    input_tokens: int,
) -> str:
    message: dict = {
        "model": "claude-opus-4-8",
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "content": [],
    }
    if message_id is not None:
        message["id"] = message_id
    return json.dumps(
        {
            "type": "assistant",
            "timestamp": timestamp,
            "sessionId": "claude-recovered",
            "message": message,
        }
    )


def test_one_malformed_row_does_not_degrade_a_healthy_tail(tmp_path) -> None:
    """BOU-2208 latch 2: the sticky `degraded` flag never recovered.

    One unreadable row set `degraded` for the rest of the read, and the whole
    rollout is re-read on every sample, so the flag was effectively permanent.
    The supervisor discards non-confident samples entirely, so context percent
    stopped updating and rotation never fired.
    """
    rollout = tmp_path / "recovered.jsonl"
    rollout.write_text(
        "\n".join(
            [
                _recovery_row(
                    message_id="msg-1",
                    timestamp="2026-07-19T03:00:00Z",
                    input_tokens=1_000,
                ),
                '{"type":"assistant","message":{"id":"msg-2",',
                _recovery_row(
                    message_id="msg-3",
                    timestamp="2026-07-19T03:02:00Z",
                    input_tokens=150_000,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    usage = _reader(window_tokens=200_000).read_file(rollout)

    assert usage.confidence is Confidence.CONFIDENT
    assert usage.context_percent == pytest.approx(75.0)
    assert any("invalid JSON" in warning for warning in usage.warnings)


def test_malformed_row_after_the_last_usable_row_is_degraded(tmp_path) -> None:
    """A corrupt tail really does make the reported figures stale."""
    rollout = tmp_path / "stale-tail.jsonl"
    rollout.write_text(
        "\n".join(
            [
                _recovery_row(
                    message_id="msg-1",
                    timestamp="2026-07-19T03:00:00Z",
                    input_tokens=1_000,
                ),
                '{"type":"assistant","message":{"id":"msg-2",',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    usage = _reader(window_tokens=200_000).read_file(rollout)

    assert usage.confidence is Confidence.DEGRADED


def test_reader_never_returns_user_text_or_transcript_fields() -> None:
    usage = _reader(window_tokens=200_000).read_file(
        FIXTURES / "duplicate-message.jsonl"
    )
    encoded = usage.model_dump_json()

    fields = type(usage).model_fields
    assert "content" not in fields
    assert "message" not in fields
    assert "user" not in fields
    assert "tool_input" not in encoded


def test_project_discovery_reports_ambiguous_live_conversations(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    projects_root = tmp_path / "claude-projects"
    (projects_root / "one").mkdir(parents=True)
    (projects_root / "two").mkdir(parents=True)

    def row(session: str) -> str:
        return (
            json.dumps(
                {
                    "type": "assistant",
                    "timestamp": "2026-07-19T03:00:00Z",
                    "sessionId": session,
                    "cwd": str(project),
                    "message": {
                        "id": "msg",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                        "content": [],
                    },
                }
            )
            + "\n"
        )

    (projects_root / "one" / "first.jsonl").write_text(row("first"), encoding="utf-8")
    (projects_root / "two" / "second.jsonl").write_text(row("second"), encoding="utf-8")

    discovery = _reader(window_tokens=200_000).discover(
        projects_root=projects_root,
        cwd=project,
    )

    assert discovery.ambiguous is True
    assert discovery.error == "multiple Claude conversations match cwd"
    assert discovery.candidates == tuple(sorted(discovery.candidates))
    assert len(discovery.candidates) == 2


def _assistant_row(
    *, message_id: str, model: str, timestamp: str, input_tokens: int = 100_000
) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "timestamp": timestamp,
            "sessionId": "claude-window",
            "cwd": "/tmp/project",
            "message": {
                "id": message_id,
                "model": model,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "content": [],
            },
        }
    )


def test_window_is_derived_from_a_1m_model_without_an_explicit_window(tmp_path) -> None:
    """BOU-2211: a suffixed rollout model remains a valid fallback signal."""
    rollout = tmp_path / "one-million.jsonl"
    rollout.write_text(
        _assistant_row(
            message_id="msg-1m",
            model="claude-opus-4-8[1m]",
            timestamp="2026-07-19T03:00:00Z",
            input_tokens=200_000,
        )
        + "\n",
        encoding="utf-8",
    )

    usage = _reader(window_tokens=None).read_file(rollout)

    assert usage.window_tokens == 1_000_000
    assert usage.context_percent == pytest.approx(20.0)
    assert usage.confidence is Confidence.CONFIDENT


def test_explicit_window_wins_when_rollout_omits_the_1m_suffix(tmp_path) -> None:
    """BOU-2237: Claude's bare rollout model cannot identify a 1M launch."""
    rollout = tmp_path / "bare-model-from-1m-launch.jsonl"
    rollout.write_text(
        _assistant_row(
            message_id="msg-bare-1m",
            model="claude-opus-4-8",
            timestamp="2026-07-20T03:00:00Z",
            input_tokens=100_000,
        )
        + "\n",
        encoding="utf-8",
    )

    usage = _reader(window_tokens=1_000_000).read_file(rollout)

    assert usage.window_tokens == 1_000_000
    assert usage.context_percent == pytest.approx(10.0)


def test_mixed_model_rollout_uses_the_most_recent_model(tmp_path) -> None:
    """BOU-2211: model switching is routine and must not disable usage sampling.

    A `/model` switch mid-session leaves two model identities in one rollout.
    The latest assistant message is the one whose window is actually in force.
    """
    rollout = tmp_path / "mixed-model.jsonl"
    rollout.write_text(
        "\n".join(
            [
                _assistant_row(
                    message_id="msg-old",
                    model="claude-opus-4-6",
                    timestamp="2026-07-19T03:00:00Z",
                ),
                _assistant_row(
                    message_id="msg-new",
                    model="claude-opus-4-8[1m]",
                    timestamp="2026-07-19T04:00:00Z",
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    usage = _reader(window_tokens=None).read_file(rollout)

    assert usage.window_tokens == 1_000_000
    assert usage.confidence is Confidence.CONFIDENT


def test_unknown_model_uses_the_explicit_window(tmp_path) -> None:
    """An explicit window keeps an unmapped model usable rather than failing.

    Refusing to produce a sample would stop rotation entirely, which is the very
    failure BOU-2195 exists to prevent.
    """
    rollout = tmp_path / "unknown-model.jsonl"
    rollout.write_text(
        _assistant_row(
            message_id="msg-unknown",
            model="claude-not-a-real-model",
            timestamp="2026-07-19T03:00:00Z",
        )
        + "\n",
        encoding="utf-8",
    )

    usage = _reader(window_tokens=200_000).read_file(rollout)

    assert usage.window_tokens == 200_000
    assert usage.context_percent == pytest.approx(50.0)
