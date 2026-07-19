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


def _reader(*, window_tokens: int):
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
    rollout = tmp_path / "cache.jsonl"
    rollout.write_text(
        '{"type":"assistant","timestamp":"2026-07-19T03:00:00Z",'
        '"sessionId":"claude-cache","cwd":"/tmp/project","message":{'
        '"id":"msg-cache","model":"claude-opus-4-6","usage":{'
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
