from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from agent_session_harness.adapters.codex import CodexUsageReader
    from agent_session_harness.models import Confidence, Runtime
except ModuleNotFoundError:
    CodexUsageReader = None
    Confidence = None
    Runtime = None


FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def _reader():
    assert CodexUsageReader is not None, "CodexUsageReader is not implemented"
    return CodexUsageReader()


def test_fork_lineage_subtracts_each_child_baseline_per_dimension() -> None:
    usage = _reader().read_lineage(
        [FIXTURES / "grandchild.jsonl", FIXTURES / "root.jsonl", FIXTURES / "child.jsonl"]
    )

    assert [session.session_id for session in usage.sessions] == [
        "root",
        "child",
        "grandchild",
    ]
    root, child, grandchild = usage.sessions
    assert root.runtime is Runtime.CODEX
    assert root.baseline_total_tokens == 0
    assert root.incremental_total_tokens == 100
    assert child.baseline_total_tokens == 100
    assert child.incremental_input_tokens == 20
    assert child.incremental_cached_input_tokens == 5
    assert child.incremental_output_tokens == 5
    assert child.incremental_reasoning_output_tokens == 0
    assert child.incremental_total_tokens == 30
    assert grandchild.baseline_total_tokens == 130
    assert grandchild.incremental_total_tokens == 20
    assert usage.incremental_total_tokens == 150
    assert usage.naive_total_tokens == 380


def test_live_context_uses_latest_turn_not_cumulative_total() -> None:
    usage = _reader().read_file(FIXTURES / "child.jsonl")

    assert usage.context_tokens == 80
    assert usage.window_tokens == 200
    assert usage.context_percent == pytest.approx(40.0)
    assert usage.final_total_tokens == 130


def test_missing_fork_baseline_is_degraded_and_never_guessed(tmp_path) -> None:
    rollout = tmp_path / "orphan.jsonl"
    rollout.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-07-19T03:05:00Z",
                        "type": "session_meta",
                        "payload": {
                            "id": "orphan",
                            "timestamp": "2026-07-19T03:05:00Z",
                            "source": {
                                "subagent": {
                                    "thread_spawn": {"parent_thread_id": "missing"}
                                }
                            },
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-07-19T03:06:00Z",
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {"total_tokens": 12},
                                "last_token_usage": {"total_tokens": 8},
                                "model_context_window": 100,
                            },
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    usage = _reader().read_lineage([rollout])

    assert usage.sessions[0].confidence is Confidence.DEGRADED
    assert usage.sessions[0].baseline_total_tokens is None
    assert usage.sessions[0].incremental_total_tokens is None
    assert usage.incremental_total_tokens is None
    assert any("baseline" in warning for warning in usage.sessions[0].warnings)


def test_lineage_cycle_is_rejected(tmp_path) -> None:
    def rollout(path: Path, session_id: str, parent_id: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "timestamp": "2026-07-19T03:00:00Z",
                    "type": "session_meta",
                    "payload": {
                        "id": session_id,
                        "timestamp": "2026-07-19T03:00:00Z",
                        "source": {
                            "subagent": {
                                "thread_spawn": {"parent_thread_id": parent_id}
                            }
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    rollout(tmp_path / "a.jsonl", "a", "b")
    rollout(tmp_path / "b.jsonl", "b", "a")

    with pytest.raises(ValueError, match="cycle"):
        _reader().read_lineage([tmp_path / "a.jsonl", tmp_path / "b.jsonl"])


def test_usage_models_never_expose_transcript_payloads() -> None:
    usage = _reader().read_lineage(
        [FIXTURES / "root.jsonl", FIXTURES / "child.jsonl"]
    )
    encoded = usage.model_dump_json()

    fields = type(usage.sessions[0]).model_fields
    assert "content" not in fields
    assert "message" not in fields
    assert "user" not in fields
    assert "tool_input" not in encoded
