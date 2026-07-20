from __future__ import annotations

import pytest

from agent_session_harness.adapters import claude, discovery


def test_iter_files_is_sorted_and_recursive(tmp_path) -> None:
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "second.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "first.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("", encoding="utf-8")

    found = discovery.iter_files(tmp_path)

    assert found == tuple(sorted(found))
    assert {path.name for path in found} == {"first.jsonl", "second.jsonl"}


def test_iter_files_returns_nothing_for_a_missing_root(tmp_path) -> None:
    assert discovery.iter_files(tmp_path / "absent") == ()


def test_iter_files_fails_closed_past_the_entry_bound(tmp_path) -> None:
    for index in range(3):
        (tmp_path / f"rollout-{index}.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="entry limit"):
        discovery.iter_files(tmp_path, max_entries=2)


def test_select_conversation_rollout_requires_exactly_one_match(tmp_path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "conversation-1.jsonl").write_text("", encoding="utf-8")

    assert discovery.select_conversation_rollout(
        (first, second), conversation_id="conversation-1"
    ) == (first / "conversation-1.jsonl")

    (second / "conversation-1.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or ambiguous"):
        discovery.select_conversation_rollout(
            (first, second), conversation_id="conversation-1"
        )


def test_select_conversation_rollout_supports_embedded_identities(tmp_path) -> None:
    (tmp_path / "rollout-conversation-1.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="missing or ambiguous"):
        discovery.select_conversation_rollout(
            (tmp_path,), conversation_id="conversation-1"
        )

    assert discovery.select_conversation_rollout(
        (tmp_path,), conversation_id="conversation-1", substring_match=True
    ) == (tmp_path / "rollout-conversation-1.jsonl")


def test_claude_discovery_uses_the_shared_enumeration(tmp_path, monkeypatch) -> None:
    calls: list[object] = []
    real = discovery.iter_files

    def spy(root, *args, **kwargs):
        calls.append(root)
        return real(root, *args, **kwargs)

    monkeypatch.setattr(claude, "iter_files", spy)
    projects = tmp_path / "projects"
    projects.mkdir()

    result = claude.ClaudeUsageReader(window_tokens=200_000).discover(
        projects_root=projects, cwd=tmp_path
    )

    assert calls == [projects]
    assert result.candidates == ()
