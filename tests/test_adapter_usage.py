from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_session_harness import cli
from agent_session_harness.adapters import usage


FIXTURES = Path(__file__).parent / "fixtures" / "adapters"
UNKNOWN = {
    "conversation_id": "unresolved",
    "context_percent": 0.0,
    "confidence": "unknown",
    "context_tokens": None,
    "window_tokens": None,
    "cumulative_tokens": None,
}


def _usage_request() -> dict[str, object]:
    return json.loads((FIXTURES / "usage-request-v1.json").read_text(encoding="utf-8"))


def _write_ledger(tmp_path: Path, *, runtime: str, conversation_id: str) -> Path:
    root = tmp_path / "worktree"
    root.mkdir(exist_ok=True)
    template = (FIXTURES / "lifecycle-v1.jsonl").read_text(encoding="utf-8")
    template = template.replace("__WORKTREE__", str(root))
    template = template.replace('"runtime":"claude"', f'"runtime":"{runtime}"')
    template = template.replace("claude-session-1", conversation_id)
    ledger = tmp_path / "session.lifecycle"
    ledger.write_text(template, encoding="utf-8")
    return ledger


def _copy_rollout(tmp_path: Path, fixture: str, relative: str) -> Path:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((FIXTURES / fixture).read_text(encoding="utf-8"), encoding="utf-8")
    return path


class _ClaudeReader:
    staged_text = ""
    observed_window_tokens: int | None = None

    def __init__(self, *, window_tokens: int):
        type(self).observed_window_tokens = window_tokens

    def read_file(self, path: Path):
        type(self).staged_text = Path(path).read_text(encoding="utf-8")
        return SimpleNamespace(
            conversation_id="claude-session-1",
            context_percent=0.8,
            confidence=SimpleNamespace(value="confident"),
            context_tokens=1_600,
            window_tokens=200_000,
            cumulative_total_tokens=1_600,
        )


class _CodexReader:
    staged_text = ""

    def read_file(self, path: Path):
        type(self).staged_text = Path(path).read_text(encoding="utf-8")
        return SimpleNamespace(
            session_id="codex-session-1",
            context_percent=0.5125,
            confidence=SimpleNamespace(value="confident"),
            context_tokens=1_025,
            window_tokens=200_000,
            incremental_total_tokens=2_450,
        )


def test_usage_maps_exact_claude_conversation_without_staging_private_text(
    tmp_path,
) -> None:
    ledger = _write_ledger(
        tmp_path, runtime="claude", conversation_id="claude-session-1"
    )
    root = tmp_path / "claude-projects"
    _copy_rollout(root, "claude-rollout-v1.jsonl", "project/claude-session-1.jsonl")

    result = usage.sample_usage(
        _usage_request(),
        ledger_paths=(ledger,),
        cwd=tmp_path / "worktree",
        claude_roots=(root,),
        codex_roots=(),
        claude_reader_factory=_ClaudeReader,
        codex_reader_factory=_CodexReader,
    )

    assert result == {
        "conversation_id": "claude-session-1",
        "context_percent": 0.8,
        "confidence": "confident",
        "context_tokens": 1_600,
        "window_tokens": 200_000,
        "cumulative_tokens": 1_600,
    }
    assert "private text" not in _ClaudeReader.staged_text
    assert "private answer" not in _ClaudeReader.staged_text
    assert '"input"' not in _ClaudeReader.staged_text
    assert '"usage"' in _ClaudeReader.staged_text


def test_usage_takes_the_window_from_the_model_the_rollout_names(tmp_path) -> None:
    # BOU-2211: the rollout is authoritative, and a caller-supplied window is a
    # fallback for unrecognized identities only. A long-context session must
    # not be measured against the base window.
    ledger = _write_ledger(
        tmp_path, runtime="claude", conversation_id="claude-session-1"
    )
    root = tmp_path / "claude-projects"
    rollout = _copy_rollout(
        root, "claude-rollout-v1.jsonl", "project/claude-session-1.jsonl"
    )
    rollout.write_text(
        rollout.read_text(encoding="utf-8").replace(
            "claude-opus-4-6", "claude-opus-4-8[1m]"
        ),
        encoding="utf-8",
    )

    usage.sample_usage(
        _usage_request(),
        ledger_paths=(ledger,),
        cwd=tmp_path / "worktree",
        claude_roots=(root,),
        codex_roots=(),
        claude_fallback_window_tokens=200_000,
        claude_reader_factory=_ClaudeReader,
        codex_reader_factory=_CodexReader,
    )

    assert _ClaudeReader.observed_window_tokens == 1_000_000


def test_usage_explicit_window_wins_over_bare_rollout_model(tmp_path) -> None:
    """BOU-2237: launch-time context selection is authoritative when present."""
    ledger = _write_ledger(
        tmp_path, runtime="claude", conversation_id="claude-session-1"
    )
    root = tmp_path / "claude-projects"
    _copy_rollout(root, "claude-rollout-v1.jsonl", "project/claude-session-1.jsonl")

    usage.sample_usage(
        _usage_request(),
        ledger_paths=(ledger,),
        cwd=tmp_path / "worktree",
        claude_roots=(root,),
        codex_roots=(),
        claude_window_tokens=1_000_000,
        claude_reader_factory=_ClaudeReader,
        codex_reader_factory=_CodexReader,
    )

    assert _ClaudeReader.observed_window_tokens == 1_000_000


def test_usage_maps_codex_incremental_burn_and_token_only_rollout(tmp_path) -> None:
    ledger = _write_ledger(tmp_path, runtime="codex", conversation_id="codex-session-1")
    root = tmp_path / "codex-sessions"
    _copy_rollout(
        root, "codex-rollout-v1.jsonl", "2026/07/19/rollout-codex-session-1.jsonl"
    )

    result = usage.sample_usage(
        _usage_request(),
        ledger_paths=(ledger,),
        cwd=tmp_path / "worktree",
        claude_roots=(),
        codex_roots=(root,),
        claude_reader_factory=_ClaudeReader,
        codex_reader_factory=_CodexReader,
    )

    assert result["conversation_id"] == "codex-session-1"
    assert result["cumulative_tokens"] == 2_450
    assert result["context_tokens"] == 1_025
    assert "private_note" not in _CodexReader.staged_text
    assert '"private"' not in _CodexReader.staged_text
    assert '"token_count"' in _CodexReader.staged_text


def test_usage_returns_unknown_for_an_unmapped_claude_model(tmp_path) -> None:
    ledger = _write_ledger(
        tmp_path, runtime="claude", conversation_id="claude-session-1"
    )
    root = tmp_path / "claude-projects"
    rollout = _copy_rollout(
        root, "claude-rollout-v1.jsonl", "project/claude-session-1.jsonl"
    )
    rollout.write_text(
        rollout.read_text(encoding="utf-8").replace(
            "claude-opus-4-6", "claude-unknown-window"
        ),
        encoding="utf-8",
    )

    result = usage.sample_usage(
        _usage_request(),
        ledger_paths=(ledger,),
        cwd=tmp_path / "worktree",
        claude_roots=(root,),
        codex_roots=(),
        claude_reader_factory=_ClaudeReader,
        codex_reader_factory=_CodexReader,
    )

    assert result["confidence"] == "unknown"
    assert result["window_tokens"] is None


def test_usage_returns_unknown_when_latest_claude_usage_is_malformed(tmp_path) -> None:
    ledger = _write_ledger(
        tmp_path, runtime="claude", conversation_id="claude-session-1"
    )
    root = tmp_path / "claude-projects"
    rollout = _copy_rollout(
        root, "claude-rollout-v1.jsonl", "project/claude-session-1.jsonl"
    )
    malformed = {
        "type": "assistant",
        "sessionId": "claude-session-1",
        "timestamp": "2026-07-19T08:01:02Z",
        "message": {
            "id": "msg-2",
            "model": "claude-opus-4-6",
            "usage": {
                "input_tokens": 1_500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            "content": [],
        },
    }
    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(malformed) + "\n")

    result = usage.sample_usage(
        _usage_request(),
        ledger_paths=(ledger,),
        cwd=tmp_path / "worktree",
        claude_roots=(root,),
        codex_roots=(),
        claude_reader_factory=_ClaudeReader,
        codex_reader_factory=_CodexReader,
    )

    assert result["confidence"] == "unknown"
    assert result["context_tokens"] is None


@pytest.mark.parametrize(
    "failure",
    [
        "wrong-owner",
        "wrong-generation",
        "wrong-cwd",
        "ambiguous-rollout",
        "symlink",
        "oversize",
    ],
)
def test_usage_fails_closed_without_private_diagnostics(tmp_path, failure) -> None:
    request = _usage_request()
    ledger = _write_ledger(
        tmp_path, runtime="claude", conversation_id="claude-session-1"
    )
    root = tmp_path / "claude-projects"
    rollout = _copy_rollout(
        root, "claude-rollout-v1.jsonl", "project/claude-session-1.jsonl"
    )
    other = tmp_path / "other-worktree"
    if failure == "wrong-owner":
        request["process"]["pid"] = 9999
    elif failure == "wrong-generation":
        request["process"]["registry_key"] = "chain-1:1"
    elif failure == "wrong-cwd":
        other.mkdir()
    elif failure == "ambiguous-rollout":
        _copy_rollout(root, "claude-rollout-v1.jsonl", "other/claude-session-1.jsonl")
    elif failure == "symlink":
        target = tmp_path / "private-rollout.jsonl"
        target.write_text("secret=do-not-leak", encoding="utf-8")
        rollout.unlink()
        rollout.symlink_to(target)

    sampled_cwd = other if failure == "wrong-cwd" else tmp_path / "worktree"
    max_rollout_bytes = 10 if failure == "oversize" else usage.MAX_ROLLOUT_BYTES

    result = usage.sample_usage(
        request,
        ledger_paths=(ledger,),
        cwd=sampled_cwd,
        claude_roots=(root,),
        codex_roots=(),
        max_rollout_bytes=max_rollout_bytes,
        claude_reader_factory=_ClaudeReader,
        codex_reader_factory=_CodexReader,
    )

    assert result == UNKNOWN
    assert "do-not-leak" not in json.dumps(result)


def test_usage_cli_emits_bounded_unknown_for_an_invalid_request(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(
        "sys.stdin", io.TextIOWrapper(io.BytesIO(b'{"schema_version":2}'))
    )

    assert cli.main(["adapter", "usage"]) == 0

    assert json.loads(capsys.readouterr().out) == UNKNOWN
