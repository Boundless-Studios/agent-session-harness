"""Bounded rollout-file enumeration shared by every runtime adapter.

Discovery lives here exactly once. Runtime adapters differ in how they select
a candidate (by working directory, by conversation identity) but never in how
they enumerate, bound, or order the files they select from.
"""

from __future__ import annotations

import os
from pathlib import Path


MAX_DISCOVERY_ENTRIES = 20_000

ROLLOUT_SUFFIX = "*.jsonl"
LIFECYCLE_SUFFIX = "*.lifecycle"


def iter_files(
    root: str | os.PathLike[str],
    pattern: str = ROLLOUT_SUFFIX,
    *,
    max_entries: int = MAX_DISCOVERY_ENTRIES,
) -> tuple[Path, ...]:
    """Return every matching regular file under ``root``, sorted and bounded."""

    if max_entries <= 0:
        raise ValueError("discovery entry bound must be positive")
    base = Path(root).expanduser()
    if not base.is_dir():
        return ()
    found: list[Path] = []
    for path in sorted(base.rglob(pattern)):
        if len(found) >= max_entries:
            raise ValueError("search root exceeds entry limit")
        found.append(path)
    return tuple(found)


def select_conversation_rollout(
    roots: tuple[str | os.PathLike[str], ...],
    *,
    conversation_id: str,
    substring_match: bool = False,
) -> Path:
    """Return the single rollout owned by one conversation, or fail closed.

    ``substring_match`` supports runtimes such as Codex whose rollout file
    names embed, rather than equal, the conversation identity.
    """

    if not conversation_id:
        raise ValueError("conversation identity is required")
    candidates: list[Path] = []
    for root in roots:
        for path in iter_files(root):
            stem = path.stem
            matched = (
                conversation_id in stem if substring_match else stem == conversation_id
            )
            if matched:
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("rollout is missing or ambiguous")
    return candidates[0]
