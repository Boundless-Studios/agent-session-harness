"""Claude PostToolUse adapter for repository-neutral context advisories.

After each tool call, works out how full the context window is and, as it fills,
emits an ``additionalContext`` nudge telling the agent to CHECKPOINT its work
and keep going.

It never tells an agent to stop. Rotation belongs to agent-session-harness,
which measures context properly, waits for quiescence, and hands off via a
verified durable capsule. This hook only makes sure work is checkpointed
before that happens, and tells the human that the conversation is long enough
to be worth restarting. An agent that stops mid-task on a context warning
strands in-flight work for no reason -- that is the incident this hook caused
and now guards against.

Occupancy is MEASURED, not estimated, whenever the transcript allows it: every
assistant turn records ``usage`` (``input_tokens`` + ``cache_read_input_tokens``
+ ``cache_creation_input_tokens``), which is exactly what is resident in the
window. Only when no usage block is available does this fall back to the legacy
estimate -- ``bytes / 4`` of each tool response, accumulated in a per-session
ledger.

That fallback is a poor proxy and must never be trusted over a measurement: it
only ever grows (it is cleared on PreCompact, which a long 1M session never
reaches) and tool output is only a fraction of what occupies a window. Trusting
it caused the incident this hook now guards against -- see
``_promote_window_to_fit`` and the regression test of the same name.

Thresholds scale with the ACTIVE MODEL'S CONTEXT WINDOW, not a fixed number:

  - CHECKPOINT at 65% of the window
  - CRITICAL at 85% of the window

Why the window and not the 48K output limit the original spec cited: what this
hook accumulates is *tool results*, which are input tokens. They consume context,
they are not part of the model's output. A qwen3 running a 128K window and an
Opus running 1M have wildly different headroom for the same pile of tool results,
so one hardcoded number either cries wolf on the big model or fires too late on
the small one. Anchoring to the window makes the warning mean the same thing
everywhere: "you are N% of the way through what this model can hold."

Model resolution order (first hit wins):
  1. ``AGENT_SESSION_HARNESS_CONTEXT_WINDOW_TOKENS`` -- explicit numeric override
  2. ``model`` field on the hook payload
  3. last ``"model"`` recorded in the session transcript (``transcript_path``)
  4. ``ANTHROPIC_MODEL`` / ``CLAUDE_MODEL`` env
  5. ``DEFAULT_CONTEXT_TOKENS`` fallback

Claude Code writes the bare model id to the transcript and drops the ``[1m]``
suffix, so a 1M session resolves to 200K by step 3. Two things repair that:

  - the CONFIGURED model spec (``ANTHROPIC_MODEL``/``CLAUDE_MODEL``, then
    ``settings.local.json`` -> ``settings.json`` -> ``~/.claude/settings.json``)
    still carries the suffix (``"model": "opus[1m]"``). When it names the same
    model family as the transcript did, its window wins.
  - failing that, the window is widened to fit measured occupancy: if 435K is
    demonstrably resident, the window is provably not 200K.

Occupancy alone is not enough, which is why the configured spec was added. It
can only repair a window that residency has already DISPROVED -- above 200K --
whereas the false nudges fire at 130K and 170K. Every 1M session was warned
twice, and told at 173K that it had 27K of headroom left, while 830K was in fact
free. That is the incident this hook exists to prevent, reappearing one band
lower than the fix for it reached.

Each tier fires at most once per session so a session past a threshold does not
emit an identical nudge on every subsequent tool call. Invoked with ``--reset``
(registered on PreCompact) it clears the session's ledger entry, because
compaction is exactly the recovery the warning was asking for.

State file: .agent-session-harness/context-advisory.json
  -- ``{session_id: {tokens, calls, warned_tier, last_tool, last_ts,
       context_window, observed_context}}``

Advisory only: never blocks, never modifies context. Any error is swallowed
so telemetry cannot wedge a turn.

Repositories may inject their own renderer. They should keep tracker names,
project scripts, and local remediation policy in that renderer rather than in
this reusable observation and de-duplication adapter.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .context_advisory import (
    AdvisoryEmissionState,
    AdvisoryTier,
    ContextAdvisoryDecision,
    ContextAdvisoryPolicy,
    ContextObservation,
    evaluate_context_advisory,
)

# ---------------------------------------------------------------------------
# Thresholds -- fractions of the active model's context window
# ---------------------------------------------------------------------------
# These MUST track agent-session-harness, which owns rotation
# (.claude/skills/agent-context-usage/SKILL.md: "Warning begins at 65%; safe
# drain/rotation begins at 70%"). Two systems watching one window with different
# numbers is how an agent gets told to wrap up by one while the other says it is
# fine -- this hook nudges, the harness rotates.
CHECKPOINT_FRACTION = 0.65  # matches the harness warning threshold
URGENT_FRACTION = 0.85      # harness rotation has not fired; something is stuck

# Used only when the model cannot be resolved at all. 200K is the common
# Claude window and is the safe middle ground: it does not cry wolf the way a
# 48K assumption did, nor stay silent until a small model has already blown up.
DEFAULT_CONTEXT_TOKENS = 200_000

# Context windows by model-id substring, longest/most-specific match first.
# Substring matching keeps this robust to date stamps ("claude-haiku-4-5-20251001").
#
# It is NOT sufficient for window variants. Claude Code writes the plain model id
# ("claude-opus-5") to the transcript and drops the "[1m]" suffix, so a 1M session
# matches the generic "claude" rule and gets a 200K window. That understatement is
# what _promote_window_to_fit() below repairs from measured occupancy.
CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("[1m]", 1_000_000),        # explicit 1M-window variants
    ("-1m", 1_000_000),
    ("deepseek-v4", 1_000_000),
    ("qwen3", 128_000),
    ("qwen", 128_000),
    ("gpt-5", 400_000),
    ("gemini", 1_000_000),
    ("claude", 200_000),        # default Claude window
)

# The suffixes that name a window variant rather than a model. These are the
# leading CONTEXT_WINDOWS entries, split out because they are also matched
# against the CONFIGURED model spec -- see _configured_model_spec().
EXPLICIT_WINDOW_MARKERS: tuple[tuple[str, int], ...] = (
    ("[1m]", 1_000_000),
    ("-1m", 1_000_000),
)

# Settings files consulted for the configured model spec, most specific first.
# Mirrors Claude Code's own precedence (local project > project > user).
SETTINGS_RELATIVE_PATHS: tuple[str, ...] = (
    "settings.local.json",
    "settings.json",
)

# Model families used to check that a configured spec and the id recorded in the
# transcript describe the SAME model. "opus[1m]" must lend its window to
# "claude-opus-5" and to nothing else -- a session switched to Sonnet with
# /model mid-flight has to fall back to the 200K default.
MODEL_FAMILIES: tuple[str, ...] = ("opus", "sonnet", "haiku", "fable")

# Bytes-to-tokens ratio (Claude Code approximation)
TOKENS_PER_BYTE = 0.25

# Windows a session could plausibly be running with, ascending. Used to repair a
# detected window that measured occupancy has already disproved.
KNOWN_WINDOW_TIERS: tuple[int, ...] = (128_000, 200_000, 400_000, 1_000_000)

# Transcript tail sizes tried in order when looking for a usage record. A single
# JSONL record can be larger than any fixed window, so this escalates rather
# than going blind on exactly the oversized records worth measuring.
TAIL_STEPS: tuple[int, ...] = (200_000, 1_000_000, 4_000_000, 16_000_000)


def _state_path() -> Path:
    """Return the context advisory state path for the active project."""
    configured = os.environ.get("AGENT_SESSION_HARNESS_CONTEXT_ADVISORY_STATE", "")
    if configured:
        return Path(configured)
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    state_root = Path(project_dir) / ".agent-session-harness"
    return state_root / "context-advisory.json"


def _load_ledger(state_path: Path) -> dict:
    """Load accumulated budget ledger, creating if absent."""
    try:
        if state_path.exists():
            with open(state_path, encoding="utf-8") as fh:
                return json.load(fh)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_ledger(state_path: Path, ledger: dict) -> None:
    """Persist accumulated budget ledger."""
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(ledger, fh)
    except OSError:
        pass  # advisory; failure to write is non-fatal


def _estimate_response_tokens(tool_name: str, tool_response: dict | None) -> int:
    """Estimate the token cost of a tool response.

    For MCP observability tools and Bash (which can produce large output),
    measures actual JSON-serialized byte size of the response payload.

    Returns estimated tokens (bytes * TOKENS_PER_BYTE).
    """
    if not isinstance(tool_response, dict):
        # Non-dict responses (simple values, errors) are negligible
        return 0

    # Serialize the response to measure bytes
    try:
        payload = json.dumps(tool_response, default=str)
    except (TypeError, ValueError):
        return 0

    response_bytes = len(payload.encode("utf-8", errors="replace"))
    return int(response_bytes * TOKENS_PER_BYTE)


def _read_tail(path: Path, nbytes: int) -> tuple[str, bool]:
    """Return (decoded tail of at most nbytes, whether it reached file start)."""
    with open(path, "rb") as fh:
        try:
            fh.seek(-nbytes, os.SEEK_END)
            from_start = fh.tell() == 0
        except OSError:
            fh.seek(0)  # file shorter than the requested window
            from_start = True
        return fh.read().decode("utf-8", errors="replace"), from_start


def _transcript_tail(transcript_path: str) -> str:
    """Return the first-step tail of a JSONL transcript, or '' when unreadable."""
    try:
        path = Path(transcript_path)
        if not path.is_file():
            return ""
        return _read_tail(path, TAIL_STEPS[0])[0]
    except OSError:
        return ""


def _model_from_transcript(transcript_path: str) -> str:
    """Return the last model id recorded in the session transcript."""
    tail = _transcript_tail(transcript_path)
    matches = re.findall(r'"model"\s*:\s*"([^"]+)"', tail)
    return matches[-1] if matches else ""


def _explicit_window(model_spec: str) -> int:
    """Return the window a spec names outright, or 0 when it names none."""
    lowered = (model_spec or "").lower()
    for needle, window in EXPLICIT_WINDOW_MARKERS:
        if needle in lowered:
            return window
    return 0


def _model_families(model_spec: str) -> frozenset[str]:
    """Return the model families named by a spec ("opus[1m]" -> {"opus"})."""
    lowered = (model_spec or "").lower()
    return frozenset(name for name in MODEL_FAMILIES if name in lowered)


def _same_family(configured: str, resolved: str) -> bool:
    """True when both specs name at least one model family in common.

    An empty intersection is a REFUSAL, not a maybe: a configured "opus[1m]"
    must not widen the window of a session that /model switched to Sonnet, and
    it must not widen an unidentifiable model either.
    """
    return bool(_model_families(configured) & _model_families(resolved))


def _configured_model_spec() -> str:
    """Return the model spec this session was configured with, if any.

    This exists for exactly one reason: Claude Code writes the BARE model id to
    the transcript ("claude-opus-5") and drops the "[1m]" suffix, so the window
    variant is unrecoverable from the transcript. The configured spec keeps it
    ("opus[1m]"), and is the only surviving record of which variant is running.
    """
    for env_key in ("ANTHROPIC_MODEL", "CLAUDE_MODEL"):
        value = os.environ.get(env_key, "").strip()
        if value:
            return value

    project_root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()) / ".claude"
    candidates = [project_root / name for name in SETTINGS_RELATIVE_PATHS]
    candidates.append(Path.home() / ".claude" / "settings.json")

    for path in candidates:
        try:
            with open(path, encoding="utf-8") as fh:
                settings = json.load(fh)
        except (OSError, json.JSONDecodeError, ValueError):
            continue  # absent or malformed settings are simply no signal
        if not isinstance(settings, dict):
            continue
        model = settings.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    return ""


def _latest_usage_record(tail: str, *, drop_partial_first: bool) -> tuple[int, str]:
    """Newest complete record carrying usage: (token total, model id).

    The model is taken from the SAME record as the usage. Reading it from a
    separate, narrower scan is how a stripped Claude session ends up measured
    correctly but classified on the generic ladder.
    """
    lines = tail.splitlines()
    if drop_partial_first and lines:
        # The seek landed mid-record; that leading fragment is not parseable
        # JSON and must not be mistaken for a complete entry.
        lines = lines[1:]

    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        message = entry.get("message")
        usage = message.get("usage") if isinstance(message, dict) else None
        if not isinstance(usage, dict):
            continue
        total = 0
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                total += int(value)
        if total > 0:
            model = message.get("model")
            return total, model if isinstance(model, str) else ""
    return 0, ""


def _observed_context(data: dict) -> tuple[int, str]:
    """Tokens genuinely resident in the context window, per the transcript.

    Claude Code records true occupancy on every assistant turn. That is the
    number this guard should reason about; the accumulated tool-result estimate
    is a poor proxy because it only ever grows (it is cleared on PreCompact,
    which a long 1M session never reaches) and because tool output is only a
    fraction of what occupies the window.

    The tail is widened until a complete usage record is found. A single record
    can exceed any fixed window -- and an oversized record is precisely the
    context-heavy case worth measuring -- so a fixed tail would go blind exactly
    when it matters.

    Returns (0, "") when no usage block exists at all, leaving the estimate in
    charge. The model id comes from the same record as the usage so that window
    classification and occupancy can never disagree about which turn they mean.
    """
    transcript_path = data.get("transcript_path")
    if not isinstance(transcript_path, str) or not transcript_path:
        return 0, ""

    try:
        path = Path(transcript_path)
        if not path.is_file():
            return 0, ""
    except OSError:
        return 0, ""

    for nbytes in TAIL_STEPS:
        try:
            tail, from_start = _read_tail(path, nbytes)
        except OSError:
            return 0, ""
        total, model = _latest_usage_record(tail, drop_partial_first=not from_start)
        if total > 0:
            return total, model
        if from_start:
            break  # already read the whole file; there is nothing further back
    return 0, ""


def _candidate_windows(model: str) -> tuple[int, ...]:
    """Windows this model could plausibly have, ascending.

    Promotion must stay on the model's own ladder. Claude ships 200K and 1M and
    there is no 400K Claude to stop at on the way up: promoting a bare
    ``claude-opus-5`` at 340K into the GPT-shaped 400K tier reports a fake 85%
    and latches the urgent tier, which then suppresses the real 85% at 850K.
    """
    lowered = (model or "").lower()
    if "[1m]" in lowered or "-1m" in lowered:
        return (1_000_000,)
    if "claude" in lowered:
        return (200_000, 1_000_000)
    if "gemini" in lowered:
        return (1_000_000,)
    if "gpt-5" in lowered:
        return (400_000,)
    if "qwen" in lowered:
        return (128_000,)
    if "deepseek-v4" in lowered:
        return (1_000_000,)
    return KNOWN_WINDOW_TIERS


def _promote_window_to_fit(
    window_tokens: int,
    observed_tokens: int,
    candidates: tuple[int, ...] = KNOWN_WINDOW_TIERS,
) -> int:
    """Widen a detected window that measured occupancy has already disproved.

    If 435K is demonstrably resident, the window is not 200K — whatever the
    model id claimed. Snap up to the smallest window the model actually offers
    that fits, falling back to the generic ladder for unrecognised models.
    """
    if observed_tokens <= window_tokens:
        return window_tokens
    for tier in candidates:
        if tier >= observed_tokens:
            return tier
    for tier in KNOWN_WINDOW_TIERS:
        if tier >= observed_tokens:
            return tier
    return observed_tokens


def _resolve_model(data: dict, transcript_model: str = "") -> str:
    """Best-effort identify the model driving this session.

    ``transcript_model`` is the id read off the same record the usage came from;
    it beats the cheap regex tail because that tail can miss the record entirely
    when an oversized entry follows it.
    """
    direct = data.get("model")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    if transcript_model.strip():
        return transcript_model.strip()

    transcript_path = data.get("transcript_path")
    if isinstance(transcript_path, str) and transcript_path:
        from_transcript = _model_from_transcript(transcript_path)
        if from_transcript:
            return from_transcript

    for env_key in ("ANTHROPIC_MODEL", "CLAUDE_MODEL"):
        value = os.environ.get(env_key, "").strip()
        if value:
            return value

    return ""


def _context_window_tokens(
    data: dict, observed_tokens: int = 0, transcript_model: str = ""
) -> int:
    """Return the active model's context window in tokens.

    An explicit AGENT_SESSION_HARNESS_CONTEXT_WINDOW_TOKENS override always wins, so a
    session on an unrecognised model can still get sane thresholds without a
    code change.
    """
    override = os.environ.get(
        "AGENT_SESSION_HARNESS_CONTEXT_WINDOW_TOKENS", ""
    ).strip()
    if override:
        try:
            parsed = int(override)
            if parsed > 0:
                return parsed
        except ValueError:
            pass  # malformed override -> fall through to model detection

    model = _resolve_model(data, transcript_model).lower()
    detected = DEFAULT_CONTEXT_TOKENS
    if model:
        for needle, window in CONTEXT_WINDOWS:
            if needle in model:
                detected = window
                break

    # Recover a window variant the transcript stripped. _promote_window_to_fit()
    # below repairs an understated window only once occupancy DISPROVES it, and
    # for a 1M Claude session that proof needs >200K resident -- while the false
    # nudges fire at 130K (65%) and 170K (85%). The whole 130K-200K band was
    # unrescuable, which is where sessions were told they had 27K of headroom
    # left with 830K actually free. The configured spec closes that band.
    if not _explicit_window(model):
        configured = _configured_model_spec()
        marked = _explicit_window(configured)
        if marked > detected and _same_family(configured, model):
            detected = marked

    return _promote_window_to_fit(detected, observed_tokens, _candidate_windows(model))


def _thresholds(window_tokens: int) -> tuple[int, int]:
    """Return (checkpoint_tokens, urgent_tokens) for a context window."""
    return (
        int(window_tokens * CHECKPOINT_FRACTION),
        int(window_tokens * URGENT_FRACTION),
    )


def _tier_for(tokens: int, window_tokens: int) -> int:
    """Return the alert tier for a context occupancy.

    0 = quiet, 1 = checkpoint (>= 65% of window), 2 = urgent (>= 85%).
    """
    checkpoint_tokens, urgent_tokens = _thresholds(window_tokens)
    if tokens >= urgent_tokens:
        return 2
    if tokens >= checkpoint_tokens:
        return 1
    return 0


def _session_id_from(data: dict) -> str:
    return str(data.get("session_id") or os.environ.get("CLAUDE_SESSION_ID") or "")


def _reset() -> int:
    """Clear the calling session's ledger entry (registered on PreCompact).

    Compaction drops the accumulated tool results from context, so the running
    total no longer reflects what Claude is carrying.
    """
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    session_id = _session_id_from(data)
    if not session_id:
        return 0

    state_path = _state_path()
    ledger = _load_ledger(state_path)
    if session_id in ledger:
        del ledger[session_id]
        _save_ledger(state_path, ledger)
    return 0


def render_context_advisory(decision: ContextAdvisoryDecision) -> str:
    """Render a safe repository-neutral advisory."""
    model_note = f" [{decision.model}]" if decision.model else ""
    source = "Context in use" if decision.measured else "Estimated context in use"
    tier = decision.tier.value.upper()
    return (
        f"[Context {tier}]{model_note} {source}: "
        f"{decision.context_tokens:,} tokens "
        f"({int(decision.context_percent)}% of the "
        f"{decision.window_tokens:,} token context window). "
        f"~{decision.headroom_tokens:,} tokens of headroom remain.\n"
        "Checkpoint the current unit durably and keep working; do not abandon "
        "in-flight work. Managed rotation remains the supervisor's responsibility."
    )


def main(
    *,
    policy: ContextAdvisoryPolicy | None = None,
    renderer: Callable[[ContextAdvisoryDecision], str] = render_context_advisory,
) -> int:
    """Process a single PostToolUse event and update the session budget ledger."""
    # Read hook payload from stdin
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    if not isinstance(data, dict):
        return 0

    session_id = _session_id_from(data)

    if not session_id:
        return 0  # No session context -- skip tracking

    tool_response = data.get("tool_response")
    if tool_response is None:
        return 0

    # Estimate token cost of this response. A negligible one is NOT a reason to
    # skip the check: measured occupancy does not depend on the current response
    # being large, and bailing here left a session at 90% unexamined whenever a
    # tool happened to return `{}`.
    tokens = _estimate_response_tokens(
        str(data.get("tool_name", "")), tool_response
    )

    # Update ledger
    state_path = _state_path()
    ledger = _load_ledger(state_path)

    entry = ledger.get(session_id, {"tokens": 0, "calls": 0})
    if not isinstance(entry, dict):
        entry = {"tokens": 0, "calls": 0}

    entry["tokens"] = entry.get("tokens", 0) + tokens
    entry["calls"] = entry.get("calls", 0) + 1
    entry["last_tool"] = str(data.get("tool_name", ""))[:60]
    entry["last_ts"] = datetime.now(UTC).isoformat()

    # Fire each tier at most once per session; a session parked above 40K should
    # not repeat the same nudge after every tool call.
    warned_tier = entry.get("warned_tier", 0)
    if not isinstance(warned_tier, int):
        warned_tier = 0

    # Prefer what the transcript measured; fall back to the running estimate only
    # when no usage block is available (early in a session, or a foreign runtime).
    #
    # The newest usage snapshot predates THIS tool result -- PostToolUse fires
    # before any assistant turn has recorded it -- so the measurement is always
    # one result stale. Add the current response, or the very result that
    # crosses the threshold sails through unremarked.
    observed_tokens, transcript_model = _observed_context(data)
    window_tokens = _context_window_tokens(data, observed_tokens, transcript_model)
    occupancy = observed_tokens + tokens if observed_tokens else entry["tokens"]

    # A tier is only meaningful against the window it was computed with. If the
    # window has since been revised, the recorded tier is stale and must not gag
    # a genuine crossing of the new threshold. The reset must be PERSISTED, not
    # just local: when the post-promotion occupancy is too low to emit, nothing
    # else writes warned_tier, the next call sees an unchanged window, and the
    # stale tier survives to suppress the real crossing.
    previous_window = entry.get("context_window")
    if isinstance(previous_window, int) and previous_window != window_tokens:
        warned_tier = 0
        entry["warned_tier"] = 0

    entry["context_window"] = window_tokens
    entry["observed_context"] = observed_tokens
    previous = AdvisoryEmissionState(
        tier={
            0: AdvisoryTier.QUIET,
            1: AdvisoryTier.CHECKPOINT,
            2: AdvisoryTier.URGENT,
        }.get(warned_tier, AdvisoryTier.QUIET),
        window_tokens=previous_window if isinstance(previous_window, int) else None,
    )
    decision = evaluate_context_advisory(
        ContextObservation(
            session_id=session_id,
            context_tokens=occupancy,
            window_tokens=window_tokens,
            model=_resolve_model(data, transcript_model) or None,
            measured=bool(observed_tokens),
        ),
        policy=policy,
        previous=previous,
    )
    tier = {
        AdvisoryTier.QUIET: 0,
        AdvisoryTier.CHECKPOINT: 1,
        AdvisoryTier.URGENT: 2,
    }[decision.tier]
    should_warn = decision.emit
    if should_warn:
        entry["warned_tier"] = tier

    ledger[session_id] = entry
    _save_ledger(state_path, ledger)

    if should_warn:
        message = renderer(decision)
        output = json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": message,
            },
        })
        print(output)
        sys.stdout.flush()

    return 0


def cli_main() -> int:
    """Run the fail-open advisory command."""
    try:
        if "--reset" in sys.argv[1:]:
            return _reset()
        return main()
    except Exception:
        return 0  # Advisory telemetry cannot wedge a runtime turn.


if __name__ == "__main__":
    sys.exit(cli_main())
