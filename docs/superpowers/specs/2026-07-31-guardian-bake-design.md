# Guardian Bake Design

## Purpose

BOU-2714 adds an opt-in operational bake around the conservative guardian. The bake records durable, redacted evidence on macOS and Linux before any upload, summarizes repeated states, and can report only to the existing Linear issue BOU-2704. Observe-only remains the default, and changing execution mode never deletes the local report spool.

## Boundaries

The implementation is one reviewable PR with four isolated units:

1. `guardian_bake.py` defines the frozen report, observation-window, usage, decision, error, and configuration contracts. A report's deduplication key is derived from canonical redacted content rather than supplied by callers.
2. `guardian_bake_spool.py` stores reports in a private, locked JSON document. It persists before upload, bounds file size and record count, tracks delivery separately, and compacts consecutive identical states into a repeat count.
3. `adapters/guardian_bake_linear.py` accepts an injected GraphQL transport but hard-codes BOU-2704 as its only issue target. Its public API has no create-issue operation. Successful delivery is acknowledged in the spool only after the comment mutation succeeds.
4. `guardian_bake_runtime.py` validates opt-in configuration, enforces a bounded bake window, records heartbeat/report data, and separates `observe_only` from `enabled_reasons`. Switching back to observe-only preserves the spool.

## Data and privacy

`GuardianBakeReport` includes guardian version, platform, observation window, resource and usage high-water marks, proposed or performed reap decisions with evidence, refused decisions, inspection or cleanup errors, and a heartbeat timestamp. All free-form strings pass through one redactor before model construction and persistence. The redactor removes credential-like assignments, authorization tokens, home-directory prefixes, and command arguments; structured reason codes and bounded exception class names remain.

The spool is the durable source of truth. Append and delivery acknowledgement use the existing symlink-safe private-file and exclusive-lock primitives. Corrupt, oversized, duplicate-ID, or unsupported-version state fails closed. The bounded spool evicts only delivered records; it refuses new data rather than discarding undelivered evidence.

## Reporting semantics

The sink sends a compact Markdown summary bearing the report deduplication key and repeat count. Before creating a comment, it scans existing BOU-2704 comments for that marker so retries are idempotent. Transport outages leave the report pending. The sink cannot accept a different issue ID and cannot create Linear issues.

## Operational semantics

Installation is represented by explicit persisted configuration rather than implicit daemon mutation. `enabled=false` performs no collection or upload. `mode=observe_only` records proposed reaps but authorizes none. `mode=reap` requires an explicit non-empty reason-code allowlist and still relies on the guardian's existing fencing authorization. Once the configured end time is reached, collection and upload stop automatically.

Bake exit assessment is deterministic: zero performed live-resource reaps, evidence for every enabled reason code, and guardian overhead within configured CPU/memory bounds. It produces a typed pass/fail result; it never broadens cleanup policy automatically.

## Verification

Tests cover contract validation, stable deduplication, redaction before disk, offline persistence, restart recovery, identical-state compaction, spool bounds, fixed-ticket enforcement, retry idempotency, opt-in/window/mode behavior, rollback preservation, and exit assessment. Existing guardian tests remain unchanged and green on Python 3.11–3.13.
