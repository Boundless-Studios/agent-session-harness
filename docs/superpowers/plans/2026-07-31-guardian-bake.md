# Guardian Bake Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in, durable guardian bake that records redacted evidence locally and can idempotently report only to BOU-2704.

**Architecture:** Frozen Pydantic contracts feed a bounded private spool; a fixed-ticket Linear adapter drains pending summaries; a small runtime/configuration layer controls opt-in collection, bake-window expiry, observe-only versus reason-allowlisted reap mode, and exit assessment. Network transport and clocks are injected so every behavior is deterministic in tests.

**Tech Stack:** Python 3.11+, Pydantic 2, existing secure-file helpers, injected Linear GraphQL transport, pytest, Ruff.

---

### Task 1: Report contract and redaction

**Files:**
- Create: `src/agent_session_harness/guardian_bake.py`
- Create: `tests/test_guardian_bake.py`

- [ ] Write failing tests proving timezone-aware windows, deterministic content-derived deduplication, bounded structured fields, and redaction of credentials, home prefixes, and command arguments.
- [ ] Run `uv run pytest -q tests/test_guardian_bake.py` and confirm failures are caused by the missing module.
- [ ] Implement frozen Pydantic models for usage, counts, decisions, errors, report, and exit assessment plus a single `redact_guardian_text` boundary. Compute SHA-256 over canonical JSON excluding heartbeat and deduplication key.
- [ ] Rerun the focused tests and commit `feat: define guardian bake report contract`.

### Task 2: Durable deduplicating spool

**Files:**
- Create: `src/agent_session_harness/guardian_bake_spool.py`
- Create: `tests/test_guardian_bake_spool.py`

- [ ] Write failing tests for persist-before-send, restart recovery, consecutive-state compaction, delivery acknowledgement, private file mode, corrupt/oversized input, duplicate record IDs, and refusal to evict undelivered evidence.
- [ ] Run `uv run pytest -q tests/test_guardian_bake_spool.py` and confirm the missing spool fails.
- [ ] Implement a locked versioned JSON document using `exclusive_lock`, `read_private_text`, and `atomic_write_private_text`; retain pending records, compact identical keys by incrementing `repeat_count`, and evict only oldest delivered records at the configured bound.
- [ ] Rerun spool and secure-file tests and commit `feat: add durable guardian bake spool`.

### Task 3: Fixed-ticket Linear sink

**Files:**
- Create: `src/agent_session_harness/adapters/guardian_bake_linear.py`
- Create: `tests/test_guardian_bake_linear.py`

- [ ] Write failing tests proving all reads/writes target BOU-2704, existing deduplication markers suppress duplicate comments, successful read-back acknowledges delivery, and transport/read-back failures leave reports pending.
- [ ] Run `uv run pytest -q tests/test_guardian_bake_linear.py` and confirm the missing adapter fails.
- [ ] Implement the injected-transport sink with constant `TARGET_ISSUE = "BOU-2704"`, bounded comment scanning, marker-based idempotency, compact Markdown formatting, and no create-issue surface.
- [ ] Rerun adapter tests and commit `feat: report guardian bake evidence to fixed Linear issue`.

### Task 4: Opt-in runtime controls and exit assessment

**Files:**
- Create: `src/agent_session_harness/guardian_bake_runtime.py`
- Create: `tests/test_guardian_bake_runtime.py`
- Modify: `src/agent_session_harness/__init__.py`
- Modify: `README.md`

- [ ] Write failing tests for disabled no-op behavior, automatic window expiry, observe-only default, explicit allowlisted reap mode, rollback to observe-only without spool deletion, heartbeat collection, and exit pass/fail criteria.
- [ ] Run `uv run pytest -q tests/test_guardian_bake_runtime.py` and confirm the missing runtime fails.
- [ ] Implement frozen configuration and runtime orchestration that records before any send, never deletes the spool on mode change, and returns typed exit assessment without enabling cleanup.
- [ ] Document opt-in configuration, platform support, rollback, privacy, and BOU-2704-only reporting; export only stable contracts.
- [ ] Rerun runtime tests and commit `feat: add opt-in guardian bake runtime`.

### Task 5: Integrated verification and shipping

**Files:**
- Verify all files above.

- [ ] Run focused guardian/bake tests with `uv run pytest -q tests/test_guardian_bake.py tests/test_guardian_bake_spool.py tests/test_guardian_bake_linear.py tests/test_guardian_bake_runtime.py tests/test_guardian_executor.py tests/test_guardian_service.py tests/test_guardian_singleton.py tests/test_resource_registry.py`.
- [ ] Run `uvx --from ruff==0.16.1 ruff check src tests` and `uvx --from ruff==0.16.1 ruff format --check src tests`.
- [ ] Run the full isolated suite with `uv run pytest -q`; distinguish any host timing failures from changed code using focused reruns.
- [ ] Push `bou-2714-guardian-bake`, open one ready PR linked to BOU-2714, and maintain CI/review feedback through two consecutive clean observations before merge.
