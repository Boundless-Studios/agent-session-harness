# Conservative Resource Guardian Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a portable host-resource contract and deterministic orphan decision engine that can reap only from affirmative ownership/liveness proof.

**Architecture:** Keep native process observation in `process_identity.py`, and introduce `resource_guardian.py` as a pure policy boundary. Registrations describe managed resources and cleanup adapter identities; observations describe process, worktree, owner, and fenced-lease state; the decision engine emits stable reason/evidence without executing cleanup. A later PR will persist registrations, own the per-user singleton lease, and invoke configured adapters.

**Tech Stack:** Python 3.11+, Pydantic 2, existing macOS/Linux `ProcessIdentity`/`ProcessObservation`, pytest, Ruff, uv.

---

## File structure

- `src/agent_session_harness/resource_guardian.py`: versioned resource/observation/decision contracts and pure decision engine.
- `src/agent_session_harness/__init__.py`: public exports.
- `tests/test_resource_guardian_contract.py`: wire contract, timestamp, bounded-input, and forward-compatibility tests.
- `tests/test_resource_guardian_decision.py`: conservative decision matrix.
- `README.md`: contract and safety-policy documentation.

### Task 1: Publish managed-resource contracts

- [ ] Write failing tests for `WorktreeIdentity`, `OwnerLease`, `ManagedResource`, `GuardianEvidence`, `GuardianObservation`, and `GuardianDecision`.
- [ ] Prove `ManagedResource` carries `schema_version`, `kind`, stable `resource_key`, optional process/worktree/lease identity, and a configured `cleanup_adapter` without Gaia paths or process names.
- [ ] Prove timestamps are timezone-aware, inputs are bounded, unsupported major versions reject, and additive fields round-trip.
- [ ] Run `uv run pytest -q tests/test_resource_guardian_contract.py` and confirm the missing module/API is the RED failure.
- [ ] Implement the minimal frozen forward-compatible Pydantic contracts and run the focused tests green.
- [ ] Commit the contract slice.

### Task 2: Implement conservative orphan decisions

- [ ] Write failing tests for live managed owners, deleted worktrees, missing children, PID reuse, zombies, unknown identity, and inspection failure.
- [ ] Assert exact live owners/resources retain; unknown identity or inspection failure alerts; deleted worktree plus no live owner reaps; missing/zombie managed children reap registry cleanup; expired fenced identity reaps only when process identity is no longer live.
- [ ] Assert a live exact process with no affirmative orphan proof alerts rather than reaps. Age, process count, and RSS are intentionally absent from the decision input.
- [ ] Run focused decision tests and confirm the engine is absent.
- [ ] Implement a pure `decide_guardian_action` function with stable reason codes and complete evidence.
- [ ] Run focused tests green and commit the decision slice.

### Task 3: Publish the contract/engine PR

- [ ] Export the public contract and engine from `agent_session_harness`.
- [ ] Document observe-only default behavior and independently enabled reap reason codes; state that this PR executes no cleanup.
- [ ] Run `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .`, and `uv build`.
- [ ] Request an independent code review, address Critical/Important findings, push one focused BOU-2712 contract/decision PR, and settle CI/review feedback.

### Follow-on PR 1: Durable registration and singleton service

- [ ] Add atomic private per-user registration storage with eager register/unregister and crash persistence.
- [ ] Add one fenced guardian singleton per OS user, restart recovery, duplicate-guardian rejection, and observe-only configuration.
- [ ] Cover guardian restart and races with session shutdown.

### Follow-on PR 2: Cleanup adapters and execution

- [ ] Add explicitly registered adapters for rotation runtimes, hook/peon children, worktree tunnels, and language-server trees.
- [ ] Gate reaping independently by stable reason code, execute bounded cleanup, and record cleanup failure without converting ambiguity into permission.
- [ ] Cover cleanup failure and macOS/Linux service integration.

## Self-review

- Spec coverage: the first PR covers the mandated contract/decision boundary and portable safety matrix; singleton persistence, adapters, cleanup failure, and shutdown races are explicitly staged as follow-ons allowed by the issue’s PR boundary.
- Placeholder scan: no TBD/TODO implementation placeholders.
- Type consistency: registration, observation, evidence, and decision names are defined once and reused by every planned stage.
