# Durable Guardian Service Plan

**Goal:** Persist explicit managed-resource registrations and run one fenced,
observe-only guardian per OS user.

**Architecture:** A private atomic registry owns durable registration envelopes.
Each logical `(kind, resource_key)` has a fresh incarnation token, so stale
sessions cannot unregister replacements. A guardian service snapshots the
registry, inspects outside the registry lock, evaluates the pure policy engine,
and revalidates its fenced singleton lease before publishing reap-authorizing
decisions. Cleanup execution remains out of scope.

## Slice 1: durable registration

- Add a bounded, versioned `ResourceRegistration` envelope.
- Add atomic register/list/unregister operations using secure-file primitives.
- Require exact incarnation tokens for unregister.
- Fail closed on corrupt, oversized, unsupported, or symlinked state.
- Cover restart, replacement races, permissions, and concurrent mutation.

## Slice 2: fenced observe-only service

- Acquire one coordinator-backed claim per OS user and advance lease epochs.
- Reject duplicate active guardians and stale predecessor operations.
- Evaluate explicit registrations only and publish decisions without cleanup.
- Revalidate ownership before publishing any reap-authorizing decision.
- Cover restart recovery, lost ownership, observer failure, and shutdown races.

## Slice 3: publish follow-on PR

- Export contracts, add CLI/config entry points, and document operational paths.
- Run full tests, current CI Ruff, formatting, and build.
- Request adversarial review and settle CI/review feedback.
