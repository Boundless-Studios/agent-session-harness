# Process Identity and Observation Contract

## Context

BOU-2709 establishes the one reusable process-liveness primitive for session
supervision, waiter ownership, orphan proof, and worktree safety. The current
`PosixProcessDriver` contains private birth-token parsing, while downstream
consumers still have incentives to combine PID existence with ad hoc process
inspection. That is unsafe under PID reuse, same-second successors, zombies,
permission failures, and malformed native output.

This change publishes the contract and native macOS/Linux adapters in
`agent-session-harness`. Higher-level guardian policy, cleanup decisions, and
repository integrations remain separate.

## Goals

- Publish versioned, provider-neutral `ProcessIdentity` and
  `ProcessObservation` JSON contracts.
- Distinguish a process lifetime using the finest stable native start token.
- Return `unknown` for ambiguous inspection instead of guessing.
- Ensure zombies never qualify as live owners.
- Give future guardian resources a typed but decoupled reference.
- Move native birth-token parsing behind one reusable adapter boundary.

## Non-goals

- Guardian singleton ownership, cleanup policy, or reaping.
- Gaia, GitHub, Linear, beads, PR, worktree, or session-rotation policy.
- A new long-running service or subprocess inspection protocol.
- Automatic migration of external consumers.

## Public contracts

All public models use Pydantic, `schema_version: Literal[1]`, bounded strings
and collections, timezone-aware timestamps, and `extra="allow"`. Additive
unknown JSON fields survive validation and serialization. Any unsupported
schema version is rejected.

### `ProcessIdentity`

Fields:

- `schema_version`
- `platform`: `linux` or `darwin`
- `pid`: positive integer
- `opaque_start_token`: bounded non-empty string
- `executable_identity`: bounded non-empty string
- `captured_at`: timezone-aware UTC timestamp

Consumers compare process lifetimes through `same_process(left, right)`.
Lifetime equivalence requires the same supported schema, platform, PID, and
opaque start token. The executable identity is provenance evidence, not part
of lifetime equivalence, because a successful `execve` preserves the process
lifetime. An executable change is recorded in observation evidence.

### `ManagedResourceReference`

Fields:

- `schema_version`
- `kind`: bounded non-empty string
- `resource_key`: bounded non-empty string

This is a typed link, not the future guardian's `ManagedResource` model. It
lets BOU-2712 correlate observations without creating a dependency from
low-level process inspection to guardian internals.

### `ProcessEvidence`

Fields:

- `source`: bounded non-empty string
- `code`: bounded non-empty stable code
- `detail`: optional bounded diagnostic

Evidence must not include raw command output, environment values, or secrets.

### `ProcessObservation`

Fields:

- `schema_version`
- `identity`: the expected `ProcessIdentity`
- `state`: `running`, `zombie`, `missing`, or `unknown`
- `parent_identity`: optional `ProcessIdentity`
- `managed_resource`: optional `ManagedResourceReference`
- `evidence`: bounded list of `ProcessEvidence`
- `observed_at`: timezone-aware UTC timestamp

## API and data flow

`ProcessInspector` selects a native reader from the requested identity's
platform:

1. `capture(pid)` reads a trustworthy current native record and returns a
   `ProcessIdentity`. It returns `None` when no trustworthy identity can be
   captured; it never substitutes PID existence.
2. `observe(expected_identity, managed_resource=None)` reads the current native
   record and always returns a `ProcessObservation` for the expected identity.
3. A missing native record produces `missing`.
4. A matching start token with zombie native state produces `zombie`.
5. A matching start token with a live state produces `running`.
6. A different start token at the same PID produces `missing` with
   `pid_reused` evidence.
7. Permission failures, unsupported platforms, malformed records, incomplete
   executable identity, or other ambiguous inspection produce `unknown`.
8. Parent identity is best effort. Failure to identify a parent omits it and
   adds evidence without weakening a trustworthy child state.

The existing `PosixProcessDriver` delegates birth-token capture and fingerprint
generation to this module. Its persisted string fingerprint remains stable, so
this PR does not invalidate existing supervisor snapshots. Full downstream
consumers migrate to structured observations separately.

## Native adapters

### Linux

- Read `/proc/<pid>/stat` once and parse after the final closing parenthesis so
  spaces and parentheses in command names are safe.
- Use field 3 for native state, field 4 for parent PID, and field 22 for the
  start token.
- Resolve `/proc/<pid>/exe` for executable identity.
- Treat `Z` as zombie.
- Classify a matching zombie from state and start token before requiring an
  executable path, because zombie processes may no longer expose one.
- Treat `FileNotFoundError`/`ProcessLookupError` as missing.
- For non-zombies, treat permission, malformed fields, empty tokens, and
  read-link failures as unknown.

### macOS

- Use `libproc` `proc_pidinfo(PROC_PIDTBSDINFO)` for PID, parent PID, status,
  and microsecond start time.
- Use `proc_pidpath` for executable identity.
- The opaque token contains seconds and microseconds; it never truncates to a
  one-second textual timestamp.
- Treat the native zombie status as zombie.
- Classify a matching zombie before requiring `proc_pidpath`.
- A zero/short result with a disappearance errno is missing; permission or
  structurally invalid data is unknown.

The native system-call layer converts platform results into a small internal
record. Deterministic fixture tests exercise the parsers and state mapping
without requiring the host platform. Real Linux and macOS CI tests exercise
the current process and a short-lived child.

## Error and compatibility semantics

- `unknown` is not live and is not permission to reap.
- `missing` means the expected process lifetime is absent, including PID reuse.
- `zombie` is terminal and never a live owner.
- No API returns `running` from `kill(pid, 0)` or PID existence alone.
- Unsupported platforms return `unknown` when observing an expected identity
  and return no captured identity for initial capture.
- Unsupported schema versions fail validation before native inspection.
- Additive unknown fields are accepted to allow minor-version evolution.

## Test strategy

Contract-first tests are written and observed RED before implementation.

- JSON schema/version round trips and additive unknown fields.
- Unsupported major versions and invalid bounded fields.
- Linux fixtures: running, zombie, PID reuse, same-second successor, missing,
  permission failure, malformed `stat`, malformed executable link.
- macOS fixtures: microsecond successors, zombie, missing, permission failure,
  short/malformed native records, path failure.
- Cross-platform comparison semantics and executable-change evidence.
- Typed managed-resource reference propagation.
- Existing `PosixProcessDriver` fingerprint behavior delegates to the shared
  capture implementation.
- Real Linux and macOS smoke tests in the existing CI jobs.

## Delivery boundary

One upstream contract/adapters PR containing the public module, colocated
tests/fixtures, the narrow `PosixProcessDriver` delegation, package exports,
and contract documentation. Guardian and downstream consumer migrations remain
separate PRs.
