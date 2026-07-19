# agent-session-harness

Crash-safe context accounting and fresh-session supervision for Claude Code and Codex.

The harness keeps long agent tasks moving without letting a single conversation grow indefinitely. It measures native usage, warns at 65% of the model window, drains at 70%, verifies a durable handoff, fences the predecessor, and launches a genuinely fresh successor. It never uses Claude `--continue`/`--resume` or Codex `resume --last` for an automatic rotation.

## Guarantees

- Claude assistant rows are deduplicated by stable API message ID. Missing IDs are deterministic but explicitly degraded.
- Codex forked rollouts subtract each child’s inherited pre-spawn baseline instead of charging copied history again.
- Live context and cumulative spend are separate values.
- Lifecycle hooks retain only identifiers, timestamps, normalized tool names, and working directories. Conversation bodies are not stored.
- Unknown context, stale/corrupt lifecycle state, an unverified required checkpoint, or unknown fencing state disables automatic termination.
- Required checkpoint stores are written and read back by exact capsule fingerprint before ownership is fenced.
- Rotation stops the predecessor before launching the successor, and coordinator lease epochs reject stale owners.
- Every supervisor effect has durable before/after evidence and an idempotency key. Process launches are keyed by chain and generation.

## Install

Version `v0.1.0` pins [`agent-coordinator` v0.2.0](https://github.com/Boundless-Studios/agent-coordinator/releases/tag/v0.2.0) by immutable Git tag.

```bash
python -m pip install \
  'agent-session-harness @ git+https://github.com/Boundless-Studios/agent-session-harness.git@v0.1.0'

agent-session-harness doctor --json
```

For development:

```bash
uv sync --extra dev
uv run pytest -q
```

## Inspect usage

Claude:

```bash
agent-session-harness inspect \
  --runtime claude \
  --path ~/.claude/projects/<project>/<conversation>.jsonl \
  --window-tokens 1000000 \
  --json
```

Codex, including a fork lineage:

```bash
agent-session-harness inspect \
  --runtime codex \
  --lineage ~/.codex/sessions/<root>.jsonl \
  --lineage ~/.codex/sessions/<child>.jsonl \
  --json
```

The Codex result reports both the naïve sum of cumulative snapshots and the corrected incremental total. A missing child baseline is `degraded`; it is never guessed.

## Configuration and observe-only mode

Configuration precedence is explicit path, project `.agent-session-harness.toml`, then the platform user config directory.

```toml
observe_only = false

[governor]
warn_percent = 65.0
rotate_percent = 70.0
stale_event_timeout_seconds = 30.0
```

Managed rotation is enabled only when the host declares every required capability known. Otherwise the loaded configuration is forced to observe-only. Preflight a host integration without starting a model:

```bash
agent-session-harness supervise \
  --runtime codex \
  --cwd "$PWD" \
  --chain-id <stable-chain-id> \
  --task-type linear \
  --task-id BOU-2195 \
  --task-fingerprint <stable-task-fingerprint> \
  --state .agent-session-harness/supervisor.json \
  --required-capabilities-known \
  --check --json
```

The long-lived integration surface is `agent_session_harness.supervisor.Supervisor`. A host supplies four small protocols: native usage reader, checkpoint manager, fenced coordinator, and process driver. This keeps Linear, beads, PR dashboards, worktree launchers, and project safety policy outside the reusable package. The deterministic E2E test uses a real child process and proves root → checkpoint → fence → stop → fresh successor → acknowledgement with no overlap.

## Durable handoff capsule

`HandoffCapsule` contains only bounded operational state:

- task identifiers, objective, exact next action, and completed/remaining criteria;
- repository, branch, HEAD, dirty paths, and file/symbol anchors;
- test results, decisions, blockers, and allowlisted process summaries;
- predecessor conversation, target generation, creation time, and SHA-256 fingerprint.

Unknown fields are rejected. Required adapters receive canonical JSON on stdin and are invoked as an argv array, never through a shell:

```json
{
  "schema_version": 1,
  "operation": "write",
  "idempotency_key": "chain-1:1",
  "capsule": {"schema_version": 1, "fingerprint": "..."}
}
```

They return one bounded object:

```json
{
  "ok": true,
  "fingerprint": "...",
  "retryable": false,
  "error": null
}
```

Required adapters must pass `write` and independent `read` fingerprint checks. Mirror failures go to a locked `0600` outbox and can be retried later:

```bash
agent-session-harness outbox replay \
  --path .agent-session-harness/mirrors.jsonl \
  --adapter 'linear=["/absolute/path/to/linear-adapter"]' \
  --json
```

## Native hooks

Install owned hook fragments additively; unrelated hooks and their order are preserved.

```bash
agent-session-harness hooks install \
  --runtime claude --path ~/.claude/settings.json --json

agent-session-harness hooks install \
  --runtime codex --path .codex/hooks.json --json

agent-session-harness hooks check \
  --runtime codex --path .codex/hooks.json --json
```

`--dry-run` and `hooks uninstall` are supported. Invalid JSON is never mutated. Hook execution requires `AGENT_SESSION_HARNESS_MANAGED=1`, writes only to the local lifecycle ledger, performs no network work, and applies a 1 MiB input bound.

At a Stop event, normal sessions are allowed to stop. A draining session receives one continuation request listing the configured durable checkpoints. A recursion marker prevents repeated blocking. Once the capsule fingerprint is verified, Stop is allowed immediately.

## Successor acknowledgement

The supervisor gives a successor only the capsule path, fingerprint, target generation, and a short instruction. The successor verifies the capsule locally and acknowledges it:

```bash
agent-session-harness acknowledge \
  --state .agent-session-harness/supervisor.json \
  --generation 1 \
  --fingerprint <sha256> \
  --conversation-id <native-conversation-id> \
  --json
```

Normal dispatch stays disabled until the expected generation and fingerprint match. Automatic rotation is equivalent to a safe `/clear`: it creates a new native conversation and deliberately has no resume path. Manual resume remains the responsibility of the calling launcher.

## Status consumers

`report --json` emits the stable downstream status contract used by terminal and dashboard integrations: runtime, governor state, context percentage/confidence, quiescence, active turn/tool/subagent/critical counts, chain/conversation/generation IDs, last checkpoint fingerprint, and outbox depth.

```bash
agent-session-harness report \
  --state .agent-session-harness/supervisor.json \
  --ledger .agent-session-harness/supervisor.json.lifecycle \
  --outbox .agent-session-harness/mirrors.jsonl \
  --json
```

Consumers should project this record; they should not infer lifecycle ownership from CPU usage or launch their own model-backed summarizer.

## Integration boundaries

- [`agent-coordinator`](https://github.com/Boundless-Studios/agent-coordinator) owns atomic claims and lease-epoch fencing.
- `worktree-deck` may route fresh actions through a managed command while preserving its explicit manual-resume action.
- `agentic-pr-dash` may ingest lifecycle events and canonical status reports, but PR concepts never enter this package.
- Project adapters own beads/Linear updates, repository safety checks, and any project-specific hook policy.

## Privacy and recovery

Usage parsers whitelist only accounting metadata. Capsules reject unknown fields, adapter diagnostics are bounded and credential-shaped assignments are redacted, ledgers/outboxes/state files use `0600`, and corrupt records fail closed.

On restart, the supervisor resumes from the last durable phase. Checkpoints, fencing, stopping, same-owner claims, and launch keys are idempotent. If a crash occurs after a child starts but before completion is recorded, the process registry finds the existing chain/generation instead of launching a duplicate.

MIT licensed. See [LICENSE](LICENSE).
